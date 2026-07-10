from typing import Any, List, Dict, Optional, Union
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, AgentInput, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

from .sparsedrive_config import SparseDriveConfig
from .sparsedrive_model import SparseDriveModel
from .sparsedrive_features import SparseDriveFeatureBuilder, SparseDriveTargetBuilder
from .sparsedrive_callback import CheckpointCallback


class SparseDriveAgent(AbstractAgent):
    """
    SparseDriveV2 智能体实现类，继承自 AbstractAgent。
    
    该类是端到端自动驾驶规划模型的核心入口，负责：
    1. 初始化模型结构（SparseDriveModel）
    2. 定义传感器配置
    3. 提供特征和目标构建器
    4. 实现前向传播和损失计算
    5. 提供优化器和训练回调
    
    核心架构：
    - 视觉骨干网络（ResNet-34）提取图像特征
    - 可分解轨迹词汇表（路径+速度）
    - Transformer Decoder 进行分层评分
    - PDM指标监督优化
    """

    def __init__(
        self,
        config: SparseDriveConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        test_every_n_epochs: int = 5,
        test_batchsize: int = 4,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        SparseDriveAgent 构造函数，初始化智能体的核心组件。
        
        :param config: SparseDriveConfig 配置对象，包含模型结构、训练超参数等
        :param lr: 学习率，用于优化器
        :param checkpoint_path: 预训练权重路径，推理或fine-tune时使用，默认为None
        :param test_every_n_epochs: 测试间隔（每N个epoch测试一次），默认5
        :param test_batchsize: 测试时的batch size，默认4
        :param trajectory_sampling: 轨迹采样配置，定义时间范围和间隔
                                   默认: time_horizon=4秒, interval_length=0.5秒
        """
        # 调用父类构造函数，传入轨迹采样配置
        super().__init__(trajectory_sampling)

        # 保存配置参数
        self._config = config                    # 模型配置对象
        self._lr = lr                            # 学习率
        self.test_every_n_epochs = test_every_n_epochs  # 测试间隔
        self.test_batchsize = test_batchsize            # 测试batch size

        # 保存预训练权重路径
        self._checkpoint_path = checkpoint_path
        
        # 核心：实例化 SparseDriveModel 模型
        # SparseDriveModel 包含：
        # 1. ResNet-34 视觉骨干网络
        # 2. Ego状态编码器
        # 3. TrajectoryHead（Transformer Decoder + 可分解词汇表）
        self._sparsedrive_model = SparseDriveModel(config)

    def name(self) -> str:
        """
        返回智能体名称，用于标识和日志记录。
        
        :return: 类名字符串（SparseDriveAgent）
        """
        return self.__class__.__name__

    def initialize(self) -> None:
        """
        初始化智能体，加载预训练权重。
        
        在推理阶段调用此方法加载训练好的模型权重，支持CPU和GPU环境。
        权重文件是PyTorch Lightning格式，需要移除"agent."前缀。
        """
        # 根据是否有GPU选择加载设备
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        
        # 移除LightningModule添加的"agent."前缀，正确加载权重
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """
        获取传感器配置，定义智能体需要的传感器数据。
        
        SparseDriveV2 仅使用相机输入（8个视角），不使用LiDAR。
        每个相机获取4帧历史数据（索引0-3）用于时序建模。
        
        :return: SensorConfig 对象，包含相机和LiDAR配置
        """
        return SensorConfig(
            cam_f0=[0, 1, 2, 3],   # 前视相机，取4帧历史
            cam_l0=[0, 1, 2, 3],   # 左前相机
            cam_l1=[0, 1, 2, 3],   # 左中相机
            cam_l2=[0, 1, 2, 3],   # 左后相机
            cam_r0=[0, 1, 2, 3],   # 右前相机
            cam_r1=[0, 1, 2, 3],   # 右中相机
            cam_r2=[0, 1, 2, 3],   # 右后相机
            cam_b0=[0, 1, 2, 3],   # 后视相机
            lidar_pc=[],            # 不使用LiDAR，空列表
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """
        获取目标构建器列表，用于训练时生成监督信号。
        
        :return: 包含一个 SparseDriveTargetBuilder 的列表
        """
        return [SparseDriveTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        获取特征构建器列表，用于数据预处理和特征提取。
        
        :return: 包含一个 SparseDriveFeatureBuilder 的列表
        """
        return [SparseDriveFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播方法，定义模型的计算流程。
        
        :param features: 输入特征字典，包含相机图像和ego状态
        :param targets: 目标标签字典，用于训练时的损失计算，推理时可为None
        :return: 预测结果和损失字典
        """
        if targets is None:
            targets = {}
        return self._sparsedrive_model(features, targets)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        重写父类方法，处理 camera_feature 为 list 的情况。
        """
        self.eval()
        device = next(self.parameters()).device
        
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        
        builder = self.get_feature_builders()[0]
        features, _, _ = builder.pipeline(features, {}, "", test_mode=True)
        
        features["status_feature"] = features["status_feature"].unsqueeze(0).to(device)
        
        for key in features["camera_feature"]:
            val = features["camera_feature"][key]
            if isinstance(val, torch.Tensor):
                features["camera_feature"][key] = val.unsqueeze(0).to(device)
            elif isinstance(val, np.ndarray):
                features["camera_feature"][key] = torch.tensor(val).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = self.forward(features)
            output = predictions[0] if isinstance(predictions, tuple) else predictions
            poses = output["trajectory"].squeeze(0).cpu().numpy()
        
        return Trajectory(poses, self._trajectory_sampling)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算训练损失。
        
        :param features: 输入特征字典
        :param targets: 目标标签字典
        :param predictions: 模型预测结果（包含output和loss_dict）
        :return: 损失字典，由模型内部计算并返回
        """
        output, loss_dict = predictions
        return loss_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """
        返回优化器配置。
        
        使用 Adam 优化器，学习率由配置指定。
        
        :return: Adam优化器实例
        """
        return torch.optim.Adam(self._sparsedrive_model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        返回训练回调函数列表。
        
        :return: 包含 CheckpointCallback 的列表，用于模型保存
        """
        return [CheckpointCallback()]
