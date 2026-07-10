from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.common.enums import StateSE2Index

from .sparsedrive_config import SparseDriveConfig
from .sparsedrive_backbone import SparseBackbone
from .custom_decoder import CustomTransformerDecoder


class SparseDriveModel(nn.Module):
    """
    SparseDriveV2 核心模型类，继承自 PyTorch 的 nn.Module。
    
    该模型实现了端到端自动驾驶规划的核心逻辑：
    1. 使用 ResNet-34 作为视觉骨干网络提取图像特征
    2. 对 ego 状态进行编码
    3. 使用 Transformer Decoder 进行分层轨迹评分
    4. 基于可分解词汇表（路径+速度）生成最优轨迹
    
    核心架构：
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  多视角图像  │ ──→ │  ResNet-34  │ ──→ │ 多尺度特征  │
    └─────────────┘     └─────────────┘     └──────┬──────┘
                                                  │
    ┌─────────────┐     ┌─────────────┐            │
    │  ego状态    │ ──→ │ 状态编码器   │ ──→ ┌─────┴─────┐
    └─────────────┘     └─────────────┘     │ Trajectory │
                                           │   Head     │ ──→ 最优轨迹
                                           └─────┬─────┘
                                                 │
                                    ┌────────────┴────────────┐
                                    ▼                         ▼
                              路径词汇表(1024)           速度词汇表(256)
    """

    def __init__(self, config: SparseDriveConfig):
        """
        SparseDriveModel 构造函数，初始化模型的核心组件。
        
        :param config: SparseDriveConfig 配置对象，包含所有模型超参数
        """
        # 调用父类构造函数
        super().__init__()

        # 保存配置对象，供后续使用
        self._config = config
        
        # 1. 初始化视觉骨干网络（ResNet-34）
        # SparseBackbone 包含：
        # - ResNet-34 预训练模型
        # - 可选的 GridMask 数据增强
        # - 多尺度特征融合 Neck
        self._backbone = SparseBackbone(config)
        
        # 2. 初始化 ego 状态编码器
        # 输入维度：4(驾驶命令) + 2(速度) + 2(加速度) = 8
        # 输出维度：d_model (默认256)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.d_model)
        
        # 3. 初始化轨迹预测头（TrajectoryHead）
        # 包含可分解轨迹词汇表和 Transformer Decoder
        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,  # 轨迹点数，默认8
            d_ffn=config.d_ffn,                              # FFN维度，默认1024
            d_model=config.d_model,                          # 模型维度，默认256
            config=config,                                   # 完整配置对象
        )

    def forward(self, features: Dict[str, torch.Tensor], targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        模型前向传播，定义完整的计算流程。
        
        :param features: 输入特征字典，包含：
                         - camera_feature: 相机图像特征（多视角图像及投影参数）
                         - status_feature: ego状态特征（驾驶命令、速度、加速度）
        :param targets: 目标标签字典，用于训练时的损失计算，推理时可为None
        :return: (output, loss_dict) - 输出字典和损失字典
        """
        # 1. 提取输入特征
        camera_feature: Dict[str, torch.Tensor] = features["camera_feature"]
        status_feature: torch.Tensor = features["status_feature"]
        
        # 2. 获取 batch size
        batch_size = status_feature.shape[0]
        
        # 3. 对 ego 状态进行编码
        # 输入: [B, 8] (4命令 + 2速度 + 2加速度)
        # 输出: [B, d_model]
        status_encoding = self._status_encoding(status_feature)
        
        # 4. 通过视觉骨干网络提取图像特征
        # 输入: [B, num_cams, 3, H, W] - 多视角相机图像
        # 输出: 多尺度特征图列表
        imgs = camera_feature["imgs"]
        feature_maps = self._backbone(imgs) #feature_maps (List[Tensor])
        
        # 将特征图添加到 camera_feature 字典中，供后续使用
        camera_feature["feature_maps"] = feature_maps
        
        # 5. 通过轨迹头进行预测
        # 输入: camera_feature(图像特征), status_encoding(ego状态编码), targets(训练目标)
        # 输出: trajectory(预测轨迹), loss_dict(损失字典)
        output = {}
        trajectory, loss_dict = self._trajectory_head(camera_feature, status_encoding, targets)
        output.update(trajectory)
        
        # 6. 训练模式下计算总损失
        if self.training:
            loss_dict["loss"] = sum(loss_dict.values())
        
        # 返回输出和损失
        return output, loss_dict


class TrajectoryHead(nn.Module):
    """
    轨迹预测头，实现可分解轨迹词汇表和分层评分机制。
    
    核心创新：将轨迹分解为路径（path）和速度（velocity）两个独立词汇表，
    通过组合得到超密集的候选轨迹集（1024 × 256 = 262,144 种组合）。
    
    工作流程：
    1. 加载预定义的路径和速度词汇表（通过 K-Means 聚类得到）
    2. 对词汇表进行位置编码
    3. 使用 Transformer Decoder 进行分层评分
    4. 通过粗粒度筛选和细粒度评分选择最优轨迹
    """

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, config: SparseDriveConfig = None):
        """
        TrajectoryHead 构造函数，初始化词汇表和解码器。
        
        :param num_poses: 轨迹点数量（默认8，对应4秒×0.5秒间隔）
        :param d_ffn: 前馈网络维度（默认1024）
        :param d_model: 模型维度（默认256）
        :param config: SparseDriveConfig 配置对象
        """
        super(TrajectoryHead, self).__init__()

        # 保存配置参数
        self._num_poses = num_poses    # 轨迹点数量
        self._d_model = d_model        # 模型维度
        self._d_ffn = d_ffn            # FFN维度

        # ============ 加载预定义词汇表（Anchor）============
        
        # 1. 路径词汇表 [K_PATH, len_path, 3]
        # K_PATH=1024, len_path=50, 每个路径包含50个点(x,y,heading)
        self.path_vocab = nn.Parameter(
            torch.from_numpy(np.load(config.path_anchor)).float(),
            requires_grad=False  # 固定不训练
        )
        
        # 2. 速度词汇表 [K_VELOCITY, len_vel_seq]
        # K_VELOCITY=256, len_vel_seq=8, 每个速度序列包含8个时间步的速度值
        self.vel_vocab = nn.Parameter(
            torch.from_numpy(np.load(config.velocity_anchor)).float(),
            requires_grad=False  # 固定不训练
        )
        
        # 3. 完整轨迹词汇表（预计算的路径+速度组合）
        # [K_PATH, K_VELOCITY, num_poses, 3] - 262K 条完整轨迹
        #ckpt/kmeans/trajectory_1024_256.npz
        trajectory_data = np.load(config.trajectory_anchor)
        self.traj_vocab = nn.Parameter(
            torch.from_numpy(trajectory_data["trajectory"]).float(),
            requires_grad=False
        )
        #- trajectory: 组合轨迹 [1024, 256, 8, 3]
        #- trajectory_mask: 有效掩码 [1024, 256, 8]  
        # 轨迹掩码，标记有效轨迹点
        # [K_PATH, K_VELOCITY, num_poses]
        #trajectory_mask[i, j] = [1, 1, 1, 1, 0, 0, 0, 0]（后4个点无效）
        self.traj_mask = nn.Parameter(
            torch.from_numpy(trajectory_data["trajectory_mask"]).float(),
            requires_grad=False
        )

        # ============ 位置编码器（Positional Encoding）============
        
        # 路径位置编码器：将路径坐标转换为 d_model 维度的嵌入
        # 输入: [B, K_PATH, len_path * 3] -> 输出: [B, K_PATH, d_model]
        self.path_pos_embed = nn.Sequential(
            nn.Linear(config.len_path * 3, d_ffn),  # 50*3=150 -> 1024
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),              # 1024 -> 256
        )
        
        # 速度位置编码器：将速度序列转换为 d_model 维度的嵌入
        # 输入: [B, K_VELOCITY, len_vel_seq] -> 输出: [B, K_VELOCITY, d_model]
        self.vel_pos_embed = nn.Sequential(
            nn.Linear(config.len_vel_seq, d_ffn),   # 8 -> 1024
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),              # 1024 -> 256
        )

        # ============ Transformer Decoder ============
        # 自定义解码器，实现分层评分和筛选机制
        self.decoder = CustomTransformerDecoder(num_poses, d_model, d_ffn, config)

    def forward(self, camera_feature, status_encoding, targets) -> Dict[str, torch.Tensor]:
        """
        轨迹预测前向传播。
        
        :param camera_feature: 相机特征字典，包含多尺度特征图
        :param status_encoding: ego状态编码 [B, d_model]
        :param targets: 训练目标字典
        :return: decoder_outputs - 包含预测轨迹和损失
        """
        # 获取 batch size
        B = status_encoding.shape[0]

        # ============ 准备词汇表（扩展到 batch 维度）============
        
        # 路径词汇表: [K_PATH, len_path, 3] -> [B, K_PATH, len_path, 3]
        path_vocab = self.path_vocab.data[None].repeat(B, 1, 1, 1)
        
        # 速度词汇表: [K_VELOCITY, len_vel_seq] -> [B, K_VELOCITY, len_vel_seq]
        vel_vocab = self.vel_vocab.data[None].repeat(B, 1, 1)
        # [None] 在第0维添加一个维度  第0维重复 B 次（batch size）
        # 轨迹词汇表: [K_PATH, K_VELOCITY, num_poses, 3] -> [B, K_PATH, K_VELOCITY, num_poses, 3]
        traj_vocab = self.traj_vocab.data[None].repeat(B, 1, 1, 1, 1)
        
        # 轨迹掩码: [K_PATH, K_VELOCITY, num_poses] -> [B, K_PATH, K_VELOCITY, num_poses]
        traj_mask = self.traj_mask.data[None].repeat(B, 1, 1, 1)

        # ============ 位置编码 ============
        
        # 路径位置编码: [B, K_PATH, len_path, 3] -> [B, K_PATH, d_model]
        path_embed = self.path_pos_embed(path_vocab.flatten(-2, -1))
        
        # 速度位置编码: [B, K_VELOCITY, len_vel_seq] -> [B, K_VELOCITY, d_model]
        vel_embed = self.vel_pos_embed(vel_vocab)

        # ============ Transformer Decoder 推理 ============
        
        # 调用解码器进行分层评分
        # 输入:
        #   - 词汇表嵌入和原始数据
        #   - 相机特征、ego状态、训练目标
        decoder_outputs = self.decoder(
            (path_embed, vel_embed, path_vocab, vel_vocab, traj_vocab, traj_mask),
            (camera_feature, status_encoding, targets),
        )

        # 返回解码器输出（包含预测轨迹和损失）
        return decoder_outputs
