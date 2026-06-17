import logging                                                              # 日志模块，用于记录训练过程信息
from pathlib import Path                                                    # 路径处理模块，用于文件路径操作
from typing import Tuple                                                     # 类型提示，用于函数返回值类型标注

import hydra                                                                # Hydra配置管理框架，用于动态加载配置和实例化对象
import pytorch_lightning as pl                                              # PyTorch Lightning训练管理框架
from hydra.utils import instantiate                                         # Hydra实例化工具，根据配置动态创建对象
from omegaconf import DictConfig                                             # OmegaConf配置字典类型，支持配置合并和变量替换
from torch.utils.data import DataLoader                                      # PyTorch数据加载器，用于批量加载数据

from navsim.agents.abstract_agent import AbstractAgent                       # 智能体抽象基类，定义了智能体的标准接口
from navsim.common.dataclasses import SceneFilter                            # 场景过滤器数据类，用于筛选符合条件的场景
from navsim.common.dataloader import SceneLoader                             # 场景加载器，负责从日志文件中加载和解析场景数据
from navsim.planning.training.agent_lightning_module import AgentLightningModule  # Lightning模块包装器，将智能体适配为Lightning兼容形式
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset       # 数据集类：Dataset(完整数据集)、CacheOnlyDataset(仅缓存模式)

logger = logging.getLogger(__name__)                                        # 日志记录器实例，记录该模块的运行日志

# ==================== 配置路径常量 ====================
CONFIG_PATH = "config/training"                                             # 训练配置文件所在目录（相对路径）
CONFIG_NAME = "default_training"                                            # 默认训练配置文件名（不含.yaml扩展名）


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    从Omega配置构建训练和验证数据集
    
    :param cfg: OmegaConf配置字典，包含所有训练参数（如数据路径、批大小、训练轮数等）
    :param agent: NAVSIM智能体实例，提供特征构建器、目标构建器和传感器配置
    :return: (train_data, val_data) - 训练数据集和验证数据集的元组
    """
    # 创建训练场景过滤器，根据配置筛选训练场景
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        # 如果过滤器已指定日志列表，只保留在train_logs中的日志
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        # 否则使用配置中的train_logs列表
        train_scene_filter.log_names = cfg.train_logs

    # 创建验证场景过滤器，根据配置筛选验证场景
    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        # 如果过滤器已指定日志列表，只保留在val_logs中的日志
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        # 否则使用配置中的val_logs列表
        val_scene_filter.log_names = cfg.val_logs

    # 获取数据路径（配置中定义的绝对路径）
    data_path: Path = Path(cfg.navsim_log_path)                              # 日志数据根目录，如 /home/xqb/DATA2/E2E_Project/sparsev2/navsim_logs/mini
    original_sensor_path: Path = Path(cfg.original_sensor_path)             # 原始传感器数据根目录，如 /home/xqb/DATA2/E2E_Project/sparsev2/sensor_blobs/mini

    # 创建训练场景加载器，负责加载训练日志文件
    train_scene_loader: SceneLoader = SceneLoader(
        original_sensor_path=original_sensor_path,                          # 传感器数据路径，用于加载相机/LiDAR数据
        data_path=data_path,                                                # 日志数据路径，用于加载场景元数据和帧数据
        scene_filter=train_scene_filter,                                    # 场景过滤器，决定加载哪些场景
        sensor_config=agent.get_sensor_config(),                            # 传感器配置，从智能体获取（如分辨率、帧率等）
    )

    # 创建验证场景加载器，负责加载验证日志文件
    val_scene_loader: SceneLoader = SceneLoader(
        original_sensor_path=original_sensor_path,                          # 传感器数据路径
        data_path=data_path,                                                # 日志数据路径
        scene_filter=val_scene_filter,                                      # 验证场景过滤器
        sensor_config=agent.get_sensor_config(),                            # 传感器配置
    )

    # 创建训练数据集，封装场景加载器和特征/目标构建器
    train_data: Dataset = Dataset(
        scene_loader=train_scene_loader,                                    # 训练场景加载器实例
        feature_builders=agent.get_feature_builders(),                      # 特征构建器列表，用于计算输入特征（如相机特征、状态特征）
        target_builders=agent.get_target_builders(),                        # 目标构建器列表，用于计算训练目标（如轨迹、路径、速度）
        cache_path=cfg.cache_path,                                          # 缓存路径，如 exp/data_cache_navmini
        force_cache_computation=cfg.force_cache_computation,                # 布尔值，是否强制重新计算缓存（覆盖已有缓存）
        cfg=cfg,                                                            # 完整配置字典，传递给数据集内部使用
    )

    # 创建验证数据集
    val_data: Dataset = Dataset(
        scene_loader=val_scene_loader,                                      # 验证场景加载器实例
        feature_builders=agent.get_feature_builders(),                      # 特征构建器列表
        target_builders=agent.get_target_builders(),                        # 目标构建器列表
        cache_path=cfg.cache_path,                                          # 缓存路径
        force_cache_computation=cfg.force_cache_computation,                # 是否强制重新计算缓存
        cfg=cfg,                                                            # 完整配置字典
    )

    return train_data, val_data                                             # 返回训练和验证数据集元组


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    训练智能体的主入口函数
    
    :param cfg: OmegaConf配置字典，包含所有训练参数，由Hydra自动加载
    """

    # 设置全局随机种子，确保训练结果可复现
    pl.seed_everything(cfg.seed, workers=True)                              # seed: 整数，随机种子值（默认0）
    logger.info(f"Global Seed set to {cfg.seed}")

    # 记录输出目录路径
    logger.info(f"Path where all results are stored: {cfg.output_dir}")     # output_dir: 字符串，实验输出目录，如 exp/sparsedrive_agent/2026.06.17.17.46.52

    # 构建智能体实例（根据配置动态创建）
    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)                           # agent: 智能体实例，如 SparseDriveAgent，通过Hydra根据配置创建

    # 构建PyTorch Lightning模块，将智能体包装为Lightning兼容形式
    logger.info("Building Lightning Module")
    lightning_module: AgentLightningModule = AgentLightningModule(
        agent=agent,                                                        # 传入智能体实例，Lightning模块内部调用agent的forward/loss等方法
    )

    # 判断是否使用缓存模式加载数据（跳过SceneLoader构建）
    if cfg.use_cache_without_dataset:                                       # use_cache_without_dataset: 布尔值，是否使用纯缓存模式
        # 缓存模式：直接从缓存目录加载预计算的特征和目标
        logger.info("Using cached data without building SceneLoader")
        
        # 参数校验：缓存模式下必须禁用强制计算缓存
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        
        # 参数校验：缓存模式下必须提供缓存路径
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        
        # 创建训练数据集（仅使用缓存，不加载原始日志）
        train_data: CacheOnlyDataset = CacheOnlyDataset(
            cache_path=cfg.cache_path,                                      # 缓存目录路径，如 exp/data_cache_navmini
            test_mode=False,                                                # 布尔值，False表示训练模式（启用数据增强）
            feature_builders=agent.get_feature_builders(),                  # 特征构建器列表
            target_builders=agent.get_target_builders(),                    # 目标构建器列表
            log_names=cfg.train_logs,                                       # 训练日志名称列表，用于筛选缓存文件
        )
        
        # 创建验证数据集（仅使用缓存）
        val_data: CacheOnlyDataset = CacheOnlyDataset(
            cache_path=cfg.cache_path,                                      # 缓存目录路径
            test_mode=True,                                                 # 布尔值，True表示测试/验证模式（禁用数据增强）
            feature_builders=agent.get_feature_builders(),                  # 特征构建器列表
            target_builders=agent.get_target_builders(),                    # 目标构建器列表
            log_names=cfg.val_logs,                                         # 验证日志名称列表
        )
    else:
        # 非缓存模式：从原始日志文件构建数据集（会自动检查和使用缓存）
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)                   # 调用数据集构建函数，返回(train_data, val_data)

    # 预加载第10个样本，确保数据集能正常工作（调试用）
    train_data.__getitem__(10)                                              # 返回一个样本的特征和目标字典

    # 创建数据加载器，将数据集包装为批量加载形式
    logger.info("Building Datasets")
    train_dataloader: DataLoader = DataLoader(
        train_data, 
        **cfg.dataloader.params,                                            # 数据加载器参数，如 batch_size=8, num_workers=4
        shuffle=True                                                        # 训练时打乱数据顺序，增加随机性
    )
    logger.info("Num training samples: %d", len(train_data))                # len(train_data): 整数，训练样本总数（如321）
    
    val_dataloader: DataLoader = DataLoader(
        val_data, 
        **cfg.dataloader.params,                                            # 数据加载器参数
        shuffle=False                                                       # 验证时不打乱数据顺序
    )
    logger.info("Num validation samples: %d", len(val_data))                # len(val_data): 整数，验证样本总数（如75）

    # 创建PyTorch Lightning训练器，管理整个训练流程
    logger.info("Building Trainer")
    trainer: pl.Trainer = pl.Trainer(
        **cfg.trainer.params,                                               # 训练器参数，如 max_epochs=10, accelerator='gpu'
        callbacks=agent.get_training_callbacks()                            # 训练回调函数列表，如模型保存、学习率调度等
    )

    # 开始训练
    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,                                             # Lightning模块实例，包含模型和损失函数
        train_dataloaders=train_dataloader,                                 # 训练数据加载器
        # val_dataloaders=val_dataloader,                                   # 验证数据加载器（当前注释掉，暂不进行验证）
    )


if __name__ == "__main__":
    # 当脚本直接运行时调用main函数（Hydra入口）
    main()
