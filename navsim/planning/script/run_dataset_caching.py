"""
数据集特征缓存脚本 - Dataset Feature Caching Script

该脚本的核心功能是将原始传感器数据（图像、激光雷达等）预处理为模型可直接使用的特征格式，
并序列化存储到磁盘，以加速后续训练过程。

主要流程：
1. 初始化工作池（支持多线程/分布式）
2. 加载场景数据（SceneLoader）
3. 按日志文件分组数据点
4. 并行执行特征提取和缓存
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def cache_features(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
    """
    特征缓存核心函数 - 负责处理一组数据点的特征提取和缓存

    :param args: 缓存参数列表，每个元素包含:
                 - cfg: 配置对象
                 - log_file: 日志文件名
                 - tokens: 该日志文件对应的场景 token 列表
    :return: 返回空列表（此函数主要用于副作用，即写入缓存文件）
    """
    print("\n" + "="*80)
    print("🔄 [cache_features] 开始处理数据点...")
    print("="*80)

    # 获取当前节点 ID（分布式训练时使用，默认为 0）
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    print(f"📌 [cache_features] 节点ID: {node_id}, 线程ID: {thread_id}")

    # 从参数中提取所有日志文件名
    log_names = [a["log_file"] for a in args]
    print(f"📂 [cache_features] 本次处理的日志文件数量: {len(log_names)}")
    print(f"   日志文件列表: {log_names}")

    # 展平所有 tokens 到一个列表
    tokens = [t for a in args for t in a["tokens"]]
    print(f"🔑 [cache_features] 本次处理的场景token数量: {len(tokens)}")

    # 获取配置对象（所有参数共享同一个配置）
    cfg: DictConfig = args[0]["cfg"]
    print(f"⚙️  [cache_features] 配置信息:")
    print(f"   - agent: {cfg.agent}")
    print(f"   - cache_path: {cfg.cache_path}")
    print(f"   - force_cache_computation: {cfg.force_cache_computation}")

    # 实例化智能体（如 SparseDriveAgent）
    # 智能体包含特征构建器和目标构建器
    print(f"\n🤖 [cache_features] 步骤1: 实例化智能体...")
    agent: AbstractAgent = instantiate(cfg.agent)
    print(f"   ✅ 智能体实例化成功: {type(agent).__name__}")

    # 获取并显示特征构建器信息
    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()
    print(f"\n📦 [cache_features] 步骤2: 获取构建器...")
    print(f"   - 特征构建器数量: {len(feature_builders)}")
    for i, fb in enumerate(feature_builders):
        print(f"     [{i+1}] {type(fb).__name__}")
    print(f"   - 目标构建器数量: {len(target_builders)}")
    for i, tb in enumerate(target_builders):
        print(f"     [{i+1}] {type(tb).__name__}")

    # 实例化场景过滤器（用于过滤需要处理的场景）
    print(f"\n🎯 [cache_features] 步骤3: 创建场景过滤器...")
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    print(f"   ✅ 场景过滤器配置完成")
    print(f"   - 过滤的日志文件: {log_names}")
    print(f"   - 过滤的token数量: {len(tokens)}")

    # 创建场景加载器（负责加载原始数据）
    print(f"\n📁 [cache_features] 步骤4: 创建场景加载器...")
    print(f"   路径配置:")
    print(f"   - synthetic_sensor_path: {cfg.synthetic_sensor_path}")
    print(f"   - original_sensor_path: {cfg.original_sensor_path}")
    print(f"   - data_path: {cfg.navsim_log_path}")
    print(f"   - synthetic_scenes_path: {cfg.synthetic_scenes_path}")

    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    print(f"   ✅ 场景加载器创建成功")
    print(f"   - 加载的场景数量: {len(scene_loader.tokens)}")

    # 创建数据集对象（核心：执行特征提取和缓存）
    print(f"\n💾 [cache_features] 步骤5: 创建数据集并执行缓存...")
    print(f"   缓存配置:")
    print(f"   - cache_path: {cfg.cache_path}") #sparsev2/exp/data_cache_navmini_log_save
    print(f"   - force_cache_computation: {cfg.force_cache_computation}")
    print(f"   开始缓存数据，这可能需要一些时间...")

    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        cfg=cfg,
    )

    print(f"   ✅ 数据集缓存完成！")
    print(f"   已处理 {len(scene_loader.tokens)} 个场景")
    print("="*80)
    print(f"✅ [cache_features] 线程 {thread_id} 处理完成")
    print("="*80 + "\n")

    return []


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None) #Hydra 自动加载机制
def main(cfg: DictConfig) -> None:
    """
    主入口函数 - 数据集缓存脚本的起点

    :param cfg: OmegaConf 配置字典，由 Hydra 自动加载
    """
    print("\n" + "="*80)
    print("🚀 [main] 数据集缓存脚本启动")
    print("="*80)

    # 设置全局随机种子为 0（确保结果可复现）
    print(f"\n🎲 [main] 步骤1: 设置随机种子")
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)
    print(f"   ✅ 随机种子设置为 0")

    # 显示配置信息
    print(f"\n📋 [main] 步骤2: 加载配置信息")
    print(f"   配置路径: {CONFIG_PATH}/{CONFIG_NAME}") #config/training/default_training
    print(f"   关键配置:")
    print(f"   - agent: {cfg.agent}") #agent=sparsedrive_agent  “agent/sparsedrive_agent.yaml “
    print(f"   - train_test_split: {cfg.train_test_split}")  #“train_test_split/navmini.yaml”
    print(f"   - cache_path: {cfg.cache_path}")  #/home/xqb/DATA2/E2E_Project/sparsev2/exp/data_cache_navmini_log_save
    print(f"   - worker: {cfg.worker}")

    # 构建工作池（支持 sequential/ray 等模式）
    print(f"\n🔧 [main] 步骤3: 创建工作池")
    print(f"   worker类型: {cfg.worker}") #config/common/worker/sequential.yaml
    worker: WorkerPool = instantiate(cfg.worker)
    print(f"   ✅ 工作池创建成功: {type(worker).__name__}")

    # 构建场景加载器（用于初步加载场景元数据）
    print(f"\n📂 [main] 步骤4: 创建初始场景加载器（仅加载元数据）")
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)

    data_path = Path(cfg.navsim_log_path) #sparsev2/navsim_logs/mini
    synthetic_sensor_path = Path(cfg.synthetic_sensor_path)
    original_sensor_path = Path(cfg.original_sensor_path)

    print(f"   路径配置:")
    print(f"   - navsim_log_path: {cfg.navsim_log_path}") #navsim/planning/script/config/common/default_dataset_paths.yaml有定义/home/xqb/DATA2/E2E_Project/sparsev2/navsim_logs/mini
    print(f"   - synthetic_sensor_path: {cfg.synthetic_sensor_path}") #sparsev2/navhard_two_stage/sensor_blobs
    print(f"   - original_sensor_path: {cfg.original_sensor_path}") #/sparsev2/sensor_blobs/mini
    print(f"   - synthetic_scenes_path: {cfg.synthetic_scenes_path}") #sparsev2/navhard_two_stage/synthetic_scene_pickles

    # 创建初始场景加载器（使用空传感器配置，仅加载元数据）
    scene_loader = SceneLoader(
        synthetic_sensor_path=synthetic_sensor_path,
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),  # 空传感器配置
    )
    print(f"   ✅ 场景加载器创建成功")
    print(f"   - 检测到的场景总数: {len(scene_loader)}")

    # 获取每个日志文件的token列表
    tokens_per_log = scene_loader.get_tokens_list_per_log()
    print(f"\n📊 [main] 步骤5: 统计日志文件分布")
    print(f"   日志文件总数: {len(tokens_per_log)}")
    for log_name, tokens in list(tokens_per_log.items())[:5]:  # 只显示前5个
        print(f"   - {log_name}: {len(tokens)} 个场景")
    if len(tokens_per_log) > 5:
        print(f"   ... 还有 {len(tokens_per_log) - 5} 个日志文件")

    # 按日志文件分组数据点（每个日志文件作为一个处理单元）
    print(f"\n📦 [main] 步骤6: 准备数据点列表")
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    print(f"   数据点总数: {len(data_points)}") #52 因为navmini里面只有52个
    print(f"   每个数据点代表一个日志文件及其所有场景")

    # 显示数据点详情
    print(f"\n📋 [main] 数据点详情（前5个）:")
    for i, dp in enumerate(data_points[:5]):
        print(f"   [{i+1}] log_file: {dp['log_file']}, tokens数量: {len(dp['tokens'])}")
    if len(data_points) > 5:
        print(f"   ... 还有 {len(data_points) - 5} 个数据点")

    # 使用工作池并行执行缓存任务
    print(f"\n▶️  [main] 步骤7: 开始并行缓存")
    print(f"   开始时间: (见下方进度条)")
    print(f"   工作池类型: {type(worker).__name__}")
    print("-"*80)

    _ = worker_map(worker, cache_features, data_points)

    print("-"*80)
    print(f"✅ [main] 步骤8: 缓存完成")
    print(f"   已处理 {len(scene_loader)} 个场景")
    print(f"   缓存输出路径: {cfg.cache_path}")
    print("="*80)
    print("🎉 [main] 数据集缓存脚本执行完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()