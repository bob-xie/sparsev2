"""
PDM Score 快速评估脚本 (v2版本)

功能说明:
    该脚本用于对训练好的导航模型进行PDM(Planning Decision Making)评分评估。
    评估流程:
    1. 加载模型并进行推理，获取所有场景的预测轨迹
    2. 使用PDM模拟器和评分器对预测轨迹进行仿真和评分
    3. 计算两帧扩展舒适度指标
    4. 汇总所有场景的评分结果并输出

数据类说明:
    - AbstractAgent: 智能体抽象基类，定义了compute_trajectory等接口
    - PDMSimulator: PDM模拟器，用于轨迹仿真
    - PDMScorer: PDM评分器，用于计算各项指标
    - MetricCacheLoader: 指标缓存加载器，加载预计算的metric缓存
    - SceneLoader: 场景加载器，加载场景数据
    - SceneFilter: 场景过滤器，筛选特定的场景和token
    - CacheOnlyDataset: 仅使用缓存的数据集类
    - AgentLightningModule: Lightning模块包装器
    - PDMResults: PDM评估结果数据类，包含各项指标字段
    - TrajectorySampling: 轨迹采样配置
    - WeightedMetricIndex: 加权指标索引枚举
"""

# ==================== 系统标准库导入 ====================

import logging  # 日志模块，用于记录运行过程信息
import os  # 操作系统接口，用于环境变量、路径操作等
import traceback  # 异常堆栈跟踪，用于捕获和打印异常详细信息
import uuid  # 唯一标识符生成器，用于生成线程ID
from dataclasses import fields  # 数据类字段提取工具，用于获取PDMResults的字段名
from datetime import datetime  # 日期时间模块，用于生成时间戳
from pathlib import Path  # 路径处理模块，用于文件路径操作
from typing import Dict, List, Union  # 类型提示，用于函数参数和返回值类型标注
from functools import partial  # 偏函数工具，用于固定函数参数

# ==================== PyTorch相关导入 ====================

import torch.distributed as dist  # PyTorch分布式训练模块，用于多进程通信
from torch.utils.data import DataLoader  # PyTorch数据加载器，用于批量加载数据
import pytorch_lightning as pl  # PyTorch Lightning训练框架

# ==================== Hydra配置和第三方库导入 ====================

import hydra  # Hydra配置管理框架，用于动态加载配置和实例化对象
import numpy as np  # NumPy数值计算库
import pandas as pd  # Pandas数据分析库，用于处理评估结果
from hydra.utils import instantiate  # Hydra实例化工具，根据配置动态创建对象

# ==================== nuPlan相关导入 ====================

from nuplan.common.actor_state.state_representation import StateSE2  # SE2状态表示，包含x,y,heading
from nuplan.common.geometry.convert import relative_to_absolute_poses  # 相对位姿转绝对位姿
from nuplan.planning.script.builders.logging_builder import build_logger  # 日志构建器
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # 轨迹采样配置
from nuplan.planning.utils.multithreading.worker_utils import worker_map  # 多线程工作映射工具
from omegaconf import DictConfig  # OmegaConf配置字典类型

# ==================== NAVSIM项目内部导入 ====================

from navsim.agents.abstract_agent import AbstractAgent  # 智能体抽象基类
from navsim.common.dataclasses import PDMResults, SensorConfig  # 数据类：评估结果、传感器配置
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader  # 数据加载相关类
from navsim.common.enums import SceneFrameType  # 场景帧类型枚举(ORIGINAL/SYNTHETIC)
from navsim.evaluate.pdm_score import pdm_score  # PDM评分函数(原始版本)
from navsim.evaluate.pdm_score_fix_bug import pdm_score as pdm_score_fix_bug  # PDM评分函数(修复bug版本)
from navsim.planning.script.builders.worker_pool_builder import build_worker  # 工作池构建器
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer  # PDM评分器
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator  # 场景聚合器
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator  # PDM模拟器
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex  # 加权指标索引
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy  # 交通代理策略基类
from navsim.planning.training.dataset import CacheOnlyDataset  # 仅缓存数据集
from navsim.planning.training.agent_lightning_module import AgentLightningModule  # Lightning模块包装器

# ==================== 全局变量 ====================

logger = logging.getLogger(__name__)  # 创建日志记录器实例

CONFIG_PATH = "config/pdm_scoring"  # Hydra配置文件目录(相对路径)
CONFIG_NAME = "default_run_pdm_score_fast"  # 默认配置文件名


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]], pdm_score_fn) -> List[pd.DataFrame]:
    """
    PDM评估工作函数，在工作线程中执行单个或多个场景的评估
    
    参数说明:
        args: List[Dict] - 工作参数列表，每个元素包含一个log_file和对应的tokens
              每个dict结构: {"cfg": DictConfig, "log_file": str, "tokens": List[str], "model_trajectory": Dict}
        pdm_score_fn: 评分函数，可为pdm_score(原始版)或pdm_score_fix_bug(修复版)
    
    返回值:
        List[pd.DataFrame] - 每个场景的评估结果DataFrame列表
    """
    # 获取节点ID，用于分布式评估时标识不同节点
    node_id = int(os.environ.get("NODE_RANK", 0))
    # 生成唯一线程ID，用于日志追踪
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    # 从args中提取所有log文件名
    log_names = [a["log_file"] for a in args]
    # 从args中提取所有tokens(展平嵌套列表)
    tokens = [t for a in args for t in a["tokens"]]
    # 获取配置对象(所有args共享同一个cfg)
    cfg: DictConfig = args[0]["cfg"]
    # 获取模型预测轨迹字典 {token: Trajectory}
    model_trajectory = args[0]['model_trajectory']
    # 过滤出模型有预测结果的tokens
    tokens = [t for t in tokens if t in model_trajectory]

    # 根据配置实例化PDM模拟器(用于轨迹仿真)
    simulator: PDMSimulator = instantiate(cfg.simulator)
    # 根据配置实例化PDM评分器(用于计算各项指标)
    scorer: PDMScorer = instantiate(cfg.scorer)
    # 断言校验：模拟器和评分器的轨迹采样配置必须一致
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    # 根据配置选择交通代理策略
    if cfg.traffic_agents == "non_reactive":
        # 非反应式策略：交通代理按固定轨迹行驶
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
        )
    elif cfg.traffic_agents == "reactive":
        # 反应式策略：交通代理会根据ego车辆的行为做出反应
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
        )
    
    # 创建指标缓存加载器，用于加载预计算的metric缓存
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    # 根据配置实例化场景过滤器
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    # 设置要评估的log文件名列表
    scene_filter.log_names = log_names
    # 设置要评估的token列表
    scene_filter.tokens = tokens
    # 创建场景加载器
    scene_loader = SceneLoader(
        original_sensor_path=Path(cfg.original_sensor_path),  # 原始传感器数据路径
        data_path=Path(cfg.navsim_log_path),  # 日志数据路径
        scene_filter=scene_filter,  # 场景过滤器
    )

    # 取scene_loader和metric_cache_loader中都存在的tokens作为最终评估列表
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    # 存储每个场景的评估结果
    pdm_results: List[pd.DataFrame] = []
    
    # 遍历所有待评估的token
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            # 从metric缓存加载该token的预计算指标
            metric_cache = metric_cache_loader.get_from_token(token)
            # 获取模型对该token的预测轨迹
            trajectory = model_trajectory[token]

            # 调用PDM评分函数进行评估
            # 返回值: score_row(各项指标DataFrame), ego_simulated_states(仿真后的ego状态序列)
            score_row, ego_simulated_states = pdm_score_fn(
                metric_cache=metric_cache,  # 预计算的metric缓存
                model_trajectory=trajectory,  # 模型预测轨迹
                future_sampling=simulator.proposal_sampling,  # 轨迹采样配置
                simulator=simulator,  # PDM模拟器
                scorer=scorer,  # PDM评分器
                traffic_agents_policy=traffic_agents_policy,  # 交通代理策略
            )
            
            # 标记该场景评估成功
            score_row["valid"] = True
            # 记录场景所属的log文件名
            score_row["log_name"] = metric_cache.log_name
            # 记录场景帧类型(ORIGINAL/SYNTHETIC)
            score_row["frame_type"] = metric_cache.scene_type
            # 记录场景开始时间(秒)
            score_row["start_time"] = metric_cache.timepoint.time_s
            
            # 创建预测轨迹终点的SE2状态对象
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],      # 终点x坐标(相对坐标系)
                y=trajectory.poses[-1, 1],      # 终点y坐标(相对坐标系)
                heading=trajectory.poses[-1, 2], # 终点朝向(相对坐标系)
            )
            # 将相对终点坐标转换为绝对世界坐标
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            # 记录终点绝对坐标
            score_row["endpoint_x"] = absolute_endpoint.x
            score_row["endpoint_y"] = absolute_endpoint.y
            # 记录起点绝对坐标(ego后轴位置)
            score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
            # 保存仿真后的ego状态序列(用于两帧扩展舒适度计算)
            score_row["ego_simulated_states"] = [ego_simulated_states]

        except Exception:
            # 评估失败时记录警告日志
            logger.warning(f"----------- Agent failed for token {token}:")
            # 打印异常堆栈信息
            traceback.print_exc()
            # 创建空结果DataFrame(所有指标为默认值)
            score_row = pd.DataFrame([PDMResults.get_empty_results()])
            # 标记该场景评估失败
            score_row["valid"] = False
        
        # 添加token字段标识该场景
        score_row["token"] = token

        # 将该场景的评估结果添加到列表中
        pdm_results.append(score_row)
    
    # 返回所有场景的评估结果列表
    return pdm_results


def infer_start_adjacent_mapping(score_df: pd.DataFrame, time_gap_threshold: float = 0.55) -> Dict[str, str]:
    """
    根据场景开始时间推断相邻帧映射关系
    
    功能说明:
        对于原始场景帧(ORIGINAL)，按log_name分组并按时间排序，
        将相邻时间间隔小于阈值的帧进行配对，用于计算两帧扩展舒适度指标
    
    参数说明:
        score_df: pd.DataFrame - 包含token, log_name, start_time字段的评分DataFrame
        time_gap_threshold: float - 相邻帧时间间隔阈值(秒)，默认0.55秒
    
    返回值:
        Dict[str, str] - 当前帧token到前一帧token的映射字典
    """
    # 初始化相邻帧映射字典
    adjacent_mapping: Dict[str, str] = {}

    # 仅处理原始场景帧(排除合成帧)，按log_name分组
    for log_name, group_df in score_df[score_df["frame_type"] == SceneFrameType.ORIGINAL].groupby("log_name"):
        # 按开始时间排序并重置索引
        group_df = group_df.sort_values(by="start_time").reset_index(drop=True)

        # 遍历排序后的帧，寻找相邻帧
        for i in range(1, len(group_df)):
            # 获取前一帧数据
            prev_row = group_df.iloc[i - 1]
            # 获取当前帧数据
            current_row = group_df.iloc[i]

            # 提取前一帧和当前帧的token
            prev_token = prev_row["token"]
            current_token = current_row["token"]
            # 计算时间差
            time_diff = current_row["start_time"] - prev_row["start_time"]

            # 如果时间差小于阈值，则建立相邻映射
            if abs(time_diff) <= time_gap_threshold:
                adjacent_mapping[current_token] = prev_token

    # 返回相邻帧映射字典
    return adjacent_mapping


def compute_final_scores(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算最终综合评分
    
    功能说明:
        将两帧扩展舒适度指标融入加权指标计算，最终得分 = 乘法指标乘积 × 加权指标得分
    
    参数说明:
        pdm_score_df: pd.DataFrame - 包含所有评估指标的DataFrame
    
    返回值:
        pd.DataFrame - 包含最终score字段的DataFrame
    """
    # 复制DataFrame避免修改原数据
    df = pdm_score_df.copy()

    # 提取两帧扩展舒适度分数数组 (shape: (N, ))
    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()
    # 提取加权指标数组 (shape: (N, M), M为指标数量)
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())
    # 提取加权权重数组 (shape: (N, M))
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())

    # 创建NaN掩码(两帧舒适度未计算的情况)
    mask = np.isnan(two_frame_scores)
    # 获取两帧扩展舒适度在加权指标中的索引
    two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT

    # 将NaN位置的指标值和权重设为0(不参与计算)
    weighted_metrics[mask, two_frame_idx] = 0.0
    weighted_metrics_array[mask, two_frame_idx] = 0.0

    # 创建非NaN掩码
    non_mask = ~mask
    # 将有效的两帧舒适度分数填入加权指标数组
    weighted_metrics[non_mask, two_frame_idx] = two_frame_scores[non_mask]

    # 计算加权和: sum(metric × weight)
    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    # 计算总权重
    total_weight = weighted_metrics_array.sum(axis=1)
    # 将总权重为0的情况设为NaN(避免除零错误)
    total_weight[total_weight == 0.0] = np.nan
    # 计算加权指标得分: weighted_sum / total_weight
    weighted_metric_scores = weighted_sum / total_weight

    # 计算最终得分: 乘法指标乘积 × 加权指标得分
    df["score"] = df["multiplicative_metrics_prod"].to_numpy() * weighted_metric_scores
    
    # 删除中间计算字段，只保留最终结果
    df.drop(
        columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"],
        inplace=True,
    )

    # 返回包含最终得分的DataFrame
    return df


def create_scene_aggregators(
    all_mappings: Dict[str, str],
    full_score_df: pd.DataFrame,
    proposal_sampling: TrajectorySampling,
) -> pd.DataFrame:
    """
    创建场景聚合器，计算两帧扩展舒适度指标
    
    参数说明:
        all_mappings: Dict[str, str] - 当前帧token到前一帧token的映射字典
        full_score_df: pd.DataFrame - 包含所有场景评估结果的DataFrame
        proposal_sampling: TrajectorySampling - 轨迹采样配置
    
    返回值:
        pd.DataFrame - 包含两帧扩展舒适度指标的DataFrame
    """
    # 初始化两帧扩展舒适度列为NaN
    full_score_df["two_frame_extended_comfort"] = np.nan
    # 将token设为索引，方便按token查找
    full_score_df = full_score_df.set_index("token")

    # 存储所有更新结果
    all_updates = []

    # 遍历所有相邻帧映射
    for now_frame, previous_frame in all_mappings.items():
        # 创建场景聚合器，用于计算两帧之间的舒适度
        aggregator = SceneAggregator(
            now_frame=now_frame,           # 当前帧token
            previous_frame=previous_frame, # 前一帧token
            score_df=full_score_df,        # 完整评分DataFrame
            proposal_sampling=proposal_sampling,  # 轨迹采样配置
        )
        # 计算两帧扩展舒适度(one_stage_only=True表示单阶段评估)
        updated_rows = aggregator.aggregate_scores(one_stage_only=True)

        # 将更新结果添加到列表
        all_updates.append(updated_rows)

    # 如果有更新结果(避免空列表导致concat报错)
    if all_updates:
        # 合并所有更新结果并按token设置索引
        all_updates_df = pd.concat(all_updates, ignore_index=True).set_index("token")
        # 更新full_score_df中的两帧舒适度值
        full_score_df.update(all_updates_df)
    
    # 重置索引(将token从索引恢复为列)
    full_score_df.reset_index(inplace=True)
    # 删除ego_simulated_states字段(已完成两帧舒适度计算，不再需要)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"])

    # 返回更新后的DataFrame
    return full_score_df


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    PDMS评估主入口函数
    
    参数说明:
        cfg: DictConfig - Hydra加载的配置字典，包含所有评估参数
    
    执行流程:
        1. 初始化模型和Lightning模块
        2. 加载测试数据集并进行推理预测
        3. 收集分布式预测结果
        4. 初始化场景加载器和metric缓存加载器
        5. 分布式执行PDM评分
        6. 计算两帧扩展舒适度指标
        7. 计算最终综合评分
        8. 保存评估结果到CSV文件
    """
    # 设置全局随机种子，确保评估结果可复现
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    # 记录输出目录路径
    logger.info(f"Path where all results are stored: {cfg.output_dir}")
    
    # 根据配置实例化智能体(如SparseDriveAgent)
    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    # 初始化智能体(加载checkpoint等)
    agent.initialize()

    # 创建Lightning模块包装器
    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent  # 传入智能体实例
    )

    # 创建测试数据集
    logger.info("Building Datasets")
    logger.info(f"Loading test set features from: {cfg.test_cache_path}")
    # 根据配置实例化测试场景过滤器
    test_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    # 创建仅使用缓存的测试数据集
    test_dataset = CacheOnlyDataset(
        cache_path=cfg.test_cache_path,           # 缓存数据路径
        feature_builders=agent.get_feature_builders(),  # 特征构建器列表
        target_builders=agent.get_target_builders(),    # 目标构建器列表
        log_names=test_scene_filter.log_names,    # 要评估的log文件名列表
    )
    # 创建测试数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.dataloader.params.batch_size,  # 批大小
        num_workers=cfg.dataloader.params.batch_size, # 工作进程数
        shuffle=False,              # 不打乱顺序
        drop_last=False,            # 不丢弃最后一批不足batch_size的数据
    )
    # 打印测试样本数量
    logger.info("Num test samples: %d", len(test_dataset))

    # 创建Lightning Trainer(用于推理)
    logger.info("Building Trainer")
    # 获取智能体的训练回调函数
    original_callbacks = agent.get_training_callbacks()
    callbacks = original_callbacks
    # 根据配置创建Trainer
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks)

    # 使用Trainer进行推理预测
    logger.info("Starting Validation")
    predictions = trainer.predict(
        model=lightning_module,   # Lightning模块
        dataloaders=test_dataloader,  # 测试数据加载器
    )

    # 分布式屏障(等待所有进程完成预测)
    dist.barrier()
    # 初始化预测结果收集列表(每个进程一个槽位)
    all_predictions = [None for _ in range(dist.get_world_size())]

    # 如果启用了分布式训练
    if dist.is_initialized():
        # 收集所有进程的预测结果
        dist.all_gather_object(all_predictions, predictions)
    else:
        # 非分布式模式，直接添加预测结果
        all_predictions.append(predictions)

    # 非主进程(非rank=0)直接返回，不参与后续处理
    if dist.get_rank() != 0:
        return None

    # 合并所有进程的预测结果
    merged_predictions = {}
    for proc_prediction in all_predictions:
        for d in proc_prediction:
            merged_predictions.update(d)

    # 构建日志记录器
    build_logger(cfg)
    # 根据配置创建工作池(用于并行评估)
    worker = build_worker(cfg)

    # 创建场景加载器(用于获取token列表，不加载传感器数据)
    scene_loader = SceneLoader(
        original_sensor_path=None,  # 不加载原始传感器数据(已在缓存中)
        data_path=Path(cfg.navsim_log_path),  # 日志数据路径
        scene_filter=instantiate(cfg.train_test_split.scene_filter),  # 场景过滤器
        sensor_config=SensorConfig.build_no_sensors(),  # 不使用传感器
    )
    # 创建指标缓存加载器
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    # 取scene_loader和metric_cache_loader中都存在的tokens
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    # 计算缺失metric缓存的token数量
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    # 计算未使用的metric缓存token数量
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    
    # 记录警告信息
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    
    # 记录开始评估的场景数量
    logger.info(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")
    
    # 构建工作参数列表，按log_file分组
    data_points = [
        {
            "cfg": cfg,                        # 配置对象
            "log_file": log_file,              # log文件名
            "tokens": tokens_list,             # 该log_file下的token列表
            "model_trajectory": merged_predictions  # 所有模型预测轨迹
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    
    ################################## bug_version (原始版本)
    # 使用工作池并行执行PDM评分(使用原始pdm_score函数)
    score_rows: List[pd.DataFrame] = worker_map(worker, partial(run_pdm_score, pdm_score_fn=pdm_score), data_points)
 
    # 合并所有场景的评分结果
    pdm_score_df = pd.concat(score_rows)

    # 推断相邻帧映射关系
    start_adjacent_mapping = infer_start_adjacent_mapping(pdm_score_df)
    # 创建场景聚合器，计算两帧扩展舒适度
    pdm_score_df = create_scene_aggregators(
        start_adjacent_mapping, pdm_score_df, instantiate(cfg.simulator.proposal_sampling)
    )
    # 计算最终综合评分
    pdm_score_df = compute_final_scores(pdm_score_df)

    # 统计成功和失败的场景数量
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    # 获取失败的token列表
    if num_failed_scenarios > 0:
        failed_tokens = pdm_score_df[~pdm_score_df["valid"]]["token"].to_list()
    else:
        failed_tokens = []

    # 筛选需要保存的评分列(PDMResults字段 + two_frame_extended_comfort + score)
    score_cols = [
        c
        for c in pdm_score_df.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]

    # 计算所有场景的平均评分
    average_row = pdm_score_df[score_cols].mean(skipna=True)
    average_row["token"] = "average_all_frames"
    average_row["valid"] = pdm_score_df["valid"].all()

    # 只保留需要的列
    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
    # 将平均评分行添加到DataFrame末尾
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    # 获取保存路径
    save_path = Path(cfg.output_dir)
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    # 保存原始版本评估结果到CSV
    pdm_score_df.to_csv(save_path / "navtest_v2.csv")

    # 记录评估完成信息
    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / "navtest_v2.csv"}.
        """
    )

    # 如果启用详细模式，打印最后三行详细结果
    if cfg.verbose:
        logger.info(
            f"""
            Detailed results:
            {pdm_score_df.iloc[-3:].T}
            """
        )
    # 如果有失败场景，打印失败token列表
    if num_failed_scenarios > 0:
        logger.info(
            f"""
            List of failed tokens:
            {failed_tokens}
            """
        )

    ################################## bug_fix version (修复bug版本)
    # 使用工作池并行执行PDM评分(使用修复bug后的pdm_score_fix_bug函数)
    score_rows: List[pd.DataFrame] = worker_map(worker, partial(run_pdm_score, pdm_score_fn=pdm_score_fix_bug), data_points)

    # 合并所有场景的评分结果
    pdm_score_df = pd.concat(score_rows)

    # 推断相邻帧映射关系
    start_adjacent_mapping = infer_start_adjacent_mapping(pdm_score_df)
    # 创建场景聚合器，计算两帧扩展舒适度
    pdm_score_df = create_scene_aggregators(
        start_adjacent_mapping, pdm_score_df, instantiate(cfg.simulator.proposal_sampling)
    )
    # 计算最终综合评分
    pdm_score_df = compute_final_scores(pdm_score_df)

    # 统计成功和失败的场景数量
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    # 获取失败的token列表
    if num_failed_scenarios > 0:
        failed_tokens = pdm_score_df[~pdm_score_df["valid"]]["token"].to_list()
    else:
        failed_tokens = []

    # 筛选需要保存的评分列
    score_cols = [
        c
        for c in pdm_score_df.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]

    # 计算所有场景的平均评分
    average_row = pdm_score_df[score_cols].mean(skipna=True)
    average_row["token"] = "average_all_frames"
    average_row["valid"] = pdm_score_df["valid"].all()

    # 只保留需要的列
    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
    # 将平均评分行添加到DataFrame末尾
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    # 获取保存路径
    save_path = Path(cfg.output_dir)
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    # 保存修复bug版本评估结果到CSV
    pdm_score_df.to_csv(save_path / "navtest_v2_bug_fix.csv")

    # 记录评估完成信息
    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / "navtest_v2_bug_fix.csv"}.
        """
    )

    # 如果启用详细模式，打印最后三行详细结果
    if cfg.verbose:
        logger.info(
            f"""
            Detailed results:
            {pdm_score_df.iloc[-3:].T}
            """
        )
    # 如果有失败场景，打印失败token列表
    if num_failed_scenarios > 0:
        logger.info(
            f"""
            List of failed tokens:
            {failed_tokens}
            """
        )


if __name__ == "__main__":
    # 程序入口，调用main函数(Hydra装饰器会自动处理配置加载)
    main()