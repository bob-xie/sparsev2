from typing import Any, List, Dict, Optional, Union, Tuple, Sequence
import os
import multiprocessing as mp
import concurrent.futures as cf
from omegaconf import OmegaConf
from hydra.utils import instantiate
import lzma
import pickle

import numpy as np
import numpy.typing as npt
import torch

from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)

from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

from .pdm_score_v2 import pdm_score


def _init_pool():
    """
    进程池初始化函数，在每个子进程启动时执行一次
    
    功能：初始化PDM仿真器、评分器和交通代理策略，作为全局变量供子进程使用
    
    全局变量：
    - SIMULATOR: PDMSimulator实例，用于轨迹仿真
    - SCORER: PDMScorer实例，用于计算轨迹评分
    - TRAFFIC_AGENT_POLICY: 交通代理策略，用于模拟其他车辆行为
    """
    global SIMULATOR, SCORER, TRAFFIC_AGENT_POLICY
    pdm_cfg = OmegaConf.load('navsim/planning/script/config/pdm_scoring/run_pdm_train.yaml')
    SIMULATOR = instantiate(pdm_cfg.simulator)
    SCORER    = instantiate(pdm_cfg.scorer)
    SCORER.train_mode = True
    TRAFFIC_AGENT_POLICY = instantiate(
        pdm_cfg.non_reactive, SIMULATOR.proposal_sampling
    )


_pdm_pool = cf.ProcessPoolExecutor(
    max_workers=16,
    mp_context=mp.get_context("spawn"),
    initializer=_init_pool,
)

def get_pdm_score_para(trajectory, metric_cache_path):
    """
    使用多进程并行计算PDM轨迹评分
    
    参数：
    - trajectory: torch.Tensor，形状 [B, G, T, C]
                  B: batch size（批次大小）
                  G: 轨迹候选数量（如200条候选轨迹）
                  T: 轨迹时间步数（如8个时间点）
                  C: 轨迹维度（3，包含 x, y, heading）
    - metric_cache_path: List[str]，长度为B
                         每个元素是一个lzma压缩的pickle文件路径，包含该样本的度量缓存
    
    返回：
    - sub_scores: List[PDMResults]，长度为B
                   每个元素是一个PDMResults对象，包含该样本所有轨迹候选的评分结果
    """
    B, G = trajectory.shape[:2]
    traj_np = trajectory.detach().cpu().numpy()

    ## single worker debug
    debug = False
    if debug:
        pdm_cfg = OmegaConf.load('navsim/planning/script/config/pdm_scoring/run_pdm_train.yaml')
        SIMULATOR = instantiate(pdm_cfg.simulator)
        SCORER    = instantiate(pdm_cfg.scorer)
        SCORER.train_mode = True
        TRAFFIC_AGENT_POLICY = instantiate(pdm_cfg.non_reactive, SIMULATOR.proposal_sampling)

        with lzma.open(metric_cache_path[0], "rb") as f:
            metric_cache = pickle.load(f)

        results = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=traj_np[0],                # (G, T, C)
            future_sampling=SIMULATOR.proposal_sampling,
            simulator=SIMULATOR,                    # 全局对象，见 initializer
            scorer=SCORER,
            traffic_agents_policy=TRAFFIC_AGENT_POLICY,
        )
   
        return results

    futures = [
        _pdm_pool.submit(
            _pdm_worker,
            (metric_cache_path[b], traj_np[b]),
        )
        for b in range(B)
    ]

    sub_scores  = [f.result() for f in futures]
    return sub_scores


def _pdm_worker(args):
    """
    进程池工作函数，处理单个样本的PDM评分计算
    
    参数：
    - args: Tuple[str, np.ndarray]
            - cache: str，度量缓存文件路径（.xz格式）
            - traj_np: np.ndarray，形状 [G, T, C]
                       单个样本的G条候选轨迹
    
    返回：
    - results: PDMResults对象，包含G条轨迹的评分结果
    """
    cache, traj_np = args
    with lzma.open(cache, "rb") as f:
        metric_cache = pickle.load(f)
    
    results = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=traj_np,                # (G, T, C)
        future_sampling=SIMULATOR.proposal_sampling,
        simulator=SIMULATOR,                    # 全局对象，见 initializer
        scorer=SCORER,
        traffic_agents_policy=TRAFFIC_AGENT_POLICY,
    )
    return results