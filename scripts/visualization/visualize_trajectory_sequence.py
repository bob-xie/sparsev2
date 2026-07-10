#!/usr/bin/env python3
"""
时间序列轨迹对比可视化脚本
在场景的每个时间步使用当前帧输入预测未来轨迹，并展示整个时间序列中轨迹的演变过程
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).parent.parent.parent
os.chdir(str(project_root))

os.environ['OPENSCENE_DATA_ROOT'] = str(project_root)
os.environ['NAVSIM_DEVKIT_ROOT'] = str(project_root)
os.environ['NAVSIM_EXP_ROOT'] = str(project_root / 'exp')
os.environ['NUPLAN_MAPS_ROOT'] = str(project_root / 'maps/nuplan-maps-v1.0')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import hydra
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig, Scene, AgentInput, EgoStatus, Cameras, Lidar, Trajectory
from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent
from navsim.visualization.plots import configure_bev_ax, configure_ax
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array
from nuplan.common.actor_state.state_representation import StateSE2


def get_agent_input_at_frame(scene: Scene, frame_idx: int) -> AgentInput:
    """
    获取指定帧索引处的agent输入
    使用从该帧开始的历史帧数据，如果历史数据不足则使用可用数据

    :param scene: navsim scene dataclass
    :param frame_idx: 当前帧索引（作为历史帧的最后一帧）
    :return: agent input dataclass
    """
    num_history_frames = scene.scene_metadata.num_history_frames
    
    start_idx = max(0, frame_idx - num_history_frames + 1)
    
    global_ego_poses = []
    for idx in range(start_idx, frame_idx + 1):
        global_ego_poses.append(scene.frames[idx].ego_status.ego_pose)
    
    origin = StateSE2(*global_ego_poses[-1])
    local_ego_poses = convert_absolute_to_relative_se2_array(origin, np.array(global_ego_poses, dtype=np.float64))
    
    ego_statuses: List[EgoStatus] = []
    cameras: List[Cameras] = []
    lidars: List[Lidar] = []
    
    for i, idx in enumerate(range(start_idx, frame_idx + 1)):
        frame_ego_status = scene.frames[idx].ego_status
        ego_statuses.append(
            EgoStatus(
                ego_pose=local_ego_poses[i],
                ego_velocity=frame_ego_status.ego_velocity,
                ego_acceleration=frame_ego_status.ego_acceleration,
                driving_command=frame_ego_status.driving_command,
            )
        )
        cameras.append(scene.frames[idx].cameras)
        lidars.append(scene.frames[idx].lidar)
    
    return AgentInput(ego_statuses, cameras, lidars)


def get_future_trajectory_from_frame(scene: Scene, frame_idx: int) -> Trajectory:
    """
    获取从指定帧开始的未来轨迹（ground truth）

    :param scene: navsim scene dataclass
    :param frame_idx: 当前帧索引
    :return: 未来轨迹
    """
    num_future_frames = scene.scene_metadata.num_future_frames
    
    global_ego_poses = []
    for idx in range(frame_idx, min(frame_idx + num_future_frames, len(scene.frames))):
        global_ego_poses.append(scene.frames[idx].ego_status.ego_pose)
    
    origin = StateSE2(*global_ego_poses[0])
    local_ego_poses = convert_absolute_to_relative_se2_array(origin, np.array(global_ego_poses, dtype=np.float64))
    
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    
    return Trajectory(
        local_ego_poses,
        TrajectorySampling(
            num_poses=len(local_ego_poses),
            interval_length=0.5,
        ),
    )


def load_agent(ckpt_path: str, cfg: dict) -> SparseDriveAgent:
    agent: SparseDriveAgent = instantiate(cfg)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'agent' in checkpoint:
        state_dict = checkpoint['agent']
    else:
        state_dict = checkpoint
    
    state_dict = {k.replace("agent.", ""): v for k, v in state_dict.items()}
    agent.load_state_dict(state_dict)
    agent.eval()
    
    if torch.cuda.is_available():
        agent = agent.cuda()
        print("模型已移至 CUDA")
    
    return agent


def plot_bev_with_sequence_trajectory(scene: Scene, agent: SparseDriveAgent, frame_idx: int) -> tuple:
    """
    绘制单个帧的BEV视图，包含当前帧的预测轨迹和ground truth轨迹

    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :param frame_idx: 当前帧索引
    :return: figure and ax object of matplotlib
    """
    agent_input = get_agent_input_at_frame(scene, frame_idx)
    agent_trajectory = agent.compute_trajectory(agent_input)
    human_trajectory = get_future_trajectory_from_frame(scene, frame_idx)
    
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)
    
    ax.set_title(f"Frame {frame_idx}", fontsize=12)
    
    return fig, ax


def plot_sequence_summary(scene: Scene, agent: SparseDriveAgent, start_frame: int, end_frame: int) -> tuple:
    """
    绘制时间序列轨迹汇总图，展示所有帧的预测轨迹叠加

    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :param start_frame: 起始帧索引
    :param end_frame: 结束帧索引
    :return: figure and ax object of matplotlib
    """
    num_history_frames = scene.scene_metadata.num_history_frames
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    frame_idx = start_frame
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    
    history_poses = []
    for idx in range(start_frame + 1):
        history_poses.append(scene.frames[idx].ego_status.ego_pose)
    
    history_poses = np.array(history_poses, dtype=np.float64)
    origin = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)
    local_history = convert_absolute_to_relative_se2_array(origin, history_poses)
    ax.plot(local_history[:, 1], local_history[:, 0], 'k-', linewidth=2, label='History')
    
    all_human_x = []
    all_human_y = []
    all_agent_x = []
    all_agent_y = []
    
    for frame_idx in tqdm(range(start_frame, end_frame + 1), desc="Computing trajectories"):
        agent_input = get_agent_input_at_frame(scene, frame_idx)
        agent_trajectory = agent.compute_trajectory(agent_input)
        human_trajectory = get_future_trajectory_from_frame(scene, frame_idx)
        
        all_human_x.extend(human_trajectory.poses[:, 1])
        all_human_y.extend(human_trajectory.poses[:, 0])
        all_agent_x.extend(agent_trajectory.poses[:, 1])
        all_agent_y.extend(agent_trajectory.poses[:, 0])
    
    ax.scatter(all_human_x, all_human_y, c='blue', s=20, alpha=0.6, label='Human Future (all frames)')
    ax.scatter(all_agent_x, all_agent_y, c='red', s=20, alpha=0.6, label='Agent Prediction (all frames)')
    
    configure_bev_ax(ax)
    configure_ax(ax)
    
    ax.legend(fontsize=10)
    ax.set_title("Trajectory Prediction Sequence Summary", fontsize=14)
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description='时间序列轨迹对比可视化')
    parser.add_argument('--token', required=True, help='场景 token')
    parser.add_argument('--ckpt', required=True, help='模型 checkpoint 路径')
    parser.add_argument('--output', default='exp/visualization/trajectory_sequence', help='输出目录')
    parser.add_argument('--start-frame', type=int, default=None, help='起始帧索引')
    parser.add_argument('--end-frame', type=int, default=None, help='结束帧索引')
    args = parser.parse_args()
    
    token = args.token
    ckpt_path = args.ckpt
    output_dir = args.output
    
    output_path = Path(project_root / output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"项目根目录: {project_root}")
    print(f"当前工作目录: {os.getcwd()}")
    
    hydra.initialize(config_path="../../navsim/planning/script/config/common/train_test_split/scene_filter", version_base=None)
    cfg_filter = hydra.compose(config_name="all_scenes")
    scene_filter: SceneFilter = instantiate(cfg_filter)
    
    openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))
    
    scene_loader = SceneLoader(
        openscene_data_root / "navsim_logs/mini",
        openscene_data_root / "sensor_blobs/mini",
        scene_filter,
        sensor_config=SensorConfig.build_all_sensors(),
    )
    
    print(f"加载场景: {token}")
    scene = scene_loader.get_scene_from_token(token)
    print(f"场景加载成功, 帧数: {len(scene.frames)}")
    print(f"历史帧数: {scene.scene_metadata.num_history_frames}")
    print(f"未来帧数: {scene.scene_metadata.num_future_frames}")
    
    print(f"加载模型: {ckpt_path}")
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../../navsim/planning/script/config/training", version_base=None)
    cfg_train = hydra.compose(config_name='default_training', overrides=[
        'train_test_split=navmini',
        'agent=sparsedrive_agent',
    ])
    
    agent = load_agent(ckpt_path, cfg_train.agent)
    print("模型加载成功")
    
    num_history_frames = scene.scene_metadata.num_history_frames
    num_future_frames = scene.scene_metadata.num_future_frames
    
    start_frame = args.start_frame if args.start_frame is not None else num_history_frames - 1
    end_frame = args.end_frame if args.end_frame is not None else len(scene.frames) - 1
    
    start_frame = max(start_frame, 0)
    end_frame = min(end_frame, len(scene.frames) - 1)
    
    print(f"\n生成时间序列可视化，帧范围: [{start_frame}, {end_frame}]")
    print(f"注意: 部分帧的未来轨迹可能不完整（不足{num_future_frames}帧）")
    
    print("\n1. 生成时间序列轨迹汇总图...")
    fig, ax = plot_sequence_summary(scene, agent, start_frame, end_frame)
    fig.savefig(output_path / f'trajectory_sequence_summary_{token}.png', dpi=150, bbox_inches='tight')
    print(f"保存轨迹汇总图: {output_path / f'trajectory_sequence_summary_{token}.png'}")
    plt.close(fig)
    
    print("\n2. 生成时间序列轨迹动画 GIF...")
    frame_indices = list(range(start_frame, end_frame + 1))
    
    images = []
    for frame_idx in tqdm(frame_indices, desc="Rendering frames"):
        fig, ax = plot_bev_with_sequence_trajectory(scene, agent, frame_idx)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        images.append(Image.open(buf).copy())
        
        buf.close()
        plt.close(fig)
    
    total_duration_ms = 5000
    frame_duration = max(100, total_duration_ms // len(images))
    gif_path = output_path / f'trajectory_sequence_{token}.gif'
    images[0].save(str(gif_path), save_all=True, append_images=images[1:], duration=frame_duration, loop=0)
    print(f"保存轨迹动画: {gif_path} (时长: {len(images) * frame_duration / 1000:.1f}秒)")
    
    print("\n3. 生成逐帧预测轨迹图...")
    for frame_idx in tqdm(frame_indices, desc="Saving individual frames"):
        fig, ax = plot_bev_with_sequence_trajectory(scene, agent, frame_idx)
        fig.savefig(output_path / f'trajectory_frame_{frame_idx:04d}_{token}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n所有可视化文件已保存至: {output_path}")


if __name__ == "__main__":
    import io
    main()
