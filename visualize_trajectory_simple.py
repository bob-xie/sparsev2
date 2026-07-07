#!/usr/bin/env python3
"""
轨迹对比可视化简化脚本
直接在项目根目录运行
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import hydra
from hydra.utils import instantiate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent
from navsim.visualization.plots import plot_bev_with_agent, frame_plot_to_gif, plot_bev_frame


def load_agent(ckpt_path: str, cfg: dict) -> SparseDriveAgent:
    """加载训练好的智能体"""
    agent: SparseDriveAgent = instantiate(cfg)
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('agent.'):
                new_k = k[6:]
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        agent.load_state_dict(new_state_dict, strict=False)
    elif 'agent' in checkpoint:
        agent.load_state_dict(checkpoint['agent'])
    else:
        agent.load_state_dict(checkpoint)
    
    agent.eval()
    return agent


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='轨迹对比可视化')
    parser.add_argument('--token', required=True, help='场景 token')
    parser.add_argument('--ckpt', default="exp/sparsedrive_agent/2026.06.17.17.46.52/periodic_pdm_ckpts/ep0010.ckpt", help='模型 checkpoint 路径')
    parser.add_argument('--output', default="exp/visualization/trajectory", help='输出目录')
    args = parser.parse_args()
    
    os.environ['OPENSCENE_DATA_ROOT'] = '/home/xqb/DATA2/E2E_Project/sparsev2'
    os.environ['NAVSIM_DEVKIT_ROOT'] = '/home/xqb/DATA2/E2E_Project/sparsev2'
    os.environ['NAVSIM_EXP_ROOT'] = '/home/xqb/DATA2/E2E_Project/sparsev2/exp'
    os.environ['NUPLAN_MAPS_ROOT'] = '/home/xqb/DATA2/E2E_Project/sparsev2/maps/nuplan-maps-v1.0'
    
    token = args.token
    ckpt_path = args.ckpt
    output_dir = args.output
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    hydra.initialize(config_path="./navsim/planning/script/config/common/train_test_split/scene_filter", version_base=None)
    cfg_filter = hydra.compose(config_name="all_scenes")
    scene_filter: SceneFilter = instantiate(cfg_filter)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    
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
    
    print(f"加载模型: {ckpt_path}")
    hydra.initialize(config_path="./navsim/planning/script/config/training", version_base=None)
    cfg_train = hydra.compose(config_name='default_training', overrides=[
        'train_test_split=navmini',
        'agent=sparsedrive_agent',
    ])
    
    agent = load_agent(ckpt_path, cfg_train.agent)
    print("模型加载成功")
    
    print("绘制轨迹对比图...")
    
    agent_input = scene.get_agent_input()
    features = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))
    
    if hasattr(agent.get_feature_builders()[0], 'pipeline'):
        features, _, _ = agent.get_feature_builders()[0].pipeline(features, {}, token, True)
    
    batch_features = {}
    for k, v in features.items():
        if isinstance(v, torch.Tensor):
            batch_features[k] = v.unsqueeze(0)
        elif isinstance(v, dict):
            sub_batch = {}
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, torch.Tensor):
                    sub_batch[sub_k] = sub_v.unsqueeze(0)
                elif isinstance(sub_v, np.ndarray):
                    sub_batch[sub_k] = torch.tensor(sub_v).unsqueeze(0)
                else:
                    sub_batch[sub_k] = sub_v
            batch_features[k] = sub_batch
        elif isinstance(v, np.ndarray):
            batch_features[k] = torch.tensor(v).unsqueeze(0)
        else:
            batch_features[k] = v
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = agent.to(device)
    
    for k, v in batch_features.items():
        if isinstance(v, torch.Tensor):
            batch_features[k] = v.to(device)
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, torch.Tensor):
                    v[sub_k] = sub_v.to(device)
    
    with torch.no_grad():
        predictions, _ = agent.forward(batch_features, {})
        poses = predictions["trajectory"].squeeze(0).cpu().numpy()
    
    from navsim.common.dataclasses import Trajectory
    agent_trajectory = Trajectory(poses, agent._trajectory_sampling)
    human_trajectory = scene.get_future_trajectory()
    
    import matplotlib.pyplot as plt
    from navsim.visualization.plots import add_configured_bev_on_ax, add_trajectory_to_bev_ax, configure_bev_ax, configure_ax
    from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG
    
    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)
    
    fig.savefig(output_path / f'trajectory_{token}.png', dpi=150, bbox_inches='tight')
    print(f"保存轨迹对比图: {output_path / f'trajectory_{token}.png'}")
    
    print("生成轨迹动画 GIF...")
    
    from navsim.visualization.plots import plot_bev_frame
    
    frame_plot_to_gif(
        file_name=str(output_path / f'trajectory_{token}.gif'),
        callable_frame_plot=plot_bev_frame,
        scene=scene,
        frame_indices=list(range(scene.scene_metadata.num_history_frames, len(scene.frames))),
        duration=300,
    )
    print(f"保存轨迹动画: {output_path / f'trajectory_{token}.gif'}")


if __name__ == "__main__":
    main()
