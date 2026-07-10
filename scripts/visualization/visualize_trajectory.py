#!/usr/bin/env python3
"""
轨迹对比可视化脚本
对比训练模型的预测轨迹与人类驾驶轨迹（ground truth）
"""

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
os.chdir(str(project_root))

os.environ['OPENSCENE_DATA_ROOT'] = str(project_root)
os.environ['NAVSIM_DEVKIT_ROOT'] = str(project_root)
os.environ['NAVSIM_EXP_ROOT'] = str(project_root / 'exp')
os.environ['NUPLAN_MAPS_ROOT'] = str(project_root / 'maps/nuplan-maps-v1.0')

import matplotlib
matplotlib.use('Agg')

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent
from navsim.visualization.plots import plot_bev_with_agent, frame_plot_to_gif


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


def main():
    parser = argparse.ArgumentParser(description='轨迹对比可视化')
    parser.add_argument('--token', required=True, help='场景 token')
    parser.add_argument('--ckpt', required=True, help='模型 checkpoint 路径')
    parser.add_argument('--output', default='exp/visualization/trajectory', help='输出目录')
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
    
    print("绘制轨迹对比图...")
    fig, ax = plot_bev_with_agent(scene, agent)
    fig.savefig(output_path / f'trajectory_{token}.png', dpi=150, bbox_inches='tight')
    print(f"保存轨迹对比图: {output_path / f'trajectory_{token}.png'}")
    
    print("生成轨迹动画 GIF...")
    frame_plot_to_gif(
        file_name=str(output_path / f'trajectory_{token}.gif'),
        callable_frame_plot=lambda s, f: plot_bev_with_agent(s, agent),
        scene=scene,
        frame_indices=list(range(scene.scene_metadata.num_history_frames, len(scene.frames))),
        duration=300,
    )
    print(f"保存轨迹动画: {output_path / f'trajectory_{token}.gif'}")


if __name__ == "__main__":
    main()
