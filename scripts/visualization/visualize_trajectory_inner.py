#!/usr/bin/env python3
"""
轨迹对比可视化内部脚本
在项目根目录下运行
"""

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
os.chdir(str(project_root))

import matplotlib
matplotlib.use('Agg')

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent
from navsim.visualization.plots import plot_bev_with_agent, frame_plot_to_gif


def load_scene_by_token(token: str, cfg: DictConfig) -> 'Scene':
    """根据 token 加载场景"""
    scene_loader = SceneLoader(
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_all_sensors(),
    )
    
    return scene_loader.get_scene_from_token(token)


def load_agent(ckpt_path: str, cfg: DictConfig) -> SparseDriveAgent:
    """加载训练好的智能体"""
    agent: SparseDriveAgent = instantiate(cfg.agent)
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['state_dict'])
    elif 'agent' in checkpoint:
        agent.load_state_dict(checkpoint['agent'])
    else:
        agent.load_state_dict(checkpoint)
    
    agent.eval()
    return agent


def main():
    """主函数"""
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
    
    with hydra.initialize(config_path='navsim/planning/script/config/training', version_base=None):
        cfg = hydra.compose(config_name='default_training', overrides=[
            'agent=sparsedrive_agent',
            'train_test_split=navmini',
        ])
        
        print(f"加载场景: {token}")
        scene = load_scene_by_token(token, cfg)
        
        print(f"加载模型: {ckpt_path}")
        agent = load_agent(ckpt_path, cfg)
        
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
