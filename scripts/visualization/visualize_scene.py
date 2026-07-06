#!/usr/bin/env python3
"""
场景可视化脚本
可视化单个场景的摄像头图像、BEV鸟瞰图、LiDAR点云
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

project_root = Path(__file__).parent.parent.parent

os.environ['OPENSCENE_DATA_ROOT'] = str(project_root)
os.environ['NAVSIM_DEVKIT_ROOT'] = str(project_root)
os.environ['NAVSIM_EXP_ROOT'] = str(project_root / 'exp')
os.environ['NUPLAN_MAPS_ROOT'] = str(project_root / 'maps/nuplan-maps-v1.0')

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import (
    plot_bev_frame,
    plot_cameras_frame,
    plot_cameras_frame_with_lidar,
    plot_cameras_frame_with_annotations,
    frame_plot_to_gif,
)


def main():
    parser = argparse.ArgumentParser(description='场景可视化')
    parser.add_argument('--token', required=True, help='场景 token')
    parser.add_argument('--output', default='exp/visualization/scene', help='输出目录')
    parser.add_argument('--frame', type=int, default=3, help='要可视化的帧索引')
    args = parser.parse_args()
    
    token = args.token
    output_dir = args.output
    frame_idx = args.frame
    
    output_path = Path(project_root / output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"当前工作目录: {os.getcwd()}")
    print(f"project_root: {project_root}")
    
    os.chdir(str(project_root))
    
    print(f"切换后工作目录: {os.getcwd()}")
    
    hydra.initialize(config_path="./navsim/planning/script/config/training", version_base=None)
    cfg = hydra.compose(config_name='default_training', overrides=[
        'train_test_split=navmini',
    ])
    
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
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
    print(f"场景帧数: {len(scene.frames)}, 历史帧数: {scene.scene_metadata.num_history_frames}")
    
    # 1. 绘制单帧 BEV 图
    print("绘制 BEV 鸟瞰图...")
    fig, ax = plot_bev_frame(scene, frame_idx)
    fig.savefig(output_path / f'bev_{token}.png', dpi=150, bbox_inches='tight')
    print(f"保存 BEV 图: {output_path / f'bev_{token}.png'}")
    
    # 2. 绘制摄像头 + BEV 全景图
    print("绘制摄像头 + BEV 全景图...")
    fig, ax = plot_cameras_frame(scene, frame_idx)
    fig.savefig(output_path / f'cameras_{token}.png', dpi=150, bbox_inches='tight')
    print(f"保存摄像头图: {output_path / f'cameras_{token}.png'}")
    
    # 3. 绘制摄像头 + LiDAR 投影图
    print("绘制摄像头 + LiDAR 投影图...")
    try:
        fig, ax = plot_cameras_frame_with_lidar(scene, frame_idx)
        fig.savefig(output_path / f'cameras_lidar_{token}.png', dpi=150, bbox_inches='tight')
        print(f"保存 LiDAR 投影图: {output_path / f'cameras_lidar_{token}.png'}")
    except Exception as e:
        print(f"LiDAR 可视化失败: {e}")
    
    # 4. 绘制摄像头 + 标注框图
    print("绘制摄像头 + 标注框图...")
    fig, ax = plot_cameras_frame_with_annotations(scene, frame_idx)
    fig.savefig(output_path / f'cameras_annotations_{token}.png', dpi=150, bbox_inches='tight')
    print(f"保存标注框图: {output_path / f'cameras_annotations_{token}.png'}")
    
    # 5. 生成 BEV 动画 GIF
    print("生成 BEV 动画 GIF...")
    frame_plot_to_gif(
        file_name=str(output_path / f'bev_animation_{token}.gif'),
        callable_frame_plot=plot_bev_frame,
        scene=scene,
        frame_indices=list(range(len(scene.frames))),
        duration=300,
    )
    print(f"保存 BEV 动画: {output_path / f'bev_animation_{token}.gif'}")


if __name__ == "__main__":
    main()
