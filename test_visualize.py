#!/usr/bin/env python3
"""
简单测试可视化脚本
"""

import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import plot_bev_frame


def main():
    os.environ['OPENSCENE_DATA_ROOT'] = '/home/xqb/DATA2/E2E_Project/sparsev2'
    
    hydra.initialize(config_path="./navsim/planning/script/config/common/train_test_split/scene_filter", version_base=None)
    cfg = hydra.compose(config_name="all_scenes")
    scene_filter: SceneFilter = instantiate(cfg)
    
    openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))
    
    scene_loader = SceneLoader(
        openscene_data_root / "navsim_logs/mini",
        openscene_data_root / "sensor_blobs/mini",
        scene_filter,
        sensor_config=SensorConfig.build_all_sensors(),
    )
    
    token = "1fc1dd0dc3d157ae"
    print(f"加载场景: {token}")
    scene = scene_loader.get_scene_from_token(token)
    print(f"场景加载成功, 帧数: {len(scene.frames)}")
    
    frame_idx = scene.scene_metadata.num_history_frames - 1
    print(f"绘制第 {frame_idx} 帧...")
    fig, ax = plot_bev_frame(scene, frame_idx)
    
    output_path = Path("exp/visualization/test")
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path / f'bev_{token}.png', dpi=150, bbox_inches='tight')
    print(f"保存成功: {output_path / f'bev_{token}.png'}")


if __name__ == "__main__":
    main()
