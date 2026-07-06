#!/bin/bash

# 场景可视化运行脚本
# 使用方式: bash scripts/visualization/run_visualize_scene.sh <token>

# 加载路径配置
source "$(dirname "$0")/../cache/path_export.sh"

# 激活 conda 环境
source /home/xqb/DATA2/E2E_Project/miniconda3/etc/profile.d/conda.sh
conda activate navsim

# 切换到项目根目录
cd "$NAVSIM_DEVKIT_ROOT"

# 运行可视化脚本
python scripts/visualization/visualize_scene.py \
    --token "$1" \
    --output "exp/visualization/scene"
