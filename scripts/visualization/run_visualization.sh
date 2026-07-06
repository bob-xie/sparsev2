#!/bin/bash

# 可视化脚本运行器
# 使用方式: bash scripts/visualization/run_visualization.sh <script_name> <args>

CONDA_PATH="/home/xqb/DATA2/E2E_Project/miniconda3"
PROJECT_ROOT="/home/xqb/DATA2/E2E_Project/sparsev2"
CONDA_ENV="navsim"

# 激活 conda 环境
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 运行指定的可视化脚本
python "scripts/visualization/$@"
