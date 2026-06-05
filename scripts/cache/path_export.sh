#!/bin/bash

# SparseDriveV2 数据路径配置
# 使用方法: source scripts/cache/path_export.sh

# 项目根目录
export NAVSIM_DEVKIT_ROOT=/home/xqb/DATA2/E2E_Project/sparsev2

# 数据集根目录
export OPENSCENE_DATA_ROOT=/home/xqb/DATA2/E2E_Project/sparsev2

# 实验输出目录
export NAVSIM_EXP_ROOT=/home/xqb/DATA2/E2E_Project/sparsev2/exp

# CUDA 工具包路径
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

# 更新 PATH 和 LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH

# 添加项目到 PYTHONPATH
export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:$PYTHONPATH

# 地图数据路径
export NUPLAN_MAPS_ROOT=$NAVSIM_DEVKIT_ROOT/maps/nuplan-maps-v1.0

# 打印配置信息
echo "=== SparseDriveV2 路径配置已加载 ==="
echo "NAVSIM_DEVKIT_ROOT: $NAVSIM_DEVKIT_ROOT"
echo "OPENSCENE_DATA_ROOT: $OPENSCENE_DATA_ROOT"
echo "NAVSIM_EXP_ROOT: $NAVSIM_EXP_ROOT"
echo "CUDA_HOME: $CUDA_HOME"