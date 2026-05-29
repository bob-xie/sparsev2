# SparseDriveV2 环境配置指南

## 一、环境要求

### 1.1 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **GPU** | NVIDIA GPU (8GB+ 显存) | NVIDIA RTX 3090 或更新 |
| **内存** | 16GB RAM | 32GB+ RAM |
| **存储** | 100GB+ 可用空间 | 200GB+ (用于数据集) |
| **CUDA** | CUDA 11.0+ | CUDA 11.8 或 12.1 |

### 1.2 软件要求

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| **Python** | 3.9 | 建议使用 conda 环境 |
| **PyTorch** | 2.0.1+ | 需要 CUDA 支持 |
| **CUDA** | 11.0+ | 用于 GPU 训练和 CUDA 扩展编译 |
| **cuDNN** | 8.0+ | 配合 CUDA 使用 |

## 二、安装步骤

### 2.1 创建 conda 环境

```bash
# 创建 conda 环境
conda create -n sparsedrive python=3.9 -y
conda activate sparsedrive

# 进入项目目录
cd /path/to/SparseDriveV2
```

### 2.2 安装 PyTorch (CUDA 版本)

```bash
# 安装 PyTorch with CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### 2.3 安装项目依赖

```bash
# 安装核心依赖
pip install pytorch-lightning==2.2.1 tensorboard==2.16.2 timm scikit-learn positional-encodings

# 安装 nuplan-devkit (需要先安装)
pip install nuplan-devkit

# 安装地理信息相关库
pip install shapely geopandas rasterio rtree fiona pyogrio
```

### 2.4 编译 CUDA 扩展

```bash
# 进入 ops 目录
cd navsim/agents/sparsedrive/ops

# 编译 CUDA 扩展
python setup.py develop

# 返回项目根目录
cd ../..
```

### 2.5 下载数据和模型权重

```bash
# 创建权重目录
mkdir -p ckpt/kmeans ckpt/resnet34.bin

# 下载 Anchor 文件 (从 HuggingFace)
# https://huggingface.co/wenchaosun/SparseDriveV2
# - path_1024.npy
# - velocity_256.npy
# - trajectory_1024_256.npz

# 下载 ResNet-34 预训练权重
# https://huggingface.co/timm/resnet34.a1_in1k/blob/main/pytorch_model.bin

# 下载 SparseDriveV2 模型权重
# https://huggingface.co/wenchaosun/SparseDriveV2
# - sparsedrive_navsimv1_92p2.ckpt
# - sparsedrive_navsimv2_90p3.ckpt

# 下载 NAVSIM 数据集
bash download/download_navtrain_hf.sh
bash download/download_navtest.sh
```

### 2.6 数据预处理

```bash
# 数据缓存
sh scripts/cache/run_dataset_caching_navtrain.sh
sh scripts/cache/run_dataset_caching_navtest.sh

# 指标缓存 (navsimv1)
sh scripts/cache/run_metric_caching_navtrain_v1.sh
sh scripts/cache/run_metric_caching_navtest_v1.sh

# 指标缓存 (navsimv2)
sh scripts/cache/run_metric_caching_navtrain_v2.sh
sh scripts/cache/run_metric_caching_navtest_v2.sh
```

## 三、测试环境

### 3.1 运行测试

```bash
# 运行模型测试
python test_sparsedrive_simple.py

# 预期输出:
# ✓ PyTorch: 2.0.1+cu118
# ✓ CUDA 可用: True
# ✓ TIMM: xxx
# ✓ SparseBackbone 创建成功
# ✓ CustomTransformerDecoder 创建成功
```

### 3.2 运行训练 (可选)

```bash
# 训练模型 (navsimv1)
sh scripts/training/sparsedrive_navsimv1.sh

# 训练模型 (navsimv2)
sh scripts/training/sparsedrive_navsimv2.sh
```

### 3.3 运行评估

```bash
# 评估模型 (navsimv1)
sh scripts/evaluation/run_pdm_score_navtest_v1.sh

# 评估模型 (navsimv2)
sh scripts/evaluation/run_pdm_score_navtest_v2.sh
```

## 四、常见问题

### 4.1 CUDA 扩展编译失败

**问题**: `RuntimeError: CUDA was not found on the system`

**解决方案**:
1. 确认已安装 NVIDIA GPU
2. 安装 CUDA Toolkit
3. 设置环境变量:
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

### 4.2 缺少 nuplan 依赖

**问题**: `ModuleNotFoundError: No module named 'shapely'`

**解决方案**:
```bash
pip install nuplan-devkit shapely geopandas
```

### 4.3 GPU 显存不足

**问题**: `OutOfMemoryError`

**解决方案**:
1. 减小 batch_size
2. 使用更小的模型 (SparseDrive-S)
3. 使用梯度累积

### 4.4 数据集下载失败

**问题**: 网络问题导致数据集下载失败

**解决方案**:
1. 使用代理
2. 使用 AWS CLI 下载
3. 手动下载并放置到正确位置

## 五、快速验证清单

运行以下命令验证环境配置:

```bash
# 1. 检查 Python 版本
python --version  # 应该显示 Python 3.9.x

# 2. 检查 PyTorch 和 CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 3. 检查 CUDA 扩展
cd navsim/agents/sparsedrive/ops
python -c "from ops import deformable_aggregation_ext; print('CUDA 扩展正常')"

# 4. 检查模型导入
cd ../..
python -c "from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel; print('模型导入成功')"

# 5. 检查数据集
ls -la exp/data_cache_navtrain/  # 应该显示数据缓存目录
```

## 六、性能预期

| 配置 | GPU 显存 | 训练速度 | 评估速度 |
|------|---------|---------|---------|
| SparseDrive-S (小模型) | 8GB | ~2h/epoch | ~1min/scene |
| SparseDrive-B (基础模型) | 16GB | ~1h/epoch | ~30sec/scene |
| SparseDrive-L (大模型) | 24GB+ | ~30min/epoch | ~15sec/scene |

## 七、参考资源

- 项目主页: https://github.com/swc-17/SparseDriveV2
- 论文: https://arxiv.org/abs/2603.29163
- HuggingFace 模型: https://huggingface.co/wenchaosun/SparseDriveV2
- NAVSIM 数据集: https://github.com/autonomousvision/navsim
