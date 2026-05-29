# SparseDriveV2 完整学习与部署指南

## 目录

1. [理论学习阶段](#阶段-1理论学习)
2. [环境搭建阶段](#阶段-2x86电脑环境搭建)
3. [本地训练与评估](#阶段-3x86本地训练与评估)
4. [模型导出与优化](#阶段-4模型导出与优化)
5. [Orin 部署](#阶段-5orin部署)

---

## 阶段 1：理论学习

### 1.1 核心论文与资源

| 资源 | 链接/来源 | 说明 |
|------|---------|-----|
| **论文** | [arXiv:2603.29163](https://arxiv.org/abs/2603.29163) | SparseDriveV2 原论文 |
| **代码** | 本仓库 | `/workspace/SparseDriveV2` |
| **基础** | 了解端到端自动驾驶、注意力机制、轨迹预测 | 前置知识 |

### 1.2 核心概念学习

#### 1.2.1 SparseDriveV2 核心创新

1. **分解式轨迹词汇表 (Factorized Trajectory Vocabulary)**
   - 将轨迹分解为几何路径 (Geometric Path) 和速度剖面 (Velocity Profile)
   - 组合覆盖动作空间，大幅增加轨迹密度

2. **两阶段评分策略 (Two-stage Scoring)**
   - 粗评分：对路径和速度分别评分，快速筛选
   - 细评分：对组合后的轨迹进行精细评分

3. **轻量级 Backbone**
   - 使用 ResNet-34，兼顾速度与性能

#### 1.2.2 关键代码模块阅读顺序

| 顺序 | 文件位置 | 学习内容 |
|------|---------|---------|
| 1 | [`navsim/agents/sparsedrive/sparsedrive_config.py`](file:///workspace/SparseDriveV2/navsim/agents/sparsedrive/sparsedrive_config.py) | 配置和超参数 |
| 2 | [`navsim/agents/sparsedrive/sparsedrive_model.py`](file:///workspace/SparseDriveV2/navsim/agents/sparsedrive/sparsedrive_model.py) | 主模型结构 |
| 3 | [`navsim/agents/sparsedrive/custom_decoder.py`](file:///workspace/SparseDriveV2/navsim/agents/sparsedrive/custom_decoder.py) | 解码器和评分机制 |
| 4 | [`navsim/agents/sparsedrive/blocks.py`](file:///workspace/SparseDriveV2/navsim/agents/sparsedrive/blocks.py) | 可变形特征聚合 |
| 5 | [`navsim/agents/sparsedrive/sparsedrive_backbone.py`](file:///workspace/SparseDriveV2/navsim/agents/sparsedrive/sparsedrive_backbone.py) | 视觉 Backbone |

---

## 阶段 2：x86 电脑环境搭建

### 2.1 硬件要求（推荐）

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **GPU** | NVIDIA GPU 6GB VRAM | NVIDIA RTX 3060/4070 12GB+ |
| **CPU** | 4+ 核 | 8+ 核 |
| **内存** | 16GB | 32GB+ |
| **存储** | 100GB+ SSD | 200GB+ SSD |

### 2.2 环境安装（参考 [`INSTALL_GUIDE.md`](file:///workspace/SparseDriveV2/INSTALL_GUIDE.md)）

#### 2.2.1 基础依赖

```bash
# 1. 确认 CUDA 可用
nvidia-smi  # 应该看到 NVIDIA GPU 信息

# 2. 安装 Conda (如果还没有)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 3. 创建并激活环境
cd /workspace/SparseDriveV2
conda env create -f environment.yml
conda activate navsim

# 4. 安装 Python 依赖
pip install -r requirements.txt
```

#### 2.2.2 安装 SparseDriveV2 自定义 CUDA 扩展

```bash
cd navsim/agents/sparsedrive/ops
python setup.py develop
```

### 2.3 数据准备

#### 2.3.1 下载数据

```bash
# 提供的下载脚本
cd /workspace/SparseDriveV2

# 下载 mini 数据集（快速尝鲜）
sh download/download_mini.sh

# 或下载完整 navtrain 数据集
sh download/download_navtrain_hf.sh
sh download/download_maps.sh
```

#### 2.3.2 数据缓存

```bash
# 运行数据集缓存脚本
sh scripts/cache/run_dataset_caching_navtrain.sh

# 运行指标缓存
sh scripts/cache/run_metric_caching_navtrain_v2.sh
```

#### 2.3.3 锚点准备

```bash
# 下载预训练的锚点，或自己生成
mkdir -p ckpt/kmeans
# 从 https://huggingface.co/wenchaosun/SparseDriveV2 下载并放到 ckpt/kmeans/

# 或者自己聚类生成
python scripts/cluster/cluster_anchor.py
```

#### 2.3.4 检查点准备

```bash
mkdir -p ckpt

# 1. 下载 ResNet-34 backbone
# 从 https://huggingface.co/timm/resnet34.a1_in1k 下载并放到 ckpt/resnet34.bin

# 2. 下载 SparseDriveV2 预训练权重（可选）
# 从 https://huggingface.co/wenchaosun/SparseDriveV2 下载
```

---

## 阶段 3：x86 本地训练与评估

### 3.1 训练

#### 3.1.1 使用提供的脚本

```bash
cd /workspace/SparseDriveV2

# NavSimV2 训练
sh scripts/training/sparsedrive_navsimv2.sh

# 或 NavSimV1 训练
sh scripts/training/sparsedrive_navsimv1.sh
```

#### 3.1.2 自定义训练参数

你可以直接修改脚本或使用 Hydra 参数：

```bash
# 示例：调整 batch size 和 epoch
python navsim/planning/script/run_training.py \
    --config-name default_training \
    agent=sparsedrive_agent \
    dataloader.params.batch_size=8 \
    trainer.params.max_epochs=5 \
    agent.lr=0.0001
```

### 3.2 评估

```bash
# NavSimV2 评估
sh scripts/evaluation/run_pdm_score_navtest_v2.sh

# NavSimV1 评估
sh scripts/evaluation/run_pdm_score_navtest_v1.sh
```

### 3.3 训练监控

- 查看 `output_dir` 中的训练日志
- 使用 TensorBoard（如果配置了）
- 检查保存的 checkpoint 文件

---

## 阶段 4：模型导出与优化

### 4.1 导出为 TorchScript（用于 LibTorch 部署）

关键步骤：将 `use_deformable_func` 设置为 **False**

#### 4.1.1 创建导出脚本

```python
# /workspace/SparseDriveV2/export_model.py
import torch
from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel

def main():
    # 1. 加载配置
    from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
    config = SparseDriveConfig()
    config.use_deformable_func = False  # 重要！使用纯 PyTorch 实现
    
    # 2. 创建模型
    model = SparseDriveModel(config)
    
    # 3. 加载训练好的权重
    checkpoint = torch.load("path/to/your/checkpoint.ckpt")
    model.load_state_dict(checkpoint["state_dict"])
    
    # 4. 设置为评估模式
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # 5. 导出 TorchScript
    # 这里需要根据实际输入创建 dummy input
    # 可以使用 trace 或 script 方法
    print("TorchScript 导出需要根据实际输入形状调整")
    
if __name__ == "__main__":
    main()
```

### 4.2 可选：模型量化（用于 Orin 加速）

```python
# 量化模型（使用 PyTorch 自带的量化或 TensorRT）
import torch.quantization
```

---

## 阶段 5：Orin 部署

### 5.1 Orin 环境准备

#### 5.1.1 硬件要求

- Jetson Orin Nano / Orin NX / Orin AGX
- JetPack 5.1+ / JetPack 6.0+

#### 5.1.2 软件安装

在 Orin 上：

```bash
# 1. 安装 JetPack（使用 NVIDIA SDK Manager）

# 2. 安装 Miniconda (aarch64 版本)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh

# 3. 创建环境
# 可以使用与 x86 相同的 environment.yml，或调整为 aarch64 版本

# 4. 注意：Orin 上不需要编译自定义 CUDA 扩展（使用纯 PyTorch 实现）
```

### 5.2 方案选择

| 方案 | 难度 | 性能 | 灵活性 | 推荐度 |
|------|------|------|-------|-------|
| **方案 A：纯 Python 部署** | 简单 | 良好 | 高 | ⭐⭐⭐⭐⭐ |
| **方案 B：LibTorch (C++)** | 中等 | 更好 | 中 | ⭐⭐⭐ |
| **方案 C：TensorRT** | 难 | 最好 | 低 | ⭐⭐ |

#### 5.2.1 方案 A：Orin 上纯 Python 部署（推荐）

简单直接，不需要 LibTorch，直接使用 PyTorch：

```bash
# 在 Orin 上
# 1. 传输模型权重和导出的 TorchScript
# 2. 直接使用 Python 加载和推理
python inference_script.py
```

#### 5.2.2 方案 B：LibTorch (C++) 部署

参考我们的 [`PURE_PYTORCH_IMPLEMENTATION.md`](file:///workspace/SparseDriveV2/PURE_PYTORCH_IMPLEMENTATION.md)：

```bash
# 在 x86 上导出模型后
# 将 TorchScript 文件和 C++ 代码传到 Orin
# 在 Orin 上编译和运行
```

### 5.3 Orin 性能优化

```bash
# 1. 开启 Orin 性能模式
sudo nvpmodel -m 0  # MAXN 模式
sudo jetson_clocks

# 2. 或使用自定义电源模式
```

### 5.4 部署验证

1. 使用测试数据集验证
2. 记录推理延迟和吞吐量
3. 对比 x86 和 Orin 的性能

---

## 完整流程总结

```
┌───────────────────────────────────────────────────────────────────┐
│  1. 理论学习                                                      │
│     - 阅读论文                                                    │
│     - 理解分解式词汇表和两阶段评分                                │
└───────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│  2. x86 环境搭建                                                  │
│     - 安装 Conda 环境                                             │
│     - 编译 CUDA 扩展                                              │
│     - 下载和准备数据                                              │
└───────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│  3. 训练与评估                                                    │
│     - 运行训练脚本                                                │
│     - 评估模型性能                                                │
│     - 保存检查点                                                  │
└───────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│  4. 模型导出                                                      │
│     - 设置 use_deformable_func=False                              │
│     - 导出 TorchScript                                           │
└───────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│  5. Orin 部署                                                     │
│     - 传输模型和代码                                              │
│     - 选择部署方案（Python 或 C++）                               │
│     - 测试与优化                                                  │
└───────────────────────────────────────────────────────────────────┘
```

---

## 常见问题

### Q1: x86 没有 GPU 可以吗？
A: 可以学习代码，但训练会非常慢。建议使用云 GPU 或有 NVIDIA GPU 的电脑。

### Q2: Orin 上可以用自定义 CUDA 扩展吗？
A: 可以，但需要在 Orin 上本地编译。对于部署，推荐使用 `use_deformable_func=False` 的纯 PyTorch 版本。

### Q3: 数据太大下载慢怎么办？
A: 可以先用 mini 数据集测试完整流程。

### Q4: 训练到部署可以端到端吗？
A: 是的，只要：
- 训练时用 `use_deformable_func=True` (加速训练)
- 导出时用 `use_deformable_func=False` (兼容部署)
- 权重可以直接共享

---

## 下一步

1. 阅读本指南和 [`docs/train_eval.md`](file:///workspace/SparseDriveV2/docs/train_eval.md)
2. 开始环境搭建和数据准备
3. 逐步完成各阶段
