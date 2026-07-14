# SparseDriveV2 模型部署指南

本文档详细说明如何使用 `export_onnx_direct.py` 和 `export_tensorrt_engine.py` 脚本进行模型导出和部署。

---

## 目录

1. [工作流程概览](#1-工作流程概览)
2. [前置依赖](#2-前置依赖)
3. [脚本1：ONNX 导出 (export_onnx_direct.py)](#3-脚本1onnx-导出-export_onnx_directpy)
4. [脚本2：TensorRT 引擎编译 (export_tensorrt_engine.py)](#4-脚本2tensorrt-引擎编译-export_tensorrt_enginepy)
5. [完整部署流程](#5-完整部署流程)
6. [常见问题](#6-常见问题)
7. [附录：输入输出格式](#7-附录输入输出格式)

---

## 1. 工作流程概览

```
.ckpt (训练权重)
    │
    ▼  [本地电脑]
export_onnx_direct.py
    │
    ▼
model.onnx (中间格式)
    │
    └── scp 传输到 Orin
    │
    ▼  [Orin 控制器]
export_tensorrt_engine.py
    │
    ▼
model.engine (TensorRT 优化引擎)
    │
    ▼
C++ 推理程序加载并运行
```

| 步骤 | 执行位置 | 脚本 | 输出 |
|------|----------|------|------|
| 1. 导出 ONNX | 本地电脑 | `export_onnx_direct.py` | `model.onnx` |
| 2. 传输文件 | SCP | - | - |
| 3. 编译引擎 | Orin | `export_tensorrt_engine.py` | `model.engine` |
| 4. 运行推理 | Orin | C++ 程序 | 预测轨迹 |

---

## 2. 前置依赖

### 2.1 本地电脑（ONNX 导出）

```bash
# 安装依赖
conda activate navsim
pip install torch torchvision onnx onnxruntime
```

### 2.2 Orin 控制器（TensorRT 引擎编译）

```bash
# 检查 TensorRT 版本（要求 >= 8.0）
python3 -c "import tensorrt as trt; print(f'TensorRT: {trt.__version__}')"
# 输出示例: TensorRT: 10.3.0

# 检查 CUDA 版本
nvcc --version
# 输出示例: release 12.5
```

---

## 3. 脚本1：ONNX 导出 (export_onnx_direct.py)

### 3.1 功能说明

将训练好的 `.ckpt` 模型直接导出为 ONNX 格式，**跳过 TorchScript 中间步骤**，避免 Transformer 层的 ONNX 兼容性问题。

### 3.2 使用方法

```bash
cd /path/to/sparsev2

# 激活环境
source scripts/cache/path_export.sh

# 基础用法
python scripts/deployment/export_onnx_direct.py \
    --ckpt exp/sparsedrive_agent/xxx/periodic_pdm_ckpts/ep0010.ckpt \
    --output exp/deployment/model.onnx

# 指定 opset 版本
python scripts/deployment/export_onnx_direct.py \
    --ckpt ep0010.ckpt \
    --output model.onnx \
    --opset 17
```

### 3.3 命令行参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--ckpt` | str | ✅ | - | 训练好的模型权重路径（.ckpt 文件） |
| `--output` | str | ❌ | model.onnx | ONNX 模型输出路径 |
| `--opset` | int | ❌ | 17 | ONNX 算子集版本（建议 17） |

### 3.4 输出示例

```
============================================================
SparseDriveV2 ONNX Direct Export
============================================================

Loading checkpoint from: exp/sparsedrive_agent/xxx/periodic_pdm_ckpts/ep0010.ckpt

Creating model with simplified config (bypassing nuplan)...
✓ Model loaded successfully!

Running forward pass for trace...
✓ Forward pass completed!

Exporting to ONNX (opset version: 17)...
✓ ONNX model exported to: exp/deployment/model.onnx

Verifying ONNX model...
✓ ONNX model is valid!
✓ ONNX vs PT output max difference: 0.000123
✓ Output consistency check passed!

ONNX model info:
  Inputs: ['imgs', 'status_feature', 'lidar2img', 'lidar2cam', 'cam2lidar', 'cam_intrinsic']
  Outputs: ['trajectory']
    imgs: ['batch_size', 3, 3, 256, 512]
    status_feature: ['batch_size', 8]
    lidar2img: ['batch_size', 3, 4, 4]
    lidar2cam: ['batch_size', 3, 4, 4]
    cam2lidar: ['batch_size', 3, 4, 4]
    cam_intrinsic: ['batch_size', 3, 3, 3]
    trajectory: ['batch_size', 8, 3]

============================================================
ONNX export completed!
============================================================
```

---

## 4. 脚本2：TensorRT 引擎编译 (export_tensorrt_engine.py)

### 4.1 功能说明

将 ONNX 模型编译为针对 Orin GPU (sm_87) 优化的推理引擎，支持 FP16 混合精度加速。

**⚠️ 必须在 Orin 上执行**，因为引擎是针对特定 GPU 架构编译的。

### 4.2 使用方法

```bash
# 在 Orin 上执行
cd /disk1/lg/truck/test_torch

# 基础用法（启用 FP16）
python export_tensorrt_engine.py \
    --onnx model_direct.onnx \
    --output model.engine \
    --fp16

# 使用 FP32 精度
python export_tensorrt_engine.py \
    --onnx model_direct.onnx \
    --output model.engine \
    --fp32

# 指定工作空间大小（2GB）和批次大小
python export_tensorrt_engine.py \
    --onnx model_direct.onnx \
    --output model.engine \
    --fp16 \
    --workspace 2 \
    --batch-size 1
```

### 4.3 命令行参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--onnx` | str | ✅ | - | ONNX 模型文件路径 |
| `--output` | str | ❌ | model.engine | TensorRT 引擎输出路径 |
| `--fp16` | flag | ❌ | True | 启用 FP16 混合精度 |
| `--fp32` | flag | ❌ | False | 使用 FP32 精度（覆盖 --fp16） |
| `--workspace` | int | ❌ | 1 | 工作空间大小（GB） |
| `--batch-size` | int | ❌ | 1 | 推理批次大小 |

### 4.4 输出示例

```
============================================================
SparseDriveV2 TensorRT Engine Build
============================================================

Building engine from ONNX: model_direct.onnx
[TRT] [I] [MemUsageChange] Init CUDA: CPU +13, GPU +0, now: CPU 35, GPU 3527 (MiB)

Parsing ONNX model...
✓ ONNX model parsed successfully!
  Input 'imgs': shape=(-1, 3, 3, 256, 512)
  Input 'status_feature': shape=(-1, 8)
  Input 'lidar2img': shape=(-1, 3, 4, 4)
  ✓ Set shape for 'imgs' using string + list
  ✓ Set shape for 'status_feature' using string + list
  ✓ Set shape for 'lidar2img' using string + list

✓ FP16 mode enabled

Building engine (this may take several minutes)...
✓ Engine built successfully!

Saving engine to: model.engine
✓ Engine saved successfully!

============================================================
Engine Build Summary
============================================================
  ONNX path: model_direct.onnx
  Output path: model.engine
  FP16 enabled: True
  Workspace size: 1.0 GB
  Batch size: 1
  Number of layers: 124
  Number of inputs: 6
  Number of outputs: 1
  Input 0: imgs - (1, 3, 3, 256, 512)
  Input 1: status_feature - (1, 8)
  Input 2: lidar2img - (1, 3, 4, 4)
  Input 3: lidar2cam - (1, 3, 4, 4)
  Input 4: cam2lidar - (1, 3, 4, 4)
  Input 5: cam_intrinsic - (1, 3, 3, 3)
  Output 0: trajectory - (1, 8, 3)

============================================================
TensorRT engine build completed!
============================================================
```

---

## 5. 完整部署流程

### 5.1 步骤1：本地导出 ONNX

```bash
# 在本地电脑上
cd /home/xqb/DATA2/E2E_Project/sparsev2
source scripts/cache/path_export.sh

# 导出 ONNX（使用最新的 checkpoint）
python scripts/deployment/export_onnx_direct.py \
    --ckpt exp/sparsedrive_agent/2026.06.17.17.46.52/periodic_pdm_ckpts/ep0010.ckpt \
    --output exp/deployment/model_direct.onnx

# 验证文件是否生成
ls -lh exp/deployment/model_direct.onnx
```

### 5.2 步骤2：传输到 Orin

```bash
# 将 ONNX 文件和编译脚本传输到 Orin
scp exp/deployment/model_direct.onnx root@orin-ip:/disk1/lg/truck/test_torch/
scp scripts/deployment/export_tensorrt_engine.py root@orin-ip:/disk1/lg/truck/test_torch/
```

### 5.3 步骤3：Orin 上编译引擎

```bash
# 在 Orin 上
cd /disk1/lg/truck/test_torch

# 编译 TensorRT 引擎（FP16）
python export_tensorrt_engine.py \
    --onnx model_direct.onnx \
    --output model.engine \
    --fp16

# 验证引擎文件
ls -lh model.engine
```

### 5.4 步骤4：运行 C++ 推理

```bash
# 在 Orin 上编译并运行 C++ 程序
cd /path/to/cpp
bash build.sh
./sparsedrive_infer --engine model.engine --test
```

---

## 6. 常见问题

### 6.1 ONNX 导出失败

**问题**：`ModuleNotFoundError: No module named 'nuplan'`

**解决**：脚本已内置简化配置，无需安装 nuplan。确保运行前执行：
```bash
source scripts/cache/path_export.sh
```

### 6.2 TensorRT 编译错误

**问题**：`AttributeError: 'tensorrt.Builder' object has no attribute 'build_engine'`

**原因**：TensorRT 10.x API 变化，`build_engine` 已被移除。

**解决**：使用最新版脚本，已兼容新旧 API。

### 6.3 FP16 不支持

**问题**：`FP16 not supported on this platform`

**解决**：使用 `--fp32` 参数回退到 FP32 精度：
```bash
python export_tensorrt_engine.py --onnx model.onnx --output model.engine --fp32
```

### 6.4 内存不足

**问题**：`CUDA out of memory`

**解决**：
1. 增加工作空间：`--workspace 2`（2GB）
2. 使用 FP16 模式（默认启用）
3. 减小 batch_size：`--batch-size 1`

### 6.5 ONNX 与 PyTorch 输出不一致

**问题**：`Output difference is larger than expected`

**解决**：差异小于 1e-3 属于正常范围（浮点精度差异）。若差异过大，检查：
1. `opset` 版本是否正确（建议 17）
2. 模型是否在 eval 模式
3. 是否使用了相同的输入数据

---

## 7. 附录：输入输出格式

### 7.1 输入格式

| 输入名称 | 形状 | 说明 |
|----------|------|------|
| `imgs` | `[B, 3, 3, 256, 512]` | 多视角相机图像（B=batch, 3个相机, RGB, H=256, W=512） |
| `status_feature` | `[B, 8]` | Ego 状态（4命令 + 2速度 + 2加速度） |
| `lidar2img` | `[B, 3, 4, 4]` | LiDAR到图像投影矩阵 |
| `lidar2cam` | `[B, 3, 4, 4]` | LiDAR到相机外参矩阵 |
| `cam2lidar` | `[B, 3, 4, 4]` | 相机到LiDAR变换矩阵 |
| `cam_intrinsic` | `[B, 3, 3, 3]` | 相机内参矩阵 |

### 7.2 输出格式

| 输出名称 | 形状 | 说明 |
|----------|------|------|
| `trajectory` | `[B, 8, 3]` | 预测轨迹（8个点，每个点包含 x, y, heading） |

### 7.3 形状参数说明

```python
B = 1              # batch_size，推理时固定为1
num_cams = 3       # 相机数量（cam_l0, cam_f0, cam_r0）
H, W = 256, 512    # 图像尺寸
num_poses = 8      # 轨迹点数量（4秒预测，每0.5秒一个点）
```

---

## 版本兼容性

| TensorRT 版本 | 测试状态 | 说明 |
|--------------|----------|------|
| 10.x | ✅ 通过 | 需要使用 `build_serialized_network` API |
| 9.x | ✅ 通过 | 兼容新旧 API |
| 8.x | ✅ 通过 | 使用 `build_engine` API |
| < 8.0 | ⚠️ 未测试 | 可能需要调整 API 调用方式 |

---

**最后更新**: 2026-07-14  
**适用项目**: SparseDriveV2  
**目标平台**: NVIDIA Jetson AGX Orin (sm_87)
