# Orin GPU 模型导出使用说明

## 概述

本脚本用于在 Orin 控制器上使用 GPU 重新导出模型，确保模型可以在 GPU 上正常运行。

## 问题背景

在训练机器上导出的模型（`model_scripted_cpu.pt`）是在 CPU 上 trace 的，某些操作在 GPU 上无法执行，会出现以下错误：

```
RuntimeError: GET was unable to find an engine to execute this computation
```

## 解决方案

在 Orin 上使用 GPU 重新导出模型，确保所有操作在 GPU 上可执行。

## 准备工作

### 1. 传输文件到 Orin

需要传输以下文件到 Orin：

```bash
# 传输模型代码
scp -r navsim/ root@orin_ip:/etc/lg/truck/test_torch/

# 传输导出脚本
scp scripts/deployment/export_torchscript_gpu.py root@orin_ip:/etc/lg/truck/test_torch/

# 传输训练好的 checkpoint
scp exp/sparsedrive_agent/2026.06.17.17.46.52/periodic_pdm_ckpts/ep0010.ckpt root@orin_ip:/etc/lg/truck/test_torch/
```

### 2. 安装依赖

确保 Orin 上已安装以下依赖：

```bash
# 检查 torch 版本
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

输出应该类似：
```
2.6.0+cu126
True
```

如果未安装，请使用 `lib_torch_orin.tar.gz` 离线安装。

## 使用方法

### 在 Orin 上执行

```bash
cd /etc/lg/truck/test_torch/

# 导出模型（使用 GPU）
python3 export_torchscript_gpu.py \
    --ckpt ep0010.ckpt \
    --output model_scripted_gpu.pt
```

### 命令参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ckpt` | 训练好的 checkpoint 文件路径（必填） | - |
| `--output` | 导出模型的保存路径 | `model_scripted_gpu.pt` |

### 预期输出

```
Loading checkpoint from: ep0010.ckpt
Using CUDA device
Model loaded successfully!
Running forward pass for trace...
Exporting to TorchScript with GPU...

Model exported to: model_scripted_gpu.pt

Verifying exported model...
✓ Verification successful!
  Device: cuda
  Output shape: torch.Size([1, 8, 3])
  Output example:
tensor([[0.xxx, 0.xxx, 0.xxx],
        [0.xxx, 0.xxx, 0.xxx],
        [0.xxx, 0.xxx, 0.xxx]], device='cuda:0')
```

## 测试导出的模型

### Python 测试

```bash
python3 -c "
import torch

model = torch.jit.load('model_scripted_gpu.pt')
model = model.cuda()
model.eval()

# 准备测试输入
B = 1
num_cams = 3
H, W = 256, 512

imgs = torch.randn(B, num_cams, 3, H, W).cuda()
status = torch.randn(B, 8).cuda()
lidar2img = torch.randn(B, num_cams, 4, 4).cuda()
lidar2cam = torch.randn(B, num_cams, 4, 4).cuda()
cam2lidar = torch.randn(B, num_cams, 4, 4).cuda()
cam_intrinsic = torch.randn(B, num_cams, 3, 3).cuda()

# 推理
with torch.no_grad():
    output = model(imgs, status, lidar2img, lidar2cam, cam2lidar, cam_intrinsic)

print('Output shape:', output.shape)
print('Output:', output)
"
```

### C++ 测试

```bash
cd /etc/lg/truck/test_torch/cpp/
./run.sh ../model_scripted_gpu.pt
```

## 注意事项

1. **必须在支持 CUDA 的设备上运行**：脚本会自动检测 CUDA 是否可用
2. **模型大小**：GPU 导出的模型大小与 CPU 版本类似（约 238MB）
3. **兼容性**：导出的模型只能在兼容的 CUDA 版本上运行
4. **checkpoint 文件**：需要使用训练好的 `.ckpt` 文件，不能使用已导出的 `.pt` 文件

## 常见问题

### Q1: 提示找不到 navsim 模块

**解决方案**：确保 navsim 目录已传输到 Orin，并且在正确的路径下。

```bash
ls /etc/lg/truck/test_torch/navsim/
```

### Q2: CUDA 不可用

**解决方案**：检查 torch 是否正确安装，并且 NVIDIA 驱动是否正常工作。

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Q3: 导出时内存不足

**解决方案**：Orin 的 GPU 内存约 61GB，通常足够。如果出现 OOM 错误，可以尝试：

```bash
# 清理 GPU 缓存
python3 -c "import torch; torch.cuda.empty_cache()"
```

## 完整流程

```
1. 传输文件到 Orin
   ├── navsim/ (模型代码)
   ├── export_torchscript_gpu.py (导出脚本)
   └── ep0010.ckpt (训练好的权重)

2. 在 Orin 上安装依赖
   └── torch-2.6.0+cu126 (使用 lib_torch_orin.tar.gz)

3. 在 Orin 上导出模型
   └── python3 export_torchscript_gpu.py --ckpt ep0010.ckpt --output model_scripted_gpu.pt

4. 测试模型
   ├── Python: python3 test_deployment.py --model model_scripted_gpu.pt
   └── C++: ./cpp/run.sh model_scripted_gpu.pt

5. 部署到生产环境
   └── 使用 model_scripted_gpu.pt 进行推理
```