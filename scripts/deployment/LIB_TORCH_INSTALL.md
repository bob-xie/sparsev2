# lib_torch_orin.tar.gz 安装说明

## 概述

本压缩包包含在 Orin 控制器上离线安装 PyTorch 所需的所有文件，支持 CUDA 12.6。

## 文件内容

```
lib_torch_orin/
├── torch-2.6.0+cu126-cp310-cp310-linux_aarch64.whl       # PyTorch 主包（CUDA 12.6）
├── torchvision-0.21.0-cp310-cp310-linux_aarch64.whl    # torchvision
├── numpy-2.2.6-cp310-cp310-manylinux_2_17_aarch64.whl   # NumPy
├── sympy-1.13.1-py3-none-any.whl                       # sympy（torch 依赖）
├── typing_extensions-4.16.0-py3-none-any.whl            # typing-extensions
├── install.sh                                          # 安装脚本
└── verify.sh                                           # 验证脚本
```

## 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Ubuntu 22.04 LTS (aarch64) |
| Python | 3.10.x |
| CUDA | 12.6 |
| NVIDIA 驱动 | >= 540.4.0 |

## 安装步骤

### 步骤 1: 传输文件到 Orin

```bash
# 使用 scp 传输
scp lib_torch_orin.tar.gz root@orin_ip:/home/root/

# 或者使用 USB 拷贝
# 将文件复制到 USB 盘，然后挂载到 Orin 上
```

### 步骤 2: 解压文件

```bash
cd /home/root/
tar -xzvf lib_torch_orin.tar.gz
cd lib_torch_orin
```

### 步骤 3: 卸载旧版本（如果有）

```bash
pip3 uninstall torch torchvision -y
```

### 步骤 4: 运行安装脚本

```bash
./install.sh
```

### 步骤 5: 验证安装

```bash
./verify.sh
```

## 安装脚本内容说明

安装顺序很重要，必须按以下顺序安装：

```bash
# 1. sympy - torch 的依赖
pip3 install --no-index --find-links=. sympy-1.13.1-py3-none-any.whl

# 2. numpy - 基础数值计算库
pip3 install --no-index --find-links=. numpy-2.2.6-cp310-cp310-manylinux_2_17_aarch64.whl

# 3. typing-extensions - Python 类型支持
pip3 install --no-index --find-links=. typing_extensions-4.16.0-py3-none-any.whl

# 4. torch - 主包（使用 --no-deps 跳过依赖解析）
pip3 install --no-index --find-links=. --no-deps torch-2.6.0+cu126-cp310-cp310-linux_aarch64.whl

# 5. torchvision - 视觉库（使用 --no-deps）
pip3 install --no-index --find-links=. --no-deps torchvision-0.21.0-cp310-cp310-linux_aarch64.whl
```

## 验证结果示例

```
==========================================
Verifying torch installation on Orin
==========================================

Python version:
Python 3.10.12

PyTorch version:
2.6.0+cu126

CUDA availability:
CUDA available: True

CUDA device:
Device: Orin
CUDA version: 12.6
GPU memory: 61.34 GB

Torchvision version:
0.21.0

NumPy version:
2.2.6

==========================================
Verification completed!
==========================================
```

## 常见问题

### Q1: pip 安装失败，提示权限问题

**解决方案**：使用 `--user` 参数或在虚拟环境中安装。

```bash
pip3 install --user --no-index --find-links=. torch-2.6.0+cu126-cp310-cp310-linux_aarch64.whl
```

### Q2: CUDA 不可用

**解决方案**：检查 NVIDIA 驱动是否正常工作。

```bash
nvidia-smi
```

如果驱动未加载，尝试：

```bash
sudo modprobe nvidia
nvidia-smi
```

### Q3: 安装后其他 Python 包出现依赖冲突

**解决方案**：创建虚拟环境。

```bash
python3 -m venv torch_env
source torch_env/bin/activate
pip3 install --no-index --find-links=. torch-2.6.0+cu126-cp310-cp310-linux_aarch64.whl
```

### Q4: 安装后 Python 找不到 torch

**解决方案**：检查 Python 路径。

```bash
python3 -c "import sys; print(sys.path)"
ls /usr/local/lib/python3.10/dist-packages/torch/
```

### Q5: 安装速度慢

**解决方案**：安装脚本使用了 `--no-index` 和 `--find-links=.` 参数，不会联网下载，速度应该很快。如果慢，可能是磁盘 I/O 问题。

## 文件大小

| 文件 | 大小 |
|------|------|
| torch-2.6.0+cu126 wheel | ~2.3 GB |
| torchvision-0.21.0 wheel | ~19 MB |
| numpy-2.2.6 wheel | ~14 MB |
| sympy-1.13.1 wheel | ~5.9 MB |
| typing-extensions wheel | ~45 KB |
| **tar.gz 压缩包** | **~2.36 GB** |

## 注意事项

1. **Python 版本**：必须使用 Python 3.10.x，其他版本不兼容
2. **CUDA 版本**：torch-2.6.0+cu126 需要 CUDA 12.6，与 Orin 的驱动兼容
3. **安装顺序**：必须按脚本中的顺序安装，否则会有依赖问题
4. **磁盘空间**：解压后需要约 5 GB 空间
5. **权限**：建议使用 root 用户安装，避免权限问题