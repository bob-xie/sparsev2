# SparseDriveV2 环境启动指南

## 一、快速启动

每次打开新终端后，按以下步骤启动项目环境：

### 方法一：手动启动（推荐）

```bash
# 1. 进入项目目录
cd /home/ubuntu22/E2E_project/sparsev2

# 2. 初始化 conda 并激活环境
eval "$('/home/ubuntu22/miniconda3/bin/conda' 'shell.bash' 'hook')"
conda activate ros_yolov5

# 3. 加载项目路径配置
source scripts/cache/path_export_home.sh
```

### 方法二：使用一键启动脚本

创建启动脚本 `start_env.sh`：

```bash
#!/bin/bash
cd /home/ubuntu22/E2E_project/sparsev2
eval "$('/home/ubuntu22/miniconda3/bin/conda' 'shell.bash' 'hook')"
conda activate ros_yolov5
source scripts/cache/path_export_home.sh
echo "=== SparseDriveV2 环境启动完成 ==="
echo "当前环境: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version 2>&1)"
```

使用方法：

```bash
chmod +x start_env.sh
./start_env.sh
```
conda deactivate
exit
### 方法三：添加到 bashrc（自动启动）

在 `~/.bashrc` 末尾添加以下内容：

```bash
# SparseDriveV2 环境启动
if [ -z "$SPARSEDRIVE_ACTIVATED" ]; then
    export SPARSEDRIVE_ACTIVATED=1
    eval "$('/home/ubuntu22/miniconda3/bin/conda' 'shell.bash' 'hook')"
    conda activate ros_yolov5
    if [ -f "/home/ubuntu22/E2E_project/sparsev2/scripts/cache/path_export_home.sh" ]; then
        source /home/ubuntu22/E2E_project/sparsev2/scripts/cache/path_export_home.sh
    fi
fi
```

**注意**：此方法会在每次打开终端时自动激活环境，可能影响其他项目。

---

## 二、环境验证

启动后验证环境是否正确：

```bash
# 1. 检查 conda 环境
echo $CONDA_DEFAULT_ENV  # 应输出: ros_yolov5

# 2. 检查 Python 版本
python --version  # 应输出: Python 3.10.12

# 3. 检查 PyTorch 和 CUDA
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# 4. 检查环境变量
echo $NAVSIM_DEVKIT_ROOT  # 应输出: /home/ubuntu22/E2E_project/sparsev2
echo $CUDA_HOME           # 应输出: /usr/local/cuda-12.2
```

---

## 三、环境配置说明

### Conda 环境

| 项目 | 值 |
|------|-----|
| 环境名称 | ros_yolov5 |
| 安装路径 | /home/ubuntu22/miniconda3/envs/ros_yolov5 |
| Python 版本 | 3.10.12 |
| 主要依赖 | PyTorch 2.2.2+cu118, torchvision 0.17.2+cu118 |

### 项目路径配置

| 环境变量 | 路径 | 说明 |
|---------|------|------|
| NAVSIM_DEVKIT_ROOT | /home/ubuntu22/E2E_project/sparsev2 | 项目根目录 |
| OPENSCENE_DATA_ROOT | /home/ubuntu22/E2E_project/sparsev2 | 数据集根目录 |
| NAVSIM_EXP_ROOT | /home/ubuntu22/E2E_project/sparsev2/exp | 实验输出目录 |
| CUDA_HOME | /usr/local/cuda-12.2 | CUDA 工具包路径 |
| NUPLAN_MAPS_ROOT | /home/ubuntu22/E2E_project/sparsev2/maps/nuplan-maps-v1.0 | 地图数据路径 |

### 软链接

| 链接路径 | 目标路径 | 说明 |
|---------|---------|------|
| /home/ubuntu22/E2E_project/.conda_env | /home/ubuntu22/miniconda3/envs/ros_yolov5 | conda 环境软链接 |

---

## 四、常见问题

### Q1: 执行 `conda activate` 时提示 "conda: command not found"

**原因**：当前终端未初始化 conda。

**解决方案**：

```bash
# 手动初始化 conda
eval "$('/home/ubuntu22/miniconda3/bin/conda' 'shell.bash' 'hook')"
```

### Q2: 执行 `source scripts/cache/path_export_home.sh` 时报错

**原因**：路径配置文件中的路径不正确。

**解决方案**：

```bash
# 检查并编辑路径配置文件
vim scripts/cache/path_export_home.sh
```

确保以下路径指向正确位置：

```bash
export NAVSIM_DEVKIT_ROOT=/home/ubuntu22/E2E_project/sparsev2
export CUDA_HOME=/usr/local/cuda-12.2
```

### Q3: PyTorch 无法检测到 CUDA

**原因**：CUDA 环境变量未正确设置。

**解决方案**：

```bash
# 检查 CUDA 版本
nvcc --version

# 确保 CUDA_HOME 正确
echo $CUDA_HOME
# 应输出: /usr/local/cuda-12.2
```

### Q4: 如何退出 conda 环境

```bash
conda deactivate
```

---

## 五、启动脚本示例

完整的一键启动脚本 `scripts/start_env.sh`：

```bash
#!/bin/bash

echo "=== SparseDriveV2 环境启动 ==="

# 进入项目目录
cd /home/ubuntu22/E2E_project/sparsev2

# 初始化 conda
eval "$('/home/ubuntu22/miniconda3/bin/conda' 'shell.bash' 'hook')"

# 激活环境
conda activate ros_yolov5
echo "✓ Conda 环境已激活: $CONDA_DEFAULT_ENV"

# 加载路径配置
source scripts/cache/path_export_home.sh

# 验证环境
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== SparseDriveV2 环境启动完成 ==="
echo "当前目录: $(pwd)"
```

使用方法：

```bash
chmod +x scripts/start_env.sh
./scripts/start_env.sh
```
