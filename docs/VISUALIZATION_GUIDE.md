# SparseDriveV2 可视化指南

## 概述

本指南介绍如何使用项目中的可视化工具，包括：

1. **轨迹对比可视化** - 对比模型预测轨迹与人类驾驶轨迹
2. **场景可视化** - 可视化摄像头、BEV鸟瞰图、LiDAR点云
3. **训练结果可视化** - 可视化损失曲线、指标变化

***

## 可视化脚本位置

```
scripts/visualization/
├── visualize_trajectory.py              # 单帧轨迹对比可视化
├── visualize_trajectory_sequence.py     # 时间序列轨迹对比可视化
├── visualize_scene.py                   # 场景可视化
└── visualize_training.py                # 训练结果可视化
```

***

## 1. 轨迹对比可视化

### 用途

对比训练模型的预测轨迹与人类驾驶轨迹（ground truth），评估模型的轨迹预测精度。

### 使用时机

- **训练完成后**：评估模型在验证集上的轨迹预测效果
- **推理阶段**：查看模型对特定场景的预测结果

### 使用方式

```bash
# 激活 conda 环境
conda activate navsim

# 运行轨迹对比可视化
python scripts/visualization/visualize_trajectory.py \
    --token <场景token> \
    --ckpt <模型checkpoint路径> \
    --output <输出目录>
```

### 参数说明

| 参数         | 必选 | 说明                                     |
| ---------- | -- | -------------------------------------- |
| `--token`  | 是  | 场景的唯一标识符（16位字符串）                       |
| `--ckpt`   | 是  | 训练好的模型 checkpoint 路径                   |
| `--output` | 否  | 输出目录，默认 `exp/visualization/trajectory` |

### 示例

```bash
# 激活 conda 环境
source /home/xqb/DATA2/E2E_Project/sparsev2/scripts/cache/path_export.sh && source /home/xqb/DATA2/E2E_Project/miniconda3/etc/profile.d/conda.sh && conda activate navsim && cd /home/xqb/DATA2/E2E_Project/sparsev2 

# 运行场景可视化
python scripts/visualization/visualize_trajectory.py \
    --token 6774548111cb5ba4 \
    --ckpt exp/sparsedrive_agent/2026.06.17.17.46.52/periodic_pdm_ckpts/ep0010.ckpt \
    --output exp/visualization/trajectory
```

### 输出结果

| 文件                       | 说明                |
| ------------------------ | ----------------- |
| `trajectory_{token}.png` | 轨迹对比静态图（BEV视角）    |
| `trajectory_{token}.gif` | 轨迹动画（展示预测轨迹随时间变化） |

### 可视化内容

- **蓝色线**：人类驾驶轨迹（ground truth）
- **红色线**：模型预测轨迹
- **绿色点**：当前帧位置
- **灰色区域**：可行驶区域

***

## 1.1 时间序列轨迹对比可视化

### 用途

在场景的每个时间步使用当前帧的输入来预测未来轨迹，展示模型在整个场景中的预测能力演变过程。与单帧可视化不同，此脚本生成一段连续的时间序列动画，便于观察模型预测的动态变化。

### 使用时机

- **训练完成后**：评估模型在整个场景时间序列上的预测一致性
- **推理阶段**：查看模型对特定场景的动态预测效果
- **调试阶段**：分析模型在不同时刻的预测偏差

### 使用方式

```bash
# 激活 conda 环境
conda activate navsim

# 运行时间序列轨迹对比可视化
python scripts/visualization/visualize_trajectory_sequence.py \
    --token <场景token> \
    --ckpt <模型checkpoint路径> \
    --output <输出目录> \
    [--start-frame <起始帧>] \
    [--end-frame <结束帧>]
```

### 参数说明

| 参数           | 必选 | 说明                                     |
| ------------ | -- | -------------------------------------- |
| `--token`    | 是  | 场景的唯一标识符（16位字符串）                       |
| `--ckpt`     | 是  | 训练好的模型 checkpoint 路径                   |
| `--output`   | 否  | 输出目录，默认 `exp/visualization/trajectory_sequence` |
| `--start-frame` | 否  | 起始帧索引，默认使用最后一帧历史帧（索引3）              |
| `--end-frame` | 否  | 结束帧索引，默认使用场景最后一帧                      |

### 示例

```bash
# 激活环境并运行
source /home/xqb/DATA2/E2E_Project/sparsev2/scripts/cache/path_export.sh && source /home/xqb/DATA2/E2E_Project/miniconda3/etc/profile.d/conda.sh && conda activate navsim && cd /home/xqb/DATA2/E2E_Project/sparsev2

# 运行时间序列轨迹对比可视化
python scripts/visualization/visualize_trajectory_sequence.py \
    --token 6774548111cb5ba4 \
    --ckpt exp/sparsedrive_agent/2026.06.17.17.46.52/periodic_pdm_ckpts/ep0010.ckpt \
    --output exp/visualization/trajectory_sequence
```

### 输出结果

| 文件                                       | 说明                                   |
| ---------------------------------------- | ------------------------------------ |
| `trajectory_sequence_summary_{token}.png` | 时间序列轨迹汇总图（所有帧预测轨迹叠加）               |
| `trajectory_sequence_{token}.gif`         | 时间序列轨迹动画（至少5秒，展示预测轨迹随时间演变）          |
| `trajectory_frame_{frame_idx}_{token}.png` | 逐帧预测轨迹图（每个时间步单独保存一张图）              |

### 可视化内容

**汇总图**：
- **黑色线**：历史轨迹（已行驶路径）
- **蓝色点**：所有帧的人类未来轨迹（ground truth）
- **红色点**：所有帧的模型预测轨迹

**动画和逐帧图**：
- **蓝色线**：当前帧的人类未来轨迹（ground truth）
- **红色线**：当前帧的模型预测轨迹
- **绿色点**：当前帧位置
- **灰色区域**：可行驶区域

### 时间序列说明

```
场景帧序列: [frame_0, frame_1, frame_2, frame_3, frame_4, ..., frame_13]
              ↓         ↓         ↓         ↓         ↓
           历史帧1   历史帧2   历史帧3   当前帧    未来帧1
                                         ↓
                                   模型在此帧预测未来轨迹
                                      ↓
                    预测: [frame_4, frame_5, ..., frame_13]
                    
时间序列可视化会在每个帧位置（frame_3 ~ frame_13）进行预测，
展示模型从不同位置出发的预测能力。
```

***

## 2. 场景可视化

### 用途

可视化单个场景的原始传感器数据，包括摄像头图像、BEV鸟瞰图、LiDAR点云投影。

### 使用时机

- **数据准备阶段**：检查原始数据质量
- **训练前**：查看场景的具体内容
- **调试阶段**：分析模型输入数据

### 使用方式

```bash
# 激活 conda 环境
conda activate navsim

# 运行场景可视化
python scripts/visualization/visualize_scene.py \
    --token <场景token> \
    --frame <帧索引> \
    --output <输出目录>
```

### 参数说明

| 参数         | 必选 | 说明                                |
| ---------- | -- | --------------------------------- |
| `--token`  | 是  | 场景的唯一标识符                          |
| `--frame`  | 否  | 要可视化的帧索引，默认 3（当前帧）                |
| `--output` | 否  | 输出目录，默认 `exp/visualization/scene` |

### 示例

```bash
# 可视化场景的第4帧（当前帧）
python scripts/visualization/visualize_scene.py \
    --token 6774548111cb5ba4 \
    --frame 3 \
    --output exp/visualization/scene
```

### 输出结果

| 文件                                | 说明                    |
| --------------------------------- | --------------------- |
| `bev_{token}.png`                 | BEV鸟瞰图                |
| `cameras_{token}.png`             | 8摄像头 + BEV 全景图（3x3网格） |
| `cameras_lidar_{token}.png`       | 摄像头 + LiDAR点云投影图      |
| `cameras_annotations_{token}.png` | 摄像头 + 标注框图            |
| `bev_animation_{token}.gif`       | BEV动画（场景所有帧）          |

### 场景帧说明

```
场景帧序列: [frame_0, frame_1, frame_2, frame_3, ..., frame_13]
              ↓         ↓         ↓         ↓
           历史帧1   历史帧2   历史帧3   当前帧(索引3)
                                         ↑
                                     --frame 3（默认）
```

***

## 3. 训练结果可视化

### 用途

可视化训练过程中的损失曲线、指标变化、学习率变化，分析训练效果。

### 使用时机

- **训练完成后**：分析训练过程
- **训练过程中**：实时监控训练进度（配合 TensorBoard）
- **调参阶段**：评估不同超参数的效果

### 使用方式

```bash
# 激活 conda 环境
conda activate navsim

# 运行训练结果可视化
python scripts/visualization/visualize_training.py \
    --log_dir <lightning_logs目录> \
    --output <输出目录>
```

### 参数说明

| 参数          | 必选 | 说明                                           |
| ----------- | -- | -------------------------------------------- |
| `--log_dir` | 是  | TensorBoard 日志目录（`lightning_logs/version_X`） |
| `--output`  | 否  | 输出目录，默认 `exp/visualization/training`         |

### 示例

```bash
# 可视化最新训练的结果
python scripts/visualization/visualize_training.py \
    --log_dir exp/sparsedrive_agent/2026.06.17.17.46.52/lightning_logs/version_0 \
    --output exp/visualization/training
```

### 输出结果

| 文件                  | 说明                     |
| ------------------- | ---------------------- |
| `loss_curves.png`   | 训练损失曲线（总损失、轨迹损失、速度损失等） |
| `metric_curves.png` | 训练指标曲线（如 ADE、FDE 等）    |
| `learning_rate.png` | 学习率变化曲线（对数坐标）          |

### 分析要点

1. **损失曲线**：应呈下降趋势，若震荡或上升说明训练不稳定
2. **验证损失**：若验证损失不再下降说明模型已收敛
3. **学习率**：应按预定策略衰减（如余弦退火）

***

## 获取场景 Token

### 方法1：查看缓存目录

```bash
# 查看 data_cache_navmini 中的 token
ls exp/data_cache_navmini/*/unknown/
```

### 方法2：查看 metric\_cache 目录

```bash
# 查看 metric_cache_navmini 中的 token
ls exp/metric_cache_navmini/*/unknown/
```

### 方法3：查看训练日志

训练过程中会打印场景 token：

```
[cache_scenarios_internal] Processing scenario 1 / 19 in thread_id=xxx
  - token: 6774548111cb5ba4
```

***

## 完整工作流

```
1. 数据准备
   ↓
2. 生成数据缓存 (run_dataset_caching.py)
   ↓
3. 生成 metric 缓存 (run_metric_caching.py)
   ↓
4. 训练模型 (run_training.py)
   ↓
5. 可视化训练结果 (visualize_training.py)
   ↓
6. 可视化场景数据 (visualize_scene.py)
   ↓
7. 可视化轨迹对比 (visualize_trajectory.py)
```

***

## 常见问题

### Q1：可视化脚本报错 "ModuleNotFoundError"

**解决**：确保在 conda 环境 `navsim` 中运行：

```bash
conda activate navsim
```

### Q2：轨迹对比可视化报错 "Token not found"

**解决**：确保 token 存在于数据缓存中，且 `train_test_split` 配置正确。

### Q3：场景可视化看不到图像

**解决**：确保已下载原始传感器数据（`sensor_blobs/mini`），且路径配置正确。

### Q4：训练结果可视化没有数据

**解决**：确保训练过程中启用了 TensorBoard 日志记录，且 `--log_dir` 指向正确的版本目录。

***

## 注意事项

1. **环境变量**：所有脚本已内置环境变量设置，无需手动配置
2. **输出目录**：可视化结果保存在 `exp/visualization/` 下，按类型分类
3. **内存占用**：场景可视化会加载原始图像，建议一次只可视化一个场景
4. **GPU加速**：轨迹对比可视化支持 GPU，确保 GPU 可用时速度更快

***

## 相关模块

可视化功能基于项目中的 `navsim/visualization/` 模块：

| 文件          | 功能        |
| ----------- | --------- |
| `plots.py`  | 核心可视化接口   |
| `bev.py`    | BEV鸟瞰图绘制  |
| `camera.py` | 摄像头图像绘制   |
| `lidar.py`  | LiDAR点云绘制 |
| `config.py` | 可视化配置参数   |

<br />

### 1. 轨迹对比可视化

- 脚本 : visualize\_trajectory\_simple.py
- 输出 :
  - exp/visualization/trajectory/trajectory\_1fc1dd0dc3d157ae.png (75KB)
  - exp/visualization/trajectory/trajectory\_1fc1dd0dc3d157ae.gif (176KB)
- 内容 : BEV鸟瞰图中对比人类轨迹（红色）和模型预测轨迹（蓝色）

### 2. 场景可视化

- 脚本 : visualize\_scene\_simple.py
- 输出 :
  - exp/visualization/scene/bev\_6774548111cb5ba4.png (70KB) - BEV鸟瞰图
  - exp/visualization/scene/cameras\_6774548111cb5ba4.png (2.5MB) - 8摄像头+BEV全景图
  - exp/visualization/scene/cameras\_lidar\_6774548111cb5ba4.png (2.9MB) - 摄像头+LiDAR投影
  - exp/visualization/scene/cameras\_annotations\_6774548111cb5ba4.png (2.5MB) - 摄像头+标注框
  - exp/visualization/scene/bev\_animation\_6774548111cb5ba4.gif (241KB) - BEV动画

### 3. 训练结果可视化

- 脚本 : scripts/visualization/visualize\_training.py
- 输出 :
  - exp/visualization/training/loss\_curves.png (256KB) - 损失曲线
  - exp/visualization/training/metric\_curves.png (33KB) - 指标曲线

