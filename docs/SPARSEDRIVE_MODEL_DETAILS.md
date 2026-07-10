# SparseDriveV2 模型细节说明

本文档详细介绍 SparseDriveV2 模型的两个核心问题：

1. [模型参数量分布](#一-模型参数量分布)
2. [模型保存机制](#二-模型保存机制)

---

## 一、模型参数量分布

### 1.1 整体统计

| 指标 | 数值 | 占比 |
|------|------|------|
| **总参数量** | **50.88M** (50,881,469) | - |
| **可训练参数** | **42.34M** (42,337,213) | 83.21% |
| **固定参数** | **8.54M** (8,544,256) | 16.79% |

### 1.2 各模块参数量汇总

| 模块 | 总参数 | 可训练 | 固定 | 占比 |
|------|--------|--------|------|------|
| **词汇表(anchor)** | 8.54M | 0 | 8.54M | 16.79% |
| **ResNet-34** | ~21.7M | 21.7M | 0 | 42.65% |
| **FPN Neck** | ~2.2M | 2.2M | 0 | 4.32% |
| **状态编码器** | 2.3K | 2.3K | 0 | <0.01% |
| **位置编码器** | ~1.05M | 1.05M | 0 | 2.06% |
| **Transformer Decoder** | ~16.5M | 16.5M | 0 | 32.43% |
| **评分头/指标头** | ~2.5M | 2.5M | 0 | 4.91% |

### 1.3 固定参数（词汇表）- 8.54M

这些参数通过 K-Means 聚类预计算得到，训练时不更新（`requires_grad=False`）。

| 参数 | 形状 | 参数量 | 文件位置 | 说明 |
|------|------|--------|----------|------|
| `_trajectory_head.path_vocab` | [1024, 50, 3] | **153.6K** | `sparsedrive_model.py:153-156` | 路径词汇表（1024条路径，每条50个点） |
| `_trajectory_head.vel_vocab` | [256, 8] | **2K** | `sparsedrive_model.py:160-163` | 速度词汇表（256个速度序列） |
| `_trajectory_head.traj_vocab` | [1024, 256, 8, 3] | **6.29M** | `sparsedrive_model.py:169-172` | 完整轨迹词汇表（路径1024 × 速度256 × 8个姿态点 × 3维坐标） |
| `_trajectory_head.traj_mask` | [1024, 256, 8] | **2.097M** | `sparsedrive_model.py:178-181` | 轨迹有效性掩码 |

**设计原理**：通过可分解词汇表（路径1024 + 速度256 = 262K组合）实现高效轨迹搜索。

### 1.4 视觉骨干网络（ResNet-34 + FPN）- 23.89M

#### 1.4.1 ResNet-34 骨干

**模块创建位置**: `navsim/agents/sparsedrive/sparsedrive_backbone.py:68-74`

```python
self.img_backbone = timm.create_model(
    config.image_architecture,  # "resnet34"
    pretrained=True,
    features_only=True,
    pretrained_cfg_overlay=dict(file=config.bkb_path),
    out_indices=(1, 2, 3, 4)[-config.num_levels:]
)
```

**参数实际定义位置**: `miniconda3/envs/navsim/lib/python3.9/site-packages/timm/models/resnet.py`

| 参数 | 形状 | 参数量 | 说明 |
|------|------|--------|------|
| `conv1.weight` | [64, 3, 7, 7] | 94K | 输入卷积层 |
| `bn1.weight/bias` | [64] | 128 | 输入批归一化 |
| `layer1.0.conv1/conv2.weight` | [64, 64, 3, 3] | 36.9K × 2 | 第1层第1个BasicBlock |
| `layer1.1.conv1/conv2.weight` | [64, 64, 3, 3] | 36.9K × 2 | 第1层第2个BasicBlock |
| `layer1.2.conv1/conv2.weight` | [64, 64, 3, 3] | 36.9K × 2 | 第1层第3个BasicBlock |
| `layer2.0.conv1.weight` | [128, 64, 3, 3] | 73.7K | 第2层第1个BasicBlock（下采样） |
| `layer2.0.conv2.weight` | [128, 128, 3, 3] | 147.5K | 第2层第1个BasicBlock |
| `layer2.0.downsample.0.weight` | [128, 64, 1, 1] | 73.7K | 第2层下采样卷积 |
| `layer2.1-3.conv1/conv2.weight` | [128, 128, 3, 3] | 147.5K × 6 | 第2层第2-4个BasicBlock |
| `layer3.0.conv1.weight` | [256, 128, 3, 3] | 294.9K | 第3层第1个BasicBlock（下采样） |
| `layer3.0.conv2.weight` | [256, 256, 3, 3] | 589.8K | 第3层第1个BasicBlock |
| `layer3.0.downsample.0.weight` | [256, 128, 1, 1] | 294.9K | 第3层下采样卷积 |
| `layer3.1-5.conv1/conv2.weight` | [256, 256, 3, 3] | 589.8K × 10 | 第3层第2-6个BasicBlock |
| `layer4.0.conv1.weight` | [512, 256, 3, 3] | 1.18M | 第4层第1个BasicBlock（下采样） |
| `layer4.0.conv2.weight` | [512, 512, 3, 3] | 2.36M | 第4层第1个BasicBlock |
| `layer4.0.downsample.0.weight` | [512, 256, 1, 1] | 131K | 第4层下采样卷积 |
| `layer4.1-2.conv1/conv2.weight` | [512, 512, 3, 3] | 2.36M × 4 | 第4层第2-3个BasicBlock |
| **ResNet-34 总计** | - | **~21.7M** | 使用预训练权重初始化 |

#### 1.4.2 FPN Neck

**模块创建位置**: `navsim/agents/sparsedrive/sparsedrive_backbone.py:77-81`

```python
if self.with_img_neck:
    self.img_neck = FPN(
        in_channels_list=[64, 128, 256, 512][-config.num_levels:],
        out_channels=self.embed_dims,  # 256
    )
```

**参数实际定义位置**: `miniconda3/envs/navsim/lib/python3.9/site-packages/torchvision/ops/fpn.py`

| 参数 | 形状 | 参数量 | 说明 |
|------|------|--------|------|
| `inner_blocks[0].weight` | [256, 64, 1, 1] | 16.4K | P2层1×1卷积 |
| `inner_blocks[1].weight` | [256, 128, 1, 1] | 32.8K | P3层1×1卷积 |
| `inner_blocks[2].weight` | [256, 256, 1, 1] | 65.5K | P4层1×1卷积 |
| `inner_blocks[3].weight` | [256, 512, 1, 1] | 131K | P5层1×1卷积 |
| `layer_blocks[0].weight` | [256, 256, 3, 3] | 589.8K | P2层3×3卷积 |
| `layer_blocks[1].weight` | [256, 256, 3, 3] | 589.8K | P3层3×3卷积 |
| `layer_blocks[2].weight` | [256, 256, 3, 3] | 589.8K | P4层3×3卷积 |
| `layer_blocks[3].weight` | [256, 256, 3, 3] | 589.8K | P5层3×3卷积 |
| **FPN 总计** | - | **~2.2M** | 多尺度特征融合 |

### 1.5 状态编码器 - 2.3K

**代码位置**: `navsim/agents/sparsedrive/sparsedrive_model.py:62`

```python
self._status_encoding = nn.Linear(4 + 2 + 2, config.d_model)  # Linear(8→256)
```

| 参数 | 形状 | 参数量 | 实际定义位置 | 说明 |
|------|------|--------|-------------|------|
| `_status_encoding.weight` | [256, 8] | 2.048K | `torch/nn/modules/linear.py` | Linear层权重 |
| `_status_encoding.bias` | [256] | 256 | `torch/nn/modules/linear.py` | Linear层偏置 |

### 1.6 位置编码器 - ~1.05M

#### 1.6.1 路径位置编码器

**代码位置**: `navsim/agents/sparsedrive/sparsedrive_model.py:187-191`

```python
self.path_pos_embed = nn.Sequential(
    nn.Linear(config.len_path * 3, d_ffn),  # 50*3=150 -> 1024
    nn.ReLU(),
    nn.Linear(d_ffn, d_model),              # 1024 -> 256
)
```

| 参数 | 形状 | 参数量 | 实际定义位置 | 说明 |
|------|------|--------|-------------|------|
| `path_pos_embed.0.weight` | [1024, 150] | 153.6K | `torch/nn/modules/linear.py` | 第一层Linear权重 |
| `path_pos_embed.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 第一层Linear偏置 |
| `path_pos_embed.2.weight` | [256, 1024] | 262.1K | `torch/nn/modules/linear.py` | 第二层Linear权重 |
| `path_pos_embed.2.bias` | [256] | 256 | `torch/nn/modules/linear.py` | 第二层Linear偏置 |
| **路径编码器总计** | - | **~417K** (417,024) | - | |

#### 1.6.2 速度位置编码器

**代码位置**: `navsim/agents/sparsedrive/sparsedrive_model.py:195-199`

```python
self.vel_pos_embed = nn.Sequential(
    nn.Linear(config.len_vel_seq, d_ffn),   # 8 -> 1024
    nn.ReLU(),
    nn.Linear(d_ffn, d_model),              # 1024 -> 256
)
```

| 参数 | 形状 | 参数量 | 实际定义位置 | 说明 |
|------|------|--------|-------------|------|
| `vel_pos_embed.0.weight` | [1024, 8] | 8.2K | `torch/nn/modules/linear.py` | 第一层Linear权重 |
| `vel_pos_embed.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 第一层Linear偏置 |
| `vel_pos_embed.2.weight` | [256, 1024] | 262.1K | `torch/nn/modules/linear.py` | 第二层Linear权重 |
| `vel_pos_embed.2.bias` | [256] | 256 | `torch/nn/modules/linear.py` | 第二层Linear偏置 |
| **速度编码器总计** | - | **~272K** (271,616) | - | |

### 1.7 Transformer Decoder - ~16.5M

Transformer Decoder 定义在 `navsim/agents/sparsedrive/custom_decoder.py` 中，包含2层解码器，每层有路径、速度分支，最后一层额外有轨迹分支。

#### 1.7.1 可变形特征聚合（DeformableFeatureAggregation）

**代码位置**: `navsim/agents/sparsedrive/custom_decoder.py:137-148`（路径分支）和 `custom_decoder.py:229-240`（轨迹分支）

```python
self.p_deform_model = DeformableFeatureAggregation(
    config=config, embed_dims=d_model, num_groups=8,
    num_levels=self._config.num_levels,  # 4层特征
    num_cams=len(config.cams),           # 3个相机
    num_pts=self._config.len_path,       # 50个路径点
    attn_drop=0.0, use_deformable_func=True,
    use_camera_embed=True, residual_mode="add",
)

self.t_deform_model = DeformableFeatureAggregation(
    config=config, embed_dims=d_model, num_groups=8,
    num_levels=self._config.num_levels,
    num_cams=len(config.cams),
    num_pts=num_poses,                   # 8个轨迹点
    attn_drop=0.0, use_deformable_func=True,
    use_camera_embed=True, residual_mode="add",
)
```

**关键点数量计算**（`blocks.py:551`）：

```python
# sparsedrive_config.py:53-54
fix_height = (0., -0.25, -0.5, 0.25, 0.5)  # len=5
num_learnable_pts = 2

# blocks.py:551
self.num_pts = num_sample * len(fix_height) * num_learnable_pts

# 路径分支: num_sample=50（路径点数）
num_pts = 50 × 5 × 2 = 500
learnable_fc输出维度 = 500 × 2 = 1000  # 每个关键点有x,y两个偏移量

# 轨迹分支: num_sample=8（轨迹姿态数）
num_pts = 8 × 5 × 2 = 80
learnable_fc输出维度 = 80 × 2 = 160
```

| 参数 | 形状 | 参数量 | 实际定义位置 | 说明 |
|------|------|--------|-------------|------|
| `p_deform_model.weights_fc.weight` | [16000, 256] | 4.096M | `blocks.py:163-165` | 路径分支可变形注意力权重（8组×4层×500关键点） |
| `p_deform_model.weights_fc.bias` | [16000] | 16K | `blocks.py:163-165` | 路径分支可变形注意力偏置 |
| `p_deform_model.kps_generator.learnable_fc.weight` | [1000, 256] | 256K | `blocks.py:553` | 路径分支关键点生成器（50×5×2×2=1000） |
| `p_deform_model.kps_generator.learnable_fc.bias` | [1000] | 1K | `blocks.py:553` | 路径分支关键点生成器偏置 |
| `p_deform_model.camera_encoder.0.weight` | [256, 12] | 3.1K | `blocks.py:159-161` | 相机编码器第一层 |
| `p_deform_model.camera_encoder.2.weight` | [256, 256] | 65.5K | `blocks.py:159-161` | 相机编码器第二层 |
| `p_deform_model.output_proj.weight` | [256, 256] | 65.5K | `blocks.py:154` | 路径分支输出投影 |
| `p_deform_model.output_proj.bias` | [256] | 256 | `blocks.py:154` | 路径分支输出投影偏置 |
| `t_deform_model.weights_fc.weight` | [2560, 256] | 655.4K | `blocks.py:163-165` | 轨迹分支可变形注意力权重（8组×4层×80关键点） |
| `t_deform_model.weights_fc.bias` | [2560] | 2.6K | `blocks.py:163-165` | 轨迹分支可变形注意力偏置 |
| `t_deform_model.kps_generator.learnable_fc.weight` | [160, 256] | 40.96K | `blocks.py:553` | 轨迹分支关键点生成器（8×5×2×2=160） |
| `t_deform_model.kps_generator.learnable_fc.bias` | [160] | 160 | `blocks.py:553` | 轨迹分支关键点生成器偏置 |
| **可变形注意力总计** | - | **~18.3M** | - | 核心创新点 |

#### 1.7.2 多头注意力（MultiheadAttention）

**代码位置**: `navsim/agents/sparsedrive/custom_decoder.py:152-157`（路径）、`custom_decoder.py:185-190`（图像）、`custom_decoder.py:194-199`（速度）、`custom_decoder.py:244-249`（轨迹）

```python
self.p_attention = nn.MultiheadAttention(config.d_model, config.num_head, dropout=config.dropout, batch_first=True)
self.v_img_attention = nn.MultiheadAttention(config.d_model, config.num_head, dropout=config.dropout, batch_first=True)
self.v_attention = nn.MultiheadAttention(config.d_model, config.num_head, dropout=config.dropout, batch_first=True)
self.t_attention = nn.MultiheadAttention(config.d_model, config.num_head, dropout=config.dropout, batch_first=True)
```

| 参数 | 形状 | 参数量 | 实际定义位置 | 说明 |
|------|------|--------|-------------|------|
| `p_attention.in_proj_weight` | [768, 256] | 196.6K | `torch/nn/modules/attention.py` | 输入投影（Q+K+V，8头） |
| `p_attention.in_proj_bias` | [768] | 768 | `torch/nn/modules/attention.py` | 输入投影偏置 |
| `p_attention.out_proj.weight` | [256, 256] | 65.5K | `torch/nn/modules/attention.py` | 输出投影 |
| `p_attention.out_proj.bias` | [256] | 256 | `torch/nn/modules/attention.py` | 输出投影偏置 |
| `v_img_attention.in_proj_weight` | [768, 256] | 196.6K | `torch/nn/modules/attention.py` | 速度-图像交叉注意力 |
| `v_img_attention.in_proj_bias` | [768] | 768 | `torch/nn/modules/attention.py` | 偏置 |
| `v_img_attention.out_proj.weight` | [256, 256] | 65.5K | `torch/nn/modules/attention.py` | 输出投影 |
| `v_img_attention.out_proj.bias` | [256] | 256 | `torch/nn/modules/attention.py` | 偏置 |
| `v_attention.in_proj_weight` | [768, 256] | 196.6K | `torch/nn/modules/attention.py` | 速度自注意力 |
| `v_attention.in_proj_bias` | [768] | 768 | `torch/nn/modules/attention.py` | 偏置 |
| `v_attention.out_proj.weight` | [256, 256] | 65.5K | `torch/nn/modules/attention.py` | 输出投影 |
| `v_attention.out_proj.bias` | [256] | 256 | `torch/nn/modules/attention.py` | 偏置 |
| `t_attention.in_proj_weight` | [768, 256] | 196.6K | `torch/nn/modules/attention.py` | 轨迹自注意力（仅最后一层） |
| `t_attention.in_proj_bias` | [768] | 768 | `torch/nn/modules/attention.py` | 偏置 |
| `t_attention.out_proj.weight` | [256, 256] | 65.5K | `torch/nn/modules/attention.py` | 输出投影 |
| `t_attention.out_proj.bias` | [256] | 256 | `torch/nn/modules/attention.py` | 偏置 |
| **多头注意力总计** | - | **~1.84M** | - | |

#### 1.7.3 前馈网络（FFN）

**代码位置**: `navsim/agents/sparsedrive/custom_decoder.py:161-165`（路径）、`custom_decoder.py:203-207`（速度）、`custom_decoder.py:252-256`（轨迹）

```python
self.p_ffn = nn.Sequential(
    nn.Linear(config.d_model, config.d_ffn),  # 256 → 1024
    nn.ReLU(),
    nn.Linear(config.d_ffn, config.d_model),  # 1024 → 256
)
self.v_ffn = nn.Sequential(...)
self.t_ffn = nn.Sequential(...)
```

| 参数 | 形状 | 参数量 | 实际定义位置 | 说明 |
|------|------|--------|-------------|------|
| `p_ffn.0.weight` | [1024, 256] | 262.1K | `torch/nn/modules/linear.py` | 路径FFN第一层 |
| `p_ffn.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 路径FFN第一层偏置 |
| `p_ffn.2.weight` | [256, 1024] | 262.1K | `torch/nn/modules/linear.py` | 路径FFN第二层 |
| `p_ffn.2.bias` | [256] | 256 | `torch/nn/modules/linear.py` | 路径FFN第二层偏置 |
| `v_ffn.0.weight` | [1024, 256] | 262.1K | `torch/nn/modules/linear.py` | 速度FFN第一层 |
| `v_ffn.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 速度FFN第一层偏置 |
| `v_ffn.2.weight` | [256, 1024] | 262.1K | `torch/nn/modules/linear.py` | 速度FFN第二层 |
| `v_ffn.2.bias` | [256] | 256 | `torch/nn/modules/linear.py` | 速度FFN第二层偏置 |
| `t_ffn.0.weight` | [1024, 256] | 262.1K | `torch/nn/modules/linear.py` | 轨迹FFN第一层（仅最后一层） |
| `t_ffn.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 轨迹FFN第一层偏置 |
| `t_ffn.2.weight` | [256, 1024] | 262.1K | `torch/nn/modules/linear.py` | 轨迹FFN第二层（仅最后一层） |
| `t_ffn.2.bias` | [256] | 256 | `torch/nn/modules/linear.py` | 轨迹FFN第二层偏置 |
| **FFN 总计** | - | **~3.14M** | - | |

### 1.8 评分头和指标头 - ~2.5M

#### 1.8.1 路径/速度/轨迹评分头

**代码位置**: `navsim/agents/sparsedrive/custom_decoder.py:175-179`（路径）、`custom_decoder.py:213-221`（速度）、`custom_decoder.py:266-270`（轨迹）

```python
self.path_mlp = nn.Sequential(
    nn.Linear(d_model, d_ffn),  # 256 → 1024
    nn.ReLU(),
    nn.Linear(d_ffn, 1),        # 1024 → 1
)
self.vel_mlp = nn.Sequential(...)
self.traj_mlp = nn.Sequential(...)
```

| 参数 | 形状 | 参数量 | 实际定义位置 | 说明 |
|------|------|--------|-------------|------|
| `path_mlp.0.weight` | [1024, 256] | 262.1K | `torch/nn/modules/linear.py` | 路径评分头第一层 |
| `path_mlp.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 路径评分头第一层偏置 |
| `path_mlp.2.weight` | [1, 1024] | 1.0K | `torch/nn/modules/linear.py` | 路径评分头第二层 |
| `path_mlp.2.bias` | [1] | 1 | `torch/nn/modules/linear.py` | 路径评分头第二层偏置 |
| `vel_mlp.0.weight` | [1024, 256] | 262.1K | `torch/nn/modules/linear.py` | 速度评分头第一层 |
| `vel_mlp.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 速度评分头第一层偏置 |
| `vel_mlp.2.weight` | [1, 1024] | 1.0K | `torch/nn/modules/linear.py` | 速度评分头第二层 |
| `vel_mlp.2.bias` | [1] | 1 | `torch/nn/modules/linear.py` | 速度评分头第二层偏置 |
| `traj_mlp.0.weight` | [1024, 256] | 262.1K | `torch/nn/modules/linear.py` | 轨迹评分头第一层 |
| `traj_mlp.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 轨迹评分头第一层偏置 |
| `traj_mlp.2.weight` | [1, 1024] | 1.0K | `torch/nn/modules/linear.py` | 轨迹评分头第二层 |
| `traj_mlp.2.bias` | [1] | 1 | `torch/nn/modules/linear.py` | 轨迹评分头第二层偏置 |
| **评分头总计** | - | **~1.31M** | - | |

#### 1.8.2 PDM指标预测头（8个指标）

**代码位置**: `navsim/agents/sparsedrive/custom_decoder.py:278-284`

```python
self.metric_heads = nn.ModuleDict()
for metric in self._config.metrics:
    self.metric_heads[metric] = nn.Sequential(
        nn.Linear(d_model, d_ffn),  # 256 → 1024
        nn.ReLU(),
        nn.Linear(d_ffn, 1),        # 1024 → 1
    )
# metrics = ["no_at_fault_collisions", "drivable_area_compliance", 
#            "driving_direction_compliance", "traffic_light_compliance",
#            "time_to_collision_within_bound", "ego_progress", 
#            "lane_keeping", "history_comfort"]
```

| 参数 | 形状 | 参数量 | 实际定义位置 | 说明 |
|------|------|--------|-------------|------|
| `metric_heads.no_at_fault_collisions.0.weight` | [1024, 256] | 262.1K | `torch/nn/modules/linear.py` | 无责任碰撞指标 |
| `metric_heads.no_at_fault_collisions.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.no_at_fault_collisions.2.weight` | [1, 1024] | 1.0K | `torch/nn/modules/linear.py` | 输出层 |
| `metric_heads.no_at_fault_collisions.2.bias` | [1] | 1 | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.drivable_area_compliance.0.weight` | [1024, 256] | 262.1K | `torch/nn/modules/linear.py` | 可行驶区域合规指标 |
| `metric_heads.drivable_area_compliance.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.drivable_area_compliance.2.weight` | [1, 1024] | 1.0K | `torch/nn/modules/linear.py` | 输出层 |
| `metric_heads.drivable_area_compliance.2.bias` | [1] | 1 | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.driving_direction_compliance.0.weight` | [1024, 256] | 262.1K | `torch/nn/modules/linear.py` | 行驶方向合规指标 |
| `metric_heads.driving_direction_compliance.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.driving_direction_compliance.2.weight` | [1, 1024] | 1.0K | `torch/nn/modules/linear.py` | 输出层 |
| `metric_heads.driving_direction_compliance.2.bias` | [1] | 1 | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.traffic_light_compliance.0.weight` | [1024, 256] | 262.1K | `torch/nn/modules/linear.py` | 交通灯合规指标 |
| `metric_heads.traffic_light_compliance.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.traffic_light_compliance.2.weight` | [1, 1024] | 1.0K | `torch/nn/modules/linear.py` | 输出层 |
| `metric_heads.traffic_light_compliance.2.bias` | [1] | 1 | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.time_to_collision_within_bound.0.weight` | [1024, 256] | 262.1K | `torch/nn/modules/linear.py` | 碰撞时间边界指标 |
| `metric_heads.time_to_collision_within_bound.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.time_to_collision_within_bound.2.weight` | [1, 1024] | 1.0K | `torch/nn/modules/linear.py` | 输出层 |
| `metric_heads.time_to_collision_within_bound.2.bias` | [1] | 1 | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.ego_progress.0.weight` | [1024, 256] | 262.1K | `torch/nn/modules/linear.py` | 自车进度指标 |
| `metric_heads.ego_progress.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.ego_progress.2.weight` | [1, 1024] | 1.0K | `torch/nn/modules/linear.py` | 输出层 |
| `metric_heads.ego_progress.2.bias` | [1] | 1 | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.lane_keeping.0.weight` | [1024, 256] | 262.1K | `torch/nn/modules/linear.py` | 车道保持指标 |
| `metric_heads.lane_keeping.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.lane_keeping.2.weight` | [1, 1024] | 1.0K | `torch/nn/modules/linear.py` | 输出层 |
| `metric_heads.lane_keeping.2.bias` | [1] | 1 | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.history_comfort.0.weight` | [1024, 256] | 262.1K | `torch/nn/modules/linear.py` | 历史舒适性指标 |
| `metric_heads.history_comfort.0.bias` | [1024] | 1.0K | `torch/nn/modules/linear.py` | 偏置 |
| `metric_heads.history_comfort.2.weight` | [1, 1024] | 1.0K | `torch/nn/modules/linear.py` | 输出层 |
| `metric_heads.history_comfort.2.bias` | [1] | 1 | `torch/nn/modules/linear.py` | 偏置 |
| **指标头总计** | - | **~2.1M** | - | 8个安全/合规/效率/舒适性指标 |

---

## 二、模型保存机制

### 2.1 触发时机

模型保存由 `CheckpointCallback` 在每个 epoch 结束时触发：

**文件**: `navsim/agents/sparsedrive/sparsedrive_callback.py:15-27`

```python
class CheckpointCallback(Callback):
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        trainer.strategy.barrier()  # 多GPU同步
        epoch = trainer.current_epoch
        ckpt_dir = Path(trainer.default_root_dir) / "periodic_pdm_ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"ep{epoch+1:04d}.ckpt"
        trainer.save_checkpoint(str(ckpt_path))  # ← 核心调用
        trainer.strategy.barrier()
        print(f"[PDM] saved ckpt: {ckpt_path}")
```

**保存路径**: `exp/sparsedrive_agent/<训练时间戳>/periodic_pdm_ckpts/ep{epoch}.ckpt`

### 2.2 完整代码路径

从 `trainer.save_checkpoint()` 到最终写入文件，经过以下步骤：

```
sparsev2/navsim/agents/sparsedrive/sparsedrive_callback.py:23
    trainer.save_checkpoint(str(ckpt_path))
           ↓
pytorch_lightning/trainer/trainer.py:1357-1382
    checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
           ↓
pytorch_lightning/trainer/connectors/checkpoint_connector.py:404-494
    生成checkpoint字典
           ↓
pytorch_lightning/strategies/strategy.py:479-491
    self.checkpoint_io.save_checkpoint(checkpoint, filepath)
           ↓
lightning_fabric/plugins/io/torch_io.py:37-58
    _atomic_save(checkpoint, path)
           ↓
lightning_fabric/utilities/cloud_io.py:65-80
    torch.save(checkpoint, bytesbuffer)
           ↓
写入文件: exp/sparsedrive_agent/xxx/periodic_pdm_ckpts/ep0010.ckpt
```

### 2.3 步骤详解

#### 步骤1：Trainer.save_checkpoint()

**文件**: `pytorch_lightning/trainer/trainer.py:1357-1382`

```python
def save_checkpoint(
    self, filepath: _PATH, weights_only: bool = False, storage_options: Optional[Any] = None
) -> None:
    if self.model is None:
        raise AttributeError("Saving a checkpoint is only possible if a model is attached...")
    
    checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
    self.strategy.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
    self.strategy.barrier("Trainer.save_checkpoint")
```

#### 步骤2：dump_checkpoint() - 生成checkpoint字典

**文件**: `pytorch_lightning/trainer/connectors/checkpoint_connector.py:404-494`

```python
def dump_checkpoint(self, weights_only: bool = False) -> dict:
    checkpoint = {
        "epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        "pytorch-lightning_version": pl.__version__,
        "state_dict": self._get_lightning_module_state_dict(),  # 模型权重
        "loops": self._get_loops_state_dict(),
    }
    if not weights_only:
        checkpoint["callbacks"] = call._call_callbacks_state_dict(trainer)
        checkpoint["optimizer_states"] = [trainer.strategy.optimizer_state(opt) for opt in trainer.optimizers]
        checkpoint["lr_schedulers"] = [config.scheduler.state_dict() for config in trainer.lr_scheduler_configs]
    return checkpoint
```

#### 步骤3：strategy.save_checkpoint()

**文件**: `pytorch_lightning/strategies/strategy.py:479-491`

```python
def save_checkpoint(self, checkpoint, filepath, storage_options=None):
    if self.is_global_zero:
        self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
```

#### 步骤4：_atomic_save() - 最终写入文件

**文件**: `lightning_fabric/utilities/cloud_io.py:65-80`

```python
def _atomic_save(checkpoint, filepath):
    bytesbuffer = io.BytesIO()
    torch.save(checkpoint, bytesbuffer)  # ← PyTorch原生保存
    with fsspec.open(filepath, "wb") as f:
        f.write(bytesbuffer.getvalue())
```

### 2.4 checkpoint 文件内容详解

| 字段 | 类型 | 内容说明 |
|------|------|----------|
| `epoch` | int | 当前训练到第几个 epoch |
| `global_step` | int | 全局训练步数（batch数） |
| `state_dict` | Dict[str, Tensor] | **模型权重**，key 为参数名，value 为张量 |
| `optimizer_states` | List[Dict] | **优化器状态**，包含动量 `exp_avg`、`exp_avg_sq` 等 |
| `lr_schedulers` | List[Dict] | 学习率调度器状态 |
| `callbacks` | Dict | 各回调的状态 |
| `hparams` | Dict | 超参数配置 |

### 2.5 state_dict 实际内容示例

```python
{
    "agent._sparsedrive_model._backbone.img_backbone.conv1.weight": tensor([[[[...]]]]),
    "agent._sparsedrive_model._status_encoding.weight": tensor([...]),
    "agent._sparsedrive_model._trajectory_head.traj_vocab": tensor([...]),
    # ... 所有参数
}
```

加载时需要移除 `agent.` 前缀：

```python
state_dict = torch.load(checkpoint_path)["state_dict"]
self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})
```

---

## 三、参数量统计脚本

```python
import torch
from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel

config = SparseDriveConfig()
model = SparseDriveModel(config)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"总参数量: {total/1e6:.2f}M")
print(f"可训练参数: {trainable/1e6:.2f}M")
print(f"固定参数: {(total-trainable)/1e6:.2f}M")
```

**运行方式**:
```bash
source scripts/cache/path_export.sh
python count_parameters_detailed.py
```