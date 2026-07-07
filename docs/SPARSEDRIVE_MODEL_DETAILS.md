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
| `_trajectory_head.traj_vocab` | [1024, 256, 8, 3] | **6.29M** | `sparsedrive_model.py:169-172` | 完整轨迹词汇表（路径1024 × 速度256 × 8个姿态点 × 3维坐标） |
| `_trajectory_head.traj_mask` | [1024, 256, 8] | **2.097M** | `sparsedrive_model.py:178-181` | 轨迹有效性掩码 |
| `_trajectory_head.path_vocab` | [1024, 50, 3] | **153.6K** | `sparsedrive_model.py:153-156` | 路径词汇表（1024条路径，每条50个点） |
| `_trajectory_head.vel_vocab` | [256, 8] | **2K** | `sparsedrive_model.py:160-163` | 速度词汇表（256个速度序列） |

**设计原理**：通过可分解词汇表（路径1024 + 速度256 = 262K组合）实现高效轨迹搜索。

### 1.4 视觉骨干网络（ResNet-34 + FPN）- 23.89M

#### 1.4.1 ResNet-34 骨干

| 参数 | 形状 | 参数量 | 文件位置 | 说明 |
|------|------|--------|----------|------|
| `conv1.weight` | [64, 3, 7, 7] | 94K | `sparsedrive_backbone.py:68-74` | 输入卷积层 |
| `layer1.conv1/conv2.weight` | [64, 64, 3, 3] | 36.9K × 6 | `sparsedrive_backbone.py:68-74` | 第1层3个BasicBlock |
| `layer2.conv1/conv2.weight` | [128, 128, 3, 3] | 147.5K × 6 | `sparsedrive_backbone.py:68-74` | 第2层4个BasicBlock |
| `layer2.downsample.0.weight` | [128, 64, 1, 1] | 73.7K | `sparsedrive_backbone.py:68-74` | 第2层下采样 |
| `layer3.conv1/conv2.weight` | [256, 256, 3, 3] | 589.8K × 10 | `sparsedrive_backbone.py:68-74` | 第3层6个BasicBlock |
| `layer3.downsample.0.weight` | [256, 128, 1, 1] | 294.9K | `sparsedrive_backbone.py:68-74` | 第3层下采样 |
| `layer4.conv1/conv2.weight` | [512, 512, 3, 3] | 2.36M × 5 | `sparsedrive_backbone.py:68-74` | 第4层3个BasicBlock |
| `layer4.downsample.0.weight` | [512, 256, 1, 1] | 131K | `sparsedrive_backbone.py:68-74` | 第4层下采样 |
| **ResNet-34 总计** | - | **~21.7M** | - | 使用预训练权重初始化 |

#### 1.4.2 FPN Neck

| 参数 | 形状 | 参数量 | 文件位置 | 说明 |
|------|------|--------|----------|------|
| `img_neck.inner_blocks[0-3].0.weight` | [256, 64/128/256/512, 1, 1] | 65.5K × 2 + 131K + 131K | `sparsedrive_backbone.py:78-81` | 1×1卷积降维 |
| `img_neck.layer_blocks[0-3].0.weight` | [256, 256, 3, 3] | 589.8K × 4 | `sparsedrive_backbone.py:78-81` | 3×3卷积融合 |
| **FPN 总计** | - | **~2.2M** | - | 多尺度特征融合 |

### 1.5 状态编码器 - 2.3K

| 参数 | 形状 | 参数量 | 文件位置 | 说明 |
|------|------|--------|----------|------|
| `_status_encoding.weight` | [256, 8] | 2.048K | `sparsedrive_model.py:62` | Linear(8→256) |
| `_status_encoding.bias` | [256] | 256 | `sparsedrive_model.py:62` | 偏置 |

### 1.6 位置编码器 - ~1.05M

#### 1.6.1 路径位置编码器

| 参数 | 形状 | 参数量 | 文件位置 | 说明 |
|------|------|--------|----------|------|
| `path_pos_embed.0.weight` | [1024, 150] | 153.6K | `sparsedrive_model.py:187-191` | Linear(150→1024) |
| `path_pos_embed.2.weight` | [256, 1024] | 262.1K | `sparsedrive_model.py:187-191` | Linear(1024→256) |
| **路径编码器总计** | - | **~417.8K** | - | |

#### 1.6.2 速度位置编码器

| 参数 | 形状 | 参数量 | 文件位置 | 说明 |
|------|------|--------|----------|------|
| `vel_pos_embed.0.weight` | [1024, 8] | 8.2K | `sparsedrive_model.py:195-199` | Linear(8→1024) |
| `vel_pos_embed.2.weight` | [256, 1024] | 262.1K | `sparsedrive_model.py:195-199` | Linear(1024→256) |
| **速度编码器总计** | - | **~270.3K** | - | |

### 1.7 Transformer Decoder - ~16.5M

#### 1.7.1 可变形特征聚合（DeformableFeatureAggregation）

每个解码器层包含2个（路径+速度），最后一层额外1个（轨迹）：

| 参数 | 形状 | 参数量 | 文件位置 | 说明 |
|------|------|--------|----------|------|
| `weights_fc.weight` | [16000, 256] | 4.096M × 4 | `custom_decoder.py:137-148` | 可变形注意力权重（路径分支） |
| `weights_fc.weight` | [2560, 256] | 655.4K | `custom_decoder.py:229-240` | 可变形注意力权重（轨迹分支） |
| `kps_generator.learnable_fc.weight` | [1000, 256] | 256K × 4 | `custom_decoder.py:137-148` | 关键点生成器 |
| `kps_generator.learnable_fc.weight` | [160, 256] | 40.96K | `custom_decoder.py:229-240` | 关键点生成器（轨迹分支） |
| `camera_encoder` | - | ~131K × 5 | `custom_decoder.py:137-148` | 相机嵌入编码器 |
| `output_proj.weight` | [256, 256] | 65.5K × 5 | `custom_decoder.py:137-148` | 输出投影 |
| **可变形注意力总计** | - | **~18.3M** | - | 核心创新点 |

#### 1.7.2 多头注意力（MultiheadAttention）

每个解码器层包含4个（路径+速度+图像+轨迹）：

| 参数 | 形状 | 参数量 | 文件位置 | 说明 |
|------|------|--------|----------|------|
| `in_proj_weight` | [768, 256] | 196.6K × 7 | `custom_decoder.py:152-157` | 输入投影（8头×3矩阵） |
| `out_proj.weight` | [256, 256] | 65.5K × 7 | `custom_decoder.py:152-157` | 输出投影 |
| **多头注意力总计** | - | **~1.84M** | - | |

#### 1.7.3 前馈网络（FFN）

每个解码器层包含3个（路径+速度+轨迹）：

| 参数 | 形状 | 参数量 | 文件位置 | 说明 |
|------|------|--------|----------|------|
| `p_ffn.0.weight` | [1024, 256] | 262.1K × 6 | `custom_decoder.py:161-165` | FFN第一层 |
| `p_ffn.2.weight` | [256, 1024] | 262.1K × 6 | `custom_decoder.py:161-165` | FFN第二层 |
| **FFN 总计** | - | **~3.14M** | - | |

### 1.8 评分头和指标头 - ~2.5M

#### 1.8.1 路径/速度/轨迹评分头

| 参数 | 形状 | 参数量 | 文件位置 | 说明 |
|------|------|--------|----------|------|
| `path_mlp.0.weight` | [1024, 256] | 262.1K × 2 | `custom_decoder.py:175-179` | 路径评分头 |
| `vel_mlp.0.weight` | [1024, 256] | 262.1K × 2 | `custom_decoder.py:217-221` | 速度评分头 |
| `traj_mlp.0.weight` | [1024, 256] | 262.1K | `custom_decoder.py:266-270` | 轨迹评分头 |
| **评分头总计** | - | **~1.31M** | - | |

#### 1.8.2 PDM指标预测头（8个指标）

| 参数 | 形状 | 参数量 | 文件位置 | 说明 |
|------|------|--------|----------|------|
| `metric_heads[metric].0.weight` | [1024, 256] | 262.1K × 8 | `custom_decoder.py:278-284` | 每个指标一个MLP |
| **指标头总计** | - | **~2.1M** | - | 安全/合规/效率/舒适性指标 |

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
    
    # 生成checkpoint字典
    checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
    
    # 保存到文件
    self.strategy.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
    
    # 同步所有进程
    self.strategy.barrier("Trainer.save_checkpoint")
```

#### 步骤2：dump_checkpoint() - 生成checkpoint字典

**文件**: `pytorch_lightning/trainer/connectors/checkpoint_connector.py:404-494`

这是最核心的代码，负责生成保存的字典内容：

```python
def dump_checkpoint(self, weights_only: bool = False) -> dict:
    trainer = self.trainer
    model = trainer.lightning_module
    datamodule = trainer.datamodule

    checkpoint = {
        "epoch": trainer.current_epoch,           # 当前epoch
        "global_step": trainer.global_step,       # 全局步数
        "pytorch-lightning_version": pl.__version__,
        "state_dict": self._get_lightning_module_state_dict(),  # 模型权重
        "loops": self._get_loops_state_dict(),    # 训练循环状态
    }

    if not weights_only:
        # 回调状态
        checkpoint["callbacks"] = call._call_callbacks_state_dict(trainer)

        # 优化器状态
        optimizer_states = []
        for i, optimizer in enumerate(trainer.optimizers):
            optimizer_state = trainer.strategy.optimizer_state(optimizer)
            optimizer_states.append(optimizer_state)
        checkpoint["optimizer_states"] = optimizer_states

        # 学习率调度器
        lr_schedulers = []
        for config in trainer.lr_scheduler_configs:
            lr_schedulers.append(config.scheduler.state_dict())
        checkpoint["lr_schedulers"] = lr_schedulers

        # 精度插件（混合精度等）
        prec_plugin = trainer.precision_plugin
        prec_plugin_state_dict = prec_plugin.state_dict()
        if prec_plugin_state_dict:
            checkpoint[prec_plugin.__class__.__qualname__] = prec_plugin_state_dict

    # 超参数
    for obj in (model, datamodule):
        if obj and obj.hparams:
            checkpoint[obj.CHECKPOINT_HYPER_PARAMS_KEY] = obj.hparams

    # DataModule状态
    if datamodule is not None:
        datamodule_state_dict = call._call_lightning_datamodule_hook(trainer, "state_dict")
        if datamodule_state_dict:
            checkpoint[datamodule.__class__.__qualname__] = datamodule_state_dict

    # 自定义保存钩子
    call._call_callbacks_on_save_checkpoint(trainer, checkpoint)
    call._call_lightning_module_hook(trainer, "on_save_checkpoint", checkpoint)
    
    return checkpoint
```

#### 步骤3：strategy.save_checkpoint()

**文件**: `pytorch_lightning/strategies/strategy.py:479-491`

```python
def save_checkpoint(
    self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
) -> None:
    # 只在主进程(rank 0)保存
    if self.is_global_zero:
        self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
```

#### 步骤4：TorchCheckpointIO.save_checkpoint()

**文件**: `lightning_fabric/plugins/io/torch_io.py:37-58`

```python
def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
    fs = get_filesystem(path)
    fs.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_save(checkpoint, path)
```

#### 步骤5：_atomic_save() - 最终写入文件

**文件**: `lightning_fabric/utilities/cloud_io.py:65-80`

```python
def _atomic_save(checkpoint: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """原子保存checkpoint，避免生成不完整的文件"""
    bytesbuffer = io.BytesIO()
    torch.save(checkpoint, bytesbuffer)  # ← PyTorch原生保存
    with fsspec.open(filepath, "wb") as f:
        f.write(bytesbuffer.getvalue())
```

### 2.4 checkpoint 文件内容详解

最终保存的 `.ckpt` 文件是一个 PyTorch 字典，包含以下关键字段：

| 字段 | 类型 | 内容说明 |
|------|------|----------|
| `epoch` | int | 当前训练到第几个 epoch |
| `global_step` | int | 全局训练步数（batch数） |
| `pytorch-lightning_version` | str | PyTorch Lightning 版本号 |
| `state_dict` | Dict[str, Tensor] | **模型权重**，key 为参数名，value 为张量 |
| `optimizer_states` | List[Dict] | **优化器状态**，包含动量 `exp_avg`、`exp_avg_sq` 等 |
| `lr_schedulers` | List[Dict] | 学习率调度器状态 |
| `callbacks` | Dict | 各回调的状态（如 ModelCheckpoint） |
| `loops` | Dict | 训练循环的进度状态 |
| `hparams` | Dict | 超参数配置 |
| `MixedPrecision` | Dict | 混合精度训练的 scaler 状态（如果使用） |

### 2.5 state_dict 实际内容示例

以 SparseDriveAgent 为例，`state_dict` 的结构如下：

```python
{
    "agent._sparsedrive_model._backbone.img_backbone.conv1.weight": tensor([[[[...]]]]),
    "agent._sparsedrive_model._backbone.img_backbone.bn1.weight": tensor([...]),
    "agent._sparsedrive_model._status_encoding.weight": tensor([...]),
    "agent._sparsedrive_model._trajectory_head.path_pos_embed.0.weight": tensor([...]),
    "agent._sparsedrive_model._trajectory_head.traj_vocab": tensor([...]),  # 固定参数也会保存
    # ... 所有参数
}
```

**注意**：前缀为 `agent.`，因为 `SparseDriveAgent` 是 LightningModule 的子模块。加载时需要移除前缀：

```python
state_dict = torch.load(checkpoint_path)["state_dict"]
self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})
```

### 2.6 optimizer_states 实际内容示例

以 Adam 优化器为例：

```python
[{
    "param_groups": [{
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "params": [0, 1, 2, ...]
    }],
    "state": {
        0: {
            "step": 12345,
            "exp_avg": tensor([...]),    # m_t (一阶动量)
            "exp_avg_sq": tensor([...]), # v_t (二阶动量)
        },
        # ... 每个参数的状态
    }
}]
```

### 2.7 设计亮点

1. **分层设计**：从 Trainer → Strategy → CheckpointIO → torch.save，职责清晰
2. **分布式支持**：Strategy 层处理多 GPU 同步，只在主进程写入
3. **原子写入**：先写入内存缓冲区，再一次性写入文件，避免断电导致文件损坏
4. **可扩展性**：可以替换 CheckpointIO 实现远程存储（S3、GCS等）
5. **完整恢复**：不仅保存模型权重，还保存优化器状态，支持断点续训

---

## 三、参数量统计脚本

以下脚本可用于计算模型的精确参数量：

```python
# count_parameters_detailed.py
import torch
from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel

def count_parameters(model):
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return total, trainable

config = SparseDriveConfig()
model = SparseDriveModel(config)
total, trainable = count_parameters(model)

print(f"总参数量: {total/1e6:.2f}M")
print(f"可训练参数: {trainable/1e6:.2f}M")
print(f"固定参数: {(total-trainable)/1e6:.2f}M")
```

**运行方式**:
```bash
source scripts/cache/path_export.sh
python count_parameters_detailed.py
```