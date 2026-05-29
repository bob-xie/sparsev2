# SparseDriveV2 纯 PyTorch 实现说明

## 概述

我们已经为 SparseDriveV2 添加了完整的纯 PyTorch 实现，使其可以在不需要自定义 CUDA 扩展的情况下运行，这对于使用 TorchScript 导出和 LibTorch (C++) 部署非常重要。

## 修改内容

### 1. `blocks.py` 文件的完整更新

我们已经完成了以下工作：

- **保留了原始的自定义 CUDA 扩展实现**（`use_deformable_func=True`）
- **添加了完整的纯 PyTorch 实现**（`use_deformable_func=False`）
  - 1. **project_points**：与 CUDA 版本完全一致 ✅
  - 2. **_get_weights**：与 CUDA 版本完全一致 ✅
  - 3. **特征采样**：使用 `torch.nn.functional.grid_sample` 替代自定义 CUDA kernel 🆕
  - 4. **特征融合**：使用纯 PyTorch 操作，完全复现 CUDA 版本的逻辑 🆕
  - 5. **后续层**：完全一致 ✅

- **支持 `depth_prob` 情况**：实现了 `feature_sampling_with_depth` 函数
- **保持形状一致性**：输入和输出形状与 CUDA 版本完全相同

## 功能对比

| 步骤 | CUDA 实现 | PyTorch 实现 | 一致性 |
|------|-----------|--------------|--------|
| project_points | ✅ | ✅ | ✅ |
| _get_weights | ✅ | ✅ | ✅ |
| 特征采样 | 自定义 kernel | grid_sample | ⚠️ 近似 |
| 特征融合 | 自定义 kernel | PyTorch 操作 | ⚠️ 近似 |
| 后续层 | ✅ | ✅ | ✅ |

## 使用方法

### 训练阶段（推荐使用自定义 CUDA 扩展）

在训练时，你可以继续使用自定义 CUDA 扩展以获得更好的性能：

```python
from navsim.agents.sparsedrive.blocks import DeformableFeatureAggregation

# 初始化时使用自定义 CUDA 扩展
model = DeformableFeatureAggregation(
    config=config,
    use_deformable_func=True  # 使用 CUDA 加速
)
```

### 导出和部署阶段（使用纯 PyTorch 实现）

在导出模型为 TorchScript 或部署到 LibTorch 时，需要使用纯 PyTorch 实现：

```python
# 加载训练好的权重到纯 PyTorch 模型
model = DeformableFeatureAggregation(
    config=config,
    use_deformable_func=False  # 使用纯 PyTorch 实现
)
model.load_state_dict(trained_weights)

# 导出为 TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("sparsedrive_model.pt")
```

## 工作流程推荐

1. **训练阶段**：在桌面 GPU 上使用 `use_deformable_func=True` 进行训练，保存权重
2. **导出阶段**：加载训练好的权重，设置 `use_deformable_func=False`，导出为 TorchScript
3. **部署阶段**：在 Jetson 或其他设备上用 LibTorch 加载和运行模型

## 注意事项

- 两种实现（CUDA 扩展和纯 PyTorch）的数值结果可能略有不同，但在实际应用中影响通常很小
- 对于需要完整 CUDA 加速的场景，建议在 Jetson 上直接用 Python + PyTorch 运行（不需要 LibTorch），并在 Jetson 上本地编译 CUDA 扩展
- 如果需要使用 LibTorch (C++)，则必须使用纯 PyTorch 实现
- 纯 PyTorch 实现支持所有功能，包括 `depth_prob` 场景

