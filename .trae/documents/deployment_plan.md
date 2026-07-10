# SparseDriveV2 部署到 Jetson Orin 控制器计划

## 一、当前状态分析

### 1.1 已完成工作
| 任务 | 状态 | 说明 |
|------|------|------|
| 训练 | ✅ 完成 | 使用 navmini 数据集训练，模型保存在 `exp/sparsedrive_agent/2026.06.17.17.46.52/periodic_pdm_ckpts/ep0010.ckpt` |
| 评估 | ✅ 完成 | 在 navmini 数据集上评估，平均分数 0.848 |
| 纯 PyTorch 实现 | ✅ 可用 | `blocks.py` 支持 `use_deformable_func=False` |

### 1.2 目标
将训练好的 SparseDriveV2 模型部署到 **Jetson Orin** 控制器上运行测试用例。

### 1.3 关键技术路径
```
训练好的模型 (.ckpt)
    ↓
加载到纯 PyTorch 实现 (use_deformable_func=False)
    ↓
导出为 TorchScript (.pt)
    ↓
复制到 Jetson Orin
    ↓
LibTorch (C++) 加载并推理
```

---

## 二、部署计划

### 步骤 1：创建模型导出脚本

**目标**：将训练好的模型导出为 TorchScript 格式，支持脱离 Python 运行时部署。

**文件**：`scripts/deployment/export_torchscript.py`

**关键要点**：
- 使用纯 PyTorch 实现（`use_deformable_func=False`）
- 处理模型初始化和权重加载
- 导出完整的推理接口
- 验证导出模型可以正常推理

### 步骤 2：创建 Python 测试用例

**目标**：验证导出的 TorchScript 模型在 Python 环境中可以正常运行。

**文件**：`scripts/deployment/test_torchscript.py`

**测试内容**：
- 加载 TorchScript 模型
- 准备测试输入数据
- 执行推理并验证输出
- 对比原始模型和导出模型的输出一致性

### 步骤 3：创建 C++ 部署代码

**目标**：创建基于 LibTorch 的 C++ 推理代码。

**文件**：
- `scripts/deployment/infer.cpp` - 推理主程序
- `scripts/deployment/CMakeLists.txt` - CMake 构建配置

**功能**：
- 加载 TorchScript 模型
- 准备相机图像输入
- 执行推理
- 输出预测轨迹

### 步骤 4：在 Jetson Orin 上编译和部署

**目标**：在目标硬件上编译并运行部署代码。

**步骤**：
1. 安装依赖（LibTorch、CUDA、TensorRT）
2. 复制模型文件和源代码
3. 编译 C++ 代码
4. 运行测试用例

---

## 三、关键技术要点

### 3.1 版本匹配要求

| 组件 | 版本 | 说明 |
|------|------|------|
| PyTorch | 2.0.1 | 训练和导出环境 |
| LibTorch | 2.0.1 (cxx11 ABI) | 部署环境 |
| CUDA | 11.8 | 训练和部署环境 |
| TensorRT | 8.6.1 | 可选，用于推理加速 |

### 3.2 模型导出注意事项

1. **必须使用纯 PyTorch 实现**：
   - 设置 `use_deformable_func=False`
   - 避免自定义 CUDA 扩展
   - 确保所有操作都使用标准 PyTorch API

2. **输入输出格式**：
   - 输入：相机图像（多视角）+ ego 状态
   - 输出：预测轨迹（8个点，每个点包含 x,y,heading）

3. **动态形状处理**：
   - 确保模型支持 batch size=1 的推理
   - 避免硬编码输入形状

### 3.3 Jetson Orin 环境准备

1. **安装 JetPack**：
   ```bash
   sudo apt-get update
   sudo apt-get install nvidia-jetpack
   ```

2. **安装 LibTorch**：
   - 下载地址：https://download.pytorch.org/libtorch/
   - 选择：`libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip`

3. **设置环境变量**：
   ```bash
   export LD_LIBRARY_PATH=/usr/local/libtorch/lib:$LD_LIBRARY_PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

---

## 四、风险与应对措施

### 4.1 潜在风险

| 风险 | 描述 | 应对措施 |
|------|------|----------|
| 版本不匹配 | 编译和部署环境版本不一致导致加载失败 | 严格按照版本要求安装依赖 |
| 自定义 CUDA 扩展 | 导出时使用了自定义 CUDA kernel | 强制使用 `use_deformable_func=False` |
| GPU 架构不兼容 | 在不同 GPU 架构上编译的模型不兼容 | 在目标 GPU 上重新导出或使用通用架构 |
| 内存不足 | Jetson Orin 内存有限 | 使用较小的 batch size，优化内存使用 |
| 性能问题 | 推理速度不够实时 | 使用 TensorRT 优化，FP16 推理 |

### 4.2 验证方案

1. **Python 环境验证**：
   - 导出模型后在训练环境中验证
   - 确保输出与原始模型一致

2. **Jetson 环境验证**：
   - 先运行简单的测试用例
   - 逐步增加复杂度
   - 监控内存和性能指标

---

## 五、预期输出

### 5.1 导出阶段
- `model_scripted.pt` - TorchScript 模型文件

### 5.2 测试阶段
- Python 测试报告（输出形状、推理时间、与原始模型的差异）
- C++ 测试报告（输出形状、推理时间）

### 5.3 部署阶段
- 可执行文件 `infer`
- 推理测试结果

---

## 六、执行顺序

1. ✅ 分析当前状态和需求
2. 创建模型导出脚本
3. 创建 Python 测试用例
4. 创建 C++ 部署代码
5. 在训练环境中测试导出模型
6. 在 Jetson Orin 上编译和测试

---

*计划生成时间：2026-07-10*