# TensorRT 加速部署方案

## 一、需求分析

### 当前问题
- Orin 控制器上 CPU 推理时间约 8-10 秒，无法满足实时性要求（目标 < 100ms）
- 需要使用 TensorRT 进行 GPU 加速

### 目标
- 推理时间 < 100ms（达到 10+ FPS）
- 使用 FP16 精度进行优化
- 保持与原模型相同的输出结果

### 参考文档
`docs/readme-训练部署.md` 中 TensorRT 编译部分

---

## 二、技术方案

### 整体流程
```
PyTorch 模型 (Python)
        ↓
torch.jit.trace
        ↓
TorchScript 模型 (.pt)
        ↓
torch_tensorrt.compile (在 Orin 上执行)
        ↓
TensorRT 优化模型 (.ts)
        ↓
C++ 部署 (LibTorch + TensorRT)
```

### 关键要点

| 要点 | 说明 |
|------|------|
| **TensorRT 编译位置** | 必须在 Orin 上编译（不同 GPU 架构的 engine 不兼容） |
| **精度选择** | FP16 通常提供 2-3 倍加速，INT8 需要校准 |
| **保存格式** | 优化后的模型仍是 TorchScript 格式 (`.ts`) |
| **运行时依赖** | 需要安装 `libtorch` + `libnvinfer` |

---

## 三、文件结构

```
scripts/deployment/
├── export_tensorrt.py         # 新增：TensorRT 导出脚本
├── cpp/
│   ├── include/
│   │   └── sparsedrive_infer.h
│   ├── src/
│   │   ├── main.cpp
│   │   └── sparsedrive_infer.cpp
│   ├── CMakeLists.txt
│   ├── build.sh
│   └── run.sh
```

---

## 四、实现步骤

### 步骤 1: 创建 TensorRT 导出脚本 (`export_tensorrt.py`)

功能：
- 加载现有的 TorchScript 模型
- 使用 `torch_tensorrt.compile` 进行优化
- 支持 FP16 精度
- 保存优化后的模型

### 步骤 2: 在 Orin 上安装依赖

需要安装：
- `torch_tensorrt`
- TensorRT 库

### 步骤 3: 在 Orin 上执行导出

```bash
python export_tensorrt.py --model model_scripted_cpu.pt --output model_trt.ts
```

### 步骤 4: 更新 C++ 代码支持 TensorRT

修改 `sparsedrive_infer.cpp` 和 `main.cpp`：
- 添加 TensorRT 相关的错误处理
- 确保正确加载 `.ts` 格式的 TensorRT 模型

### 步骤 5: 更新编译脚本

确保编译时链接 TensorRT 库。

---

## 五、关键依赖

| 依赖 | 版本 | 安装方式 |
|------|------|----------|
| torch | 2.5.0+cu126 | pip |
| torch_tensorrt | 对应 torch 版本 | pip |
| TensorRT | 8.x+ | apt/pip |
| CUDA | 12.6 | 系统安装 |

---

## 六、风险与注意事项

### 风险 1: torch_tensorrt 安装失败
- 解决方案：使用与 torch 版本匹配的 torch_tensorrt
- 可能需要从源码编译

### 风险 2: TensorRT 编译失败
- 解决方案：检查模型中是否有不支持的操作
- 使用 `truncate_long_and_double=True` 参数

### 风险 3: 精度损失
- 解决方案：先使用 FP16，必要时使用 INT8 校准

### 风险 4: 模型格式不兼容
- 解决方案：确保在 Orin 上编译，使用相同版本的 torch

---

## 七、验证计划

1. **导出验证**: 确认 TensorRT 模型可以成功导出
2. **加载验证**: 确认 C++ 程序可以加载模型
3. **推理验证**: 确认模型能正常推理
4. **性能验证**: 测量推理时间，确认达到目标
5. **结果验证**: 对比原始模型和 TensorRT 模型的输出差异

---

## 八、交付物

- TensorRT 导出脚本
- 更新后的 C++ 推理代码
- 编译和运行脚本
- 使用说明文档

---

## 九、前置条件

- Orin 上已安装兼容的 torch 版本（2.5.0+cu126）
- Orin 上已安装 TensorRT
- 原始 TorchScript 模型已传输到 Orin

---

## 十、预期性能提升

| 设备 | 原始推理时间 | 预期推理时间 | 提升倍数 |
|------|-------------|-------------|----------|
| CPU | ~8-10 秒 | ~8-10 秒 | 1x |
| GPU (PyTorch) | ~500ms-1s | ~500ms-1s | 1x |
| GPU (TensorRT FP16) | - | ~50-100ms | 5-10x |