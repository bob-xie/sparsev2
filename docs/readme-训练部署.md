# PyTorch 训练到部署完整指南

> 本文档汇总了关于 TorchScript、PyTorch TensorRT 和 LibTorch 在训练、加速、部署中的使用方法和配合方案。

---

## 目录

1. [核心组件概述](#一核心组件概述)
2. [TorchScript 导出与 TensorRT 加速](#二torchscript-导出与-tensorrt-加速)
3. [完整部署流程](#三完整部署流程)
4. [常见问题澄清](#四常见问题澄清)
5. [版本匹配与依赖](#五版本匹配与依赖)

---

## 一、核心组件概述

### 1. TorchScript

**核心作用**：将 Python 编写的动态 PyTorch 模型转换为可序列化、可优化的静态图表示。

| 特性 | 说明 |
|------|------|
| **Tracing** | 通过示例输入追踪模型执行路径 |
| **Scripting** | 直接编译 Python 代码，支持控制流 |
| **中间表示** | 生成与 Python 无关的 `.pt` 文件 |

**典型使用流程**：

```python
import torch

# 1. 训练好的模型
model = MyModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 2. 转换为 TorchScript (Tracing 方式)
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# 3. 保存为可部署格式
traced_model.save("model.pt")

# 4. 加载并推理（无需 Python 环境）
loaded_model = torch.jit.load("model.pt")
output = loaded_model(input_tensor)
```

**适用场景**：
- 需要脱离 Python 运行时部署
- 模型需要跨平台（C++、移动端）
- 作为进一步优化的中间格式

---

### 2. PyTorch TensorRT

**核心作用**：利用 NVIDIA TensorRT 对 PyTorch 模型进行深度学习推理优化，显著提升 GPU 上的推理速度。

| 特性 | 说明 |
|------|------|
| **自动转换** | 将 TorchScript/FX Graph 转为 TensorRT 引擎 |
| **混合精度** | 支持 FP16/INT8 量化加速 |
| **层融合** | 自动合并算子减少内存访问 |
| **动态形状** | 支持变长输入 |

**典型使用流程**：

```python
import torch
import torch_tensorrt

# 1. 准备模型
model = MyModel().eval().cuda()

# 2. 编译为 TensorRT 优化版本
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224])],
    enabled_precisions={torch.float16},  # FP16 加速
)

# 3. 保存优化后的模型
torch.jit.save(trt_model, "model_trt.ts")

# 4. 推理（自动使用 TensorRT 后端）
with torch.no_grad():
    output = trt_model(input_tensor.cuda())
```

**加速效果**：
- 通常可获得 **2-5 倍** 的推理速度提升
- 显存占用降低（通过层融合和内存优化）

---

### 3. LibTorch

**核心作用**：提供 PyTorch 的 C++ 前端，用于在生产环境中加载和运行 TorchScript 模型。

| 特性 | 说明 |
|------|------|
| **无 Python 依赖** | 纯 C++ 库，适合嵌入式/服务端 |
| **加载 TorchScript** | 直接运行 `.pt` 文件 |
| **GPU/CPU 支持** | 完整的张量运算能力 |
| **线程安全** | 适合高并发服务 |

**典型使用流程**：

```cpp
#include <torch/torch.h>
#include <torch/script.h>

int main() {
    // 1. 加载 TorchScript 模型
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("model.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model\n";
        return -1;
    }
    
    // 2. 移动到 GPU（如需要）
    module.to(torch::kCUDA);
    
    // 3. 准备输入
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}).to(torch::kCUDA));
    
    // 4. 执行推理
    at::Tensor output = module.forward(inputs).toTensor();
    
    return 0;
}
```

---

## 二、TorchScript 导出与 TensorRT 加速

### 问题：TorchScript 导出后可以用 TensorRT 加速吗？

**答案：完全可以！**

TorchScript 导出后完全可以使用 TensorRT 进行加速。实际上，这是 NVIDIA GPU 部署的推荐流程之一。

### 方案一：torch-tensorrt 直接编译（推荐）

```python
import torch
import torch_tensorrt

# 1. 加载或创建 TorchScript 模型
# 方式 A: 从 Python 模型直接转换
model = MyModel().eval().cuda()
traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224).cuda())

# 方式 B: 加载已有的 TorchScript 文件
# traced_model = torch.jit.load("model.pt")

# 2. 使用 torch-tensorrt 编译加速
trt_model = torch_tensorrt.compile(
    traced_model,
    inputs=[
        torch_tensorrt.Input(
            min_shape=[1, 3, 224, 224],
            opt_shape=[4, 3, 224, 224],
            max_shape=[8, 3, 224, 224],
            dtype=torch.float32
        )
    ],
    enabled_precisions={torch.float32, torch.float16},  # FP16 混合精度
    workspace_size=1 << 30,  # 1GB 工作空间
)

# 3. 保存 TensorRT 优化后的模型
torch.jit.save(trt_model, "model_trt.ts")

# 4. 推理（自动使用 TensorRT 后端）
with torch.no_grad():
    output = trt_model(input_tensor.cuda())
```

### 方案二：TRT Engine 导出（C++ 部署用）

```python
import torch_tensorrt

# 编译并保存为 TensorRT engine
trt_model = torch_tensorrt.compile(
    traced_model,
    inputs=[torch_tensorrt.Input([1, 3, 224, 224])],
    enabled_precisions={torch.float16},
)

# 保存的 .ts 文件包含 TensorRT engine
torch.jit.save(trt_model, "model_trt.ts")
```

**C++ 加载方式**：
```cpp
#include <torch/torch.h>
#include <torch/script.h>

// 加载 TensorRT 优化的 TorchScript 模型
auto module = torch::jit::load("model_trt.ts");
module.to(torch::kCUDA);

// 推理时会自动调用 TensorRT
auto output = module.forward(inputs).toTensor();
```

### 关键要点

| 要点 | 说明 |
|------|------|
| **输入格式** | TorchScript 是 TensorRT 编译的理想输入格式 |
| **动态形状** | 支持通过 `min/opt/max_shape` 配置动态 batch/尺寸 |
| **精度选择** | FP16 通常提供 2-3 倍加速，INT8 需要校准 |
| **保存格式** | 优化后的模型仍是 TorchScript 格式 (`.ts`) |
| **运行时依赖** | 部署环境需要安装 `libtorch` + `libnvinfer` |

---

## 三、完整部署流程

### 问题：TorchScript → TensorRT → C++ 部署的完整方案

**完整流程图**：

```
PyTorch 模型 (Python)
        ↓
torch.jit.script
        ↓
TorchScript 模型 (.pt)
        ↓
torch_tensorrt.compile
        ↓
TensorRT 优化模型 (.ts)
        ↓
C++ 部署 (LibTorch + TensorRT)
```

**关键结论**：LibTorch 可以加载 `.ts` 格式的 TensorRT 优化模型，但需要满足以下条件：

| 条件 | 说明 |
|------|------|
| 编译和部署使用 **相同版本** 的 LibTorch | 版本不匹配会导致加载失败 |
| 部署环境安装 **TensorRT 库** | `libnvinfer.so` 等必须存在 |
| 部署环境有 **CUDA 驱动和运行时** | GPU 环境必需 |
| 使用 **相同的 GPU 架构** | 不同架构的 engine 不兼容 |

---

### 第一步：Python 端导出 TensorRT 优化模型

```python
# export_model.py
import torch
import torch_tensorrt

# ========== 1. 定义或加载模型 ==========
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(64 * 56 * 56, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 加载训练好的权重（如果有）
model = MyModel().eval().cuda()

# ========== 2. 转换为 TorchScript ==========
example_input = torch.randn(1, 3, 224, 224).cuda()

try:
    # 先尝试 script（支持控制流）
    scripted_model = torch.jit.script(model)
    print("使用 torch.jit.script 成功")
except Exception as e:
    # 失败则使用 trace
    print(f"script 失败，使用 trace: {e}")
    scripted_model = torch.jit.trace(model, example_input)

# 保存纯 TorchScript 版本（可选）
scripted_model.save("model_scripted.pt")
print("TorchScript 模型已保存: model_scripted.pt")

# ========== 3. TensorRT 编译优化 ==========
print("开始 TensorRT 编译...")

trt_model = torch_tensorrt.compile(
    scripted_model,  # TorchScript 模型作为输入
    inputs=[
        torch_tensorrt.Input(
            min_shape=[1, 3, 224, 224],      # 最小 batch
            opt_shape=[4, 3, 224, 224],      # 最优 batch
            max_shape=[8, 3, 224, 224],      # 最大 batch
            dtype=torch.float32,
            name="input_0"                   # 输入名称（可选）
        )
    ],
    enabled_precisions={torch.float32, torch.float16},  # 启用 FP16
    workspace_size=1 << 30,  # 1GB 工作空间
    truncate_long_and_double=True,  # 处理 long/double 类型
)

# ========== 4. 保存 TensorRT 优化模型 ==========
torch.jit.save(trt_model, "model_trt.ts")
print("TensorRT 优化模型已保存: model_trt.ts")

# ========== 5. 验证模型可以正常加载和推理 ==========
print("验证模型...")
loaded_model = torch.jit.load("model_trt.ts")
test_input = torch.randn(1, 3, 224, 224).cuda()

with torch.no_grad():
    output = loaded_model(test_input)
    print(f"输出形状: {output.shape}")
    print("验证成功！")
```

---

### 第二步：C++ LibTorch 加载和推理

#### C++ 推理代码

```cpp
// infer.cpp
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <chrono>
#include <vector>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path-to-model_trt.ts>" << std::endl;
        return -1;
    }

    const std::string model_path = argv[1];

    // ========== 1. 检查 CUDA 可用性 ==========
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available! TensorRT model requires GPU." << std::endl;
        return -1;
    }
    std::cout << "CUDA is available. Device count: " << torch::cuda::device_count() << std::endl;

    // ========== 2. 加载 TensorRT 优化模型 ==========
    torch::jit::script::Module module;
    try {
        std::cout << "Loading model from: " << model_path << std::endl;
        module = torch::jit::load(model_path);
        std::cout << "Model loaded successfully!" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        std::cerr << "Make sure the model was compiled with the same PyTorch/TensorRT version." << std::endl;
        return -1;
    }

    // ========== 3. 将模型移动到 GPU ==========
    try {
        module.to(torch::kCUDA);
        std::cout << "Model moved to CUDA." << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error moving model to CUDA: " << e.what() << std::endl;
        return -1;
    }

    // ========== 4. 准备输入数据 ==========
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    torch::Tensor input_tensor = torch::randn(input_shape).to(torch::kCUDA);
    std::cout << "Input tensor shape: " << input_tensor.sizes() << std::endl;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    // ========== 5. 预热推理 ==========
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        auto output = module.forward(inputs);
    }
    torch::cuda::synchronize();

    // ========== 6. 正式推理并计时 ==========
    const int num_iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    torch::Tensor output;
    for (int i = 0; i < num_iterations; ++i) {
        auto result = module.forward(inputs);
        output = result.toTensor();
    }
    
    torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // ========== 7. 输出结果 ==========
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float avg_latency_ms = duration.count() / 1000.0f / num_iterations;

    std::cout << "\n========== Inference Results ==========" << std::endl;
    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << "Output dtype: " << output.dtype() << std::endl;
    std::cout << "Average latency: " << avg_latency_ms << " ms" << std::endl;
    std::cout << "Throughput: " << 1000.0f / avg_latency_ms << " FPS" << std::endl;

    return 0;
}
```

#### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tensorrt_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========== 查找 LibTorch ==========
set(Torch_DIR "/path/to/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# ========== 查找 CUDA ==========
find_package(CUDA REQUIRED)

# ========== 查找 TensorRT ==========
set(TENSORRT_ROOT "/usr/local/tensorrt")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found!")
endif()

message(STATUS "Found TensorRT: ${TENSORRT_LIBRARY}")
message(STATUS "Found Torch: ${TORCH_LIBRARIES}")

# ========== 创建可执行文件 ==========
add_executable(infer infer.cpp)

# ========== 链接库 ==========
target_link_libraries(infer 
    ${TORCH_LIBRARIES}
    ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

# ========== 包含目录 ==========
target_include_directories(infer PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# ========== 编译选项 ==========
target_compile_options(infer PRIVATE 
    -O3 
    -D_GLIBCXX_USE_CXX11_ABI=1
)

set_property(TARGET infer PROPERTY CXX_STANDARD 17)
```

---

### 第三步：编译和运行

```bash
# ========== 编译 ==========
mkdir build && cd build
cmake ..
make -j$(nproc)

# ========== 运行 ==========
./infer /path/to/model_trt.ts
```

---

## 四、常见问题澄清

### 问题：TorchScript 能否被 TensorRT 优化？

**澄清**：

| 方案 | 是否可行 | 说明 |
|------|----------|------|
| **TensorRT 原生 API** 直接加载 `.pt` 文件 | ❌ 不可行 | TensorRT 原生不支持 TorchScript 格式 |
| **torch-tensorrt** 编译 TorchScript 模型 | ✅ **完全可行** | PyTorch 官方提供的集成方案 |
| **ONNX 中间转换** | ✅ 可行 | TorchScript → ONNX → TensorRT |

**正确理解 torch-tensorrt**：

`torch-tensorrt` 是 **PyTorch 与 TensorRT 的官方集成工具**，它的工作原理是：

```
TorchScript 模型
      ↓
torch-tensorrt 解析计算图
      ↓
转换为 TensorRT Network Definition
      ↓
TensorRT Builder 优化生成 Engine
      ↓
包装为 TorchScript 可执行模块
```

**关键点**：`torch-tensorrt.compile()` 的输入可以是：
- `torch.nn.Module`（PyTorch 模型）
- `torch.jit.ScriptModule`（TorchScript 模型）✅

### 验证代码

```python
import torch
import torch_tensorrt

# 创建 TorchScript 模型
class MyModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x * 2)

model = MyModel().eval().cuda()
traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224).cuda())

# 验证：traced_model 是 TorchScript 类型
print(type(traced_model))  # <class 'torch.jit._script.ScriptModule'>

# ✅ 可以直接用 torch-tensorrt 编译
trt_model = torch_tensorrt.compile(
    traced_model,  # TorchScript 模型作为输入
    inputs=[torch_tensorrt.Input([1, 3, 224, 224])],
    enabled_precisions={torch.float16},
)

# 保存并部署
torch.jit.save(trt_model, "model_trt.ts")
```

---

## 五、版本匹配与依赖

### 版本匹配要求（非常重要！）

| 组件 | 编译环境版本 | 部署环境版本 |
|------|-------------|-------------|
| PyTorch | 2.0.1 | 2.0.1 (LibTorch) |
| TensorRT | 8.6.1 | 8.6.1 |
| CUDA | 11.8 | 11.8 |
| cuDNN | 8.9 | 8.9 |

**版本不匹配会导致加载失败！**

### 部署环境依赖

```bash
# 需要安装的库文件
# 1. LibTorch
# 下载地址: https://download.pytorch.org/libtorch/
# 选择: libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip

# 2. TensorRT
# 下载地址: https://developer.nvidia.com/tensorrt
# 需要设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `Error loading the model` | 版本不匹配 | 确保编译和部署使用相同版本 |
| `CUDA error: no kernel image` | GPU 架构不匹配 | 在目标 GPU 上重新编译 |
| `undefined symbol` | ABI 不匹配 | 使用 `-D_GLIBCXX_USE_CXX11_ABI=1` |
| `TensorRT not found` | 缺少 TensorRT 库 | 安装 TensorRT 并设置路径 |

---

## 六、场景选择建议

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| **快速原型/研究** | Python + PyTorch | 开发效率高，调试方便 |
| **内部 API 服务** | TorchScript + Python | 部署简单，性能可接受 |
| **高性能生产服务** | TensorRT + C++/TRITON | 最大化 GPU 利用率 |
| **移动端/嵌入式** | TorchScript + PyTorch Mobile | 轻量级，跨平台 |
| **边缘设备 (Jetson)** | TensorRT | 针对嵌入式 GPU 优化 |
| **跨平台 C++ 应用** | LibTorch | 无 Python 依赖，可移植 |

---

## 七、完整流程总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Python 环境                                 │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────┐ │
│  │ PyTorch 模型 │ → │ torch.jit.script │ → │ torch_tensorrt.compile│ │
│  └─────────────┘    └─────────────────┘    └─────────────────────┘ │
│                                                      ↓              │
│                                               model_trt.ts          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ 复制到部署机器
┌─────────────────────────────────────────────────────────────────────┐
│                          C++ 部署环境                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ LibTorch + CUDA │ → │ torch::jit::load │ → │ module.forward() │ │
│  │  + TensorRT     │    │ ("model_trt.ts") │    │ (自动使用 TensorRT)│ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**核心要点**：
- TorchScript 是连接 Python 训练与生产部署的桥梁
- TensorRT 是针对 NVIDIA GPU 的深度优化层
- LibTorch 是脱离 Python 的生产环境执行引擎
- **版本必须严格匹配**是成功部署的关键

---

*文档生成时间：2026-05-31*
