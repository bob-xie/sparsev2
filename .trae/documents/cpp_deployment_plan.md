# C++ 部署实现计划

## 一、需求分析

用户希望使用 Orin 控制器上已有的 libtorch-2.5.0 进行 C++ 部署，替代 Python 部署方案。

### 现有资源
- **LibTorch 版本**: 2.5.0 (aarch64)
- **路径**: `/etc/lg/truck/3rdparty/aarch64/libtorch-2.5.0/`
- **模型文件**: `exp/deployment/model_scripted_cpu.pt`

### 目标
- 创建 C++ 推理代码
- 支持 GPU 推理（利用 Orin 的 CUDA）
- 提供完整的编译和运行脚本

---

## 二、文件结构

```
scripts/deployment/
├── cpp/                      # 新增目录
│   ├── include/              # 头文件
│   │   └── sparsedrive_infer.h
│   ├── src/                  # 源文件
│   │   ├── main.cpp
│   │   └── sparsedrive_infer.cpp
│   ├── CMakeLists.txt        # CMake 配置
│   ├── build.sh              # 编译脚本
│   └── run.sh                # 运行脚本
```

---

## 三、实现步骤

### 步骤 1: 创建头文件 (`sparsedrive_infer.h`)

定义推理类接口：
- 模型加载
- 推理执行
- 输入输出处理

### 步骤 2: 创建实现文件 (`sparsedrive_infer.cpp`)

实现核心功能：
- 使用 `torch::jit::load()` 加载模型
- 设置 CUDA 设备
- 构建输入张量
- 执行推理并返回结果

### 步骤 3: 创建主程序 (`main.cpp`)

提供测试入口：
- 命令行参数解析
- 加载模型
- 准备测试输入
- 执行推理并输出结果

### 步骤 4: 创建 CMakeLists.txt

配置编译环境：
- 设置 LibTorch 路径
- 链接 CUDA 库
- 设置编译选项

### 步骤 5: 创建编译和运行脚本

- `build.sh`: 设置环境变量并编译
- `run.sh`: 运行推理程序

---

## 四、关键依赖

| 依赖 | 版本 | 路径 |
|------|------|------|
| LibTorch | 2.5.0 | `/etc/lg/truck/3rdparty/aarch64/libtorch-2.5.0/` |
| CUDA | 12.6 | 系统安装 |
| cuDNN | 9.x | 系统安装 |

---

## 五、风险与注意事项

### 风险 1: LibTorch 路径配置
- 确保 CMake 正确指向 libtorch-2.5.0
- 设置 `CMAKE_PREFIX_PATH` 环境变量

### 风险 2: CUDA 兼容性
- libtorch-2.5.0 需要匹配的 CUDA 版本
- 确保 Orin 的 CUDA 版本与 libtorch 兼容

### 风险 3: 模型格式兼容性
- TorchScript 模型需要在相同或兼容的 PyTorch 版本下导出
- 如果版本不兼容，需要重新导出模型

---

## 六、验证计划

1. **编译验证**: 确认项目能成功编译
2. **推理验证**: 确认模型能正常推理
3. **GPU 验证**: 确认 CUDA 加速生效
4. **结果验证**: 对比 Python 和 C++ 输出结果

---

## 七、交付物

- C++ 推理代码
- CMake 配置文件
- 编译和运行脚本
- 使用说明文档

---

## 八、执行时间估计

| 步骤 | 时间 |
|------|------|
| 创建代码文件 | 30 分钟 |
| 编译调试 | 30 分钟 |
| 验证测试 | 30 分钟 |
| 总计 | 90 分钟 |

---

## 九、前置条件

- 模型文件 `model_scripted_cpu.pt` 已导出
- Orin 上已安装 CUDA 工具链
- LibTorch 2.5.0 路径正确