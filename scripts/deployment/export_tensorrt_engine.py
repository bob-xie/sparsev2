"""
TensorRT 引擎编译脚本 - 在 Orin 上执行

功能：将 ONNX 模型编译为针对 Orin GPU (sm_87) 优化的推理引擎

使用方法：
    python3 export_tensorrt_engine.py --onnx model.onnx --output model.engine --fp16

关键要点：
    1. 必须在 Orin 上执行（GPU 架构 sm_87）
    2. 支持 FP16 混合精度加速
    3. 支持动态形状配置
"""

# 导入命令行参数解析模块，用于处理脚本输入参数
import argparse
# 导入操作系统接口模块，用于文件路径处理
import os
# 导入系统模块，用于程序退出等操作
import sys

# 导入 TensorRT 库，用于模型优化和引擎编译
import tensorrt as trt


def build_engine(onnx_path: str, output_path: str, use_fp16: bool = True,
                 workspace_size: int = 1 << 30, batch_size: int = 1):
    """
    核心函数：将 ONNX 模型编译为 TensorRT 推理引擎
    
    参数：
        onnx_path: str - ONNX 模型文件路径
        output_path: str - 编译后的 TensorRT 引擎输出路径
        use_fp16: bool - 是否使用 FP16 混合精度，默认 True
        workspace_size: int - TensorRT 工作空间大小（字节），默认 1GB (1 << 30)
        batch_size: int - 推理批次大小，默认 1
    """
    # 打印编译标题
    print("=" * 60)
    print("SparseDriveV2 TensorRT Engine Build")
    print("=" * 60)
    print(f"\nBuilding engine from ONNX: {onnx_path}")

    # 检查 ONNX 文件是否存在
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found - {onnx_path}")
        sys.exit(1)  # 文件不存在时退出程序

    # 创建 TensorRT 日志记录器，级别为 INFO（输出编译过程信息）
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # 创建 TensorRT Builder 对象，用于构建引擎
    # Builder 是 TensorRT 的核心组件，负责优化和编译模型
    builder = trt.Builder(TRT_LOGGER)
    
    # 创建 TensorRT Network 对象，用于表示计算图
    # EXPLICIT_BATCH 标志表示使用显式批处理模式（推荐）
    # 1 << int(...) 将枚举值转换为位掩码
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 创建 ONNX 解析器，用于将 ONNX 模型解析为 TensorRT Network
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print("\nParsing ONNX model...")
    # 以二进制只读模式打开 ONNX 文件
    with open(onnx_path, "rb") as f:
        # 解析 ONNX 文件内容
        if not parser.parse(f.read()):
            # 解析失败时打印错误信息
            print("ERROR: Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            sys.exit(1)
    print("✓ ONNX model parsed successfully!")

    # 打印输入层信息
    for i in range(network.num_inputs):
        layer = network.get_input(i)
        print(f"  Input '{layer.name}': shape={layer.shape}")
    
    # 创建优化配置文件（Optimization Profile）
    # 用于指定输入张量的最小、最优、最大形状，支持动态形状推理
    profile = builder.create_optimization_profile()
    
    # 遍历所有输入张量，为每个输入设置形状配置
    for i in range(network.num_inputs):
        layer = network.get_input(i)
        input_name = layer.name
        shape = list(layer.shape)
        
        # 设置 batch_size（替换第一维）
        shape[0] = batch_size
        
        # 尝试多种方式设置形状（兼容不同版本）
        try:
            # 方式1：使用 ITensor 对象（新版 API）
            profile.set_shape(layer, shape, shape, shape)
            print(f"  ✓ Set shape for '{input_name}' using ITensor")
        except TypeError:
            try:
                # 方式2：使用字符串名称 + list（新版 API）
                profile.set_shape(input_name, shape, shape, shape)
                print(f"  ✓ Set shape for '{input_name}' using string + list")
            except TypeError:
                try:
                    # 方式3：使用字符串名称 + trt.Dims（旧版 API）
                    profile.set_shape(input_name, trt.Dims(shape), trt.Dims(shape), trt.Dims(shape))
                    print(f"  ✓ Set shape for '{input_name}' using string + Dims")
                except Exception as e:
                    print(f"  ✗ Failed to set shape for '{input_name}': {e}")

    # 创建 Builder 配置对象，用于设置编译选项
    config = builder.create_builder_config()
    # 将优化配置文件添加到 Builder 配置中
    config.add_optimization_profile(profile)
    # 设置工作空间内存限制（用于中间张量计算）
    # trt.MemoryPoolType.WORKSPACE 表示工作空间内存池
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    # 检查是否启用 FP16 且当前平台支持快速 FP16 计算
    if use_fp16 and builder.platform_has_fast_fp16:
        # 启用 FP16 混合精度模式
        config.set_flag(trt.BuilderFlag.FP16)
        print("\n✓ FP16 mode enabled")
    elif use_fp16:
        # 当前平台不支持 FP16，回退到 FP32
        print("\n⚠ FP16 not supported on this platform, using FP32")

    print("\nBuilding engine (this may take several minutes)...")
    serialized_engine = None  # 用于保存序列化后的引擎字节流
    
    try:
        # TensorRT 10.x API 变化：build_engine 已移除
        # 新方式：先构建序列化网络，再反序列化为引擎
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("ERROR: build_serialized_network returned None")
            sys.exit(1)
        
        # 反序列化为引擎（用于后续验证）
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        if engine is None:
            print("ERROR: Deserialization returned None")
            sys.exit(1)
            
    except AttributeError:
        # 兼容旧版 TensorRT（< 10.x）
        try:
            engine = builder.build_engine(network, config)
            if engine is None:
                print("ERROR: Engine build returned None")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to build engine: {e}")
            sys.exit(1)
    except Exception as e:
        # 捕获其他异常并打印错误信息
        print(f"ERROR: Failed to build engine: {e}")
        sys.exit(1)

    print("✓ Engine built successfully!")

    print(f"\nSaving engine to: {output_path}")
    # 以二进制写入模式打开输出文件
    with open(output_path, "wb") as f:
        # TensorRT 10.x 使用 serialized_engine（已经是字节流）
        # 旧版使用 engine.serialize() 获取字节流
        if serialized_engine is not None:
            f.write(serialized_engine)
        else:
            f.write(engine.serialize())

    print(f"✓ Engine saved successfully!")

    # 打印引擎构建摘要信息
    print("\n" + "=" * 60)
    print("Engine Build Summary")
    print("=" * 60)
    print(f"  ONNX path: {onnx_path}")          # ONNX 模型路径
    print(f"  Output path: {output_path}")    # 引擎输出路径
    print(f"  FP16 enabled: {use_fp16}")      # 是否启用 FP16
    print(f"  Workspace size: {workspace_size / (1 << 30):.1f} GB")  # 工作空间大小（GB）
    print(f"  Batch size: {batch_size}")      # 批次大小
    print(f"  Number of layers: {network.num_layers}")      # 网络层数
    print(f"  Number of inputs: {network.num_inputs}")      # 输入数量
    print(f"  Number of outputs: {network.num_outputs}")    # 输出数量

    # 打印所有输入层的详细信息
    for i in range(network.num_inputs):
        input_layer = network.get_input(i)
        print(f"  Input {i}: {input_layer.name} - {input_layer.shape}")

    # 打印所有输出层的详细信息
    for i in range(network.num_outputs):
        output_layer = network.get_output(i)
        print(f"  Output {i}: {output_layer.name} - {output_layer.shape}")

    print("\n" + "=" * 60)
    print("TensorRT engine build completed!")
    print("=" * 60)


def main():
    """
    主函数：解析命令行参数并调用引擎编译函数
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX model")
    
    # 添加 --onnx 参数：ONNX 模型路径（必需）
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Path to the ONNX model"
    )
    
    # 添加 --output 参数：TensorRT 引擎输出路径（可选，默认 model.engine）
    parser.add_argument(
        "--output",
        type=str,
        default="model.engine",
        help="Output path for the TensorRT engine"
    )
    
    # 添加 --fp16 参数：启用 FP16 精度（可选，默认启用）
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 precision (default: True)"
    )
    
    # 添加 --fp32 参数：使用 FP32 精度（可选，覆盖 --fp16）
    parser.add_argument(
        "--fp32",
        action="store_true",
        default=False,
        help="Use FP32 precision (overrides --fp16)"
    )
    
    # 添加 --workspace 参数：工作空间大小（GB）（可选，默认1）
    parser.add_argument(
        "--workspace",
        type=int,
        default=1,
        help="Workspace size in GB (default: 1)"
    )
    
    # 添加 --batch-size 参数：批次大小（可选，默认1）
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 确定是否使用 FP16：启用 --fp16 且未启用 --fp32
    use_fp16 = args.fp16 and not args.fp32
    # 将工作空间大小从 GB 转换为字节（1 << 30 = 1GB）
    workspace_size = args.workspace << 30

    # 获取输出目录路径
    output_dir = os.path.dirname(args.output)
    # 如果输出目录不存在，创建目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 调用核心编译函数
    build_engine(args.onnx, args.output, use_fp16, workspace_size, args.batch_size)


# 当脚本直接运行时执行 main 函数
if __name__ == "__main__":
    main()
