"""
SparseDriveV2 TensorRT 导出脚本

功能：将 TorchScript 模型编译为 TensorRT 优化模型

使用方法：
    python export_tensorrt.py --model model_scripted_cpu.pt --output model_trt.ts

关键要点：
    1. 必须在目标 GPU 上编译（不同架构不兼容）
    2. 使用 FP16 精度获得最佳性能
    3. 需要安装 torch_tensorrt
"""

import argparse
import os
import sys
import time
import torch

try:
    import torch_tensorrt
    print("✓ torch_tensorrt 已安装")
except ImportError:
    print("✗ torch_tensorrt 未安装，请先安装：")
    print("  pip install torch_tensorrt --index-url https://download.pytorch.org/whl/cu126")
    sys.exit(1)


def export_tensorrt(model_path: str, output_path: str):
    """导出 TensorRT 优化模型"""
    print("=" * 60)
    print("SparseDriveV2 TensorRT 导出")
    print("=" * 60)
    
    print(f"\n加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        return False
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用，TensorRT 需要 GPU")
        return False
    
    print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
    
    # 加载 TorchScript 模型
    start_time = time.time()
    model = torch.jit.load(model_path)
    model = model.cuda()
    model.eval()
    load_time = time.time() - start_time
    print(f"模型加载时间: {load_time:.2f}秒")
    
    # 准备示例输入
    B = 1
    num_cams = 3
    H, W = 256, 512
    
    example_imgs = torch.randn(B, num_cams, 3, H, W).cuda()
    example_status = torch.randn(B, 8).cuda()
    example_lidar2img = torch.randn(B, num_cams, 4, 4).cuda()
    example_lidar2cam = torch.randn(B, num_cams, 4, 4).cuda()
    example_cam2lidar = torch.randn(B, num_cams, 4, 4).cuda()
    example_cam_intrinsic = torch.randn(B, num_cams, 3, 3).cuda()
    
    # 先运行一次预热
    print("\n预热模型...")
    with torch.no_grad():
        model(example_imgs, example_status, example_lidar2img, example_lidar2cam, example_cam2lidar, example_cam_intrinsic)
    
    # TensorRT 编译
    print("\n开始 TensorRT 编译...")
    print("精度: FP16")
    print("工作空间: 1GB")
    
    start_time = time.time()
    try:
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[
                torch_tensorrt.Input(
                    shape=[B, num_cams, 3, H, W],
                    dtype=torch.float32,
                    name="imgs"
                ),
                torch_tensorrt.Input(
                    shape=[B, 8],
                    dtype=torch.float32,
                    name="status"
                ),
                torch_tensorrt.Input(
                    shape=[B, num_cams, 4, 4],
                    dtype=torch.float32,
                    name="lidar2img"
                ),
                torch_tensorrt.Input(
                    shape=[B, num_cams, 4, 4],
                    dtype=torch.float32,
                    name="lidar2cam"
                ),
                torch_tensorrt.Input(
                    shape=[B, num_cams, 4, 4],
                    dtype=torch.float32,
                    name="cam2lidar"
                ),
                torch_tensorrt.Input(
                    shape=[B, num_cams, 3, 3],
                    dtype=torch.float32,
                    name="cam_intrinsic"
                ),
            ],
            enabled_precisions={torch.float16},
            workspace_size=1 << 30,
            truncate_long_and_double=True,
        )
        
        compile_time = time.time() - start_time
        print(f"✓ TensorRT 编译成功! 耗时: {compile_time:.2f}秒")
    except Exception as e:
        print(f"✗ TensorRT 编译失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 保存模型
    print(f"\n保存模型: {output_path}")
    torch.jit.save(trt_model, output_path)
    print("✓ 模型保存成功!")
    
    # 验证模型
    print("\n验证模型...")
    loaded_model = torch.jit.load(output_path)
    
    with torch.no_grad():
        output = loaded_model(example_imgs, example_status, example_lidar2img, example_lidar2cam, example_cam2lidar, example_cam_intrinsic)
    
    print(f"✓ 验证成功!")
    print(f"  输出形状: {output.shape}")
    
    # 性能测试
    print("\n性能测试...")
    times = []
    with torch.no_grad():
        for _ in range(3):
            loaded_model(example_imgs, example_status, example_lidar2img, example_lidar2cam, example_cam2lidar, example_cam_intrinsic)
        
        for i in range(10):
            t_start = time.time()
            loaded_model(example_imgs, example_status, example_lidar2img, example_lidar2cam, example_cam2lidar, example_cam_intrinsic)
            torch.cuda.synchronize()
            t_end = time.time()
            times.append(t_end - t_start)
    
    avg_time = sum(times) / len(times)
    print(f"平均推理时间: {avg_time * 1000:.2f}ms")
    print(f"FPS: {1 / avg_time:.2f}")
    
    print("\n" + "=" * 60)
    print("导出完成!")
    print("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Export SparseDriveV2 model to TensorRT")
    parser.add_argument(
        "--model",
        type=str,
        default="model_scripted_cpu.pt",
        help="Path to the TorchScript model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_trt.ts",
        help="Output path for the TensorRT model"
    )
    
    args = parser.parse_args()
    
    success = export_tensorrt(args.model, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()