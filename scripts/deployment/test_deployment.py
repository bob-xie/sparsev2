"""
SparseDriveV2 部署测试脚本

功能：
    1. 验证导出的 TorchScript 模型可以正常加载和推理
    2. 测试不同设备（CPU/GPU）上的推理
    3. 测量推理时间
    4. 输出模型信息

使用方法：
    python test_deployment.py --model <model_path>
"""

import argparse
import os
import sys
import time
import torch


def test_model(model_path: str):
    """测试部署模型"""
    print("=" * 60)
    print("SparseDriveV2 部署测试")
    print("=" * 60)
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    start_time = time.time()
    model = torch.jit.load(model_path)
    load_time = time.time() - start_time
    print(f"模型加载时间: {load_time:.2f}秒")
    
    # 打印模型信息
    print("\n模型信息:")
    print(model)
    
    # 准备测试输入
    B = 1
    num_cams = 3
    H, W = 256, 512
    
    print(f"\n测试输入配置:")
    print(f"  Batch size: {B}")
    print(f"  相机数量: {num_cams}")
    print(f"  图像尺寸: {H} x {W}")
    
    # 测试 CPU 推理
    print("\n" + "=" * 60)
    print("CPU 推理测试")
    print("=" * 60)
    
    model_cpu = model.to(torch.device("cpu"))
    model_cpu.eval()
    
    imgs_cpu = torch.randn(B, num_cams, 3, H, W)
    status_cpu = torch.randn(B, 8)
    lidar2img_cpu = torch.randn(B, num_cams, 4, 4)
    lidar2cam_cpu = torch.randn(B, num_cams, 4, 4)
    cam2lidar_cpu = torch.randn(B, num_cams, 4, 4)
    cam_intrinsic_cpu = torch.randn(B, num_cams, 3, 3)
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            model_cpu(imgs_cpu, status_cpu, lidar2img_cpu, lidar2cam_cpu, cam2lidar_cpu, cam_intrinsic_cpu)
    
    # 测量推理时间
    times = []
    with torch.no_grad():
        for i in range(10):
            t_start = time.time()
            output = model_cpu(imgs_cpu, status_cpu, lidar2img_cpu, lidar2cam_cpu, cam2lidar_cpu, cam_intrinsic_cpu)
            t_end = time.time()
            times.append(t_end - t_start)
    
    print(f"CPU 推理时间 (平均): {sum(times)/len(times):.4f}秒")
    print(f"CPU 推理时间 (最快): {min(times):.4f}秒")
    print(f"CPU 推理时间 (最慢): {max(times):.4f}秒")
    print(f"CPU FPS: {1/(sum(times)/len(times)):.2f}")
    print(f"输出形状: {output.shape}")
    print(f"输出示例:\n{output[0]}")
    
    # 测试 GPU 推理（如果可用）
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU 推理测试")
        print("=" * 60)
        
        model_gpu = model.to(torch.device("cuda"))
        model_gpu.eval()
        
        imgs_gpu = imgs_cpu.cuda()
        status_gpu = status_cpu.cuda()
        lidar2img_gpu = lidar2img_cpu.cuda()
        lidar2cam_gpu = lidar2cam_cpu.cuda()
        cam2lidar_gpu = cam2lidar_cpu.cuda()
        cam_intrinsic_gpu = cam_intrinsic_cpu.cuda()
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                model_gpu(imgs_gpu, status_gpu, lidar2img_gpu, lidar2cam_gpu, cam2lidar_gpu, cam_intrinsic_gpu)
        
        # 测量推理时间
        times = []
        with torch.no_grad():
            for i in range(10):
                t_start = time.time()
                output_gpu = model_gpu(imgs_gpu, status_gpu, lidar2img_gpu, lidar2cam_gpu, cam2lidar_gpu, cam_intrinsic_gpu)
                torch.cuda.synchronize()
                t_end = time.time()
                times.append(t_end - t_start)
        
        print(f"GPU 推理时间 (平均): {sum(times)/len(times):.4f}秒")
        print(f"GPU 推理时间 (最快): {min(times):.4f}秒")
        print(f"GPU 推理时间 (最慢): {max(times):.4f}秒")
        print(f"GPU FPS: {1/(sum(times)/len(times)):.2f}")
        print(f"输出形状: {output_gpu.shape}")
        
        # 对比 CPU 和 GPU 输出
        diff = torch.abs(output.cpu() - output_gpu.cpu())
        print(f"\nCPU/GPU 输出差异 (最大): {diff.max().item()}")
        print(f"CPU/GPU 输出差异 (平均): {diff.mean().item()}")
        
        if diff.max().item() < 1e-5:
            print("✓ CPU/GPU 输出一致")
        else:
            print("✗ CPU/GPU 输出不一致")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test SparseDriveV2 deployment")
    parser.add_argument(
        "--model",
        type=str,
        default="exp/deployment/model_scripted.pt",
        help="Path to the exported TorchScript model"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在 - {args.model}")
        sys.exit(1)
    
    success = test_model(args.model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()