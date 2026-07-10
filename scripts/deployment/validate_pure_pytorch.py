"""
验证纯 PyTorch 实现与 CUDA 实现的一致性

功能：
    1. 使用相同的输入分别运行 CUDA 版本和纯 PyTorch 版本
    2. 对比输出结果，计算差异
    3. 验证纯 PyTorch 实现的正确性
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel


def validate_models():
    """验证两种实现的一致性"""
    print("=" * 60)
    print("验证纯 PyTorch 实现与 CUDA 实现的一致性")
    print("=" * 60)
    
    # 加载权重
    ckpt_path = "exp/sparsedrive_agent/2026.06.17.17.46.52/periodic_pdm_ckpts/ep0010.ckpt"
    print(f"\n加载权重: {ckpt_path}")
    
    if torch.cuda.is_available():
        state_dict = torch.load(ckpt_path)["state_dict"]
    else:
        state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))["state_dict"]
    
    state_dict = {k.replace("agent.", "").replace("_sparsedrive_model.", ""): v for k, v in state_dict.items()}
    
    # 创建两种配置
    config_cuda = SparseDriveConfig()
    config_cuda.use_deformable_func = True  # CUDA 版本
    
    config_pure = SparseDriveConfig()
    config_pure.use_deformable_func = False  # 纯 PyTorch 版本
    
    # 创建两种模型
    print("\n创建模型...")
    model_cuda = SparseDriveModel(config_cuda)
    model_pure = SparseDriveModel(config_pure)
    
    # 准备测试输入
    B = 1
    num_cams = 3
    H, W = 256, 512
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    imgs = torch.randn(B, num_cams, 3, H, W)
    status = torch.randn(B, 8)
    lidar2img = torch.randn(B, num_cams, 4, 4)
    lidar2cam = torch.randn(B, num_cams, 4, 4)
    cam2lidar = torch.randn(B, num_cams, 4, 4)
    cam_intrinsic = torch.randn(B, num_cams, 3, 3)
    
    # 构建特征字典（包含训练时 data_adapter 生成的所有字段）
    features = {
        "camera_feature": {
            "imgs": imgs,
            "lidar2img": lidar2img,
            "lidar2cam": lidar2cam,
            "cam2lidar": cam2lidar,
            "cam_intrinsic": cam_intrinsic,
            "projection_mat": lidar2img[:, :, :3].float(),
            "image_wh": torch.tensor([[W, H]] * num_cams).repeat(B, 1, 1).float(),
        },
        "status_feature": status,
    }
    
    if torch.cuda.is_available():
        print("\n使用 GPU 进行验证...")
        model_cuda = model_cuda.cuda()
        model_pure = model_pure.cuda()
        
        # 将词汇表参数移动到 CUDA
        model_cuda._trajectory_head.path_vocab = nn.Parameter(model_cuda._trajectory_head.path_vocab.data.cuda(), requires_grad=False)
        model_cuda._trajectory_head.vel_vocab = nn.Parameter(model_cuda._trajectory_head.vel_vocab.data.cuda(), requires_grad=False)
        model_cuda._trajectory_head.traj_vocab = nn.Parameter(model_cuda._trajectory_head.traj_vocab.data.cuda(), requires_grad=False)
        model_cuda._trajectory_head.traj_mask = nn.Parameter(model_cuda._trajectory_head.traj_mask.data.cuda(), requires_grad=False)
        
        model_pure._trajectory_head.path_vocab = nn.Parameter(model_pure._trajectory_head.path_vocab.data.cuda(), requires_grad=False)
        model_pure._trajectory_head.vel_vocab = nn.Parameter(model_pure._trajectory_head.vel_vocab.data.cuda(), requires_grad=False)
        model_pure._trajectory_head.traj_vocab = nn.Parameter(model_pure._trajectory_head.traj_vocab.data.cuda(), requires_grad=False)
        model_pure._trajectory_head.traj_mask = nn.Parameter(model_pure._trajectory_head.traj_mask.data.cuda(), requires_grad=False)
        
        # 加载权重
        state_dict_cuda = {k: v.cuda() for k, v in state_dict.items()}
        model_cuda.load_state_dict(state_dict_cuda)
        model_pure.load_state_dict(state_dict_cuda)
        
        # 移动输入到 CUDA
        for k, v in features["camera_feature"].items():
            features["camera_feature"][k] = v.cuda()
        features["status_feature"] = features["status_feature"].cuda()
    else:
        print("\n使用 CPU 进行验证...")
        model_cuda.load_state_dict(state_dict)
        model_pure.load_state_dict(state_dict)
    
    # 设置为评估模式
    model_cuda.eval()
    model_pure.eval()
    
    # 运行 CUDA 版本
    print("\n运行 CUDA 版本...")
    with torch.no_grad():
        try:
            output_cuda, _ = model_cuda(features, None)
            traj_cuda = output_cuda["trajectory"]
            print("  ✓ CUDA 版本运行成功")
        except Exception as e:
            print(f"  ✗ CUDA 版本运行失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 运行纯 PyTorch 版本
    print("\n运行纯 PyTorch 版本...")
    with torch.no_grad():
        try:
            output_pure, _ = model_pure(features, None)
            traj_pure = output_pure["trajectory"]
            print("  ✓ 纯 PyTorch 版本运行成功")
        except Exception as e:
            print(f"  ✗ 纯 PyTorch 版本运行失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    
    print(f"\nCUDA 版本输出形状: {traj_cuda.shape}")
    print(f"纯 PyTorch 版本输出形状: {traj_pure.shape}")
    
    print(f"\nCUDA 版本输出:\n{traj_cuda}")
    print(f"\n纯 PyTorch 版本输出:\n{traj_pure}")
    
    # 计算差异
    diff = torch.abs(traj_cuda - traj_pure)
    print(f"\n绝对差异:\n{diff}")
    
    print(f"\n最大差异: {diff.max().item()}")
    print(f"平均差异: {diff.mean().item()}")
    print(f"均方误差: {torch.mean(diff ** 2).item()}")
    
    # 判断是否一致
    tolerance = 1e-4
    if diff.max().item() < tolerance:
        print(f"\n✓ 通过验证！两种实现的输出在容忍度 {tolerance} 内一致")
        return True
    else:
        print(f"\n✗ 验证失败！最大差异 {diff.max().item()} 超过容忍度 {tolerance}")
        return False


if __name__ == "__main__":
    success = validate_models()
    sys.exit(0 if success else 1)