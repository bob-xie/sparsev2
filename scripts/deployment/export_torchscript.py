"""
SparseDriveV2 模型导出脚本

功能：将训练好的模型导出为 TorchScript 格式，支持脱离 Python 运行时部署

使用方法：
    python export_torchscript.py --ckpt <checkpoint_path> --output <output_path>

关键要点：
    1. 使用纯 PyTorch 实现（use_deformable_func=False）
    2. 移除自定义 CUDA 扩展依赖
    3. 导出完整的推理接口
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel


class SparseDriveScriptModule(nn.Module):
    """
    用于 TorchScript 导出的包装模块
    
    将复杂的字典输入转换为简单的张量输入，方便部署
    """
    
    def __init__(self, model: SparseDriveModel):
        super().__init__()
        self.model = model
        self.model.eval()
    
    def forward(self, 
                imgs: torch.Tensor, 
                status_feature: torch.Tensor,
                lidar2img: torch.Tensor,
                lidar2cam: torch.Tensor,
                cam2lidar: torch.Tensor,
                cam_intrinsic: torch.Tensor) -> torch.Tensor:
        """
        推理前向传播
        
        参数说明：
            imgs: 多视角相机图像 [B, num_cams, 3, H, W]
                  B=batch_size, num_cams=3(cam_l0, cam_f0, cam_r0), 
                  3=RGB通道, H=256, W=512
            status_feature: ego状态特征 [B, 8]
                           8 = 4(驾驶命令) + 2(速度) + 2(加速度)
            lidar2img: 投影矩阵 [B, num_cams, 4, 4]
            lidar2cam: 相机外参 [B, num_cams, 4, 4]
            cam2lidar: 相机到lidar变换矩阵 [B, num_cams, 4, 4]
            cam_intrinsic: 相机内参 [B, num_cams, 3, 3]
        
        返回值：
            trajectory: 预测轨迹 [B, num_poses, 3]
                        num_poses=8, 3=(x, y, heading)
        """
        B, num_cams = imgs.shape[:2]
        _, H, W = imgs.shape[-3:]
            
        features = {
            "camera_feature": {
                "imgs": imgs,
                "lidar2img": lidar2img,
                "lidar2cam": lidar2cam,
                "cam2lidar": cam2lidar,
                "cam_intrinsic": cam_intrinsic,
                "projection_mat": lidar2img[:, :, :3].float(),
                "image_wh": torch.tensor([[W, H]] * num_cams).repeat(B, 1, 1).float().to(imgs.device),
            },
            "status_feature": status_feature,
        }
        
        output, _ = self.model(features, None)
        return output["trajectory"]


def export_model(ckpt_path: str, output_path: str):
    """
    导出模型为 TorchScript 格式
    
    参数：
        ckpt_path: 训练好的模型权重路径 (.ckpt)
        output_path: 导出模型的保存路径 (.pt)
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # 加载权重
    if torch.cuda.is_available():
        state_dict = torch.load(ckpt_path)["state_dict"]
    else:
        state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))["state_dict"]
    
    # 移除"agent."前缀和"_sparsedrive_model."前缀
    state_dict = {k.replace("agent.", "").replace("_sparsedrive_model.", ""): v for k, v in state_dict.items()}
    
    # 创建配置对象，使用纯 PyTorch 实现（禁用自定义 CUDA 扩展）
    config = SparseDriveConfig()
    config.use_deformable_func = False
    
    # 创建纯 PyTorch 实现的模型
    print("Creating model with pure PyTorch implementation...")
    model = SparseDriveModel(config)
    
    # 加载权重
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Model loaded successfully!")
    
    # 创建导出包装模块
    script_model = SparseDriveScriptModule(model)
    
    # 准备示例输入（根据实际输入格式）
    B = 1
    num_cams = 3
    H, W = 256, 512
    
    example_imgs = torch.randn(B, num_cams, 3, H, W)
    example_status = torch.randn(B, 8)
    example_lidar2img = torch.randn(B, num_cams, 4, 4)
    example_lidar2cam = torch.randn(B, num_cams, 4, 4)
    example_cam2lidar = torch.randn(B, num_cams, 4, 4)
    example_cam_intrinsic = torch.randn(B, num_cams, 3, 3)
    
    if torch.cuda.is_available():
        script_model = script_model.cuda()
        example_imgs = example_imgs.cuda()
        example_status = example_status.cuda()
        example_lidar2img = example_lidar2img.cuda()
        example_lidar2cam = example_lidar2cam.cuda()
        example_cam2lidar = example_cam2lidar.cuda()
        example_cam_intrinsic = example_cam_intrinsic.cuda()
    
    # 导出为 TorchScript
    print("Exporting to TorchScript...")
    try:
        # 先运行一次前向传播确保模型已初始化
        with torch.no_grad():
            script_model(example_imgs, example_status, example_lidar2img, example_lidar2cam, example_cam2lidar, example_cam_intrinsic)
        
        # 使用 trace 方式（模型包含复杂控制流，trace更稳定）
        traced_model = torch.jit.trace(
            script_model, 
            (example_imgs, example_status, example_lidar2img, example_lidar2cam, example_cam2lidar, example_cam_intrinsic),
            strict=False,
            check_trace=False
        )
        print("✓ 使用 torch.jit.trace 成功")
    except Exception as e:
        print(f"✗ trace 失败: {e}")
        return
    
    # 保存模型
    traced_model.save(output_path)
    print(f"\n✓ 模型已导出到: {output_path}")
    
    # 验证导出的模型
    print("\n验证导出的模型...")
    loaded_model = torch.jit.load(output_path)
    
    if torch.cuda.is_available():
        loaded_model = loaded_model.cuda()
    
    with torch.no_grad():
        output = loaded_model(example_imgs, example_status, example_lidar2img, example_lidar2cam, example_cam2lidar, example_cam_intrinsic)
    
    print(f"✓ 推理成功!")
    print(f"  输入形状:")
    print(f"    imgs: {example_imgs.shape}")
    print(f"    status_feature: {example_status.shape}")
    print(f"    lidar2img: {example_lidar2img.shape}")
    print(f"    lidar2cam: {example_lidar2cam.shape}")
    print(f"    cam2lidar: {example_cam2lidar.shape}")
    print(f"    cam_intrinsic: {example_cam_intrinsic.shape}")
    print(f"  输出形状: trajectory {output.shape}")
    print(f"  输出示例: {output[0, :3, :]}")


def main():
    parser = argparse.ArgumentParser(description="Export SparseDriveV2 model to TorchScript")
    parser.add_argument(
        "--ckpt", 
        type=str, 
        default="exp/sparsedrive_agent/2026.06.17.17.46.52/periodic_pdm_ckpts/ep0010.ckpt",
        help="Path to the trained checkpoint"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="exp/deployment/model_scripted.pt",
        help="Output path for the TorchScript model"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    export_model(args.ckpt, args.output)


if __name__ == "__main__":
    main()