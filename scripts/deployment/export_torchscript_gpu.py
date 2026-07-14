"""
SparseDriveV2 模型 GPU 导出脚本

功能：在 Orin 控制器上使用 GPU 重新导出模型，确保模型可以在 GPU 上运行

使用方法：
    # 在 Orin 上执行
    python export_torchscript_gpu.py --ckpt <checkpoint_path> --output <output_path>

关键要点：
    1. 必须在支持 CUDA 的设备上运行
    2. 使用纯 PyTorch 实现（use_deformable_func=False）
    3. 导出时使用 GPU 作为设备，确保所有操作在 GPU 上可执行
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel


class SparseDriveScriptModule(nn.Module):
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
    print(f"Loading checkpoint from: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found - {ckpt_path}")
        sys.exit(1)
    
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))["state_dict"]
    state_dict = {k.replace("agent.", "").replace("_sparsedrive_model.", ""): v for k, v in state_dict.items()}
    
    config = SparseDriveConfig()
    config.use_deformable_func = False
    
    model = SparseDriveModel(config)
    model.load_state_dict(state_dict)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, using CPU")
    
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    script_model = SparseDriveScriptModule(model)
    
    B = 1
    num_cams = 3
    H, W = 256, 512
    
    example_imgs = torch.randn(B, num_cams, 3, H, W).to(device)
    example_status = torch.randn(B, 8).to(device)
    example_lidar2img = torch.randn(B, num_cams, 4, 4).to(device)
    example_lidar2cam = torch.randn(B, num_cams, 4, 4).to(device)
    example_cam2lidar = torch.randn(B, num_cams, 4, 4).to(device)
    example_cam_intrinsic = torch.randn(B, num_cams, 3, 3).to(device)
    
    print("Running forward pass for trace...")
    with torch.no_grad():
        script_model(example_imgs, example_status, example_lidar2img, example_lidar2cam, example_cam2lidar, example_cam_intrinsic)
    
    print("Exporting to TorchScript with GPU...")
    traced_model = torch.jit.trace(
        script_model, 
        (example_imgs, example_status, example_lidar2img, example_lidar2cam, example_cam2lidar, example_cam_intrinsic),
        strict=False,
        check_trace=False
    )
    
    traced_model.save(output_path)
    print(f"\nModel exported to: {output_path}")
    
    print("\nVerifying exported model...")
    loaded_model = torch.jit.load(output_path)
    loaded_model = loaded_model.to(device)
    
    with torch.no_grad():
        output = loaded_model(example_imgs, example_status, example_lidar2img, example_lidar2cam, example_cam2lidar, example_cam_intrinsic)
    
    print(f"✓ Verification successful!")
    print(f"  Device: {device}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output example:\n{output[0, :3, :]}")


def main():
    parser = argparse.ArgumentParser(description="Export SparseDriveV2 model to TorchScript with GPU support")
    parser.add_argument(
        "--ckpt", 
        type=str, 
        required=True,
        help="Path to the trained checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="model_scripted_gpu.pt",
        help="Output path for the TorchScript model"
    )
    
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    export_model(args.ckpt, args.output)


if __name__ == "__main__":
    main()