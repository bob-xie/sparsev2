"""
Orin 专用 GPU 导出脚本 - 绕过 nuplan 依赖

功能：在 Orin 控制器上使用 GPU 导出模型，确保模型可以在 Orin 的 sm_87 GPU 上运行

使用方法：
    # 在 Orin 上执行
    python3 export_torchscript_gpu_orin.py --ckpt <checkpoint_path> --output <output_path>

关键要点：
    1. 必须在 Orin 上执行（GPU 架构 sm_87）
    2. 使用纯 PyTorch 实现（use_deformable_func=False）
    3. 简化配置类，绕过 nuplan 依赖
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class SparseDriveConfig:
    """简化版配置类，绕过 nuplan 依赖"""
    
    def __init__(self):
        # ================ model ================ #
        # basic setting
        self.d_model: int = 256
        self.d_ffn: int = 1024
        self.num_head: int = 8
        self.dropout: float = 0.0

        # vision backbone & neck
        self.image_architecture: str = "resnet34"
        self.bkb_path: str = "ckpt/resnet34.bin"    
        self.use_grid_mask: bool = True
        self.with_img_neck: bool = True
        self.num_levels: int = 4

        # vocabulary
        self.path_anchor: str = "ckpt/kmeans/path_1024.npy"
        self.velocity_anchor: str = "ckpt/kmeans/velocity_256.npy"
        self.trajectory_anchor: str = "ckpt/kmeans/trajectory_1024_256.npz"

        self.mode_path: int = 1024
        self.len_path: int = 50
        self.path_interval: float = 1.0
        self.mode_vel: int = 256
        self.len_vel_seq: int = 8
        self.vel_time_interval: float = 0.5

        # decoder
        self.decoder_num_layers: int = 2

        self.path_filter_num = (128, 20)
        self.velocity_filter_num = (64, 10)

        self.path_sigmas: float = 4.0
        self.velocity_sigmas: float = 4.0
        self.trajectory_sigmas: float = 4.0

        # deformable
        self.fix_height = (0., -0.25, -0.5, 0.25, 0.5)
        self.num_learnable_pts: int = 2
        self.use_deformable_func: bool = False

        # metric supervision
        self.dataset_version: str = "v2"
        self.metrics = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance",
                       "time_to_collision_within_bound", "ego_progress", "lane_keeping", "history_comfort")
        self.metric_loss_weight: float = 5.0

        # ================ data process ================ #
        self.cams = ("cam_l0", "cam_f0", "cam_r0")
        self.resize_lim = (512/1920, 512/1920)
        self.final_dim = (256, 512)
        self.bot_pct_lim = (0.0, 0.0)
        self.rot_lim = (-0., 0.)
        self.H: int = 1080
        self.W: int = 1920
        self.rand_flip: bool = False
        self.rot3d_range = (-0., 0.)
        self.photo_metric_distortion: bool = True
        self.img_mean = (123.675, 116.28, 103.53)
        self.img_std = (58.395, 57.12, 57.375)
        self.to_bgr: bool = False


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
    print("=" * 60)
    print("SparseDriveV2 GPU Export for Orin")
    print("=" * 60)
    print(f"\nLoading checkpoint from: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found - {ckpt_path}")
        sys.exit(1)
    
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))["state_dict"]
    state_dict = {k.replace("agent.", "").replace("_sparsedrive_model.", ""): v for k, v in state_dict.items()}
    
    print("\nCreating model with simplified config (bypassing nuplan)...")
    config = SparseDriveConfig()
    config.use_deformable_func = False
    
    model = SparseDriveModel(config)
    model.load_state_dict(state_dict)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA device (GPU architecture: sm_87)")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, using CPU")
    
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully!")
    
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
    
    print("\nRunning forward pass for trace...")
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
    print(f"\n✓ Model exported to: {output_path}")
    
    print("\nVerifying exported model...")
    loaded_model = torch.jit.load(output_path)
    loaded_model = loaded_model.to(device)
    
    with torch.no_grad():
        output = loaded_model(example_imgs, example_status, example_lidar2img, example_lidar2cam, example_cam2lidar, example_cam_intrinsic)
    
    print(f"✓ Verification successful!")
    print(f"  Device: {device}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output example:\n{output[0, :3, :]}")
    
    print("\n" + "=" * 60)
    print("Export completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Export SparseDriveV2 model to TorchScript on Orin")
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