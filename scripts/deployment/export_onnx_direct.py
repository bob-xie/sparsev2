"""
直接从 .ckpt 导出 ONNX 格式（跳过 TorchScript）

功能：将训练好的模型直接导出为 ONNX 格式，避免 TorchScript 的 ONNX 兼容性问题

使用方法：
    python export_onnx_direct.py --ckpt ep0010.ckpt --output model.onnx

输入格式：
    imgs: (B, num_cams, 3, H, W)  # B=1, num_cams=3, H=256, W=512
    status_feature: (B, 8)
    lidar2img: (B, num_cams, 4, 4)
    lidar2cam: (B, num_cams, 4, 4)
    cam2lidar: (B, num_cams, 4, 4)
    cam_intrinsic: (B, num_cams, 3, 3)

输出格式：
    trajectory: (B, 9, 3)  # 9个轨迹点 (x, y, heading)
"""

# 导入命令行参数解析模块，用于处理脚本输入参数
import argparse
# 导入操作系统接口模块，用于文件路径处理
import os
# 导入系统模块，用于程序退出等操作
import sys

# 导入 PyTorch 核心库，用于模型加载和 ONNX 导出
import torch
# 导入 PyTorch 神经网络模块，用于定义模型类
import torch.nn as nn


# 将项目根目录添加到 Python 路径，使其能够导入 navsim 模块
# os.path.abspath(__file__) 获取当前脚本的绝对路径
# os.path.dirname() 逐级向上获取父目录，共三次到达项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# 轨迹采样配置类，用于配置轨迹预测的关键参数
class TrajectorySamplingConfig:
    def __init__(self):
        # num_poses: int - 轨迹点数量，默认8个（对应4秒预测，每0.5秒一个点）
        self.num_poses: int = 8


# SparseDriveV2 模型配置类，定义模型的所有超参数
class SparseDriveConfig:
    def __init__(self):
        # d_model: int - Transformer 的模型维度（隐藏层大小），默认256
        self.d_model: int = 256
        # d_ffn: int - 前馈网络的隐藏层维度，默认1024
        self.d_ffn: int = 1024
        # num_head: int - MultiheadAttention 的注意力头数，默认8
        self.num_head: int = 8
        # dropout: float - Dropout 比率，默认0.0（推理时禁用）
        self.dropout: float = 0.0
        # image_architecture: str - 图像骨干网络类型，默认使用 resnet34
        self.image_architecture: str = "resnet34"
        # bkb_path: str - 预训练骨干网络权重路径
        self.bkb_path: str = "ckpt/resnet34.bin"
        # use_grid_mask: bool - 是否使用网格掩码增强，默认True
        self.use_grid_mask: bool = True
        # with_img_neck: bool - 是否使用图像颈部网络（FPN），默认True
        self.with_img_neck: bool = True
        # num_levels: int - FPN 特征图层级数，默认4层
        self.num_levels: int = 4
        # path_anchor: str - 路径词汇表（K-Means聚类结果）路径
        self.path_anchor: str = "ckpt/kmeans/path_1024.npy"
        # velocity_anchor: str - 速度词汇表路径
        self.velocity_anchor: str = "ckpt/kmeans/velocity_256.npy"
        # trajectory_anchor: str - 轨迹词汇表（组合路径和速度）路径
        self.trajectory_anchor: str = "ckpt/kmeans/trajectory_1024_256.npz"
        # mode_path: int - 路径词汇表大小，默认1024
        self.mode_path: int = 1024
        # len_path: int - 每条路径的点数，默认50
        self.len_path: int = 50
        # path_interval: float - 路径点间距（米），默认1.0米
        self.path_interval: float = 1.0
        # mode_vel: int - 速度词汇表大小，默认256
        self.mode_vel: int = 256
        # len_vel_seq: int - 速度序列长度，默认8
        self.len_vel_seq: int = 8
        # vel_time_interval: float - 速度采样时间间隔（秒），默认0.5秒
        self.vel_time_interval: float = 0.5
        # decoder_num_layers: int - Transformer Decoder 层数，默认2层
        self.decoder_num_layers: int = 2
        # path_filter_num: tuple - 解码器各层的路径过滤数量，(128, 20)表示第一层保留128条，第二层保留20条
        self.path_filter_num = (128, 20)
        # velocity_filter_num: tuple - 解码器各层的速度过滤数量，(64, 10)表示第一层保留64条，第二层保留10条
        self.velocity_filter_num = (64, 10)
        # path_sigmas: float - 路径评分的高斯核标准差，默认4.0
        self.path_sigmas: float = 4.0
        # velocity_sigmas: float - 速度评分的高斯核标准差，默认4.0
        self.velocity_sigmas: float = 4.0
        # trajectory_sigmas: float - 轨迹评分的高斯核标准差，默认4.0
        self.trajectory_sigmas: float = 4.0
        # fix_height: tuple - 可变形聚合的固定高度采样值，用于3D关键点生成
        self.fix_height = (0., -0.25, -0.5, 0.25, 0.5)
        # num_learnable_pts: int - 可学习关键点数量，默认2
        self.num_learnable_pts: int = 2
        # use_deformable_func: bool - 是否使用自定义 CUDA 可变形聚合函数，默认False（使用纯PyTorch实现）
        self.use_deformable_func: bool = False
        # dataset_version: str - 数据集版本，默认v2
        self.dataset_version: str = "v2"
        # metrics: tuple - 评估指标列表，用于训练时的指标损失计算
        self.metrics = ("no_at_fault_collisions", "drivable_area_compliance", 
                       "driving_direction_compliance", "traffic_light_compliance",
                       "time_to_collision_within_bound", "ego_progress", 
                       "lane_keeping", "history_comfort")
        # metric_loss_weight: float - 指标损失的权重系数，默认5.0
        self.metric_loss_weight: float = 5.0
        # cams: tuple - 使用的相机名称列表，(cam_l0左前, cam_f0前视, cam_r0右前)
        self.cams = ("cam_l0", "cam_f0", "cam_r0")
        # resize_lim: tuple - 图像缩放范围，用于数据增强
        self.resize_lim = (512/1920, 512/1920)
        # final_dim: tuple - 图像最终尺寸 (height, width)，默认(256, 512)
        self.final_dim = (256, 512)
        # bot_pct_lim: tuple - 底部裁剪百分比范围
        self.bot_pct_lim = (0.0, 0.0)
        # rot_lim: tuple - 旋转角度范围（弧度）
        self.rot_lim = (-0., 0.)
        # H: int - 原始图像高度，默认1080
        self.H: int = 1080
        # W: int - 原始图像宽度，默认1920
        self.W: int = 1920
        # rand_flip: bool - 是否随机水平翻转，默认False（推理时禁用）
        self.rand_flip: bool = False
        # rot3d_range: tuple - 3D旋转角度范围
        self.rot3d_range = (-0., 0.)
        # photo_metric_distortion: bool - 是否使用光度畸变增强，默认True
        self.photo_metric_distortion: bool = True
        # img_mean: tuple - 图像归一化均值（RGB通道），默认ImageNet均值
        self.img_mean = (123.675, 116.28, 103.53)
        # img_std: tuple - 图像归一化标准差（RGB通道），默认ImageNet标准差
        self.img_std = (58.395, 57.12, 57.375)
        # to_bgr: bool - 是否将RGB转换为BGR格式，默认False
        self.to_bgr: bool = False
        # trajectory_sampling: TrajectorySamplingConfig - 轨迹采样配置对象
        self.trajectory_sampling = TrajectorySamplingConfig()


# 从 navsim 模块导入 SparseDriveModel 类，这是核心模型定义
from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel


# ONNX 导出包装类，将原始模型的字典输入转换为简单的张量输入
class SparseDriveONNXModule(nn.Module):
    """
    ONNX 导出专用包装模块
    
    作用：将复杂的字典输入格式转换为扁平化的张量输入，便于 ONNX 导出和部署
    原始模型输入：字典形式（包含 camera_feature 和 status_feature）
    导出模型输入：6个独立张量
    """
    
    def __init__(self, model: SparseDriveModel):
        """
        初始化函数
        
        参数：
            model: SparseDriveModel - 已经加载权重的原始模型实例
        """
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        self.model = model  # 保存原始模型引用
        self.model.eval()   # 将模型设置为推理模式（关闭 dropout、batchnorm 训练行为）
    
    def forward(self, 
                imgs: torch.Tensor, 
                status_feature: torch.Tensor,
                lidar2img: torch.Tensor,
                lidar2cam: torch.Tensor,
                cam2lidar: torch.Tensor,
                cam_intrinsic: torch.Tensor) -> torch.Tensor:
        """
        ONNX 导出专用前向传播函数
        
        参数：
            imgs: torch.Tensor - 多视角相机图像，形状 [B, num_cams, 3, H, W]
                  B=batch_size, num_cams=3, 3=RGB通道, H=256, W=512
            status_feature: torch.Tensor - ego状态特征，形状 [B, 8]
                           8 = 4(驾驶命令one-hot) + 2(速度xy) + 2(加速度xy)
            lidar2img: torch.Tensor - LiDAR到图像的投影矩阵，形状 [B, num_cams, 4, 4]
            lidar2cam: torch.Tensor - LiDAR到相机的外参矩阵，形状 [B, num_cams, 4, 4]
            cam2lidar: torch.Tensor - 相机到LiDAR的变换矩阵，形状 [B, num_cams, 4, 4]
            cam_intrinsic: torch.Tensor - 相机内参矩阵，形状 [B, num_cams, 3, 3]
        
        返回：
            trajectory: torch.Tensor - 预测轨迹，形状 [B, num_poses, 3]
                        num_poses=8, 3=(x, y, heading)
        """
        # 从 imgs 张量形状中提取 batch_size 和相机数量
        B, num_cams = imgs.shape[:2]
        # 从 imgs 张量形状中提取图像高度和宽度（忽略通道维度3）
        _, H, W = imgs.shape[-3:]
            
        # 构造原始模型需要的特征字典格式
        features = {
            "camera_feature": {
                # 多视角图像张量
                "imgs": imgs,
                # LiDAR到图像投影矩阵
                "lidar2img": lidar2img,
                # LiDAR到相机外参矩阵
                "lidar2cam": lidar2cam,
                # 相机到LiDAR变换矩阵
                "cam2lidar": cam2lidar,
                # 相机内参矩阵
                "cam_intrinsic": cam_intrinsic,
                # 投影矩阵（取 lidar2img 的前3行，转为float类型）
                "projection_mat": lidar2img[:, :, :3].float(),
                # 图像宽高信息，形状 [B, num_cams, 2]，每个相机对应 [W, H]
                "image_wh": torch.tensor([[W, H]] * num_cams).repeat(B, 1, 1).float().to(imgs.device),
            },
            # ego状态特征
            "status_feature": status_feature,
        }
        
        # 调用原始模型的前向传播，第二个参数 targets=None 表示推理模式
        # output 是字典，包含 "trajectory" 键
        # loss_dict 在推理模式下为空
        output, _ = self.model(features, None)
        # 只返回轨迹预测结果，用于 ONNX 导出
        return output["trajectory"]


def export_onnx(ckpt_path: str, output_path: str, opset_version: int = 17):
    """
    核心导出函数：将 .ckpt 模型直接导出为 ONNX 格式
    
    参数：
        ckpt_path: str - 训练好的模型权重路径（.ckpt文件）
        output_path: str - ONNX 模型输出路径
        opset_version: int - ONNX 算子集版本，默认17（兼容性较好）
    """
    # 打印导出标题
    print("=" * 60)
    print("SparseDriveV2 ONNX Direct Export")
    print("=" * 60)
    print(f"\nLoading checkpoint from: {ckpt_path}")

    # 检查 checkpoint 文件是否存在
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found - {ckpt_path}")
        sys.exit(1)  # 文件不存在时退出程序

    # 加载 checkpoint 文件，map_location="cpu" 强制加载到 CPU 内存
    # 从 checkpoint 中提取 state_dict（模型权重字典）
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))["state_dict"]
    # 清理权重键名：移除 "agent." 和 "_sparsedrive_model." 前缀
    # 因为训练时权重保存在 LightningModule 内部，导出时需要匹配原始模型的键名
    state_dict = {k.replace("agent.", "").replace("_sparsedrive_model.", ""): v for k, v in state_dict.items()}

    print("\nCreating model with simplified config (bypassing nuplan)...")
    # 创建简化的配置对象（不依赖 nuplan 库）
    config = SparseDriveConfig()
    # 强制使用纯 PyTorch 实现，禁用 CUDA 扩展（便于跨平台部署）
    config.use_deformable_func = False
    
    # 创建模型实例
    model = SparseDriveModel(config)
    # 加载权重到模型
    model.load_state_dict(state_dict)
    # 将模型设置为推理模式
    model.eval()

    # 创建 ONNX 导出包装模块
    onnx_model = SparseDriveONNXModule(model)
    print("✓ Model loaded successfully!")

    # 定义示例输入的形状参数
    B = 1              # batch_size，推理时通常为1
    num_cams = 3       # 相机数量（左前、前视、右前）
    H, W = 256, 512    # 图像高度和宽度

    # 创建随机示例输入张量，用于 ONNX 导出时的 trace
    example_imgs = torch.randn(B, num_cams, 3, H, W)        # 多视角图像 [1, 3, 3, 256, 512]
    example_status = torch.randn(B, 8)                      # ego状态 [1, 8]
    example_lidar2img = torch.randn(B, num_cams, 4, 4)      # 投影矩阵 [1, 3, 4, 4]
    example_lidar2cam = torch.randn(B, num_cams, 4, 4)      # 外参矩阵 [1, 3, 4, 4]
    example_cam2lidar = torch.randn(B, num_cams, 4, 4)      # 变换矩阵 [1, 3, 4, 4]
    example_cam_intrinsic = torch.randn(B, num_cams, 3, 3)  # 内参矩阵 [1, 3, 3, 3]

    print("\nRunning forward pass for trace...")
    # 执行一次前向传播，确保模型所有层都已初始化（特别是懒加载层）
    with torch.no_grad():  # 禁用梯度计算，节省内存和时间
        onnx_model(example_imgs, example_status, example_lidar2img,
                   example_lidar2cam, example_cam2lidar, example_cam_intrinsic)
    print("✓ Forward pass completed!")

    print(f"\nExporting to ONNX (opset version: {opset_version})...")
    
    # 定义动态轴配置，允许 batch_size 维度可变
    dynamic_axes = {
        "imgs": {0: "batch_size"},           # imgs 的第0维是 batch_size
        "status_feature": {0: "batch_size"}, # status_feature 的第0维是 batch_size
        "lidar2img": {0: "batch_size"},      # lidar2img 的第0维是 batch_size
        "lidar2cam": {0: "batch_size"},      # lidar2cam 的第0维是 batch_size
        "cam2lidar": {0: "batch_size"},      # cam2lidar 的第0维是 batch_size
        "cam_intrinsic": {0: "batch_size"},  # cam_intrinsic 的第0维是 batch_size
        "trajectory": {0: "batch_size"},     # trajectory 的第0维是 batch_size
    }

    # 调用 torch.onnx.export 将模型导出为 ONNX 格式
    torch.onnx.export(
        onnx_model,  # 要导出的模型（包装后的模型）
        # 示例输入元组，用于 trace 模型的前向传播
        (example_imgs, example_status, example_lidar2img,
         example_lidar2cam, example_cam2lidar, example_cam_intrinsic),
        output_path,  # ONNX 文件输出路径
        opset_version=opset_version,  # ONNX 算子集版本
        # 输入张量的名称，用于后续 ONNX Runtime 或 TensorRT 加载时引用
        input_names=["imgs", "status_feature", "lidar2img",
                     "lidar2cam", "cam2lidar", "cam_intrinsic"],
        output_names=["trajectory"],  # 输出张量的名称
        dynamic_axes=dynamic_axes,  # 动态轴配置
        verbose=False,  # 是否打印详细导出信息
        do_constant_folding=True,  # 是否执行常量折叠优化
        export_params=True,  # 是否导出模型参数（权重）
        keep_initializers_as_inputs=False,  # 是否将初始化器作为输入（通常设为False）
    )

    print(f"\n✓ ONNX model exported to: {output_path}")

    print("\nVerifying ONNX model...")
    try:
        # 尝试导入 onnx 库进行模型验证
        import onnx
        # 尝试导入 onnxruntime 进行推理验证
        import onnxruntime as ort

        # 加载导出的 ONNX 模型
        onnx_model = onnx.load(output_path)
        # 检查 ONNX 模型的有效性（结构是否正确）
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid!")

        # 创建 ONNX Runtime 推理会话，使用 CPU 执行器
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

        # 构造 ONNX Runtime 输入字典，将 PyTorch 张量转换为 numpy 数组
        input_dict = {
            "imgs": example_imgs.numpy(),
            "status_feature": example_status.numpy(),
            "lidar2img": example_lidar2img.numpy(),
            "lidar2cam": example_lidar2cam.numpy(),
            "cam2lidar": example_cam2lidar.numpy(),
            "cam_intrinsic": example_cam_intrinsic.numpy(),
        }

        # 使用 ONNX Runtime 执行推理
        onnx_output = sess.run(["trajectory"], input_dict)[0]

        # 使用原始 PyTorch 模型执行推理（用于对比）
        with torch.no_grad():
            pt_output = onnx_model(example_imgs, example_status, example_lidar2img,
                                   example_lidar2cam, example_cam2lidar, example_cam_intrinsic).numpy()

        # 计算 ONNX 输出与 PyTorch 输出的最大差异
        max_diff = abs(onnx_output - pt_output).max()
        print(f"✓ ONNX vs PT output max difference: {max_diff:.6f}")

        # 判断输出一致性：差异小于 1e-3 认为通过
        if max_diff < 1e-3:
            print("✓ Output consistency check passed!")
        else:
            print(f"⚠ Output difference is larger than expected: {max_diff}")

        # 打印 ONNX 模型的输入输出信息
        print(f"\nONNX model info:")
        print(f"  Inputs: {[input.name for input in sess.get_inputs()]}")
        print(f"  Outputs: {[output.name for output in sess.get_outputs()]}")
        for input in sess.get_inputs():
            print(f"    {input.name}: {input.shape}")
        for output in sess.get_outputs():
            print(f"    {output.name}: {output.shape}")

    except ImportError:
        # 如果未安装 onnx 或 onnxruntime，跳过验证步骤
        print("⚠ ONNX/ONNX Runtime not installed, skipping verification")

    print("\n" + "=" * 60)
    print("ONNX export completed!")
    print("=" * 60)


def main():
    """
    主函数：解析命令行参数并调用导出函数
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Export SparseDriveV2 model to ONNX directly")
    # 添加 --ckpt 参数：训练好的 checkpoint 路径（必需）
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the trained checkpoint (.ckpt)"
    )
    # 添加 --output 参数：ONNX 模型输出路径（可选，默认 model.onnx）
    parser.add_argument(
        "--output",
        type=str,
        default="model.onnx",
        help="Output path for the ONNX model"
    )
    # 添加 --opset 参数：ONNX 算子集版本（可选，默认17）
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 获取输出目录路径
    output_dir = os.path.dirname(args.output)
    # 如果输出目录不存在，创建目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 调用核心导出函数
    export_onnx(args.ckpt, args.output, args.opset)


# 当脚本直接运行时执行 main 函数
if __name__ == "__main__":
    main()
