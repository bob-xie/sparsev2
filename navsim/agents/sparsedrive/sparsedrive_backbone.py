# 导入 numpy 库，用于数值计算和数组操作
import numpy as np

# 导入 PyTorch 核心库，用于张量操作和神经网络
import torch

# 导入 PyTorch 神经网络模块
import torch.nn as nn

# 导入 PyTorch 功能模块，包含激活函数、池化等操作
import torch.nn.functional as F

# 导入 timm 库，提供预训练的计算机视觉模型
import timm

# 导入 PyTorch Vision 的 FPN（特征金字塔网络）模块
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork as FPN

# 导入自定义的可变形卷积格式转换工具
from navsim.agents.sparsedrive.ops import deformable_format

# 导入自定义的 GridMask 数据增强模块
from .grid_mask import GridMask


class SparseBackbone(nn.Module):
    """
    SparseDrive 视觉骨干网络类
    
    负责从多视角相机图像中提取特征，支持：
    1. ResNet-34 骨干网络提取多尺度特征
    2. FPN（特征金字塔网络）进行多尺度特征融合
    3. GridMask 数据增强（可选）
    
    输入：多视角相机图像 [B, num_cams, 3, H, W]
    输出：多尺度特征图列表，每个特征图形状为 [B, num_cams, d_model, H', W']
    """

    def __init__(self, config):
        """
        初始化视觉骨干网络
        
        :param config: SparseDriveConfig 配置对象，包含模型参数
        """
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        
        self._config = config  # 保存配置对象
        self.embed_dims = config.d_model  # 特征嵌入维度（默认256）
        self.use_grid_mask = config.use_grid_mask  # 是否使用 GridMask 数据增强（默认True）
        self.with_img_neck = config.with_img_neck  # 是否使用 FPN Neck（特征金字塔网络，默认True）

        # 如果启用 GridMask，创建 GridMask 实例
        if self.use_grid_mask:
            self.grid_mask = GridMask(
                True, True,    # 参数：固定大小、固定宽高比
                rotate=1,       # 旋转角度数
                offset=False,   # 是否偏移
                ratio=0.5,      # 遮挡比例
                mode=1,         # 模式选择
                prob=0.7        # 应用概率
            )

        # 验证图像架构是否支持（当前仅支持 resnet34）
        assert config.image_architecture in ["resnet34"], \
            f"Image architecture {config.image_architecture} not supported."
        
        # 创建 ResNet-34 骨干网络
        self.img_backbone = timm.create_model(
            config.image_architecture,  # 模型名称："resnet34"
            pretrained=True,            # 使用预训练权重
            features_only=True,         # 仅输出特征图，不输出分类结果
            pretrained_cfg_overlay=dict(file=config.bkb_path),  # 指定预训练权重路径
            out_indices=(1, 2, 3, 4)[-config.num_levels:]  # 输出哪些层的特征（默认取最后4层）
        )
        
        # 如果启用 FPN Neck
        if self.with_img_neck:
            self.img_neck = FPN(
                in_channels_list=[64, 128, 256, 512][-config.num_levels:],  # 各层输入通道数
                out_channels=self.embed_dims,  # 统一输出通道数（256）
            )
        else:
            # 如果不启用 FPN，使用简单的 3x3 卷积将通道数调整为 d_model
            self.img_neck = nn.Conv2d(
                512,              # 输入通道数（ResNet最后一层的通道数）
                config.d_model,   # 输出通道数（256）
                kernel_size=(3, 3),  # 卷积核大小
                stride=1,         # 步长
                padding=(1, 1),   # 填充（保持尺寸不变）
                bias=True,        # 使用偏置
            )

    def forward(self, img):
        """
        前向传播：从图像中提取特征
        
        :param img: 输入图像张量，形状为 [B, num_cams, 3, H, W]（多视角）或 [B, 3, H, W]（单视角）
        :return: 多尺度特征图列表，每个特征图形状为 [B, num_cams, d_model, H', W']
        """
        bs = img.shape[0]  # 批量大小（batch size）
        
        # 判断是否为多视角输入
        if img.dim() == 5:  # 多视角：[B, num_cams, 3, H, W]
            num_cams = img.shape[1]  # 相机数量（默认3：左、前、右）
            img = img.flatten(end_dim=1)  # 展平为 [B×num_cams, 3, H, W]
        else:  # 单视角：[B, 3, H, W]
            num_cams = 1
        
        # 如果启用 GridMask，应用数据增强
        if self.use_grid_mask:
            img = self.grid_mask(img)
        
        # 使用 ResNet-34 提取特征
        # 输出为列表，包含多个尺度的特征图：
        # - feature_maps[0]: [B×C, 64, H/4, W/4]
        # - feature_maps[1]: [B×C, 128, H/8, W/8]
        # - feature_maps[2]: [B×C, 256, H/16, W/16]
        # - feature_maps[3]: [B×C, 512, H/32, W/32]
        feature_maps = self.img_backbone(img)
        
        # 如果启用 FPN，进行多尺度特征融合
        if self.with_img_neck:
            # 将特征列表转换为字典 {feat_0: tensor, feat_1: tensor, ...}
            feature_dict = {f"feat_{i}": feature_maps[i] for i in range(len(feature_maps))}
            # FPN 融合后，输出字典，转换为列表
            feature_maps = list(self.img_neck(feature_dict).values())
        else:
            # 如果不启用 FPN，仅处理最后一层特征
            feature_maps = [self.img_neck(feature_maps[-1])]

        # 将特征图重构回多相机维度
        for i, feat in enumerate(feature_maps):
            # 形状变换：[B×C, d_model, H', W'] → [B, C, d_model, H', W']
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )

        # 返回多尺度特征图列表
        # 每个元素形状：[B, num_cams, d_model, H', W']
        return feature_maps
