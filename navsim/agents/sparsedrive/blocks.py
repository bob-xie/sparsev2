# 导入类型注解模块
from typing import List, Optional, Tuple

# 导入数学运算模块
import math

# 导入 numpy 库
import numpy as np

# 导入 PyTorch 核心库
import torch

# 导入 PyTorch 神经网络模块
import torch.nn as nn

# 导入自定义的可变形聚合函数（DAF = Deformable Aggregation Function）
from navsim.agents.sparsedrive.ops import deformable_aggregation_func as DAF


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    """
    创建线性层 + ReLU + LayerNorm 的组合模块
    
    :param embed_dims: 嵌入维度（默认256）
    :param in_loops: 内层循环次数（每次循环包含一个线性层和一个ReLU）
    :param out_loops: 外层循环次数（每次外层循环后添加一个LayerNorm）
    :param input_dims: 输入维度，默认为 embed_dims
    :return: 层列表
    """
    if input_dims is None:
        input_dims = embed_dims  # 默认输入维度等于嵌入维度
    
    layers = []  # 初始化层列表
    
    for _ in range(out_loops):  # 外层循环
        for _ in range(in_loops):  # 内层循环
            layers.append(nn.Linear(input_dims, embed_dims))  # 添加线性层
            layers.append(nn.ReLU(inplace=True))  # 添加ReLU激活
            input_dims = embed_dims  # 更新输入维度
        layers.append(nn.LayerNorm(embed_dims))  # 添加LayerNorm
    
    return layers  # 返回层列表


class DeformableFeatureAggregation(nn.Module):
    """
    可变形特征聚合模块（Deformable Feature Aggregation, DFA）
    
    核心功能：将多视角、多尺度的视觉特征聚合到预定义的3D关键点上
    
    工作流程：
    1. 根据轨迹锚点生成3D关键点
    2. 将3D关键点投影到各相机的2D图像平面
    3. 使用可变形卷积从各相机、各尺度特征图中提取关键点特征
    4. 通过注意力权重对多相机、多尺度特征进行融合
    5. 输出聚合后的特征
    
    输入：
    - instance_feature: 轨迹特征嵌入 [B, num_anchor, embed_dims]
    - anchor: 轨迹锚点（路径点）[B, num_anchor, num_sample*3]
    - anchor_embed: 锚点嵌入 [B, num_anchor, embed_dims]
    - feature_maps: 多尺度特征图列表，每个元素 [B, num_cams, embed_dims, H, W]
    - metas: 元数据，包含投影矩阵等
    - depth_prob: 深度概率分布（可选）
    
    输出：
    - output: 聚合后的特征 [B, num_anchor, embed_dims]
    """

    def __init__(
        self,
        config: dict = None,               # 配置字典
        embed_dims: int = 256,             # 嵌入维度（默认256）
        num_groups: int = 8,               # 分组数（用于特征分组融合）
        num_levels: int = 4,               # 特征图层级数（默认4）
        num_cams: int = 6,                # 相机数量（默认6，实际使用3个）
        num_pts: int = 8,                 # 每个锚点的采样点数
        proj_drop: float = 0.0,           # 投影层dropout率
        attn_drop: float = 0.0,           # 注意力dropout率
        kps_generator: dict = None,       # 关键点生成器配置（未使用）
        temporal_fusion_module=None,      # 时间融合模块（未使用）
        use_temporal_anchor_embed=True,   # 是否使用时间锚点嵌入（未使用）
        use_deformable_func=False,        # 是否使用可变形聚合函数
        use_camera_embed=False,           # 是否使用相机嵌入
        residual_mode="add",              # 残差模式：add（相加）或 cat（拼接）
        filter_outlier=True,              # 是否过滤离群点
        min_depth=None,                   # 最小深度（用于深度归一化）
        max_depth=None,                   # 最大深度（用于深度归一化）
    ):
        """
        初始化可变形特征聚合模块
        
        参数说明：
        - embed_dims: 所有特征向量的维度
        - num_groups: 将特征分为num_groups组进行加权融合
        - num_levels: FPN输出的特征图层级数（4个尺度）
        - num_cams: 相机数量（默认6，实际使用cam_l0, cam_f0, cam_r0三个）
        - num_pts: 每个路径锚点采样的3D关键点数量
        - use_deformable_func: 是否使用自定义的可变形聚合函数DAF
        - use_camera_embed: 是否使用投影矩阵编码作为相机嵌入
        - residual_mode: 输出与输入的残差连接方式
        """
        super(DeformableFeatureAggregation, self).__init__()  # 调用父类初始化
        
        # 验证嵌入维度是否能被分组数整除
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        
        self.config = config                    # 保存配置
        self.group_dims = int(embed_dims / num_groups)  # 每组维度（256/8=32）
        self.embed_dims = embed_dims            # 嵌入维度
        self.num_levels = num_levels            # 特征图层级数
        self.num_groups = num_groups            # 分组数
        self.num_cams = num_cams                # 相机数量
        self.use_temporal_anchor_embed = use_temporal_anchor_embed  # 时间锚点嵌入
        
        # 如果使用可变形函数，确保DAF已定义
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func  # 是否使用可变形函数
        self.attn_drop = attn_drop              # 注意力dropout率
        self.residual_mode = residual_mode      # 残差模式
        self.filter_outlier = filter_outlier    # 是否过滤离群点
        self.min_depth = min_depth              # 最小深度
        self.max_depth = max_depth              # 最大深度

        # 投影层dropout
        self.proj_drop = nn.Dropout(proj_drop)

        # 创建3D关键点生成器
        self.kps_generator = SparsePoint3DKeyPointsGenerator(
            embed_dims=embed_dims,                           # 嵌入维度
            num_sample=num_pts,                              # 采样点数
            num_learnable_pts=config.num_learnable_pts,      # 可学习关键点数量
            fix_height=config.fix_height,                    # 固定高度列表
            ground_height=0,                                 # 地面高度
        )
        self.num_pts = self.kps_generator.num_pts  # 更新实际关键点数量
        
        # 时间融合模块（未使用）
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(
                temporal_fusion_module, PLUGIN_LAYERS
            )
        else:
            self.temp_module = None
        
        # 输出投影层（将聚合后的特征投影回embed_dims维度）
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        # 如果使用相机嵌入
        if use_camera_embed:
            # 相机编码器：将3x4投影矩阵（12维）编码为embed_dims维度
            self.camera_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)  # 2层线性+ReLU+LayerNorm
            )
            # 权重全连接层：输出维度为 num_groups * num_levels * num_pts
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            # 不使用相机嵌入
            self.camera_encoder = None
            # 权重全连接层：输出维度为 num_groups * num_cams * num_levels * num_pts
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

        # 初始化权重
        self.init_weight()

    def init_weight(self):
        """初始化模块权重"""
        # 权重全连接层初始化为0
        nn.init.constant_(self.weights_fc.weight, 0)
        nn.init.constant_(self.weights_fc.bias, 0)

        # 输出投影层使用Xavier初始化
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(
        self,
        instance_feature: torch.Tensor,   # 轨迹特征嵌入 [B, num_anchor, embed_dims]
        anchor: torch.Tensor,              # 轨迹锚点 [B, num_anchor, num_sample*3]
        anchor_embed: torch.Tensor,        # 锚点嵌入 [B, num_anchor, embed_dims]
        feature_maps: List[torch.Tensor],  # 多尺度特征图列表
        metas: dict,                       # 元数据（投影矩阵等）
        depth_prob,                         # 深度概率分布（可选）
        return_kps_features: bool = False,  # 是否返回关键点特征
        **kwargs: dict,                     # 额外参数
    ):
        """
        前向传播：将多视角特征聚合到轨迹关键点上
        
        :param instance_feature: 轨迹特征嵌入 [B, num_anchor, embed_dims]
        :param anchor: 轨迹锚点（等距采样的路径点）[B, num_anchor, num_sample*3]
        :param anchor_embed: 锚点嵌入 [B, num_anchor, embed_dims]
        :param feature_maps: 多尺度特征图列表，每个元素 [B, num_cams, embed_dims, H, W]
        :param metas: 元数据字典，包含 projection_mat（投影矩阵）等
        :param depth_prob: 深度概率分布 [B, num_cams, depth_bins, H, W]（可选）
        :return: 聚合后的特征 [B, num_anchor, embed_dims]
        """
        bs, num_anchor = instance_feature.shape[:2]  # 获取批量大小和锚点数量
        
        # 步骤1：根据锚点生成3D关键点
        # key_points形状：[B, num_anchor, num_pts, 3]
        key_points = self.kps_generator(anchor, instance_feature)

        # 步骤2：如果使用可变形函数，执行特征聚合
        if self.use_deformable_func:
            # 将3D关键点投影到各相机的2D图像平面
            # points_2d: [B, num_cams, num_anchor, num_pts, 2] - 2D坐标
            # depth: [B, num_cams, num_anchor, num_pts] - 深度值
            # mask: [B, num_cams, num_anchor, num_pts] - 有效掩码
            points_2d, depth, mask = self.project_points(
                key_points,
                metas["projection_mat"],      # 投影矩阵 [B, num_cams, 3, 4]
                metas.get("image_wh"),        # 图像宽高 [B, num_cams, 2]（可选）
            )

            # 步骤3：计算注意力权重
            # weights形状：[B, num_anchor, num_cams, num_levels, num_pts, num_groups]
            weights = self._get_weights(
                instance_feature, anchor_embed, metas, mask
            )

            # 步骤4：调整关键点2D坐标形状
            # 从 [B, num_cams, num_anchor, num_pts, 2] 变为 [B, num_anchor*num_pts, num_cams, 2]
            points_2d = points_2d.permute(0, 2, 3, 1, 4).reshape(
                bs, num_anchor * self.num_pts, -1, 2
            )
            
            # 步骤5：调整权重形状
            # 从 [B, num_anchor, num_cams, num_levels, num_pts, num_groups]
            # 变为 [B, num_anchor*num_pts, num_cams, num_levels, num_groups]
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor * self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            
            # 步骤6：如果提供了深度概率分布，使用深度感知的可变形聚合
            if depth_prob is not None:
                # 调整深度形状 [B, num_cams, num_anchor, num_pts] → [B, num_anchor*num_pts, num_cams, 1]
                depth = depth.permute(0, 2, 3, 1).reshape(
                    bs, num_anchor * self.num_pts, -1, 1
                )
                # 将深度归一化到 [0, depth_prob.shape[-1]-1] 范围
                depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
                depth = depth * (depth_prob.shape[-1] - 1)
                
                # 使用深度感知的可变形聚合函数
                features = DAF(
                    *feature_maps, points_2d, weights, depth_prob, depth
                )
            else:
                # 使用普通的可变形聚合函数
                features = DAF(*feature_maps, points_2d, weights)
            
            # 步骤7：调整特征形状并求和
            # 从 [B, num_anchor*num_pts, embed_dims] 变为 [B, num_anchor, num_pts, embed_dims]
            features = features.reshape(bs, num_anchor, self.num_pts, self.embed_dims)
            # 对所有关键点特征求和，得到每个锚点的聚合特征
            features = features.sum(dim=2)
        
        # 步骤8：投影和残差连接
        # 通过线性层投影并应用dropout
        output = self.proj_drop(self.output_proj(features))
        
        # 根据残差模式进行连接
        if self.residual_mode == "add":
            # 相加模式：output = output + instance_feature
            output = output + instance_feature
        elif self.residual_mode == "cat":
            # 拼接模式：output = [output, instance_feature]
            output = torch.cat([output, instance_feature], dim=-1)
        
        # 返回聚合后的特征 [B, num_anchor, embed_dims]
        return output

    def _get_weights(
        self, instance_feature, anchor_embed, metas=None, mask=None
    ):
        """
        计算多相机、多尺度特征融合的注意力权重
        
        :param instance_feature: 轨迹特征嵌入 [B, num_anchor, embed_dims]
        :param anchor_embed: 锚点嵌入 [B, num_anchor, embed_dims]
        :param metas: 元数据（包含投影矩阵）
        :param mask: 关键点有效掩码 [B, num_cams, num_anchor, num_pts]
        :return: 注意力权重 [B, num_anchor, num_cams, num_levels, num_pts, num_groups]
        """
        bs, num_anchor = instance_feature.shape[:2]  # 获取批量大小和锚点数量
        
        # 将实例特征和锚点嵌入相加
        if anchor_embed is not None:
            feature = instance_feature + anchor_embed
        else:
            feature = instance_feature
        
        # 如果使用相机嵌入，将投影矩阵编码后与特征相加
        if self.camera_encoder is not None:
            # 提取投影矩阵的前3行（3x3旋转+平移部分），展平为12维
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(bs, self.num_cams, -1)
            )
            # 将轨迹特征与相机嵌入相加（广播）
            feature = feature[:, :, None] + camera_embed[:, None]

        # 通过全连接层计算权重
        weights = self.weights_fc(feature)
        
        # 如果提供了掩码且需要过滤离群点
        if mask is not None and self.filter_outlier:
            # 调整掩码形状 [B, num_cams, num_anchor, num_pts] → [B, num_anchor, num_cams, num_pts, 1, 1]
            mask = mask.permute(0, 2, 1, 3)[..., None, :, None]
            
            # 将权重调整为 [B, num_anchor, num_cams, num_levels, num_pts, num_groups]
            weights = weights.reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
            
            # 将无效关键点的权重设为负无穷（softmax后为0）
            weights = weights.masked_fill(
                torch.logical_and(~mask, mask.sum(dim=2, keepdim=True) != 0),
                float("-inf"),
            )
        
        # 将权重调整形状并应用softmax归一化
        weights = (
            weights.reshape(bs, num_anchor, -1, self.num_groups)  # 展平为 [B, num_anchor, N, num_groups]
            .softmax(dim=-2)  # 在N维度上归一化
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        
        # 如果训练模式且注意力dropout率大于0，应用dropout
        if self.training and self.attn_drop > 0:
            # 创建随机掩码
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            # 按比例保留权重
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        
        # 返回注意力权重
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        """
        将3D关键点投影到2D图像平面
        
        :param key_points: 3D关键点 [B, num_anchor, num_pts, 3]
        :param projection_mat: 投影矩阵 [B, num_cams, 3, 4]
        :param image_wh: 图像宽高 [B, num_cams, 2]（可选）
        :return: (points_2d, depth, mask)
                 - points_2d: 2D坐标 [B, num_cams, num_anchor, num_pts, 2]
                 - depth: 深度值 [B, num_cams, num_anchor, num_pts]
                 - mask: 有效掩码 [B, num_cams, num_anchor, num_pts]
        """
        bs, num_anchor, num_pts = key_points.shape[:3]  # 获取维度

        # 将3D点扩展为齐次坐标 [B, num_anchor, num_pts, 4]
        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        
        # 使用投影矩阵将3D点投影到2D
        # projection_mat: [B, num_cams, 3, 4]
        # pts_extend: [B, num_anchor, num_pts, 4]
        # points_2d: [B, num_cams, num_anchor, num_pts, 3]（齐次坐标）
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        
        # 提取深度（齐次坐标的第三个分量）
        depth = points_2d[..., 2]
        
        # 掩码：深度大于阈值的点为有效
        mask = depth > 1e-5
        
        # 将齐次坐标转换为归一化像素坐标（除以深度）
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        
        # 更新掩码：只保留在图像边界内的点
        mask = mask & (points_2d[..., 0] > 0) & (points_2d[..., 1] > 0)
        
        # 如果提供了图像宽高，进行归一化
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]  # 归一化到 [0, 1]
            mask = mask & (points_2d[..., 0] < 1) & (points_2d[..., 1] < 1)
        
        # 返回2D坐标、深度和掩码
        return points_2d, depth, mask

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        使用双线性插值从特征图中采样关键点特征（静态方法，未使用可变形卷积）
        
        :param feature_maps: 多尺度特征图列表
        :param key_points: 3D关键点 [B, num_anchor, num_pts, 3]
        :param projection_mat: 投影矩阵 [B, num_cams, 3, 4]
        :param image_wh: 图像宽高（可选）
        :return: 采样的特征 [B, num_anchor, num_cams, num_levels, num_pts, embed_dims]
        """
        num_levels = len(feature_maps)  # 特征图层级数
        num_cams = feature_maps[0].shape[1]  # 相机数量
        bs, num_anchor, num_pts = key_points.shape[:3]  # 获取维度

        # 将3D关键点投影到2D
        points_2d, _, _ = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        
        # 将坐标转换为 [-1, 1] 范围（grid_sample要求）
        points_2d = points_2d * 2 - 1
        
        # 展平为 [B*num_anchor*num_pts, num_cams, 2]
        points_2d = points_2d.flatten(end_dim=1)

        # 从各尺度特征图中采样
        features = []
        for fm in feature_maps:
            # 展平特征图 [B, num_cams, embed_dims, H, W] → [B*num_cams, embed_dims, H, W]
            # grid_sample采样：[B*num_cams, embed_dims, num_anchor*num_pts]
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        
        # 将多尺度特征堆叠 [num_levels, B*num_cams, embed_dims, num_anchor*num_pts]
        features = torch.stack(features, dim=1)
        
        # 调整形状为 [B, num_anchor, num_cams, num_levels, num_pts, embed_dims]
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        """
        多视角、多尺度特征融合（使用注意力权重加权求和）
        
        :param features: 采样的特征 [B, num_anchor, num_cams, num_levels, num_pts, embed_dims]
        :param weights: 注意力权重 [B, num_anchor, num_cams, num_levels, num_pts, num_groups]
        :return: 融合后的特征 [B, num_anchor, num_pts, embed_dims]
        """
        bs, num_anchor = weights.shape[:2]  # 获取批量大小和锚点数量
        
        # 将特征按组划分：[B, num_anchor, num_cams, num_levels, num_pts, num_groups, group_dims]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        
        # 对相机维度和尺度维度求和
        features = features.sum(dim=2).sum(dim=2)
        
        # 调整形状为 [B, num_anchor, num_pts, embed_dims]
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        
        return features


class SparsePoint3DKeyPointsGenerator(nn.Module): 
    """
    稀疏3D关键点生成器
    
    根据轨迹锚点（等距采样的路径点）生成3D关键点，支持：
    1. 固定高度采样（从地面到不同高度）
    2. 可学习的偏移量（增强关键点的表达能力）
    
    输入：
    - anchor: 轨迹锚点 [B, num_anchor, num_sample*3]（等距采样的路径点）
    - instance_feature: 轨迹特征嵌入 [B, num_anchor, embed_dims]（可选，用于可学习偏移）
    
    输出：
    - key_points: 3D关键点 [B, num_anchor, num_pts, 3]
    """

    def __init__(
        self,
        embed_dims: int = 256,              # 嵌入维度
        num_sample: int = 20,               # 每个锚点的样本数（路径点数）
        num_learnable_pts: int = 0,         # 每个路径点的可学习关键点数量
        fix_height: Tuple = (0,),           # 固定高度列表（如 (0., -0.25, -0.5, 0.25, 0.5)）
        ground_height: int = 0,             # 地面高度
    ):
        """
        初始化稀疏3D关键点生成器
        
        :param num_sample: 每个轨迹锚点的路径点数（如50个等距采样点）
        :param num_learnable_pts: 每个路径点的可学习关键点数量（0表示不可学习）
        :param fix_height: 固定高度列表，用于生成不同高度的关键点
        :param ground_height: 地面高度（默认0）
        """
        super(SparsePoint3DKeyPointsGenerator, self).__init__()  # 调用父类初始化
        #必须在子类的 __init__ 中调用 super().__init__() ，否则模块无法正常工作
        self.embed_dims = embed_dims  # 嵌入维度
        self.num_sample = num_sample  # 每个锚点的样本数
        self.num_learnable_pts = num_learnable_pts  # 可学习关键点数量
        
        # 计算总关键点数量
        if self.num_learnable_pts > 0:
            # 可学习模式：num_sample × num_fix_height × num_learnable_pts
            self.num_pts = num_sample * len(fix_height) * num_learnable_pts
            # 创建可学习偏移量的全连接层
            self.learnable_fc = nn.Linear(self.embed_dims, self.num_pts * 2)
        else:
            # 不可学习模式：num_sample × num_fix_height
            self.num_pts = num_sample * len(fix_height)

        self.fix_height = np.array(fix_height)  # 固定高度列表（numpy数组）
        self.ground_height = ground_height      # 地面高度

        # 初始化权重
        self.init_weight()

    def init_weight(self):
        """初始化可学习层的权重"""
        if self.num_learnable_pts > 0:
            nn.init.xavier_uniform_(self.learnable_fc.weight)
            nn.init.constant_(self.learnable_fc.bias, 0)

    def forward(
        self,
        anchor,                           # 轨迹锚点 [B, num_anchor, num_sample*3]
        instance_feature=None,            # 轨迹特征嵌入 [B, num_anchor, embed_dims]
        T_cur2temp_list=None,            # 当前帧到历史帧的变换矩阵（未使用）
        cur_timestamp=None,               # 当前时间戳（未使用）
        temp_timestamps=None,            # 历史时间戳（未使用）
    ):
        """
        前向传播：生成3D关键点
        
        :param anchor: 轨迹锚点（等距采样的路径点）[B, num_anchor, num_sample*3]
        :param instance_feature: 轨迹特征嵌入（用于计算可学习偏移）[B, num_anchor, embed_dims]
        :return: 3D关键点 [B, num_anchor, num_pts, 3]
        """
        bs, num_anchor, _ = anchor.shape  # 获取批量大小和锚点数量
        
        # 将锚点调整为 [B, num_anchor, num_sample, 3] 形状
        key_points = anchor.view(bs, num_anchor, self.num_sample, -1)
        
        # 如果使用可学习关键点
        if self.num_learnable_pts > 0:
            # 通过全连接层计算偏移量 [B, num_anchor, num_pts*2]
            offset = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_sample, len(self.fix_height), self.num_learnable_pts, 2)
            )        
            # 将偏移量加到路径点上（只在x,y方向）
            key_points = offset + key_points[..., None, None, :]
        else:
            # 不可学习模式，添加两个None维度
            key_points = key_points[..., None, None, :]
        
        # 添加z坐标（地面高度）
        # key_points形状：[B, num_anchor, num_sample, num_fix_height, num_learnable_pts, 3]
        key_points = torch.cat(
            [
                key_points,
                key_points.new_full(key_points.shape[:-1]+(1,), fill_value=self.ground_height),
            ],
            dim=-1,
        )
        
        # 创建高度偏移量 [num_fix_height, 3]
        fix_height = key_points.new_tensor(self.fix_height)
        height_offset = key_points.new_zeros([len(fix_height), 2])
        height_offset = torch.cat([height_offset, fix_height[:,None]], dim=-1)
        
        # 将高度偏移量加到关键点上
        key_points = key_points + height_offset[None, None, None, :, None]
        
        # 展平维度：[B, num_anchor, num_pts, 3]
        key_points = key_points.flatten(2, 4)
        
        # 如果没有时间信息，直接返回关键点
        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        # 时间融合（未使用）
        temp_key_points_list = []
        for i, t_time in enumerate(temp_timestamps):
            temp_key_points = key_points
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list