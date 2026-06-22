"""
可变形特征聚合算子 - SparseDriveV2 的核心 CUDA 算子

核心功能：
1. 实现可变形注意力机制，将 3D 路径点投影到图像平面
2. 在投影位置周围采样多尺度、多视角的视觉特征
3. 通过加权聚合生成路径/速度嵌入

实现层次：
- DeformableAggregationFunction: 标准可变形聚合（无深度）
- DeformableAggregationWithDepthFunction: 带深度的可变形聚合
- deformable_format: 特征图格式化函数（推荐使用）
- feature_maps_format: 多尺度特征图格式化函数（兼容旧接口）

依赖：deformable_aggregation_ext（C++/CUDA 实现）
"""
# ============================================================================
# 导入 PyTorch 自动求导相关模块
# ============================================================================
import torch
# Function: 自定义可微分的函数基类，支持前向和反向传播
# once_differentiable: 一次可微的装饰器，用于反向传播计算
from torch.autograd.function import Function, once_differentiable

# ============================================================================
# 导入 CUDA 算子（由 C++/CUDA 实现）
# deformable_aggregation_ext: 标准可变形聚合的 CUDA 实现
# deformable_aggregation_with_depth_ext: 带深度的可变形聚合 CUDA 实现
# ============================================================================
from . import deformable_aggregation_ext
from . import deformable_aggregation_with_depth_ext


# ============================================================================
# DeformableAggregationFunction: 标准可变形特征聚合前向/反向传播函数
# 继承自 torch.autograd.Function，实现自定义梯度计算
# ============================================================================
class DeformableAggregationFunction(Function):
    """标准可变形特征聚合算子（无深度）"""

    @staticmethod
    def forward(
        ctx,                    # 上下文，用于保存反向传播需要的变量
        mc_ms_feat,            # 输入: 多相机多尺度特征 [B, N, C]
        spatial_shape,          # 输入: 每层的空间形状 [num_levels, 2] - 每层的 (H, W)
        scale_start_index,      # 输入: 每层特征点在展平后的起始索引 [num_levels]
        sampling_location,      # 输入: 采样位置坐标 [B, N, num_pts, 2] - 每个采样点的 (u, v)
        weights,                # 输入: 聚合权重 [B, N, num_pts, num_groups]
    ):
        """
        前向传播：执行可变形特征聚合

        输入维度详解：
        - B: batch size，批量大小
        - N: 特征点总数 = num_cams × (H₁×W₁ + H₂×W₂ + ... + Hₙ×Wₙ)
        - C: 特征通道数 = 256
        - num_pts: 每个查询点的采样点数
        - num_groups: 分组数，用于分组归一化

        示例数值：
        - B = 8
        - N = 3 cameras × (64×128 + 32×64 + 16×32 + 8×16) = 3 × 12800 ≈ 38400
        - C = 256
        - num_pts = 50（路径点数量）
        - num_groups = 8

        输出：
        - output: [B, num_pts, C] - 聚合后的特征
        """
        # 确保所有输入张量在内存中是连续存储的（CUDA kernel 要求）
        mc_ms_feat = mc_ms_feat.contiguous().float()          # [B, N, C] - 多相机多尺度特征
        spatial_shape = spatial_shape.contiguous().int()      # [num_levels, 2] - 每层 (H, W)
        scale_start_index = scale_start_index.contiguous().int()  # [num_levels] - 每层起始索引
        sampling_location = sampling_location.contiguous().float()  # [B, N, num_pts, 2] - 采样坐标
        weights = weights.contiguous().float()               # [B, N, num_pts, num_groups] - 聚合权重

        # 调用 CUDA 算子执行前向计算
        output = deformable_aggregation_ext.deformable_aggregation_forward(
            mc_ms_feat,         # 输入特征
            spatial_shape,       # 空间形状
            scale_start_index,   # 层起始索引
            sampling_location,   # 采样位置
            weights,             # 聚合权重
        )
        # 输出: [B, num_pts, num_embeds] - 聚合后的特征

        # 保存反向传播需要的变量到 ctx
        ctx.save_for_backward(
            mc_ms_feat,          # 反向传播时需要计算 mc_ms_feat 的梯度
            spatial_shape,        # 反向传播时需要
            scale_start_index,   # 反向传播时需要
            sampling_location,   # 反向传播时需要计算采样位置的梯度
            weights,             # 反向传播时需要计算权重的梯度
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        反向传播：计算梯度

        grad_output: [B, num_pts, C] - 输出特征的梯度

        返回梯度（与前向输入对应）：
        - grad_mc_ms_feat: [B, N, C] - 特征梯度
        - grad_sampling_location: [B, N, num_pts, 2] - 采样位置梯度
        - grad_weights: [B, N, num_pts, num_groups] - 权重梯度
        - spatial_shape 和 scale_start_index 不需要梯度，返回 None
        """
        # 从 ctx 恢复前向保存的变量
        (
            mc_ms_feat,          # [B, N, C]
            spatial_shape,       # [num_levels, 2]
            scale_start_index,   # [num_levels]
            sampling_location,   # [B, N, num_pts, 2]
            weights,             # [B, N, num_pts, num_groups]
        ) = ctx.saved_tensors

        # 确保张量连续且类型正确
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        # 初始化梯度张量（与输入同形状，值为0）
        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)           # [B, N, C]
        grad_sampling_location = torch.zeros_like(sampling_location)  # [B, N, num_pts, 2]
        grad_weights = torch.zeros_like(weights)                 # [B, N, num_pts, num_groups]

        # 调用 CUDA 算子执行反向计算
        deformable_aggregation_ext.deformable_aggregation_backward(
            mc_ms_feat,          # 前向输入（不需要梯度）
            spatial_shape,       # 前向输入（不需要梯度）
            scale_start_index,    # 前向输入（不需要梯度）
            sampling_location,    # 前向输入
            weights,              # 前向输入
            grad_output.contiguous(),  # 输出梯度
            grad_mc_ms_feat,      # 输出的梯度
            grad_sampling_location,   # 输出的梯度
            grad_weights,         # 输出的梯度
        )

        # 返回梯度（与前向参数顺序对应）
        # spatial_shape 和 scale_start_index 无梯度，返回 None
        return (
            grad_mc_ms_feat,      # 对应 mc_ms_feat
            None,                 # 对应 spatial_shape
            None,                 # 对应 scale_start_index
            grad_sampling_location,  # 对应 sampling_location
            grad_weights,         # 对应 weights
        )


# ============================================================================
# DeformableAggregationWithDepthFunction: 带深度的可变形特征聚合
# 在标准可变形聚合基础上增加了深度感知能力
# ============================================================================
class DeformableAggregationWithDepthFunction(Function):
    """带深度的可变形特征聚合算子"""

    @staticmethod
    def forward(
        ctx,                    # 上下文
        mc_ms_feat,            # 输入: 多相机多尺度特征 [B, N, C + depth_dim]
        spatial_shape,          # 输入: 每层的空间形状 [num_levels, 2]
        scale_start_index,      # 输入: 每层特征点的起始索引 [num_levels]
        sampling_location,      # 输入: 采样位置坐标 [B, N, num_pts, 3] - (u, v, depth)
        weights,                # 输入: 聚合权重 [B, N, num_pts, num_groups]
        num_depths,             # 输入: 深度维度大小
    ):
        """
        前向传播：执行带深度的可变形特征聚合

        与标准版的区别：
        1. sampling_location 增加了深度维度：[B, N, num_pts, 3] 而不是 [B, N, num_pts, 2]
        2. mc_ms_feat 增加了深度特征通道
        3. 增加了 num_depths 参数用于处理深度

        输出：
        - output: [B, num_pts, C + depth_dim]
        """
        # 确保所有输入张量连续且类型正确
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        # 调用带深度的 CUDA 算子
        output = deformable_aggregation_with_depth_ext.deformable_aggregation_with_depth_forward(
            mc_ms_feat,         # 输入特征（包含深度特征）
            spatial_shape,       # 空间形状
            scale_start_index,   # 层起始索引
            sampling_location,   # 采样位置（包含深度）
            weights,             # 聚合权重
            num_depths,          # 深度维度大小
        )

        # 保存反向传播需要的变量
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx._num_depths = num_depths  # 保存深度维度大小到 ctx
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """反向传播：计算带深度的梯度"""
        # 恢复保存的变量
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        num_depths = ctx._num_depths  # 从 ctx 获取深度维度大小

        # 确保张量连续
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        # 初始化梯度张量
        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)

        # 调用带深度的 CUDA 反向算子
        deformable_aggregation_with_depth_ext.deformable_aggregation_with_depth_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            num_depths,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )

        # 返回梯度（num_depths 无梯度，返回 None）
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
            None,
        )


# ============================================================================
# deformable_aggregation_func: 可变形特征聚合的 Python 接口
# 根据是否有深度信息自动选择标准版或深度版
# ============================================================================
def deformable_aggregation_func(
    mc_ms_feat,                # [B, N, C] - 多相机多尺度特征
    spatial_shape,             # [num_levels, 2] - 每层 (H, W)
    scale_start_index,          # [num_levels] - 每层起始索引
    sampling_location,          # [B, N, num_pts, 2/3] - 采样坐标
    weights,                    # [B, N, num_pts, num_groups] - 聚合权重
    depth_prob=None,           # 可选: 深度概率 [B, N, num_pts, depth_dim]
    depth=None                  # 可选: 深度值 [B, N, num_pts, 1]
):
    """
    可变形特征聚合的统一接口

    参数：
    - mc_ms_feat: 多相机多尺度特征 [B, N, C]
    - spatial_shape: 每层的空间形状 [num_levels, 2]
    - scale_start_index: 每层特征点的起始索引 [num_levels]
    - sampling_location: 采样位置坐标 [B, N, num_pts, 2] 或 [B, N, num_pts, 3]（有深度时）
    - weights: 聚合权重 [B, N, num_pts, num_groups]
    - depth_prob: 可选，深度概率 [B, N, num_pts, depth_dim]
    - depth: 可选，深度值 [B, N, num_pts, 1]

    返回：
    - output: [B, num_pts, C] 或 [B, num_pts, C + depth_dim]

    选择逻辑：
    - 如果提供了 depth_prob 和 depth，使用带深度的版本
    - 否则使用标准版本
    """
    # 如果提供了深度信息，使用带深度的版本
    if depth_prob is not None and depth is not None:
        # 拼接深度特征到主特征
        # [B, N, C] + [B, N, depth_dim] → [B, N, C + depth_dim]
        mc_ms_feat = torch.cat([mc_ms_feat, depth_prob], dim=-1)

        # 拼接深度值到采样位置
        # [B, N, num_pts, 2] + [B, N, num_pts, 1] → [B, N, num_pts, 3]
        sampling_location = torch.cat([sampling_location, depth], dim=-1)

        # 调用带深度的版本
        return DeformableAggregationWithDepthFunction.apply(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            depth_prob.shape[-1],  # 深度维度大小
        )
    else:
        # 调用标准版本
        return DeformableAggregationFunction.apply(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )


# ============================================================================
# feature_maps_format: 多尺度特征图格式化函数
# 支持正向（列表→合并）和逆向（合并→列表）转换
# ============================================================================
def feature_maps_format(feature_maps, inverse=False):
    """
    多尺度特征图格式化

    参数：
    - feature_maps: 多尺度特征图列表
                    例如: [feat_0, feat_1, feat_2, feat_3]
                    feat_i: [B, num_cams, C_i, H_i, W_i]
                    - feat_0: [B, 3, 64, 64, 128] - level 0 (H/4, W/4)
                    - feat_1: [B, 3, 128, 32, 64] - level 1 (H/8, W/8)
                    - feat_2: [B, 3, 256, 16, 32] - level 2 (H/16, W/16)
                    - feat_3: [B, 3, 512, 8, 16]  - level 3 (H/32, W/32)
    - inverse: False=正向转换（列表→合并），True=逆向转换（合并→列表）

    正向输出：
    - col_feats: [B, num_cams * sum(H_i*W_i), C_total]
                 展平并拼接所有特征 [B, 3*12800, 256*4]
    - spatial_shape: [num_cams, num_levels, 2]
                     每层空间形状 [[H_0, W_0], [H_1, W_1], ...]
    - scale_start_index: [num_cams, num_levels]
                          每层在展平后的起始索引

    逆向输出：
    - mc_ms_feat: List[List[Tensor]]
                  [num_cams][num_levels] 每个元素 [B, C_i, H_i, W_i]
    """
    # ==================== 逆向转换（合并→列表） ====================
    if inverse:
        # 解包合并后的特征
        col_feats, spatial_shape, scale_start_index = feature_maps
        num_cams, num_levels = spatial_shape.shape[:2]  # 例如: 3, 4

        # 计算每层的特征点数量
        # spatial_shape: [num_cams, num_levels, 2] → [num_cams, num_levels]
        split_size = spatial_shape[..., 0] * spatial_shape[..., 1]  # 每层的 H*W
        split_size = split_size.cpu().numpy().tolist()  # 转为列表，如 [[5120], [2048], ...]

        # 处理相机分割
        # cam_split: 每组包含多少个相机
        idx = 0
        cam_split = [1]                  # 初始：第一组1个相机
        cam_split_size = [sum(split_size[0])]  # 第一组的总特征点数
        for i in range(num_cams - 1):
            # 如果当前相机与下一个相机的空间形状不同，则新开一组
            if not torch.all(spatial_shape[i] == spatial_shape[i + 1]):
                cam_split.append(0)
                cam_split_size.append(0)
            cam_split[-1] += 1  # 当前组相机数+1
            cam_split_size[-1] += sum(split_size[i + 1])  # 当前组特征点数累加

        # 分割并恢复相机维度
        # col_feats.split: 按 cam_split_size 分割
        mc_feat = [
            x.unflatten(1, (cam_split[i], -1))  # 恢复相机维度
            for i, x in enumerate(col_feats.split(cam_split_size, dim=1))
        ]

        # 恢复多尺度维度
        spatial_shape = spatial_shape.cpu().numpy().tolist()
        mc_ms_feat = []
        shape_index = 0
        for i, feat in enumerate(mc_feat):
            # 按 split_size 分割每个相机的特征
            feat = list(feat.split(split_size[shape_index], dim=2))
            for j, f in enumerate(feat):
                # 恢复空间维度
                # [B, C, H*W] → [B, C, H, W]
                feat[j] = f.unflatten(2, spatial_shape[shape_index][j])
                # [B, C, H, W] → [B, C, W, H] → [B, C, H, W] (调整维度)
                feat[j] = feat[j].permute(0, 1, 4, 2, 3)
            mc_ms_feat.append(feat)
            shape_index += cam_split[i]
        return mc_ms_feat

    # ==================== 正向转换（列表→合并） ====================
    # 递归处理嵌套列表
    if isinstance(feature_maps[0], (list, tuple)):
        # 递归格式化每个子列表
        formated = [feature_maps_format(x) for x in feature_maps]
        # 合并所有子列表的结果
        col_feats = torch.cat([x[0] for x in formated], dim=1)      # 拼接特征
        spatial_shape = torch.cat([x[1] for x in formated], dim=0)   # 拼接形状
        scale_start_index = torch.cat([x[2] for x in formated], dim=0)  # 拼接索引
        return [col_feats, spatial_shape, scale_start_index]

    # 单层多相机特征的处理
    bs, num_cams = feature_maps[0].shape[:2]  # batch size, 相机数量

    spatial_shape = []  # 存储每层的 (H, W)
    col_feats = []      # 存储展平后的特征

    for i, feat in enumerate(feature_maps):
        # 提取空间形状
        spatial_shape.append(feat.shape[-2:])  # (H, W)

        # 展平特征: [B, num_cams, C, H, W] → [B, num_cams, C, H*W]
        col_feats.append(
            torch.reshape(feat, (bs, num_cams, feat.shape[2], -1))
        )

    # 拼接所有层的特征
    # [B, num_cams, C_0, H_0*W_0], [B, num_cams, C_1, H_1*W_1], ...
    # → [B, num_cams, C_total, sum(H_i*W_i)]
    col_feats = torch.cat(col_feats, dim=-1)
    # → [B, num_cams, sum(H_i*W_i), C_total]
    col_feats = col_feats.permute(0, 1, 3, 2)
    # → [B, num_cams*sum(H_i*W_i), C_total]
    col_feats = col_feats.flatten(1, 2)

    # 构建空间形状张量
    # spatial_shape: [(H_0, W_0), (H_1, W_1), ...] → [num_levels, 2]
    spatial_shape = [spatial_shape] * num_cams  # 复制给每个相机
    spatial_shape = torch.tensor(
        spatial_shape,
        dtype=torch.int64,
        device=col_feats.device,
    )

    # 计算每层的起始索引
    # [num_cams, num_levels, 2] → [num_cams, num_levels]
    scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
    # 展平并计算累积和
    scale_start_index = scale_start_index.flatten().cumsum(dim=0)
    # 在前面加0，得到正确的起始位置
    scale_start_index = torch.cat(
        [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
    )
    # 恢复到 [num_cams, num_levels] 形状
    scale_start_index = scale_start_index.reshape(num_cams, -1)

    # 组合返回
    feature_maps = [
        col_feats,          # [B, num_cams*sum(H_i*W_i), C_total]
        spatial_shape,       # [num_cams, num_levels, 2]
        scale_start_index,   # [num_cams, num_levels]
    ]
    return feature_maps


# ============================================================================
# deformable_format: 将特征图转换为可变形注意力需要的格式（推荐接口）
# 与 feature_maps_format 的区别：
# - deformable_format: 输出 [feat_flatten, spatial_shapes, level_start_index]
# - feature_maps_format: 输出 [col_feats, spatial_shape, scale_start_index]
# ============================================================================
def deformable_format(
    feature_maps,              # 输入: 特征图列表 List[[B, C, H, W]]
    spatial_shapes=None,       # 可选: 指定空间形状 [num_levels, 2]
    level_start_index=None,    # 可选: 指定层起始索引 [num_levels]
    flat_batch=False,          # 是否扁平化 batch 维度
    batch_size=None,           # batch 大小（用于恢复形状）
):
    """
    将多尺度特征图转换为可变形注意力格式

    参数：
    - feature_maps: 多尺度特征图列表
                    List[[B, C_i, H_i, W_i]] for i in levels
                    例如:
                    - feature_maps[0]: [B, 64, 64, 128]  - level 0
                    - feature_maps[1]: [B, 128, 32, 64]  - level 1
                    - feature_maps[2]: [B, 256, 16, 32]  - level 2
                    - feature_maps[3]: [B, 512, 8, 16]   - level 3

    - spatial_shapes: 可选，手动指定每层的空间形状
                      如果为 None，从 feature_maps 自动推断

    - level_start_index: 可选，手动指定每层的起始索引
                         如果为 None，自动计算

    - flat_batch: 是否将 batch 维度与空间维度合并

    - batch_size: 用于恢复 batch 维度

    返回（当 spatial_shapes=None 时）：
    - feat_flatten: [B, num_feat_points, C] - 展平后的特征
                    num_feat_points = sum(H_i * W_i) for all levels
    - spatial_shapes: [num_levels, 2] - 每层的 (H, W)
    - level_start_index: [num_levels] - 每层的起始索引

    示例：
    输入: feature_maps = [feat_0, feat_1, feat_2, feat_3]
          feat_i: [B=8, C=64*i, H=64/2^i, W=128/2^i]
    输出:
          feat_flatten: [8, 12800, 256]  (64*128 + 32*64 + 16*32 + 8*16 = 12800)
          spatial_shapes: [[64, 128], [32, 64], [16, 32], [8, 16]]
          level_start_index: [0, 8192, 10240, 11520]
    """
    # ==================== 正向转换（特征图 → 展平格式） ====================
    if spatial_shapes is None:
        # 如果指定了 flat_batch，扁平化 batch 和空间维度
        if flat_batch and feature_maps[0].dim() > 4:
            feature_maps = [x.flatten(end_dim=-4) for x in feature_maps]

        feat_flatten = []    # 存储展平后的特征
        spatial_shapes = []  # 存储每层的 (H, W)

        # 遍历每个尺度的特征图
        for lvl, feat in enumerate(feature_maps):
            # 获取空间形状 (H, W)
            spatial_shape = torch._shape_as_tensor(feat)[-2:].to(feat.device)

            # 展平空间维度: [B, C, H, W] → [B, C, H*W] → [B, H*W, C]
            feat = feat.flatten(start_dim=-2).transpose(-1, -2)

            feat_flatten.append(feat)  # [B, H*W, C]
            spatial_shapes.append(spatial_shape)  # [2]

        # 拼接所有层的特征
        # [B, H_0*W_0, C_0], [B, H_1*W_1, C_1], ... → [B, H_0*W_0+H_1*W_1+..., C_total]
        feat_flatten = torch.cat(feat_flatten, -2)

        # 拼接空间形状: [num_levels, 2]
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)

        # 计算每层的起始索引
        # 例如: [[64, 128], [32, 64]] → [8192, 2048] → cumsum → [8192, 10240]
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),  # [0]
                spatial_shapes.prod(1).cumsum(0)[:-1],  # [8192, 10240) - 不包含最后一个
            )
        )

        # 返回展平的特征和元数据
        return feat_flatten, spatial_shapes, level_start_index

    # ==================== 逆向转换（展平格式 → 特征图） ====================
    else:
        # 计算每层的特征点数量
        split_size = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        # 例如: [[64, 128], [32, 64]] → [8192, 2048]

        # 调整维度顺序: [B, N, C] → [B, C, N]
        feature_maps = feature_maps.transpose(-1, -2)

        # 按层分割
        feature_maps = list(torch.split(feature_maps, split_size, dim=-1))

        # 恢复每层的空间维度
        for i, feat in enumerate(feature_maps):
            # [B, C, H*W] → [B, C, H, W]
            feature_maps[i] = feature_maps[i].unflatten(
                -1, (spatial_shapes[i, 0], spatial_shapes[i, 1])
            )

            # 如果指定了 batch_size，恢复 batch 维度
            if batch_size is not None:
                if isinstance(batch_size, int):
                    # [B, C, H, W] → [batch_size, -1, C, H, W]
                    feature_maps[i] = feature_maps[i].unflatten(
                        0, (batch_size, -1)
                    )
                else:
                    # batch_size 是元组的情况
                    feature_maps[i] = feature_maps[i].unflatten(
                        0, batch_size + (-1,)
                    )

        return feature_maps
