"""
CustomTransformerDecoder - SparseDriveV2 的核心解码器模块

核心创新点：
1. 可分解轨迹词汇表：将轨迹分解为路径(path)和速度(velocity)两个独立词汇表
2. 分层评分策略：先粗粒度筛选，再细粒度评分，平衡计算效率和精度
3. 可变形注意力：自适应特征聚合，增强空间理解能力
4. PDM指标监督：直接优化安全、合规、效率、舒适性指标
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.sparsedrive.ops import deformable_format
from navsim.common.dataclasses import Trajectory

from .blocks import DeformableFeatureAggregation
from .scorer.get_pdm_score_v1 import get_pdm_score_para as get_pdm_score_v1
from .scorer.get_pdm_score_v2 import get_pdm_score_para as get_pdm_score_v2


def _get_clones(module, N):
    """
    创建模块的多个副本
    
    :param module: 要复制的模块
    :param N: 副本数量
    :return: 包含N个副本的ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CustomTransformerDecoder(nn.Module):
    """
    自定义Transformer解码器，由多个解码器层堆叠组成
    
    核心设计理念：
    - 采用分层评分策略，逐层缩小候选集规模
    - 第一层：粗筛选，从1024×256=262,144候选中筛选到128×64=8,192
    - 第二层：细筛选，从8,192候选中筛选到20×10=200
    - 最终层：PDM指标预测和最优轨迹选择
    
    每层执行流程：
    1. 路径分支：可变形注意力 + MHA + FFN → 路径评分
    2. 速度分支：图像注意力 + MHA + FFN → 速度评分  
    3. 粗粒度筛选：Top-K筛选路径和速度词汇
    4. 最终层（仅最后一层）：轨迹组合 + PDM指标预测 + 最优轨迹选择
    
    :param num_poses: 轨迹姿态数量，默认值: 8（对应4秒，每0.5秒一个姿态）
    :param d_model: 模型维度，默认值: 256（所有嵌入的维度）
    :param d_ffn: FFN隐藏层维度，默认值: 1024（前馈网络中间层维度）
    :param config: 配置对象，包含decoder_num_layers、path_filter_num、velocity_filter_num等
    """
    def __init__(self, num_poses, d_model, d_ffn, config):
        super().__init__()
        # 记录API使用情况（PyTorch内部日志）
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        # 存储所有解码器层
        self.layers = nn.ModuleList()
        
        # 根据配置创建多层解码器（默认2层）
        # config.decoder_num_layers: 解码器层数，默认值: 2
        for i in range(config.decoder_num_layers):
            decoder_layer = CustomTransformerDecoderLayer(
                num_poses=num_poses,           # 轨迹姿态数量
                d_model=d_model,               # 模型维度: 256
                d_ffn=d_ffn,                   # FFN维度: 1024
                config=config,                 # 配置对象
                decoder_idx=i,                 # 层索引（0或1），用于控制筛选策略
            )
            self.layers.append(decoder_layer)
    
    def forward(self, feature, input):
        """
        解码器前向传播
        
        :param feature: 特征元组，格式为 (path_embed, vel_embed, path_vocab, vel_vocab, traj_vocab, traj_mask)
                        初始时: path_embed=[B,1024,256], vel_embed=[B,256,256]
                        第一层后: path_embed=[B,128,256], vel_embed=[B,64,256]
                        第二层后: path_embed=[B,20,256], vel_embed=[B,10,256]
        :param input: 输入元组，格式为 (camera_feature, status_encoding, targets)
        :return: (outputs, loss_dicts) - 输出字典和损失字典
                 outputs: 包含最终预测轨迹 {"trajectory": [B,8,3]}
                 loss_dicts: 包含所有层的损失 {"path_loss_0", "velocity_loss_0", "path_loss_1", "velocity_loss_1", "traj_loss_1", ...}
        """
        outputs = {}   # 存储最终输出（如预测轨迹）
        loss_dicts = {} # 存储所有层的损失
        
        # 逐层处理，每层的输出特征作为下一层的输入特征
        # 第一层（decoder_idx=0）: 粗筛选，从1024×256→128×64
        # 第二层（decoder_idx=1）: 细筛选，从128×64→20×10，并生成最终轨迹
        for i, mod in enumerate(self.layers):
            # 调用解码器层，返回更新后的特征、输出和损失
            # feature: 更新后的特征元组（经过Top-K筛选）
            # output: 当前层的输出（仅最后一层有轨迹输出）
            # loss_dict: 当前层的损失（训练时非空，测试时为空）
            feature, output, loss_dict = mod(*feature, *input)
            
            # 合并输出：outputs 只会在最后一层包含 {"trajectory": ...}
            outputs.update(output)
            
            # 合并损失：每层的损失会被累积到 loss_dicts
            # 例如第一层添加: {"path_loss_0": ..., "velocity_loss_0": ...}
            # 第二层添加: {"path_loss_1": ..., "velocity_loss_1": ..., "traj_loss_1": ..., "metric_loss_1": ...}
            loss_dicts.update(loss_dict)
        
        # 返回合并后的输出和所有层的损失
        return outputs, loss_dicts


class CustomTransformerDecoderLayer(nn.Module):
    """
    自定义Transformer解码器层，包含三个并行分支：
    - 路径分支(path): 处理路径词汇表，使用可变形注意力提取空间特征
    - 速度分支(velocity): 处理速度词汇表，使用多头注意力
    - 轨迹分支(trajectory): 仅在最后一层激活，组合路径和速度生成最终轨迹
    
    :param num_poses: 轨迹姿态数量
    :param d_model: 模型维度
    :param d_ffn: FFN隐藏层维度
    :param config: 配置对象
    :param decoder_idx: 当前层索引
    """
    def __init__(self, num_poses, d_model, d_ffn, config, decoder_idx):
        super().__init__()
        self._config = config
        self.decoder_idx = decoder_idx
        
        # ==================== 路径分支 (Path Branch) ====================
        # 可变形特征聚合模块：核心组件，负责将3D路径点投影到图像平面并聚合多视角特征
        # 工作原理：
        # 1. 使用相机内外参将每个路径点投影到所有相机的图像平面
        # 2. 在投影位置周围采样特征（可变形偏移）
        # 3. 聚合多尺度、多视角的特征到路径嵌入中
        self.p_deform_model = DeformableFeatureAggregation(
            config=config,              # 配置对象
            embed_dims=d_model,         # 嵌入维度: 256
            num_groups=8,               # 分组数，用于分组归一化，增强正则化效果
            num_levels=self._config.num_levels,      # 多尺度特征层数: 4（ResNet的4个stage输出）
            num_cams=len(config.cams),               # 相机数量: 3（左、前、右）
            num_pts=self._config.len_path,           # 每条路径的点数: 50
            attn_drop=0.0,             # 注意力dropout: 0.0（不使用）
            use_deformable_func=config.use_deformable_func,  # 是否使用可变形注意力函数
            use_camera_embed=True,     # 是否使用相机嵌入: True（区分不同相机视角）
            residual_mode="add",       # 残差连接方式: "add"（加性残差）
        )
        
        # 多头自注意力：捕捉路径之间的关系和依赖
        # config.d_model: 256, config.num_head: 8 → 每个头维度: 32
        self.p_attention = nn.MultiheadAttention(
                config.d_model,         # 输入维度: 256
                config.num_head,        # 注意力头数: 8
                dropout=config.dropout, # dropout率: 0.0
                batch_first=True,       # 输入格式: [batch, seq_len, dim]
            )
        
        # 前馈网络（FFN）：非线性变换，增强表达能力
        # 结构: Linear(256→1024) → ReLU → Linear(1024→256)
        self.p_ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ffn),  # 升维到1024
            nn.ReLU(),                                # 非线性激活
            nn.Linear(config.d_ffn, config.d_model),  # 降维回256
        )
        
        # 层归一化和Dropout：稳定训练，防止过拟合
        self.p_norm1 = nn.LayerNorm(config.d_model)   # 自注意力后归一化
        self.p_dropout1 = nn.Dropout(0.1)            # 自注意力后dropout
        self.p_norm2 = nn.LayerNorm(config.d_model)   # FFN后归一化
        self.p_dropout2 = nn.Dropout(0.1)            # FFN后dropout
        
        # 路径评分头：将路径嵌入映射为标量评分，用于Top-K筛选
        # 结构: Linear(256→1024) → ReLU → Linear(1024→1)
        self.path_mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, 1),
        )

        # ==================== 速度分支 (Velocity Branch) ====================
        # 图像注意力：速度嵌入与图像特征交互，让速度感知场景信息
        # query: vel_embed [B, K_vel, 256]
        # key/value: img_value [B, num_cams*H*W/1024, 512]（来自ResNet最后一层）
        self.v_img_attention = nn.MultiheadAttention(
            config.d_model,         # query维度: 256
            config.num_head,        # 注意力头数: 8
            dropout=config.dropout, # dropout率: 0.0
            batch_first=True,       # 输入格式: [batch, seq_len, dim]
        )
        
        # 速度自注意力：捕捉不同速度模板之间的关系
        # 例如：低速和高速模板的关联性
        self.v_attention = nn.MultiheadAttention(
                config.d_model,         # 输入维度: 256
                config.num_head,        # 注意力头数: 8
                dropout=config.dropout, # dropout率: 0.0
                batch_first=True,       # 输入格式: [batch, seq_len, dim]
            )
        
        # 前馈网络：非线性变换
        # 结构: Linear(256→1024) → ReLU → Linear(1024→256)
        self.v_ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ffn),
            nn.ReLU(),
            nn.Linear(config.d_ffn, config.d_model),
        )
        
        # 层归一化和Dropout
        self.v_norm1 = nn.LayerNorm(config.d_model)
        self.v_dropout1 = nn.Dropout(0.1)
        self.v_norm2 = nn.LayerNorm(config.d_model)
        self.v_dropout2 = nn.Dropout(0.1)
        
        # 速度评分头：将速度嵌入映射为标量评分
        # 结构: Linear(256→1024) → ReLU → Linear(1024→1)
        self.vel_mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, 1),
        )

        # ==================== 轨迹分支 (Trajectory Branch) - 仅最后一层 ====================
        # 仅在最后一层创建，因为轨迹是路径和速度的组合结果
        # 当 decoder_idx == decoder_num_layers - 1（即decoder_idx=1，第二层）时创建
        if self.decoder_idx == self._config.decoder_num_layers - 1:
            # 轨迹可变形特征聚合：与路径分支类似，但针对轨迹点
            # num_pts=num_poses=8（轨迹姿态数量）
            self.t_deform_model = DeformableFeatureAggregation(
                config=config,              # 配置对象
                embed_dims=d_model,         # 嵌入维度: 256
                num_groups=8,               # 分组数: 8
                num_levels=self._config.num_levels,  # 多尺度特征层数: 4
                num_cams=len(config.cams),           # 相机数量: 3
                num_pts=num_poses,                   # 轨迹姿态数量: 8
                attn_drop=0.0,
                use_deformable_func=config.use_deformable_func,
                use_camera_embed=True,
                residual_mode="add",
            )
            
            # 轨迹自注意力：捕捉不同轨迹之间的关系
            # 输入: traj_emed [B, 200, 256]（200=20×10，筛选后的路径×速度）
            self.t_attention = nn.MultiheadAttention(
                    config.d_model,         # 输入维度: 256
                    config.num_head,        # 注意力头数: 8
                    dropout=config.dropout, # dropout率: 0.0
                    batch_first=True,       # 输入格式: [batch, seq_len, dim]
                )
            
            # 轨迹前馈网络：非线性变换
            self.t_ffn = nn.Sequential(
                nn.Linear(config.d_model, config.d_ffn),
                nn.ReLU(),
                nn.Linear(config.d_ffn, config.d_model),
            )
            
            # 层归一化和Dropout
            self.t_norm1 = nn.LayerNorm(config.d_model)
            self.t_dropout1 = nn.Dropout(0.1)
            self.t_norm2 = nn.LayerNorm(config.d_model)
            self.t_dropout2 = nn.Dropout(0.1)
            
            # 轨迹评分头：用于对比学习损失
            # 结构: Linear(256→1024) → ReLU → Linear(1024→1)
            self.traj_mlp = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            )
            
            # PDM指标预测头：每个安全/合规/效率/舒适性指标一个独立的MLP
            # metrics列表来自config，NAVSIMv2包含8个指标：
            # ["no_at_fault_collisions", "drivable_area_compliance", 
            #  "driving_direction_compliance", "traffic_light_compliance",
            #  "time_to_collision_within_bound", "ego_progress", 
            #  "lane_keeping", "history_comfort"]
            self.metric_heads = nn.ModuleDict()
            for metric in self._config.metrics:
                self.metric_heads[metric] = nn.Sequential(
                    nn.Linear(d_model, d_ffn),  # 256 → 1024
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),        # 1024 → 1（输出logit）
                )

    def forward(self, path_embed, vel_embed, path_vocab, vel_vocab, traj_vocab, traj_mask,
                camera_feature, status_encoding, targets,
    ):
        """
        解码器层前向传播
        
        参数维度与具体涵义：
        
        【输入特征】
        - path_embed: [B, K_path, d_model] 
          * B: batch size（批量大小），典型值: 8/16
          * K_path: 路径词汇数量，默认值: 1024
          * d_model: 模型维度，默认值: 256
          * 涵义: 每个路径词汇的初始嵌入表示
        
        - vel_embed: [B, K_vel, d_model]
          * K_vel: 速度词汇数量，默认值: 256
          * 涵义: 每个速度词汇的初始嵌入表示
        
        - path_vocab: [B, K_path, len_path, 3]
          * len_path: 每条路径的点数，默认值: 50
          * 最后一维: (x, y, heading) - 路径点坐标和朝向
          * 涵义: 预定义的路径模板词汇表，通过K-means从训练数据聚类得到
        
        - vel_vocab: [B, K_vel, len_vel]
          * len_vel: 速度序列长度，默认值: 8（对应4秒，每0.5秒一个点）
          * 涵义: 预定义的速度模板词汇表，通过K-means从训练数据聚类得到
        
        - traj_vocab: [B, K_path, K_vel, num_poses, 3]
          * num_poses: 轨迹姿态数量，默认值: 8
          * 涵义: 路径和速度的笛卡尔积，共1024×256=262,144种组合
        
        - traj_mask: [B, K_path, K_vel, num_poses]
          * 涵义: 轨迹有效性掩码，标记哪些姿态点有效
        
        - camera_feature: Dict[str, Tensor]
          * "feature_maps": List[Tensor]，4个尺度的特征图
            - level 0: [B, C=64, H/4, W/4]
            - level 1: [B, C=128, H/8, W/8]
            - level 2: [B, C=256, H/16, W/16]
            - level 3: [B, C=512, H/32, W/32]
          * "lidar2img": [B, num_cams, 4, 4] - LiDAR到图像投影矩阵
          * "cam_intrinsic": [B, num_cams, 3, 3] - 相机内参矩阵
          * num_cams: 相机数量，默认值: 3（左、前、右）
        
        - status_encoding: [B, d_model]
          * 涵义: Ego车辆状态编码（4维驾驶命令 + 2维速度 + 2维加速度 → 8维 → 编码到256维）
        
        - targets: Dict[str, Tensor]（训练时使用）
          * "path": [B, len_path, 3] - 目标路径
          * "path_mask": [B, len_path] - 路径掩码
          * "velocity": [B, len_vel] - 目标速度序列
          * "trajectory": [B, num_poses, 3] - 目标轨迹
          * "token_path": List[str] - metric缓存文件路径列表
        
        返回：
        - 特征元组: (filter_path_embed, filter_vel_embed, filter_path_vocab, filter_vel_vocab, filter_traj_vocab, filter_traj_mask)
          * 经过Top-K筛选后的嵌入和词汇表
          * 第一层筛选后: path=128, vel=64; 第二层筛选后: path=20, vel=10
        - output: Dict - 输出字典，包含预测轨迹
        - loss_dict: Dict - 损失字典，包含各分支损失
        """
        num_path = path_embed.shape[1]  # 路径词汇数量: 1024（第一层）/ 128（第二层）
        num_vel = vel_embed.shape[1]    # 速度词汇数量: 256（第一层）/ 64（第二层）

        # 准备图像特征：取最后一层特征图（最大感受野），调整维度
        # [B, num_cams, C=512, H/32, W/32] → [B, num_cams*H*W/1024, 512]
        img_value = camera_feature["feature_maps"][-1].permute(0, 1, 3, 4, 2).flatten(1, 3)
        # 将特征图转换为可变形注意力格式：[B, num_cams, num_levels, C, H, W]
        deform_value = deformable_format(camera_feature["feature_maps"])

        # ==================== Ego状态注入 ====================
        # 将Ego状态编码广播后添加到每个路径/速度嵌入中
        # status_encoding: [B, d_model] → [B, 1, d_model] → 广播到 [B, K_path/K_vel, d_model]意思是把
        path_embed = path_embed + status_encoding.unsqueeze(1)
        vel_embed = vel_embed + status_encoding.unsqueeze(1)

        # ==================== 路径分支处理 ====================
        #path_vocab  [B, K_path, len_path, 3]
        # 将路径坐标展平：[B, K_path, len_path, 2] → [B, K_path, len_path*2=100]
        # 只取x, y坐标，忽略heading用于投影 取最后一个维度的前两个元素（x, y坐标，忽略heading）
        path_vocab_flat = path_vocab[..., :2].flatten(-2) #从 倒数第二个维度 开始展平，到最后一个维度结束
        # 可变形特征聚合：将3D路径点投影到图像平面，聚合多视角多尺度特征
        # 输入: path_embed [B, K_path, 256], path_vocab_flat [B, K_path, 100]
        # 输出: path_embed [B, K_path, 256]（更新后的嵌入）
        path_embed = self.p_deform_model(
            path_embed,           # 查询嵌入
            path_vocab_flat,      # 路径点坐标（用于投影到图像平面）
            None,                 # key padding mask
            deform_value,         # 多尺度特征图
            camera_feature,       # 相机参数（内外参，用于投影计算）
            None,                 # attn mask
        )
        # 多头自注意力 + 残差连接：捕捉路径之间的关系
        path_embed = path_embed + self.p_dropout1(self.p_attention(path_embed, path_embed, path_embed)[0])
        path_embed = self.p_norm1(path_embed)
        # FFN + 残差连接：非线性变换
        # 前馈层（FFN）在Transformer模型中的作用是对每个位置的词向量进行独立的非线性变换。
        # 尽管注意力机制能够捕捉序列中的全局依赖，但前馈层通过增加模型的深度和复杂度，为模型引入必要的非线性，
        # 从而增强模型的表达能力。每个编码器和解码器层都包含一个FFN，它对所有位置的表示进行相同的操作，但并不共享参数。
        path_embed = path_embed + self.p_dropout2(self.p_ffn(path_embed))
        path_embed = self.p_norm2(path_embed)
        # 路径评分：[B, K_path, 256] → [B, K_path, 1] → [B, K_path] # 移除最后一个维度，总大小不变，方便torch.topk筛选
        # 每个路径词汇的得分，用于后续Top-K筛选
        path_scores = self.path_mlp(path_embed).squeeze(-1)

        # ==================== 速度分支处理 ====================
        # 图像注意力：速度嵌入与图像特征交互，获取场景信息
        # query: vel_embed [B, K_vel, 256]
        # key/value: img_value [B, num_cams*H*W/1024, 512]
        vel_embed = vel_embed + self.v_img_attention(vel_embed, img_value, img_value)[0]
        # 速度自注意力 + 残差连接：捕捉速度之间的关系
        vel_embed = vel_embed + self.v_dropout1(self.v_attention(vel_embed, vel_embed, vel_embed)[0])
        vel_embed = self.v_norm1(vel_embed)
        # FFN + 残差连接
        vel_embed = vel_embed + self.v_dropout2(self.v_ffn(vel_embed))
        vel_embed = self.v_norm2(vel_embed)
        # 速度评分：[B, K_vel]
        vel_scores = self.vel_mlp(vel_embed).squeeze(-1)

        # ==================== 粗粒度筛选 ====================
        # 初始化筛选后的轨迹词汇表和掩码（深拷贝）
        filter_traj_vocab = traj_vocab.clone()
        filter_traj_mask = traj_mask.clone()

        # Top-K筛选路径：根据评分保留Top-K个路径
        # 第一层: 1024 → 128; 第二层: 128 → 20
        if num_path > self._config.path_filter_num[self.decoder_idx]:
            topk_path_scores, topk_path_indices = torch.topk(path_scores, self._config.path_filter_num[self.decoder_idx], dim=1)
            # 筛选路径嵌入: [B, K_path, 256] → [B, K_path_filter, 256]
            filter_path_embed = torch.gather(path_embed, 1, topk_path_indices.unsqueeze(-1).expand(-1, -1, path_embed.shape[-1]))
            # 筛选路径词汇: [B, K_path, 50, 3] → [B, K_path_filter, 50, 3]
            filter_path_vocab = torch.gather(path_vocab, 1, topk_path_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, path_vocab.shape[-2], path_vocab.shape[-1]))
            # 筛选轨迹词汇: [B, K_path, K_vel, 8, 3] → [B, K_path_filter, K_vel, 8, 3]
            filter_traj_vocab = torch.gather(filter_traj_vocab, 1, topk_path_indices[:, :, None, None, None].expand(-1, -1, filter_traj_vocab.shape[-3], filter_traj_vocab.shape[-2], filter_traj_vocab.shape[-1]))
            filter_traj_mask = torch.gather(filter_traj_mask, 1, topk_path_indices[:, :, None, None].expand(-1, -1, filter_traj_mask.shape[-2], filter_traj_mask.shape[-1]))
        else:
            filter_path_embed = path_embed
            filter_path_vocab = path_vocab

        # Top-K筛选速度：根据评分保留Top-K个速度
        # 第一层: 256 → 64; 第二层: 64 → 10
        if num_vel > self._config.velocity_filter_num[self.decoder_idx]:
            topk_vel_scores, topk_vel_indices = torch.topk(vel_scores, self._config.velocity_filter_num[self.decoder_idx], dim=1)
            filter_vel_embed = torch.gather(vel_embed, 1, topk_vel_indices.unsqueeze(-1).expand(-1, -1, vel_embed.shape[-1]))
            filter_vel_vocab = torch.gather(vel_vocab, 1, topk_vel_indices.unsqueeze(-1).expand(-1, -1, vel_vocab.shape[-1]))
            filter_traj_vocab = torch.gather(filter_traj_vocab, 2, topk_vel_indices[:, None, :, None, None].expand(-1, filter_traj_vocab.shape[-4], -1, filter_traj_vocab.shape[-2], filter_traj_vocab.shape[-1]))
            filter_traj_mask = torch.gather(filter_traj_mask, 2, topk_vel_indices[:, None, :, None].expand(-1, filter_traj_mask.shape[-3], -1, filter_traj_mask.shape[-1]))
        else:
            filter_vel_embed = vel_embed
            filter_vel_vocab = vel_vocab

        # ==================== 轨迹重构（仅最后一层） ====================
        if self.decoder_idx == self._config.decoder_num_layers - 1:
            # 组合路径和速度嵌入：外积相加
            # filter_path_embed: [B, 20, 256] → [B, 20, 1, 256]
            # filter_vel_embed: [B, 10, 256] → [B, 1, 10, 256]
            # 相加后: [B, 20, 10, 256] → 展平为 [B, 200, 256]
            traj_emed = filter_path_embed.unsqueeze(2) + filter_vel_embed.unsqueeze(1)
            # 展平为：[B, K_path*K_vel=200, d_model=256]
            traj_emed = traj_emed.flatten(1, 2)

            # 轨迹坐标展平：[B, 20, 10, 8, 2] → [B, 200, 16]
            # 只取x, y坐标用于投影
            filter_traj_vocab_flat = filter_traj_vocab[..., :2].flatten(1, 2).flatten(-2)
            # 可变形特征聚合：将轨迹点投影到图像平面，聚合视觉特征
            traj_emed = self.t_deform_model(
                traj_emed,
                filter_traj_vocab_flat,
                None,
                deform_value,
                camera_feature,
                None,
            )
            # 自注意力 + 残差：捕捉轨迹之间的关系
            traj_emed = traj_emed + self.t_dropout1(self.t_attention(traj_emed, traj_emed, traj_emed)[0])
            traj_emed = self.t_norm1(traj_emed)
            # FFN + 残差
            traj_emed = traj_emed + self.t_dropout2(self.t_ffn(traj_emed))
            traj_emed = self.t_norm2(traj_emed)
            # 轨迹评分：[B, 200, 256] → [B, 200, 1] → [B, 200]
            traj_scores = self.traj_mlp(traj_emed).squeeze(-1)
            # PDM指标预测：每个指标独立预测
            # metrics: ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", 
            #           "traffic_light_compliance", "time_to_collision_within_bound", "ego_progress", 
            #           "lane_keeping", "history_comfort"]
            metric_logit = {}
            for metric in self._config.metrics:
                metric_logit[metric] = self.metric_heads[metric](traj_emed).squeeze(-1)  # [B, 200]

        # ==================== 损失计算（训练模式） ====================
        loss_dict = {}
        if self.training:
            # ========== 路径损失：对比学习 ==========
            # 目标：让模型学习选择与真实路径最接近的路径词汇
            target_path = targets["path"]           # [B, 50, 3]
            target_path_mask = targets["path_mask"] # [B, 50]
            
            # 计算每个路径词汇与目标路径的距离（仅x, y）
            # [B, K_path, 50, 3] - [B, 1, 50, 3] → [B, K_path, 50, 2]
            diff = (path_vocab - target_path[:, None])[..., :2]
            dist = diff.pow(2).sum(-1)  # [B, K_path, 50] - 逐点距离
            mask = target_path_mask[:, None].float()  # [B, 1, 50]
            dist = dist * mask  # 只计算有效点的距离
            
            valid_cnt = mask.sum(-1).clamp(min=1.0)  # 有效点数
            dist = dist.sum(-1) / valid_cnt  # 平均距离 [B, K_path]
            dist = dist * self._config.path_sigmas * self._config.len_path  # 缩放
            
            # 对比学习损失：将距离转换为伪标签
            # (-dist).softmax(1): 距离越小，伪标签概率越大
            path_loss = F.cross_entropy(path_scores, (-dist).softmax(1))
            loss_dict[f'path_loss_{self.decoder_idx}'] = path_loss

            # ========== 速度损失：对比学习 ==========
            target_vel = targets["velocity"]  # [B, 8]
            # 计算每个速度词汇与目标速度的L1距离
            dist = (vel_vocab - target_vel[:, None]).abs()  # [B, K_vel, 8]
            dist = dist.sum(-1) * self._config.velocity_sigmas  # [B, K_vel]
            vel_loss = F.cross_entropy(vel_scores, (-dist).softmax(1))
            loss_dict[f'velocity_loss_{self.decoder_idx}'] = vel_loss

            # ========== 轨迹损失（仅最后一层） ==========
            if self.decoder_idx == self._config.decoder_num_layers - 1:
                # 轨迹对比损失
                target_traj = targets["trajectory"]  # [B, 8, 3]
                # [B, 200, 8, 2] - [B, 1, 8, 2] → [B, 200, 8, 2]
                dist = (filter_traj_vocab.flatten(1, 2) - target_traj[:, None])[..., :2] ** 2
                dist = dist.sum((-2, -1)) * self._config.trajectory_sigmas  # [B, 200]
                traj_loss = F.cross_entropy(traj_scores, (-dist).softmax(1))
                loss_dict[f'traj_loss_{self.decoder_idx}'] = traj_loss

                # ========== PDM指标损失 ==========
                trajectory = filter_traj_vocab.flatten(1,2)  # [B, 200, 8, 3]
                
                # 构建metric缓存文件路径
                # 原始路径: exp/data_cache_navtrain/{log_name}/{token}/sparsedrive_target.gz
                # metric路径: exp/metric_cache_navtrain{v1/v2}/{log_name}/unknown/{token}/metric_cache.pkl
                pdm_token_paths = []
                for token_path in targets["token_path"]:
                    pdm_token_path = token_path.replace("data_cache_navtrain", f"metric_cache_navtrain{self._config.dataset_version}")
                    pdm_token_path_parts = pdm_token_path.split('/')
                    pdm_token_path_parts.insert(-1, 'unknown')
                    pdm_token_path = '/'.join(pdm_token_path_parts) + "/metric_cache.pkl"
                    pdm_token_paths.append(pdm_token_path)
                
                # 根据数据集版本选择评分函数
                if self._config.dataset_version == "v1":
                    sub_scores = get_pdm_score_v1(trajectory, pdm_token_paths)
                elif self._config.dataset_version == "v2":
                    sub_scores = get_pdm_score_v2(trajectory, pdm_token_paths)
                
                # 计算每个指标的二分类损失
                for metric in self._config.metrics:
                    metric_pred = metric_logit[metric]  # [B, 200] - logits
                    metric_gt = torch.tensor(np.stack([sub_score[metric] for sub_score in sub_scores])).to(metric_pred)  # [B, 200]
                    metric_gt[metric_gt == 0.5] = 0.0  # 处理不确定标签（0.5表示不确定，视为负样本）
                    metric_loss = F.binary_cross_entropy_with_logits(metric_pred, metric_gt)
                    loss_dict[f'{metric}_loss_{self.decoder_idx}'] = metric_loss * self._config.metric_loss_weight
        
        # ==================== 输出（仅最后一层） ====================
        output = {}
        if self.decoder_idx == self._config.decoder_num_layers - 1:
            # ========== PDM指标评分计算 ==========
            # 根据NAVSIM评分公式计算每个候选轨迹的最终评分
            # 评分公式: 安全性指标的乘积 × (效率指标加权和)
            # 所有指标经过sigmoid激活到[0,1]区间
            
            if self._config.dataset_version == "v1":
                # NAVSIMv1评分公式
                scores = (
                    metric_logit["no_at_fault_collisions"].sigmoid() *  # 无责任碰撞 (安全)
                    metric_logit["drivable_area_compliance"].sigmoid()  # 可行驶区域遵守 (合规)
                ) * (
                    5 * metric_logit["time_to_collision_within_bound"].sigmoid() +  # 碰撞时间边界 (安全)
                    5 * metric_logit["ego_progress"].sigmoid()  +                     # 自车进度 (效率)
                    2 * metric_logit["comfort"].sigmoid()                              # 舒适性 (舒适)
                )
            if self._config.dataset_version == "v2":
                # NAVSIMv2评分公式（更严格的安全要求）
                scores = (
                    metric_logit["no_at_fault_collisions"].sigmoid() *        # 无责任碰撞 (安全)
                    metric_logit["drivable_area_compliance"].sigmoid() *     # 可行驶区域遵守 (合规)
                    metric_logit["driving_direction_compliance"].sigmoid() *  # 行驶方向遵守 (合规)
                    metric_logit["traffic_light_compliance"].sigmoid()       # 交通灯遵守 (合规)
                ) * (
                    5 * metric_logit["time_to_collision_within_bound"].sigmoid() +  # 碰撞时间边界 (安全)
                    5 * metric_logit["ego_progress"].sigmoid()  +                     # 自车进度 (效率)
                    2 * metric_logit["lane_keeping"].sigmoid() +                      # 车道保持 (舒适)
                    2 * metric_logit["history_comfort"].sigmoid()                     # 历史舒适性 (舒适)
                )
            # scores: [B, 200] - 每个候选轨迹的最终评分

            # ========== 选择最优轨迹 ==========
            # 选择评分最高的轨迹作为最终输出
            bs_indices = torch.arange(scores.shape[0], device=scores.device)  # [0, 1, ..., B-1]
            mode_indices = scores.argmax(1)
            trajectory = filter_traj_vocab.flatten(1, 2)[bs_indices, mode_indices] 
            output["trajectory"] = trajectory

        # 返回筛选后的嵌入、词汇表和输出、损失
        return (filter_path_embed, filter_vel_embed, filter_path_vocab, filter_vel_vocab, filter_traj_vocab, filter_traj_mask), output, loss_dict




