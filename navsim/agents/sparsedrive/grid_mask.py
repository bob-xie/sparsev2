# 导入 PyTorch 核心库，用于张量操作
import torch

# 导入 PyTorch 神经网络模块
import torch.nn as nn

# 导入 numpy 库，用于数值计算和数组操作
import numpy as np

# 导入 PIL 库，用于图像操作（如旋转）
from PIL import Image


class Grid(object):
    """
    GridMask 数据增强类（普通类版本）
    
    GridMask 是一种基于网格的图像遮挡增强技术，通过随机生成网格状的掩码来遮挡图像的部分区域，
    从而增强模型对局部特征的鲁棒性和泛化能力。
    
    工作原理：
    1. 生成一个比原图大1.5倍的掩码
    2. 在掩码上随机生成水平和垂直的遮挡条带
    3. 随机旋转掩码
    4. 裁剪到原图大小并应用到图像上
    
    应用场景：
    - 自动驾驶视觉模型训练
    - 提高模型对部分遮挡的鲁棒性
    - 增强模型的局部特征学习能力
    """

    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        """
        初始化 GridMask 实例
        
        :param use_h: 是否使用水平遮挡条带
        :param use_w: 是否使用垂直遮挡条带
        :param rotate: 旋转角度数（0到rotate-1度随机选择）
        :param offset: 是否在遮挡区域添加随机偏移（使遮挡更柔和）
        :param ratio: 遮挡条带宽度与网格间隔的比例（0~1）
        :param mode: 模式选择（0=遮挡区域为0，1=遮挡区域为1，其余为0）
        :param prob: 应用此增强的概率
        """
        self.use_h = use_h              # 是否使用水平遮挡
        self.use_w = use_w              # 是否使用垂直遮挡
        self.rotate = rotate            # 旋转角度数
        self.offset = offset            # 是否添加随机偏移
        self.ratio = ratio              # 遮挡条带比例
        self.mode = mode                # 模式（0或1）
        self.st_prob = prob             # 初始概率
        self.prob = prob                # 当前概率（可随训练进度调整）

    def set_prob(self, epoch, max_epoch):
        """
        设置增强概率（随训练进度线性增加）
        
        :param epoch: 当前训练轮数
        :param max_epoch: 最大训练轮数
        """
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img, label):
        """
        对图像应用 GridMask 增强
        
        :param img: 输入图像张量，形状为 [C, H, W]
        :param label: 图像标签（保持不变）
        :return: (增强后的图像, 标签)
        """
        # 根据概率决定是否应用增强
        if np.random.rand() > self.prob:
            return img, label
        
        # 获取图像尺寸
        h = img.size(1)  # 高度
        w = img.size(2)  # 宽度
        
        # 设置网格参数
        self.d1 = 2               # 最小网格间隔
        self.d2 = min(h, w)       # 最大网格间隔
        hh = int(1.5 * h)        # 掩码高度（比原图大1.5倍）
        ww = int(1.5 * w)        # 掩码宽度（比原图大1.5倍）
        
        # 随机选择网格间隔 d（在 [d1, d2) 范围内）
        d = np.random.randint(self.d1, self.d2)
        
        # 计算遮挡条带宽度 l
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        
        # 创建全1掩码（1表示保留，0表示遮挡）
        mask = np.ones((hh, ww), np.float32)
        
        # 随机选择起始位置
        st_h = np.random.randint(d)  # 水平起始位置
        st_w = np.random.randint(d)  # 垂直起始位置
        
        # 添加水平遮挡条带
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h        # 条带起始位置
                t = min(s + self.l, hh)  # 条带结束位置
                mask[s:t, :] *= 0       # 设置为0（遮挡）
        
        # 添加垂直遮挡条带
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w        # 条带起始位置
                t = min(s + self.l, ww)  # 条带结束位置
                mask[:, s:t] *= 0       # 设置为0（遮挡）

        # 随机旋转掩码
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask * 255))  # 转换为PIL图像
        mask = mask.rotate(r)                          # 旋转
        mask = np.asarray(mask) / 255.0               # 转换回numpy数组
        
        # 裁剪到原图大小（从中心裁剪）
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,    # 高度方向裁剪
            (ww - w) // 2 : (ww - w) // 2 + w,    # 宽度方向裁剪
        ]

        # 转换为PyTorch张量
        mask = torch.from_numpy(mask).float()
        
        # 如果是模式1，反转掩码（遮挡区域变为保留区域）
        if self.mode == 1:
            mask = 1 - mask

        # 将掩码扩展到与图像相同的形状
        mask = mask.expand_as(img)
        
        # 根据offset参数决定是否添加随机偏移
        if self.offset:
            # 在遮挡区域添加随机偏移（-1到1之间）
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset  # 只在遮挡区域添加偏移
            img = img * mask + offset      # 应用掩码和偏移
        else:
            # 直接应用掩码
            img = img * mask

        return img, label


class GridMask(nn.Module):
    """
    GridMask 数据增强类（PyTorch nn.Module 版本）
    
    与 Grid 类功能相同，但继承自 nn.Module，可直接作为模型的一部分使用，
    支持 GPU 加速和训练/评估模式切换。
    """

    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        """
        初始化 GridMask nn.Module
        
        :param use_h: 是否使用水平遮挡条带
        :param use_w: 是否使用垂直遮挡条带
        :param rotate: 旋转角度数
        :param offset: 是否添加随机偏移
        :param ratio: 遮挡条带比例（默认0.5）
        :param mode: 模式（0=遮挡，1=反转）
        :param prob: 应用概率（默认1.0）
        """
        super(GridMask, self).__init__()
        self.use_h = use_h              # 是否使用水平遮挡
        self.use_w = use_w              # 是否使用垂直遮挡
        self.rotate = rotate            # 旋转角度数
        self.offset = offset            # 是否添加随机偏移
        self.ratio = ratio              # 遮挡条带比例
        self.mode = mode                # 模式
        self.st_prob = prob             # 初始概率
        self.prob = prob                # 当前概率

    def set_prob(self, epoch, max_epoch):
        """
        设置增强概率（随训练进度线性增加）
        
        :param epoch: 当前训练轮数
        :param max_epoch: 最大训练轮数
        """
        self.prob = self.st_prob * epoch / max_epoch

    def forward(self, x):
        """
        前向传播：对批量图像应用 GridMask 增强
        
        :param x: 输入图像张量，形状为 [N, C, H, W]
        :return: 增强后的图像张量，形状为 [N, C, H, W]
        """
        # 训练模式下才应用增强，且根据概率决定是否应用
        if np.random.rand() > self.prob or not self.training:
            return x
        
        # 获取批量大小和图像尺寸
        n, c, h, w = x.size()
        
        # 将图像展平为 [N×C, H, W]，便于统一处理
        x = x.view(-1, h, w)
        
        # 创建比原图大1.5倍的掩码
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        
        # 随机选择网格间隔
        d = np.random.randint(2, h)
        
        # 计算遮挡条带宽度
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        
        # 创建全1掩码
        mask = np.ones((hh, ww), np.float32)
        
        # 随机选择起始位置
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        
        # 添加水平遮挡条带
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        
        # 添加垂直遮挡条带
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        # 随机旋转掩码
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask * 255))
        mask = mask.rotate(r)
        mask = np.asarray(mask) / 255.0
        
        # 裁剪到原图大小
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        # 转换为PyTorch张量并移动到GPU
        mask = torch.from_numpy(mask.copy()).float().cuda()
        
        # 如果是模式1，反转掩码
        if self.mode == 1:
            mask = 1 - mask
        
        # 扩展掩码到与图像相同的形状
        mask = mask.expand_as(x)
        
        # 根据offset参数决定是否添加随机偏移
        if self.offset:
            offset = (
                torch.from_numpy(2 * (np.random.rand(h, w) - 0.5))
                .float()
                .cuda()
            )
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        # 恢复原始形状 [N, C, H, W]
        return x.view(n, c, h, w)