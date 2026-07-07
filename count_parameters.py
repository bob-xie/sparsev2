import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
os.environ['OPENSCENE_DATA_ROOT'] = str(project_root)
os.environ['NAVSIM_DEVKIT_ROOT'] = str(project_root)
os.environ['NAVSIM_EXP_ROOT'] = str(project_root / 'exp')
os.environ['NUPLAN_MAPS_ROOT'] = str(project_root / 'maps/nuplan-maps-v1.0')

import torch

import hydra
from hydra.utils import instantiate

from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel


def count_parameters(model):
    """计算模型参数量，区分可训练参数和固定参数"""
    total_params = 0
    trainable_params = 0
    fixed_params = 0
    param_details = {}
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
        else:
            fixed_params += num_params
        
        module_name = name.split('.')[0]
        if module_name not in param_details:
            param_details[module_name] = {'total': 0, 'trainable': 0, 'fixed': 0}
        param_details[module_name]['total'] += num_params
        if param.requires_grad:
            param_details[module_name]['trainable'] += num_params
        else:
            param_details[module_name]['fixed'] += num_params
    
    return total_params, trainable_params, fixed_params, param_details


def format_params(num):
    """格式化参数量显示"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def main():
    from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
    
    config = SparseDriveConfig()
    model = SparseDriveModel(config)
    
    total, trainable, fixed, details = count_parameters(model)
    
    print("=" * 60)
    print("模型参数量统计")
    print("=" * 60)
    print(f"总参数量:        {format_params(total)} ({total:,})")
    print(f"可训练参数:      {format_params(trainable)} ({trainable:,})")
    print(f"固定参数(不训练): {format_params(fixed)} ({fixed:,})")
    print(f"可训练比例:      {trainable/total*100:.2f}%")
    print()
    print("=" * 60)
    print("各模块参数量明细")
    print("=" * 60)
    
    for module, stats in details.items():
        print(f"\n【{module}】")
        print(f"  总参数:    {format_params(stats['total'])} ({stats['total']:,})")
        print(f"  可训练:    {format_params(stats['trainable'])} ({stats['trainable']:,})")
        print(f"  固定:      {format_params(stats['fixed'])} ({stats['fixed']:,})")
    
    print()
    print("=" * 60)
    print("参数量计算说明")
    print("=" * 60)
    print("1. 固定参数(requires_grad=False): 包括词汇表(anchor)，由K-Means预计算")
    print("2. 可训练参数: ResNet-34骨干、FPN Neck、状态编码器、Transformer Decoder等")
    print("3. 词汇表参数: 路径(1024×50×3) + 速度(256×8) + 轨迹(1024×256×8×3)")


if __name__ == "__main__":
    main()