import os
from pathlib import Path

project_root = Path(__file__).parent
os.environ['OPENSCENE_DATA_ROOT'] = str(project_root)
os.environ['NAVSIM_DEVKIT_ROOT'] = str(project_root)
os.environ['NAVSIM_EXP_ROOT'] = str(project_root / 'exp')
os.environ['NUPLAN_MAPS_ROOT'] = str(project_root / 'maps/nuplan-maps-v1.0')

import torch

from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel


def format_params(num):
    if num >= 1e9:
        return f"{num/1e9:.4f}B"
    elif num >= 1e6:
        return f"{num/1e6:.4f}M"
    elif num >= 1e3:
        return f"{num/1e3:.4f}K"
    return str(num)


def count_module_params(module, prefix="", indent=0):
    total = 0
    trainable = 0
    fixed = 0
    details = []
    
    prefix_str = "  " * indent
    
    if isinstance(module, torch.nn.Module):
        for name, param in module.named_parameters(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            num_params = param.numel()
            total += num_params
            if param.requires_grad:
                trainable += num_params
            else:
                fixed += num_params
            
            details.append({
                'name': full_name,
                'shape': list(param.shape),
                'num_params': num_params,
                'trainable': param.requires_grad,
                'indent': indent + 1,
            })
        
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            child_total, child_trainable, child_fixed, child_details = count_module_params(child, child_prefix, indent + 1)
            total += child_total
            trainable += child_trainable
            fixed += child_fixed
            details.extend(child_details)
    
    return total, trainable, fixed, details


def main():
    config = SparseDriveConfig()
    model = SparseDriveModel(config)
    
    total, trainable, fixed, details = count_module_params(model)
    
    print("=" * 80)
    print("SparseDriveV2 模型参数量详细统计")
    print("=" * 80)
    print(f"总参数量:        {format_params(total)} ({total:,})")
    print(f"可训练参数:      {format_params(trainable)} ({trainable:,})")
    print(f"固定参数(不训练): {format_params(fixed)} ({fixed:,})")
    print(f"可训练比例:      {trainable/total*100:.4f}%")
    print()
    
    print("=" * 80)
    print("参数量分布详细清单")
    print("=" * 80)
    print(f"{'模块名称':<70} {'参数量':>12} {'类型':>8} {'形状'}")
    print("-" * 100)
    
    for item in sorted(details, key=lambda x: -x['num_params']):
        type_str = "可训练" if item['trainable'] else "固定"
        shape_str = str(item['shape'])
        indent_str = "  " * item['indent']
        name_display = indent_str + item['name']
        if len(name_display) > 70:
            name_display = name_display[:67] + "..."
        
        print(f"{name_display:<70} {format_params(item['num_params']):>12} {type_str:>8} {shape_str}")
    
    print()
    print("=" * 80)
    print("模块汇总统计")
    print("=" * 80)
    
    module_summary = {}
    for item in details:
        parts = item['name'].split('.')
        module_name = parts[0]
        if module_name not in module_summary:
            module_summary[module_name] = {'total': 0, 'trainable': 0, 'fixed': 0}
        module_summary[module_name]['total'] += item['num_params']
        if item['trainable']:
            module_summary[module_name]['trainable'] += item['num_params']
        else:
            module_summary[module_name]['fixed'] += item['num_params']
    
    for module_name, stats in sorted(module_summary.items(), key=lambda x: -x[1]['total']):
        print(f"\n【{module_name}】")
        print(f"  总参数:    {format_params(stats['total'])} ({stats['total']:,})")
        print(f"  可训练:    {format_params(stats['trainable'])} ({stats['trainable']:,})")
        print(f"  固定:      {format_params(stats['fixed'])} ({stats['fixed']:,})")


if __name__ == "__main__":
    main()