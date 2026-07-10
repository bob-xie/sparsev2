import os
from pathlib import Path

project_root = Path(__file__).parent
os.environ['OPENSCENE_DATA_ROOT'] = str(project_root)
os.environ['NAVSIM_DEVKIT_ROOT'] = str(project_root)
os.environ['NAVSIM_EXP_ROOT'] = str(project_root / 'exp')
os.environ['NUPLAN_MAPS_ROOT'] = str(project_root / 'maps/nuplan-maps-v1.0')

import torch
import inspect

from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel


def find_param_source(param):
    """查找参数的定义位置"""
    if hasattr(param, 'grad_fn'):
        return "计算图中间变量"
    
    try:
        # 获取参数所属模块
        module = param.owner()
        
        # 获取模块的类定义文件
        class_file = inspect.getsourcefile(type(module))
        
        # 获取模块在类中的属性名
        module_name = None
        for name, obj in type(module).__bases__[0].__dict__.items():
            if isinstance(obj, torch.nn.Module) or (isinstance(obj, property) and isinstance(obj.fget(module), torch.nn.Module)):
                pass
        
        return class_file
    except Exception as e:
        return f"未知: {e}"


def find_param_in_module(module, param_name_parts, current_path=""):
    """递归查找参数在模块中的位置"""
    for name, child in module.named_children():
        child_path = f"{current_path}.{name}" if current_path else name
        
        # 检查当前模块是否有这个参数
        for param_name, param in child.named_parameters(recurse=False):
            if param_name == param_name_parts[-1]:
                class_file = inspect.getsourcefile(type(child))
                try:
                    source, start_line = inspect.getsourcelines(type(child))
                    for i, line in enumerate(source):
                        if param_name in line and ('=' in line or 'self.' in line):
                            return class_file, start_line + i, type(child).__name__
                except:
                    return class_file, 0, type(child).__name__
        
        # 递归查找子模块
        result = find_param_in_module(child, param_name_parts, child_path)
        if result:
            return result
    
    return None


def find_all_param_locations(model):
    """查找模型中所有参数的定义位置"""
    param_locations = {}
    
    for name, param in model.named_parameters():
        param_name_parts = name.split('.')
        
        # 查找参数位置
        result = find_param_in_module(model, param_name_parts)
        
        if result:
            class_file, param_lineno, module_class = result
        else:
            class_file = "未知"
            param_lineno = 0
            module_class = "未知"
        
        param_locations[name] = {
            'file': class_file,
            'param_lineno': param_lineno,
            'shape': list(param.shape),
            'num_params': param.numel(),
            'trainable': param.requires_grad,
            'module_class': module_class,
        }
    
    return param_locations


def main():
    config = SparseDriveConfig()
    model = SparseDriveModel(config)
    
    param_locations = find_all_param_locations(model)
    
    print("=" * 120)
    print("模型参数定义位置查找")
    print("=" * 120)
    print(f"{'参数名称':<80} {'文件路径':<60} {'行号':>10} {'形状':<20} {'参数量':>12}")
    print("-" * 120)
    
    for name, info in sorted(param_locations.items(), key=lambda x: -x[1]['num_params']):
        file_path = info['file']
        if len(file_path) > 58:
            file_path = '...' + file_path[-55:]
        
        print(f"{name:<80} {file_path:<60} {info['param_lineno']:>10} {str(info['shape']):<20} {info['num_params']:>12}")
    
    print("\n" + "=" * 120)
    print("按文件汇总")
    print("=" * 120)
    
    file_summary = {}
    for name, info in param_locations.items():
        if info['file'] not in file_summary:
            file_summary[info['file']] = {'total_params': 0, 'param_count': 0}
        file_summary[info['file']]['total_params'] += info['num_params']
        file_summary[info['file']]['param_count'] += 1
    
    for file_path, stats in sorted(file_summary.items(), key=lambda x: -x[1]['total_params']):
        if len(file_path) > 80:
            display_path = '...' + file_path[-77:]
        else:
            display_path = file_path
        print(f"{display_path:<85} 参数数量: {stats['param_count']:>5} 总参数量: {stats['total_params']:,}")


if __name__ == "__main__":
    main()