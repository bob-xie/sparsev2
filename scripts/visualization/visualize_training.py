#!/usr/bin/env python3
"""
训练结果可视化脚本
可视化训练过程中的损失曲线、指标变化

使用方式:
    python scripts/visualization/visualize_training.py --log_dir <lightning_logs目录>

依赖:
    - 需要训练产生的 lightning_logs 目录
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_data(log_dir: str):
    """加载 TensorBoard 日志数据"""
    event_accumulator = EventAccumulator(log_dir, size_guidance={
        'scalars': 0,
        'histograms': 0,
        'images': 0,
        'audio': 0,
        'tensors': 0,
    })
    event_accumulator.Reload()
    return event_accumulator


def plot_loss_curves(event_accumulator, output_path: Path):
    """绘制损失曲线"""
    print("绘制损失曲线...")
    
    tags = event_accumulator.Tags()['scalars']
    loss_tags = [tag for tag in tags if 'loss' in tag.lower()]
    
    if not loss_tags:
        print("未找到损失相关标签")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for tag in loss_tags:
        events = event_accumulator.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        
        ax.plot(steps, values, label=tag)
    
    ax.set_xlabel('训练步数')
    ax.set_ylabel('损失值')
    ax.set_title('训练损失曲线')
    ax.legend()
    ax.grid(True)
    
    fig.savefig(output_path / 'loss_curves.png', dpi=150, bbox_inches='tight')
    print(f"保存损失曲线: {output_path / 'loss_curves.png'}")
    plt.close(fig)


def plot_metric_curves(event_accumulator, output_path: Path):
    """绘制指标曲线"""
    print("绘制指标曲线...")
    
    tags = event_accumulator.Tags()['scalars']
    metric_tags = [tag for tag in tags if 'loss' not in tag.lower()]
    
    if not metric_tags:
        print("未找到指标相关标签")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for tag in metric_tags:
        events = event_accumulator.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        
        ax.plot(steps, values, label=tag)
    
    ax.set_xlabel('训练步数')
    ax.set_ylabel('指标值')
    ax.set_title('训练指标曲线')
    ax.legend()
    ax.grid(True)
    
    fig.savefig(output_path / 'metric_curves.png', dpi=150, bbox_inches='tight')
    print(f"保存指标曲线: {output_path / 'metric_curves.png'}")
    plt.close(fig)


def plot_learning_rate(event_accumulator, output_path: Path):
    """绘制学习率变化曲线"""
    print("绘制学习率曲线...")
    
    tags = event_accumulator.Tags()['scalars']
    lr_tags = [tag for tag in tags if 'lr' in tag.lower() or 'learning_rate' in tag.lower()]
    
    if not lr_tags:
        print("未找到学习率相关标签")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for tag in lr_tags:
        events = event_accumulator.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        
        ax.plot(steps, values, label=tag)
    
    ax.set_xlabel('训练步数')
    ax.set_ylabel('学习率')
    ax.set_title('学习率变化曲线')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    
    fig.savefig(output_path / 'learning_rate.png', dpi=150, bbox_inches='tight')
    print(f"保存学习率曲线: {output_path / 'learning_rate.png'}")
    plt.close(fig)


def main(log_dir: str, output_dir: str = 'exp/visualization/training'):
    """主函数"""
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载 TensorBoard 数据
    print(f"加载 TensorBoard 日志: {log_dir}")
    event_accumulator = load_tensorboard_data(log_dir)
    
    # 绘制各种曲线
    plot_loss_curves(event_accumulator, output_path)
    plot_metric_curves(event_accumulator, output_path)
    plot_learning_rate(event_accumulator, output_path)
    
    print(f"\n所有训练可视化结果已保存到: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练结果可视化')
    parser.add_argument('--log_dir', required=True, help='TensorBoard 日志目录（lightning_logs/version_X）')
    parser.add_argument('--output', default='exp/visualization/training', help='输出目录')
    args = parser.parse_args()
    
    main(args.log_dir, args.output)
