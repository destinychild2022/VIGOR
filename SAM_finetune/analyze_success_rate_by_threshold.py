#!/usr/bin/env python3
"""
基于vigor_success_rate_results.txt文件，计算不同IoU阈值下的成功率
"""

import re
from pathlib import Path


def parse_results_file(file_path):
    """解析结果文件，提取SAM和SAM2的IoU值"""
    sam_ious = []
    sam2_ious = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到对比结果表格的开始位置
    in_table = False
    for line in lines:
        # 检查是否是表格头部
        if 'SAM vs SAM2 对比结果' in line:
            in_table = True
            continue
        
        if not in_table:
            continue
        
        # 跳过分隔线
        if line.strip().startswith('-'):
            continue
        
        # 解析数据行：序号 图像名 物体名 SAM IoU SAM状态 SAM2 IoU SAM2状态
        # 例如: "1      4558.png             conventional rivets            0.9930       成功           0.9868       成功"
        match = re.match(r'^\s*(\d+)\s+(\S+)\s+([^\d]+?)\s+([\d.]+)\s+(成功|失败)\s+([\d.]+)\s+(成功|失败)', line)
        if match:
            sam_iou = float(match.group(4))
            sam2_iou = float(match.group(6))
            sam_ious.append(sam_iou)
            sam2_ious.append(sam2_iou)
    
    return sam_ious, sam2_ious


def calculate_success_rate(ious, threshold):
    """计算给定阈值下的成功率"""
    if len(ious) == 0:
        return 0.0, 0, 0
    
    success_count = sum(1 for iou in ious if iou >= threshold)
    total_count = len(ious)
    success_rate = (success_count / total_count) * 100.0
    
    return success_rate, success_count, total_count


def main():
    results_file = Path('/opt/data/private/LLMSeg/vigor_success_rate_results.txt')
    
    if not results_file.exists():
        print(f"错误: 找不到结果文件 {results_file}")
        return
    
    print("正在解析结果文件...")
    sam_ious, sam2_ious = parse_results_file(results_file)
    
    if len(sam_ious) == 0 or len(sam2_ious) == 0:
        print("错误: 未能从文件中解析出IoU数据")
        return
    
    print(f"成功解析 {len(sam_ious)} 条记录\n")
    
    # 要计算的阈值
    thresholds = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 计算并输出结果
    print("=" * 90)
    print("不同IoU阈值下的成功率统计")
    print("=" * 90)
    print(f"{'阈值':<10} {'SAM成功率':<18} {'SAM成功数/总数':<20} {'SAM2成功率':<18} {'SAM2成功数/总数':<20}")
    print("-" * 90)
    
    for threshold in thresholds:
        sam_rate, sam_success, sam_total = calculate_success_rate(sam_ious, threshold)
        sam2_rate, sam2_success, sam2_total = calculate_success_rate(sam2_ious, threshold)
        
        print(f"{threshold:<10.1f} {sam_rate:>6.2f}%{'':<10} {sam_success}/{sam_total:<15} {sam2_rate:>6.2f}%{'':<10} {sam2_success}/{sam2_total:<15}")
    
    print("=" * 90)
    
    # 同时保存到文件
    output_file = Path('/opt/data/private/LLMSeg/vigor_success_rate_by_threshold.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("不同IoU阈值下的成功率统计\n")
        f.write("=" * 90 + "\n")
        f.write(f"基于文件: {results_file}\n")
        f.write(f"总记录数: {len(sam_ious)}\n\n")
        f.write(f"{'阈值':<10} {'SAM成功率':<18} {'SAM成功数/总数':<20} {'SAM2成功率':<18} {'SAM2成功数/总数':<20}\n")
        f.write("-" * 90 + "\n")
        
        for threshold in thresholds:
            sam_rate, sam_success, sam_total = calculate_success_rate(sam_ious, threshold)
            sam2_rate, sam2_success, sam2_total = calculate_success_rate(sam2_ious, threshold)
            
            f.write(f"{threshold:<10.1f} {sam_rate:>6.2f}%{'':<10} {sam_success}/{sam_total:<15} {sam2_rate:>6.2f}%{'':<10} {sam2_success}/{sam2_total:<15}\n")
        
        f.write("=" * 90 + "\n")
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
