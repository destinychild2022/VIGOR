"""
分析 VIGOR 测试结果中 IC-IoU 的成功率

统计每个 sample 的 IC-IoU (per instruction) 中有多少个超过各阈值 (0.3-0.9)

使用方法:
    python analyze_ic_iou_success_rate.py --input /path/to/vigor_test_results.txt
"""

import argparse
import re
import os
from collections import defaultdict


def parse_ic_iou_values(line: str):
    """
    解析 IC-IoU (per instruction) 行
    格式: IC-IoU (per instruction): ['0.6507', '0.6510', '0.3171']
    """
    match = re.search(r"IC-IoU \(per instruction\): \[(.*?)\]", line)
    if match:
        values_str = match.group(1)
        # 解析 '0.6507', '0.6510', '0.3171' 格式
        values = re.findall(r"'([\d.]+)'", values_str)
        return [float(v) for v in values]
    return []


def analyze_results(input_file: str, output_file: str):
    """分析结果文件并输出统计"""
    
    # 定义阈值
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 统计数据
    stats = {
        'easy': {
            'total_instructions': 0,
            'passed': {t: 0 for t in thresholds},
            'sample_count': 0,
        },
        'hard': {
            'total_instructions': 0,
            'passed': {t: 0 for t in thresholds},
            'sample_count': 0,
        }
    }
    
    # 当前解析状态
    current_section = None  # 'easy' or 'hard'
    line_count = 0
    matched_lines = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            line = line.strip()
            
            # 检测当前是 Easy 还是 Hard 部分
            if "Easy 样本详细结果" in line:
                current_section = 'easy'
            elif "Hard 样本详细结果" in line:
                current_section = 'hard'
            
            # 解析 IC-IoU (per instruction)
            if current_section and "IC-IoU (per instruction):" in line:
                matched_lines += 1
                iou_values = parse_ic_iou_values(line)
                
                if iou_values:
                    stats[current_section]['sample_count'] += 1
                    stats[current_section]['total_instructions'] += len(iou_values)
                    
                    # 统计每个阈值下超过的数量
                    for threshold in thresholds:
                        passed = sum(1 for v in iou_values if v >= threshold)
                        stats[current_section]['passed'][threshold] += passed
    
    # 打印读取统计
    print(f"\n[调试] 文件总行数: {line_count}")
    print(f"[调试] 匹配到的 IC-IoU 行数: {matched_lines}")
    print(f"[调试] Easy 样本数: {stats['easy']['sample_count']}, 指令数: {stats['easy']['total_instructions']}")
    print(f"[调试] Hard 样本数: {stats['hard']['sample_count']}, 指令数: {stats['hard']['total_instructions']}")
    print("")
    
    # 生成输出报告
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("  IC-IoU 成功率分析")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append(f"输入文件: {input_file}")
    output_lines.append("")
    
    # 汇总表格
    output_lines.append("=" * 80)
    output_lines.append("  汇总表格")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # 表头
    header = "| Category | #Samples | #Instructions |"
    for t in thresholds:
        header += f" SR@{t:.1f} |"
    output_lines.append(header)
    
    separator = "|----------|----------|---------------|"
    for _ in thresholds:
        separator += "--------|"
    output_lines.append(separator)
    
    # Easy 行
    easy = stats['easy']
    easy_sr = {}
    row = f"| Easy     | {easy['sample_count']:>8} | {easy['total_instructions']:>13} |"
    for t in thresholds:
        if easy['total_instructions'] > 0:
            sr = easy['passed'][t] / easy['total_instructions'] * 100
        else:
            sr = 0.0
        easy_sr[t] = sr
        row += f" {sr:>5.2f}% |"
    output_lines.append(row)
    
    # Hard 行
    hard = stats['hard']
    hard_sr = {}
    row = f"| Hard     | {hard['sample_count']:>8} | {hard['total_instructions']:>13} |"
    for t in thresholds:
        if hard['total_instructions'] > 0:
            sr = hard['passed'][t] / hard['total_instructions'] * 100
        else:
            sr = 0.0
        hard_sr[t] = sr
        row += f" {sr:>5.2f}% |"
    output_lines.append(row)
    
    # Average 行（按样本数加权平均）
    total_samples = easy['sample_count'] + hard['sample_count']
    total_instr = easy['total_instructions'] + hard['total_instructions']
    avg_sr = {}
    row = f"| Average  | {total_samples:>8} | {total_instr:>13} |"
    for t in thresholds:
        if total_samples > 0:
            # 加权平均：(easy_sr * easy_samples + hard_sr * hard_samples) / total_samples
            sr = (easy_sr[t] * easy['sample_count'] + hard_sr[t] * hard['sample_count']) / total_samples
        else:
            sr = 0.0
        avg_sr[t] = sr
        row += f" {sr:>5.2f}% |"
    output_lines.append(row)
    
    output_lines.append("")
    
    # 详细统计
    output_lines.append("=" * 80)
    output_lines.append("  详细统计")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    for category in ['easy', 'hard']:
        cat_stats = stats[category]
        output_lines.append(f"{category.upper()} 样本:")
        output_lines.append(f"  样本数: {cat_stats['sample_count']}")
        output_lines.append(f"  总指令数 (每个样本3条): {cat_stats['total_instructions']}")
        output_lines.append("")
        
        for t in thresholds:
            passed = cat_stats['passed'][t]
            total = cat_stats['total_instructions']
            if total > 0:
                sr = passed / total * 100
            else:
                sr = 0.0
            output_lines.append(f"  SR@{t:.1f}: {passed}/{total} = {sr:.2f}%")
        output_lines.append("")
    
    # 平均值统计
    output_lines.append("AVERAGE (加权平均):")
    output_lines.append(f"  总样本数: {total_samples}")
    output_lines.append(f"  总指令数: {total_instr}")
    output_lines.append("")
    for t in thresholds:
        output_lines.append(f"  SR@{t:.1f}: {avg_sr[t]:.2f}%")
    output_lines.append("")
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + "\n")
    
    # 同时打印到控制台
    for line in output_lines:
        print(line)
    
    print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="分析 VIGOR 测试结果中 IC-IoU 的成功率")
    parser.add_argument("--input", "-i", required=True, type=str,
                        help="输入的测试结果文件路径")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出文件路径（默认在输入文件同目录下生成）")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 生成输出文件名
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        input_name = os.path.basename(args.input)
        output_name = input_name.replace(".txt", "_ic_iou_success_rate.txt")
        args.output = os.path.join(input_dir, output_name)
    
    analyze_results(args.input, args.output)


if __name__ == "__main__":
    main()
