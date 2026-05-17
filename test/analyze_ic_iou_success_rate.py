"""
分析 VIGOR 测试结果中 IC-IoU 的成功率。

统计每个 sample 的 IC-IoU (per instruction) 中有多少个超过各阈值，
并额外统计 Hard 样本前2条指令与第3条指令的 SSR。

使用方法:
    python test/analyze_ic_iou_success_rate.py --input /path/to/vigor_test_results.txt
"""

import argparse
import os
import re


def parse_ic_iou_values(line: str):
    """
    解析 IC-IoU (per instruction) 行。
    兼容格式: IC-IoU (per instruction): ['0.6507', '0.6510', '0.3171']
    """
    match = re.search(r"IC-IoU \(per instruction\): \[(.*?)\]", line)
    if not match:
        return []
    values_str = match.group(1)
    values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", values_str)
    return [float(v) for v in values]


def new_bucket(thresholds):
    return {
        "total_instructions": 0,
        "passed": {t: 0 for t in thresholds},
        "sample_count": 0,
        "values": [],
    }


def update_bucket(bucket, values, thresholds):
    if not values:
        return
    bucket["sample_count"] += 1
    bucket["total_instructions"] += len(values)
    bucket["values"].extend(values)
    for threshold in thresholds:
        bucket["passed"][threshold] += sum(1 for v in values if v >= threshold)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sr(bucket, threshold):
    total = bucket["total_instructions"]
    return bucket["passed"][threshold] / total * 100.0 if total > 0 else 0.0


def analyze_results(input_file: str, output_file: str):
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    stats = {
        "easy": new_bucket(thresholds),
        "hard": new_bucket(thresholds),
    }
    hard_groups = {
        "first2": new_bucket(thresholds),
        "third": new_bucket(thresholds),
    }

    current_section = None
    line_count = 0
    matched_lines = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            line = line.strip()

            if "Easy 样本详细结果" in line:
                current_section = "easy"
            elif "Hard 样本详细结果" in line:
                current_section = "hard"

            if current_section and "IC-IoU (per instruction):" in line:
                iou_values = parse_ic_iou_values(line)
                if not iou_values:
                    continue
                matched_lines += 1
                update_bucket(stats[current_section], iou_values, thresholds)

                if current_section == "hard":
                    update_bucket(hard_groups["first2"], iou_values[:2], thresholds)
                    update_bucket(hard_groups["third"], iou_values[2:3], thresholds)

    print(f"\n[调试] 文件总行数: {line_count}")
    print(f"[调试] 匹配到的 IC-IoU 行数: {matched_lines}")
    print(f"[调试] Easy 样本数: {stats['easy']['sample_count']}, 指令数: {stats['easy']['total_instructions']}")
    print(f"[调试] Hard 样本数: {stats['hard']['sample_count']}, 指令数: {stats['hard']['total_instructions']}")
    print(f"[调试] Hard 前2条指令数: {hard_groups['first2']['total_instructions']}")
    print(f"[调试] Hard 第3条指令数: {hard_groups['third']['total_instructions']}")
    print("")

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("  IC-IoU 成功率分析")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append(f"输入文件: {input_file}")
    output_lines.append("")

    output_lines.append("=" * 80)
    output_lines.append("  汇总表格")
    output_lines.append("=" * 80)
    output_lines.append("")

    header = "| Category | #Samples | #Instructions | Avg IC-IoU |"
    for t in thresholds:
        header += f" SR@{t:.1f} |"
    output_lines.append(header)

    separator = "|----------|----------|---------------|------------|"
    for _ in thresholds:
        separator += "--------|"
    output_lines.append(separator)

    for category, label in [("easy", "Easy"), ("hard", "Hard")]:
        bucket = stats[category]
        row = (
            f"| {label:<8} | {bucket['sample_count']:>8} | "
            f"{bucket['total_instructions']:>13} | {mean(bucket['values']):>10.4f} |"
        )
        for t in thresholds:
            row += f" {sr(bucket, t):>5.2f}% |"
        output_lines.append(row)

    total_samples = stats["easy"]["sample_count"] + stats["hard"]["sample_count"]
    total_instr = stats["easy"]["total_instructions"] + stats["hard"]["total_instructions"]
    all_values = stats["easy"]["values"] + stats["hard"]["values"]
    row = f"| {'Average':<8} | {total_samples:>8} | {total_instr:>13} | {mean(all_values):>10.4f} |"
    avg_sr = {}
    for t in thresholds:
        if total_instr > 0:
            value = (stats["easy"]["passed"][t] + stats["hard"]["passed"][t]) / total_instr * 100.0
        else:
            value = 0.0
        avg_sr[t] = value
        row += f" {value:>5.2f}% |"
    output_lines.append(row)
    output_lines.append("")

    output_lines.append("=" * 80)
    output_lines.append("  Hard 指令分组 SSR")
    output_lines.append("=" * 80)
    output_lines.append("")
    group_header = "| Hard Group | #Samples | #Instructions | Avg IC-IoU |"
    for t in thresholds:
        group_header += f" SSR@{t:.1f} |"
    output_lines.append(group_header)
    group_separator = "|------------|----------|---------------|------------|"
    for _ in thresholds:
        group_separator += "---------|"
    output_lines.append(group_separator)

    for key, label in [("first2", "Instr1-2"), ("third", "Instr3")]:
        bucket = hard_groups[key]
        row = (
            f"| {label:<10} | {bucket['sample_count']:>8} | "
            f"{bucket['total_instructions']:>13} | {mean(bucket['values']):>10.4f} |"
        )
        for t in thresholds:
            row += f" {sr(bucket, t):>6.2f}% |"
        output_lines.append(row)
    output_lines.append("")

    output_lines.append("=" * 80)
    output_lines.append("  详细统计")
    output_lines.append("=" * 80)
    output_lines.append("")

    for category in ["easy", "hard"]:
        bucket = stats[category]
        output_lines.append(f"{category.upper()} 样本:")
        output_lines.append(f"  样本数: {bucket['sample_count']}")
        output_lines.append(f"  总指令数: {bucket['total_instructions']}")
        output_lines.append(f"  平均 IC-IoU: {mean(bucket['values']):.4f}")
        output_lines.append("")
        for t in thresholds:
            passed = bucket["passed"][t]
            total = bucket["total_instructions"]
            output_lines.append(f"  SR@{t:.1f}: {passed}/{total} = {sr(bucket, t):.2f}%")
        output_lines.append("")

    output_lines.append("HARD 指令分组:")
    for key, label in [("first2", "前2条指令"), ("third", "第3条指令")]:
        bucket = hard_groups[key]
        output_lines.append(f"  {label}:")
        output_lines.append(f"    样本数: {bucket['sample_count']}")
        output_lines.append(f"    指令数: {bucket['total_instructions']}")
        output_lines.append(f"    平均 IC-IoU: {mean(bucket['values']):.4f}")
        for t in thresholds:
            passed = bucket["passed"][t]
            total = bucket["total_instructions"]
            output_lines.append(f"    SSR@{t:.1f}: {passed}/{total} = {sr(bucket, t):.2f}%")
        output_lines.append("")

    output_lines.append("AVERAGE:")
    output_lines.append(f"  总样本数: {total_samples}")
    output_lines.append(f"  总指令数: {total_instr}")
    output_lines.append(f"  平均 IC-IoU: {mean(all_values):.4f}")
    for t in thresholds:
        output_lines.append(f"  SR@{t:.1f}: {avg_sr[t]:.2f}%")
    output_lines.append("")

    with open(output_file, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    for line in output_lines:
        print(line)

    print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="分析 VIGOR 测试结果中 IC-IoU 的成功率")
    parser.add_argument("--input", "-i", required=True, type=str, help="输入的测试结果文件路径")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return

    if args.output is None:
        input_dir = os.path.dirname(args.input)
        input_name = os.path.basename(args.input)
        output_name = input_name.replace(".txt", "_ic_iou_success_rate.txt")
        args.output = os.path.join(input_dir, output_name)

    analyze_results(args.input, args.output)


if __name__ == "__main__":
    main()
