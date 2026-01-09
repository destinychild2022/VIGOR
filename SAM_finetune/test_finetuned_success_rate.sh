#!/bin/bash
# 测试VIGOR-100K数据集中微调SAM生成的候选掩码成功率，并与原始SAM对比

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 激活虚拟环境
if [ -d ".venv" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
else
    echo "警告: 未找到虚拟环境 .venv"
fi

# 设置参数
ANNOTATIONS_FILE="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test/all_annotations.json"
DATASET_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test"
SAM_MASKS_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/sam_masks_0.8_0.8"
FINETUNED_MASKS_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test_masks_finetuned_fixed_0.8_0.8"
SUCCESS_THRESHOLD=0.4
OUTPUT_FILE="./vigor_finetuned_success_rate_results_0.8_0.8_same.txt"

# 检查路径是否存在
if [ ! -f "$ANNOTATIONS_FILE" ]; then
    echo "错误: 标注文件不存在: $ANNOTATIONS_FILE"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "错误: 数据集目录不存在: $DATASET_DIR"
    exit 1
fi

if [ ! -d "$SAM_MASKS_DIR" ]; then
    echo "警告: SAM掩码目录不存在: $SAM_MASKS_DIR"
    echo "将只测试微调SAM的结果"
    SAM_MASKS_DIR=""
fi

if [ ! -d "$FINETUNED_MASKS_DIR" ]; then
    echo "错误: 微调SAM掩码目录不存在: $FINETUNED_MASKS_DIR"
    exit 1
fi

echo "=========================================="
echo "VIGOR-100K Test Set Success Rate Evaluation"
echo "微调SAM vs 原始SAM对比测试"
echo "=========================================="
echo "Annotations file: $ANNOTATIONS_FILE"
echo "Dataset dir: $DATASET_DIR"
echo "Original SAM masks dir: $SAM_MASKS_DIR"
echo "Finetuned SAM masks dir: $FINETUNED_MASKS_DIR"
echo "Success threshold: IoU >= $SUCCESS_THRESHOLD"
echo "Output file: $OUTPUT_FILE"
echo "=========================================="
echo ""

# 构建命令参数
CMD="python SAM_finetune/test_sam_success_rate.py \
    --dataset_type vigor \
    --vigor_annotations_file \"$ANNOTATIONS_FILE\" \
    --vigor_dataset_dir \"$DATASET_DIR\" \
    --success_threshold \"$SUCCESS_THRESHOLD\" \
    --output_file \"$OUTPUT_FILE\""

# 添加原始SAM掩码目录（如果存在）
if [ ! -z "$SAM_MASKS_DIR" ]; then
    CMD="$CMD --sam_masks_dir \"$SAM_MASKS_DIR\""
fi

# 添加微调SAM掩码目录
CMD="$CMD --sam_masks2_dir \"$FINETUNED_MASKS_DIR\""

echo "运行测试命令:"
echo "$CMD"
echo ""

# 运行测试
eval $CMD

echo ""
echo "测试完成！结果已保存到: $OUTPUT_FILE"
echo ""
echo "结果摘要将在文件中包含："
echo "  - 原始SAM的成功率"
echo "  - 微调SAM的成功率"
echo "  - 性能提升对比"
