#!/bin/bash
# 测试VIGOR-100K数据集中SAM和SAM2生成的候选掩码成功率

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
SAM_MASKS_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test/sam_masks"
SAM2_MASKS_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test/sam2_masks"
SUCCESS_THRESHOLD=0.4
OUTPUT_FILE="./vigor_success_rate_results.txt"

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
fi

if [ ! -d "$SAM2_MASKS_DIR" ]; then
    echo "警告: SAM2掩码目录不存在: $SAM2_MASKS_DIR"
fi

echo "=========================================="
echo "VIGOR-100K Test Set Success Rate Evaluation"
echo "=========================================="
echo "Annotations file: $ANNOTATIONS_FILE"
echo "Dataset dir: $DATASET_DIR"
echo "SAM masks dir: $SAM_MASKS_DIR"
echo "SAM2 masks dir: $SAM2_MASKS_DIR"
echo "Success threshold: IoU >= $SUCCESS_THRESHOLD"
echo "Output file: $OUTPUT_FILE"
echo "=========================================="
echo ""

# 运行测试
python SAM_finetune/test_sam_success_rate.py \
    --dataset_type vigor \
    --vigor_annotations_file "$ANNOTATIONS_FILE" \
    --vigor_dataset_dir "$DATASET_DIR" \
    --sam_masks_dir "$SAM_MASKS_DIR" \
    --sam2_masks_dir "$SAM2_MASKS_DIR" \
    --success_threshold "$SUCCESS_THRESHOLD" \
    --output_file "$OUTPUT_FILE"

echo ""
echo "测试完成！结果已保存到: $OUTPUT_FILE"
