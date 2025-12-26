#!/bin/bash

# 按object名称测试IoU脚本

# 数据集路径
ANNOTATIONS_DIR="/opt/data/private/LLMSeg/dataset/GT_mask"
GT_MASKS_DIR="/opt/data/private/LLMSeg/dataset/GT_mask"
IMAGES_DIR="/opt/data/private/LLMSeg/dataset/raw_pic"

# 候选mask目录（原始SAM生成的）
CANDIDATE_MASK_DIR="/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_robot_arm_point2/test_vis_sam_origin"

# 输出目录
OUTPUT_BASE_DIR="/opt/data/private/LLMSeg/runs/finetune_llmseg_robot_arm2/test_vis_object"
OUTPUT_FILE="${OUTPUT_BASE_DIR}/test_object_iou_results.txt"
VIS_OUTPUT_DIR="${OUTPUT_BASE_DIR}"

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# 激活虚拟环境（如果需要）
if [ -d ".venv" ]; then
    echo "激活虚拟环境..."
    source ".venv/bin/activate"
else
    echo "警告: 未找到虚拟环境，使用系统Python"
fi

echo "=== 按Object名称测试IoU ==="
echo "Annotations目录: $ANNOTATIONS_DIR"
echo "GT Masks目录: $GT_MASKS_DIR"
echo "原图目录: $IMAGES_DIR"
echo "候选mask目录: $CANDIDATE_MASK_DIR"
echo "输出文件: $OUTPUT_FILE"
echo "可视化输出目录: $VIS_OUTPUT_DIR"
echo ""

# 检查目录是否存在
if [ ! -d "$ANNOTATIONS_DIR" ]; then
    echo "错误: Annotations目录不存在: $ANNOTATIONS_DIR"
    exit 1
fi

if [ ! -d "$GT_MASKS_DIR" ]; then
    echo "错误: GT Masks目录不存在: $GT_MASKS_DIR"
    exit 1
fi

if [ ! -d "$CANDIDATE_MASK_DIR" ]; then
    echo "错误: 候选mask目录不存在: $CANDIDATE_MASK_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_BASE_DIR"

# 运行测试脚本
python test_object_iou.py \
    --annotations_dir "$ANNOTATIONS_DIR" \
    --gt_masks_dir "$GT_MASKS_DIR" \
    --images_dir "$IMAGES_DIR" \
    --candidate_mask_dir "$CANDIDATE_MASK_DIR" \
    --output_file "$OUTPUT_FILE" \
    --vis_output_dir "$VIS_OUTPUT_DIR" \
    --view_names robot_arm_01 robot_arm_02 robot_arm_03

echo ""
echo "测试完成！"
echo "结果文件: $OUTPUT_FILE"
echo "可视化结果: $VIS_OUTPUT_DIR（按object名称分目录）"
