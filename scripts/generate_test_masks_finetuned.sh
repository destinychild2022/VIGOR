#!/bin/bash

# 使用微调后的SAM为VIGOR-100K/test数据集生成候选掩码

# 设置路径
FINETUNED_CHECKPOINT="/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_vigor_point/best_model.pth"
DATASET_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/train"
OUTPUT_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/train_masks_finetuned_fixed_0.8_0.8"

# 设置GPU
GPU_IDS="0"
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 切换到脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# 激活虚拟环境
if [ -d ".venv" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
fi

echo "=== 使用微调后的SAM为VIGOR-100K/test生成候选掩码 ==="
echo "微调模型: $FINETUNED_CHECKPOINT"
echo "数据集目录: $DATASET_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "GPU: $GPU_IDS"
echo ""

# 检查模型文件是否存在
if [ ! -f "$FINETUNED_CHECKPOINT" ]; then
    echo "❌ 错误: 微调模型文件不存在: $FINETUNED_CHECKPOINT"
    exit 1
fi

# 检查数据集目录是否存在
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ 错误: 数据集目录不存在: $DATASET_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "🚀 开始生成候选掩码..."
echo ""

# 运行掩码生成脚本（处理前400张图像）
python prepare_datasets/prepare_vigor_sam2.py \
    --model_path "$FINETUNED_CHECKPOINT" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --use_finetuned \
    --device "cuda" \
    --save_format "png" \

echo ""
echo "✅ 候选掩码生成完成！"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "查看生成的结果："
echo "  - 每张图像都有独立的文件夹"
echo "  - masks/: 包含所有候选掩码"
echo "  - mask_info.json: 掩码信息"
echo "  - overlay_all_masks.png: 可视化叠加图"
echo "  - original_image.png: 原图"
