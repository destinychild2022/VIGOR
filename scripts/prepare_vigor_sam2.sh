#!/bin/bash

# 使用SAM2/SAM为VIGOR-100K/test数据集生成候选掩码

# 设置模型路径
# SAM2模型路径（如果使用SAM2）
SAM2_MODEL_PATH="/opt/data/private/model/sam2-hiera-large"
# SAM模型路径（可以是目录或checkpoint文件路径）
SAM_MODEL_PATH="/opt/data/private/model/SAM-vit-h"
# 或者直接指定checkpoint文件：
# SAM_MODEL_PATH="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# 微调后的SAM模型路径
FINETUNED_CHECKPOINT="/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_vigor_point2/best_model.pth"

# 设置数据集路径
DATASET_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/unseen"

# 设置输出目录
OUTPUT_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/unseen_masks_sam_0.8_0.8"

# 设置GPU
GPU_IDS="0"
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 选择使用的模型（sam2 或 sam）
USE_MODEL="sam"  # 改为 "sam2" 使用SAM2模型

# 是否使用微调后的模型（true/false）
USE_FINETUNED="false"  # 改为 "true" 使用微调后的SAM模型

# 设置其他参数
MAX_IMAGES=""  # 留空处理所有图像，或设置数字限制（如100用于测试）
SAVE_FORMAT="png"  # png 或 npy

# 切换到脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# 激活虚拟环境
if [ -d ".venv" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
fi

# 检查模型路径
if [ "$USE_FINETUNED" = "true" ]; then
    # 使用微调后的模型
    MODEL_PATH="$FINETUNED_CHECKPOINT"
    USE_SAM2_FLAG=""
    FALLBACK_FLAG=""
    USE_FINETUNED_FLAG="--use_finetuned"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Error: Finetuned model checkpoint not found: $MODEL_PATH"
        exit 1
    fi
    echo "Using finetuned SAM model: $MODEL_PATH"
elif [ "$USE_MODEL" = "sam2" ]; then
    MODEL_PATH="$SAM2_MODEL_PATH"
    USE_SAM2_FLAG="--use_sam2"
    FALLBACK_FLAG="--fallback_sam_path $SAM_MODEL_PATH"
    USE_FINETUNED_FLAG=""
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Error: SAM2 model path not found: $MODEL_PATH"
        exit 1
    fi
    if [ ! -d "$SAM_MODEL_PATH" ]; then
        echo "Warning: SAM fallback path not found: $SAM_MODEL_PATH"
        echo "         If SAM2 is not available, script will fail"
        FALLBACK_FLAG=""
    fi
else
    MODEL_PATH="$SAM_MODEL_PATH"
    USE_SAM2_FLAG=""
    FALLBACK_FLAG=""
    USE_FINETUNED_FLAG=""
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Error: SAM model path not found: $MODEL_PATH"
        exit 1
    fi
fi

# 检查数据集目录
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 显示配置
echo "=========================================="
echo "VIGOR-100K Test Set Mask Generation"
echo "=========================================="
echo "Model: $USE_MODEL"
if [ "$USE_FINETUNED" = "true" ]; then
    echo "Using Finetuned Model: YES"
else
    echo "Using Finetuned Model: NO"
fi
echo "Model Path: $MODEL_PATH"
echo "Dataset Dir: $DATASET_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "GPU: $GPU_IDS"
echo "Save Format: $SAVE_FORMAT"
if [ -n "$MAX_IMAGES" ]; then
    echo "Max Images: $MAX_IMAGES (limited)"
fi
echo "=========================================="
echo ""

# 运行脚本
python prepare_datasets/prepare_vigor_sam2.py \
    --model_path "$MODEL_PATH" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device "cuda" \
    --save_format "$SAVE_FORMAT" \
    $USE_SAM2_FLAG \
    $FALLBACK_FLAG \
    $USE_FINETUNED_FLAG \
    $([ -n "$MAX_IMAGES" ] && echo "--max_images $MAX_IMAGES") \

echo ""
echo "=========================================="
echo "Mask generation completed!"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
