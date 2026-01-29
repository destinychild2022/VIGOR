#!/bin/bash
# ========================================================================
# LLMSeg 单张图片推理脚本
# 
# 【重要】LLMSeg 的推理需要 SAM 候选 mask！
# 它是从候选 mask 中选择相似度最高的，而不是直接生成 mask
# ========================================================================

# ========== 模型路径配置 ==========
LISA_MODEL_PATH="/opt/data/private/model/LISA_Plus_7b"
CHECKPOINT_PATH="/opt/data/private/LLMSeg/runs/finetune_llmseg_vigor_simple/ckpt_model"
CLIP_PATH="/opt/data/private/model/clip-vit-large-patch14"
SAM_VIT_PATH="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# ========== 输入配置（请修改为你的参数）==========
# 输入图片路径
IMAGE_PATH="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test/7.png"

# 输入指令
INSTRUCTION="Please hand over the spring washers."

# SAM 候选 mask 目录 (必需！)
SAM_MASKS_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test_mask/sam_masks3"

# GT mask 路径 (可选，用于对比可视化)
# 例如: GT_MASK_PATH="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test/masks/7_lockwashers_mask.png"
GT_MASK_PATH=""

# ========== 输出配置 ==========
OUTPUT_DIR="./infer_output"

# ========== 测试配置 ==========
PRECISION="bf16"

# ========================================================================

cd "$(dirname "$0")/.." || exit 1

echo "========================================================================"
echo "  LLMSeg 单张图片推理"
echo "========================================================================"
echo "输入图片: ${IMAGE_PATH}"
echo "指令: ${INSTRUCTION}"
echo "SAM候选mask: ${SAM_MASKS_DIR}"
if [ -n "$GT_MASK_PATH" ]; then
    echo "GT mask: ${GT_MASK_PATH}"
fi
echo "========================================================================"

# 构建命令参数
CMD="python test/infer_single_llmseg.py \
    --version=\"${LISA_MODEL_PATH}\" \
    --checkpoint=\"${CHECKPOINT_PATH}\" \
    --vision_tower=\"${CLIP_PATH}\" \
    --vision_pretrained=\"${SAM_VIT_PATH}\" \
    --image=\"${IMAGE_PATH}\" \
    --instruction=\"${INSTRUCTION}\" \
    --sam_masks_dir=\"${SAM_MASKS_DIR}\" \
    --output_dir=\"${OUTPUT_DIR}\" \
    --precision=\"${PRECISION}\" \
    --use_mm_start_end"

# 添加 GT mask 参数（如果提供）
if [ -n "$GT_MASK_PATH" ]; then
    CMD="$CMD --gt_mask=\"${GT_MASK_PATH}\""
fi

eval $CMD

echo ""
echo "========================================================================"
echo "  推理完成"
echo "========================================================================"
