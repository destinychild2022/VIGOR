#!/bin/bash
# ========================================================================
# LLMSeg VIGOR-100K Unseen 数据集测试脚本
# 
# 【重要】LLMSeg 的推理需要 SAM 候选 mask！
# 它是从候选 mask 中选择相似度最高的，而不是直接生成 mask
# 
# 计算指标：IC-IoU、ICR、SSR@0.5
# ========================================================================

# ========== 模型路径配置 ==========
LISA_MODEL_PATH="/opt/data/private/model/LISA_Plus_7b"
CHECKPOINT_PATH="/opt/data/private/LLMSeg/runs/finetune_llmseg_vigor_simple/ckpt_model"
CLIP_PATH="/opt/data/private/model/clip-vit-large-patch14"
SAM_VIT_PATH="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# ========== 数据集配置 ==========
# Unseen 数据集路径
TEST_DATA_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/unseen"
# SAM 候选 mask 目录 (必需！)
SAM_MASKS_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/unseen_masks_sam_0.8_0.8"

# ========== 输出配置 ==========
OUTPUT_DIR="./result_unseen"
VIS_DIR="./vis_output_unseen"  # 可视化输出目录
SAVE_VIS="true"  # 是否保存可视化图片 (true/false)

# ========== 测试配置 ==========
PRECISION="bf16"
ICR_THRESHOLDS="0.3,0.4,0.5,0.6,0.7,0.8,0.9"
SSR_THRESHOLD="0.5"  # SSR 成功率阈值

# ========== 调试配置 ==========
DEBUG="${DEBUG:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

# ========================================================================

cd "$(dirname "$0")/.." || exit 1

echo "========================================================================"
echo "  LLMSeg VIGOR-100K Unseen 数据集测试"
echo "========================================================================"
echo "基础模型: ${LISA_MODEL_PATH}"
echo "微调权重: ${CHECKPOINT_PATH}"
echo "测试数据: ${TEST_DATA_DIR}"
echo "SAM候选mask: ${SAM_MASKS_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "可视化目录: ${VIS_DIR}"
echo "SSR阈值: ${SSR_THRESHOLD}"
echo "========================================================================"

# 构建参数
ARGS="
    --version=${LISA_MODEL_PATH}
    --checkpoint=${CHECKPOINT_PATH}
    --vision_tower=${CLIP_PATH}
    --vision_pretrained=${SAM_VIT_PATH}
    --data_dir=${TEST_DATA_DIR}
    --sam_masks_dir=${SAM_MASKS_DIR}
    --output_dir=${OUTPUT_DIR}
    --vis_dir=${VIS_DIR}
    --precision=${PRECISION}
    --icr_thresholds=${ICR_THRESHOLDS}
    --ssr_threshold=${SSR_THRESHOLD}
    --use_mm_start_end
"

# 添加可视化保存参数
if [ "$SAVE_VIS" = "true" ]; then
    ARGS="${ARGS} --save_vis"
    echo "可视化保存: 开启"
fi

# 添加调试参数
if [ "$DEBUG" = "1" ] || [ "$DEBUG" = "true" ]; then
    ARGS="${ARGS} --debug"
    echo "调试模式: 开启"
fi

# 添加最大样本数限制
if [ -n "$MAX_SAMPLES" ]; then
    ARGS="${ARGS} --max_samples=${MAX_SAMPLES}"
    echo "最大样本数: ${MAX_SAMPLES}"
fi

echo ""

# 执行测试
python test/test_llmseg_vigor_unseen.py ${ARGS}

echo ""
echo "========================================================================"
echo "  Unseen 测试完成"
echo "========================================================================"
