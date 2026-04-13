#!/bin/bash
# ========================================================================
# LLMSeg VIGOR-100K 测试脚本
# 
# 【重要】LLMSeg 的推理需要 SAM 候选 mask！
# 它是从候选 mask 中选择相似度最高的，而不是直接生成 mask
# ========================================================================

# ========== 模型路径配置 ==========
LISA_MODEL_PATH="../root/autodl-tmp/model/LISA_Plus_7b"
CHECKPOINT_PATH="../root/autodl-tmp/runs/finetune_llmseg_vigor_simple-object/ckpt_model"
CLIP_PATH="../root/autodl-tmp/model/clip-vit-large-patch14"
SAM_VIT_PATH="../root/autodl-tmp/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# ========== 数据集配置 ==========
# 测试数据集路径
TEST_DATA_DIR="../root/autodl-tmp/VIGOR-100K_new/test"
# SAM 候选 mask 目录 (必需！)
SAM_MASKS_DIR="../root/autodl-tmp/test_mask/sam_masks"

# ========== 输出配置 ==========
OUTPUT_DIR="../root/autodl-tmp/test_results"
VIS_DIR="../root/autodl-tmp/test_vis_output"  # 可视化输出目录
SAVE_VIS="true"  # 是否保存可视化图片 (true/false)

# ========== 测试配置 ==========
PRECISION="bf16"
ICR_THRESHOLDS="0.3,0.4,0.5,0.6,0.7,0.8,0.9"

# ========== 调试配置 ==========
DEBUG="${DEBUG:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GPU_ID="${GPU_ID:-0}"
SPLIT="${SPLIT:-hard}"  # 可选: both, easy, hard

# ========================================================================

cd "$(dirname "$0")/.." || exit 1

echo "========================================================================"
echo "  LLMSeg VIGOR-100K 测试"
echo "========================================================================"
echo "基础模型: ${LISA_MODEL_PATH}"
echo "微调权重: ${CHECKPOINT_PATH}"
echo "测试数据: ${TEST_DATA_DIR}"
echo "SAM候选mask: ${SAM_MASKS_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "可视化目录: ${VIS_DIR}"
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
    --device="cuda:${GPU_ID}"
    --split=${SPLIT}
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
python test/test_llmseg_vigor.py ${ARGS}

echo ""
echo "========================================================================"
echo "  测试完成"
echo "========================================================================"
