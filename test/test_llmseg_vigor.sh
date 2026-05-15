#!/bin/bash
set -euo pipefail
# ========================================================================
# LLMSeg VIGOR-100K 测试脚本
# 
# 【重要】LLMSeg 的推理需要 SAM 候选 mask！
# 它是从候选 mask 中选择相似度最高的，而不是直接生成 mask
# ========================================================================

# ========== 模型路径配置 ==========
LISA_MODEL_PATH="/root/autodl-tmp/model/LISA_Plus_7b"
CHECKPOINT_BASE="/root/autodl-tmp/runs/finetune_llmseg_vigor_simple-spatial-1/ckpt_model"
CLIP_PATH="/root/autodl-tmp/model/clip-vit-large-patch14"
SAM_VIT_PATH="/root/autodl-tmp/model/SAM-vit-h/sam_vit_h_4b8939.pth"
export TORCH_HOME="/root/autodl-tmp/torch_cache"

# ========== 数据集配置 ==========
# 测试数据集路径
TEST_DATA_DIR="/root/autodl-tmp/VIGOR-100K_new/test"
DEPTH_DIR="${TEST_DATA_DIR}/depth"
# SAM 候选 mask 目录 (必需！)
SAM_MASKS_DIR="/root/autodl-tmp/test_mask/sam_masks3"

# ========== 输出配置 ==========
OUTPUT_DIR="/root/autodl-tmp/result"
VIS_DIR="/root/autodl-tmp/vis_output2_hard"  # 可视化输出目录
SAVE_VIS="false"  # 是否保存可视化图片 (true/false)

# ========== 测试配置 ==========
PRECISION="bf16"
ICR_THRESHOLDS="0.3,0.4,0.5,0.6,0.7,0.8,0.9"
WORKERS=12
BATCH_SIZE=1
MAX_INSTRUCTIONS=3

# 候选 mask 选择方式:
#   similarity: 选 pred_similarity 最大的单个候选 mask
#   iou:        合并 pred_iou > IOU_THRESHOLD 的候选 mask（与 validate_threshold 一致）
MASK_SELECTION_MODE="similarity"
IOU_THRESHOLD=0.5

# 一次性测试的 checkpoint 子目录。需要改哪些权重就直接改这里。
CKPT_NAMES="best latest"

# ========== 调试配置 ==========
DEBUG=0
MAX_SAMPLES=""
GPU_ID=0
SPLIT="both"  # 可选: both, easy, hard

# ========================================================================

cd "$(dirname "$0")/.." || exit 1

PYTHON_BIN="./.venv3124/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

echo "========================================================================"
echo "  LLMSeg VIGOR-100K 测试"
echo "========================================================================"
echo "基础模型: ${LISA_MODEL_PATH}"
echo "权重根目录: ${CHECKPOINT_BASE}"
echo "测试权重: ${CKPT_NAMES}"
echo "Torch Hub缓存: ${TORCH_HOME}/hub"
echo "测试数据: ${TEST_DATA_DIR}"
echo "Depth目录: ${DEPTH_DIR}"
echo "SAM候选mask: ${SAM_MASKS_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "可视化目录: ${VIS_DIR}"
echo "候选选择: ${MASK_SELECTION_MODE} (IOU_THRESHOLD=${IOU_THRESHOLD})"
echo "Workers: ${WORKERS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Python: ${PYTHON_BIN}"
echo "========================================================================"

# 构建公共参数
COMMON_ARGS="
    --version=${LISA_MODEL_PATH}
    --vision_tower=${CLIP_PATH}
    --vision_pretrained=${SAM_VIT_PATH}
    --data_dir=${TEST_DATA_DIR}
    --depth_dir=${DEPTH_DIR}
    --sam_masks_dir=${SAM_MASKS_DIR}
    --precision=${PRECISION}
    --icr_thresholds=${ICR_THRESHOLDS}
    --device=cuda:${GPU_ID}
    --split=${SPLIT}
    --workers=${WORKERS}
    --batch_size=${BATCH_SIZE}
    --max_instructions=${MAX_INSTRUCTIONS}
    --mask_selection_mode=${MASK_SELECTION_MODE}
    --iou_threshold=${IOU_THRESHOLD}
    --use_mm_start_end
"

# 添加可视化保存参数
if [ "$SAVE_VIS" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --save_vis"
    echo "可视化保存: 开启"
fi

# 添加调试参数
if [ "$DEBUG" = "1" ] || [ "$DEBUG" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --debug"
    echo "调试模式: 开启"
fi

# 添加最大样本数限制
if [ -n "$MAX_SAMPLES" ]; then
    COMMON_ARGS="${COMMON_ARGS} --max_samples=${MAX_SAMPLES}"
    echo "最大样本数: ${MAX_SAMPLES}"
fi

echo ""

# 执行测试
FOUND_CKPT=0
for CKPT_NAME in ${CKPT_NAMES}; do
    CHECKPOINT_PATH="${CHECKPOINT_BASE}/${CKPT_NAME}"
    if [ ! -d "${CHECKPOINT_PATH}" ]; then
        echo "[警告] 跳过不存在的权重目录: ${CHECKPOINT_PATH}"
        continue
    fi
    FOUND_CKPT=1

    CKPT_OUTPUT_DIR="${OUTPUT_DIR}/${CKPT_NAME}/${MASK_SELECTION_MODE}"
    mkdir -p "${CKPT_OUTPUT_DIR}"

    echo ""
    echo "========================================================================"
    echo "  开始测试权重: ${CKPT_NAME}"
    echo "  Checkpoint: ${CHECKPOINT_PATH}"
    echo "  Result: ${CKPT_OUTPUT_DIR}"
    echo "========================================================================"

    ${PYTHON_BIN} test/test_llmseg_vigor.py ${COMMON_ARGS} \
        --checkpoint="${CHECKPOINT_PATH}" \
        --checkpoint_tag="${CKPT_NAME}" \
        --output_dir="${CKPT_OUTPUT_DIR}" \
        --vis_dir="${VIS_DIR}"
done

if [ "${FOUND_CKPT}" -eq 0 ]; then
    echo "[错误] 没有找到任何可测试的权重目录，请检查 CHECKPOINT_BASE 和 CKPT_NAMES。"
    exit 1
fi

echo ""
echo "========================================================================"
echo "  测试完成"
echo "========================================================================"
