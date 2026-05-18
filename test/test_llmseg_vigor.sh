#!/bin/bash
# ========================================================================
# LLMSeg VIGOR-100K 测试脚本
# 
# 【重要】LLMSeg 的推理需要 SAM 候选 mask！
# 它是从候选 mask 中选择相似度最高的，而不是直接生成 mask
# ========================================================================

# ========== 模型路径配置 ==========
LISA_MODEL_PATH="/opt/data/private/model/LISA_Plus_7b"
CHECKPOINT_PATH="/opt/data/private/LLMSeg/runs/finetune_llmseg_vigor_simple-object/ckpt_model/epoch_20"
CLIP_PATH="/opt/data/private/model/clip-vit-large-patch14"
SAM_VIT_PATH="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# ========== 数据集配置 ==========
# 测试数据集路径
TEST_DATA_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K_new/test"
TEST_EASY_JSON="open_vocab_grasp_easy_object_mix.json"
TEST_HARD_JSON="open_vocab_grasp_hard_object_mix.json"
# SAM 候选 mask 目录 (必需！)
SAM_MASKS_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test_mask/sam_masks3"

# ========== 输出配置 ==========
OUTPUT_DIR="./result"
VIS_DIR="./vis_output_object_topk_testset"  # 可视化输出目录
SAVE_VIS="true"  # 是否保存可视化图片 (true/false)
TOPK_MASK="${TOPK_MASK:-true}"  # 是否保存 similarity 排序的 Top-K mask (true/false)
TOPK_MASK_K="${TOPK_MASK_K:-5}"  # Top-K mask 最大保存数量
TOPK_RECALL_IOU_THRESHOLD="${TOPK_RECALL_IOU_THRESHOLD:-0.5}"  # object recall IoU 阈值

# ========== 测试配置 ==========
PRECISION="bf16"
ICR_THRESHOLDS="0.3,0.4,0.5,0.6,0.7,0.8,0.9"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,out_proj"

# ========== 调试配置 ==========
DEBUG="${DEBUG:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GPU_ID="${GPU_ID:-1}"
SPLIT="${SPLIT:-both}"  # 可选: both, easy, hard
WORKERS="${WORKERS:-12}"

# ========================================================================

cd "$(dirname "$0")/.." || exit 1

echo "========================================================================"
echo "  LLMSeg VIGOR-100K 测试"
echo "========================================================================"
echo "基础模型: ${LISA_MODEL_PATH}"
echo "微调权重: ${CHECKPOINT_PATH}"
echo "测试数据: ${TEST_DATA_DIR}"
echo "Easy JSON: ${TEST_EASY_JSON}"
echo "Hard JSON: ${TEST_HARD_JSON}"
echo "SAM候选mask: ${SAM_MASKS_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "可视化目录: ${VIS_DIR}"
echo "Top-K mask: ${TOPK_MASK}"
echo "Top-K mask K: ${TOPK_MASK_K}"
echo "Top-K recall IoU threshold: ${TOPK_RECALL_IOU_THRESHOLD}"
echo "DataLoader workers: ${WORKERS}"
echo "========================================================================"

# 构建参数
ARGS="
    --version=${LISA_MODEL_PATH}
    --checkpoint=${CHECKPOINT_PATH}
    --vision_tower=${CLIP_PATH}
    --vision_pretrained=${SAM_VIT_PATH}
    --data_dir=${TEST_DATA_DIR}
    --easy_json_file=${TEST_EASY_JSON}
    --hard_json_file=${TEST_HARD_JSON}
    --sam_masks_dir=${SAM_MASKS_DIR}
    --output_dir=${OUTPUT_DIR}
    --vis_dir=${VIS_DIR}
    --topk_mask_k=${TOPK_MASK_K}
    --topk_recall_iou_threshold=${TOPK_RECALL_IOU_THRESHOLD}
    --precision=${PRECISION}
    --icr_thresholds=${ICR_THRESHOLDS}
    --lora_r=${LORA_R}
    --lora_alpha=${LORA_ALPHA}
    --lora_dropout=${LORA_DROPOUT}
    --lora_target_modules=${LORA_TARGET_MODULES}
    --device="cuda:${GPU_ID}"
    --split=${SPLIT}
    --workers=${WORKERS}
    --use_mm_start_end
"

# 添加可视化保存参数
if [ "$SAVE_VIS" = "true" ]; then
    ARGS="${ARGS} --save_vis"
    echo "可视化保存: 开启"
fi

if [ "$TOPK_MASK" = "true" ] || [ "$TOPK_MASK" = "1" ]; then
    ARGS="${ARGS} --save_topk_masks"
    echo "Top-K mask 保存: 开启"
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
