#! /bin/bash

# ========================================================================
# 简化版 VIGOR 训练脚本 (使用 finetune_llmseg_vigor_simple.py)
# - 去掉了复杂的 safetensors 保存逻辑
# - 去掉了内存清理逻辑
# - 只使用 DeepSpeed 原生的 checkpoint 保存
# ========================================================================

# ========== 模型路径配置 ==========
MODEL_PATH="/root/autodl-tmp/model/LISA_Plus_7b"
CLIP_PATH="/root/autodl-tmp/model/clip-vit-large-patch14"
VISION_PATH="/root/autodl-tmp/model/SAM-vit-h/sam_vit_h_4b8939.pth"
export TORCH_HOME="/root/autodl-tmp/torch_cache"

# ========== VIGOR 数据集配置 ==========
VIGOR_DATA_DIR="/root/autodl-tmp/VIGOR-100K_new"
VIGOR_TRAIN_SPLIT="train"
VIGOR_VAL_SPLIT="test"
VIGOR_EASY_JSON_FILE="open_vocab_grasp_easy_object_1.json"
VIGOR_HARD_JSON_FILE="open_vocab_grasp_hard_object_1.json"
# SAM候选masks路径
VIGOR_TRAIN_SAM_MASKS="/root/autodl-tmp/train_masks_sam_0.8_0.8"
VIGOR_VAL_SAM_MASKS="/root/autodl-tmp/test_mask/sam_masks3"
# 是否每个 epoch 后运行验证。false 时只保存 ckpt_model/latest。
ENABLE_VALIDATION=false
# 是否只使用 hard 样本 (不使用 easy 样本)
VIGOR_ONLY_HARD=false

# ========== 输出配置 ==========
LOG_DIR="/root/autodl-tmp/runs"
EXP_NAME="finetune_llmseg_vigor_simple-spatial-object"
TRAIN_VIS_DIR="train_vis"
VAL_VIS_DIR="val_vis"
EVAL_VIS_DIR="eval_vis_iop"

# ========== 训练超参数 ==========
EPOCHS=20
STEPS_PER_EPOCH=1000
BATCH_SIZE=16
GRAD_ACCUMULATION_STEPS=1
TRAIN_WORKERS=10
VAL_WORKERS=4
LR=2e-5
PRECISION="bf16"
ALIGN_TEMP=0.05
VIGOR_MAX_INSTRUCTIONS=3

# 验证集配置 (test 分片中 scene ID <= 1000 的所有 Easy+Hard 样本)
VAL_MAX_SCENE_ID=1000
# 每个 epoch 可视化样本数
MAX_VIS_SAMPLES=4

# ========== LoRA 配置 ==========
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,out_proj"

# ========== GPU 配置 ==========
GPU_IDS="0"
MASTER_PORT=24375

# ========== Checkpoint 配置 ==========
# 留空时训练脚本会优先从 ${LOG_DIR}/${EXP_NAME}/ckpt_model/latest 自动恢复。
RESUME_PATH=""
# 默认只每轮更新 latest；需要验证集最优权重时改为 true，并保持 ENABLE_VALIDATION=true。
SAVE_BEST=false

# ========== 分布式超时配置 ==========
# 60 minutes, to avoid validation all-reduce timeout when ranks finish unevenly.
export NCCL_TIMEOUT=7200

# ========================================================================

cd "$(dirname "$0")/.." || exit 1

# Prefer the project's venv
DEEPSPEED_BIN="./.venv3124/bin/deepspeed"
if [ ! -x "$DEEPSPEED_BIN" ]; then
  DEEPSPEED_BIN="deepspeed"
fi

echo "========================================================================"
echo "  VIGOR 简化版微调训练"
echo "========================================================================"
echo "模型路径: ${MODEL_PATH}"
echo "Torch Hub缓存: ${TORCH_HOME}/hub"
echo "数据目录: ${VIGOR_DATA_DIR}"
echo "Easy JSON: ${VIGOR_EASY_JSON_FILE}"
echo "Hard JSON: ${VIGOR_HARD_JSON_FILE}"
echo "GPU 显卡: ${GPU_IDS}"
echo "实验名称: ${EXP_NAME}"
echo "训练Workers: ${TRAIN_WORKERS}"
echo "验证Workers: ${VAL_WORKERS}"
echo "验证开关: ${ENABLE_VALIDATION}"
if [ "${ENABLE_VALIDATION}" = true ] && [ "${SAVE_BEST}" = true ]; then
  echo "权重保存: ckpt_model/latest (最新) + ckpt_model/best (验证集最优)"
else
  echo "权重保存: ckpt_model/latest (每轮更新)"
fi
if [ "${ENABLE_VALIDATION}" = true ]; then
  echo "最大验证场景 ID: ${VAL_MAX_SCENE_ID}"
  echo "验证数据: Easy + Hard 混合验证"
else
  echo "验证数据: 已关闭"
fi
echo "========================================================================"

# 额外的可选参数
EXTRA_ARGS=""
if [ "${VIGOR_ONLY_HARD}" = true ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --vigor_only_hard"
fi
if [ "${ENABLE_VALIDATION}" != true ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --no_eval"
fi
if [ "${ENABLE_VALIDATION}" = true ] && [ "${SAVE_BEST}" = true ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --save_best"
fi

# 执行训练
$DEEPSPEED_BIN --include localhost:${GPU_IDS} \
  --master_port=${MASTER_PORT} finetune_llmseg_vigor_simple.py \
  --version="${MODEL_PATH}" \
  --vision-tower="${CLIP_PATH}" \
  --vision_pretrained="${VISION_PATH}" \
  --dataset="vigor" \
  --sample_rates="1" \
  --vigor_data_base_dir="${VIGOR_DATA_DIR}" \
  --vigor_easy_json_file="${VIGOR_EASY_JSON_FILE}" \
  --vigor_hard_json_file="${VIGOR_HARD_JSON_FILE}" \
  --vigor_split="${VIGOR_TRAIN_SPLIT}" \
  --vigor_val_split="${VIGOR_VAL_SPLIT}" \
  --vigor_train_sam_masks_dir="${VIGOR_TRAIN_SAM_MASKS}" \
  --vigor_val_sam_masks_dir="${VIGOR_VAL_SAM_MASKS}" \
  --vigor_val_max_samples=${VAL_MAX_SCENE_ID} \
  --max_vis_samples=${MAX_VIS_SAMPLES} \
  --exp_name="${EXP_NAME}" \
  --log_base_dir="${LOG_DIR}" \
  --steps_per_epoch=${STEPS_PER_EPOCH} \
  --lr=${LR} \
  --epochs=${EPOCHS} \
  --batch_size=${BATCH_SIZE} \
  --grad_accumulation_steps=${GRAD_ACCUMULATION_STEPS} \
  --workers=${TRAIN_WORKERS} \
  --val_workers=${VAL_WORKERS} \
  --lora_r=${LORA_R} \
  --lora_alpha=${LORA_ALPHA} \
  --lora_dropout=${LORA_DROPOUT} \
  --lora_target_modules="${LORA_TARGET_MODULES}" \
  --precision="${PRECISION}" \
  --visualize \
  --resume="${RESUME_PATH}" \
  --train_vis_dir="${TRAIN_VIS_DIR}" \
  --val_vis_dir="${VAL_VIS_DIR}" \
  --eval_vis_dir="${EVAL_VIS_DIR}" \
  --align_temperature=${ALIGN_TEMP} \
  --vigor_max_instructions=${VIGOR_MAX_INSTRUCTIONS} \
  --iou_selection_only \
  ${EXTRA_ARGS}

echo "========================================================================"
echo "  训练结束"
echo "========================================================================"
