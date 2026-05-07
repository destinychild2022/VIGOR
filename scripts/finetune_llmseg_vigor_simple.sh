#! /bin/bash

# ========================================================================
# 简化版 VIGOR 训练脚本 (使用 finetune_llmseg_vigor_simple.py)
# - 去掉了复杂的 safetensors 保存逻辑
# - 去掉了内存清理逻辑
# - 只使用 DeepSpeed 原生的 checkpoint 保存
# ========================================================================

# ========== 模型路径配置 ==========
MODEL_PATH="/opt/data/private/model/LISA_Plus_7b"
CLIP_PATH="/opt/data/private/model/clip-vit-large-patch14"
VISION_PATH="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# ========== VIGOR 数据集配置 ==========
VIGOR_DATA_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K_new"
VIGOR_TRAIN_SPLIT="train"
VIGOR_VAL_SPLIT="test"
VIGOR_EASY_JSON="open_vocab_grasp_easy_object_2.json"
VIGOR_HARD_JSON="open_vocab_grasp_hard_object_2.json"
# SAM候选masks路径
VIGOR_TRAIN_SAM_MASKS="/opt/data/private/LLMSeg/dataset/VIGOR-100K/train_masks_sam_0.8_0.8"
VIGOR_VAL_SAM_MASKS="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test_mask/sam_masks3"
# 是否只使用 hard 样本 (不使用 easy 样本)
VIGOR_ONLY_HARD=false

# ========== 输出配置 ==========
LOG_DIR="./runs"
EXP_NAME="finetune_llmseg_vigor_simple-newdata"
TRAIN_VIS_DIR="train_vis"
VAL_VIS_DIR="val_vis"
EVAL_VIS_DIR="eval_vis_iop"
DISABLE_TENSORBOARD=true

# ========== 训练超参数 ==========
EPOCHS=20
STEPS_PER_EPOCH=1000
BATCH_SIZE=8
GRAD_ACCUMULATION_STEPS=1
LR=2e-5
PRECISION="bf16"
ALIGN_TEMP=0.05
VIGOR_MAX_INSTRUCTIONS=3

# 验证集配置 (test 分片中 scene ID <= 1000 的所有 Easy+Hard 样本)
RUN_VALIDATION=false
VAL_MAX_SCENE_ID=1000
# 每个 epoch 可视化样本数
MAX_VIS_SAMPLES=4

# ========== LoRA 配置 ==========
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,out_proj"

# ========== GPU 配置 ==========
GPU_IDS="0,1"
MASTER_PORT=24375

# ========== Checkpoint 配置 ==========
RESUME_PATH=""
CHECKPOINT_SAVE_INTERVAL=5
SAVE_ONLY_TARGET_EPOCH=true
TARGET_SAVE_EPOCH=20

# ========== 分布式超时配置 ==========
DISTRIBUTED_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=$((DISTRIBUTED_TIMEOUT_SEC * 1000))

# ========================================================================

cd "$(dirname "$0")/.." || exit 1

# Prefer the project's venv
DEEPSPEED_BIN="./.venv/bin/deepspeed"
if [ ! -x "$DEEPSPEED_BIN" ]; then
  DEEPSPEED_BIN="deepspeed"
fi

echo "========================================================================"
echo "  VIGOR 简化版微调训练"
echo "========================================================================"
echo "模型路径: ${MODEL_PATH}"
echo "数据目录: ${VIGOR_DATA_DIR}"
echo "GPU 显卡: ${GPU_IDS}"
echo "实验名称: ${EXP_NAME}"
echo "分布式超时: ${DISTRIBUTED_TIMEOUT_SEC}s"
echo "Easy JSON: ${VIGOR_EASY_JSON}"
echo "Hard JSON: ${VIGOR_HARD_JSON}"
if [ "${SAVE_ONLY_TARGET_EPOCH}" = true ]; then
  echo "权重保存: 每轮保存最新 checkpoint，仅保留一个；到 epoch_${TARGET_SAVE_EPOCH} 后自动停止"
else
  echo "权重保存: ckpt_model/best (最优) + 每 ${CHECKPOINT_SAVE_INTERVAL} 轮定期存档"
fi
echo "最大验证场景 ID: ${VAL_MAX_SCENE_ID}"
if [ "${RUN_VALIDATION}" = true ]; then
  echo "验证数据: Easy + Hard 混合验证"
else
  echo "验证数据: 已关闭每轮验证"
fi
echo "========================================================================"

# 额外的可选参数
EXTRA_ARGS=""
if [ "${VIGOR_ONLY_HARD}" = true ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --vigor_only_hard"
fi
if [ "${RUN_VALIDATION}" != true ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --no_eval"
fi
if [ "${SAVE_ONLY_TARGET_EPOCH}" = true ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --save_only_target_epoch --target_save_epoch=${TARGET_SAVE_EPOCH}"
fi
if [ "${DISABLE_TENSORBOARD}" = true ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --disable_tensorboard"
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
  --vigor_split="${VIGOR_TRAIN_SPLIT}" \
  --vigor_val_split="${VIGOR_VAL_SPLIT}" \
  --vigor_easy_json_file="${VIGOR_EASY_JSON}" \
  --vigor_hard_json_file="${VIGOR_HARD_JSON}" \
  --vigor_train_sam_masks_dir="${VIGOR_TRAIN_SAM_MASKS}" \
  --vigor_val_sam_masks_dir="${VIGOR_VAL_SAM_MASKS}" \
  --vigor_val_max_samples=${VAL_MAX_SCENE_ID} \
  --max_vis_samples=${MAX_VIS_SAMPLES} \
  --exp_name="${EXP_NAME}" \
  --log_base_dir="${LOG_DIR}" \
  --steps_per_epoch=${STEPS_PER_EPOCH} \
  --lr=${LR} \
  --distributed_timeout_sec=${DISTRIBUTED_TIMEOUT_SEC} \
  --epochs=${EPOCHS} \
  --checkpoint_save_interval=${CHECKPOINT_SAVE_INTERVAL} \
  --batch_size=${BATCH_SIZE} \
  --grad_accumulation_steps=${GRAD_ACCUMULATION_STEPS} \
  --workers=12 \
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
