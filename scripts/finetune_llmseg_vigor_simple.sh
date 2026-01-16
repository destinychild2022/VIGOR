#! /bin/bash

# ========================================================================
# 简化版 VIGOR 训练脚本 (使用 finetune_llmseg_vigor_simple.py)
# - 去掉了复杂的 safetensors 保存逻辑
# - 去掉了内存清理逻辑
# - 只使用 DeepSpeed 原生的 checkpoint 保存
# ========================================================================

# 基础模型路径
llava_path="/opt/data/private/model/LISA_Plus_7b"
clip_path="/opt/data/private/model/clip-vit-large-patch14"
vision_path="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# VIGOR-100K数据集路径
vigor_data_base_dir="/opt/data/private/LLMSeg/dataset/VIGOR-100K"
vigor_split="train"
vigor_val_split="test"

# SAM候选masks路径
vigor_train_sam_masks="${vigor_data_base_dir}/train_masks_sam_0.8_0.8"
vigor_val_sam_masks="${vigor_data_base_dir}/test_mask/sam_masks"

log_path="./runs"
exp_name="finetune_llmseg_vigor_simple"

# DeepSpeed checkpoint 路径
resume_path=""

cd "$(dirname "$0")/.." || exit 1

# Prefer the project's venv
DEEPSPEED_BIN="./.venv/bin/deepspeed"
if [ ! -x "$DEEPSPEED_BIN" ]; then
  DEEPSPEED_BIN="deepspeed"
fi

# 双卡训练
$DEEPSPEED_BIN --include localhost:0,1 \
  --master_port=24374 finetune_llmseg_vigor_simple.py \
  --version="$llava_path" \
  --vision-tower="$clip_path" \
  --vision_pretrained="$vision_path" \
  --dataset="vigor" \
  --sample_rates="1" \
  --vigor_data_base_dir="$vigor_data_base_dir" \
  --vigor_split="$vigor_split" \
  --vigor_val_split="$vigor_val_split" \
  --vigor_train_sam_masks_dir="$vigor_train_sam_masks" \
  --vigor_val_sam_masks_dir="$vigor_val_sam_masks" \
  --exp_name="$exp_name" \
  --log_base_dir="$log_path" \
  --steps_per_epoch=1000 \
  --lr=2e-5 \
  --epochs=70 \
  --batch_size=8 \
  --grad_accumulation_steps=1 \
  --workers=4 \
  --lora_r=8 \
  --lora_alpha=16 \
  --lora_dropout=0.1 \
  --lora_target_modules="q_proj,k_proj,v_proj,out_proj" \
  --precision="bf16" \
  --visualize \
  --resume="$resume_path" \
  --train_vis_dir="train_vis" \
  --val_vis_dir="val_vis" \
  --eval_vis_dir="eval_vis_iop" \
  --align_temperature=0.05
