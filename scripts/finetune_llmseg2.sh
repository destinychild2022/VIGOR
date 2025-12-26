#! /bin/bash

# 基础模型路径（用于初始化模型结构）
llava_path="/opt/data/private/model/LISA_Plus_7b"
# CLIP 模型路径（vision tower，用于图像编码）
clip_path="/opt/data/private/model/clip-vit-large-patch14"
# SAM 模型权重路径（.pth 文件）
vision_path="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# 数据集路径（当前结构，在训练脚本中硬编码）
# 原图：/opt/data/private/LLMSeg/dataset/raw_pic
dataset_path="/opt/data/private/LLMSeg/dataset/raw_pic"
# GT mask：/opt/data/private/LLMSeg/dataset/GT_mask
gt_mask_path="/opt/data/private/LLMSeg/dataset/GT_mask"
# SAM候选：/opt/data/private/LLMSeg/dataset/sam_candidate
sam_candidate_path="/opt/data/private/LLMSeg/dataset/sam_candidate"
# 数据集包含三个视角：robot_arm_01, robot_arm_02, robot_arm_03
# 每个视角只使用前100张标注好的图片进行训练

log_path="./runs"
exp_name="finetune_llmseg_robot_arm3"
# 权重保存路径（训练过程中自动保存）
# 实际保存路径：{log_base_dir}/{exp_name}/ckpt_model
checkpoint_save_path="${log_path}/${exp_name}/ckpt_model"
# DeepSpeed checkpoint 路径（用于恢复训练，包含已微调的模型权重）
# 如果要从已微调的模型继续训练，应该指向 LLM-Seg-deepspeed
# 如果不需要恢复训练，设置为空字符串 ""
resume_path="/opt/data/private/model/LLM-Seg-deepspeed"

# ========== 调试模式控制 ==========
# 设置为 "1" 或 "true" 开启调试模式（打印所有debug信息）
# 设置为 "0" 或 "false" 关闭调试模式（不打印debug信息）
# 默认关闭调试模式
# 
# 使用方法：
#   开启调试模式：ENABLE_DEBUG=1 bash scripts/finetune_llmseg2.sh
#   关闭调试模式：bash scripts/finetune_llmseg2.sh （默认）
ENABLE_DEBUG="${ENABLE_DEBUG:-0}"

cd "$(dirname "$0")/.." || exit 1

# Prefer the project's venv deepspeed if available (avoids PATH/activation issues)
DEEPSPEED_BIN="./.venv/bin/deepspeed"
if [ -x "$DEEPSPEED_BIN" ]; then
  DS="$DEEPSPEED_BIN"
else
  DS="deepspeed"
fi

# 使用单 GPU 训练以降低内存占用
$DS --include localhost:0 \
  --master_port=24374 finetune_llmseg_copy2.py \
  --version="$llava_path" \
  --vision-tower="$clip_path" \
  --vision_pretrained="$vision_path" \
  --dataset="robot_arm" \
  --sample_rates="1" \
  --dataset_base_dir="$dataset_path" \
  --gt_mask_base_dir="$gt_mask_path" \
  --sam_masks_base_dir="$sam_candidate_path" \
  --exp_name="$exp_name" \
  --log_base_dir="$log_path" \
  --steps_per_epoch=35 \
  --lr=1e-5 \
  --epochs=70 \
  --batch_size=4 \
  --grad_accumulation_steps=2 \
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
  --align_temperature=0.05 \
  $([ "$ENABLE_DEBUG" = "1" ] || [ "$ENABLE_DEBUG" = "true" ] && echo "--debug_epoch_shapes")
