#! /bin/bash

# 基础模型路径（用于初始化模型结构）
llava_path="/opt/data/private/model/LISA_Plus_7b"
# CLIP 模型路径（vision tower，用于图像编码）
clip_path="/opt/data/private/model/clip-vit-large-patch14"
# SAM 模型权重路径（.pth 文件）
vision_path="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# VIGOR-100K数据集路径
vigor_data_base_dir="/opt/data/private/LLMSeg/dataset/VIGOR-100K"
vigor_split="train"  # train/test/unseen
vigor_val_split="test"  # 验证集使用test split

log_path="./runs"
exp_name="finetune_llmseg_vigor"
# 权重保存路径（训练过程中自动保存）
# 实际保存路径：{log_base_dir}/{exp_name}/ckpt_model
checkpoint_save_path="${log_path}/${exp_name}/ckpt_model"
# DeepSpeed checkpoint 路径（用于恢复训练，包含已微调的模型权重）
# 如果要从已微调的模型继续训练，应该指向 LLM-Seg-deepspeed
# 如果不需要恢复训练，设置为空字符串 ""
resume_path=""

# ========== 调试模式控制 ==========
# 设置为 "1" 或 "true" 开启调试模式（打印所有debug信息）
# 设置为 "0" 或 "false" 关闭调试模式（不打印debug信息）
# 默认关闭调试模式
# 
# 使用方法：
#   开启调试模式：ENABLE_DEBUG=1 bash scripts/finetune_llmseg_vigor.sh
#   关闭调试模式：bash scripts/finetune_llmseg_vigor.sh （默认）
ENABLE_DEBUG="${ENABLE_DEBUG:-0}"

# ========== CPU 调试模式 ==========
# 设置为 "1" 开启CPU模式（绕过DeepSpeed，使用纯Python运行）
# 注意：CPU模式只用于快速调试数据加载逻辑，不适合实际训练
# 
# 使用方法：
#   CPU调试模式：CPU_ONLY=1 bash scripts/finetune_llmseg_vigor.sh
CPU_ONLY="${CPU_ONLY:-0}"

cd "$(dirname "$0")/.." || exit 1

# Prefer the project's venv python/deepspeed if available
PYTHON_BIN="./.venv/bin/python"
DEEPSPEED_BIN="./.venv/bin/deepspeed"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

if [ ! -x "$DEEPSPEED_BIN" ]; then
  DEEPSPEED_BIN="deepspeed"
fi

# 根据CPU_ONLY选择运行方式
if [ "$CPU_ONLY" = "1" ] || [ "$CPU_ONLY" = "true" ]; then
  echo "=========================================="
  echo "  CPU调试模式 (无GPU)"
  echo "  注意: DeepSpeed已禁用, 使用fp32精度"
  echo "=========================================="
  
  # 设置环境变量禁用CUDA
  export CUDA_VISIBLE_DEVICES=""
  
  $PYTHON_BIN finetune_llmseg_vigor.py \
    --version="$llava_path" \
    --vision-tower="$clip_path" \
    --vision_pretrained="$vision_path" \
    --dataset="vigor" \
    --sample_rates="1" \
    --vigor_data_base_dir="$vigor_data_base_dir" \
    --vigor_split="$vigor_split" \
    --vigor_val_split="$vigor_val_split" \
    --exp_name="${exp_name}_cpu_debug" \
    --log_base_dir="$log_path" \
    --steps_per_epoch=2 \
    --lr=1e-5 \
    --epochs=1 \
    --batch_size=1 \
    --grad_accumulation_steps=1 \
    --workers=4 \
    --lora_r=8 \
    --lora_alpha=16 \
    --lora_dropout=0.1 \
    --lora_target_modules="q_proj,k_proj,v_proj,out_proj" \
    --precision="fp32" \
    --resume="$resume_path" \
    --train_vis_dir="train_vis" \
    --val_vis_dir="val_vis" \
    --eval_vis_dir="eval_vis_iop" \
    --align_temperature=0.05 \
    --debug_epoch_shapes
else
  # 正常GPU训练模式 (DeepSpeed)
  $DEEPSPEED_BIN --include localhost:0,1 \
    --master_port=24374 finetune_llmseg_vigor.py \
    --version="$llava_path" \
    --vision-tower="$clip_path" \
    --vision_pretrained="$vision_path" \
    --dataset="vigor" \
    --sample_rates="1" \
    --vigor_data_base_dir="$vigor_data_base_dir" \
    --vigor_split="$vigor_split" \
    --vigor_val_split="$vigor_val_split" \
    --exp_name="$exp_name" \
    --log_base_dir="$log_path" \
    --steps_per_epoch=250 \
    --lr=2e-5 \
    --epochs=100 \
    --batch_size=16 \
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
    --align_temperature=0.05 \
    $([ "$ENABLE_DEBUG" = "1" ] || [ "$ENABLE_DEBUG" = "true" ] && echo "--debug_epoch_shapes")
fi

