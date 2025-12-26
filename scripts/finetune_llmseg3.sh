#! /bin/bash


# 基础模型路径（用于初始化模型结构）
llava_path="/mnt/data-oss/rap-prod-bak/GLOVER/model/LISA_Plus_7b"
# CLIP 模型路径（vision tower，用于图像编码）
clip_path="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2"
# SAM 模型权重路径（.pth 文件）
vision_path="/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth"
# 数据集路径
dataset_path="/mnt/data-cpfs/workspace_xl/code/GLOVER-ZH/SAM_finetune/dataset"
sam_masks_path="/mnt/data-cpfs/workspace_xl/code/GLOVER-ZH/SAM_finetune/test_output_point"
log_path="./runs"
exp_name="finetune_llmseg_robot_arm"
# 权重保存路径（训练过程中自动保存）
# 实际保存路径：{log_base_dir}/{exp_name}/ckpt_model
checkpoint_save_path="${log_path}/${exp_name}/ckpt_model"
# DeepSpeed checkpoint 路径（用于恢复训练，包含已微调的模型权重）
# 如果要从已微调的模型继续训练，应该指向 LLM-Seg-deepspeed
# 如果不需要恢复训练，设置为空字符串 ""
resume_path="/mnt/data-oss/rap-prod-bak/GLOVER/model/LLM-Seg-deepspeed"

cd "$(dirname "$0")/.." || exit 1

deepspeed --include localhost:1,2,3 \
  --master_port=24374 finetune_llmseg_copy.py \
  --version="$llava_path" \
  --vision-tower="$clip_path" \
  --dataset_dir="$dataset_path" \
  --sam_masks_dir="$sam_masks_path" \
  --vision_pretrained="$vision_path" \
  --dataset="sem_seg||refer_seg||reason_seg" \
  --sample_rates="9,3,1" \
  --exp_name="$exp_name" \
  --log_base_dir="$log_path" \
  --steps_per_epoch=25 \
  --lr=1e-5 \
  --epochs=70 \
  --batch_size=8 \
  --workers=16 \
  --lora_r=32 \
  --lora_alpha=64 \
  --lora_dropout=0.1 \
  --lora_target_modules="q_proj,k_proj,v_proj,out_proj" \
  --resume="$resume_path" \
  --train_vis_dir="/mnt/data-cpfs/workspace_xl/code/LLMSeg/runs/finetune_llmseg_robot_arm/train_vis3" \
  --val_vis_dir="/mnt/data-cpfs/workspace_xl/code/LLMSeg/runs/finetune_llmseg_robot_arm/val_vis3" \
  --eval_vis_dir="/mnt/data-cpfs/workspace_xl/code/LLMSeg/runs/finetune_llmseg_robot_arm/eval_vis_iop" \
  --align_temperature=0.05 \
