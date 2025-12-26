#!/bin/bash

# 测试脚本：加载训练好的 DeepSpeed checkpoint 并在300张标注图片上计算IoU

# 基础模型路径
llava_path="/opt/data/private/model/LISA_Plus_7b"
clip_path="/opt/data/private/model/clip-vit-large-patch14"
vision_path="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# 数据集路径
dataset_path="/opt/data/private/LLMSeg/dataset/raw_pic"
gt_mask_path="/opt/data/private/LLMSeg/dataset/GT_mask"
sam_candidate_path="/opt/data/private/LLMSeg/dataset/sam_candidate"

# Checkpoint路径
checkpoint_path="/opt/data/private/LLMSeg/runs/finetune_llmseg_robot_arm2/ckpt_model"

# 输出文件路径
output_file="./test_results_iou.txt"

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

# 激活虚拟环境
if [ -d ".venv" ]; then
    echo "激活项目根目录虚拟环境: $(pwd)/.venv"
    source .venv/bin/activate
else
    echo "警告: 未找到虚拟环境，使用系统Python"
fi

# Prefer the project's venv deepspeed if available
DEEPSPEED_BIN="./.venv/bin/deepspeed"
if [ -x "$DEEPSPEED_BIN" ]; then
    DS="$DEEPSPEED_BIN"
else
    DS="deepspeed"
fi

echo "=== LLMSeg 模型测试 ==="
echo "Checkpoint: $checkpoint_path"
echo "数据集: $dataset_path"
echo "输出文件: $output_file"
echo "模式: DeepSpeed引擎加载（启用CPU offload，支持大模型）"
echo "说明: 30GB checkpoint无法完全放入23GB GPU，使用CPU offload自动管理内存"
echo ""

# 检查并设置进程内存限制（ulimit）
echo "【检查进程内存限制】"
echo "  注意：即使系统内存充足，进程内存限制也可能导致OOM"
echo ""

# 检查虚拟内存限制
CURRENT_ULIMIT_V=$(ulimit -v 2>/dev/null || echo "unlimited")
if [ "$CURRENT_ULIMIT_V" = "unlimited" ] || [ -z "$CURRENT_ULIMIT_V" ]; then
    echo "  虚拟内存限制(ulimit -v): 无限制 ✓"
else
    CURRENT_ULIMIT_V_GB=$((CURRENT_ULIMIT_V / 1024 / 1024))
    echo "  虚拟内存限制(ulimit -v): ${CURRENT_ULIMIT_V_GB} GB"
    echo "  ⚠️  警告：进程内存限制可能不足（30GB checkpoint需要~75GB峰值内存）"
    echo "  尝试解除限制..."
    ulimit -v unlimited 2>/dev/null && echo "  ✓ 已解除虚拟内存限制" || echo "  ✗ 无法解除限制（可能需要root权限）"
fi

# 检查数据段限制（也可能影响内存分配）
CURRENT_ULIMIT_D=$(ulimit -d 2>/dev/null || echo "unlimited")
if [ "$CURRENT_ULIMIT_D" = "unlimited" ] || [ -z "$CURRENT_ULIMIT_D" ]; then
    echo "  数据段限制(ulimit -d): 无限制 ✓"
else
    CURRENT_ULIMIT_D_GB=$((CURRENT_ULIMIT_D / 1024 / 1024))
    echo "  数据段限制(ulimit -d): ${CURRENT_ULIMIT_D_GB} GB"
    echo "  尝试解除限制..."
    ulimit -d unlimited 2>/dev/null && echo "  ✓ 已解除数据段限制" || echo "  ✗ 无法解除限制"
fi

# 检查文件大小限制（虽然不直接影响内存，但可能影响checkpoint读取）
CURRENT_ULIMIT_F=$(ulimit -f 2>/dev/null || echo "unlimited")
if [ "$CURRENT_ULIMIT_F" != "unlimited" ] && [ -n "$CURRENT_ULIMIT_F" ]; then
    CURRENT_ULIMIT_F_GB=$((CURRENT_ULIMIT_F / 1024 / 1024))
    if [ $CURRENT_ULIMIT_F_GB -lt 50 ]; then
        echo "  文件大小限制(ulimit -f): ${CURRENT_ULIMIT_F_GB} GB"
        echo "  尝试解除限制..."
        ulimit -f unlimited 2>/dev/null && echo "  ✓ 已解除文件大小限制" || echo "  ✗ 无法解除限制"
    fi
fi

echo ""

# 检查系统内存
echo ""
echo "【系统内存情况】"
if command -v free &> /dev/null; then
    free -h | head -2
else
    echo "  无法检查（free命令不可用）"
fi

echo ""

# 使用单 GPU 测试（DeepSpeed引擎 + CPU offload）
# 原因：30GB checkpoint无法完全放入23GB GPU，需要使用CPU offload
$DS --include localhost:0 \
    --master_port=24375 test_llmseg_checkpoint.py \
    --version="$llava_path" \
    --vision-tower="$clip_path" \
    --vision_pretrained="$vision_path" \
    --checkpoint="$checkpoint_path" \
    --dataset_base_dir="$dataset_path" \
    --gt_mask_base_dir="$gt_mask_path" \
    --sam_masks_base_dir="$sam_candidate_path" \
    --image_size=896 \
    --model_max_length=512 \
    --precision="bf16" \
    --lora_r=8 \
    --lora_alpha=16 \
    --lora_dropout=0.1 \
    --lora_target_modules="q_proj,k_proj,v_proj,out_proj" \
    --threshold=0.5 \
    --output_file="$output_file" \
    --local_rank=0 \
    --workers=0 \
    --use_deepspeed_load
    # 使用DeepSpeed引擎加载模式：
    # 1. 启用CPU offload，参数存储在CPU，推理时按需加载到GPU
    # 2. 支持加载大于GPU显存的checkpoint（30GB > 23GB GPU）
    # 3. 自动管理内存，避免OOM错误

echo ""
echo "测试完成！结果保存在: $output_file"
