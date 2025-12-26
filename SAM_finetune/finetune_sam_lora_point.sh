#!/bin/bash

# SAM模型LoRA微调启动脚本（使用affordance points）
# 使用工业零件标注数据集微调SAM模型

# 设置模型路径
SAM_CHECKPOINT="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# 设置数据集路径
# 数据集包含：原图、mask和annotations.json
# 注意：代码会自动遍历picture目录下的robot_arm_01, robot_arm_02, robot_arm_03三个子目录
DATASET_DIR="/opt/data/private/LLMSeg/dataset"  # 保留用于兼容性，实际不使用
IMAGES_DIR="/opt/data/private/LLMSeg/dataset/raw_pic"  # 父目录，代码会自动遍历子目录

# 设置输出目录
OUTPUT_DIR="./sam_output/sam_finetuned_robot_arm_point2"

# 设置GPU
GPU_IDS="0"
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 检查模型文件是否存在
if [ ! -f "$SAM_CHECKPOINT" ]; then
    echo "Error: SAM checkpoint not found at $SAM_CHECKPOINT"
    exit 1
fi

# 检查数据集目录是否存在
# 检查原图目录（应该包含robot_arm_01, robot_arm_02, robot_arm_03子目录）
if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found at $IMAGES_DIR"
    exit 1
fi

# 检查mask目录
MASKS_DIR="/opt/data/private/LLMSeg/dataset/GT_mask"
if [ ! -d "$MASKS_DIR" ]; then
    echo "Error: Masks directory not found at $MASKS_DIR"
    exit 1
fi

# 检查是否有至少一个数据集子目录
if [ ! -d "$IMAGES_DIR/robot_arm_01" ] && [ ! -d "$IMAGES_DIR/robot_arm_02" ] && [ ! -d "$IMAGES_DIR/robot_arm_03" ]; then
    echo "Warning: No dataset subdirectories found in $IMAGES_DIR (expected robot_arm_01, robot_arm_02, or robot_arm_03)"
fi

# 显示参数
echo "=== SAM模型LoRA微调（使用affordance points） ==="
echo "SAM Checkpoint: $SAM_CHECKPOINT"
echo "Images Directory: $IMAGES_DIR (将自动遍历robot_arm_01/02/03子目录)"
echo "Masks Directory: $MASKS_DIR (将自动遍历robot_arm_01/02/03子目录)"
echo "Output Directory: $OUTPUT_DIR"
echo "GPU IDs: $GPU_IDS"
echo "SwanLab: 已启用，将记录训练曲线"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置SwanLab API Key
SWANLAB_API_KEY="BBd5HKuM6sIhTwyWmgZ6Z"
export SWANLAB_API_KEY=$SWANLAB_API_KEY

# 切换到脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 激活虚拟环境（优先使用项目根目录的 .venv）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "激活项目根目录虚拟环境: $PROJECT_ROOT/.venv"
    source "$PROJECT_ROOT/.venv/bin/activate"

else
    echo "警告: 未找到虚拟环境，使用系统Python"
fi

# 启动微调
echo "Starting SAM LoRA fine-tuning (使用 affordance points)..."
python "$SCRIPT_DIR/finetune_sam_lora_point.py" \
    --sam_checkpoint "$SAM_CHECKPOINT" \
    --dataset_dir "$DATASET_DIR" \
    --images_dir "$IMAGES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device "cuda" \
    --batch_size 8 \
    --epochs 200 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,v_proj,k_proj,out_proj" \
    --val_split 0.1 \
    --num_workers 8 \
    --save_every 20 \
    --swanlab_api_key "$SWANLAB_API_KEY" \
    --swanlab_project "SAM-Finetune" \
    --swanlab_experiment_name "SAM-LoRA-robot-arm-point"

echo ""
echo "Fine-tuning completed!"
echo "Model saved to: $OUTPUT_DIR"

