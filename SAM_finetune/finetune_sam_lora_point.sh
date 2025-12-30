#!/bin/bash

# SAM模型LoRA微调启动脚本（使用affordance points）
# 使用工业零件标注数据集微调SAM模型

# 设置模型路径
SAM_CHECKPOINT="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# 设置数据集路径
# VIGOR-100K数据集
DATASET_TYPE="vigor"
VIGOR_ANNOTATIONS_FILE="/opt/data/private/LLMSeg/dataset/VIGOR-100K/train/all_annotations.json"
IMAGES_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/train"  # VIGOR训练集图像目录
DATASET_DIR="/opt/data/private/LLMSeg/dataset"  # 保留用于兼容性，实际不使用

# 设置输出目录
OUTPUT_DIR="./sam_output/sam_finetuned_vigor_point"

# ✅ 设置是否从checkpoint恢复训练（如果需要继续训练，设置为checkpoint路径）
# 例如：RESUME_CHECKPOINT="./sam_output/sam_finetuned_vigor_point/best_model.pth"
# 如果不需要恢复，设置为空字符串 "" 或注释掉
# 修复：从第5个epoch的checkpoint恢复（最新的checkpoint）
RESUME_CHECKPOINT="/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_vigor_point/checkpoint/checkpoint_epoch_4.pth"

# 设置GPU - 使用双GPU分布式训练
GPU_IDS="0,1"
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 检查模型文件是否存在
if [ ! -f "$SAM_CHECKPOINT" ]; then
    echo "Error: SAM checkpoint not found at $SAM_CHECKPOINT"
    exit 1
fi

# 检查数据集目录是否存在
if [ "$DATASET_TYPE" = "vigor" ]; then
    # VIGOR-100K数据集检查
    if [ ! -d "$IMAGES_DIR" ]; then
        echo "Error: VIGOR images directory not found at $IMAGES_DIR"
        exit 1
    fi
    
    if [ ! -f "$VIGOR_ANNOTATIONS_FILE" ]; then
        echo "Error: VIGOR annotations file not found at $VIGOR_ANNOTATIONS_FILE"
        exit 1
    fi
    
    MASKS_DIR="$IMAGES_DIR/masks"
    if [ ! -d "$MASKS_DIR" ]; then
        echo "Warning: VIGOR masks directory not found at $MASKS_DIR"
    fi
else
    # robot_arm数据集检查（原有逻辑）
    if [ ! -d "$IMAGES_DIR" ]; then
        echo "Error: Images directory not found at $IMAGES_DIR"
        exit 1
    fi
    
    MASKS_DIR="/opt/data/private/LLMSeg/dataset/GT_mask"
    if [ ! -d "$MASKS_DIR" ]; then
        echo "Error: Masks directory not found at $MASKS_DIR"
        exit 1
    fi
    
    # 检查是否有至少一个数据集子目录
    if [ ! -d "$IMAGES_DIR/robot_arm_01" ] && [ ! -d "$IMAGES_DIR/robot_arm_02" ] && [ ! -d "$IMAGES_DIR/robot_arm_03" ]; then
        echo "Warning: No dataset subdirectories found in $IMAGES_DIR (expected robot_arm_01, robot_arm_02, or robot_arm_03)"
    fi
fi

# 显示参数
echo "=== SAM模型LoRA微调（使用points） ==="
echo "Dataset Type: $DATASET_TYPE"
if [ "$DATASET_TYPE" = "vigor" ]; then
    echo "SAM Checkpoint: $SAM_CHECKPOINT"
    echo "VIGOR Annotations File: $VIGOR_ANNOTATIONS_FILE"
    echo "Images Directory: $IMAGES_DIR"
    echo "Masks Directory: $MASKS_DIR"
else
    echo "SAM Checkpoint: $SAM_CHECKPOINT"
    echo "Images Directory: $IMAGES_DIR (将自动遍历robot_arm_01/02/03子目录)"
    echo "Masks Directory: $MASKS_DIR (将自动遍历robot_arm_01/02/03子目录)"
fi
# 切换到脚本所在目录（需要在路径检查之前切换）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ✅ 将相对路径转换为绝对路径（用于checkpoint检查）
# 注意：这个转换需要在cd到SCRIPT_DIR之后进行
if [ -n "$RESUME_CHECKPOINT" ]; then
    # 如果路径是相对路径，转换为基于SCRIPT_DIR的绝对路径
    if [[ "$RESUME_CHECKPOINT" != /* ]]; then
        RESUME_CHECKPOINT="$SCRIPT_DIR/$RESUME_CHECKPOINT"
    fi
fi

echo "Output Directory: $OUTPUT_DIR"
echo "GPU IDs: $GPU_IDS"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resume Checkpoint: $RESUME_CHECKPOINT"
    if [ -f "$RESUME_CHECKPOINT" ]; then
        echo "  ✅ Checkpoint文件存在，将从checkpoint恢复训练"
    else
        echo "  ⚠️  Checkpoint文件不存在，将从头开始训练"
        # 如果文件不存在，清空RESUME_CHECKPOINT，避免传递无效参数
        RESUME_CHECKPOINT=""
    fi
else
    echo "Resume Checkpoint: 无（从头开始训练）"
fi
echo "SwanLab: 已启用，将记录训练曲线"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置SwanLab API Key
SWANLAB_API_KEY="BBd5HKuM6sIhTwyWmgZ6Z"
export SWANLAB_API_KEY=$SWANLAB_API_KEY

# 激活虚拟环境（优先使用项目根目录的 .venv）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "激活项目根目录虚拟环境: $PROJECT_ROOT/.venv"
    source "$PROJECT_ROOT/.venv/bin/activate"

else
    echo "警告: 未找到虚拟环境，使用系统Python"
fi

# 启动微调
if [ "$DATASET_TYPE" = "vigor" ]; then
    echo "Starting SAM LoRA fine-tuning with VIGOR-100K dataset (使用 points)..."
    # 设置PyTorch内存优化（避免内存碎片）
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # 计算GPU数量
    NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
    echo "使用 $NUM_GPUS 个GPU进行分布式训练"
    
    if [ $NUM_GPUS -gt 1 ]; then
        # 多GPU分布式训练
        torchrun --nproc_per_node=$NUM_GPUS \
            --master_port=29500 \
            "$SCRIPT_DIR/finetune_sam_lora_point.py" \
            --sam_checkpoint "$SAM_CHECKPOINT" \
            --dataset_dir "$DATASET_DIR" \
            --images_dir "$IMAGES_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --device "cuda" \
            --batch_size 3 \
            --epochs 200 \
            --lr 1e-4 \
            --weight_decay 1e-4 \
            --use_lora \
            --lora_r 32 \
            --lora_alpha 64 \
            --lora_dropout 0.1 \
            --lora_target_modules "q_proj,v_proj,k_proj,out_proj" \
            --val_split 0.1 \
            --num_workers 6 \
            --save_every 5 \
            --dataset_type "vigor" \
            --vigor_annotations_file "$VIGOR_ANNOTATIONS_FILE" \
            $([ -n "$RESUME_CHECKPOINT" ] && echo "--resume $RESUME_CHECKPOINT") \
            --swanlab_api_key "$SWANLAB_API_KEY" \
            --swanlab_project "SAM-Finetune" \
            --swanlab_experiment_name "SAM-LoRA-vigor-point"
    else
        # 单GPU训练
        python "$SCRIPT_DIR/finetune_sam_lora_point.py" \
            --sam_checkpoint "$SAM_CHECKPOINT" \
            --dataset_dir "$DATASET_DIR" \
            --images_dir "$IMAGES_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --device "cuda" \
            --batch_size 2 \
            --epochs 200 \
            --lr 1e-4 \
            --weight_decay 1e-4 \
            --use_lora \
            --lora_r 32 \
            --lora_alpha 64 \
            --lora_dropout 0.1 \
            --lora_target_modules "q_proj,v_proj,k_proj,out_proj" \
            --val_split 0.1 \
            --num_workers 6 \
            --save_every 5 \
            --dataset_type "vigor" \
            --vigor_annotations_file "$VIGOR_ANNOTATIONS_FILE" \
            $([ -n "$RESUME_CHECKPOINT" ] && echo "--resume $RESUME_CHECKPOINT") \
            --swanlab_api_key "$SWANLAB_API_KEY" \
            --swanlab_project "SAM-Finetune" \
            --swanlab_experiment_name "SAM-LoRA-vigor-point"
    fi
else
    echo "Starting SAM LoRA fine-tuning with robot_arm dataset (使用 affordance points)..."
    # 计算GPU数量
    NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
    echo "使用 $NUM_GPUS 个GPU进行分布式训练"
    
    if [ $NUM_GPUS -gt 1 ]; then
        # 多GPU分布式训练
        torchrun --nproc_per_node=$NUM_GPUS \
            --master_port=29500 \
            "$SCRIPT_DIR/finetune_sam_lora_point.py" \
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
            --dataset_type "robot_arm" \
            $([ -n "$RESUME_CHECKPOINT" ] && echo "--resume $RESUME_CHECKPOINT") \
            --swanlab_api_key "$SWANLAB_API_KEY" \
            --swanlab_project "SAM-Finetune" \
            --swanlab_experiment_name "SAM-LoRA-robot-arm-point"
    else
        # 单GPU训练
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
            --dataset_type "robot_arm" \
            $([ -n "$RESUME_CHECKPOINT" ] && echo "--resume $RESUME_CHECKPOINT") \
            --swanlab_api_key "$SWANLAB_API_KEY" \
            --swanlab_project "SAM-Finetune" \
            --swanlab_experiment_name "SAM-LoRA-robot-arm-point"
    fi
fi

echo ""
echo "Fine-tuning completed!"
echo "Model saved to: $OUTPUT_DIR"
