#!/bin/bash

# 测试训练好的SAM模型的分割成功率

# 设置模型路径
SAM_CHECKPOINT="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# 训练好的SAM模型checkpoint路径
TRAINED_SAM_MODEL="/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_robot_arm_point2/best_model.pth"

# 数据集路径
ANNOTATIONS_DIR="/opt/data/private/LLMSeg/dataset/GT_mask"
IMAGES_DIR="/opt/data/private/LLMSeg/dataset/raw_pic"
GT_MASKS_DIR="/opt/data/private/LLMSeg/dataset/GT_mask"

# 是否使用原始SAM权重（未训练的）进行测试
# 设置为 "true" 使用原始SAM，设置为 "false" 使用训练好的SAM
# 注意：在compute_only模式下，这个选项决定从哪个目录读取候选masks
#   - "true": 从 test_vis_sam_origin 读取（原始SAM生成的masks）
#   - "false": 从 test_vis 读取（训练好的SAM生成的masks）
USE_ORIGINAL_SAM="true"

# 输出文件（根据是否使用原始SAM自动调整文件名）
if [ "$USE_ORIGINAL_SAM" = "true" ]; then
    OUTPUT_FILE="/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_robot_arm_point2/success_rate_results_original_sam.txt"
else
    OUTPUT_FILE="/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_robot_arm_point2/success_rate_results.txt"
fi

# 可视化输出目录（训练好的SAM）
VIS_OUTPUT_DIR="/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_robot_arm_point2/test_vis"

# 原始SAM可视化输出目录
VIS_OUTPUT_DIR_ORIGINAL="/opt/data/private/LLMSeg/SAM_finetune/sam_output/sam_finetuned_robot_arm_point2/test_vis_sam_origin"

# 均匀网格点参数（参考test_automatic_mask_generator.sh）
POINTS_PER_SIDE=32        # 每边的点数（总点数为 points_per_side^2）
POINTS_PER_BATCH=64        # 每批处理的点数
PRED_IOU_THRESH=0.88       # 预测IoU阈值
STABILITY_SCORE_THRESH=0.95  # 稳定性分数阈值
BOX_NMS_THRESH=0.7        # NMS IoU阈值
MAX_AREA_RATIO=0.25       # 最大mask面积比例（相对于图像总面积）

# 设置GPU
GPU_IDS="0"
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 切换到脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 检查文件是否存在（仅计算模式不需要检查模型文件）
if [ "$COMPUTE_ONLY" != "true" ]; then
    if [ ! -f "$SAM_CHECKPOINT" ]; then
        echo "Error: SAM checkpoint not found at $SAM_CHECKPOINT"
        exit 1
    fi
    
    # 如果使用原始SAM，不需要检查训练好的模型文件
    if [ "$USE_ORIGINAL_SAM" != "true" ]; then
        if [ ! -f "$TRAINED_SAM_MODEL" ]; then
            echo "Error: Trained SAM model not found at $TRAINED_SAM_MODEL"
            exit 1
        fi
    fi
fi

if [ ! -d "$ANNOTATIONS_DIR" ]; then
    echo "Error: Annotations directory not found at $ANNOTATIONS_DIR"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found at $IMAGES_DIR"
    exit 1
fi

if [ "$COMPUTE_ONLY" = "true" ]; then
    if [ "$USE_ORIGINAL_SAM" = "true" ]; then
        if [ -n "$VIS_OUTPUT_DIR_ORIGINAL" ] && [ ! -d "$VIS_OUTPUT_DIR_ORIGINAL" ]; then
            echo "Error: Visualization output directory not found at $VIS_OUTPUT_DIR_ORIGINAL"
            exit 1
        fi
    else
        if [ -n "$VIS_OUTPUT_DIR" ] && [ ! -d "$VIS_OUTPUT_DIR" ]; then
            echo "Error: Visualization output directory not found at $VIS_OUTPUT_DIR"
            exit 1
        fi
    fi
fi

# 显示参数
if [ "$COMPUTE_ONLY" = "true" ]; then
    echo "=== SAM模型分割成功率测试（仅计算模式） ==="
else
    if [ "$USE_ORIGINAL_SAM" = "true" ]; then
        echo "=== SAM模型分割成功率测试（使用原始SAM权重） ==="
    else
        echo "=== SAM模型分割成功率测试（使用训练好的SAM权重） ==="
    fi
fi

if [ "$COMPUTE_ONLY" != "true" ]; then
    echo "SAM Checkpoint: $SAM_CHECKPOINT"
    if [ "$USE_ORIGINAL_SAM" = "true" ]; then
        echo "使用原始SAM权重: 是（未训练的）"
    else
        echo "Trained SAM Model: $TRAINED_SAM_MODEL"
        echo "使用原始SAM权重: 否（使用训练权重）"
    fi
fi
echo "Annotations Directory: $ANNOTATIONS_DIR"
echo "Images Directory: $IMAGES_DIR"
echo "GT Masks Directory: $GT_MASKS_DIR"
echo "Success Threshold: IoU >= $success_threshold"
echo "Output File: $OUTPUT_FILE"
if [ "$USE_ORIGINAL_SAM" = "true" ]; then
    if [ -n "$VIS_OUTPUT_DIR_ORIGINAL" ]; then
        echo "Visualization Output Dir (原始SAM): $VIS_OUTPUT_DIR_ORIGINAL"
    fi
else
    if [ -n "$VIS_OUTPUT_DIR" ]; then
        echo "Visualization Output Dir (训练好的SAM): $VIS_OUTPUT_DIR"
    fi
fi
if [ "$COMPUTE_ONLY" != "true" ]; then
    echo "GPU ID: $GPU_IDS"
    echo ""
    echo "均匀网格点参数:"
    echo "  - Points per side: $POINTS_PER_SIDE (总点数: $((POINTS_PER_SIDE * POINTS_PER_SIDE)))"
    echo "  - Points per batch: $POINTS_PER_BATCH"
    echo "  - IoU threshold: $PRED_IOU_THRESH"
    echo "  - Stability threshold: $STABILITY_SCORE_THRESH"
    echo "  - NMS threshold: $BOX_NMS_THRESH"
    echo "  - Max area ratio: $MAX_AREA_RATIO"
fi
echo ""

# 激活虚拟环境（如果需要）
if [ -d "../.venv" ]; then
    echo "激活虚拟环境..."
    source "../.venv/bin/activate"
else
    echo "警告: 未找到虚拟环境，使用系统Python"
fi

# 是否仅计算模式（从已保存的masks计算IoU，不重新生成）
# 设置为 "true" 启用仅计算模式，设置为 "false" 正常模式（生成masks并计算）
COMPUTE_ONLY="true"
success_threshold=0.3
# 运行测试脚本
if [ "$COMPUTE_ONLY" = "true" ]; then
    echo "开始计算（仅计算模式，从已保存的masks加载）..."
    if [ "$USE_ORIGINAL_SAM" = "true" ]; then
        python test_sam_success_rate.py \
            --sam_checkpoint "$SAM_CHECKPOINT" \
            --annotations_dir "$ANNOTATIONS_DIR" \
            --images_dir "$IMAGES_DIR" \
            --gt_masks_dir "$GT_MASKS_DIR" \
            --success_threshold "$success_threshold" \
            --output_file "$OUTPUT_FILE" \
            --vis_output_dir "$VIS_OUTPUT_DIR" \
            --vis_output_dir_original "$VIS_OUTPUT_DIR_ORIGINAL" \
            --compute_only \
            --use_original_sam
    else
        python test_sam_success_rate.py \
            --sam_checkpoint "$SAM_CHECKPOINT" \
            --annotations_dir "$ANNOTATIONS_DIR" \
            --images_dir "$IMAGES_DIR" \
            --gt_masks_dir "$GT_MASKS_DIR" \
            --success_threshold "$success_threshold" \
            --output_file "$OUTPUT_FILE" \
            --vis_output_dir "$VIS_OUTPUT_DIR" \
            --vis_output_dir_original "$VIS_OUTPUT_DIR_ORIGINAL" \
            --compute_only
    fi
else
    if [ "$USE_ORIGINAL_SAM" = "true" ]; then
        echo "开始测试（使用原始SAM权重，生成masks并计算）..."
        python test_sam_success_rate.py \
            --sam_checkpoint "$SAM_CHECKPOINT" \
            --annotations_dir "$ANNOTATIONS_DIR" \
            --images_dir "$IMAGES_DIR" \
            --gt_masks_dir "$GT_MASKS_DIR" \
            --success_threshold "$success_threshold" \
            --output_file "$OUTPUT_FILE" \
            --points_per_side "$POINTS_PER_SIDE" \
            --points_per_batch "$POINTS_PER_BATCH" \
            --pred_iou_thresh "$PRED_IOU_THRESH" \
            --stability_score_thresh "$STABILITY_SCORE_THRESH" \
            --box_nms_thresh "$BOX_NMS_THRESH" \
            --max_area_ratio "$MAX_AREA_RATIO" \
            --use_original_sam \
            --vis_output_dir "$VIS_OUTPUT_DIR" \
            --vis_output_dir_original "$VIS_OUTPUT_DIR_ORIGINAL"
    else
        echo "开始测试（使用训练好的SAM权重，生成masks并计算）..."
        python test_sam_success_rate.py \
            --sam_model_path "$TRAINED_SAM_MODEL" \
            --sam_checkpoint "$SAM_CHECKPOINT" \
            --annotations_dir "$ANNOTATIONS_DIR" \
            --images_dir "$IMAGES_DIR" \
            --gt_masks_dir "$GT_MASKS_DIR" \
            --success_threshold "$success_threshold" \
            --use_lora \
            --output_file "$OUTPUT_FILE" \
            --points_per_side "$POINTS_PER_SIDE" \
            --points_per_batch "$POINTS_PER_BATCH" \
            --pred_iou_thresh "$PRED_IOU_THRESH" \
            --stability_score_thresh "$STABILITY_SCORE_THRESH" \
            --box_nms_thresh "$BOX_NMS_THRESH" \
            --max_area_ratio "$MAX_AREA_RATIO" \
            --vis_output_dir "$VIS_OUTPUT_DIR" \
            --vis_output_dir_original "$VIS_OUTPUT_DIR_ORIGINAL"
    fi
fi

echo ""
echo "测试完成！结果保存在: $OUTPUT_FILE"
