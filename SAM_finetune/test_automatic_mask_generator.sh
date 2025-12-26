#!/bin/bash

# 使用训练好的SAM模型生成所有mask的测试脚本

# 设置模型路径
SAM_CHECKPOINT="/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# 设置训练好的checkpoint路径（如果USE_ORIGINAL_SAM=true则不需要）
TRAINED_CHECKPOINT="/mnt/data-oss/rap-prod-bak/GLOVER/output/sam_finetuned_robot_arm_point2/best_model.pth"

# 设置测试图像目录
TEST_IMAGES_DIR="/mnt/data-cpfs/workspace_xl/code/GLOVER-ZH/SAM_finetune/dataset/picture/robot_arm_01"

# 设置输出目录
OUTPUT_DIR="/mnt/data-cpfs/workspace_xl/code/GLOVER-ZH/SAM_finetune/test_output_point/robot_arm_01"

# 设置GPU
GPU_IDS="2"
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 是否使用原始SAM权重（不加载训练权重）
# 设置为 "true" 使用原始SAM权重，设置为 "false" 使用训练权重
USE_ORIGINAL_SAM="false"

# 设置测试模式：affordance点模式、bbox模式 或 均匀网格点模式
# 可选值: "affordance"、"bbox" 或 "grid"
TEST_MODE="grid"  # 改为 "affordance" 使用affordance点模式，改为 "bbox" 使用bbox模式

# 网格点模式参数（仅在TEST_MODE="grid"时使用）
POINTS_PER_SIDE=32        # 每边的点数（总点数为 points_per_side^2）
POINTS_PER_BATCH=64        # 每批处理的点数
PRED_IOU_THRESH=0.85       # 预测IoU阈值
STABILITY_SCORE_THRESH=0.86  # 稳定性分数阈值
BOX_NMS_THRESH=0.7        # NMS IoU阈值
MAX_AREA_RATIO=0.2       # 最大mask面积比例（相对于图像总面积），超过此比例的mask将被过滤，默认0.25（1/4）

# 标注文件路径（仅在TEST_MODE="affordance"或"bbox"时使用）
ANNOTATIONS_FILE="/mnt/data-cpfs/workspace_xl/code/GLOVER-ZH/SAM_finetune/dataset/mask/robot_arm_03/annotations.json"

# 检查文件是否存在
if [ ! -f "$SAM_CHECKPOINT" ]; then
    echo "Error: SAM checkpoint not found at $SAM_CHECKPOINT"
    exit 1
fi

# 只有在不使用原始SAM时才检查训练权重文件
if [ "$USE_ORIGINAL_SAM" != "true" ]; then
    if [ ! -f "$TRAINED_CHECKPOINT" ]; then
        echo "Error: Trained checkpoint not found at $TRAINED_CHECKPOINT"
        exit 1
    fi
fi

if [ ! -d "$TEST_IMAGES_DIR" ]; then
    echo "Error: Test images directory not found at $TEST_IMAGES_DIR"
    exit 1
fi

# 只在affordance或bbox模式下检查标注文件
if [ "$TEST_MODE" = "affordance" ] || [ "$TEST_MODE" = "bbox" ]; then
    if [ ! -f "$ANNOTATIONS_FILE" ]; then
        echo "Error: Annotations file not found at $ANNOTATIONS_FILE"
        exit 1
    fi
fi

# 显示参数
if [ "$TEST_MODE" = "affordance" ]; then
    echo "=== SAM自动Mask生成测试（使用Affordance点） ==="
    TEST_MODE_DESC="使用标注中的affordance点"
elif [ "$TEST_MODE" = "bbox" ]; then
    echo "=== SAM自动Mask生成测试（使用Bbox） ==="
    TEST_MODE_DESC="使用标注中的bbox（与训练时一致）"
else
    echo "=== SAM自动Mask生成测试（使用均匀网格点） ==="
    TEST_MODE_DESC="使用均匀网格点分布生成所有mask"
fi

echo "SAM Checkpoint: $SAM_CHECKPOINT"
if [ "$USE_ORIGINAL_SAM" = "true" ]; then
    echo "使用原始SAM权重: 是（不加载训练权重）"
else
    echo "Trained Checkpoint: $TRAINED_CHECKPOINT"
    echo "使用原始SAM权重: 否（使用训练权重）"
fi
echo "Test Images Directory: $TEST_IMAGES_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "GPU ID: $GPU_IDS"
echo "测试模式: $TEST_MODE_DESC"

if [ "$TEST_MODE" = "grid" ]; then
    echo "网格点参数:"
    echo "  - Points per side: $POINTS_PER_SIDE (总点数: $((POINTS_PER_SIDE * POINTS_PER_SIDE)))"
    echo "  - Points per batch: $POINTS_PER_BATCH"
    echo "  - IoU threshold: $PRED_IOU_THRESH"
    echo "  - Stability threshold: $STABILITY_SCORE_THRESH"
    echo "  - NMS threshold: $BOX_NMS_THRESH"
    echo "  - Max area ratio: $MAX_AREA_RATIO (过滤面积超过图像${MAX_AREA_RATIO}倍的mask)"
elif [ "$TEST_MODE" = "affordance" ] || [ "$TEST_MODE" = "bbox" ]; then
    echo "Annotations File: $ANNOTATIONS_FILE"
fi
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 激活虚拟环境（如果需要）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/.venv" ]; then
    echo "激活虚拟环境..."
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    echo "警告: 未找到虚拟环境，使用系统Python"
fi

# 运行测试脚本
if [ "$TEST_MODE" = "affordance" ]; then
    echo "开始生成masks（使用affordance点）..."
    if [ "$USE_ORIGINAL_SAM" = "true" ]; then
        python test_automatic_mask_generator.py \
            --sam_checkpoint "$SAM_CHECKPOINT" \
            --test_images_dir "$TEST_IMAGES_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --annotations_file "$ANNOTATIONS_FILE" \
            --use_affordance_points \
            --use_original_sam
    else
        python test_automatic_mask_generator.py \
            --checkpoint "$TRAINED_CHECKPOINT" \
            --sam_checkpoint "$SAM_CHECKPOINT" \
            --test_images_dir "$TEST_IMAGES_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --annotations_file "$ANNOTATIONS_FILE" \
            --use_affordance_points \
            --use_lora
    fi
elif [ "$TEST_MODE" = "bbox" ]; then
    echo "开始生成masks（使用bbox）..."
    if [ "$USE_ORIGINAL_SAM" = "true" ]; then
        python test_automatic_mask_generator.py \
            --sam_checkpoint "$SAM_CHECKPOINT" \
            --test_images_dir "$TEST_IMAGES_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --annotations_file "$ANNOTATIONS_FILE" \
            --use_bbox \
            --use_original_sam
    else
        python test_automatic_mask_generator.py \
            --checkpoint "$TRAINED_CHECKPOINT" \
            --sam_checkpoint "$SAM_CHECKPOINT" \
            --test_images_dir "$TEST_IMAGES_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --annotations_file "$ANNOTATIONS_FILE" \
            --use_bbox \
            --use_lora
    fi
else
    echo "开始生成masks（使用均匀网格点）..."
    if [ "$USE_ORIGINAL_SAM" = "true" ]; then
        python test_automatic_mask_generator.py \
            --sam_checkpoint "$SAM_CHECKPOINT" \
            --test_images_dir "$TEST_IMAGES_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --points_per_side "$POINTS_PER_SIDE" \
            --points_per_batch "$POINTS_PER_BATCH" \
            --pred_iou_thresh "$PRED_IOU_THRESH" \
            --stability_score_thresh "$STABILITY_SCORE_THRESH" \
            --box_nms_thresh "$BOX_NMS_THRESH" \
            --max_area_ratio "$MAX_AREA_RATIO" \
            --use_original_sam
    else
        python test_automatic_mask_generator.py \
            --checkpoint "$TRAINED_CHECKPOINT" \
            --sam_checkpoint "$SAM_CHECKPOINT" \
            --test_images_dir "$TEST_IMAGES_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --points_per_side "$POINTS_PER_SIDE" \
            --points_per_batch "$POINTS_PER_BATCH" \
            --pred_iou_thresh "$PRED_IOU_THRESH" \
            --stability_score_thresh "$STABILITY_SCORE_THRESH" \
            --box_nms_thresh "$BOX_NMS_THRESH" \
            --max_area_ratio "$MAX_AREA_RATIO" \
            --use_lora
    fi
fi

echo ""
echo "测试完成！"
echo "结果保存在: $OUTPUT_DIR"

