#!/bin/bash

# GraspNet 服务启动脚本 (graspnet_service.py)
# 运行环境: conda (graspnet)
#apt-get update && apt-get install -y libgl1-mesa-glx


# 1. 配置
PORT="5556"
CHECKPOINT="/opt/data/private/LLMSeg/graspnet-baseline/logs/checkpoint-rs.tar"
export CUDA_VISIBLE_DEVICES=2
DEVICE="cuda:0" # 物理卡 2 现在映射为逻辑 0
COLLISION_THRESH="0.08"
VOXEL_SIZE="0.01"
COLLISION_APPROACH_DIST="0.02"
TABLE_Z="0.40"
TABLE_SAFETY_MARGIN="0.01"
GRIPPER_FINGER_WIDTH="0.0175"
GRIPPER_FINGER_LENGTH="0.05"
GRASP_WIDTH_SCALE="1.0"
GRASP_DEPTH_SCALE="1.0"
GRASP_HEIGHT_SCALE="1.0"
MAX_GRASP_WIDTH="0.14"

# 2. 激活环境 (根据你的服务器配置调整 conda 路径)
CONDA_PATH="/opt/data/private/anaconda3/etc/profile.d/conda.sh" # 常见的 conda 路径
if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
    conda activate graspnet
    echo "-> 已激活 conda 环境: graspnet"
else
    echo "-> [Warning] 没找到 conda.sh, 请手动确保处于 graspnet 环境中"
fi

# 获取当前脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 3. 启动服务
echo "=========================================="
echo "🎯 启动 GraspNet 推理服务"
echo "   Port:      $PORT"
echo "   Device:    $DEVICE"
echo "   Table Z:   $TABLE_Z"
echo "   Coll Th:   $COLLISION_THRESH"
echo "   Coll App:  $COLLISION_APPROACH_DIST"
echo "   Finger W:  $GRIPPER_FINGER_WIDTH"
echo "   Finger L:  $GRIPPER_FINGER_LENGTH"
echo "   Max Width: $MAX_GRASP_WIDTH"
echo "=========================================="

python "$SCRIPT_DIR/graspnet_service.py" \
    --port "$PORT" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    --collision_thresh "$COLLISION_THRESH" \
    --voxel_size "$VOXEL_SIZE" \
    --collision_approach_dist "$COLLISION_APPROACH_DIST" \
    --table_z "$TABLE_Z" \
    --table_safety_margin "$TABLE_SAFETY_MARGIN" \
    --gripper_finger_width "$GRIPPER_FINGER_WIDTH" \
    --gripper_finger_length "$GRIPPER_FINGER_LENGTH" \
    --grasp_width_scale "$GRASP_WIDTH_SCALE" \
    --grasp_depth_scale "$GRASP_DEPTH_SCALE" \
    --grasp_height_scale "$GRASP_HEIGHT_SCALE" \
    --max_grasp_width "$MAX_GRASP_WIDTH" \
    --vis_dir "$SCRIPT_DIR/../vigor_grasp_vis"
