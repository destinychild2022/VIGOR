#!/bin/bash

# GraspNet 服务启动脚本 (graspnet_service.py)
# 运行环境: conda (graspnet)
#apt-get update && apt-get install -y libgl1-mesa-glx


# 1. 配置
PORT="5556"
CHECKPOINT="/opt/data/private/LLMSeg/graspnet-baseline/logs/checkpoint-rs.tar"
export CUDA_VISIBLE_DEVICES=2
DEVICE="cuda:0" # 物理卡 2 现在映射为逻辑 0

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
echo "=========================================="

python "$SCRIPT_DIR/graspnet_service.py" \
    --port "$PORT" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    --vis_dir "$SCRIPT_DIR/../vigor_grasp_vis"
