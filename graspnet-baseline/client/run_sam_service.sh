#!/bin/bash

# SAM 独立分割服务启动脚本
# 运行环境: .venv (uv)

# 1. 配置
PORT="5557"
CHECKPOINT="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"
export CUDA_VISIBLE_DEVICES=0
DEVICE="cuda:0" # 物理卡 0 现在映射为逻辑 0

# 2. 激活环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="/opt/data/private/LLMSeg/.venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
fi

# 3. 启动服务
echo "=========================================="
echo "🎯 启动 SAM (ViT-H) 分割服务"
echo "   Port:      $PORT"
echo "   Device:    Physical GPU 0 ($DEVICE)"
echo "=========================================="

python "$SCRIPT_DIR/sam_service.py" \
    --port "$PORT" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE"
