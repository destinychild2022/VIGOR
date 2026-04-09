#!/bin/bash

# VIGOR 主控启动脚本 (vigor_client_main.py)
# 运行环境: .venv (uv)

# 1. 网络配置
SERVER_IP="219.223.182.106"  # 恢复正确的 OmniGibson Server 的 IP
SERVER_PORT="5555"

# --- [网络配置] ---
# GraspNet 在本地
GRASPNET_IP="127.0.0.1"
GRASPNET_PORT="5556"

# SAM 在同机另一个容器，已填入其确切的【内网 IP】
SAM_IP="127.0.0.1"  #"10.106.195.30" 
SAM_PORT="5557"

# 2. 模型 & 显存配置
export CUDA_VISIBLE_DEVICES=2  # 运行 VIGOR (Llama-2 7B) 的显卡

VIGOR_DEVICE="cuda:0" # 这里映射为物理 1 号卡
SAM_CKPT="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"
VIGOR_VERSION="/opt/data/private/model/LISA_Plus_7b"
VIGOR_CKPT="/opt/data/private/LLMSeg/runs/finetune_llmseg_vigor_simple/ckpt_model"
VISION_TOWER="/opt/data/private/model/clip-vit-large-patch14"
VISION_PRETRAINED="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# 3. 运行逻辑 (支持场景索引)
SCENE_IDX=${1:-0}
DATASET_PATH="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test/open_vocab_grasp_easy.json"

# 4. 激活环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="/opt/data/private/LLMSeg/.venv/bin/activate"

if [ -f "$VENV_PATH" ]; then
    echo "-> 激活虚拟环境: $VENV_PATH"
    source "$VENV_PATH"
fi

# 5. 启动主程序
echo "=========================================="
echo "🚀 启动 VIGOR 主控流水线 (Dual-GPU Mode)"
echo "   VIGOR Device: $VIGOR_DEVICE"
echo "   SAM Device:   $SAM_DEVICE"
echo "   Scene Index:  $SCENE_IDX"
echo "=========================================="

python "$SCRIPT_DIR/vigor_client_main.py" \
    --server_ip "$SERVER_IP" \
    --server_port "$SERVER_PORT" \
    --graspnet_port "$GRASPNET_PORT" \
    --sam_checkpoint "$SAM_CKPT" \
    --vigor_version "$VIGOR_VERSION" \
    --vigor_checkpoint "$VIGOR_CKPT" \
    --vision_tower "$VISION_TOWER" \
    --vision_pretrained "$VISION_PRETRAINED" \
    --vigor_device "$VIGOR_DEVICE" \
    --sam_port "$SAM_PORT" \
    --sam_ip "$SAM_IP" \
    --graspnet_ip "$GRASPNET_IP" \
    --graspnet_port "$GRASPNET_PORT" \
    --precision "bf16" \
    --dataset_path "$DATASET_PATH" \
    --scene_idx "$SCENE_IDX" \
    --vis_dir "$SCRIPT_DIR/../vigor_grasp_vis"

echo "=========================================="
echo "✅ 程序已退出"
