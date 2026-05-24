#!/bin/bash

# VIGOR 主控启动脚本 (vigor_client_main.py)
# 运行环境: .venv (uv)

# 1. 网络配置
SERVER_IP="127.0.0.1"  # 恢复正确的 OmniGibson Server 的 IP
SERVER_PORT="5555"

# --- [网络配置] ---
# GraspNet 在本地
GRASPNET_IP="127.0.0.1"
GRASPNET_PORT="5556"

# SAM 在同机另一个容器，已填入其确切的【内网 IP】
SAM_IP="127.0.0.1"   #"10.106.195.59"  #"10.106.195.30"
SAM_PORT="5557"

# 2. 模型 & 显存配置
export CUDA_VISIBLE_DEVICES=2  # 运行 VIGOR (Llama-2 7B) 的显卡

VIGOR_DEVICE="cuda:0" # 这里映射为物理 2 号卡
VIGOR_VERSION="/opt/data/private/model/LISA_Plus_7b"
VIGOR_CKPT="/opt/data/private/LLMSeg/runs/finetune_llmseg_vigor_simple-new/ckpt_model/epoch_20"
VISION_TOWER="/opt/data/private/model/clip-vit-large-patch14"
VISION_PRETRAINED="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# 3. 运行逻辑
AUTO="true"  #"false"
SCENE_ID="19"
OBJECT_IDX="0"
INSTRUCTION_IDX="0"
MAX_ATTEMPTS="8"
MAX_SCENE_ID="100" #"7"     # 自动模式下为空表示跑完整 JSON；否则只跑 scene <= 该值
RESULTS_PATH="/opt/data/private/LLMSeg/graspnet-baseline/client/results_auto.json"     # 自动模式下为空表示不保存结果文件
DATASET_PATH="/opt/data/private/LLMSeg/dataset/VIGOR-100K_new/test/open_vocab_grasp_easy_new_1.json"
RECORD_VIDEO="true"  # 仅自动模式生效；true 时每条 instruction 保存一段 server 端视频
SERVER_VIDEO_DIR="/home/harrison/workspace/BEHAVIOR-1K/vigor_instruction_videos"
VIDEO_FPS="12"
VIDEO_FRAME_STRIDE="2"

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
echo "   SAM Service:  $SAM_IP:$SAM_PORT"
echo "   Auto Mode:    $AUTO"
echo "   Scene/Object: $SCENE_ID / $OBJECT_IDX"
echo "   Instruction:  $INSTRUCTION_IDX"
echo "   Record Video: $RECORD_VIDEO"
echo "=========================================="

EXTRA_ARGS=()
if [ "$AUTO" = "true" ]; then
    EXTRA_ARGS+=("--auto")
    if [ -n "$MAX_SCENE_ID" ]; then
        EXTRA_ARGS+=("--max_scene_id" "$MAX_SCENE_ID")
    fi
    if [ -n "$RESULTS_PATH" ]; then
        EXTRA_ARGS+=("--results_path" "$RESULTS_PATH")
    fi
    if [ "$RECORD_VIDEO" = "true" ]; then
        EXTRA_ARGS+=("--record_video")
        EXTRA_ARGS+=("--server_video_dir" "$SERVER_VIDEO_DIR")
        EXTRA_ARGS+=("--video_fps" "$VIDEO_FPS")
        EXTRA_ARGS+=("--video_frame_stride" "$VIDEO_FRAME_STRIDE")
    fi
else
    EXTRA_ARGS+=("--scene_id" "$SCENE_ID")
    EXTRA_ARGS+=("--object_idx" "$OBJECT_IDX")
    EXTRA_ARGS+=("--instruction_idx" "$INSTRUCTION_IDX")
fi

python "$SCRIPT_DIR/vigor_client_main.py" \
    --server_ip "$SERVER_IP" \
    --server_port "$SERVER_PORT" \
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
    --max_attempts "$MAX_ATTEMPTS" \
    --vis_dir "$SCRIPT_DIR/../vigor_grasp_vis" \
    "${EXTRA_ARGS[@]}"

echo "=========================================="
echo "✅ 程序已退出"
