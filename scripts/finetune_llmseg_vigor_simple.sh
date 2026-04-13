#! /bin/bash

# ========================================================================
# 简化版 VIGOR 训练脚本 (使用 finetune_llmseg_vigor_simple.py)
# - 可直接关闭 SwanLab、训练可视化、验证，方便看每 step 性能/显存
# ========================================================================

# ========== 模型路径配置 ==========
MODEL_PATH="../root/autodl-tmp/model/LISA_Plus_7b"
CLIP_PATH="../root/autodl-tmp/model/clip-vit-large-patch14"
VISION_PATH="../root/autodl-tmp/model/SAM-vit-h/sam_vit_h_4b8939.pth"
DINO_PATH="../root/autodl-tmp/model/dinov2_vitl14"

# torch.hub/DINOv2 代码缓存放数据盘，避免占用系统盘并避免每次重下。
TORCH_HOME="/root/autodl-tmp/torch_cache"
export TORCH_HOME
export DINOV2_LOCAL_PATH="${DINO_PATH}"

# ========== VIGOR 数据集配置 ==========
VIGOR_DATA_DIR="../root/autodl-tmp/VIGOR-100K_new"
VIGOR_TRAIN_SPLIT="train"
VIGOR_VAL_SPLIT="test"
# SAM候选masks路径
VIGOR_TRAIN_SAM_MASKS="../root/autodl-tmp/train_masks_sam_0.8_0.8"
VIGOR_VAL_SAM_MASKS="../root/autodl-tmp/test_mask/sam_masks"
# 是否只使用 hard 样本 (不使用 easy 样本)
VIGOR_ONLY_HARD=false
# 训练启动前自动扫描 train/depth，用训练集分位数确定 depth range。
AUTO_VIGOR_DEPTH_RANGE=true
VIGOR_DEPTH_RANGE_LOW_P=0.1
VIGOR_DEPTH_RANGE_HIGH_P=99.9
VIGOR_DEPTH_SCAN_SAMPLES_PER_FILE=4096
# 自动扫描失败时的保守 fallback；dataset 会将 depth clamp/map 到 [0,1]。
VIGOR_DEPTH_MIN=0.6
VIGOR_DEPTH_MAX=1.85

# ========== 输出配置 ==========
LOG_DIR="../root/autodl-tmp/runs"
EXP_NAME="finetune_llmseg_vigor_simple-DFormer"
TRAIN_VIS_DIR="train_vis"
VAL_VIS_DIR="val_vis"
EVAL_VIS_DIR="eval_vis_iop"

# ========== 训练超参数 ==========
EPOCHS=20
STEPS_PER_EPOCH=1000
BATCH_SIZE=16
GRAD_ACCUMULATION_STEPS=1
LR=2e-5
PRECISION="bf16"
ALIGN_TEMP=0.05
VIGOR_MAX_INSTRUCTIONS=3

# ========== 性能测试开关 ==========
# 手动测试不同 batch size 时，只改上面的 BATCH_SIZE 和 EXP_NAME 即可。
# DISABLE_SWANLAB=true  -> 传 --disable_swanlab
# VISUALIZE=false      -> 不传 --visualize，关闭训练/验证可视化输出
# NO_EVAL=true         -> 传 --no_eval，跳过验证
PROFILE_PERF=false
DISABLE_SWANLAB=false
VISUALIZE=true
NO_EVAL=false

# 验证集配置 (test 分片中 scene ID <= 1000 的所有 Easy+Hard 样本)
VAL_MAX_SCENE_ID=1000
# 每个 epoch 可视化样本数
MAX_VIS_SAMPLES=4

# ========== LoRA 配置 ==========
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,out_proj"

# ========== GPU 配置 ==========
GPU_IDS="0"
MASTER_PORT=24375

# ========== Checkpoint 配置 ==========
RESUME_PATH=""

# ========================================================================

cd "$(dirname "$0")/.." || exit 1

# Prefer the project's venv. This machine currently uses .venv3124.
DEEPSPEED_BIN="./.venv/bin/deepspeed"
if [ ! -x "$DEEPSPEED_BIN" ] && [ -x "./.venv3124/bin/deepspeed" ]; then
  DEEPSPEED_BIN="./.venv3124/bin/deepspeed"
fi
if [ ! -x "$DEEPSPEED_BIN" ]; then
  DEEPSPEED_BIN="deepspeed"
fi
PYTHON_BIN="./.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ] && [ -x "./.venv3124/bin/python" ]; then
  PYTHON_BIN="./.venv3124/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

if [ "${AUTO_VIGOR_DEPTH_RANGE}" = true ]; then
  DEPTH_RANGE_OUTPUT=$("$PYTHON_BIN" - <<PY
import os
from pathlib import Path
import numpy as np

depth_dir = Path("${VIGOR_DATA_DIR}") / "${VIGOR_TRAIN_SPLIT}" / "depth"
low_p = float("${VIGOR_DEPTH_RANGE_LOW_P}")
high_p = float("${VIGOR_DEPTH_RANGE_HIGH_P}")
samples_per_file = int("${VIGOR_DEPTH_SCAN_SAMPLES_PER_FILE}")
files = sorted(depth_dir.glob("*.npy"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
if not files:
    raise SystemExit(f"no depth npy files found in {depth_dir}")
rng = np.random.default_rng(20260413)
chunks = []
exact_min = np.inf
exact_max = -np.inf
for p in files:
    arr = np.load(p, mmap_mode="r")
    vals = np.asarray(arr).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        continue
    exact_min = min(exact_min, float(vals.min()))
    exact_max = max(exact_max, float(vals.max()))
    take = min(samples_per_file, vals.size)
    if take > 0:
        idx = rng.choice(vals.size, size=take, replace=False)
        chunks.append(np.asarray(vals[idx], dtype=np.float32))
if not chunks:
    raise SystemExit(f"no finite depth values found in {depth_dir}")
sample = np.concatenate(chunks)
lo = float(np.percentile(sample, low_p))
hi = float(np.percentile(sample, high_p))
lo = min(lo, exact_min)
hi = max(hi, exact_max)
margin = max((hi - lo) * 0.01, 1e-4)
lo -= margin
hi += margin
print(f"{lo:.8f} {hi:.8f} {len(files)} {exact_min:.8f} {exact_max:.8f}")
PY
)
  if [ $? -eq 0 ]; then
    read -r VIGOR_DEPTH_MIN VIGOR_DEPTH_MAX VIGOR_DEPTH_FILES VIGOR_DEPTH_EXACT_MIN VIGOR_DEPTH_EXACT_MAX <<< "$DEPTH_RANGE_OUTPUT"
    echo "[DepthRange] auto range from train/depth: files=${VIGOR_DEPTH_FILES} exact=[${VIGOR_DEPTH_EXACT_MIN}, ${VIGOR_DEPTH_EXACT_MAX}] range=[${VIGOR_DEPTH_MIN}, ${VIGOR_DEPTH_MAX}]"
  else
    echo "[DepthRange] auto scan failed; using fallback range=[${VIGOR_DEPTH_MIN}, ${VIGOR_DEPTH_MAX}]"
  fi
fi
export VIGOR_DEPTH_MIN
export VIGOR_DEPTH_MAX

echo "========================================================================"
echo "  VIGOR 简化版微调训练"
echo "========================================================================"
echo "模型路径: ${MODEL_PATH}"
echo "DINOv2路径: ${DINO_PATH}"
echo "torch hub缓存: ${TORCH_HOME}"
echo "数据目录: ${VIGOR_DATA_DIR}"
echo "Depth归一化: [${VIGOR_DEPTH_MIN}, ${VIGOR_DEPTH_MAX}]"
echo "GPU 显卡: ${GPU_IDS}"
echo "实验名称: ${EXP_NAME}"
echo "batch_size: ${BATCH_SIZE}"
echo "性能输出: PROFILE_PERF=${PROFILE_PERF}"
echo "关闭 SwanLab: DISABLE_SWANLAB=${DISABLE_SWANLAB}"
echo "关闭可视化: VISUALIZE=${VISUALIZE}"
echo "跳过验证: NO_EVAL=${NO_EVAL}"
echo "========================================================================"

# 额外的可选参数
EXTRA_ARGS=""
if [ "${VIGOR_ONLY_HARD}" = true ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --vigor_only_hard"
fi
if [ "${PROFILE_PERF}" = true ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --profile_perf"
fi
if [ "${DISABLE_SWANLAB}" = true ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --disable_swanlab"
fi
if [ "${VISUALIZE}" = true ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --visualize"
fi
if [ "${NO_EVAL}" = true ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --no_eval"
fi

# 执行训练
$DEEPSPEED_BIN --include localhost:${GPU_IDS} \
  --master_port=${MASTER_PORT} finetune_llmseg_vigor_simple.py \
  --version="${MODEL_PATH}" \
  --vision-tower="${CLIP_PATH}" \
  --vision_pretrained="${VISION_PATH}" \
  --dataset="vigor" \
  --sample_rates="1" \
  --vigor_data_base_dir="${VIGOR_DATA_DIR}" \
  --vigor_split="${VIGOR_TRAIN_SPLIT}" \
  --vigor_val_split="${VIGOR_VAL_SPLIT}" \
  --vigor_train_sam_masks_dir="${VIGOR_TRAIN_SAM_MASKS}" \
  --vigor_val_sam_masks_dir="${VIGOR_VAL_SAM_MASKS}" \
  --vigor_val_max_samples=${VAL_MAX_SCENE_ID} \
  --max_vis_samples=${MAX_VIS_SAMPLES} \
  --exp_name="${EXP_NAME}" \
  --log_base_dir="${LOG_DIR}" \
  --steps_per_epoch=${STEPS_PER_EPOCH} \
  --lr=${LR} \
  --epochs=${EPOCHS} \
  --batch_size=${BATCH_SIZE} \
  --grad_accumulation_steps=${GRAD_ACCUMULATION_STEPS} \
  --workers=8 \
  --lora_r=${LORA_R} \
  --lora_alpha=${LORA_ALPHA} \
  --lora_dropout=${LORA_DROPOUT} \
  --lora_target_modules="${LORA_TARGET_MODULES}" \
  --precision="${PRECISION}" \
  --resume="${RESUME_PATH}" \
  --train_vis_dir="${TRAIN_VIS_DIR}" \
  --val_vis_dir="${VAL_VIS_DIR}" \
  --eval_vis_dir="${EVAL_VIS_DIR}" \
  --align_temperature=${ALIGN_TEMP} \
  --vigor_max_instructions=${VIGOR_MAX_INSTRUCTIONS} \
  --iou_selection_only \
  --print_freq=1 \
  ${EXTRA_ARGS}

echo "========================================================================"
echo "  训练结束"
echo "========================================================================"
