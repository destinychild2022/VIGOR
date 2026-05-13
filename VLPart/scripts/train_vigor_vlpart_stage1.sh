#!/usr/bin/env bash
set -euo pipefail

# ========================================================================
# VLPart VIGOR Stage 1 fine-tuning
#
# Stage 1 freezes the Swin-B bottom_up backbone and trains the remaining
# FPN/RPN/ROI/mask heads from the existing VLPart SwinBase + IN parsed weight.
# The dataset loader reads the VIGOR easy JSON directly.
# ========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLPART_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLPART_ROOT}"

# ========================================================================
# Editable settings
# Change the values in this block directly before running the script.
# ========================================================================

# ========== Environment ==========
CONDA_ENV="vlpart"
CONDA_ROOT="/opt/data/private/miniconda3"
CUDA_VISIBLE_DEVICES_LIST="1,2"
NUM_GPUS="2"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_LIST}"
export LD_LIBRARY_PATH="${CONDA_ROOT}/envs/${CONDA_ENV}/lib:${CONDA_ROOT}/envs/${CONDA_ENV}/targets/x86_64-linux/lib:${CONDA_ROOT}/envs/${CONDA_ENV}/lib/python3.10/site-packages/torch/lib"

# ========== Training config ==========
CONFIG_FILE="configs/vigor/swinbase_vigor_easy_stage1.yaml"
RESUME="false"

# The config intentionally keeps the original SwinBase + IN parsed total
# iteration count: SOLVER.MAX_ITER = 90000.
# With 2 x 4090, per-GPU batch is 8 and global batch is 16 in the config.

# ========== VIGOR vocabulary classifier ==========
VOCAB_MAPPING="configs/vigor/vigor_easy_object_to_vocabulary.json"
VOCAB_CLIP_WEIGHT="datasets/metadata/vigor_easy_clip_RN50_a+cname.npy"

# ========== SwanLab logging ==========
SWANLAB_ENABLED="true"
SWANLAB_API_KEY="17UKzqoPx2VI4PLzCHYdH"  # Fill this in if this machine is not already logged in.
SWANLAB_PROJECT="VLPart"
SWANLAB_EXP_NAME="vigor_swinbase_easy_stage1_bs16_lr4e-5"
SWANLAB_LOG_PERIOD="20"

export VLPART_SWANLAB_ENABLED="${SWANLAB_ENABLED}"
export VLPART_SWANLAB_PROJECT="${SWANLAB_PROJECT}"
export VLPART_SWANLAB_EXP_NAME="${SWANLAB_EXP_NAME}"
export VLPART_SWANLAB_LOG_PERIOD="${SWANLAB_LOG_PERIOD}"
if [[ -n "${SWANLAB_API_KEY}" ]]; then
  export SWANLAB_API_KEY="${SWANLAB_API_KEY}"
fi

echo "========================================================================"
echo "  VLPart VIGOR Stage 1 fine-tuning"
echo "========================================================================"
echo "VLPart root: ${VLPART_ROOT}"
echo "Conda env: ${CONDA_ENV}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Config: ${CONFIG_FILE}"
echo "Resume: ${RESUME}"
echo "Vocabulary mapping: ${VOCAB_MAPPING}"
echo "Vocabulary CLIP weight: ${VOCAB_CLIP_WEIGHT}"
echo "SwanLab enabled: ${SWANLAB_ENABLED}"
echo "SwanLab project: ${SWANLAB_PROJECT}"
echo "SwanLab experiment: ${SWANLAB_EXP_NAME}"
echo "SwanLab log period: ${SWANLAB_LOG_PERIOD}"
if [[ -n "${SWANLAB_API_KEY}" ]]; then
  echo "SwanLab API key: configured"
else
  echo "SwanLab API key: not set in script"
fi
echo "========================================================================"

if [[ ! -f "${VOCAB_CLIP_WEIGHT}" ]]; then
  echo "Generating VIGOR CLIP classifier: ${VOCAB_CLIP_WEIGHT}"
  conda run -n "${CONDA_ENV}" python tools/vigor_clip_name.py \
    --mapping "${VOCAB_MAPPING}" \
    --output "${VOCAB_CLIP_WEIGHT}"
fi

ARGS=(
  --num-gpus "${NUM_GPUS}"
  --config-file "${CONFIG_FILE}"
)

if [[ "${RESUME}" == "true" || "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
fi

conda run -n "${CONDA_ENV}" python train_vigor_stage1.py "${ARGS[@]}"

echo ""
echo "========================================================================"
echo "  VLPart VIGOR Stage 1 training finished"
echo "========================================================================"
