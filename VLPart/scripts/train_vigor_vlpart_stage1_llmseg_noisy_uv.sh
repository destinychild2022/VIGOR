#!/usr/bin/env bash
set -euo pipefail

# ========================================================================
# VLPart VIGOR Stage 1 training with the full clean easy set plus extra
# GT-nearby noisy inputs. Noisy duplicates keep the same supervision but
# union the target GT object region with a nearby GT object region as input.
# ========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLPART_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${VLPART_ROOT}/.." && pwd)"
cd "${VLPART_ROOT}"

# ========== Environment ==========
VENV_ROOT="${REPO_ROOT}/.venv"
PYTHON_BIN="${VENV_ROOT}/bin/python"
CUDA_VISIBLE_DEVICES_LIST="0"
NUM_GPUS="1"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

PY_SITE="$("${PYTHON_BIN}" -c "import site; print(site.getsitepackages()[0])")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_LIST}"
export PATH="/usr/local/cuda/bin:${VENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${PY_SITE}/torch/lib"
export PYTHONPATH="${REPO_ROOT}/detectron2:${VLPART_ROOT}:${VLPART_ROOT}/demo:${PYTHONPATH:-}"

# ========== Training config ==========
CONFIG_FILE="configs/vigor/swinbase_vigor_easy_stage1_llmseg_noisy.yaml"
RESUME="false"

# ========== Noisy input config ==========
# Extra noisy duplicates relative to the full clean easy set. 0.2 means
# total training records are clean_N + round(clean_N * 0.2).
NOISY_RATIO="0.2"
NEARBY_TOPK="1"
MAX_CENTER_DISTANCE="150.0"
DATASET_RANDOM_SEED="42"

# ========== VIGOR vocabulary classifier ==========
VOCAB_MAPPING="configs/vigor/vigor_easy_object_to_vocabulary.json"
VOCAB_CLIP_WEIGHT="datasets/metadata/vigor_easy_clip_RN50_a+cname.npy"

# ========== SwanLab logging ==========
SWANLAB_ENABLED="true"
SWANLAB_API_KEY="17UKzqoPx2VI4PLzCHYdH"
SWANLAB_PROJECT="VLPart"
SWANLAB_EXP_NAME="vigor_swinbase_easy_stage1_gt_plus_nearby_noisy_ratio20_top1_dist150"
SWANLAB_LOG_PERIOD="20"

export VLPART_SWANLAB_ENABLED="${SWANLAB_ENABLED}"
export VLPART_SWANLAB_PROJECT="${SWANLAB_PROJECT}"
export VLPART_SWANLAB_EXP_NAME="${SWANLAB_EXP_NAME}"
export VLPART_SWANLAB_LOG_PERIOD="${SWANLAB_LOG_PERIOD}"
if [[ -n "${SWANLAB_API_KEY}" ]]; then
  export SWANLAB_API_KEY="${SWANLAB_API_KEY}"
fi

echo "========================================================================"
echo "  VLPart VIGOR Stage 1 GT + nearby-noisy training (.venv)"
echo "========================================================================"
echo "VLPart root: ${VLPART_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Config: ${CONFIG_FILE}"
echo "Resume: ${RESUME}"
echo "Extra noisy ratio: ${NOISY_RATIO}"
echo "Nearby top-K: ${NEARBY_TOPK}"
echo "Max center distance: ${MAX_CENTER_DISTANCE}"
echo "Dataset random seed: ${DATASET_RANDOM_SEED}"
echo "Vocabulary mapping: ${VOCAB_MAPPING}"
echo "Vocabulary CLIP weight: ${VOCAB_CLIP_WEIGHT}"
echo "SwanLab enabled: ${SWANLAB_ENABLED}"
echo "SwanLab project: ${SWANLAB_PROJECT}"
echo "SwanLab experiment: ${SWANLAB_EXP_NAME}"
echo "SwanLab log period: ${SWANLAB_LOG_PERIOD}"
if [[ -n "${SWANLAB_API_KEY}" ]]; then
  echo "SwanLab API key: configured"
else
  echo "SwanLab API key: not set"
fi
echo "========================================================================"

if [[ ! -f "${VOCAB_CLIP_WEIGHT}" ]]; then
  echo "Generating VIGOR CLIP classifier: ${VOCAB_CLIP_WEIGHT}"
  "${PYTHON_BIN}" tools/vigor_clip_name.py --mapping "${VOCAB_MAPPING}" --output "${VOCAB_CLIP_WEIGHT}"
fi

ARGS=(
  --num-gpus "${NUM_GPUS}"
  --config-file "${CONFIG_FILE}"
  VIGOR_NOISY.NOISY_RATIO "${NOISY_RATIO}"
  VIGOR_NOISY.NEARBY_TOPK "${NEARBY_TOPK}"
  VIGOR_NOISY.MAX_CENTER_DISTANCE "${MAX_CENTER_DISTANCE}"
  VIGOR_NOISY.RANDOM_SEED "${DATASET_RANDOM_SEED}"
)

if [[ "${RESUME}" == "true" || "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
fi

"${PYTHON_BIN}" train_vigor_stage1_llmseg_noisy.py "${ARGS[@]}"

echo ""
echo "========================================================================"
echo "  VLPart VIGOR Stage 1 GT + nearby-noisy training finished"
echo "========================================================================"
