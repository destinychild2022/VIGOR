#!/usr/bin/env bash
set -euo pipefail

# ========================================================================
# VLPart VIGOR Stage 1 GT Object Prior upper-bound fine-tuning.
# This adds object_prior as a tensor input and leaves the original scripts
# untouched.
# ========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLPART_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${VLPART_ROOT}/.." && pwd)"
cd "${VLPART_ROOT}"

# ========== Environment ==========
VENV_ROOT="${REPO_ROOT}/.venv"
PYTHON_BIN="${VENV_ROOT}/bin/python"
CUDA_VISIBLE_DEVICES_LIST="0,1"
NUM_GPUS="2"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

PY_SITE="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_LIST}"
export PATH="/usr/local/cuda/bin:${VENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${PY_SITE}/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/detectron2:${VLPART_ROOT}:${VLPART_ROOT}/demo:${PYTHONPATH:-}"

# ========== Training config ==========
CONFIG_FILE="configs/vigor/swinbase_vigor_stage1_gt_object_prior.yaml"
RESUME="false"

# ========== VIGOR vocabulary classifier ==========
VOCAB_MAPPING="configs/vigor/vigor_easy_object_to_vocabulary.json"
VOCAB_CLIP_WEIGHT="datasets/metadata/vigor_easy_clip_RN50_a+cname.npy"

# ========== SwanLab logging ==========
SWANLAB_ENABLED="true"
SWANLAB_API_KEY="17UKzqoPx2VI4PLzCHYdH"
SWANLAB_PROJECT="VLPart"
SWANLAB_EXP_NAME="vigor_swinbase_gt_object_prior_all"
SWANLAB_LOG_PERIOD="20"

export VLPART_SWANLAB_ENABLED="${SWANLAB_ENABLED}"
export VLPART_SWANLAB_PROJECT="${SWANLAB_PROJECT}"
export VLPART_SWANLAB_EXP_NAME="${SWANLAB_EXP_NAME}"
export VLPART_SWANLAB_LOG_PERIOD="${SWANLAB_LOG_PERIOD}"
if [[ -n "${SWANLAB_API_KEY}" ]]; then
  export SWANLAB_API_KEY="${SWANLAB_API_KEY}"
fi

echo "========================================================================"
echo "  VLPart VIGOR GT Object Prior fine-tuning (.venv)"
echo "========================================================================"
echo "VLPart root: ${VLPART_ROOT}"
echo "Python: ${PYTHON_BIN}"
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
echo "========================================================================"

if [[ ! -f "${VOCAB_CLIP_WEIGHT}" ]]; then
  echo "Generating VIGOR CLIP classifier: ${VOCAB_CLIP_WEIGHT}"
  "${PYTHON_BIN}" tools/vigor_clip_name.py \
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

"${PYTHON_BIN}" train_vigor_stage1_gt_object_prior.py "${ARGS[@]}"

echo ""
echo "========================================================================"
echo "  VLPart VIGOR GT Object Prior training finished"
echo "========================================================================"
