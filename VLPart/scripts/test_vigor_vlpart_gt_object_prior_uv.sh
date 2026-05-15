#!/usr/bin/env bash
set -euo pipefail

# ========================================================================
# VLPart VIGOR-100K GT Object Prior upper-bound evaluation.
# Full RGB image + fixed affordance vocabulary + GT object_prior tensor.
# ========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLPART_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${VLPART_ROOT}/.." && pwd)"
cd "${VLPART_ROOT}"

# ========== Environment ==========
VENV_ROOT="${REPO_ROOT}/.venv"
PYTHON_BIN="${VENV_ROOT}/bin/python"
GPU_ID="0"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

PY_SITE="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
export PATH="/usr/local/cuda/bin:${VENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${PY_SITE}/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/detectron2:${VLPART_ROOT}:${VLPART_ROOT}/demo:${PYTHONPATH:-}"

# ========== VLPart model ==========
CONFIG_FILE="configs/vigor/swinbase_vigor_stage1_gt_object_prior.yaml"
WEIGHTS="/opt/data/private/LLMSeg/VLPart/output/VLPart/vigor_swinbase_stage1_gt_object_prior_bs16_lr4e-5/model_final.pth"
CONFIDENCE_THRESHOLD="0.05"

# ========== Dataset ==========
DATA_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K_new/test"
EASY_JSON="open_vocab_grasp_easy_object_mix.json"
HARD_JSON="open_vocab_grasp_hard_object_mix.json"
SPLIT="both"

# ========== Vocabulary ==========
VOCABULARY="custom"
CUSTOM_VOCABULARY="cylindrical side surface,hexagonal side face,flat side surface,whole object"

# ========== Output ==========
OUTPUT_ROOT="vlpart_vigor_outputs/gt_object_prior_all"
OUTPUT_DIR="${OUTPUT_ROOT}/results"
VIS_DIR="${OUTPUT_ROOT}/visualizations"
PRED_MASKS_DIR="${OUTPUT_ROOT}/pred_masks"
SAVE_VIS="true"
SAVE_PRED_MASKS="true"

# ========== Evaluation ==========
MASK_SELECTION="top1"
SSR_THRESHOLD="0.5"
ICR_THRESHOLD="0.7"
MAX_SAMPLES=""
PREDICTION_CACHE_SIZE="256"
DEBUG="0"

echo "========================================================================"
echo "  VLPart VIGOR-100K GT Object Prior evaluation (.venv)"
echo "========================================================================"
echo "VLPart root: ${VLPART_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Config: ${CONFIG_FILE}"
echo "Weights: ${WEIGHTS}"
echo "Data dir: ${DATA_DIR}"
echo "Split: ${SPLIT}"
echo "ICR threshold: ${ICR_THRESHOLD}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Visualization dir: ${VIS_DIR}"
echo "Pred masks dir: ${PRED_MASKS_DIR}"
echo "========================================================================"

ARGS=(
  --config-file "${CONFIG_FILE}"
  --weights "${WEIGHTS}"
  --data_dir "${DATA_DIR}"
  --easy_json_file "${EASY_JSON}"
  --hard_json_file "${HARD_JSON}"
  --split "${SPLIT}"
  --output_dir "${OUTPUT_DIR}"
  --vis_dir "${VIS_DIR}"
  --pred_masks_dir "${PRED_MASKS_DIR}"
  --vocabulary "${VOCABULARY}"
  --custom_vocabulary "${CUSTOM_VOCABULARY}"
  --confidence-threshold "${CONFIDENCE_THRESHOLD}"
  --mask_selection "${MASK_SELECTION}"
  --ssr_threshold "${SSR_THRESHOLD}"
  --icr_threshold "${ICR_THRESHOLD}"
  --device "cuda:${GPU_ID}"
  --prediction_cache_size "${PREDICTION_CACHE_SIZE}"
)

if [[ "${SAVE_VIS}" == "true" || "${SAVE_VIS}" == "1" ]]; then
  ARGS+=(--save_vis)
fi

if [[ "${SAVE_PRED_MASKS}" == "true" || "${SAVE_PRED_MASKS}" == "1" ]]; then
  ARGS+=(--save_pred_masks)
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

if [[ "${DEBUG}" == "true" || "${DEBUG}" == "1" ]]; then
  ARGS+=(--debug)
fi

"${PYTHON_BIN}" tools/test_vigor_vlpart_gt_object_prior.py "${ARGS[@]}"

echo ""
echo "========================================================================"
echo "  VLPart VIGOR GT Object Prior evaluation finished"
echo "========================================================================"
