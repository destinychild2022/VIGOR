#!/usr/bin/env bash
set -euo pipefail

# ========================================================================
# VLPart VIGOR-100K evaluation with LLMSeg top-K object masks.
# Selection uses an equal-weight bounded score:
#   run VLPart on every top-K masked RGB image, flatten all predicted masks,
#   map llmseg_similarity from [-1, 1] to [0, 1], use the other
#   three scores directly, then choose max weighted score.
# GT affordance masks are used only for final evaluation.
# ========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLPART_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${VLPART_ROOT}/.." && pwd)"
cd "${VLPART_ROOT}"

# ========== Environment ==========
VENV_ROOT="${REPO_ROOT}/.venv"
PYTHON_BIN="${VENV_ROOT}/bin/python"
GPU_ID="1"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

PY_SITE="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
export PATH="/usr/local/cuda/bin:${VENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${PY_SITE}/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/detectron2:${VLPART_ROOT}:${VLPART_ROOT}/demo:${PYTHONPATH:-}"

# ========== VLPart model ==========
CONFIG_FILE="configs/vigor/swinbase_vigor_easy_stage1.yaml"
WEIGHTS="/opt/data/private/LLMSeg/VLPart/output/VLPart/vigor_swinbase_easy_stage1_bs16_lr4e-5/model_final.pth"
CONFIDENCE_THRESHOLD="0.05"

# ========== Dataset ==========
DATA_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K_new/test"
EASY_JSON="open_vocab_grasp_easy_object_mix.json"
HARD_JSON="open_vocab_grasp_hard_object_mix.json"
SPLIT="both"

# ========== LLMSeg Top-K object masks ==========
LLMSEG_TOPK_MASKS_DIR="/opt/data/private/LLMSeg/vis_output_object_topk_testset/topk_masks"
TOPK_MASK_K="3"

# ========== Vocabulary ==========
VOCABULARY="custom"
CUSTOM_VOCABULARY="cylindrical side surface,hexagonal side face,flat side surface,whole object"

# ========== Output ==========
OUTPUT_ROOT="vlpart_vigor_outputs/llmseg_topk_weighted_score_k${TOPK_MASK_K}"
OUTPUT_DIR="${OUTPUT_ROOT}/results"
VIS_DIR="${OUTPUT_ROOT}/visualizations"
PRED_MASKS_DIR="${OUTPUT_ROOT}/pred_masks"
SAVE_VIS="false"
SAVE_PRED_MASKS="false"

# ========== Evaluation ==========
SSR_THRESHOLD="0.5"
ICR_THRESHOLD="0.7"
MAX_SAMPLES=""
PREDICTION_CACHE_SIZE="256"
DEBUG="0"

echo "========================================================================"
echo "  VLPart VIGOR-100K LLMSeg Top-K weighted-score evaluation"
echo "========================================================================"
echo "VLPart root: ${VLPART_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Config: ${CONFIG_FILE}"
echo "Weights: ${WEIGHTS}"
echo "Data dir: ${DATA_DIR}"
echo "Split: ${SPLIT}"
echo "LLMSeg top-K masks: ${LLMSEG_TOPK_MASKS_DIR}"
echo "Top-K K: ${TOPK_MASK_K}"
echo "Selection: 0.25 * scaled llmseg_similarity + 0.25 * llmseg_predicted_iou + 0.25 * vlpart_score + 0.25 * containment_score"
echo "Output dir: ${OUTPUT_DIR}"
echo "========================================================================"

ARGS=(
  --config-file "${CONFIG_FILE}"
  --weights "${WEIGHTS}"
  --data_dir "${DATA_DIR}"
  --easy_json_file "${EASY_JSON}"
  --hard_json_file "${HARD_JSON}"
  --split "${SPLIT}"
  --llmseg_topk_masks_dir "${LLMSEG_TOPK_MASKS_DIR}"
  --topk_mask_k "${TOPK_MASK_K}"
  --output_dir "${OUTPUT_DIR}"
  --vis_dir "${VIS_DIR}"
  --pred_masks_dir "${PRED_MASKS_DIR}"
  --vocabulary "${VOCABULARY}"
  --custom_vocabulary "${CUSTOM_VOCABULARY}"
  --confidence-threshold "${CONFIDENCE_THRESHOLD}"
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

"${PYTHON_BIN}" tools/test_vigor_vlpart_llmseg_topk_weighted_score.py "${ARGS[@]}"

echo ""
echo "========================================================================"
echo "  VLPart top-K weighted-score evaluation finished"
echo "========================================================================"
