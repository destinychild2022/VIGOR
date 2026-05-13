#!/usr/bin/env bash
set -euo pipefail

# ========================================================================
# VLPart VIGOR-100K batch evaluation
#
# The Python evaluator reads:
#   ${DATA_DIR}/${EASY_JSON}
#   ${DATA_DIR}/${HARD_JSON}
# and uses gt_object_path as input image, gt_mask_path as GT mask.
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
GPU_ID="1"

export LD_LIBRARY_PATH="${CONDA_ROOT}/envs/${CONDA_ENV}/lib:${CONDA_ROOT}/envs/${CONDA_ENV}/targets/x86_64-linux/lib:${CONDA_ROOT}/envs/${CONDA_ENV}/lib/python3.10/site-packages/torch/lib"

# ========== VLPart model ==========
CONFIG_FILE="configs/vigor/swinbase_vigor_easy_stage1.yaml"
WEIGHTS="/opt/data/private/LLMSeg/VLPart/output/VLPart/vigor_swinbase_easy_stage1_bs16_lr4e-5/model_final.pth"
CONFIDENCE_THRESHOLD="0.05"

# ========== Dataset ==========
DATA_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K_new/test"
EASY_JSON="open_vocab_grasp_easy_object_mix.json"
HARD_JSON="open_vocab_grasp_hard_object_mix.json"
SPLIT="both"  # both, easy, hard

# ========== Vocabulary ==========
VOCABULARY="custom"
CUSTOM_VOCABULARY="cylindrical side surface,hexagonal side face,flat side surface,whole object"

# ========== Output ==========
OUTPUT_ROOT="vlpart_vigor_outputs/stage1_bs16_lr4e-5_final"
OUTPUT_DIR="${OUTPUT_ROOT}/results"
VIS_DIR="${OUTPUT_ROOT}/visualizations"
PRED_MASKS_DIR="${OUTPUT_ROOT}/pred_masks"
SAVE_VIS="true"
SAVE_PRED_MASKS="true"

# ========== Evaluation ==========
MASK_SELECTION="top1"  # top1 or union
SSR_THRESHOLD="0.5"
MAX_SAMPLES=""  # Empty means full split. Example: "10" for quick debugging.
PREDICTION_CACHE_SIZE="256"
DEBUG="0"

echo "========================================================================"
echo "  VLPart VIGOR-100K batch evaluation"
echo "========================================================================"
echo "VLPart root: ${VLPART_ROOT}"
echo "Conda env: ${CONDA_ENV}"
echo "Config: ${CONFIG_FILE}"
echo "Weights: ${WEIGHTS}"
echo "Data dir: ${DATA_DIR}"
echo "Easy JSON: ${EASY_JSON}"
echo "Hard JSON: ${HARD_JSON}"
echo "Split: ${SPLIT}"
echo "Vocabulary: ${VOCABULARY}"
echo "Custom vocabulary: ${CUSTOM_VOCABULARY}"
echo "Mask selection: ${MASK_SELECTION}"
echo "Confidence threshold: ${CONFIDENCE_THRESHOLD}"
echo "SSR threshold: ${SSR_THRESHOLD}"
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

conda run -n "${CONDA_ENV}" python tools/test_vigor_vlpart.py "${ARGS[@]}"

echo ""
echo "========================================================================"
echo "  VLPart VIGOR evaluation finished"
echo "========================================================================"
