#!/usr/bin/env bash
set -euo pipefail

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

# ========== Output / cache ==========
OUTPUT_ROOT="vlpart_vigor_outputs/llmseg_topk_reranker_k${TOPK_MASK_K}"
CHECKPOINT_PATH="${OUTPUT_ROOT}/checkpoints/reranker.pt"
OUTPUT_DIR="${OUTPUT_ROOT}/feature_correlation"
FEATURE_CACHE_DIR="${OUTPUT_ROOT}/candidate_feature_cache_test"
SAVE_FEATURE_ROWS="false"

# ========== Analysis ==========
SSR_THRESHOLD="0.5"
MAX_SAMPLES=""
PREDICTION_CACHE_SIZE="128"
DEBUG="0"

echo "========================================================================"
echo "  Analyze VLPart top-K reranker input feature correlation"
echo "========================================================================"
echo "Top-K masks: ${LLMSEG_TOPK_MASKS_DIR}"
echo "Top-K K: ${TOPK_MASK_K}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Feature cache: ${FEATURE_CACHE_DIR}"
echo "Save feature rows CSV: ${SAVE_FEATURE_ROWS}"
echo "========================================================================"

ARGS=(
  --mode analyze_features
  --config-file "${CONFIG_FILE}"
  --weights "${WEIGHTS}"
  --data_dir "${DATA_DIR}"
  --easy_json_file "${EASY_JSON}"
  --hard_json_file "${HARD_JSON}"
  --split "${SPLIT}"
  --llmseg_topk_masks_dir "${LLMSEG_TOPK_MASKS_DIR}"
  --topk_mask_k "${TOPK_MASK_K}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --vocabulary "${VOCABULARY}"
  --custom_vocabulary "${CUSTOM_VOCABULARY}"
  --confidence-threshold "${CONFIDENCE_THRESHOLD}"
  --ssr_threshold "${SSR_THRESHOLD}"
  --device "cuda:${GPU_ID}"
  --prediction_cache_size "${PREDICTION_CACHE_SIZE}"
  --candidate_feature_cache_dir "${FEATURE_CACHE_DIR}"
)

if [[ "${SAVE_FEATURE_ROWS}" == "true" || "${SAVE_FEATURE_ROWS}" == "1" ]]; then
  ARGS+=(--save_feature_rows)
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

if [[ "${DEBUG}" == "true" || "${DEBUG}" == "1" ]]; then
  ARGS+=(--debug)
fi

"${PYTHON_BIN}" tools/vigor_vlpart_topk_reranker.py "${ARGS[@]}"

echo ""
echo "========================================================================"
echo "  VLPart top-K feature correlation analysis finished"
echo "========================================================================"
