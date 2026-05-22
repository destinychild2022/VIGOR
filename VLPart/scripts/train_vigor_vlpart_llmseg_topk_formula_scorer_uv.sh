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
EXTRACT_GPUS=("0" "2")
PARALLEL_EXTRACT="true"

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
DATA_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K_new/train"
EASY_JSON="open_vocab_grasp_easy_object_mix.json"
HARD_JSON="open_vocab_grasp_hard_object_mix.json"
SPLIT="both"

# ========== LLMSeg Top-K object masks ==========
LLMSEG_TOPK_MASKS_DIR="/opt/data/private/LLMSeg/vis_output_object_topk_trainset/topk_masks"
TOPK_MASK_K="3"

# ========== Vocabulary ==========
VOCABULARY="custom"
CUSTOM_VOCABULARY="cylindrical side surface,hexagonal side face,flat side surface,whole object"

# ========== Output / cache ==========
OUTPUT_ROOT="vlpart_vigor_outputs/llmseg_topk_reranker_k${TOPK_MASK_K}"
OUTPUT_DIR="${OUTPUT_ROOT}/train_logs"
CHECKPOINT_PATH="${OUTPUT_ROOT}/checkpoints/formula_scorer.pt"
# Reuse the full 4-D candidate feature cache produced by the reranker pipeline.
FEATURE_CACHE_DIR="vlpart_vigor_outputs/llmseg_topk_reranker_k${TOPK_MASK_K}/candidate_feature_cache_train"
EXTRACT_LOG_DIR="${OUTPUT_ROOT}/extract_logs"

# ========== Training ==========
# Formula: score = (alpha * pred_similarity + (1 - alpha) * pred_iou) * vlpart_score
# pred_similarity uses the raw value from LLMSeg top-K mask filenames.
# Loss: one-hot cross entropy over all candidates in each instruction group.
# The candidate with the highest GT IoU is the positive class; all others are 0.
# alpha is stored as one nn.Parameter through a sigmoid, so the effective alpha stays in [0, 1].
EPOCHS="20"
LR="5e-2"
WEIGHT_DECAY="0.0"
RANKING_TEMPERATURE="1.0"
FORMULA_ALPHA_INIT="0.5"
SEED="42"
MAX_SAMPLES="1000"
PREDICTION_CACHE_SIZE="128"
FEATURE_CACHE_CHUNK_SIZE="1000"
BALANCE_TRAIN_EASY_HARD="false"
DEBUG="0"

# ========== SwanLab ==========
SWANLAB_ENABLED="true"
SWANLAB_API_KEY="17UKzqoPx2VI4PLzCHYdH"
SWANLAB_PROJECT="VLPart"
SWANLAB_EXP_NAME="vigor_topk_formula_scorer_k${TOPK_MASK_K}"
if [[ -n "${SWANLAB_API_KEY}" ]]; then
  export SWANLAB_API_KEY="${SWANLAB_API_KEY}"
fi

echo "========================================================================"
echo "  Train VLPart top-K formula scorer"
echo "========================================================================"
echo "Formula: (alpha * pred_similarity + (1 - alpha) * pred_iou) * vlpart_score"
echo "Loss: one-hot CE, positive candidate = max GT IoU in each group"
echo "Initial alpha: ${FORMULA_ALPHA_INIT}"
echo "Top-K masks: ${LLMSEG_TOPK_MASKS_DIR}"
echo "Top-K K: ${TOPK_MASK_K}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Feature cache: ${FEATURE_CACHE_DIR}"
echo "Feature cache chunk size: ${FEATURE_CACHE_CHUNK_SIZE}"
echo "Balance train easy/hard: ${BALANCE_TRAIN_EASY_HARD}"
echo "Parallel extract: ${PARALLEL_EXTRACT}"
echo "Extract GPUs: ${EXTRACT_GPUS[*]}"
echo "Train GPU: ${GPU_ID}"
echo "Epochs: ${EPOCHS}"
echo "LR: ${LR}"
echo "SwanLab enabled: ${SWANLAB_ENABLED}"
echo "SwanLab project: ${SWANLAB_PROJECT}"
echo "SwanLab experiment: ${SWANLAB_EXP_NAME}"
echo "========================================================================"

COMMON_ARGS=(
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
  --prediction_cache_size "${PREDICTION_CACHE_SIZE}"
  --candidate_feature_cache_dir "${FEATURE_CACHE_DIR}"
  --feature_cache_chunk_size "${FEATURE_CACHE_CHUNK_SIZE}"
  --seed "${SEED}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  COMMON_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

if [[ "${DEBUG}" == "true" || "${DEBUG}" == "1" ]]; then
  COMMON_ARGS+=(--debug)
fi

if [[ "${BALANCE_TRAIN_EASY_HARD}" == "true" || "${BALANCE_TRAIN_EASY_HARD}" == "1" ]]; then
  COMMON_ARGS+=(--balance_train_easy_hard)
fi

mkdir -p "${OUTPUT_ROOT}" "${EXTRACT_LOG_DIR}"
NUM_EXTRACT_SHARDS="${#EXTRACT_GPUS[@]}"

if [[ "${PARALLEL_EXTRACT}" == "true" || "${PARALLEL_EXTRACT}" == "1" ]]; then
  echo ""
  echo "========================================================================"
  echo "  Step 1/2: build train feature cache with ${#EXTRACT_GPUS[@]} GPU shards"
  echo "========================================================================"
  PIDS=()
  for SHARD_INDEX in "${!EXTRACT_GPUS[@]}"; do
    EXTRACT_GPU="${EXTRACT_GPUS[${SHARD_INDEX}]}"
    LOG_FILE="${EXTRACT_LOG_DIR}/extract_shard${SHARD_INDEX}_gpu${EXTRACT_GPU}.log"
    echo "Starting shard ${SHARD_INDEX}/${NUM_EXTRACT_SHARDS} on cuda:${EXTRACT_GPU}; log: ${LOG_FILE}"
    (
      PYTHONUNBUFFERED=1 "${PYTHON_BIN}" tools/vigor_vlpart_topk_reranker.py \
        --mode extract_train_cache \
        "${COMMON_ARGS[@]}" \
        --device "cuda:${EXTRACT_GPU}" \
        --extract_shard_index "${SHARD_INDEX}" \
        --extract_num_shards "${NUM_EXTRACT_SHARDS}" \
        2>&1 | tee "${LOG_FILE}"
    ) &
    PIDS+=("$!")
  done

  FAILED=0
  for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
      FAILED=1
    fi
  done
  if [[ "${FAILED}" != "0" ]]; then
    echo "One or more feature-cache extraction shards failed. See logs in ${EXTRACT_LOG_DIR}" >&2
    exit 1
  fi
fi

echo ""
echo "========================================================================"
echo "  Step 2/2: train formula scorer from cached features"
echo "========================================================================"

ARGS=(
  --mode train_formula_scorer
  "${COMMON_ARGS[@]}"
  --device "cuda:${GPU_ID}"
  --extract_num_shards "${NUM_EXTRACT_SHARDS}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --ranking_temperature "${RANKING_TEMPERATURE}"
  --formula_alpha_init "${FORMULA_ALPHA_INIT}"
  --swanlab_project "${SWANLAB_PROJECT}"
  --swanlab_exp_name "${SWANLAB_EXP_NAME}"
)

if [[ "${SWANLAB_ENABLED}" == "true" || "${SWANLAB_ENABLED}" == "1" ]]; then
  ARGS+=(--swanlab_enabled)
fi

"${PYTHON_BIN}" tools/vigor_vlpart_topk_reranker.py "${ARGS[@]}"

echo ""
echo "========================================================================"
echo "  VLPart top-K formula scorer training finished"
echo "========================================================================"
