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

# ========== Output ==========
OUTPUT_ROOT="vlpart_vigor_outputs/llmseg_topk_reranker_k${TOPK_MASK_K}"
OUTPUT_DIR="${OUTPUT_ROOT}/train_logs"
CHECKPOINT_PATH="${OUTPUT_ROOT}/checkpoints/reranker.pt"
FEATURE_CACHE_DIR="${OUTPUT_ROOT}/candidate_feature_cache_train"
EXTRACT_LOG_DIR="${OUTPUT_ROOT}/extract_logs"

# ========== Training ==========
EPOCHS="20"
LR="1e-3"
WEIGHT_DECAY="1e-4"
HIDDEN_DIM="32"
DROPOUT="0.0"
KL_WEIGHT="1.0"
MSE_WEIGHT="1.0"
TARGET_TEMPERATURE="0.10"
RANKING_TEMPERATURE="1.0"
SEED="42"
MAX_SAMPLES=""
PREDICTION_CACHE_SIZE="128"
DEBUG="0"

# ========== SwanLab ==========
SWANLAB_ENABLED="true"
SWANLAB_API_KEY="17UKzqoPx2VI4PLzCHYdH"
SWANLAB_PROJECT="VLPart"
SWANLAB_EXP_NAME="vigor_topk_reranker_k${TOPK_MASK_K}"
if [[ -n "${SWANLAB_API_KEY}" ]]; then
  export SWANLAB_API_KEY="${SWANLAB_API_KEY}"
fi

echo "========================================================================"
echo "  Train VLPart top-K affordance reranker"
echo "========================================================================"
echo "Top-K masks: ${LLMSEG_TOPK_MASKS_DIR}"
echo "Top-K K: ${TOPK_MASK_K}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Feature cache: ${FEATURE_CACHE_DIR}"
echo "Parallel extract: ${PARALLEL_EXTRACT}"
echo "Extract GPUs: ${EXTRACT_GPUS[*]}"
echo "Train GPU: ${GPU_ID}"
echo "Epochs: ${EPOCHS}"
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
  --seed "${SEED}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  COMMON_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

if [[ "${DEBUG}" == "true" || "${DEBUG}" == "1" ]]; then
  COMMON_ARGS+=(--debug)
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
echo "  Step 2/2: train reranker from cached features"
echo "========================================================================"

ARGS=(
  --mode train
  "${COMMON_ARGS[@]}"
  --device "cuda:${GPU_ID}"
  --extract_num_shards "${NUM_EXTRACT_SHARDS}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --hidden_dim "${HIDDEN_DIM}"
  --dropout "${DROPOUT}"
  --kl_weight "${KL_WEIGHT}"
  --mse_weight "${MSE_WEIGHT}"
  --target_temperature "${TARGET_TEMPERATURE}"
  --ranking_temperature "${RANKING_TEMPERATURE}"
  --swanlab_project "${SWANLAB_PROJECT}"
  --swanlab_exp_name "${SWANLAB_EXP_NAME}"
)

if [[ "${SWANLAB_ENABLED}" == "true" || "${SWANLAB_ENABLED}" == "1" ]]; then
  ARGS+=(--swanlab_enabled)
fi

"${PYTHON_BIN}" tools/vigor_vlpart_topk_reranker.py "${ARGS[@]}"

echo ""
echo "========================================================================"
echo "  VLPart top-K reranker training finished"
echo "========================================================================"
