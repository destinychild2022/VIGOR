#!/usr/bin/env bash
set -euo pipefail

# ========================================================================
# Quick checks before GT Object Prior training:
#   1) visual overlays for full RGB / GT object prior / GT affordance / pred mask
#   2) proposal prior_score distribution
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

# ========== Model ==========
CONFIG_FILE="configs/vigor/swinbase_vigor_stage1_gt_object_prior.yaml"
WEIGHTS="/opt/data/private/LLMSeg/VLPart/output/VLPart/vigor_swinbase_easy_stage1_bs16_lr4e-5/model_final.pth"
CONFIDENCE_THRESHOLD="0.05"

# ========== Dataset ==========
DATA_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K_new"
EASY_JSON="open_vocab_grasp_easy_object_mix.json"
HARD_JSON="open_vocab_grasp_hard_object_mix.json"

# ========== Output / Sampling ==========
OUTPUT_DIR="vlpart_vigor_outputs/gt_object_prior_quick_check"
SAMPLES_PER_SPLIT="20"
PRIOR_SCORE_SAMPLES="8"
TOPK_PROPOSALS="10"
SEED="7"
SKIP_MODEL="false"

# ========== Vocabulary ==========
CUSTOM_VOCABULARY="cylindrical side surface,hexagonal side face,flat side surface,whole object"

echo "========================================================================"
echo "  VLPart VIGOR GT Object Prior quick check"
echo "========================================================================"
echo "Config: ${CONFIG_FILE}"
echo "Weights: ${WEIGHTS}"
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Samples per split: ${SAMPLES_PER_SPLIT}"
echo "Prior-score samples: ${PRIOR_SCORE_SAMPLES}"
echo "GPU: ${GPU_ID}"
echo "========================================================================"

ARGS=(
  --config-file "${CONFIG_FILE}"
  --weights "${WEIGHTS}"
  --data_dir "${DATA_DIR}"
  --easy_json_file "${EASY_JSON}"
  --hard_json_file "${HARD_JSON}"
  --output_dir "${OUTPUT_DIR}"
  --custom_vocabulary "${CUSTOM_VOCABULARY}"
  --samples_per_split "${SAMPLES_PER_SPLIT}"
  --prior_score_samples "${PRIOR_SCORE_SAMPLES}"
  --topk_proposals "${TOPK_PROPOSALS}"
  --confidence-threshold "${CONFIDENCE_THRESHOLD}"
  --device "cuda:${GPU_ID}"
  --seed "${SEED}"
)

if [[ "${SKIP_MODEL}" == "true" || "${SKIP_MODEL}" == "1" ]]; then
  ARGS+=(--skip_model)
fi

"${PYTHON_BIN}" tools/check_vigor_gt_object_prior.py "${ARGS[@]}"

echo ""
echo "========================================================================"
echo "  Quick check finished"
echo "========================================================================"
