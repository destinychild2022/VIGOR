#!/usr/bin/env bash
set -euo pipefail

# ========================================================================
# LLMSeg -> region RGB -> VLPart VIGOR-100K evaluation
#
# Stage 1: LLMSeg selects one SAM candidate mask for each instruction.
# Stage 2: the selected mask is projected back to scene RGB, then VLPart
#          predicts the final mask on that region RGB image.
# Metrics and visualization format follow test/test_llmseg_vigor.sh.
# ========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ========== Python environment ==========
VENV_ROOT="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-${VENV_ROOT}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

PY_SITE="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
export PATH="/usr/local/cuda/bin:${VENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${PY_SITE}/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/detectron2:${REPO_ROOT}/VLPart:${REPO_ROOT}/VLPart/demo:${PYTHONPATH:-}"

# ========== LLMSeg model ==========
LISA_MODEL_PATH="/opt/data/private/model/LISA_Plus_7b"
CHECKPOINT_PATH="/opt/data/private/LLMSeg/runs/finetune_llmseg_vigor_simple-object/ckpt_model/epoch_20"
CLIP_PATH="/opt/data/private/model/clip-vit-large-patch14"
SAM_VIT_PATH="/opt/data/private/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# ========== Dataset ==========
TEST_DATA_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K_new/test"
TEST_EASY_JSON="open_vocab_grasp_easy_object_mix.json"
TEST_HARD_JSON="open_vocab_grasp_hard_object_mix.json"
SAM_MASKS_DIR="/opt/data/private/LLMSeg/dataset/VIGOR-100K/test_mask/sam_masks3"

# ========== VLPart model ==========
VLPART_CONFIG_FILE="configs/vigor/swinbase_vigor_easy_stage1.yaml"
VLPART_WEIGHTS="/opt/data/private/LLMSeg/VLPart/output/VLPart/vigor_swinbase_easy_stage1_llmseg_noisy_ratio20_from_gtbaseline_lr4e-6_3ep/model_final.pth"
VLPART_CONFIDENCE_THRESHOLD="0.05"
VLPART_MASK_SELECTION="top1"
VLPART_VOCABULARY="custom"
VLPART_CUSTOM_VOCABULARY="cylindrical side surface,hexagonal side face,flat side surface,whole object"
VLPART_PREDICTION_CACHE_SIZE="256"

# ========== Output ==========
OUTPUT_ROOT="./result_llmseg_vlpart"
OUTPUT_DIR="${OUTPUT_ROOT}/results"
VIS_DIR="${OUTPUT_ROOT}/visualizations"
REGION_RGB_DIR="${OUTPUT_ROOT}/region_rgb"
LLMSEG_MASKS_DIR="${OUTPUT_ROOT}/llmseg_masks"
VLPART_PRED_MASKS_DIR="${OUTPUT_ROOT}/vlpart_pred_masks"
SAVE_VIS="false"
SAVE_PRED_MASKS="false"

# ========== Test settings ==========
PRECISION="bf16"
ICR_THRESHOLDS="0.3,0.4,0.5,0.6,0.7,0.8,0.9"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,out_proj"

DEBUG="${DEBUG:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GPU_ID="${GPU_ID:-1}"
VLPART_GPU_ID="${VLPART_GPU_ID:-${GPU_ID}}"
SPLIT="${SPLIT:-both}"
WORKERS="${WORKERS:-12}"

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ ("${SAVE_VIS}" == "true" || "${SAVE_VIS}" == "1") && -e "${VIS_DIR}" ]]; then
  VIS_DIR="${VIS_DIR}_${RUN_TIMESTAMP}"
fi
if [[ ("${SAVE_PRED_MASKS}" == "true" || "${SAVE_PRED_MASKS}" == "1") && -e "${VLPART_PRED_MASKS_DIR}" ]]; then
  VLPART_PRED_MASKS_DIR="${VLPART_PRED_MASKS_DIR}_${RUN_TIMESTAMP}"
fi

echo "========================================================================"
echo "  LLMSeg -> region RGB -> VLPart VIGOR-100K evaluation"
echo "========================================================================"
echo "Python: ${PYTHON_BIN}"
echo "LLMSeg checkpoint: ${CHECKPOINT_PATH}"
echo "VLPart config: ${VLPART_CONFIG_FILE}"
echo "VLPart weights: ${VLPART_WEIGHTS}"
echo "Data dir: ${TEST_DATA_DIR}"
echo "Split: ${SPLIT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Visualization dir: ${VIS_DIR}"
echo "Region RGB dir: ${REGION_RGB_DIR}"
echo "VLPart pred masks dir: ${VLPART_PRED_MASKS_DIR}"
echo "========================================================================"

ARGS=(
  --version "${LISA_MODEL_PATH}"
  --checkpoint "${CHECKPOINT_PATH}"
  --vision_tower "${CLIP_PATH}"
  --vision_pretrained "${SAM_VIT_PATH}"
  --data_dir "${TEST_DATA_DIR}"
  --easy_json_file "${TEST_EASY_JSON}"
  --hard_json_file "${TEST_HARD_JSON}"
  --sam_masks_dir "${SAM_MASKS_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --vis_dir "${VIS_DIR}"
  --region_rgb_dir "${REGION_RGB_DIR}"
  --llmseg_masks_dir "${LLMSEG_MASKS_DIR}"
  --vlpart_pred_masks_dir "${VLPART_PRED_MASKS_DIR}"
  --precision "${PRECISION}"
  --icr_thresholds "${ICR_THRESHOLDS}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --lora_target_modules "${LORA_TARGET_MODULES}"
  --device "cuda:${GPU_ID}"
  --vlpart_device "cuda:${VLPART_GPU_ID}"
  --split "${SPLIT}"
  --workers "${WORKERS}"
  --vlpart_config_file "${VLPART_CONFIG_FILE}"
  --vlpart_weights "${VLPART_WEIGHTS}"
  --vlpart_confidence_threshold "${VLPART_CONFIDENCE_THRESHOLD}"
  --vlpart_mask_selection "${VLPART_MASK_SELECTION}"
  --vlpart_vocabulary "${VLPART_VOCABULARY}"
  --vlpart_custom_vocabulary "${VLPART_CUSTOM_VOCABULARY}"
  --vlpart_prediction_cache_size "${VLPART_PREDICTION_CACHE_SIZE}"
  --use_mm_start_end
)

if [[ "${SAVE_VIS}" == "true" || "${SAVE_VIS}" == "1" ]]; then
  ARGS+=(--save_vis)
fi

if [[ "${SAVE_PRED_MASKS}" == "true" || "${SAVE_PRED_MASKS}" == "1" ]]; then
  ARGS+=(--save_pred_masks)
fi

if [[ "${DEBUG}" == "true" || "${DEBUG}" == "1" ]]; then
  ARGS+=(--debug)
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

"${PYTHON_BIN}" test/test_llmseg_vlpart_vigor.py "${ARGS[@]}"

echo ""
echo "========================================================================"
echo "  LLMSeg+VLPart evaluation finished"
echo "========================================================================"
