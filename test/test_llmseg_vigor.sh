#!/bin/bash
# ========================================================================
# LLMSeg VIGOR-100K 测试脚本
# 
# 【重要】LLMSeg 的推理需要 SAM 候选 mask！
# 它是从候选 mask 中选择相似度最高的，而不是直接生成 mask
# ========================================================================

# ========== 模型路径配置 ==========
LISA_MODEL_PATH="../root/autodl-tmp/model/LISA_Plus_7b"
CKPT_ROOT="../root/autodl-tmp/runs/finetune_llmseg_vigor_simple-DFormer/ckpt_model"
CKPT_LIST="${CKPT_LIST:-best epoch_5 epoch_10 epoch_15 epoch_20}"
CLIP_PATH="../root/autodl-tmp/model/clip-vit-large-patch14"
SAM_VIT_PATH="../root/autodl-tmp/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# ========== 数据集配置 ==========
# 测试数据集路径
TEST_DATA_DIR="../root/autodl-tmp/VIGOR-100K_new/test"
# SAM 候选 mask 目录 (必需！)
SAM_MASKS_DIR="../root/autodl-tmp/test_mask/sam_masks3"
# 必须和训练/验证时的 depth normalization range 一致；默认按训练脚本同样扫描 train/depth。
AUTO_VIGOR_DEPTH_RANGE="${AUTO_VIGOR_DEPTH_RANGE:-true}"
VIGOR_DEPTH_RANGE_LOW_P="${VIGOR_DEPTH_RANGE_LOW_P:-0.1}"
VIGOR_DEPTH_RANGE_HIGH_P="${VIGOR_DEPTH_RANGE_HIGH_P:-99.9}"
VIGOR_DEPTH_SCAN_SAMPLES_PER_FILE="${VIGOR_DEPTH_SCAN_SAMPLES_PER_FILE:-4096}"
VIGOR_DEPTH_MIN="${VIGOR_DEPTH_MIN:-0.6}"
VIGOR_DEPTH_MAX="${VIGOR_DEPTH_MAX:-1.85}"

# ========== 输出配置 ==========
OUTPUT_DIR="../root/autodl-tmp/result"
VIS_DIR="../root/autodl-tmp/vis_output_DFormer"  # 可视化输出目录
SAVE_VIS="false"  # 是否保存可视化图片 (true/false)

# ========== 测试配置 ==========
PRECISION="bf16"
TEST_WORKERS="${TEST_WORKERS:-8}"
IOU_SELECTION_ONLY="${IOU_SELECTION_ONLY:-false}"    # true: 根据pred_similarity选择 mask
IOU_THRESHOLD="${IOU_THRESHOLD:-0.5}"
ICR_THRESHOLDS="0.3,0.4,0.5,0.6,0.7,0.8,0.9"

# ========== 调试配置 ==========
DEBUG="${DEBUG:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GPU_ID="${GPU_ID:-0}"
SPLIT="${SPLIT:-both}"  # 可选: both, easy, hard

# ========================================================================

cd "$(dirname "$0")/.." || exit 1

if [ "${AUTO_VIGOR_DEPTH_RANGE}" = true ]; then
  DEPTH_RANGE_OUTPUT=$(python - <<PY
from pathlib import Path
import numpy as np

test_dir = Path("${TEST_DATA_DIR}")
depth_dir = test_dir.parent / "train" / "depth"
low_p = float("${VIGOR_DEPTH_RANGE_LOW_P}")
high_p = float("${VIGOR_DEPTH_RANGE_HIGH_P}")
samples_per_file = int("${VIGOR_DEPTH_SCAN_SAMPLES_PER_FILE}")
files = sorted(depth_dir.glob("*.npy"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
if not files:
    raise SystemExit(f"no depth npy files found in {depth_dir}")
rng = np.random.default_rng(20260413)
chunks = []
exact_min = np.inf
exact_max = -np.inf
for p in files:
    arr = np.load(p, mmap_mode="r")
    vals = np.asarray(arr).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        continue
    exact_min = min(exact_min, float(vals.min()))
    exact_max = max(exact_max, float(vals.max()))
    take = min(samples_per_file, vals.size)
    if take > 0:
        idx = rng.choice(vals.size, size=take, replace=False)
        chunks.append(np.asarray(vals[idx], dtype=np.float32))
if not chunks:
    raise SystemExit(f"no finite depth values found in {depth_dir}")
sample = np.concatenate(chunks)
lo = float(np.percentile(sample, low_p))
hi = float(np.percentile(sample, high_p))
lo = min(lo, exact_min)
hi = max(hi, exact_max)
margin = max((hi - lo) * 0.01, 1e-4)
lo -= margin
hi += margin
print(f"{lo:.8f} {hi:.8f} {len(files)} {exact_min:.8f} {exact_max:.8f}")
PY
)
  if [ $? -eq 0 ]; then
    read -r VIGOR_DEPTH_MIN VIGOR_DEPTH_MAX VIGOR_DEPTH_FILES VIGOR_DEPTH_EXACT_MIN VIGOR_DEPTH_EXACT_MAX <<< "$DEPTH_RANGE_OUTPUT"
    echo "[DepthRange] auto range from train/depth: files=${VIGOR_DEPTH_FILES} exact=[${VIGOR_DEPTH_EXACT_MIN}, ${VIGOR_DEPTH_EXACT_MAX}] range=[${VIGOR_DEPTH_MIN}, ${VIGOR_DEPTH_MAX}]"
  else
    echo "[DepthRange] auto scan failed; using fallback range=[${VIGOR_DEPTH_MIN}, ${VIGOR_DEPTH_MAX}]"
  fi
fi
export VIGOR_DEPTH_MIN
export VIGOR_DEPTH_MAX

echo "========================================================================"
echo "  LLMSeg VIGOR-100K 批量测试"
echo "========================================================================"
echo "基础模型: ${LISA_MODEL_PATH}"
echo "权重根目录: ${CKPT_ROOT}"
echo "测试权重: ${CKPT_LIST}"
echo "测试数据: ${TEST_DATA_DIR}"
echo "SAM候选mask: ${SAM_MASKS_DIR}"
echo "Depth归一化: [${VIGOR_DEPTH_MIN}, ${VIGOR_DEPTH_MAX}]"
echo "测试 workers: ${TEST_WORKERS}"
echo "Mask选择方式: $([ "${IOU_SELECTION_ONLY}" = "true" ] && echo "pred_similarity_argmax" || echo "pred_iou_threshold_${IOU_THRESHOLD}")"
echo "输出根目录: ${OUTPUT_DIR}"
echo "可视化保存: ${SAVE_VIS}"
echo "========================================================================"

# 公共参数
COMMON_ARGS="
    --version=${LISA_MODEL_PATH}
    --vision_tower=${CLIP_PATH}
    --vision_pretrained=${SAM_VIT_PATH}
    --data_dir=${TEST_DATA_DIR}
    --sam_masks_dir=${SAM_MASKS_DIR}
    --depth_min=${VIGOR_DEPTH_MIN}
    --depth_max=${VIGOR_DEPTH_MAX}
    --precision=${PRECISION}
    --workers=${TEST_WORKERS}
    --iou_threshold=${IOU_THRESHOLD}
    --icr_thresholds=${ICR_THRESHOLDS}
    --device=cuda:${GPU_ID}
    --split=${SPLIT}
    --use_mm_start_end
"

# 添加 mask 选择方式；true 时和训练脚本 --iou_selection_only 的 validate() 路径一致。
if [ "$IOU_SELECTION_ONLY" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --iou_selection_only"
fi

# 添加可视化保存参数；默认 false 时不会生成可视化图片。
if [ "$SAVE_VIS" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --save_vis --vis_dir=${VIS_DIR}"
    echo "可视化保存: 开启"
fi

# 添加调试参数
if [ "$DEBUG" = "1" ] || [ "$DEBUG" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --debug"
    echo "调试模式: 开启"
fi

# 添加最大样本数限制
if [ -n "$MAX_SAMPLES" ]; then
    COMMON_ARGS="${COMMON_ARGS} --max_samples=${MAX_SAMPLES}"
    echo "最大样本数: ${MAX_SAMPLES}"
fi

for CKPT_NAME in ${CKPT_LIST}; do
    CHECKPOINT_PATH="${CKPT_ROOT}/${CKPT_NAME}"
    CKPT_OUTPUT_DIR="${OUTPUT_DIR}/${CKPT_NAME}"

    echo ""
    echo "========================================================================"
    echo "  测试 checkpoint: ${CKPT_NAME}"
    echo "========================================================================"
    echo "微调权重: ${CHECKPOINT_PATH}"
    echo "输出目录: ${CKPT_OUTPUT_DIR}"

    if [ ! -d "${CHECKPOINT_PATH}" ]; then
        echo "[警告] checkpoint 目录不存在，跳过: ${CHECKPOINT_PATH}"
        continue
    fi

    python test/test_llmseg_vigor.py ${COMMON_ARGS} \
        --checkpoint=${CHECKPOINT_PATH} \
        --output_dir=${CKPT_OUTPUT_DIR}
done

echo ""
echo "========================================================================"
echo "  批量测试完成"
echo "========================================================================"
