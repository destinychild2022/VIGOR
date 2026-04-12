#!/usr/bin/env bash
set -euo pipefail

# Download the model weights needed by the VIGOR LLMSeg training/test scripts.
# Usage:
#   bash scripts/download_model_weights.sh
# Optional:
#   MODEL_ROOT=/opt/data/private/model HF_ENDPOINT=https://hf-mirror.com bash scripts/download_model_weights.sh

MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/model}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
LISA_DIR="${LISA_DIR:-${MODEL_ROOT}/LISA_Plus_7b}"
CLIP_DIR="${CLIP_DIR:-${MODEL_ROOT}/clip-vit-large-patch14}"
SAM_DIR="${SAM_DIR:-${MODEL_ROOT}/SAM-vit-h}"
DINO_DIR="${DINO_DIR:-${MODEL_ROOT}/dinov2_vitl14}"

LISA_REPO="${LISA_REPO:-JiaaZ/GLOVER_plus}"
CLIP_REPO="${CLIP_REPO:-openai/clip-vit-large-patch14}"
SAM_URL="${SAM_URL:-https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth}"
DINO_URL="${DINO_URL:-https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth}"

mkdir -p "${LISA_DIR}" "${CLIP_DIR}" "${SAM_DIR}" "${DINO_DIR}"

download_url() {
  local url="$1"
  local output="$2"

  if [ -s "${output}" ]; then
    echo "[skip] ${output} already exists"
    return
  fi

  echo "[download] ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --continue-at - --output "${output}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "${output}" "${url}"
  else
    python - "$url" "$output" <<'PY'
import sys
import urllib.request

url, output = sys.argv[1], sys.argv[2]
urllib.request.urlretrieve(url, output)
PY
  fi
}

python - "$LISA_REPO" "$LISA_DIR" "$CLIP_REPO" "$CLIP_DIR" <<'PY'
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is not installed. Install requirements first, "
        "for example: pip install -r requirements_fixed_new.txt"
    ) from exc

lisa_repo, lisa_dir, clip_repo, clip_dir = sys.argv[1:5]

common_kwargs = {}
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    common_kwargs["token"] = hf_token
force_download = os.environ.get("FORCE_DOWNLOAD", "0").lower() in {"1", "true", "yes"}

print(f"[download] {lisa_repo} -> {lisa_dir}")
snapshot_download(
    repo_id=lisa_repo,
    local_dir=lisa_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    force_download=force_download,
    **common_kwargs,
)

print(f"[download] {clip_repo} -> {clip_dir}")
snapshot_download(
    repo_id=clip_repo,
    local_dir=clip_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    force_download=force_download,
    allow_patterns=[
        "config.json",
        "merges.txt",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
    **common_kwargs,
)
PY

download_url "${SAM_URL}" "${SAM_DIR}/sam_vit_h_4b8939.pth"
download_url "${DINO_URL}" "${DINO_DIR}/dinov2_vitl14_pretrain.pth"

echo
echo "Done. Model files are under: ${MODEL_ROOT}"
du -sh "${LISA_DIR}" "${CLIP_DIR}" "${SAM_DIR}/sam_vit_h_4b8939.pth" "${DINO_DIR}" 2>/dev/null || true
