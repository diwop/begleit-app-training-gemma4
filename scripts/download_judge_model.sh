#!/usr/bin/env bash
set -euo pipefail

# Determine repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SGLANG_CONTAINER_DIR="${WORKSPACE_ROOT}/images/sglang_sandbox"
CONTAINER_PYTHON="/usr/bin/python3"
HF_CACHE_DIR="${HOME}/.cache/huggingface"

# Target Judge Model
JUDGE_MODEL="${JUDGE_MODEL_NAME:-nvidia/Llama-4-Scout-17B-16E-Instruct-FP8}"

echo "============================================================"
echo " Downloading LLM-as-a-Judge Model Snapshot (Llama 4 Scout)"
echo " (Run this on hsuper-login01 where internet is available)"
echo "============================================================"
echo "[INFO] Cache Directory : ${HF_CACHE_DIR}"
echo "[INFO] Container Python: ${CONTAINER_PYTHON}"
echo "[INFO] Target Model    : ${JUDGE_MODEL}"
echo "============================================================"

if [ ! -d "${SGLANG_CONTAINER_DIR}" ]; then
  echo "[ERROR] SGLang container directory '${SGLANG_CONTAINER_DIR}' not found."
  echo "[INFO] Run 'bash scripts/prepare_images.sh eval' first."
  exit 1
fi

apptainer exec \
  --env HF_HOME="${HF_CACHE_DIR}" \
  --env HF_HUB_ENABLE_HF_TRANSFER=1 \
  --bind "${WORKSPACE_ROOT}:/repo" \
  --bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}" \
  --bind "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "${SGLANG_CONTAINER_DIR}" \
  "${CONTAINER_PYTHON}" -c "
import os
import sys
from huggingface_hub import snapshot_download

target_model = '${JUDGE_MODEL}'
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
if not token:
    token_file = os.path.expanduser('~/.cache/huggingface/token')
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            token = f.read().strip()
if not token:
    token_file_alt = '${HF_CACHE_DIR}/token'
    if os.path.exists(token_file_alt):
        with open(token_file_alt, 'r') as f:
            token = f.read().strip()

print(f'[INFO] Downloading snapshot for {target_model} (token present: {bool(token)})...')
try:
    path = snapshot_download(
        repo_id=target_model,
        token=token,
        resume_download=True,
        max_workers=8,
    )
    print(f'[SUCCESS] Downloaded {target_model} to: {path}')
except Exception as e:
    print(f'[ERROR] Failed to download {target_model}: {e}', file=sys.stderr)
    sys.exit(1)
"

echo "============================================================"
echo "[SUCCESS] Judge model download complete!"
echo "============================================================"
