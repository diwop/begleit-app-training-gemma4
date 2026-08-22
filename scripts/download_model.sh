#!/usr/bin/env bash
set -euo pipefail

# Determine repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SGLANG_CONTAINER_DIR="${WORKSPACE_ROOT}/images/sglang_sandbox"
CONTAINER_PYTHON="/usr/bin/python3"
HF_CACHE_DIR="${HOME}/.cache/huggingface"

# Models required for training, evaluation, and dynamic few-shot retrieval
MODELS=(
  "google/gemma-4-26b-a4b-it"
  "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
  "intfloat/multilingual-e5-base"
)

echo "============================================================"
echo " Preparing Training & Evaluation Dependencies and Models"
echo " (Run this on hsuper-login01 where internet is available)"
echo "============================================================"
echo "[INFO] Cache Directory : ${HF_CACHE_DIR}"
echo "[INFO] Container Python: ${CONTAINER_PYTHON}"
for m in "${MODELS[@]}"; do
  echo "[INFO] Target Model    : ${m}"
done
echo "============================================================"

mkdir -p "${HF_CACHE_DIR}" "${HOME}/.local"

# 1. Ensure textstat and sentence-transformers are installed into user site packages
echo "[INFO] Installing/verifying 'textstat', 'sentence-transformers', and 'llmcompressor' in SGLang environment..."
apptainer exec \
  --bind "${HOME}/.local:${HOME}/.local" \
  "${SGLANG_CONTAINER_DIR}" \
  "${CONTAINER_PYTHON}" -m pip install --user --no-cache-dir --break-system-packages textstat sentence-transformers llmcompressor

# 2. Download model snapshots into shared Hugging Face cache
echo "[INFO] Downloading Hugging Face model snapshots..."
apptainer exec \
  --env HF_HOME="${HF_CACHE_DIR}" \
  --bind "${WORKSPACE_ROOT}:/repo" \
  --bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}" \
  --bind "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "${SGLANG_CONTAINER_DIR}" \
  "${CONTAINER_PYTHON}" -c "
import os
from huggingface_hub import snapshot_download

models = [
    'google/gemma-4-26b-a4b-it',
    'RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic',
    'intfloat/multilingual-e5-base',
]
token = os.environ.get('HF_TOKEN', None)

for model_name in models:
    print(f'[INFO] Checking/downloading snapshot for {model_name}...')
    path = snapshot_download(repo_id=model_name, token=token)
    print(f'[SUCCESS] Snapshot ready at: {path}')
"

echo "============================================================"
echo "[SUCCESS] All model snapshots and dependencies ready!"
echo "============================================================"
