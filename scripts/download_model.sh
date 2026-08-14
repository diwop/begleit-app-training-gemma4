#!/usr/bin/env bash
set -euo pipefail

# Determine repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_CONTAINER_DIR="${WORKSPACE_ROOT}/images/vllm_sandbox"
CONTAINER_PYTHON="/usr/bin/python3"
HF_CACHE_DIR="${HOME}/.cache/huggingface"
MODEL_NAME="${1:-RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic}"

echo "============================================================"
echo " Downloading Hugging Face Model to Shared Cluster Cache"
echo " (Run this on hsuper-login01 where internet is available)"
echo "============================================================"
echo "[INFO] Model Name      : ${MODEL_NAME}"
echo "[INFO] Cache Directory : ${HF_CACHE_DIR}"
echo "[INFO] Container Python: ${CONTAINER_PYTHON}"
echo "============================================================"

mkdir -p "${HF_CACHE_DIR}"

apptainer exec \
  --env HF_HOME="${HF_CACHE_DIR}" \
  --bind "${WORKSPACE_ROOT}:/repo" \
  --bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}" \
  --bind "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "${VLLM_CONTAINER_DIR}" \
  "${CONTAINER_PYTHON}" -c "
import os
from huggingface_hub import snapshot_download

model_name = '${MODEL_NAME}'
token = os.environ.get('HF_TOKEN', None)

print(f'[INFO] Downloading snapshot for {model_name}...')
path = snapshot_download(repo_id=model_name, token=token)
print(f'[SUCCESS] Model snapshot downloaded to: {path}')
"

echo "============================================================"
echo "[SUCCESS] Model download complete and cached on shared filesystem!"
echo "============================================================"
