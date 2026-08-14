#!/usr/bin/env bash
set -euo pipefail

# Determine repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_CONTAINER_DIR="${WORKSPACE_ROOT}/images/vllm_sandbox"
CONTAINER_PYTHON="/usr/bin/python3"
HF_CACHE_DIR="${HOME}/.cache/huggingface"

echo "============================================================"
echo " Starting Gemma 4 Baseline Evaluation (vLLM)"
echo "============================================================"
echo "[INFO] Workspace       : ${WORKSPACE_ROOT}"
echo "[INFO] Container       : ${VLLM_CONTAINER_DIR}"
echo "[INFO] Container Python: ${CONTAINER_PYTHON}"
echo "[INFO] HF Cache        : ${HF_CACHE_DIR}"
echo "============================================================"

if [ ! -d "${VLLM_CONTAINER_DIR}" ]; then
  echo "[ERROR] vLLM container directory '${VLLM_CONTAINER_DIR}' not found."
  echo "[INFO] Run 'bash scripts/prepare_images.sh' on the login node first."
  exit 1
fi

mkdir -p "${HF_CACHE_DIR}"

# Execute evaluation.py inside the vLLM container in offline mode
apptainer exec \
  --nv \
  --env HF_HOME="${HF_CACHE_DIR}" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --bind "${WORKSPACE_ROOT}:/repo" \
  --bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}" \
  --bind "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  --pwd /repo \
  "${VLLM_CONTAINER_DIR}" \
  "${CONTAINER_PYTHON}" /repo/src-eval/evaluation.py

echo "============================================================"
echo "[SUCCESS] Gemma 4 Baseline Evaluation finished!"
echo "============================================================"
