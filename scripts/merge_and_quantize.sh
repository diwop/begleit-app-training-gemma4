#!/usr/bin/env bash
set -euo pipefail

# Determine repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AXOLOTL_CONTAINER_DIR="${WORKSPACE_ROOT}/images/axolotl_sandbox"
HF_CACHE_DIR="${HOME}/.cache/huggingface"

echo "============================================================"
echo " Starting Gemma 4 Adapter Merge & FP8 Quantization"
echo "============================================================"
echo "[INFO] Workspace       : ${WORKSPACE_ROOT}"
echo "[INFO] Container       : ${AXOLOTL_CONTAINER_DIR}"
echo "[INFO] HF Cache        : ${HF_CACHE_DIR}"

LOCAL_SCRATCH_ARGS=()
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR}" ]; then
  SCRATCH_ROOT="${SLURM_TMPDIR}/${SLURM_JOB_ID:-merge_scratch}"
  mkdir -p "${SCRATCH_ROOT}/tmp" "${SCRATCH_ROOT}/triton" "${SCRATCH_ROOT}/torch_ext"
  echo "[INFO] Local Scratch   : ${SCRATCH_ROOT} (NVMe SSD $SLURM_TMPDIR)"
  LOCAL_SCRATCH_ARGS=(
    --bind "${SCRATCH_ROOT}:/scratch_local"
    --env TMPDIR="/scratch_local/tmp"
    --env TRITON_CACHE_DIR="/scratch_local/triton"
    --env TORCH_EXTENSIONS_DIR="/scratch_local/torch_ext"
  )
else
  echo "[INFO] Local Scratch   : Using default /tmp (SLURM_TMPDIR not set)"
fi
echo "============================================================"

if [ ! -d "${AXOLOTL_CONTAINER_DIR}" ]; then
  echo "[ERROR] Axolotl container directory '${AXOLOTL_CONTAINER_DIR}' not found."
  echo "[INFO] Run 'bash scripts/prepare_images.sh' on the login node first."
  exit 1
fi

mkdir -p "${HF_CACHE_DIR}" "${HOME}/.local" "${WORKSPACE_ROOT}/local/models" "${WORKSPACE_ROOT}/logs"

# Execute merge and quantization inside container
apptainer exec \
  --nv \
  --env HF_HOME="${HF_CACHE_DIR}" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DISABLE_TELEMETRY=1 \
  "${LOCAL_SCRATCH_ARGS[@]}" \
  --bind "${WORKSPACE_ROOT}:/repo" \
  --bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}" \
  --bind "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  --bind "${HOME}/.local:${HOME}/.local" \
  --pwd /repo \
  "${AXOLOTL_CONTAINER_DIR}" \
  /workspace/axolotl-venv/bin/python /repo/src-train/merge_and_quantize.py "$@"

echo "============================================================"
echo "[SUCCESS] Merge & FP8 Quantization completed successfully!"
echo "============================================================"
