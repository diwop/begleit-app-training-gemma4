#!/usr/bin/env bash
set -euo pipefail

# Determine repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SGLANG_CONTAINER_DIR="${WORKSPACE_ROOT}/images/sglang_sandbox"
CONTAINER_PYTHON="/usr/bin/python3"
HF_CACHE_DIR="${HOME}/.cache/huggingface"

# Disable all external telemetry and tracking in host and container environments
export DO_NOT_TRACK=1
export AXOLOTL_DO_NOT_TRACK=1
export POSTHOG_DISABLED=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export WANDB_MODE=offline
export ANONYMIZED_TELEMETRY=False
export DISABLE_TELEMETRY=1

echo "============================================================"
echo " Starting Gemma 4 Evaluation (SGLang)"
echo "============================================================"
echo "[INFO] Workspace       : ${WORKSPACE_ROOT}"
echo "[INFO] Container       : ${SGLANG_CONTAINER_DIR}"
echo "[INFO] Container Python: ${CONTAINER_PYTHON}"
echo "[INFO] HF Cache        : ${HF_CACHE_DIR}"
echo "============================================================"

if [ ! -d "${SGLANG_CONTAINER_DIR}" ]; then
  echo "[ERROR] SGLang container directory '${SGLANG_CONTAINER_DIR}' not found."
  echo "[INFO] Run 'bash scripts/prepare_images.sh eval' on the login node first."
  exit 1
fi

# Configure local fast SSD scratch via SLURM_TMPDIR if available (auto-cleaned by Slurm)
LOCAL_SCRATCH_ARGS=()
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR}" ]; then
  SCRATCH_ROOT="${SLURM_TMPDIR}/${SLURM_JOB_ID:-eval_scratch}"
  mkdir -p "${SCRATCH_ROOT}/tmp" "${SCRATCH_ROOT}/triton" "${SCRATCH_ROOT}/torch_ext"
  echo "[INFO] Local Scratch   : ${SCRATCH_ROOT} (NVMe SSD $SLURM_TMPDIR)"
  LOCAL_SCRATCH_ARGS=(
    --bind "${SCRATCH_ROOT}:/scratch_local"
    --bind "${SCRATCH_ROOT}/tmp:/tmp"
    --env TMPDIR="/scratch_local/tmp"
    --env TRITON_CACHE_DIR="/scratch_local/triton"
    --env TORCH_EXTENSIONS_DIR="/scratch_local/torch_ext"
  )
else
  echo "[INFO] Local Scratch   : Using default /tmp (SLURM_TMPDIR not set)"
  LOCAL_SCRATCH_ARGS=(
    --bind "/tmp:/tmp"
    --env TMPDIR="/tmp"
  )
fi

# Execute evaluation.py inside the SGLang container in offline mode
apptainer exec \
  --nv \
  --env HF_HOME="${HF_CACHE_DIR}" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DISABLE_TELEMETRY=1 \
  --env DO_NOT_TRACK=1 \
  --env AXOLOTL_DO_NOT_TRACK=1 \
  --env POSTHOG_DISABLED=1 \
  --env WANDB_DISABLED=true \
  --env WANDB_MODE=offline \
  --env ANONYMIZED_TELEMETRY=False \
  --env DISABLE_TELEMETRY=1 \
  --env NCCL_P2P_DISABLE=1 \
  --env NCCL_IB_DISABLE=1 \
  --env TORCH_NCCL_BLOCKING_WAIT=1 \
  --env PYTHONUSERBASE="${HOME}/.local" \
  --env PYTHONNOUSERSITE=0 \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH="/repo/src-eval:${HOME}/.local/lib/python3.12/site-packages:${HOME}/.local/lib/python3.11/site-packages:${PYTHONPATH:-}" \
  "${LOCAL_SCRATCH_ARGS[@]}" \
  --bind "${WORKSPACE_ROOT}:/repo" \
  --bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}" \
  --bind "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  --bind "${HOME}/.local:${HOME}/.local" \
  --pwd /repo \
  "${SGLANG_CONTAINER_DIR}" \
  "${CONTAINER_PYTHON}" /repo/src-eval/evaluation.py

echo "============================================================"
echo "[SUCCESS] Gemma 4 Evaluation finished!"
echo "============================================================"
