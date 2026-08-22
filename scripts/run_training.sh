#!/usr/bin/env bash
set -euo pipefail

# Determine repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AXOLOTL_CONTAINER_DIR="${WORKSPACE_ROOT}/images/axolotl_sandbox"
HF_CACHE_DIR="${HOME}/.cache/huggingface"
CONFIG_PATH="/repo/src-train/config.yml"

echo "============================================================"
echo " Starting Gemma 4 26B-A4B Adapter Fine-Tuning (Axolotl)"
echo "============================================================"
echo "[INFO] Workspace       : ${WORKSPACE_ROOT}"
echo "[INFO] Container       : ${AXOLOTL_CONTAINER_DIR}"
echo "[INFO] HF Cache        : ${HF_CACHE_DIR}"
echo "[INFO] Config          : ${CONFIG_PATH}"

# Configure local fast SSD scratch via SLURM_TMPDIR if available
LOCAL_SCRATCH_ARGS=()
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR}" ]; then
  SCRATCH_ROOT="${SLURM_TMPDIR}/${SLURM_JOB_ID:-training_scratch}"
  mkdir -p "${SCRATCH_ROOT}/tmp" "${SCRATCH_ROOT}/triton" "${SCRATCH_ROOT}/torch_ext" "${SCRATCH_ROOT}/axolotl"
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

# Detect available GPUs
if command -v nvidia-smi &>/dev/null; then
  NUM_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
else
  # Fallback to PyTorch detection inside container or CUDA_VISIBLE_DEVICES
  NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
fi

echo "[INFO] Detected available GPUs: ${NUM_GPUS}"

if [ "${NUM_GPUS}" -lt 2 ]; then
  echo "============================================================"
  echo "[ERROR] Insufficient GPUs: Found ${NUM_GPUS} GPU(s)."
  echo "Gemma 4 26B-A4B MoE LoRA training under ZeRO-3 requires AT LEAST 2 GPUs."
  echo "Please allocate 2 or more GPUs (e.g. salloc --gpus 2 / #SBATCH --gpus=2)."
  echo "============================================================"
  exit 1
fi

mkdir -p "${HF_CACHE_DIR}" "${HOME}/.local" "${WORKSPACE_ROOT}/local/adapters" "${WORKSPACE_ROOT}/logs"

# Execute Axolotl training using accelerate launch inside the Apptainer container
echo "[INFO] Launching accelerate distributed training across ${NUM_GPUS} GPUs..."
apptainer exec \
  --nv \
  --env HF_HOME="${HF_CACHE_DIR}" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DISABLE_TELEMETRY=1 \
  --env NCCL_P2P_DISABLE=1 \
  --env NCCL_IB_DISABLE=1 \
  --env TORCH_NCCL_BLOCKING_WAIT=1 \
  "${LOCAL_SCRATCH_ARGS[@]}" \
  --bind "${WORKSPACE_ROOT}:/repo" \
  --bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}" \
  --bind "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  --bind "${HOME}/.local:${HOME}/.local" \
  --pwd /repo \
  "${AXOLOTL_CONTAINER_DIR}" \
  /workspace/axolotl-venv/bin/accelerate launch \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=bf16 \
    -m axolotl.cli.train "${CONFIG_PATH}"

echo "============================================================"
echo "[SUCCESS] Gemma 4 Adapter Fine-Tuning finished successfully!"
echo "============================================================"
