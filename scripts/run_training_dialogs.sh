#!/usr/bin/env bash
set -euo pipefail

# Determine repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AXOLOTL_CONTAINER_DIR="${WORKSPACE_ROOT}/images/axolotl_sandbox"
HF_CACHE_DIR="${HOME}/.cache/huggingface"
CONFIG_PATH="/repo/src-train/config_dialogs.yml"

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
echo " Starting Gemma 4 26B-A4B Dialogs End-to-End Pipeline"
echo " (1. LoRA Fine-Tuning  ->  2. Merge & FP8 Quantization)"
echo "============================================================"
echo "[INFO] Workspace       : ${WORKSPACE_ROOT}"
echo "[INFO] Container       : ${AXOLOTL_CONTAINER_DIR}"
echo "[INFO] HF Cache        : ${HF_CACHE_DIR}"
echo "[INFO] Config          : ${CONFIG_PATH}"

# Configure local fast SSD scratch via SLURM_TMPDIR if available
LOCAL_SCRATCH_ARGS=()
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR}" ]; then
  SCRATCH_ROOT="${SLURM_TMPDIR}/${SLURM_JOB_ID:-training_dialogs_scratch}"
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

mkdir -p "${HF_CACHE_DIR}" "${HOME}/.local" "${WORKSPACE_ROOT}/local/adapters" "${WORKSPACE_ROOT}/local/models" "${WORKSPACE_ROOT}/logs"

# Ensure stale prepared dataset and adapter output directories are cleared
rm -rf "${WORKSPACE_ROOT}/last_run_prepared"
rm -rf "${WORKSPACE_ROOT}/local/adapters/gemma-4-26b-a4b-it-lora_dialogs"
rm -rf "${WORKSPACE_ROOT}/local/merged"

# ------------------------------------------------------------------------------
# STEP 1: Axolotl LoRA Fine-Tuning
# ------------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " [STEP 1/2] Launching Axolotl Distributed LoRA Training (Dialogs)"
echo "============================================================"
echo "[INFO] Running accelerate across ${NUM_GPUS} GPUs..."
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
  "${LOCAL_SCRATCH_ARGS[@]}" \
  --bind "${WORKSPACE_ROOT}:/repo" \
  --bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}" \
  --bind "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  --bind "${HOME}/.local:${HOME}/.local" \
  --pwd /repo \
  "${AXOLOTL_CONTAINER_DIR}" \
  /workspace/axolotl-venv/bin/accelerate launch \
    --multi_gpu \
    --num_machines=1 \
    --num_processes="${NUM_GPUS}" \
    --dynamo_backend=no \
    --mixed_precision=bf16 \
    /repo/src-train/train_patched.py "${CONFIG_PATH}"

echo "[SUCCESS] Step 1: Dialogs adapter fine-tuning completed successfully!"

# ------------------------------------------------------------------------------
# STEP 2: Merge LoRA Adapter & Compress to FP8-Dynamic
# ------------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " [STEP 2/2] Merging Adapter into Base Model & Quantizing to FP8 (Dialogs)"
echo "============================================================"
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
  --env PYTHONUSERBASE="${HOME}/.local" \
  --env PYTHONNOUSERSITE=0 \
  --env PYTHONPATH="${HOME}/.local/lib/python3.12/site-packages:${HOME}/.local/lib/python3.11/site-packages:${PYTHONPATH:-}" \
  "${LOCAL_SCRATCH_ARGS[@]}" \
  --bind "${WORKSPACE_ROOT}:/repo" \
  --bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}" \
  --bind "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  --bind "${HOME}/.local:${HOME}/.local" \
  --pwd /repo \
  "${AXOLOTL_CONTAINER_DIR}" \
  /workspace/axolotl-venv/bin/python /repo/src-train/merge_and_quantize_dialogs.py

echo ""
echo "============================================================"
echo "[SUCCESS] Dialogs Pipeline Complete! Fine-tuned FP8 model is ready at:"
echo "          ${WORKSPACE_ROOT}/local/models/gemma-4-26b-a4b-it-fp8_dialogs"
echo "============================================================"
