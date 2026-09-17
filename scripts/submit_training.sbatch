#!/usr/bin/env bash
#SBATCH --job-name=gemma4-train-fp8
#SBATCH --partition=small_gpu8
#SBATCH --nodes=1-1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j-err.log

# Navigate to submission directory where sbatch was executed
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

# Ensure output directory for logs exists
mkdir -p logs

echo "=== Slurm Job Details ==="
echo "Job ID       : ${SLURM_JOB_ID:-N/A}"
echo "Node Name    : ${SLURM_JOB_NODELIST:-N/A}"
echo "Partition    : ${SLURM_JOB_PARTITION:-N/A}"
echo "Submit Dir   : ${SLURM_SUBMIT_DIR:-$PWD}"
echo "========================="

# Execute combined training, merge, and FP8 quantization pipeline
bash scripts/run_training.sh
