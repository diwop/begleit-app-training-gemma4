#!/usr/bin/env bash
#SBATCH --job-name=gemma4-eval
#SBATCH --partition=small_gpu8
#SBATCH --gpus=2
#SBATCH --time=00:05:00
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j-err.log

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

# Execute vLLM evaluation helper script
bash scripts/run_evaluation.sh
