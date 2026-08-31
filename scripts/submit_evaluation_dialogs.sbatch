#!/usr/bin/env bash
#SBATCH --job-name=gemma4-eval-dialogs
#SBATCH --partition=small_gpu8
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_dialogs_%j.log
#SBATCH --error=logs/eval_dialogs_%j-err.log

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

# Execute SGLang dialogs evaluation helper script
bash scripts/run_evaluation_dialogs.sh
