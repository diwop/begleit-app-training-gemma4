#!/usr/bin/env bash
#SBATCH --job-name=judge-llama4
#SBATCH --partition=small_gpu8
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=logs/judge_%j.log
#SBATCH --error=logs/judge_%j-err.log

# Navigate to submission directory where sbatch was executed
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

# Ensure output directory for logs exists
mkdir -p logs

echo "=== Slurm Job Details ==="
echo "Job ID              : ${SLURM_JOB_ID}"
echo "Job Name            : ${SLURM_JOB_NAME}"
echo "Submit Directory    : ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Allocated Nodes     : ${SLURM_JOB_NODELIST}"
echo "Allocated GPUs      : ${SLURM_GPUS:-4}"
echo "Allocated CPUs      : ${SLURM_CPUS_PER_TASK:-32}"
echo "Allocated Memory    : 240G"
echo "========================="

# Execute the judge evaluation runner script
bash scripts/run_judge_evaluation.sh
