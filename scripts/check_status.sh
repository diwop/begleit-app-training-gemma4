#!/usr/bin/env bash
set -uo pipefail

echo "==================== [1. SLURM QUEUE] ===================="
squeue -u wulfc

echo "==================== [2. SACCT HISTORY] ===================="
sacct -u wulfc -X --format=JobID,JobName%30,State,ExitCode,Start,End,Elapsed

echo "==================== [3. JUDGE EVALUATION LOGS] ===================="
tail -n 35 /beegfs/home/w/wulfc/begleit-app-training-gemma4-judge/logs/*.log 2>/dev/null || echo "No logs found yet."

echo "==================== [4. JUDGE DATA FILES] ===================="
ls -lht /beegfs/home/w/wulfc/begleit-app-training-gemma4-judge/data/ 2>/dev/null || echo "No data directory found."

echo "==================== [5. ACTIVE SMALL_GPU8 JOBS] ===================="
squeue -p small_gpu8 -t RUNNING -o '%.10i %.10u %.2t %.10M %.10l %.6D %R'

echo "==================== [6. PENDING PRIORITY QUEUE] ===================="
squeue -p small_gpu8 -t PENDING -o '%i %u %l %b %R' | grep '(Priority)'

