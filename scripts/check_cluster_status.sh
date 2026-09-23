#!/usr/bin/env bash
# ==============================================================================
# check_cluster_status.sh
# Checks Slurm job status on the HSUper cluster once per invocation.
# Detects VPC/VPN disconnection and exits with code 2 if unreachable.
# ==============================================================================

set -uo pipefail

CLUSTER_HOST="wulfc@hsuper-login01.hsu-hh.de"
SSH_OPTS="-o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

# 1. Connectivity / VPC / VPN check
echo "=== [1. Checking Connection to Cluster / VPC] ==="
if ! ssh ${SSH_OPTS} "${CLUSTER_HOST}" "true" 2>/dev/null; then
  echo "[ERROR] VPC_DISCONNECTED: Cannot reach cluster login node (${CLUSTER_HOST})."
  echo "Please verify that the VPN / VPC connection is active."
  exit 2
fi
echo "[OK] Connection to cluster established."

# 2. Remote Slurm status queries
echo ""
echo "=== [2. Active Slurm Jobs for User: wulfc] ==="
ssh ${SSH_OPTS} "${CLUSTER_HOST}" "squeue -u wulfc -o '%.10i %.18j %.10u %.2t %.10M %.10l %.6D %R'"

echo ""
echo "=== [3. Recent Slurm Job History (sacct)] ==="
ssh ${SSH_OPTS} "${CLUSTER_HOST}" "sacct -u wulfc -X --format=JobID,JobName%25,State,Elapsed,Start,End -S \$(date +%Y-%m-%d)"

echo ""
echo "=== [4. Recent Log Activity (Dialogs Training / Eval)] ==="
ssh ${SSH_OPTS} "${CLUSTER_HOST}" "
  cd ~/begleit-app-training-gemma4 2>/dev/null || exit 0
  LATEST_TRAIN=\$(ls -t logs/train_dialogs_*.log 2>/dev/null | head -n 1)
  LATEST_EVAL=\$(ls -t logs/eval_dialogs_*.log 2>/dev/null | head -n 1)

  if [ -n \"\${LATEST_TRAIN}\" ]; then
    echo \"--- Latest Training Log (\${LATEST_TRAIN}) ---\"
    tail -n 15 \"\${LATEST_TRAIN}\"
  fi

  if [ -n \"\${LATEST_EVAL}\" ]; then
    echo \"--- Latest Evaluation Log (\${LATEST_EVAL}) ---\"
    tail -n 15 \"\${LATEST_EVAL}\"
  fi
"

echo ""
echo "=== [5. Cluster Partition Load (small_gpu8)] ==="
ssh ${SSH_OPTS} "${CLUSTER_HOST}" "
  echo -n 'Running jobs on small_gpu8: '
  squeue -p small_gpu8 -t RUNNING -h | wc -l
  echo -n 'Pending jobs on small_gpu8: '
  squeue -p small_gpu8 -t PENDING -h | wc -l
"

echo ""
echo "=== Status Check Complete ==="
