#!/usr/bin/env bash
set -euo pipefail

# Helper script to check the status of a Slurm job on the HSUper cluster.
# Can be called periodically without modifying any cluster directories.

JOB_ID="${1:-1332687}"
CLUSTER_HOST="wulfc@hsuper-login01.hsu-hh.de"

ssh -o ConnectTimeout=10 -o BatchMode=yes "${CLUSTER_HOST}" "
  SQUEUE_OUT=\$(squeue -h -j ${JOB_ID} -o '%T|%M|%N|%r' 2>/dev/null || true)
  if [ -n \"\${SQUEUE_OUT}\" ]; then
    STATE=\$(echo \"\${SQUEUE_OUT}\" | awk -F'|' '{print \$1}' | tr -d ' ')
    TIME=\$(echo \"\${SQUEUE_OUT}\" | awk -F'|' '{print \$2}' | tr -d ' ')
    NODE=\$(echo \"\${SQUEUE_OUT}\" | awk -F'|' '{print \$3}' | tr -d ' ')
    REASON=\$(echo \"\${SQUEUE_OUT}\" | awk -F'|' '{print \$4}' | tr -d ' ')
    echo \"JOB_ID: ${JOB_ID}\"
    echo \"STATE: \${STATE}\"
    echo \"TIME: \${TIME}\"
    echo \"NODE: \${NODE}\"
    echo \"REASON_OR_NODELIST: \${REASON}\"
    if [ \"\${STATE}\" = \"RUNNING\" ]; then
      echo '--- LOG SNIPPET (STDOUT) ---'
      tail -n 15 ~/begleit-app-training-gemma4-reasoning/logs/eval_${JOB_ID}.log 2>/dev/null || true
    fi
  else
    SACCT_OUT=\$(sacct -j ${JOB_ID} -X -n -o 'State%20,Elapsed,ExitCode' 2>/dev/null || true)
    echo \"JOB_ID: ${JOB_ID}\"
    echo \"STATE: FINISHED\"
    echo \"SACCT: \${SACCT_OUT}\"
    echo '--- LOG SNIPPET (STDOUT) ---'
    tail -n 25 ~/begleit-app-training-gemma4-reasoning/logs/eval_${JOB_ID}.log 2>/dev/null || true
  fi
"
