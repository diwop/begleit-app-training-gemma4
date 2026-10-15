#!/usr/bin/env bash
set -euo pipefail

# Helper script to check the status of a Slurm job on the HSUper cluster.
# Can be called periodically without modifying any cluster directories.

JOB_ID="${1:-1383307}"
CLUSTER_HOST="wulfc@hsuper-login01.hsu-hh.de"

ssh -o ConnectTimeout=10 -o BatchMode=yes "${CLUSTER_HOST}" "
  SQUEUE_OUT=\$(squeue -h -j ${JOB_ID} -o '%T|%M|%L|%N|%r' 2>/dev/null || true)
  if [ -n \"\${SQUEUE_OUT}\" ]; then
    STATE=\$(echo \"\${SQUEUE_OUT}\" | awk -F'|' '{print \$1}' | tr -d ' ')
    TIME=\$(echo \"\${SQUEUE_OUT}\" | awk -F'|' '{print \$2}' | tr -d ' ')
    TIME_LEFT=\$(echo \"\${SQUEUE_OUT}\" | awk -F'|' '{print \$3}' | tr -d ' ')
    NODE=\$(echo \"\${SQUEUE_OUT}\" | awk -F'|' '{print \$4}' | tr -d ' ')
    REASON=\$(echo \"\${SQUEUE_OUT}\" | awk -F'|' '{print \$5}' | tr -d ' ')
    echo \"JOB_ID: ${JOB_ID}\"
    echo \"STATE: \${STATE}\"
    echo \"TIME_ELAPSED: \${TIME}\"
    echo \"TIME_LEFT: \${TIME_LEFT}\"
    echo \"NODE: \${NODE}\"
    echo \"REASON_OR_NODELIST: \${REASON}\"
    if [ \"\${STATE}\" = \"RUNNING\" ]; then
      echo '--- LOG SNIPPET (STDOUT) ---'
      tail -n 20 ~/begleit-app-training-gemma4-unidirectional/logs/*_${JOB_ID}.log ~/begleit-app-training-gemma4/logs/*_${JOB_ID}.log ~/begleit-app-training-gemma4-reasoning/logs/*_${JOB_ID}.log 2>/dev/null || true
    fi
  else
    SACCT_OUT=\$(sacct -j ${JOB_ID} -X -n -o 'State%20,Elapsed,ExitCode' 2>/dev/null || true)
    echo \"JOB_ID: ${JOB_ID}\"
    echo \"STATE: FINISHED\"
    echo \"SACCT: \${SACCT_OUT}\"
    echo '--- LOG SNIPPET (STDOUT) ---'
    tail -n 30 ~/begleit-app-training-gemma4-unidirectional/logs/*_${JOB_ID}.log ~/begleit-app-training-gemma4/logs/*_${JOB_ID}.log ~/begleit-app-training-gemma4-reasoning/logs/*_${JOB_ID}.log 2>/dev/null || true
  fi
"

