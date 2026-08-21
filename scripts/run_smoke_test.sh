#!/usr/bin/env bash
# ==============================================================================
# Helper script to run Axolotl (Train) and SGLang (Eval) smoke tests in sequence inside Apptainer.
# Usage:
#   bash scripts/run_smoke_test.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

AXOLOTL_SANDBOX="${WORKSPACE_ROOT}/images/axolotl_sandbox"
SGLANG_SANDBOX="${WORKSPACE_ROOT}/images/sglang_sandbox"

MISSING=0
if [ ! -d "${AXOLOTL_SANDBOX}" ]; then
  echo "[ERROR] Axolotl container sandbox directory not found at: ${AXOLOTL_SANDBOX}"
  MISSING=1
fi

if [ ! -d "${SGLANG_SANDBOX}" ]; then
  echo "[ERROR] SGLang container sandbox directory not found at: ${SGLANG_SANDBOX}"
  MISSING=1
fi

if [ "${MISSING}" -ne 0 ]; then
  echo "============================================================"
  echo "GPU compute nodes do not have internet access to pull Docker images directly."
  echo "Please run the following command on the LOGIN node (hsuper-login01) first:"
  echo ""
  echo "    bash scripts/prepare_images.sh"
  echo "============================================================"
  exit 1
fi

echo "============================================================"
echo " Starting Full Training & Evaluation Cluster Smoke Test"
echo "============================================================"

# Step 1: Run Axolotl Training Smoke Test
echo ""
echo "[STEP 1/2] Executing Axolotl Training Container Smoke Test..."
echo "[INFO] Container: ${AXOLOTL_SANDBOX}"
apptainer exec --nv \
  --bind "${WORKSPACE_ROOT}:/repo" \
  --pwd /repo \
  "${AXOLOTL_SANDBOX}" \
  /workspace/axolotl-venv/bin/python /repo/src-train/smoke_test.py

# Step 2: Run SGLang Evaluation Smoke Test
echo ""
echo "[STEP 2/2] Executing SGLang Evaluation Container Smoke Test..."
echo "[INFO] Container: ${SGLANG_SANDBOX}"
apptainer exec --nv \
  --bind "${WORKSPACE_ROOT}:/repo" \
  --pwd /repo \
  "${SGLANG_SANDBOX}" \
  /usr/bin/python3 /repo/src-eval/smoke_test.py

echo ""
echo "============================================================"
echo "[SUCCESS] Both Training (Axolotl) and Evaluation (SGLang) smoke tests completed!"
echo "============================================================"
