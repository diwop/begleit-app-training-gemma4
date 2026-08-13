#!/usr/bin/env bash
# ==============================================================================
# Helper script to run the Axolotl smoke test inside Apptainer on HSUper cluster.
# Usage:
#   bash scripts/run_smoke_test.sh [SANDBOX_PATH]
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_SANDBOX="${WORKSPACE_ROOT}/images/axolotl_sandbox"
IMAGE="${1:-${AXOLOTL_IMAGE:-${DEFAULT_SANDBOX}}}"

if [ ! -d "${IMAGE}" ]; then
  echo "============================================================"
  echo "[ERROR] Container sandbox directory not found at:"
  echo "        ${IMAGE}"
  echo ""
  echo "GPU compute nodes do not have internet access to pull Docker images directly."
  echo "Please run the following command on the LOGIN node (hsuper-login01) first:"
  echo ""
  echo "    bash scripts/prepare_image.sh"
  echo "============================================================"
  exit 1
fi

echo "[INFO] Running Axolotl GPU Smoke Test Spike"
echo "[INFO] Workspace: ${WORKSPACE_ROOT}"
echo "[INFO] Apptainer Container: ${IMAGE}"

# Execute Apptainer with NVIDIA GPU pass-through (--nv).
# Bind project root to /repo (instead of /workspace) to preserve container's /workspace/axolotl-venv
apptainer exec --nv \
  --bind "${WORKSPACE_ROOT}:/repo" \
  --pwd /repo \
  "${IMAGE}" \
  /workspace/axolotl-venv/bin/python /repo/src-train/smoke_test.py
