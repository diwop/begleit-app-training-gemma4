#!/usr/bin/env bash
# ==============================================================================
# Helper script to pull dataset files from AWS S3 via DVC inside Apptainer.
# Run this on the LOGIN node (hsuper-login01) where internet access is available.
#
# Usage:
#   bash scripts/pull_data.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AXOLOTL_SANDBOX="${WORKSPACE_ROOT}/images/axolotl_sandbox"
DVC_VENV="${WORKSPACE_ROOT}/.dvc-venv"

echo "============================================================"
echo " Pulling Dataset from AWS S3 Remote via DVC"
echo " (Run this on hsuper-login01 where internet is available)"
echo "============================================================"
echo "[INFO] Workspace : ${WORKSPACE_ROOT}"
echo "============================================================"

if [ ! -d "${AXOLOTL_SANDBOX}" ]; then
  echo "[ERROR] Axolotl container '${AXOLOTL_SANDBOX}' not found."
  echo "[INFO] Run 'bash scripts/prepare_images.sh' on the login node first."
  exit 1
fi

# 1. Initialize workspace .dvc-venv if not already present
if [ ! -f "${DVC_VENV}/bin/dvc" ]; then
  echo "[INFO] Initializing workspace DVC virtual environment (.dvc-venv)..."
  apptainer exec \
    --bind "${WORKSPACE_ROOT}:/repo" \
    --pwd /repo \
    "${AXOLOTL_SANDBOX}" \
    uv venv /repo/.dvc-venv

  echo "[INFO] Installing 'dvc[s3]' into workspace virtual environment..."
  apptainer exec \
    --bind "${WORKSPACE_ROOT}:/repo" \
    --pwd /repo \
    "${AXOLOTL_SANDBOX}" \
    uv pip install --python /repo/.dvc-venv "dvc[s3]"
fi

# 2. Execute dvc pull inside Apptainer
echo "[INFO] Running 'dvc pull'..."
apptainer exec \
  --bind "${WORKSPACE_ROOT}:/repo" \
  --bind "${HOME}/.aws:${HOME}/.aws" \
  --pwd /repo \
  "${AXOLOTL_SANDBOX}" \
  /repo/.dvc-venv/bin/dvc pull

echo "============================================================"
echo "[SUCCESS] DVC dataset successfully pulled to shared filesystem!"
echo "============================================================"
