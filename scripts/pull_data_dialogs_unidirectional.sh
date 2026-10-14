#!/usr/bin/env bash
# ==============================================================================
# Helper script to pull unidirectional dialogs dataset files from AWS S3 via DVC inside Apptainer.
# Run this on the LOGIN node (hsuper-login01) where internet access is available.
#
# Usage:
#   bash scripts/pull_data_dialogs_unidirectional.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AXOLOTL_SANDBOX="${WORKSPACE_ROOT}/images/axolotl_sandbox"
DVC_VENV="${WORKSPACE_ROOT}/.dvc-venv"

echo "============================================================"
echo " Pulling Unidirectional Dialogs Dataset from AWS S3 Remote via DVC"
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
echo "[INFO] Running 'dvc pull' for unidirectional dialogs..."
TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
  if [ -f "${WORKSPACE_ROOT}/data/raw_dialogs.dvc" ]; then
    TARGETS=("data/raw_dialogs.dvc" "prepare_data_dialogs_unidirectional" "prepare_few_shots_dialogs_unidirectional")
  else
    TARGETS=("prepare_data_dialogs_unidirectional" "prepare_few_shots_dialogs_unidirectional")
  fi
fi

if ! apptainer exec \
  --bind "${WORKSPACE_ROOT}:/repo" \
  --bind "${HOME}/.aws:${HOME}/.aws" \
  --pwd /repo \
  "${AXOLOTL_SANDBOX}" \
  /repo/.dvc-venv/bin/dvc pull --force "${TARGETS[@]}"; then
  echo "[INFO] Some unidirectional stage outputs were not in S3 cache; pulling raw_dialogs and reproducing stages locally..."
  apptainer exec \
    --bind "${WORKSPACE_ROOT}:/repo" \
    --bind "${HOME}/.aws:${HOME}/.aws" \
    --pwd /repo \
    "${AXOLOTL_SANDBOX}" \
    /repo/.dvc-venv/bin/dvc pull --force data/raw_dialogs.dvc
  apptainer exec \
    --bind "${WORKSPACE_ROOT}:/repo" \
    --pwd /repo \
    "${AXOLOTL_SANDBOX}" \
    /repo/.dvc-venv/bin/dvc repro prepare_data_dialogs_unidirectional prepare_few_shots_dialogs_unidirectional
fi

echo "============================================================"
echo "[SUCCESS] Unidirectional Dialogs DVC dataset successfully pulled to shared filesystem!"
echo "============================================================"
