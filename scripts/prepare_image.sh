#!/usr/bin/env bash
# ==============================================================================
# Helper script to pull and prepare the Axolotl Apptainer container on the login node.
# Builds a Sandbox Directory container (images/axolotl_sandbox) to bypass mksquashfs/proot limits.
# NOTE: Run this command on the LOGIN node (hsuper-login01) which has internet access.
#
# Usage:
#   bash scripts/prepare_image.sh [DOCKER_IMAGE_TAG]
# ==============================================================================

set -euo pipefail

export PROOT_NO_SECCOMP=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure Apptainer temp & cache directories are inside workspace/user home
export APPTAINER_TMPDIR="${WORKSPACE_ROOT}/images/.tmp"
export APPTAINER_CACHEDIR="${WORKSPACE_ROOT}/images/.cache"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

RAW_IMAGE="${1:-${AXOLOTL_DOCKER_IMAGE:-docker://axolotlai/axolotl:0.18.0}}"

PURE_IMAGE="${RAW_IMAGE#docker://}"
DOCKER_URI="docker://${PURE_IMAGE}"

OUTPUT_DIR="${WORKSPACE_ROOT}/images"
SANDBOX_DIR="${OUTPUT_DIR}/axolotl_sandbox"
mkdir -p "${OUTPUT_DIR}"

echo "[INFO] Preparing Apptainer Sandbox Container (bypasses mksquashfs)..."
echo "[INFO] Source Image : ${DOCKER_URI}"
echo "[INFO] Target Dir   : ${SANDBOX_DIR}"

# Remove existing sandbox dir if present to allow fresh build
if [ -d "${SANDBOX_DIR}" ]; then
  echo "[INFO] Removing previous sandbox build..."
  rm -rf "${SANDBOX_DIR}"
fi

# Build directory sandbox container (does NOT use mksquashfs, bypassing PRoot restriction)
apptainer build --sandbox "${SANDBOX_DIR}" "${DOCKER_URI}"

# Cleanup temp build dirs
rm -rf "${APPTAINER_TMPDIR}"

echo "============================================================"
echo "[SUCCESS] Apptainer sandbox container ready at:"
echo "          ${SANDBOX_DIR}"
echo "Now you can run scripts on GPU nodes offline."
echo "============================================================"
