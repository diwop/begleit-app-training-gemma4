#!/usr/bin/env bash
# ==============================================================================
# Helper script to pull and prepare both Axolotl and SGLang Apptainer containers on login node.
# Reads 'train_image' and 'eval_image' strictly from README.md YAML frontmatter.
# Builds Sandbox Directory containers (images/axolotl_sandbox and images/sglang_sandbox).
# NOTE: Run this command on the LOGIN node (hsuper-login01) which has internet access.
#
# Usage:
#   bash scripts/prepare_images.sh [all|train|eval|verify]
# ==============================================================================

set -euo pipefail

export PROOT_NO_SECCOMP=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure Apptainer temp & cache directories are inside workspace/user home
export APPTAINER_TMPDIR="${WORKSPACE_ROOT}/images/.tmp"
export APPTAINER_CACHEDIR="${WORKSPACE_ROOT}/images/.cache"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

README_FILE="${WORKSPACE_ROOT}/README.md"
if [ ! -f "${README_FILE}" ]; then
  echo "[ERROR] README.md file not found at: ${README_FILE}"
  exit 1
fi

# Extract images strictly from README.md frontmatter
TRAIN_IMAGE="$(sed -n '/^---$/,/^---$/p' "${README_FILE}" | grep '^train_image:' | head -n 1 | awk -F': ' '{print $2}' | tr -d ' "\r' | sed -e "s/^'//" -e "s/'$//")"
EVAL_IMAGE="$(sed -n '/^---$/,/^---$/p' "${README_FILE}" | grep '^eval_image:' | head -n 1 | awk -F': ' '{print $2}' | tr -d ' "\r' | sed -e "s/^'//" -e "s/'$//")"

if [ -z "${TRAIN_IMAGE}" ]; then
  echo "============================================================"
  echo "[ERROR] Failed to extract 'train_image' from YAML frontmatter in:"
  echo "        ${README_FILE}"
  echo "Please ensure the frontmatter contains: train_image: <image_name>"
  echo "============================================================"
  exit 1
fi

if [ -z "${EVAL_IMAGE}" ]; then
  echo "============================================================"
  echo "[ERROR] Failed to extract 'eval_image' from YAML frontmatter in:"
  echo "        ${README_FILE}"
  echo "Please ensure the frontmatter contains: eval_image: <image_name>"
  echo "============================================================"
  exit 1
fi

TARGET="${1:-all}"
OUTPUT_DIR="${WORKSPACE_ROOT}/images"
mkdir -p "${OUTPUT_DIR}"

build_sandbox() {
  local name="$1"
  local raw_uri="$2"
  local target_dir="${OUTPUT_DIR}/${name}_sandbox"
  local pure_image="${raw_uri#docker://}"
  local docker_uri="docker://${pure_image}"

  echo "[INFO] ============================================================"
  echo "[INFO] Preparing Apptainer Sandbox Container: ${name}"
  echo "[INFO] Source Image : ${docker_uri}"
  echo "[INFO] Target Dir   : ${target_dir}"
  echo "[INFO] ============================================================"

  if [ -d "${target_dir}" ]; then
    echo "[INFO] Removing previous ${name} sandbox build..."
    rm -rf "${target_dir}"
  fi

  apptainer build --sandbox "${target_dir}" "${docker_uri}"
  echo "[SUCCESS] ${name} sandbox container ready at: ${target_dir}"
}

verify_sglang_container() {
  local sandbox_dir="${OUTPUT_DIR}/sglang_sandbox"
  if [ ! -d "${sandbox_dir}" ]; then
    echo "[WARNING] Sandbox directory '${sandbox_dir}' not found."
    return
  fi
  echo "[INFO] Running verification test inside SGLang Apptainer container..."
  apptainer exec "${sandbox_dir}" /usr/bin/python3 -c "
import sglang as sgl
print('  * [VERIFIED] SGLang version:', sgl.__version__)
import torch
print('  * [VERIFIED] PyTorch version:', torch.__version__)
import transformers
print('  * [VERIFIED] Transformers version:', transformers.__version__)
print('  * [SUCCESS] SGLang container ready and verified!')
"
}

case "${TARGET}" in
  train)
    build_sandbox "axolotl" "${TRAIN_IMAGE}"
    ;;
  eval)
    build_sandbox "sglang" "${EVAL_IMAGE}"
    echo "[INFO] Installing 'textstat' and 'sentence-transformers' in SGLang environment on login node..."
    mkdir -p "${HOME}/.local"
    apptainer exec --bind "${HOME}/.local:${HOME}/.local" "${OUTPUT_DIR}/sglang_sandbox" /usr/bin/python3 -m pip install --user --no-cache-dir --break-system-packages textstat sentence-transformers
    verify_sglang_container
    ;;
  all)
    build_sandbox "axolotl" "${TRAIN_IMAGE}"
    build_sandbox "sglang" "${EVAL_IMAGE}"
    echo "[INFO] Installing 'textstat' and 'sentence-transformers' in SGLang environment on login node..."
    mkdir -p "${HOME}/.local"
    apptainer exec --bind "${HOME}/.local:${HOME}/.local" "${OUTPUT_DIR}/sglang_sandbox" /usr/bin/python3 -m pip install --user --no-cache-dir --break-system-packages textstat sentence-transformers
    verify_sglang_container
    ;;
  verify)
    verify_sglang_container
    ;;
  *)
    echo "[ERROR] Unknown target: '${TARGET}'. Usage: $0 [all|train|eval|verify]"
    exit 1
    ;;
esac

# Cleanup temp build dirs
rm -rf "${APPTAINER_TMPDIR}"

echo "============================================================"
echo "[SUCCESS] All requested Apptainer sandbox containers ready in:"
echo "          ${OUTPUT_DIR}"
echo "Now you can run training and evaluation scripts on GPU nodes offline."
echo "============================================================"
