#!/usr/bin/env bash
# ==============================================================================
# Helper script to pull and prepare both Axolotl and vLLM Apptainer containers on login node.
# Reads 'train_image' and 'eval_image' strictly from README.md YAML frontmatter.
# Builds Sandbox Directory containers (images/axolotl_sandbox and images/vllm_sandbox).
# NOTE: Run this command on the LOGIN node (hsuper-login01) which has internet access.
#
# Usage:
#   bash scripts/prepare_images.sh [all|train|eval]
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

patch_vllm_container() {
  local sandbox_dir="${OUTPUT_DIR}/vllm_sandbox"
  if [ ! -d "${sandbox_dir}" ]; then
    echo "[WARNING] Sandbox directory '${sandbox_dir}' not found. Run with 'eval' or 'all' to build it first."
    return
  fi
  echo "[INFO] Applying Gemma 4 compatibility hotfixes directly to vLLM sandbox on login node..."

  local vllm_dir
  vllm_dir=$(find "${sandbox_dir}/usr" -type d -path "*/dist-packages/vllm" 2>/dev/null | head -n 1)

  if [ -z "${vllm_dir}" ]; then
    echo "[WARNING] Could not locate vLLM package inside ${sandbox_dir}"
    return
  fi

  echo "[INFO] Found vLLM installation at: ${vllm_dir}"

  # 1. Patch weight_utils.py (1D slice for [512] vs [256])
  local weight_utils="${vllm_dir}/model_executor/model_loader/weight_utils.py"
  if [ -f "${weight_utils}" ]; then
    python3 -c "
import re
with open('${weight_utils}', 'r', encoding='utf-8') as f:
    code = f.read()

new_default_weight_loader = '''def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    try:
        if param.size() == loaded_weight.size():
            param.data.copy_(loaded_weight)
            return
        if param.numel() <= loaded_weight.numel():
            param.data.copy_(loaded_weight.flatten()[:param.numel()].reshape(param.shape))
            return
    except Exception:
        pass
    assert param.size() == loaded_weight.size(), (
        f\"Attempted to load weight ({loaded_weight.size()}) into parameter ({param.size()})\"
    )
'''

pattern = r'def default_weight_loader\s*\([\s\S]*?(?=\n(?:def |class |\Z))'
code = re.sub(pattern, new_default_weight_loader.strip(), code)

# Verify syntax before saving
compile(code, '${weight_utils}', 'exec')
with open('${weight_utils}', 'w', encoding='utf-8') as f:
    f.write(code)
print('  -> [SUCCESS] Validated and patched weight_utils.py on disk')
"
  fi

  # 2. Patch parameter.py (safe narrow for QKV sliding window vs global layers)
  local param_file="${vllm_dir}/model_executor/parameter.py"
  if [ -f "${param_file}" ]; then
    python3 -c "
with open('${param_file}', 'r', encoding='utf-8') as f:
    code = f.read()

target = 'loaded_weight = loaded_weight.narrow(dim, start, length)'
replacement = 'loaded_weight = loaded_weight.narrow(dim, start, min(length, max(0, loaded_weight.size(dim) - start)))'

if target in code:
    code = code.replace(target, replacement, 1)

compile(code, '${param_file}', 'exec')
with open('${param_file}', 'w', encoding='utf-8') as f:
    f.write(code)
print('  -> [SUCCESS] Validated and patched parameter.py on disk')
"
  fi

  # 3. Patch mla_attention.py (Issue #43263)
  local mla_file="${vllm_dir}/model_executor/layers/attention/mla_attention.py"
  if [ -f "${mla_file}" ]; then
    python3 -c "
with open('${mla_file}', 'r', encoding='utf-8') as f:
    code = f.read()

broken = 'kv_c_normed = kv_c_normed.to(self.kv_b_proj.weight.dtype)'
fixed = 'kv_c_normed = kv_c_normed.to(_kv_b_proj_w_dtype)'

if broken in code:
    code = code.replace(broken, fixed, 1)
    compile(code, '${mla_file}', 'exec')
    with open('${mla_file}', 'w', encoding='utf-8') as f:
        f.write(code)
    print('  -> [SUCCESS] Validated and patched mla_attention.py on disk')
"
  fi
}

case "${TARGET}" in
  train)
    build_sandbox "axolotl" "${TRAIN_IMAGE}"
    ;;
  eval)
    build_sandbox "vllm" "${EVAL_IMAGE}"
    echo "[INFO] Installing 'textstat' in vLLM environment on login node..."
    mkdir -p "${HOME}/.local"
    apptainer exec --bind "${HOME}/.local:${HOME}/.local" "${OUTPUT_DIR}/vllm_sandbox" /usr/bin/python3 -m pip install --user --no-cache-dir textstat
    patch_vllm_container
    ;;
  all)
    build_sandbox "axolotl" "${TRAIN_IMAGE}"
    build_sandbox "vllm" "${EVAL_IMAGE}"
    echo "[INFO] Installing 'textstat' in vLLM environment on login node..."
    mkdir -p "${HOME}/.local"
    apptainer exec --bind "${HOME}/.local:${HOME}/.local" "${OUTPUT_DIR}/vllm_sandbox" /usr/bin/python3 -m pip install --user --no-cache-dir textstat
    patch_vllm_container
    ;;
  patch)
    patch_vllm_container
    ;;
  *)
    echo "[ERROR] Unknown target: '${TARGET}'. Usage: $0 [all|train|eval|patch]"
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
