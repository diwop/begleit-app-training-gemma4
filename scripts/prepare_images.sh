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

  # 2. Write canonical Gemma 4 patched parameter.py
  local param_file="${vllm_dir}/model_executor/parameter.py"
  if [ -f "${param_file}" ]; then
    python3 -c "
code = '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Hashable
from fractions import Fraction
from typing import Any
from weakref import WeakValueDictionary

import torch
from torch.nn import Parameter

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger

__all__ = [
    \"BasevLLMParameter\",
    \"PackedvLLMParameter\",
    \"PerTensorScaleParameter\",
    \"ModelWeightParameter\",
    \"ChannelQuantScaleParameter\",
    \"GroupQuantScaleParameter\",
    \"BlockQuantScaleParameter\",
    \"PackedColumnParameter\",
    \"RowvLLMParameter\",
]

logger = init_logger(__name__)


class BasevLLMParameter(Parameter):
    def __new__(cls, data: torch.Tensor | None, **kwargs):
        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(self, data: torch.Tensor, weight_loader: Callable):
        from vllm.platforms import current_platform

        if current_platform.use_sync_weight_loader():
            weight_loader = current_platform.make_synced_weight_loader(weight_loader)

        self._weight_loader = weight_loader
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

    @property
    def weight_loader(self) -> Callable:
        if self._weight_loader is None:
            raise AttributeError(
                f\"{self.__class__.__name__} weight_loader attribute has been deleted\"
            )
        return self._weight_loader

    @weight_loader.setter
    def weight_loader(self, value: Callable):
        self._weight_loader = value

    @weight_loader.deleter
    def weight_loader(self):
        self._weight_loader = None

    def _is_1d_and_scalar(self, loaded_weight: torch.Tensor):
        cond1 = self.data.ndim == 1 and self.data.numel() == 1
        cond2 = loaded_weight.ndim == 0 and loaded_weight.numel() == 1
        return cond1 and cond2

    def _assert_and_load(self, loaded_weight: torch.Tensor):
        if self.data.shape != loaded_weight.shape and not self._is_1d_and_scalar(loaded_weight):
            if self.data.numel() <= loaded_weight.numel():
                loaded_weight = loaded_weight.flatten()[:self.data.numel()].reshape(self.data.shape)
        assert self.data.shape == loaded_weight.shape or self._is_1d_and_scalar(loaded_weight)
        self.data.copy_(loaded_weight)

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        self._assert_and_load(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        self._assert_and_load(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._assert_and_load(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._assert_and_load(loaded_weight)


class _ColumnvLLMParameter(BasevLLMParameter):
    def __init__(self, output_dim: int, **kwargs):
        self._output_dim = output_dim
        super().__init__(**kwargs)

    @property
    def output_dim(self):
        return self._output_dim

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        shard_size = self.data.shape[self.output_dim]
        max_len = max(0, loaded_weight.size(self.output_dim) - (self.tp_rank * shard_size))
        safe_size = min(shard_size, max_len)
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, safe_size
        )
        if self.data.shape != loaded_weight.shape and self.data.numel() <= loaded_weight.numel():
            loaded_weight = loaded_weight.flatten()[:self.data.numel()].reshape(self.data.shape)
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_offset: int = kwargs[\"shard_offset\"]
        shard_size: int = kwargs[\"shard_size\"]

        if (
            isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
            and self.packed_dim == self.output_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = self.data
        max_p_len = max(0, param_data.size(self.output_dim) - shard_offset)
        safe_p_size = min(shard_size, max_p_len)
        param_data = param_data.narrow(self.output_dim, shard_offset, safe_p_size)

        max_w_len = max(0, loaded_weight.size(self.output_dim) - (self.tp_rank * shard_size))
        safe_w_size = min(shard_size, max_w_len)
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, safe_w_size
        )

        if param_data.shape != loaded_weight.shape and param_data.numel() <= loaded_weight.numel():
            loaded_weight = loaded_weight.flatten()[:param_data.numel()].reshape(param_data.shape)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_offset: int = kwargs[\"shard_offset\"]
        shard_size: int = kwargs[\"shard_size\"]
        shard_id: str = kwargs[\"shard_id\"]
        num_heads: int = kwargs[\"num_heads\"]

        if (
            isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
            and self.output_dim == self.packed_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = self.data
        shard_id_int = self.tp_rank if shard_id == \"q\" else self.tp_rank // num_heads

        max_p_len = max(0, param_data.size(self.output_dim) - shard_offset)
        safe_p_size = min(shard_size, max_p_len)
        param_data = param_data.narrow(self.output_dim, shard_offset, safe_p_size)

        max_w_len = max(0, loaded_weight.size(self.output_dim) - (shard_id_int * shard_size))
        safe_w_size = min(shard_size, max_w_len)
        loaded_weight = loaded_weight.narrow(
            self.output_dim, shard_id_int * shard_size, safe_w_size
        )

        if param_data.shape != loaded_weight.shape and param_data.numel() <= loaded_weight.numel():
            loaded_weight = loaded_weight.flatten()[:param_data.numel()].reshape(param_data.shape)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class RowvLLMParameter(BasevLLMParameter):
    def __init__(self, input_dim: int, **kwargs):
        self._input_dim = input_dim
        super().__init__(**kwargs)

    @property
    def input_dim(self):
        return self._input_dim

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        shard_size = self.data.shape[self.input_dim]
        max_w_len = max(0, loaded_weight.size(self.input_dim) - (self.tp_rank * shard_size))
        safe_w_size = min(shard_size, max_w_len)
        loaded_weight = loaded_weight.narrow(
            self.input_dim, self.tp_rank * shard_size, safe_w_size
        )

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        if self.data.shape != loaded_weight.shape and self.data.numel() <= loaded_weight.numel():
            loaded_weight = loaded_weight.flatten()[:self.data.numel()].reshape(self.data.shape)

        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)


class ModelWeightParameter(_ColumnvLLMParameter, RowvLLMParameter):
    pass


class GroupQuantScaleParameter(_ColumnvLLMParameter, RowvLLMParameter):
    pass


class BlockQuantScaleParameter(_ColumnvLLMParameter, RowvLLMParameter):
    pass


class ChannelQuantScaleParameter(_ColumnvLLMParameter):
    pass


class PerTensorScaleParameter(BasevLLMParameter):
    pass


class PackedColumnParameter(_ColumnvLLMParameter):
    def __init__(self, packed_dim: int, packed_factor: int, **kwargs):
        self.packed_dim = packed_dim
        self.packed_factor = packed_factor
        super().__init__(**kwargs)

    def adjust_shard_indexes_for_packing(self, shard_size: int, shard_offset: int):
        return _adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=self.packed_factor,
            marlin_tile_size=getattr(self, \"marlin_tile_size\", None),
        )


class PackedvLLMParameter(BasevLLMParameter):
    def __init__(self, packed_dim: int, packed_factor: int, **kwargs):
        self.packed_dim = packed_dim
        self.packed_factor = packed_factor
        super().__init__(**kwargs)

    def adjust_shard_indexes_for_packing(self, shard_size: int, shard_offset: int):
        return _adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=self.packed_factor,
            marlin_tile_size=getattr(self, \"marlin_tile_size\", None),
        )


def _adjust_shard_indexes_for_marlin(shard_size, shard_offset, marlin_tile_size):
    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


def _adjust_shard_indexes_for_packing(
    shard_size, shard_offset, packed_factor, marlin_tile_size
):
    shard_size = round(shard_size // packed_factor)
    shard_offset = round(shard_offset // packed_factor)
    if marlin_tile_size is not None:
        return _adjust_shard_indexes_for_marlin(
            shard_size=shard_size,
            shard_offset=shard_offset,
            marlin_tile_size=marlin_tile_size,
        )
    return shard_size, shard_offset


class SharedWeightParameter(BasevLLMParameter):
    def __init__(self, partitions: dict[str | int, BasevLLMParameter], **kwargs):
        self.partitions = partitions
        self.kwargs = kwargs
        super().__init__(data=None, weight_loader=self._fake_weight_loader)

    def _shard_id_as_int(self, shard_id: str | int | None) -> int:
        if shard_id is None:
            return 0
        if isinstance(shard_id, int):
            return shard_id
        return int(shard_id)

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor, **kwargs):
        partition_id = self._shard_id_as_int(kwargs.pop(\"shard_id\", None))
        partition = self.partitions[partition_id]
        partition.load_column_parallel_weight(loaded_weight, **kwargs)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor, **kwargs):
        partition_id = self._shard_id_as_int(kwargs.pop(\"shard_id\", None))
        partition = self.partitions[partition_id]
        partition.load_row_parallel_weight(loaded_weight, **kwargs)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        partition_id = self._shard_id_as_int(kwargs.pop(\"shard_id\", None))
        partition = self.partitions[partition_id]
        partition.load_merged_column_weight(loaded_weight, **kwargs)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        partition_id = self._shard_id_as_int(kwargs.pop(\"shard_id\", None))
        partition = self.partitions[partition_id]

        input_dim = self.kwargs.get(\"input_dim\")
        shard_size = partition.data.size(input_dim) // self.tp_size if input_dim is not None else partition.data.size(0) // self.tp_size
        shard_offset = self.tp_rank * shard_size
        shard_id = \"q\"
        num_heads = kwargs.get(\"num_heads\")

        ModelWeightParameter.load_qkv_weight(
            partition,
            loaded_weight,
            shard_offset=shard_offset,
            shard_size=shard_size,
            shard_id=shard_id,
            num_heads=num_heads,
        )

    def process_weights_after_loading(self):
        for key in self.partitions:
            self.partitions[key] = torch.nn.Parameter(
                data=self.partitions[key].data, requires_grad=False
            )

    @property
    def data(self):
        raise ValueError(
            \"Accessing data of a SharedWeightParameter is not allowed.\"
        )

    def get_partition(self, partition_id: str | int) -> torch.Tensor:
        return self.partitions[partition_id].data

    def _fake_weight_loader(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_weight_shard_id: str | int | None,
    ):
        raise ValueError(
            \"When loading partition weights of SharedWeightParameter, use methods provided by SharedWeightParameter\"
        )


def permute_param_layout_(
    param: BasevLLMParameter, input_dim: int, output_dim: int, **kwargs
) -> BasevLLMParameter:
    curr_input_dim = getattr(param, \"input_dim\", None)
    curr_output_dim = getattr(param, \"output_dim\", None)

    if curr_input_dim is None or curr_output_dim is None:
        assert param.data.dim() == 2, (
            \"permute_param_layout_ only supports 2D parameters when either \"
            \"input_dim or output_dim is not set\"
        )

    if curr_input_dim is None:
        assert curr_output_dim is not None, \"either input or output dim must be set\"
        curr_input_dim = (curr_output_dim + 1) % 2
    if curr_output_dim is None:
        assert curr_input_dim is not None, \"either input or output dim must be set\"
        curr_output_dim = (curr_input_dim + 1) % 2

    perm = [
        i for i in range(param.data.dim()) if i not in [curr_input_dim, curr_output_dim]
    ]
    perm.insert(input_dim, curr_input_dim)
    perm.insert(output_dim, curr_output_dim)

    if \"packed_dim\" in kwargs:
        assert (
            hasattr(param, \"packed_dim\")
            and param.packed_dim == perm[kwargs[\"packed_dim\"]]
        ), \"permute_param_layout_ currently doesn't support repacking\"

    param.data = param.data.permute(*perm)
    if hasattr(param, \"_input_dim\"):
        param._input_dim = input_dim
    if hasattr(param, \"_output_dim\"):
        param._output_dim = output_dim
    if \"packed_dim\" in kwargs and hasattr(param, \"_packed_dim\"):
        param._packed_dim = kwargs[\"packed_dim\"]

    return param
'''

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
    with open('${mla_file}', 'w', encoding='utf-8') as f:
        f.write(code)
    print('  -> [SUCCESS] Patched mla_attention.py on disk')
"
  fi

  # 4. Automated Post-Patch Container Validation Test
  echo "[INFO] Running verification test inside Apptainer container..."
  apptainer exec "${sandbox_dir}" /usr/bin/python3 -c "
import vllm
print('  * [VERIFIED] vLLM version:', vllm.__version__)
from vllm.model_executor.parameter import BasevLLMParameter, BlockQuantScaleParameter, ModelWeightParameter, RowvLLMParameter
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
print('  * [SUCCESS] All critical vLLM layers and parameters imported cleanly without error!')
"
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
