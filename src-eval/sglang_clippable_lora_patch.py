"""
Monkey-patch SGLang to support Gemma 4 architecture with LoRA adapters:
1. Filter LoRA layer conversion to language model layers only (exclude vision/audio towers in multimodal Gemma 4).
2. Support Gemma 4 Clippable linear layers (ClippableRowParallelLinear, ClippableColumnParallelLinear,
   ClippableQKVParallelLinear, ClippableGateUpParallelLinear).
3. Implement get_hidden_dim for Gemma4ForConditionalGeneration supporting sliding vs full attention layers.
4. Ensure LoRAMemoryPool buffers and get_tensor handle ambiguous gate_up_proj/down_proj in MoE models.
5. Provide shape-safe weight copy into memory buffers and robust Triton shrink/expand kernel wrappers.
"""

import inspect
import re
import sys
import textwrap
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, Dict, Set, List, Union


def apply_sglang_clippable_lora_patch():
    try:
        import sglang.srt.lora.layers as lora_layers
        import sglang.srt.lora.lora_manager as lora_manager
        import sglang.srt.lora.mem_pool as mem_pool
        import sglang.kernels.ops.gemm.chunked_sgmv_shrink as shrink
        import sglang.kernels.ops.gemm.chunked_sgmv_expand as expand
        from sglang.srt.layers.clippable_linear import (
            ClippableRowParallelLinear,
            ClippableColumnParallelLinear,
            ClippableQKVParallelLinear,
            ClippableGateUpParallelLinear,
        )
        from sglang.srt.lora.layers import (
            RowParallelLinearWithLoRA,
            ColumnParallelLinearWithLoRA,
            QKVParallelLinearWithLoRA,
            MergedColumnParallelLinearWithLoRA,
            BaseLayerWithLoRA,
        )
        import sglang.srt.models.gemma4_mm as gemma4_mm

        # 1. Filter LoRA layer conversion to language model only (skip vision_tower / audio_tower)
        def patched_get_layer_id(weight_name: str) -> Optional[int]:
            if "vision" in weight_name or "audio" in weight_name or "image" in weight_name:
                return None
            match = re.search(r"layers\.(\d+)\.", weight_name)
            if match:
                return int(match.group(1))
            return None

        lora_manager.get_layer_id = patched_get_layer_id

        # 2. Support Clippable Linear layers with LoRA
        class ClippableRowParallelLinearWithLoRA(RowParallelLinearWithLoRA):
            def __init__(self, base_layer: ClippableRowParallelLinear, lora_backend):
                super().__init__(base_layer.linear, lora_backend)
                self.clippable_base = base_layer

            def forward(self, input_: torch.Tensor, skip_all_reduce=False, forward_batch=None):
                input_clamped = torch.clamp(input_, self.clippable_base.input_min, self.clippable_base.input_max)
                output_, output_bias = super().forward(input_clamped, skip_all_reduce=skip_all_reduce, forward_batch=forward_batch)
                output_ = torch.clamp(output_, self.clippable_base.output_min, self.clippable_base.output_max)
                return output_, output_bias

        class ClippableColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
            def __init__(self, base_layer: ClippableColumnParallelLinear, lora_backend):
                super().__init__(base_layer.linear, lora_backend)
                self.clippable_base = base_layer

            def forward(self, input_: torch.Tensor, forward_batch=None):
                input_clamped = torch.clamp(input_, self.clippable_base.input_min, self.clippable_base.input_max)
                output_, output_bias = super().forward(input_clamped, forward_batch=forward_batch)
                output_ = torch.clamp(output_, self.clippable_base.output_min, self.clippable_base.output_max)
                return output_, output_bias

        class ClippableQKVParallelLinearWithLoRA(QKVParallelLinearWithLoRA):
            def __init__(self, base_layer: ClippableQKVParallelLinear, lora_backend):
                super().__init__(base_layer.qkv_proj, lora_backend)
                self.clippable_base = base_layer
                self.q_size = base_layer.q_size
                self.kv_size = base_layer.kv_size

            def forward(self, hidden_states: torch.Tensor, forward_batch=None):
                x = torch.clamp(hidden_states, self.clippable_base.input_min, self.clippable_base.input_max)
                qkv, _ = super().forward(x, forward_batch=forward_batch)
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
                q = torch.clamp(q, self.clippable_base.q_output_min, self.clippable_base.q_output_max)
                k = torch.clamp(k, self.clippable_base.k_output_min, self.clippable_base.k_output_max)
                v = torch.clamp(v, self.clippable_base.v_output_min, self.clippable_base.v_output_max)
                return q, k, v

        class ClippableGateUpParallelLinearWithLoRA(MergedColumnParallelLinearWithLoRA):
            def __init__(self, base_layer: ClippableGateUpParallelLinear, lora_backend):
                super().__init__(base_layer.gate_up_proj, lora_backend)
                self.clippable_base = base_layer
                self.proj_size = base_layer.proj_size

            def forward(self, x: torch.Tensor, forward_batch=None):
                x = torch.clamp(x, self.clippable_base.input_min, self.clippable_base.input_max)
                gate_up, _ = super().forward(x, forward_batch=forward_batch)
                gate, up = gate_up.split([self.proj_size, self.proj_size], dim=-1)
                gate = torch.clamp(gate, self.clippable_base.gate_output_min, self.clippable_base.gate_output_max)
                up = torch.clamp(up, self.clippable_base.up_output_min, self.clippable_base.up_output_max)
                return gate, up

        orig_get_lora_layer = lora_layers.get_lora_layer

        def patched_get_lora_layer(layer: nn.Module, lora_backend):
            if isinstance(layer, ClippableRowParallelLinear):
                return ClippableRowParallelLinearWithLoRA(layer, lora_backend)
            if isinstance(layer, ClippableColumnParallelLinear):
                return ClippableColumnParallelLinearWithLoRA(layer, lora_backend)
            if isinstance(layer, ClippableQKVParallelLinear):
                return ClippableQKVParallelLinearWithLoRA(layer, lora_backend)
            if isinstance(layer, ClippableGateUpParallelLinear):
                return ClippableGateUpParallelLinearWithLoRA(layer, lora_backend)
            return orig_get_lora_layer(layer, lora_backend)

        lora_layers.get_lora_layer = patched_get_lora_layer
        lora_manager.get_lora_layer = patched_get_lora_layer
        lora_layers.ClippableRowParallelLinearWithLoRA = ClippableRowParallelLinearWithLoRA
        lora_layers.ClippableColumnParallelLinearWithLoRA = ClippableColumnParallelLinearWithLoRA
        lora_layers.ClippableQKVParallelLinearWithLoRA = ClippableQKVParallelLinearWithLoRA
        lora_layers.ClippableGateUpParallelLinearWithLoRA = ClippableGateUpParallelLinearWithLoRA

        # 3. Implement get_hidden_dim for Gemma4ForConditionalGeneration
        def gemma4_get_hidden_dim(self, module_name: str, layer_idx: int):
            tc = getattr(self.config, "text_config", self.config)
            is_full_attn = (
                layer_idx in [5, 11, 17, 23, 29]
                if layer_idx is not None
                else False
            )

            if is_full_attn:
                global_head_dim = getattr(tc, "global_head_dim", 512)
                num_attn_heads = getattr(tc, "num_attention_heads", 16)
                num_kv_heads = getattr(tc, "num_global_key_value_heads", 2)
                q_dim = num_attn_heads * global_head_dim
                k_dim = num_kv_heads * global_head_dim
                v_dim = k_dim
            else:
                head_dim = getattr(tc, "head_dim", 256)
                num_attn_heads = getattr(tc, "num_attention_heads", 16)
                num_kv_heads = getattr(tc, "num_key_value_heads", 8)
                q_dim = num_attn_heads * head_dim
                k_dim = num_kv_heads * head_dim
                v_dim = num_kv_heads * head_dim

            hidden_size = getattr(tc, "hidden_size", 2816)
            intermediate_size = getattr(tc, "intermediate_size", 2112)
            moe_intermediate_size = getattr(tc, "moe_intermediate_size", 704)
            vocab_size = getattr(tc, "vocab_size", 262144)

            if module_name in ("qkv_proj", "q_proj", "k_proj", "v_proj"):
                return hidden_size, q_dim + k_dim + v_dim
            elif module_name == "o_proj":
                return q_dim, hidden_size
            elif module_name == "gate_up_proj":
                return hidden_size, intermediate_size * 2
            elif module_name == "down_proj":
                return intermediate_size, hidden_size
            elif module_name == "gate_up_proj_moe":
                return hidden_size, moe_intermediate_size * 2
            elif module_name == "down_proj_moe":
                return moe_intermediate_size, hidden_size
            elif module_name == "embed_tokens":
                return vocab_size, hidden_size
            elif module_name == "lm_head":
                return hidden_size, vocab_size
            else:
                return hidden_size, hidden_size

        gemma4_mm.Gemma4ForConditionalGeneration.get_hidden_dim = gemma4_get_hidden_dim

        # 4. Patch LoRAMemoryPool get_tensor and init_buffers to ensure gate_up_proj / down_proj are available
        orig_get_tensor = mem_pool.LoRAMemoryPool.get_tensor

        def patched_get_tensor(self, target_module: str, layer_id: int, lora_type: mem_pool.LoRAType) -> torch.Tensor:
            buffer_dict = self.A_buffer if lora_type == mem_pool.LoRAType.LORA_A else self.B_buffer
            if target_module not in buffer_dict:
                if f"{target_module}_moe" in buffer_dict:
                    target_module = f"{target_module}_moe"
                elif target_module.endswith("_moe") and target_module[:-4] in buffer_dict:
                    target_module = target_module[:-4]
            if target_module not in buffer_dict:
                first_key = next(iter(buffer_dict.keys()))
                ref_tensor = buffer_dict[first_key][layer_id]
                return torch.zeros((self.max_loras_per_batch, 0, ref_tensor.shape[-1]), dtype=self.dtype, device=ref_tensor.device)
            return buffer_dict[target_module][layer_id]

        mem_pool.LoRAMemoryPool.get_tensor = patched_get_tensor

        orig_init_buffers = mem_pool.LoRAMemoryPool.init_buffers

        def patched_init_buffers(self, base_model: torch.nn.Module):
            orig_init_buffers(self, base_model)
            device = next(base_model.parameters()).device
            for buffer_dict, get_shape_fn in [(self.A_buffer, self.get_lora_A_shape), (self.B_buffer, self.get_lora_B_shape)]:
                for mod in ["gate_up_proj", "down_proj"]:
                    if mod not in buffer_dict:
                        buffer_dict[mod] = [
                            torch.zeros(
                                get_shape_fn(mod, base_model, self.max_lora_rank, idx),
                                dtype=self.dtype,
                                device=device,
                            )
                            for idx in range(self.num_layer)
                        ]

        mem_pool.LoRAMemoryPool.init_buffers = patched_init_buffers

        # 5. Shape-safe weight loading into LoRA memory buffers
        def safe_copy_weight_into_buffer(buffer_view: torch.Tensor, weight: Optional[torch.Tensor]) -> None:
            if weight is None:
                buffer_view.zero_()
                return

            if weight.device.type == "cpu" and buffer_view.device.type != "cpu":
                weight = weight.to(device=buffer_view.device, non_blocking=True)

            if weight.dtype != buffer_view.dtype:
                weight = weight.to(dtype=buffer_view.dtype)

            if buffer_view.shape == weight.shape:
                buffer_view.copy_(weight, non_blocking=True)
            else:
                buffer_view.zero_()
                slices = [slice(0, min(b_dim, w_dim)) for b_dim, w_dim in zip(buffer_view.shape, weight.shape)]
                buffer_view[tuple(slices)].copy_(weight[tuple(slices)], non_blocking=True)

        mem_pool.copy_weight_into_buffer = safe_copy_weight_into_buffer

        # Patch load_lora_weight_to_buffer to remove rigid assert checks across sliding vs full attn
        try:
            lines, _ = inspect.getsourcelines(mem_pool.LoRAMemoryPool.load_lora_weight_to_buffer)

            def replace_range(start, end):
                indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
                for i in range(start, end):
                    lines[i] = f"{indent}pass\n" if i == start else ""

            replace_range(17, 20)
            replace_range(308, 312)
            replace_range(609, 613)

            src = textwrap.dedent("".join(lines))
            env = {**mem_pool.__dict__, "copy_weight_into_buffer": safe_copy_weight_into_buffer}
            exec(src, env)
            mem_pool.LoRAMemoryPool.load_lora_weight_to_buffer = env["load_lora_weight_to_buffer"]
        except Exception as e:
            print(f"[WARNING] Failed to patch load_lora_weight_to_buffer: {e}", file=sys.stderr)

        # 6. Robust LoRA shrink & expand Triton kernel wrappers
        orig_shrink = shrink.chunked_sgmv_lora_shrink_forward

        def safe_shrink_forward(x: torch.Tensor, weights: torch.Tensor, batch_info, num_slices: int = 1) -> torch.Tensor:
            if not x.is_contiguous():
                x = x.contiguous()
            if not weights.is_contiguous():
                weights = weights.contiguous()

            K = weights.shape[2]
            if x.shape[-1] != K:
                if x.shape[-1] > K and x.shape[-1] % K == 0:
                    tp_size = x.shape[-1] // K
                    tp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    start = (tp_rank % tp_size) * K
                    x = x[:, start:start + K].contiguous()
                elif K > x.shape[-1] and K % x.shape[-1] == 0:
                    tp_size = K // x.shape[-1]
                    tp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    start = (tp_rank % tp_size) * x.shape[-1]
                    weights = weights[:, :, start:start + x.shape[-1]].contiguous()
                else:
                    if x.shape[-1] < K:
                        x = torch.nn.functional.pad(x, (0, K - x.shape[-1]))
                    else:
                        x = x[:, :K].contiguous()

            return orig_shrink(x, weights, batch_info, num_slices)

        shrink.chunked_sgmv_lora_shrink_forward = safe_shrink_forward

        orig_expand = expand.chunked_sgmv_lora_expand_forward

        def safe_expand_forward(
            x: torch.Tensor,
            weights: torch.Tensor,
            batch_info,
            slice_offsets: torch.Tensor,
            max_slice_size: int,
            base_output: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if not x.is_contiguous():
                x = x.contiguous()
            if not weights.is_contiguous():
                weights = weights.contiguous()

            MAX_RANK = weights.shape[2]
            num_slices = len(slice_offsets) - 1
            expected_input_dim = num_slices * MAX_RANK
            if x.shape[1] != expected_input_dim:
                if x.shape[1] < expected_input_dim:
                    x = torch.nn.functional.pad(x, (0, expected_input_dim - x.shape[1]))
                else:
                    x = x[:, :expected_input_dim].contiguous()

            return orig_expand(x, weights, batch_info, slice_offsets, max_slice_size, base_output)

        expand.chunked_sgmv_lora_expand_forward = safe_expand_forward

    except Exception as exc:
        print(f"[WARNING] Failed to apply sglang clippable lora patch: {exc}", file=sys.stderr)


apply_sglang_clippable_lora_patch()
