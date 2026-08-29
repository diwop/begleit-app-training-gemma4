"""
Monkey-patch SGLang to support Gemma 4 clippable linear layers with LoRA adapters.
Gemma 4 uses ClippableRowParallelLinear, ClippableColumnParallelLinear, ClippableQKVParallelLinear,
and ClippableGateUpParallelLinear, which wrap underlying parallel linear layers with activation clamping.
"""

import sys
import torch
import torch.nn as nn
from typing import Optional, Tuple


def apply_sglang_clippable_lora_patch():
    try:
        import sglang.srt.lora.layers as lora_layers
        import sglang.srt.lora.lora_manager as lora_manager
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
        )

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
    except Exception as exc:
        pass


apply_sglang_clippable_lora_patch()
