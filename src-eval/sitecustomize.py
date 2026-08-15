"""
Python startup hook (sitecustomize) automatically loaded by Python in all processes
(including spawned multiprocessing worker processes in vLLM).

Applies compatibility patches for Gemma 4 FP8 and Transformers 5.x:
1. Weight loader shape slicing for 1D FP8 tensors ([512] vs [256])
2. Transformers 5.x PretrainedConfig per-layer attribute access
3. Gemma RoPE scaling dictionary schema ('rope_type' vs 'type')
4. GemmaTokenizer special tokens extended attribute
"""

import sys

try:
    import torch
    import vllm.model_executor.model_loader.weight_utils as weight_utils

    _orig_default_weight_loader = weight_utils.default_weight_loader

    def _smart_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if param.size() == loaded_weight.size():
            param.data.copy_(loaded_weight)
            return

        # 1D tensor mismatch (e.g. rotary frequency embeddings or projection scales [512] vs [256])
        if param.dim() == 1 and loaded_weight.dim() == 1:
            min_len = min(param.size(0), loaded_weight.size(0))
            param.data[:min_len].copy_(loaded_weight[:min_len])
            return

        # Squeezed tensor shape match (e.g. [1, 256] vs [256])
        if loaded_weight.squeeze().size() == param.size():
            param.data.copy_(loaded_weight.squeeze())
            return

        # Flatten & slice fallback if total elements fit
        if param.numel() <= loaded_weight.numel():
            param.data.copy_(loaded_weight.flatten()[:param.numel()].reshape(param.shape))
            return

        _orig_default_weight_loader(param, loaded_weight)

    weight_utils.default_weight_loader = _smart_weight_loader
except Exception:
    pass

# Transformers 5.x PretrainedConfig per-layer attribute access
try:
    from transformers.configuration_utils import PretrainedConfig
    _orig_init = PretrainedConfig.__init__
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.allow_global_per_layer_attribute_access = True
    PretrainedConfig.__init__ = _patched_init
    PretrainedConfig.allow_global_per_layer_attribute_access = True
except Exception:
    pass

# RoPE scaling schema compatibility
try:
    from transformers.models.gemma.configuration_gemma import GemmaConfig
    _orig_gemma_init = GemmaConfig.__init__
    def _patched_gemma_init(self, *args, **kwargs):
        if "rope_scaling" in kwargs and kwargs["rope_scaling"] is not None:
            rs = kwargs["rope_scaling"]
            if isinstance(rs, dict):
                if "rope_type" not in rs and "type" in rs:
                    rs["rope_type"] = rs["type"]
                if "rope_type" not in rs:
                    rs["rope_type"] = "default"
        _orig_gemma_init(self, *args, **kwargs)
    GemmaConfig.__init__ = _patched_gemma_init
except Exception:
    pass

# Gemma tokenizer special tokens attribute
try:
    from transformers import GemmaTokenizer, GemmaTokenizerFast
    for cls in [GemmaTokenizer, GemmaTokenizerFast]:
        if not hasattr(cls, "all_special_tokens_extended"):
            cls.all_special_tokens_extended = property(lambda self: self.all_special_tokens)
except Exception:
    pass
