# ADR-002 vLLM for inference

vLLM is used for inference of trained LoRA.

## Rationale

Inference is hard for fine-tuned adapters of modern MoE models.
Sglang and vLLM have been considered.
vLLM succeeded in inference of an adapter fine-tuned using Axolotl.