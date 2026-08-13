---
train_image: axolotlai/axolotl:0.18.0
eval_image: vllm/vllm-openai:v0.27.1
---

# BegleitApp Training (Gemma 4 on HSUper)

This is repository contains code to fine-tune and evaluate adapters for
[Google Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it)
on the [HSUper](https://www.hsu-hh.de/hpc/en/hsuper/) GPU infrastructure.  
[Axolotl](https://axolotl.ai) us used to train the adapter for the Mixture of
Experts model in a way that [vLLM](https://vllm.ai) is capable of providing
inference on the base model with the tuned adapter.

Training and evaluation is desined for **NVIDIA L40S GPUs** as provided by HSUper.

## Details

See [docs/implementation-details.md](docs/implementation-details.md) for details.

See [adrs](adrs/) for Architecture Decision Records.