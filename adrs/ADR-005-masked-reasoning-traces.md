# ADR-005 Masked Empty Reasoning Traces for SFT

Training datasets without explicit reasoning traces use Axolotl's `input_output` format to prepend empty reasoning delimiter tokens (`<|channel>thought\n<channel|>\n` for Gemma 4) to the model turn, masked with `label: false` (`labels = -100`). Supervised cross-entropy loss is computed exclusively on the target response (`label: true`).

## Rationale

Fine-tuning reasoning models (such as `google/gemma-4-26b-a4b-it`) on domain-specific datasets that contain only direct target translations (no-think data, such as Leichte Sprache text pairs) induces **Reasoning-Trace Collapse**. As demonstrated in [arXiv:2605.21127v1](https://arxiv.org/html/2605.21127v1) (*Reasoning-Trace Collapse: Evaluating the Loss of Explicit Reasoning During Fine-Tuning*, Twist et al.), standard SFT penalizes the opening of reasoning tags because the loss gradient forces immediate generation of the response text, collapsing the Valid Reasoning Rate (VR) to 0%.

arXiv:2605.21127v1 establishes that prepending an empty reasoning trace and masking both the prompt and the empty thought delimiters ("Response-only" masking) aligns the training and inference contexts:
1. The model does not suffer a penalty for opening reasoning delimiters during inference when thinking is enabled (`enable_thinking=True`).
2. The model is never rewarded for outputting empty reasoning blocks, because the delimiters are masked out of the loss computation.
3. Explicit reasoning capability is preserved even after domain adaptation.

Axolotl natively supports segment-level loss masking via `type: input_output` combined with `train_on_inputs: false`. In `prepare_data.py`, samples are formatted into two segments:
- Segment 0 (`label: false`): Prompt tokens plus empty thought channel delimiters (`<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n<|channel>thought\n<channel|>\n`).
- Segment 1 (`label: true`): Target translation response followed by `<turn|>\n`.
