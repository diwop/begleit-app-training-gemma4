#!/usr/bin/env python3
"""
Custom Axolotl Training Entrypoint with Runtime Monkeypatches for Gemma 4 & DeepSpeed ZeRO-3.

Axolotl's config schema automatically forces `use_reentrant=False` for Gemma 4 in distributed runs.
However, under DeepSpeed ZeRO-3, freed parameter buffers produce empty tensor shapes (Size([0]))
that trigger PyTorch's `check_recomputed_tensors_match` CheckpointError during the backward pass.
Overriding `use_reentrant=True` at runtime allows ZeRO-3 dynamic parameter gathering during backward.
"""

import sys
import fire
import axolotl.train
import axolotl.cli.train

original_train = axolotl.train.train


def patched_train(cfg, *args, **kwargs):
    print("\n" + "=" * 60)
    print("🔧 MONKEYPATCH: Overriding gradient_checkpointing_kwargs to use_reentrant=True")
    print("This bypasses the DeepSpeed ZeRO-3 parameter sharding metadata mismatch (Size([0]) vs Size([N])).")
    print("=" * 60 + "\n", flush=True)

    if hasattr(cfg, "gradient_checkpointing_kwargs") and cfg.gradient_checkpointing_kwargs:
        cfg.gradient_checkpointing_kwargs["use_reentrant"] = True
    else:
        cfg.gradient_checkpointing_kwargs = {"use_reentrant": True}

    return original_train(cfg, *args, **kwargs)


# Apply monkeypatch globally across Axolotl training hooks
axolotl.train.train = patched_train
if hasattr(axolotl.cli.train, "train"):
    axolotl.cli.train.train = patched_train

if __name__ == "__main__":
    fire.Fire(axolotl.cli.train.do_cli)
