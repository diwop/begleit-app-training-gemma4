#!/usr/bin/env python3
"""
Merges a trained LoRA adapter into the base bfloat16 Gemma 4 model and compresses
the resulting merged checkpoint to FP8-Dynamic using llmcompressor.

The exported FP8 model can be loaded natively and efficiently by SGLang or vLLM for inference.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch

# Enforce offline mode on compute nodes
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["AXOLOTL_DO_NOT_TRACK"] = "1"


def get_model_snapshot_path(model_name: str) -> str:
    """Resolve model snapshot directory from Hugging Face cache or local path."""
    if Path(model_name).exists():
        return model_name

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_folder = "models--" + model_name.replace("/", "--")
    snapshots_dir = hf_home / "hub" / repo_folder / "snapshots"

    if not snapshots_dir.exists():
        print(f"[ERROR] Hugging Face cache directory not found at: {snapshots_dir}", file=sys.stderr)
        print("[INFO] Please run 'bash scripts/download_model.sh' on the login node first.", file=sys.stderr)
        sys.exit(1)

    snapshots = sorted(
        [p for p in snapshots_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not snapshots:
        print(f"[ERROR] No snapshot directories found inside: {snapshots_dir}", file=sys.stderr)
        sys.exit(1)

    resolved_path = str(snapshots[0])
    print(f"[INFO] Resolved local model snapshot: {resolved_path}")
    return resolved_path


def merge_via_axolotl(config_path: str, adapter_dir: str, output_dir: str) -> bool:
    """Try merging using Axolotl's native CLI merge_lora for Gemma 4."""
    print(f"[INFO] Attempting merge via Axolotl CLI (axolotl.cli.merge_lora)...")
    env = os.environ.copy()
    env.setdefault("MASTER_ADDR", "localhost")
    env.setdefault("MASTER_PORT", "12345")
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("RANK", "0")
    env.setdefault("LOCAL_RANK", "0")

    cmd = [
        sys.executable, "-m", "axolotl.cli.merge_lora",
        config_path,
        f"--lora_model_dir={adapter_dir}",
        f"--output_dir={output_dir}",
    ]
    try:
        subprocess.run(cmd, check=True, env=env)
        # Check if output or nested merged/ has config.json
        nested = Path(output_dir) / "merged"
        if nested.exists() and (nested / "config.json").exists():
            for item in nested.iterdir():
                shutil.move(str(item), str(output_dir))
            shutil.rmtree(str(nested), ignore_errors=True)
        if (Path(output_dir) / "config.json").exists():
            print(f"[SUCCESS] Axolotl merge succeeded at: {output_dir}")
            return True
    except Exception as e:
        print(f"[WARNING] Axolotl merge failed or unavailable: {e}. Falling back to PEFT merge.", file=sys.stderr)
    return False


def merge_via_peft(base_model_snapshot: str, adapter_path: Path, merged_bf16_path: Path) -> None:
    """Fallback merge using transformers and PEFT merge_and_unload."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("\n[STEP 1/3] Loading base model in bfloat16 via Transformers...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_snapshot,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    print(f"[INFO] Base model loaded in {time.time() - t0:.1f}s")

    print("[STEP 1/3] Loading and merging LoRA adapter weights...")
    t1 = time.time()
    peft_model = PeftModel.from_pretrained(model, str(adapter_path))
    merged_model = peft_model.merge_and_unload()
    print(f"[INFO] Adapter merged in {time.time() - t1:.1f}s")

    print(f"[STEP 1/3] Saving merged bfloat16 checkpoint to: {merged_bf16_path}")
    merged_model.save_pretrained(str(merged_bf16_path))
    tokenizer = AutoTokenizer.from_pretrained(base_model_snapshot, trust_remote_code=True, local_files_only=True)
    tokenizer.save_pretrained(str(merged_bf16_path))

    del peft_model, model, merged_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into Gemma 4 base model and compress to FP8"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src-train/config.yml",
        help="Path to training config.yml",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-4-26b-a4b-it",
        help="Base model ID or local directory path",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="./local/adapters/gemma-4-26b-a4b-it-lora",
        help="Path to trained LoRA adapter directory",
    )
    parser.add_argument(
        "--merged-bf16-dir",
        type=str,
        default="",
        help="Path to temporary merged BF16 model (defaults to $SLURM_TMPDIR or ./local/merged)",
    )
    parser.add_argument(
        "--output-fp8-dir",
        type=str,
        default="./local/models/gemma-4-26b-a4b-it-fp8",
        help="Output directory for compressed FP8 model",
    )
    parser.add_argument(
        "--keep-merged-bf16",
        action="store_true",
        help="Do not delete intermediate merged BF16 checkpoint",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter_dir)
    if not adapter_path.exists():
        print(f"[ERROR] Adapter directory not found at: {adapter_path}", file=sys.stderr)
        print("[INFO] Please complete adapter fine-tuning before running merge & quantization.", file=sys.stderr)
        sys.exit(1)

    output_fp8_path = Path(args.output_fp8_dir)
    output_fp8_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine temporary location for merged BF16 weights
    if args.merged_bf16_dir:
        merged_bf16_path = Path(args.merged_bf16_dir)
    else:
        slurm_tmp = os.environ.get("SLURM_TMPDIR", "")
        if slurm_tmp and Path(slurm_tmp).is_dir():
            merged_bf16_path = Path(slurm_tmp) / "merged-gemma4-bf16"
        else:
            merged_bf16_path = Path("./local/merged/gemma-4-26b-a4b-it-bf16")
    merged_bf16_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("      Gemma 4 Adapter Merge & FP8 Quantization")
    print("=" * 60)
    print(f"[INFO] Base Model       : {args.base_model}")
    print(f"[INFO] Adapter Directory: {adapter_path}")
    print(f"[INFO] Merged BF16 Path : {merged_bf16_path}")
    print(f"[INFO] Output FP8 Path  : {output_fp8_path}")
    print("=" * 60)

    # Step 1: Perform BF16 Merge (Axolotl primary -> PEFT fallback)
    base_model_snapshot = get_model_snapshot_path(args.base_model)
    merged_ok = merge_via_axolotl(args.config, str(adapter_path), str(merged_bf16_path))
    if not merged_ok:
        merge_via_peft(base_model_snapshot, adapter_path, merged_bf16_path)

    # Step 2: Quantize merged model into FP8-Dynamic
    try:
        from transformers import AutoTokenizer
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.transformers import SparseMLForCausalLM
    except ImportError as e:
        print(f"[ERROR] Required package for FP8 compression missing: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[STEP 2/3] Compressing merged model to FP8-Dynamic with llmcompressor...")
    t2 = time.time()
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC")
    SparseMLForCausalLM.compress(
        model_path=str(merged_bf16_path),
        recipe=recipe,
        output_dir=str(output_fp8_path),
    )
    tokenizer = AutoTokenizer.from_pretrained(str(merged_bf16_path), trust_remote_code=True, local_files_only=True)
    tokenizer.save_pretrained(str(output_fp8_path))
    print(f"[SUCCESS] FP8 compression completed in {time.time() - t2:.1f}s")

    # Step 3: Cleanup intermediate unquantized BF16 model if requested
    if not args.keep_merged_bf16 and merged_bf16_path.exists():
        print(f"\n[STEP 3/3] Cleaning up temporary merged BF16 checkpoint at: {merged_bf16_path}")
        shutil.rmtree(merged_bf16_path, ignore_errors=True)
    else:
        print(f"\n[STEP 3/3] Preserved merged BF16 checkpoint at: {merged_bf16_path}")

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Merged & Compressed FP8 model is ready at: {output_fp8_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
