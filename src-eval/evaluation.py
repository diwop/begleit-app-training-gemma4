#!/usr/bin/env python3
"""
Runs baseline evaluation for RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic on data/dataset_eval.jsonl.
Evaluates the model twice:
1. Standard generation (without thinking: chat_template_kwargs={"enable_thinking": False})
2. Thinking-enabled generation (with thinking: chat_template_kwargs={"enable_thinking": True})

Outputs results to data/results.jsonl:
{
    "id": "<id>",
    "system": "<system-prompt>",
    "user_input": "<raw input text without template wrapper>",
    "assistant": "<ground-truth Leichte_Sprache>",
    "assistant_gemma4": "<gemma4 output without thinking>",
    "assistant_gemma4_thinking": "<gemma4 output with thinking>"
}
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import re
import sys

# Enforce offline mode on cluster compute nodes (no internet access)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("[ERROR] vLLM is not installed in the current environment.", file=sys.stderr)
    print("[INFO] Please run this script inside the vLLM container (images/vllm_sandbox).", file=sys.stderr)
    sys.exit(1)

MODEL_NAME = "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
EVAL_DATA_PATH = Path("data/dataset_eval.jsonl")
RESULTS_OUTPUT_PATH = Path("data/results.jsonl")


def resolve_model_path(model_name: str) -> str:
    """Resolve model repo ID to local disk snapshot directory to bypass network lookups."""
    # 1. Direct path check
    if Path(model_name).exists():
        print(f"[INFO] Using direct path: {model_name}")
        return model_name

    repo_folder = "models--" + model_name.replace("/", "--")
    model_short_name = model_name.split("/")[-1]

    # Candidate Hugging Face cache directories
    candidate_bases = []
    if "HF_HOME" in os.environ:
        candidate_bases.append(Path(os.environ["HF_HOME"]) / "hub")
        candidate_bases.append(Path(os.environ["HF_HOME"]))
    
    candidate_bases.extend([
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface",
        Path("/root/.cache/huggingface/hub"),
        Path("/root/.cache/huggingface"),
    ])

    # Check parent directory of home (in case running under another username/mount)
    try:
        if Path.home().parent.exists():
            for user_home in Path.home().parent.glob("*"):
                candidate_bases.append(user_home / ".cache" / "huggingface" / "hub")
    except Exception:
        pass

    print(f"[INFO] Searching local disk for snapshot of '{model_name}'...")

    for base in candidate_bases:
        if not base or not base.exists():
            continue

        # Look for exact repo folder
        target_dir = base / repo_folder
        if not target_dir.exists():
            # Try subfolder if base wasn't hub/
            target_dir = base / "hub" / repo_folder

        if target_dir.exists():
            snapshots_dir = target_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = sorted(
                    [p for p in snapshots_dir.iterdir() if p.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if snapshots:
                    resolved = str(snapshots[0])
                    print(f"[SUCCESS] Resolved local model snapshot: {resolved}")
                    return resolved

        # Fallback: glob search in base
        try:
            for match in base.glob(f"*{model_short_name}*"):
                snapshots_dir = match / "snapshots"
                if snapshots_dir.exists():
                    snapshots = sorted(
                        [p for p in snapshots_dir.iterdir() if p.is_dir()],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if snapshots:
                        resolved = str(snapshots[0])
                        print(f"[SUCCESS] Resolved local model snapshot via search: {resolved}")
                        return resolved
        except Exception:
            pass

    print(f"[WARNING] Local snapshot directory not found. Passing '{model_name}' directly.")
    return model_name


def extract_user_input(user_prompt: str) -> str:
    """Extract raw text from inside the ```input ... ``` block of the user prompt."""
    match = re.search(r"```input\n(.*?)\n```", user_prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return user_prompt.strip()


def load_eval_data(path: Path) -> list[dict[str, str]]:
    """Load evaluation samples from JSONL."""
    if not path.exists():
        print(f"[ERROR] Evaluation dataset not found at {path}", file=sys.stderr)
        print("[INFO] Run 'dvc repro' or 'python3 src-train/prepare_data.py' first.", file=sys.stderr)
        sys.exit(1)

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def main() -> None:
    print("=" * 60)
    print("      Gemma 4 Baseline Evaluation (vLLM)")
    print("=" * 60)
    print(f"[INFO] Model            : {MODEL_NAME}")
    print(f"[INFO] Input Dataset    : {EVAL_DATA_PATH}")
    print(f"[INFO] Output Results   : {RESULTS_OUTPUT_PATH}")

    gpu_count = torch.cuda.device_count()
    tensor_parallel_size = max(1, gpu_count)
    print(f"[INFO] Detected GPUs    : {gpu_count} (Tensor Parallel Size: {tensor_parallel_size})")

    records = load_eval_data(EVAL_DATA_PATH)
    print(f"[INFO] Loaded {len(records)} evaluation samples.")

    # Resolve local offline snapshot path on disk
    model_path = resolve_model_path(MODEL_NAME)

    # Initialize vLLM model
    print(f"\n[INFO] Loading model into vLLM engine from: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
    )

    # Standard chat conversations for both passes
    conversations = [
        [
            {"role": "system", "content": rec["system"]},
            {"role": "user", "content": rec["user"]},
        ]
        for rec in records
    ]

    # Sampling parameters
    sampling_params_no_thinking = SamplingParams(
        temperature=0.0,
        max_tokens=4096,
    )

    sampling_params_thinking = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
    )

    print("\n[STEP 1/2] Running inference WITHOUT thinking (enable_thinking=False)...")
    no_thinking_outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling_params_no_thinking,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=True,
    )

    print("\n[STEP 2/2] Running inference WITH thinking (enable_thinking=True)...")
    thinking_outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling_params_thinking,
        chat_template_kwargs={"enable_thinking": True},
        use_tqdm=True,
    )

    print(f"\n[INFO] Assembling and writing results to {RESULTS_OUTPUT_PATH}...")
    RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, rec in enumerate(records):
        out_no_thinking = no_thinking_outputs[idx].outputs[0].text.strip()
        out_thinking = thinking_outputs[idx].outputs[0].text.strip()
        raw_user_input = extract_user_input(rec["user"])

        result_entry = {
            "id": rec["id"],
            "system": rec["system"],
            "user_input": raw_user_input,
            "assistant": rec["assistant"],
            "assistant_gemma4": out_no_thinking,
            "assistant_gemma4_thinking": out_thinking,
        }
        results.append(result_entry)

    with RESULTS_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Wrote {len(results)} evaluated results to {RESULTS_OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
