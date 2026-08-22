#!/usr/bin/env python3
"""
Prepares training and evaluation JSONL datasets from raw text pairs and prompt templates.
Filters out outlier documents exceeding MAX_SEQUENCE_LENGTH (8192 tokens).
Fixed 10% reproducible split (seed 42) for DVC pipeline.
Calculates token distributions and prints a token statistics summary.
"""

from __future__ import annotations

import os
import json
import random
from pathlib import Path
import sys

# Single source of truth for max sequence/token length
MAX_SEQUENCE_LENGTH = 8192

RAW_DIR = Path("data/raw")
SYSTEM_PROMPT_PATH = Path("prompts/system-prompt.md")
PROMPT_TEMPLATE_PATH = Path("prompts/prompt-template.md")
TRAIN_OUTPUT = Path("data/dataset_train.jsonl")
EVAL_OUTPUT = Path("data/dataset_eval.jsonl")
EVAL_RATIO = 0.10
SEED = 42


def get_tokenizer():
    """Attempt to load Hugging Face tokenizer if available offline or in environment."""
    try:
        from transformers import AutoTokenizer
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        # Check local snapshots in cache
        for snap_dir in sorted(hf_home.glob("hub/models--*gemma*/snapshots/*"), reverse=True):
            if (snap_dir / "tokenizer.json").exists() or (snap_dir / "tokenizer_config.json").exists():
                return AutoTokenizer.from_pretrained(str(snap_dir), local_files_only=True)
        return AutoTokenizer.from_pretrained("google/gemma-2-27b", local_files_only=True)
    except Exception:
        pass
    return None


def count_tokens(text: str, tokenizer=None) -> int:
    """Calculate token count using tokenizer or German BPE approximation."""
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # German subword heuristic: German compounds average ~1.35 tokens per whitespace word
    words = len(text.split())
    chars = len(text)
    return max(1, int(round(max(chars / 3.8, words * 1.35))))


def compute_distribution(lengths: list[int]) -> dict[str, float | int]:
    """Calculate descriptive statistics for a list of token counts."""
    if not lengths:
        return {}
    s = sorted(lengths)
    n = len(s)
    return {
        "count": n,
        "min": s[0],
        "p25": s[int(n * 0.25)],
        "median": s[int(n * 0.50)],
        "mean": round(sum(s) / n, 1),
        "p75": s[int(n * 0.75)],
        "p90": s[int(n * 0.90)],
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)],
        "max": s[-1],
    }


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    print("=" * 60)
    print("      Dataset Preparation & Token Length Filtering")
    print("=" * 60)
    print(f"[INFO] Max Sequence Length Limit : {MAX_SEQUENCE_LENGTH} tokens")

    if not SYSTEM_PROMPT_PATH.exists():
        print(f"[ERROR] System prompt file not found at: {SYSTEM_PROMPT_PATH}", file=sys.stderr)
        sys.exit(1)

    if not PROMPT_TEMPLATE_PATH.exists():
        print(f"[ERROR] Prompt template file not found at: {PROMPT_TEMPLATE_PATH}", file=sys.stderr)
        sys.exit(1)

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()

    std_files = {f.name.split("_")[0]: f for f in RAW_DIR.glob("*_Standardsprache.txt")}
    ls_files = {f.name.split("_")[0]: f for f in RAW_DIR.glob("*_Leichte_Sprache.txt")}

    common_ids = sorted(std_files.keys() & ls_files.keys())
    print(f"[INFO] Found {len(common_ids)} raw document pairs in {RAW_DIR}")

    tokenizer = get_tokenizer()
    if tokenizer is not None:
        print(f"[INFO] Using tokenizer: {tokenizer.__class__.__name__}")
    else:
        print("[INFO] Using calibrated German subword token estimation (~1.35 tokens/word)")

    system_token_count = count_tokens(system_prompt, tokenizer)
    print(f"[INFO] System Prompt Tokens: {system_token_count}")

    records = []
    filtered_outliers = []
    user_tokens = []
    assistant_tokens = []
    total_tokens = []

    for doc_id in common_ids:
        std_text = std_files[doc_id].read_text(encoding="utf-8").strip()
        ls_text = ls_files[doc_id].read_text(encoding="utf-8").strip()
        user_prompt = template.replace("%INPUT%", std_text)

        u_tok = count_tokens(user_prompt, tokenizer)
        a_tok = count_tokens(ls_text, tokenizer)
        prompt_total = system_token_count + u_tok
        tot_tok = prompt_total + a_tok

        # Filter out documents where prompt or total sequence exceeds MAX_SEQUENCE_LENGTH
        if prompt_total > MAX_SEQUENCE_LENGTH or tot_tok > MAX_SEQUENCE_LENGTH:
            filtered_outliers.append({
                "id": doc_id,
                "prompt_tokens": prompt_total,
                "total_tokens": tot_tok,
            })
            continue

        user_tokens.append(u_tok)
        assistant_tokens.append(a_tok)
        total_tokens.append(tot_tok)

        records.append({
            "id": doc_id,
            "system": system_prompt,
            "user": user_prompt,
            "assistant": ls_text,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": ls_text},
            ],
        })

    print(f"[INFO] Kept {len(records)} pairs (filtered {len(filtered_outliers)} outliers exceeding {MAX_SEQUENCE_LENGTH} tokens):")
    for out in filtered_outliers:
        print(f"       - Excluded ID {out['id']}: {out['prompt_tokens']} prompt tokens / {out['total_tokens']} total tokens")

    # Deterministic shuffle & split (seed 42)
    rng = random.Random(SEED)
    shuffled = list(records)
    rng.shuffle(shuffled)

    eval_count = int(round(len(shuffled) * EVAL_RATIO))
    eval_records = sorted(shuffled[:eval_count], key=lambda r: r["id"])
    train_records = sorted(shuffled[eval_count:], key=lambda r: r["id"])

    write_jsonl(TRAIN_OUTPUT, train_records)
    write_jsonl(EVAL_OUTPUT, eval_records)

    dist_user = compute_distribution(user_tokens)
    dist_assistant = compute_distribution(assistant_tokens)
    dist_total = compute_distribution(total_tokens)

    print(f"\n[SUCCESS] Wrote {len(train_records)} train samples to {TRAIN_OUTPUT}")
    print(f"[SUCCESS] Wrote {len(eval_records)} eval samples to {EVAL_OUTPUT}")

    # Print clean summary table to console
    print("\n" + "=" * 60)
    print("      Retained Dataset Token Distribution Summary")
    print("=" * 60)
    print(f"{'Metric':<12} | {'User Prompt':<12} | {'Assistant (LS)':<14} | {'Total Sequence':<14}")
    print("-" * 60)
    print(f"{'Min':<12} | {dist_user['min']:<12} | {dist_assistant['min']:<14} | {dist_total['min']:<14}")
    print(f"{'Median (P50)':<12} | {dist_user['median']:<12} | {dist_assistant['median']:<14} | {dist_total['median']:<14}")
    print(f"{'Mean':<12} | {dist_user['mean']:<12} | {dist_assistant['mean']:<14} | {dist_total['mean']:<14}")
    print(f"{'P90':<12} | {dist_user['p90']:<12} | {dist_assistant['p90']:<14} | {dist_total['p90']:<14}")
    print(f"{'P95':<12} | {dist_user['p95']:<12} | {dist_assistant['p95']:<14} | {dist_total['p95']:<14}")
    print(f"{'P99':<12} | {dist_user['p99']:<12} | {dist_assistant['p99']:<14} | {dist_total['p99']:<14}")
    print(f"{'Max':<12} | {dist_user['max']:<12} | {dist_assistant['max']:<14} | {dist_total['max']:<14}")
    print("=" * 60)


if __name__ == "__main__":
    main()
