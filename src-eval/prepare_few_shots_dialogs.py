#!/usr/bin/env python3
"""
Prepares the dynamic few-shot database partitioned into 6 history-length buckets (0 to 5)
for evaluation from raw dialog text files.

Buckets:
- Bucket 0: history_len == 0 (partner started, 'keine Historie')
- Bucket 1: history_len == 1 (user started)
- Buckets 2-4: history_len == N (N preceding turns)
- Bucket 5: history_len >= 5 (capped to last 5 turns)

Outputs:
- data/few_shots_dialogs_train_{0..5}.jsonl (from 51 train dialogs, used by evaluation)
- data/few_shots_dialogs_eval_{0..5}.jsonl  (from 6 eval dialogs)
- data/few_shots_dialogs_full_{0..5}.jsonl  (from all 57 dialogs)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any

RAW_DIR = Path("data/raw_dialogs")
FEW_SHOTS_SAMPLE_TEMPLATE_PATH = Path(
    "prompts/prompt-template-dynamic-few-shots_dialogs_sample.md"
)

FEW_SHOTS_TRAIN_OUTPUTS = [
    Path(f"data/few_shots_dialogs_train_{i}.jsonl") for i in range(6)
]
FEW_SHOTS_EVAL_OUTPUTS = [
    Path(f"data/few_shots_dialogs_eval_{i}.jsonl") for i in range(6)
]
FEW_SHOTS_FULL_OUTPUTS = [
    Path(f"data/few_shots_dialogs_full_{i}.jsonl") for i in range(6)
]

EVAL_RATIO = 0.10
SEED = 42


def get_tokenizer():
    """Attempt to load Hugging Face tokenizer if available offline or in environment."""
    try:
        from transformers import AutoTokenizer

        hf_home = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        for snap_dir in sorted(
            hf_home.glob("hub/models--*gemma*/snapshots/*"), reverse=True
        ):
            if (snap_dir / "tokenizer.json").exists() or (
                snap_dir / "tokenizer_config.json"
            ).exists():
                return AutoTokenizer.from_pretrained(
                    str(snap_dir), local_files_only=True
                )
        return AutoTokenizer.from_pretrained(
            "google/gemma-2-27b", local_files_only=True
        )
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_dialog_file(filepath: Path) -> list[dict[str, str]]:
    """
    Parses a raw dialog file into ordered turns of user and partner exchanges.
    """
    content = filepath.read_text(encoding="utf-8")
    pattern = r"<(user|partner|partner simple translation|partner simple formatting)>(.*?)</(user|partner)>"
    matches = re.findall(pattern, content, flags=re.DOTALL)

    turns = []
    i = 0
    while i < len(matches):
        tag, text, _ = matches[i]
        if tag == "user":
            turns.append({
                "speaker": "user",
                "text": text.strip(),
            })
            i += 1
        elif tag == "partner":
            if i + 1 < len(matches) and matches[i + 1][0] in (
                "partner simple translation",
                "partner simple formatting",
            ):
                simple_tag, simple_text, _ = matches[i + 1]
                kind = (
                    "ÜBERSETZUNG"
                    if simple_tag == "partner simple translation"
                    else "FORMATIERUNG"
                )
                turns.append({
                    "speaker": "partner",
                    "text": text.strip(),
                    "kind": kind,
                    "translation": simple_text.strip(),
                })
                i += 2
            else:
                print(
                    f"[WARNING] Partner turn without simple translation/formatting in {filepath} at tag index {i}"
                )
                i += 1
        else:
            i += 1

    return turns


def format_history(turns_before: list[dict[str, str]]) -> str:
    """
    Formats preceding dialog turns into the specified history structure:

    User: user text keeping new lines

    Partner: partner text keeping new lines

    Partner (Leichte Sprache): ÜBERSETZUNG|FORMATIERUNG
    partner translation keeping markdown and new lines
    """
    if not turns_before:
        return "keine Historie"

    blocks = []
    for turn in turns_before:
        if turn["speaker"] == "user":
            blocks.append("User: " + turn["text"])
        elif turn["speaker"] == "partner":
            blocks.append("Partner: " + turn["text"])
            blocks.append(
                f"Partner (Leichte Sprache): {turn['kind']}\n{turn['translation']}"
            )

    if not blocks:
        return "keine Historie"

    return "\n\n".join(blocks)


def extract_few_shot_buckets(
    files: list[Path],
    sample_template: str,
    tokenizer=None,
) -> dict[int, list[dict[str, Any]]]:
    """
    Extracts partner turns into 6 history-length buckets:
    - Bucket 0: history_len == 0 (starts of discussion where partner started)
    - Bucket 1: history_len == 1 (starts of discussion with first partner text when user started)
    - Bucket 2: history_len == 2
    - Bucket 3: history_len == 3
    - Bucket 4: history_len == 4
    - Bucket 5: history_len >= 5 (history capped to the last 5 turns)

    Format of each record:
    {
      "id": ...,
      "dialog": ...,
      "exchange_idx": ...,
      "turn_idx": ...,
      "input": "...",
      "sample": "...",
      "tokens": ...
    }
    """
    buckets: dict[int, list[dict[str, Any]]] = {i: [] for i in range(6)}

    for f in sorted(files, key=lambda x: x.name):
        turns = parse_dialog_file(f)
        partner_count = 0
        doc_stem = f.stem

        for idx, turn in enumerate(turns):
            if turn["speaker"] == "partner":
                history_len = idx  # number of preceding turns
                bucket_idx = min(history_len, 5)

                if bucket_idx == 5:
                    turns_for_history = turns[:idx][-5:]
                else:
                    turns_for_history = turns[:idx]

                history_str = format_history(turns_for_history)
                few_shot_output = f"{turn['kind']}\n{turn['translation']}"
                sample_text = (
                    sample_template.replace("%FEW_SHOT_HISTORY%", history_str)
                    .replace("%FEW_SHOT_INPUT%", turn["text"])
                    .replace("%FEW_SHOT_OUTPUT%", few_shot_output)
                )
                tok_count = count_tokens(sample_text, tokenizer)
                sample_id = f"{doc_stem}_{partner_count:02d}"

                record = {
                    "id": sample_id,
                    "dialog": f.name,
                    "exchange_idx": partner_count,
                    "turn_idx": idx,
                    "kind": turn["kind"],
                    "input": turn["text"],
                    "user_input": turn["text"],
                    "output": few_shot_output,
                    "assistant": few_shot_output,
                    "sample": sample_text,
                    "tokens": tok_count,
                }
                buckets[bucket_idx].append(record)
                partner_count += 1

    return buckets


def main() -> None:
    print("=" * 60)
    print("      Few-Shot Dialogs Database Preparation (Buckets 0-5)")
    print("=" * 60)
    print(f"[INFO] Raw Dialogs Directory     : {RAW_DIR}")
    print(f"[INFO] Sample Template Path      : {FEW_SHOTS_SAMPLE_TEMPLATE_PATH}")

    if not FEW_SHOTS_SAMPLE_TEMPLATE_PATH.exists():
        print(
            f"[ERROR] Sample template file not found at: {FEW_SHOTS_SAMPLE_TEMPLATE_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    sample_template = FEW_SHOTS_SAMPLE_TEMPLATE_PATH.read_text(
        encoding="utf-8"
    ).strip()

    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted([
        f
        for f in RAW_DIR.glob("*.txt")
        if not f.name.startswith(".") and f.name != ".gitkeep"
    ])
    print(f"[INFO] Found {len(raw_files)} raw dialog files in {RAW_DIR}")

    if not raw_files:
        print(
            f"[WARNING] No dialog files found in {RAW_DIR}. Writing empty bucket files."
        )
        for i in range(6):
            write_jsonl(FEW_SHOTS_TRAIN_OUTPUTS[i], [])
            write_jsonl(FEW_SHOTS_EVAL_OUTPUTS[i], [])
            write_jsonl(FEW_SHOTS_FULL_OUTPUTS[i], [])
        return

    # Split dialog files as complete wholes into train and eval sets (consistent seed=42)
    rng = random.Random(SEED)
    shuffled_files = list(raw_files)
    rng.shuffle(shuffled_files)

    eval_count = max(1, int(round(len(shuffled_files) * EVAL_RATIO)))
    eval_files = sorted(shuffled_files[:eval_count], key=lambda f: f.name)
    train_files = sorted(shuffled_files[eval_count:], key=lambda f: f.name)

    print(
        f"[INFO] Split {len(raw_files)} dialog files: {len(train_files)} Train"
        f" files, {len(eval_files)} Eval file(s)"
    )

    tokenizer = get_tokenizer()

    buckets_train = extract_few_shot_buckets(
        train_files, sample_template, tokenizer=tokenizer
    )
    buckets_eval = extract_few_shot_buckets(
        eval_files, sample_template, tokenizer=tokenizer
    )
    buckets_full = extract_few_shot_buckets(
        raw_files, sample_template, tokenizer=tokenizer
    )

    for i in range(6):
        write_jsonl(FEW_SHOTS_TRAIN_OUTPUTS[i], buckets_train[i])
        write_jsonl(FEW_SHOTS_EVAL_OUTPUTS[i], buckets_eval[i])
        write_jsonl(FEW_SHOTS_FULL_OUTPUTS[i], buckets_full[i])

    print("\n" + "=" * 60)
    print("      Few-Shot Dialogs Buckets Summary")
    print("=" * 60)
    print(
        f"{'Bucket':<8} | {'History Description':<22} | {'Train':<7} | {'Eval':<6} | {'Full':<6}"
    )
    print("-" * 60)
    for i in range(6):
        desc = (
            "0 (partner started)"
            if i == 0
            else (
                "1 (user started)"
                if i == 1
                else (f"{i} turns" if i < 5 else ">= 5 turns (capped)")
            )
        )
        print(
            f"Bucket {i:<1} | {desc:<22} | {len(buckets_train[i]):<7} | {len(buckets_eval[i]):<6} | {len(buckets_full[i]):<6}"
        )
    print("-" * 60)
    total_train_fs = sum(len(b) for b in buckets_train.values())
    total_eval_fs = sum(len(b) for b in buckets_eval.values())
    total_full_fs = sum(len(b) for b in buckets_full.values())
    print(
        f"{'Total':<8} | {'All partner turns':<22} | {total_train_fs:<7} | {total_eval_fs:<6} | {total_full_fs:<6}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
