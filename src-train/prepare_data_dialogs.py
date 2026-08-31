#!/usr/bin/env python3
"""
Prepares dialogs training and evaluation JSONL datasets from raw dialog text files.
Each raw dialog file contains turns of <user>, <partner>, <partner simple translation>,
and <partner simple formatting>.

Splits dialog files as complete wholes into train and eval sets.
- Train dialogs: Creates 4 samples per dialog file (first exchange, last partner text, 25% and 75% percentiles).
- Eval dialogs: Creates 1 sample per partner text across the entire dialog.

Formats %HISTORY% and %INPUT% into prompts/prompt-template_dialogs.md.
Filters out outlier sequences exceeding MAX_SEQUENCE_LENGTH (8192 tokens).
Calculates token distributions and prints a token statistics summary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any

# Single source of truth for max sequence/token length
MAX_SEQUENCE_LENGTH = 8192

RAW_DIR = Path("data/raw_dialogs")
SYSTEM_PROMPT_PATH = Path("prompts/system-prompt_dialogs.md")
PROMPT_TEMPLATE_PATH = Path("prompts/prompt-template_dialogs.md")
TRAIN_OUTPUT = Path("data/dataset_train_dialogs.jsonl")
EVAL_OUTPUT = Path("data/dataset_eval_dialogs.jsonl")
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


def extract_partner_exchanges(
    turns: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Extracts all partner exchanges with their corresponding history.
    """
    exchanges = []
    partner_count = 0
    for idx, turn in enumerate(turns):
        if turn["speaker"] == "partner":
            history_str = format_history(turns[:idx])
            exchanges.append({
                "partner_idx": partner_count,
                "turn_idx": idx,
                "partner_text": turn["text"],
                "partner_translation": turn["translation"],
                "kind": turn["kind"],
                "history": history_str,
            })
            partner_count += 1
    return exchanges


def select_training_exchange_indices(num_exchanges: int) -> list[int]:
    """
    Selects 4 sample indices for training:
    - first verbal exchange (index 0)
    - 25% percentile
    - 75% percentile
    - last partner text (index num_exchanges - 1)
    """
    if num_exchanges <= 4:
        return list(range(num_exchanges))

    idx_first = 0
    idx_p25 = int(round(0.25 * (num_exchanges - 1)))
    idx_p75 = int(round(0.75 * (num_exchanges - 1)))
    idx_last = num_exchanges - 1

    # Return deduplicated, sorted list
    selected = sorted(list({idx_first, idx_p25, idx_p75, idx_last}))
    return selected


def main() -> None:
    print("=" * 60)
    print("      Dialogs Dataset Preparation & Token Length Filtering")
    print("=" * 60)
    print(f"[INFO] Max Sequence Length Limit : {MAX_SEQUENCE_LENGTH} tokens")
    print(f"[INFO] Raw Dialogs Directory     : {RAW_DIR}")

    if not SYSTEM_PROMPT_PATH.exists():
        print(
            f"[ERROR] System prompt file not found at: {SYSTEM_PROMPT_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not PROMPT_TEMPLATE_PATH.exists():
        print(
            f"[ERROR] Prompt template file not found at: {PROMPT_TEMPLATE_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()

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
            f"[WARNING] No dialog files found in {RAW_DIR}. Writing empty datasets."
        )
        write_jsonl(TRAIN_OUTPUT, [])
        write_jsonl(EVAL_OUTPUT, [])
        print(f"[SUCCESS] Initialized empty {TRAIN_OUTPUT} and {EVAL_OUTPUT}")
        return

    # Split dialog files as complete wholes into train and eval sets
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
    print(f"[INFO] Eval file(s): {[f.name for f in eval_files]}")
    print(f"[INFO] Train files : {[f.name for f in train_files]}")

    tokenizer = get_tokenizer()
    if tokenizer is not None:
        print(f"[INFO] Using tokenizer: {tokenizer.__class__.__name__}")
    else:
        print(
            "[INFO] Using calibrated German subword token estimation (~1.35"
            " tokens/word)"
        )

    system_token_count = count_tokens(system_prompt, tokenizer)
    print(f"[INFO] System Prompt Tokens: {system_token_count}")

    train_records = []
    eval_records = []
    filtered_outliers = []

    user_tokens = []
    assistant_tokens = []
    total_tokens = []

    # Process Training Dialog Files (4 samples per file: first, 25%, 75%, last)
    for f in train_files:
        turns = parse_dialog_file(f)
        exchanges = extract_partner_exchanges(turns)
        selected_indices = select_training_exchange_indices(len(exchanges))
        doc_stem = f.stem

        print(
            f"[INFO] Train Dialog '{f.name}': {len(exchanges)} partner"
            f" exchanges -> selected 4 samples at indices {selected_indices}"
        )

        for j in selected_indices:
            ex = exchanges[j]
            user_prompt = template.replace("%HISTORY%", ex["history"]).replace(
                "%INPUT%", ex["partner_text"]
            )
            assistant_text = ex["partner_translation"]
            sample_id = f"{doc_stem}_{j:02d}"

            u_tok = count_tokens(user_prompt, tokenizer)
            a_tok = count_tokens(assistant_text, tokenizer)
            prompt_total = system_token_count + u_tok
            tot_tok = prompt_total + a_tok

            if (
                prompt_total > MAX_SEQUENCE_LENGTH
                or tot_tok > MAX_SEQUENCE_LENGTH
            ):
                filtered_outliers.append({
                    "id": sample_id,
                    "split": "train",
                    "prompt_tokens": prompt_total,
                    "total_tokens": tot_tok,
                })
                continue

            user_tokens.append(u_tok)
            assistant_tokens.append(a_tok)
            total_tokens.append(tot_tok)

            train_records.append({
                "id": sample_id,
                "dialog": f.name,
                "exchange_idx": j,
                "system": system_prompt,
                "history": ex["history"],
                "user_input": ex["partner_text"],
                "user": user_prompt,
                "assistant": assistant_text,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_text},
                ],
            })

    # Process Evaluation Dialog Files (1 sample per partner text across the entire dialog)
    for f in eval_files:
        turns = parse_dialog_file(f)
        exchanges = extract_partner_exchanges(turns)
        doc_stem = f.stem

        print(
            f"[INFO] Eval Dialog '{f.name}': {len(exchanges)} partner exchanges"
            f" -> creating {len(exchanges)} eval samples"
        )

        for j, ex in enumerate(exchanges):
            user_prompt = template.replace("%HISTORY%", ex["history"]).replace(
                "%INPUT%", ex["partner_text"]
            )
            assistant_text = ex["partner_translation"]
            sample_id = f"{doc_stem}_{j:02d}"

            u_tok = count_tokens(user_prompt, tokenizer)
            a_tok = count_tokens(assistant_text, tokenizer)
            prompt_total = system_token_count + u_tok
            tot_tok = prompt_total + a_tok

            if (
                prompt_total > MAX_SEQUENCE_LENGTH
                or tot_tok > MAX_SEQUENCE_LENGTH
            ):
                filtered_outliers.append({
                    "id": sample_id,
                    "split": "eval",
                    "prompt_tokens": prompt_total,
                    "total_tokens": tot_tok,
                })
                continue

            user_tokens.append(u_tok)
            assistant_tokens.append(a_tok)
            total_tokens.append(tot_tok)

            eval_records.append({
                "id": sample_id,
                "dialog": f.name,
                "exchange_idx": j,
                "system": system_prompt,
                "history": ex["history"],
                "user_input": ex["partner_text"],
                "user": user_prompt,
                "assistant": assistant_text,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_text},
                ],
            })

    if filtered_outliers:
        print(
            f"[WARNING] Filtered {len(filtered_outliers)} outlier samples"
            f" exceeding {MAX_SEQUENCE_LENGTH} tokens:"
        )
        for out in filtered_outliers:
            print(
                f"       - Excluded ID {out['id']} ({out['split']}):"
                f" {out['prompt_tokens']} prompt tokens / {out['total_tokens']}"
                " total tokens"
            )

    train_records = sorted(train_records, key=lambda r: r["id"])
    eval_records = sorted(eval_records, key=lambda r: r["id"])

    write_jsonl(TRAIN_OUTPUT, train_records)
    write_jsonl(EVAL_OUTPUT, eval_records)

    dist_user = compute_distribution(user_tokens)
    dist_assistant = compute_distribution(assistant_tokens)
    dist_total = compute_distribution(total_tokens)

    print(
        f"\n[SUCCESS] Wrote {len(train_records)} train samples to {TRAIN_OUTPUT}"
    )
    print(f"[SUCCESS] Wrote {len(eval_records)} eval samples to {EVAL_OUTPUT}")

    # Print summary table
    print("\n" + "=" * 60)
    print("      Retained Dialogs Dataset Token Distribution Summary")
    print("=" * 60)
    print(
        f"{'Metric':<12} | {'User Prompt':<12} | {'Assistant (LS)':<14} |"
        f" {'Total Sequence':<14}"
    )
    print("-" * 60)
    print(
        f"{'Min':<12} | {dist_user['min']:<12} | {dist_assistant['min']:<14} |"
        f" {dist_total['min']:<14}"
    )
    print(
        f"{'Median (P50)':<12} | {dist_user['median']:<12} |"
        f" {dist_assistant['median']:<14} | {dist_total['median']:<14}"
    )
    print(
        f"{'Mean':<12} | {dist_user['mean']:<12} | {dist_assistant['mean']:<14}"
        f" | {dist_total['mean']:<14}"
    )
    print(
        f"{'P90':<12} | {dist_user['p90']:<12} | {dist_assistant['p90']:<14} |"
        f" {dist_total['p90']:<14}"
    )
    print(
        f"{'P95':<12} | {dist_user['p95']:<12} | {dist_assistant['p95']:<14} |"
        f" {dist_total['p95']:<14}"
    )
    print(
        f"{'P99':<12} | {dist_user['p99']:<12} | {dist_assistant['p99']:<14} |"
        f" {dist_total['p99']:<14}"
    )
    print(
        f"{'Max':<12} | {dist_user['max']:<12} | {dist_assistant['max']:<14} |"
        f" {dist_total['max']:<14}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
