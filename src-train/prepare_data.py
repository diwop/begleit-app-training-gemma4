#!/usr/bin/env python3
"""
Prepares training and evaluation JSONL datasets from raw text pairs and prompt templates.
Fixed 10% reproducible split (seed 42) for DVC pipeline.
"""

from pathlib import Path
import json
import random

RAW_DIR = Path("data/raw")
SYSTEM_PROMPT_PATH = Path("prompts/system-prompt.md")
PROMPT_TEMPLATE_PATH = Path("prompts/prompt-template.md")
TRAIN_OUTPUT = Path("data/dataset_train.jsonl")
EVAL_OUTPUT = Path("data/dataset_eval.jsonl")
EVAL_RATIO = 0.10
SEED = 42


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()

    std_files = {f.name.split("_")[0]: f for f in RAW_DIR.glob("*_Standardsprache.txt")}
    ls_files = {f.name.split("_")[0]: f for f in RAW_DIR.glob("*_Leichte_Sprache.txt")}

    common_ids = sorted(std_files.keys() & ls_files.keys())
    print(f"[INFO] Found {len(common_ids)} document pairs in {RAW_DIR}")

    records = []
    for doc_id in common_ids:
        std_text = std_files[doc_id].read_text(encoding="utf-8").strip()
        ls_text = ls_files[doc_id].read_text(encoding="utf-8").strip()
        user_prompt = template.replace("%INPUT%", std_text)

        records.append({
            "id": doc_id,
            "system": system_prompt,
            "user": user_prompt,
            "assistant": ls_text,
        })

    # Deterministic shuffle & split
    rng = random.Random(SEED)
    shuffled = list(records)
    rng.shuffle(shuffled)

    eval_count = int(round(len(shuffled) * EVAL_RATIO))
    eval_records = sorted(shuffled[:eval_count], key=lambda r: r["id"])
    train_records = sorted(shuffled[eval_count:], key=lambda r: r["id"])

    write_jsonl(TRAIN_OUTPUT, train_records)
    write_jsonl(EVAL_OUTPUT, eval_records)

    print(f"[SUCCESS] Wrote {len(train_records)} train samples to {TRAIN_OUTPUT}")
    print(f"[SUCCESS] Wrote {len(eval_records)} eval samples to {EVAL_OUTPUT}")


if __name__ == "__main__":
    main()
