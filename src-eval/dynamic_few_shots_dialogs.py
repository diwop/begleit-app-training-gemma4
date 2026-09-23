#!/usr/bin/env python3
"""
Dynamic Few-Shot Example Retriever for Gemma 4 Dialogs Evaluation / Inference.

Uses sentence-transformers with 'intfloat/multilingual-e5-base' to index training
dialog few-shot buckets (0 to 5, partitioned by history length) and retrieve the
most semantically relevant few-shot demonstrations for populating
prompts/prompt-template-dynamic-few-shots_dialogs.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Any

# Enforce offline mode on cluster compute nodes
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    from sentence_transformers import SentenceTransformer, util
    import torch

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    util = None
    torch = None
    HAS_SENTENCE_TRANSFORMERS = False

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_DATA_DIR = Path("data")
DEFAULT_SPLIT = "train"
FEW_SHOT_TEMPLATE_PATH = Path(
    "prompts/prompt-template-dynamic-few-shots_dialogs.md"
)
ZERO_SHOT_TEMPLATE_PATH = Path("prompts/prompt-template_dialogs.md")


def count_turns_in_history(history: str) -> int:
    """
    Counts the number of preceding turns from a formatted dialog history string.
    Each user turn starts with 'User:' and each partner turn starts with 'Partner:'
    (excluding 'Partner (Leichte Sprache):').
    """
    if not history or history.strip() == "keine Historie":
        return 0
    pattern = r"(?:^|\n)(User:|Partner:(?!\s*\(Leichte))"
    matches = re.findall(pattern, history)
    return len(matches)


def determine_bucket(history_or_record: str | dict[str, Any]) -> int:
    """
    Determines history bucket index (0 to 5) for a given history string or record dict.
    - Bucket 0: 0 turns (partner started)
    - Bucket 1: 1 turn (user started)
    - Buckets 2-4: 2-4 turns
    - Bucket 5: >= 5 turns
    """
    if isinstance(history_or_record, dict):
        if history_or_record.get("bucket") is not None:
            return min(max(int(history_or_record["bucket"]), 0), 5)
        if history_or_record.get("turn_idx") is not None:
            return min(max(int(history_or_record["turn_idx"]), 0), 5)
        hist = history_or_record.get("history", "")
    else:
        hist = str(history_or_record)

    turns = count_turns_in_history(hist)
    return min(max(turns, 0), 5)


def get_model_snapshot_path(model_name: str, required: bool = True) -> str:
    """Resolve model repo ID to local disk snapshot directory or fail fast."""
    if Path(model_name).exists():
        return model_name

    hf_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    repo_folder = "models--" + model_name.replace("/", "--")
    snapshots_dir = hf_home / "hub" / repo_folder / "snapshots"

    if not snapshots_dir.exists():
        if required:
            print(
                f"[ERROR] Embedding model cache directory not found at: {snapshots_dir}",
                file=sys.stderr,
            )
            print(
                "[INFO] Please run 'bash scripts/download_models.sh' on the login node first.",
                file=sys.stderr,
            )
            sys.exit(1)
        return ""

    snapshots = sorted(
        [p for p in snapshots_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not snapshots:
        if required:
            print(
                f"[ERROR] No snapshot directories found inside: {snapshots_dir}",
                file=sys.stderr,
            )
            print(
                "[INFO] Please run 'bash scripts/download_models.sh' on the login node first.",
                file=sys.stderr,
            )
            sys.exit(1)
        return ""

    return str(snapshots[0])


class DynamicFewShotIndex:
    """
    RAG Semantic Embedding Index over a single dialog dataset or few-shot bucket JSONL.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        model: Any | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name

        if device is None:
            if torch is not None and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.entries: list[dict[str, Any]] = []
        self.raw_inputs: list[str] = []
        self.corpus_embeddings = None

        self._load_dataset()
        if self.raw_inputs:
            if model is not None:
                self.model = model
            else:
                if not HAS_SENTENCE_TRANSFORMERS:
                    print(
                        "[WARNING] sentence-transformers not installed. Semantic search unavailable.",
                        file=sys.stderr,
                    )
                    self.model = None
                    return
                model_path = get_model_snapshot_path(self.model_name)
                print(
                    f"[INFO] Loading embedding model from: {model_path} (Device: {self.device})"
                )
                self.model = SentenceTransformer(model_path, device=self.device)
            self._encode_corpus()
        else:
            self.model = model

    def _load_dataset(self) -> None:
        """Load few-shot bucket or dialog dataset records."""
        if not self.dataset_path.exists():
            print(
                f"[WARNING] Few-shot dataset not found at '{self.dataset_path}'.",
                file=sys.stderr,
            )
            return

        self.entries = []
        self.raw_inputs = []

        with self.dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                record = json.loads(line_str)
                doc_id = record.get("id", "")
                dialog = record.get("dialog", "")
                exchange_idx = record.get("exchange_idx")
                turn_idx = record.get("turn_idx")
                tokens = record.get("tokens")
                sample = record.get("sample", "").strip()

                # Supports both bucket schema ("input", "sample") and standard schema ("user_input", "assistant")
                raw_input = record.get("input") or record.get("user_input", "")
                raw_input = raw_input.strip()

                if "assistant" in record and record["assistant"]:
                    assistant = record["assistant"].strip()
                elif sample:
                    m = re.search(r"```output\n(.*?)```", sample, re.DOTALL)
                    assistant = m.group(1).strip() if m else ""
                else:
                    assistant = ""

                history = record.get("history", "")
                if not sample and assistant and raw_input:
                    sample = (
                        "### Beispiel\n\n"
                        f"Bisheriger Dialog:\n\n```history\n{history or 'keine Historie'}\n```\n\n"
                        f"#### Eingabe: Standardsprache\n\n```input\n{raw_input}\n```\n\n"
                        f"#### Ausgabe: Übersetzung in Leichte Sprache\n\n```output\n{assistant}\n```"
                    )

                entry = {
                    "id": doc_id,
                    "dialog": dialog,
                    "exchange_idx": exchange_idx,
                    "turn_idx": turn_idx,
                    "input": raw_input,
                    "user_input": raw_input,
                    "assistant": assistant,
                    "sample": sample,
                    "history": history,
                    "tokens": tokens,
                }
                self.entries.append(entry)
                self.raw_inputs.append(raw_input)

    def _encode_corpus(self) -> None:
        """Encode all indexed passages using the loaded embedding model."""
        if self.model is None:
            return
        # Multilingual-E5 expects 'passage: ' prefix for indexed documents
        passage_texts = [f"passage: {text}" for text in self.raw_inputs]
        self.corpus_embeddings = self.model.encode(
            passage_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def get_closest_examples(
        self,
        query: str,
        k: int = 3,
        exclude_dialog: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Obtain the k semantically closest few-shot demonstrations for a given query.
        Optionally excludes demonstrations from the same dialog.
        """
        query_text = query.strip()
        if not query_text or not self.entries:
            return []

        if self.model is None or self.corpus_embeddings is None or util is None:
            # Fallback when embeddings model is not loaded (e.g. local offline test)
            results = []
            for entry in self.entries:
                if exclude_dialog and entry.get("dialog") == exclude_dialog:
                    continue
                ent = dict(entry)
                ent["score"] = 0.0
                results.append(ent)
                if len(results) >= k:
                    break
            return results

        # Fetch extra candidates in case some are filtered out by exclude_dialog
        search_k = min(max(k * 2, k + 5), len(self.entries))

        formatted_query = f"query: {query_text}"
        query_embedding = self.model.encode(
            formatted_query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        hits = util.semantic_search(
            query_embedding,
            self.corpus_embeddings,
            top_k=search_k,
        )[0]

        results = []
        for hit in hits:
            idx = hit["corpus_id"]
            entry = dict(self.entries[idx])
            if exclude_dialog and entry.get("dialog") == exclude_dialog:
                continue
            entry["score"] = round(float(hit["score"]), 4)
            results.append(entry)
            if len(results) >= k:
                break

        return results


class DynamicFewShotBucketIndex:
    """
    Manages semantic few-shot indices across all 6 history-length buckets (0 to 5).
    Reuses a single SentenceTransformer model across all buckets to minimize memory usage.
    """

    def __init__(
        self,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        split: str = DEFAULT_SPLIT,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.model_name = model_name

        if device is None:
            if torch is not None and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model: Any | None = None
        self.bucket_indices: dict[int, DynamicFewShotIndex] = {}

    def _ensure_model(self) -> Any | None:
        if self.model is None and HAS_SENTENCE_TRANSFORMERS:
            model_path = get_model_snapshot_path(self.model_name)
            print(
                f"[INFO] Loading shared embedding model from: {model_path} (Device: {self.device})"
            )
            self.model = SentenceTransformer(model_path, device=self.device)
        return self.model

    def get_bucket_index(self, bucket: int) -> DynamicFewShotIndex:
        bucket = min(max(int(bucket), 0), 5)
        if bucket not in self.bucket_indices:
            bucket_file = (
                self.data_dir / f"few_shots_dialogs_{self.split}_{bucket}.jsonl"
            )
            model = self._ensure_model()
            self.bucket_indices[bucket] = DynamicFewShotIndex(
                dataset_path=bucket_file,
                model=model,
                model_name=self.model_name,
                device=self.device,
            )
        return self.bucket_indices[bucket]

    def get_closest_examples(
        self,
        query: str,
        bucket: int = 0,
        k: int = 3,
        exclude_dialog: str | None = None,
    ) -> list[dict[str, Any]]:
        idx = self.get_bucket_index(bucket)
        return idx.get_closest_examples(
            query=query, k=k, exclude_dialog=exclude_dialog
        )


_cached_bucket_index: DynamicFewShotBucketIndex | None = None


def get_dynamic_few_shots(
    query: str,
    bucket: int = 0,
    k: int = 3,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    split: str = DEFAULT_SPLIT,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    exclude_dialog: str | None = None,
    dataset_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k semantically closest few-shot demonstrations from the specified history bucket.
    """
    global _cached_bucket_index

    # If an explicit dataset_path is provided (e.g. legacy caller)
    if dataset_path is not None:
        idx = DynamicFewShotIndex(
            dataset_path=dataset_path, model_name=model_name
        )
        return idx.get_closest_examples(
            query=query, k=k, exclude_dialog=exclude_dialog
        )

    if (
        _cached_bucket_index is None
        or _cached_bucket_index.data_dir != Path(data_dir)
        or _cached_bucket_index.split != split
        or _cached_bucket_index.model_name != model_name
    ):
        _cached_bucket_index = DynamicFewShotBucketIndex(
            data_dir=data_dir,
            split=split,
            model_name=model_name,
        )

    return _cached_bucket_index.get_closest_examples(
        query=query,
        bucket=bucket,
        k=k,
        exclude_dialog=exclude_dialog,
    )


def build_dynamic_few_shot_user_prompt(
    query_standardsprache: str,
    examples: list[dict[str, Any]],
    query_history: str = "keine Historie",
    few_shot_template_path: Path | str = FEW_SHOT_TEMPLATE_PATH,
    zero_shot_template_path: Path | str = ZERO_SHOT_TEMPLATE_PATH,
) -> str:
    """
    Build user prompt string populated with dynamic few-shot demonstrations.
    If examples is empty, cleanly falls back to prompts/prompt-template_dialogs.md.
    """
    query_text = query_standardsprache.strip()
    history_text = query_history.strip() if query_history else "keine Historie"

    if not examples:
        zero_shot_template = Path(zero_shot_template_path).read_text(
            encoding="utf-8"
        )
        return (
            zero_shot_template.replace("%HISTORY%", history_text).replace(
                "%INPUT%", query_text
            )
        )

    few_shot_template = Path(few_shot_template_path).read_text(
        encoding="utf-8"
    )

    sample_blocks = []
    for i, ex in enumerate(examples, start=1):
        s = ex.get("sample", "").strip()
        if "%NUMBER%" in s:
            s = s.replace("%NUMBER%", str(i))
        sample_blocks.append(s)

    few_shots_str = "\n\n".join(sample_blocks)

    return (
        few_shot_template.replace("%FEW_SHOTS%", few_shots_str)
        .replace("%HISTORY%", history_text)
        .replace("%INPUT%", query_text)
    )


def get_fitting_few_shot_examples(
    query: str,
    tokenizer: Any = None,
    query_history: str = "keine Historie",
    bucket: int | None = None,
    max_input_tokens: int = 24000,
    max_examples: int = 3,
    candidate_k: int = 15,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    split: str = DEFAULT_SPLIT,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    exclude_dialog: str | None = None,
    dataset_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve candidate few-shot examples from the history-length bucket and greedily select
    up to `max_examples` such that the total token count of the constructed user prompt
    does not exceed `max_input_tokens`.
    """
    if bucket is None:
        bucket = determine_bucket(query_history)
    else:
        bucket = min(max(int(bucket), 0), 5)

    candidates = get_dynamic_few_shots(
        query=query,
        bucket=bucket,
        k=candidate_k,
        data_dir=data_dir,
        split=split,
        model_name=model_name,
        exclude_dialog=exclude_dialog,
        dataset_path=dataset_path,
    )

    selected: list[dict[str, Any]] = []

    # Check baseline (0 examples)
    base_prompt = build_dynamic_few_shot_user_prompt(
        query, [], query_history=query_history
    )
    if tokenizer is not None:
        base_tokens = len(
            tokenizer.encode(base_prompt, add_special_tokens=False)
        )
    else:
        base_tokens = len(base_prompt) // 4

    if base_tokens >= max_input_tokens:
        return []

    for cand in candidates:
        test_selected = selected + [cand]
        test_prompt = build_dynamic_few_shot_user_prompt(
            query, test_selected, query_history=query_history
        )
        if tokenizer is not None:
            test_tokens = len(
                tokenizer.encode(test_prompt, add_special_tokens=False)
            )
        else:
            test_tokens = len(test_prompt) // 4

        if test_tokens <= max_input_tokens:
            selected.append(cand)
            if len(selected) >= max_examples:
                break

    return selected


if __name__ == "__main__":
    test_query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Guten Tag, Herr Schmidt. Wie geht es Ihnen heute?"
    )

    print("=" * 60)
    print("      Dynamic Few-Shot Retriever - Dialogs (Buckets 0-5)")
    print("=" * 60)
    print(f"[INFO] Query Text: {test_query}\n")

    for b in range(6):
        results = get_dynamic_few_shots(test_query, bucket=b, k=3)
        print(f"\n--- Bucket {b} (Retrieved {len(results)} examples) ---")
        for rank, ex in enumerate(results, start=1):
            print(
                f"[{rank}] ID: {ex['id']} | Score: {ex['score']:.4f} | Input:"
                f" {ex['input'][:60]}..."
            )

    sample_results = get_dynamic_few_shots(test_query, bucket=0, k=2)
    prompt = build_dynamic_few_shot_user_prompt(
        test_query, sample_results, query_history="keine Historie"
    )
    print("\n" + "=" * 60)
    print("      Generated Dynamic Few-Shot Prompt Preview (Bucket 0, 2 shots)")
    print("=" * 60)
    print(prompt)
