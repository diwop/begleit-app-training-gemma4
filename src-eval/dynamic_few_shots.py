#!/usr/bin/env python3
"""
Dynamic Few-Shot Example Retriever for Gemma 4 Evaluation / Inference.

Uses sentence-transformers with 'intfloat/multilingual-e5-base' to index training
pairs and retrieve the most semantically relevant few-shot demonstrations for
populating prompts/prompt-template-dynamic-few-shots.md.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Enforce offline mode on cluster compute nodes
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("[ERROR] 'sentence-transformers' is not installed in the environment.", file=sys.stderr)
    print("[INFO] Please run 'bash scripts/download_models.sh' on the login node.", file=sys.stderr)
    sys.exit(1)

import torch

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_TRAIN_DATASET = Path("data/dataset_train.jsonl")
DEFAULT_RAW_DIR = Path("data/raw")
FEW_SHOT_TEMPLATE_PATH = Path("prompts/prompt-template-dynamic-few-shots.md")


def get_model_snapshot_path(model_name: str) -> str:
    """Resolve model repo ID to local disk snapshot directory or fail fast."""
    if Path(model_name).exists():
        return model_name

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_folder = "models--" + model_name.replace("/", "--")
    snapshots_dir = hf_home / "hub" / repo_folder / "snapshots"

    if not snapshots_dir.exists():
        print(f"[ERROR] Embedding model cache directory not found at: {snapshots_dir}", file=sys.stderr)
        print("[INFO] Please run 'bash scripts/download_models.sh' on the login node first.", file=sys.stderr)
        sys.exit(1)

    snapshots = sorted(
        [p for p in snapshots_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not snapshots:
        print(f"[ERROR] No snapshot directories found inside: {snapshots_dir}", file=sys.stderr)
        print("[INFO] Please run 'bash scripts/download_models.sh' on the login node first.", file=sys.stderr)
        sys.exit(1)

    return str(snapshots[0])


def load_raw_standardsprache(doc_id: str, raw_dir: Path = DEFAULT_RAW_DIR) -> str:
    """Load raw Standardsprache text from data/raw/{id}_Standardsprache.txt."""
    raw_file = raw_dir / f"{doc_id}_Standardsprache.txt"
    if raw_file.exists():
        return raw_file.read_text(encoding="utf-8").strip()
    return ""


class DynamicFewShotIndex:
    """
    RAG Semantic Embedding Index over training dataset using sentence-transformers.
    """

    def __init__(
        self,
        dataset_path: Path | str = DEFAULT_TRAIN_DATASET,
        raw_dir: Path | str = DEFAULT_RAW_DIR,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.raw_dir = Path(raw_dir)
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.entries: list[dict[str, Any]] = []
        self.raw_inputs: list[str] = []

        self._load_dataset()
        self._load_model_and_encode()

    def _load_dataset(self) -> None:
        """Load dataset_train.jsonl and resolve raw Standardsprache texts by doc_id."""
        if not self.dataset_path.exists():
            print(f"[ERROR] Training dataset not found at '{self.dataset_path}'.", file=sys.stderr)
            print("[INFO] Run 'dvc repro' or 'python3 src-train/prepare_data.py' first.", file=sys.stderr)
            sys.exit(1)

        self.entries = []
        self.raw_inputs = []

        with self.dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line.strip())
                doc_id = record.get("id", "")
                assistant = record.get("assistant", "").strip()

                raw_input = load_raw_standardsprache(doc_id, self.raw_dir)
                if not raw_input:
                    raw_input = record.get("user", "").strip()

                entry = {
                    "id": doc_id,
                    "user_input": raw_input,
                    "assistant": assistant,
                }
                self.entries.append(entry)
                self.raw_inputs.append(raw_input)

    def _load_model_and_encode(self) -> None:
        """Load local embedding model snapshot and pre-encode all training passages."""
        model_path = get_model_snapshot_path(self.model_name)
        print(f"[INFO] Loading embedding model from: {model_path} (Device: {self.device})")
        self.model = SentenceTransformer(model_path, device=self.device)

        # Multilingual-E5 expects 'passage: ' prefix for indexed documents
        passage_texts = [f"passage: {text}" for text in self.raw_inputs]

        print(f"[INFO] Encoding {len(passage_texts)} training passages into semantic embeddings...")
        self.corpus_embeddings = self.model.encode(
            passage_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        print(f"[SUCCESS] Semantic index ready ({self.corpus_embeddings.shape[0]} embeddings, dim={self.corpus_embeddings.shape[1]})")

    def get_closest_examples(
        self,
        query: str,
        k: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Obtain the k semantically closest training input/output pairs for a given query.

        Args:
            query: The Standardsprache input text.
            k: Number of few-shot examples to retrieve (default: 2).

        Returns:
            List of dicts containing: 'id', 'user_input', 'assistant', and 'score'.
        """
        query_text = query.strip()
        if not query_text or not self.entries:
            return []

        k = min(k, len(self.entries))

        # Multilingual-E5 expects 'query: ' prefix for search queries
        formatted_query = f"query: {query_text}"
        query_embedding = self.model.encode(
            formatted_query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        hits = util.semantic_search(
            query_embedding,
            self.corpus_embeddings,
            top_k=k,
        )[0]

        results = []
        for hit in hits:
            idx = hit["corpus_id"]
            entry = dict(self.entries[idx])
            entry["score"] = round(float(hit["score"]), 4)
            results.append(entry)

        return results


# Cached singleton instance for fast reuse across multiple calls
_cached_index: DynamicFewShotIndex | None = None


def get_dynamic_few_shots(
    query: str,
    k: int = 2,
    dataset_path: Path | str = DEFAULT_TRAIN_DATASET,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    """
    Convenience function to get top-k few-shot examples from the training set.

    Args:
        query: Query text (Standardsprache).
        k: Number of examples to retrieve (default: 2).
        dataset_path: Path to dataset_train.jsonl.
        raw_dir: Path to raw files directory.
        model_name: Hugging Face model identifier for embeddings.

    Returns:
        List of dicts with 'id', 'user_input', 'assistant', 'score'.
    """
    global _cached_index
    if (
        _cached_index is None
        or _cached_index.dataset_path != Path(dataset_path)
        or _cached_index.model_name != model_name
    ):
        _cached_index = DynamicFewShotIndex(
            dataset_path=dataset_path,
            raw_dir=raw_dir,
            model_name=model_name,
        )
    return _cached_index.get_closest_examples(query=query, k=k)


def build_dynamic_few_shot_user_prompt(
    query_standardsprache: str,
    examples: list[dict[str, Any]],
    template_path: Path = FEW_SHOT_TEMPLATE_PATH,
) -> str:
    """
    Populate prompt-template-dynamic-few-shots.md with 2 retrieved few-shot examples and input text.

    Args:
        query_standardsprache: The Standardsprache text to translate.
        examples: Top-2 retrieved example dicts from get_closest_examples().
        template_path: Path to prompts/prompt-template-dynamic-few-shots.md.

    Returns:
        Filled user prompt string with 2 few-shot demonstrations and the target input.
    """
    template_content = template_path.read_text(encoding="utf-8").strip()

    ex1_in = examples[0]["user_input"] if len(examples) > 0 else ""
    ex1_out = examples[0]["assistant"] if len(examples) > 0 else ""
    ex2_in = examples[1]["user_input"] if len(examples) > 1 else ""
    ex2_out = examples[1]["assistant"] if len(examples) > 1 else ""

    filled = (
        template_content.replace("%FEW_SHOT_INPUT_1%", ex1_in)
        .replace("%FEW_SHOT_OUTPUT_1%", ex1_out)
        .replace("%FEW_SHOT_INPUT_2%", ex2_in)
        .replace("%FEW_SHOT_OUTPUT_2%", ex2_out)
        .replace("%INPUT%", query_standardsprache.strip())
    )
    return filled


if __name__ == "__main__":
    test_query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Informationen zur Barrierefreiheit und Leichten Sprache bei Behörden und Ämtern."
    )

    print("=" * 60)
    print("      Dynamic Few-Shot Retriever (sentence-transformers)")
    print("=" * 60)
    print(f"[INFO] Query Text: {test_query}\n")

    retriever = DynamicFewShotIndex()
    results = retriever.get_closest_examples(test_query, k=2)
    print(f"\n[INFO] Top {len(results)} Semantically Closest Training Pairs:\n")

    for rank, ex in enumerate(results, start=1):
        print(f"[{rank}] ID: {ex['id']} | Similarity Score: {ex['score']:.4f}")
        print(f"    Input (Standardsprache preview): {ex['user_input'][:120]}...")
        print(f"    Target (Leichte Sprache preview): {ex['assistant'][:120]}...\n")

    if FEW_SHOT_TEMPLATE_PATH.exists():
        prompt = build_dynamic_few_shot_user_prompt(test_query, results)
        print("=" * 60)
        print("      Generated Dynamic Few-Shot Prompt Preview")
        print("=" * 60)
        print(prompt[:600] + "\n...\n")
