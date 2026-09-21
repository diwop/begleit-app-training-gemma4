#!/usr/bin/env python3
"""
End-to-end test for prepare_data.py with temporary raw documents.
Verifies that generated JSONL entries have valid input_output segments and masked empty reasoning traces.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

# Add src-train to path
SRC_TRAIN_DIR = Path(__file__).resolve().parent.parent / "src-train"
sys.path.insert(0, str(SRC_TRAIN_DIR))

import prepare_data


def test_prepare_data_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = Path(tmp_dir) / "raw"
        raw_dir.mkdir(parents=True)

        std_file = raw_dir / "doc001_Standardsprache.txt"
        ls_file = raw_dir / "doc001_Leichte_Sprache.txt"

        std_file.write_text("Das ist ein komplexer deutscher Text für einen Antrag.", encoding="utf-8")
        ls_file.write_text("Das ist ein einfacher Text.", encoding="utf-8")

        train_out = Path(tmp_dir) / "dataset_train.jsonl"
        eval_out = Path(tmp_dir) / "dataset_eval.jsonl"

        # Temporarily patch module paths
        orig_raw = prepare_data.RAW_DIR
        orig_train = prepare_data.TRAIN_OUTPUT
        orig_eval = prepare_data.EVAL_OUTPUT

        prepare_data.RAW_DIR = raw_dir
        prepare_data.TRAIN_OUTPUT = train_out
        prepare_data.EVAL_OUTPUT = eval_out

        try:
            prepare_data.main()

            assert train_out.exists() or eval_out.exists()

            # Read created record
            out_file = train_out if train_out.exists() and train_out.stat().st_size > 0 else eval_out
            lines = out_file.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 1

            record = json.loads(lines[0])
            assert record["id"] == "doc001"
            assert "segments" in record
            assert len(record["segments"]) == 2

            seg0 = record["segments"][0]
            seg1 = record["segments"][1]

            assert seg0["label"] is False
            assert "<|channel>thought\n<channel|>\n" in seg0["text"]
            assert "Das ist ein komplexer deutscher Text" in seg0["text"]

            assert seg1["label"] is True
            assert "Das ist ein einfacher Text.<turn|>\n" in seg1["text"]

            print("[SUCCESS] End-to-end data preparation test passed!")
        finally:
            prepare_data.RAW_DATA_DIR = orig_raw
            prepare_data.TRAIN_OUTPUT = orig_train
            prepare_data.EVAL_OUTPUT = orig_eval


if __name__ == "__main__":
    test_prepare_data_pipeline()
