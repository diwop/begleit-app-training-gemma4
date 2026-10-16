#!/usr/bin/env python3
"""
End-to-end test for prepare_data_dialogs_unidirectional.py,
prepare_few_shots_dialogs_unidirectional.py, and dynamic_few_shots_dialogs_unidirectional.py.
Verifies:
1. User turns (<user>) are completely excluded from unidirectional history.
2. When <user> starts a dialog, the first <partner> turn still gets 'keine Historie' and Bucket 0.
3. Subsequent <partner> turns include preceding 'Partner:' and 'Partner (Leichte Sprache):' blocks.
4. Few-shot buckets 0..5 count only preceding partner turns and cap Bucket 5 to the last 5 partner turns.
5. Prompt templates use 'Bisherige Aussagen des Gesprächspartners:'.
"""

import json
from pathlib import Path
import sys
import tempfile

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src-train"))
sys.path.insert(0, str(ROOT_DIR / "src-eval"))

import dynamic_few_shots_dialogs_unidirectional
import prepare_data_dialogs_unidirectional
import prepare_few_shots_dialogs_unidirectional

SAMPLE_DIALOG_USER_STARTS = """<user>
Guten Tag, ich habe eine Frage zu meinem Bescheid.
</user>
<partner>
Guten Tag! Nehmen Sie gerne Platz. Um welchen Bescheid handelt es sich?
</partner>
<partner simple formatting>
Guten Tag.  
Nehmen Sie gerne Platz.  
Um welchen Bescheid handelt es sich?
</partner>
<user>
Hier ist der Brief vom Amt wegen der Grundsicherung.
</user>
<partner>
Der Bewilligungszeitraum endet nächsten Monat, sodass ein Weiterbewilligungsantrag gestellt werden muss.
</partner>
<partner simple translation>
Ihr Geld vom Amt läuft nächsten Monat ab.  
Deshalb müssen Sie einen neuen Antrag stellen.
</partner>
"""

SAMPLE_DIALOG_LONG = "\n".join(
    [
        f"<user>\nNutzerfrage Nummer {i}\n</user>\n"
        f"<partner>\nPartneraussage Nummer {i} mit Fachbegriffen.\n</partner>\n"
        f"<partner simple translation>\nEinfache Aussage Nummer {i}.\n</partner>"
        for i in range(1, 8)
    ]
)


def test_unidirectional_data_and_few_shots():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_dir = tmp_path / "raw_dialogs"
        raw_dir.mkdir(parents=True)

        (raw_dir / "dialog01.txt").write_text(
            SAMPLE_DIALOG_USER_STARTS, encoding="utf-8"
        )
        (raw_dir / "dialog02.txt").write_text(
            SAMPLE_DIALOG_LONG, encoding="utf-8"
        )

        # 1. Test prepare_data_dialogs_unidirectional
        train_out = tmp_path / "dataset_train_dialogs_unidirectional.jsonl"
        eval_out = tmp_path / "dataset_eval_dialogs_unidirectional.jsonl"

        orig_raw = prepare_data_dialogs_unidirectional.RAW_DIR
        orig_train = prepare_data_dialogs_unidirectional.TRAIN_OUTPUT
        orig_eval = prepare_data_dialogs_unidirectional.EVAL_OUTPUT

        prepare_data_dialogs_unidirectional.RAW_DIR = raw_dir
        prepare_data_dialogs_unidirectional.TRAIN_OUTPUT = train_out
        prepare_data_dialogs_unidirectional.EVAL_OUTPUT = eval_out

        try:
            prepare_data_dialogs_unidirectional.main()

            assert train_out.exists() and eval_out.exists()
            train_records = [
                json.loads(line)
                for line in train_out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            eval_records = [
                json.loads(line)
                for line in eval_out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            all_records = train_records + eval_records
            assert len(all_records) > 0

            for rec in all_records:
                # Ensure no User: turn ever appears in history or user prompt
                assert "User:" not in rec["history"]
                assert "Nutzerfrage" not in rec["history"]
                assert "Frage zu meinem Bescheid" not in rec["history"]
                assert (
                    "Bisherige Aussagen des Gesprächspartners:" in rec["user"]
                )
                assert "Bisheriger Dialog:" not in rec["user"]

                # First partner exchange in any dialog must have 'keine Historie' even when <user> started
                if rec["exchange_idx"] == 0:
                    assert rec["history"] == "keine Historie"
                else:
                    assert "Partner:" in rec["history"]
                    assert "Partner (Leichte Sprache):" in rec["history"]

        finally:
            prepare_data_dialogs_unidirectional.RAW_DIR = orig_raw
            prepare_data_dialogs_unidirectional.TRAIN_OUTPUT = orig_train
            prepare_data_dialogs_unidirectional.EVAL_OUTPUT = orig_eval

        # 2. Test prepare_few_shots_dialogs_unidirectional & dynamic_few_shots_dialogs_unidirectional
        orig_fs_raw = prepare_few_shots_dialogs_unidirectional.RAW_DIR
        orig_fs_train = (
            prepare_few_shots_dialogs_unidirectional.FEW_SHOTS_TRAIN_OUTPUTS
        )
        orig_fs_eval = (
            prepare_few_shots_dialogs_unidirectional.FEW_SHOTS_EVAL_OUTPUTS
        )
        orig_fs_full = (
            prepare_few_shots_dialogs_unidirectional.FEW_SHOTS_FULL_OUTPUTS
        )

        fs_train_outs = [
            tmp_path / f"few_shots_dialogs_unidirectional_train_{i}.jsonl"
            for i in range(6)
        ]
        fs_eval_outs = [
            tmp_path / f"few_shots_dialogs_unidirectional_eval_{i}.jsonl"
            for i in range(6)
        ]
        fs_full_outs = [
            tmp_path / f"few_shots_dialogs_unidirectional_full_{i}.jsonl"
            for i in range(6)
        ]

        prepare_few_shots_dialogs_unidirectional.RAW_DIR = raw_dir
        prepare_few_shots_dialogs_unidirectional.FEW_SHOTS_TRAIN_OUTPUTS = (
            fs_train_outs
        )
        prepare_few_shots_dialogs_unidirectional.FEW_SHOTS_EVAL_OUTPUTS = (
            fs_eval_outs
        )
        prepare_few_shots_dialogs_unidirectional.FEW_SHOTS_FULL_OUTPUTS = (
            fs_full_outs
        )

        try:
            prepare_few_shots_dialogs_unidirectional.main()

            # Check full buckets across dialog01 (2 partner turns -> buckets 0, 1)
            # and dialog02 (7 partner turns -> buckets 0, 1, 2, 3, 4, 5, 5)
            full_buckets = [
                [
                    json.loads(line)
                    for line in p.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                for p in fs_full_outs
            ]
            assert len(full_buckets[0]) == 2  # first partner turn of both dialogs
            assert len(full_buckets[1]) == 2  # second partner turn of both dialogs
            assert len(full_buckets[2]) == 1
            assert len(full_buckets[3]) == 1
            assert len(full_buckets[4]) == 1
            assert len(full_buckets[5]) == 2  # 6th and 7th partner turns of dialog02

            # Check Bucket 5 capping (at most 5 preceding partner turns)
            for rec5 in full_buckets[5]:
                assert "User:" not in rec5["sample"]
                assert (
                    "Bisherige Aussagen des Gesprächspartners:"
                    in rec5["sample"]
                )
                assert (
                    dynamic_few_shots_dialogs_unidirectional.count_turns_in_history(
                        rec5["history"]
                    )
                    == 5
                )
                assert (
                    dynamic_few_shots_dialogs_unidirectional.determine_bucket(
                        rec5
                    )
                    == 5
                )

            # Check prompt building in dynamic_few_shots_dialogs_unidirectional
            prompt = dynamic_few_shots_dialogs_unidirectional.build_dynamic_few_shot_user_prompt(
                "Test-Eingabe",
                full_buckets[1][:1],
                query_history=full_buckets[1][0]["history"],
            )
            assert "Bisherige Aussagen des Gesprächspartners:" in prompt
            assert "Bisheriger Dialog:" not in prompt
            assert "User:" not in prompt

        finally:
            prepare_few_shots_dialogs_unidirectional.RAW_DIR = orig_fs_raw
            prepare_few_shots_dialogs_unidirectional.FEW_SHOTS_TRAIN_OUTPUTS = (
                orig_fs_train
            )
            prepare_few_shots_dialogs_unidirectional.FEW_SHOTS_EVAL_OUTPUTS = (
                orig_fs_eval
            )
            prepare_few_shots_dialogs_unidirectional.FEW_SHOTS_FULL_OUTPUTS = (
                orig_fs_full
            )

        # 3. Test evaluation_dialogs_unidirectional multi-model helpers & constants
        import evaluation_dialogs_unidirectional

        assert (
            evaluation_dialogs_unidirectional.GEMMA4_31B_MODEL_NAME
            == "RedHatAI/gemma-4-31B-it-FP8-dynamic"
        )
        assert (
            evaluation_dialogs_unidirectional.QWEN3_8_27B_MODEL_NAME
            == "Qwen/Qwen3.8-27B-FP8"
        )

        # Gemma 4 reasoning extraction
        r_g4, ans_g4 = (
            evaluation_dialogs_unidirectional.extract_gemma4_reasoning(
                "<|channel>thought\nKurze Überlegung<channel|>\nÜBERSETZUNG\nDas ist leicht."
            )
        )
        assert r_g4 == "Kurze Überlegung"
        assert ans_g4 == "ÜBERSETZUNG\nDas ist leicht."

        # Qwen pre-seeded <think> reasoning extraction
        raw_qwen = evaluation_dialogs_unidirectional.extract_output_text(
            {"text": "Schritt 1\n</think>\n\nÜBERSETZUNG\nDas ist leicht."},
            preseeded_think=True,
        )
        assert raw_qwen.startswith("<think>\n")
        r_qwen, ans_qwen = (
            evaluation_dialogs_unidirectional.extract_gemma4_reasoning(
                raw_qwen
            )
        )
        assert r_qwen == "Schritt 1"
        assert ans_qwen == "ÜBERSETZUNG\nDas ist leicht."

        struct_qwen = (
            evaluation_dialogs_unidirectional.compute_structural_reasoning_metrics(
                [
                    {
                        "text": (
                            "Schritt 1\n</think>\n\nÜBERSETZUNG\nDas ist leicht."
                        )
                    }
                ],
                preseeded_think=True,
            )
        )
        assert struct_qwen["valid_reasoning_rate"] == 100.0

        # Integrity check uses unidirectional template
        ic = evaluation_dialogs_unidirectional.get_integrity_checks(
            "System Prompt"
        )
        assert len(ic) == 2
        assert "Bisherige Aussagen des Gesprächspartners:" in ic[1]["user"]

    print(
        "[SUCCESS] All unidirectional dialogs data preparation, few-shot, and evaluation tests passed!"
    )


if __name__ == "__main__":
    test_unidirectional_data_and_few_shots()

