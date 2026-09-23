#!/usr/bin/env python3
"""
Recomputes German readability metrics (FRE and WSTF) on data/results.jsonl
and updates data/results.jsonl and data/results-metadata.json.
"""

import json
from pathlib import Path
import textstat

textstat.set_lang("de")


def get_raw_metrics(text: str | None) -> dict[str, float] | None:
    if not text or not text.strip() or len(text.strip().split()) < 3:
        return {"fre": 0.0, "wstf": 0.0}
    try:
        raw_fre = float(textstat.flesch_reading_ease(text))
        fre = round(max(0.0, min(100.0, raw_fre)), 1)
    except Exception:
        fre = 0.0
    try:
        raw_wstf = float(textstat.wiener_sachtextformel(text, 1))
        wstf = round(max(1.0, min(15.0, raw_wstf)), 1)
    except Exception:
        wstf = 0.0
    return {"fre": fre, "wstf": wstf}


def main():
    results_path = Path("data/results.jsonl")
    metadata_path = Path("data/results-metadata.json")

    with results_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    fre_scores = {
        "input": [],
        "ground_truth": [],
        "gemma4": [],
        "gemma4_thinking": [],
        "gemma4_dynamic_few_shots": [],
        "gemma4_merged_adapter_8bit": [],
    }
    wstf_scores = {
        "input": [],
        "ground_truth": [],
        "gemma4": [],
        "gemma4_thinking": [],
        "gemma4_dynamic_few_shots": [],
        "gemma4_merged_adapter_8bit": [],
    }

    updated_records = []
    for r in records:
        r["user_input_metrics"] = get_raw_metrics(r.get("user_input"))
        r["assistant_metrics"] = get_raw_metrics(r.get("assistant"))
        r["assistant_gemma4_metrics"] = get_raw_metrics(r.get("assistant_gemma4"))
        r["assistant_gemma4_thinking_metrics"] = get_raw_metrics(r.get("assistant_gemma4_thinking"))
        r["assistant_gemma4_dynamic_few_shots_metrics"] = get_raw_metrics(r.get("assistant_gemma4_dynamic_few_shots"))
        r["assistant_gemma4_merged_adapter_8bit_metrics"] = get_raw_metrics(r.get("assistant_gemma4_merged_adapter_8bit"))

        if r["assistant_metrics"] is not None:
            if r["user_input_metrics"]:
                fre_scores["input"].append(r["user_input_metrics"]["fre"])
                wstf_scores["input"].append(r["user_input_metrics"]["wstf"])
            fre_scores["ground_truth"].append(r["assistant_metrics"]["fre"])
            wstf_scores["ground_truth"].append(r["assistant_metrics"]["wstf"])
            if r["assistant_gemma4_metrics"]:
                fre_scores["gemma4"].append(r["assistant_gemma4_metrics"]["fre"])
                wstf_scores["gemma4"].append(r["assistant_gemma4_metrics"]["wstf"])
            if r["assistant_gemma4_thinking_metrics"]:
                fre_scores["gemma4_thinking"].append(r["assistant_gemma4_thinking_metrics"]["fre"])
                wstf_scores["gemma4_thinking"].append(r["assistant_gemma4_thinking_metrics"]["wstf"])
            if r["assistant_gemma4_dynamic_few_shots_metrics"]:
                fre_scores["gemma4_dynamic_few_shots"].append(r["assistant_gemma4_dynamic_few_shots_metrics"]["fre"])
                wstf_scores["gemma4_dynamic_few_shots"].append(r["assistant_gemma4_dynamic_few_shots_metrics"]["wstf"])
            if r["assistant_gemma4_merged_adapter_8bit_metrics"] and r.get("assistant_gemma4_merged_adapter_8bit"):
                fre_scores["gemma4_merged_adapter_8bit"].append(r["assistant_gemma4_merged_adapter_8bit_metrics"]["fre"])
                wstf_scores["gemma4_merged_adapter_8bit"].append(r["assistant_gemma4_merged_adapter_8bit_metrics"]["wstf"])

        updated_records.append(r)

    with results_path.open("w", encoding="utf-8") as f:
        for r in updated_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    num_eval = len(fre_scores["ground_truth"])
    avg_metrics = {
        "input_standardsprache": {
            "fre": round(sum(fre_scores["input"]) / len(fre_scores["input"]), 1) if fre_scores["input"] else None,
            "wstf": round(sum(wstf_scores["input"]) / len(wstf_scores["input"]), 1) if wstf_scores["input"] else None,
        },
        "ground_truth": {
            "fre": round(sum(fre_scores["ground_truth"]) / num_eval, 1) if num_eval else None,
            "wstf": round(sum(wstf_scores["ground_truth"]) / num_eval, 1) if num_eval else None,
        },
        "assistant_gemma4": {
            "fre": round(sum(fre_scores["gemma4"]) / len(fre_scores["gemma4"]), 1) if fre_scores["gemma4"] else None,
            "wstf": round(sum(wstf_scores["gemma4"]) / len(wstf_scores["gemma4"]), 1) if wstf_scores["gemma4"] else None,
            "speed_tokens_per_sec": metadata.get("model_speeds_tokens_per_second", {}).get("assistant_gemma4_speed"),
        },
        "assistant_gemma4_thinking": {
            "fre": round(sum(fre_scores["gemma4_thinking"]) / len(fre_scores["gemma4_thinking"]), 1) if fre_scores["gemma4_thinking"] else None,
            "wstf": round(sum(wstf_scores["gemma4_thinking"]) / len(wstf_scores["gemma4_thinking"]), 1) if wstf_scores["gemma4_thinking"] else None,
            "speed_tokens_per_sec": metadata.get("model_speeds_tokens_per_second", {}).get("assistant_gemma4_thinking_speed"),
        },
        "assistant_gemma4_dynamic_few_shots": {
            "fre": round(sum(fre_scores["gemma4_dynamic_few_shots"]) / len(fre_scores["gemma4_dynamic_few_shots"]), 1) if fre_scores["gemma4_dynamic_few_shots"] else None,
            "wstf": round(sum(wstf_scores["gemma4_dynamic_few_shots"]) / len(wstf_scores["gemma4_dynamic_few_shots"]), 1) if wstf_scores["gemma4_dynamic_few_shots"] else None,
            "speed_tokens_per_sec": metadata.get("model_speeds_tokens_per_second", {}).get("assistant_gemma4_dynamic_few_shots_speed"),
        },
        "assistant_gemma4_merged_adapter_8bit": {
            "fre": round(sum(fre_scores["gemma4_merged_adapter_8bit"]) / len(fre_scores["gemma4_merged_adapter_8bit"]), 1) if fre_scores["gemma4_merged_adapter_8bit"] else None,
            "wstf": round(sum(wstf_scores["gemma4_merged_adapter_8bit"]) / len(wstf_scores["gemma4_merged_adapter_8bit"]), 1) if wstf_scores["gemma4_merged_adapter_8bit"] else None,
            "speed_tokens_per_sec": metadata.get("model_speeds_tokens_per_second", {}).get("assistant_gemma4_merged_adapter_8bit_speed"),
        },
    }

    metadata["average_metrics"] = avg_metrics

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("      Evaluation Summary Metrics (Dataset Averages)")
    print("=" * 60)
    print(f"  * Input Standardsprache         : FRE = {avg_metrics['input_standardsprache']['fre']} | WSTF = {avg_metrics['input_standardsprache']['wstf']}")
    print(f"  * Ground Truth (Target)         : FRE = {avg_metrics['ground_truth']['fre']} | WSTF = {avg_metrics['ground_truth']['wstf']}")
    print(f"  * Gemma 4 (Zero-Shot)           : FRE = {avg_metrics['assistant_gemma4']['fre']} | WSTF = {avg_metrics['assistant_gemma4']['wstf']}")
    print(f"  * Gemma 4 (With Thinking)       : FRE = {avg_metrics['assistant_gemma4_thinking']['fre']} | WSTF = {avg_metrics['assistant_gemma4_thinking']['wstf']}")
    print(f"  * Gemma 4 (Few-Shots + Thinking): FRE = {avg_metrics['assistant_gemma4_dynamic_few_shots']['fre']} | WSTF = {avg_metrics['assistant_gemma4_dynamic_few_shots']['wstf']}")
    print(f"  * Gemma 4 (Merged 8-bit + Think): FRE = {avg_metrics['assistant_gemma4_merged_adapter_8bit']['fre']} | WSTF = {avg_metrics['assistant_gemma4_merged_adapter_8bit']['wstf']}")
    print("=" * 60)
    print("Structural reasoning metrics:")
    print(json.dumps(metadata.get("structural_reasoning_metrics", {}), indent=2))


if __name__ == "__main__":
    main()
