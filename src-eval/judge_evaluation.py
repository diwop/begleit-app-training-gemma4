#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation Pipeline using Llama 4 Scout (FP8 on 4 GPUs).

Evaluates Leichte Sprache translations across multiple model variants against:
  1. Official Leichte Sprache rules (Rule Adherence)
  2. Source text (Factual & Semantic Completeness)
  3. Readability, natural tone, and respectful formulation
  4. Relative quality comparison to the Human Ground Truth reference

Outputs:
  - data/judge_results.jsonl: Per-sample, per-candidate detailed critiques and numerical scores
  - data/judge_summary.json: Aggregated model benchmarks, win-rates, and overall rankings
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import sglang as sgl
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
DEFAULT_JUDGE_MODEL = "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8"
JUDGE_MODEL_NAME = os.environ.get("JUDGE_MODEL_NAME", DEFAULT_JUDGE_MODEL)
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "4"))
MAX_EVAL_SAMPLES = int(os.environ.get("MAX_EVAL_SAMPLES", "0"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQUENCE_LENGTH", "16384"))

EVAL_RESULTS_PATH = Path("data/results.jsonl")
JUDGE_SYSTEM_PROMPT_PATH = Path("prompts/judge-system-prompt.md")
JUDGE_PROMPT_TEMPLATE_PATH = Path("prompts/judge-prompt-template.md")
TRANSLATOR_SYSTEM_PROMPT_PATH = Path("prompts/system-prompt.md")
JUDGE_RESULTS_PATH = Path("data/judge_results.jsonl")
JUDGE_SUMMARY_PATH = Path("data/judge_summary.json")

# Models evaluated from data/results.jsonl
CANDIDATE_MODELS = [
    ("assistant_gemma4", "Zero-Shot (Base Gemma 4)"),
    ("assistant_gemma4_thinking", "Zero-Shot + Thinking (Base Gemma 4)"),
    ("assistant_gemma4_dynamic_few_shots", "Dynamic Few-Shot (Base Gemma 4)"),
    ("assistant_gemma4_merged_adapter_8bit", "Fine-Tuned Merged 8-bit Adapter"),
    ("assistant", "Human Reference (Ground Truth Control)"),
]


def load_file_content(path: Path, default: str = "") -> str:
    """Load text file content with fallback."""
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return default


def get_model_snapshot_path(model_name: str) -> str:
    """Resolve model path from Hugging Face offline cache or local directory."""
    if Path(model_name).exists():
        return model_name

    hf_cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_dir = hf_cache_dir / "hub"
    formatted_name = f"models--{model_name.replace('/', '--')}"
    target_dir = hub_dir / formatted_name / "snapshots"

    if target_dir.exists():
        snapshots = [d for d in target_dir.iterdir() if d.is_dir()]
        if snapshots:
            snapshots.sort(key=lambda s: s.stat().st_mtime, reverse=True)
            return str(snapshots[0])

    print(f"[WARN] Local snapshot for '{model_name}' not found under {target_dir}. Using Hugging Face identifier.")
    return model_name


def parse_judge_json(raw_text: str) -> dict[str, Any] | None:
    """Extract and parse strict JSON from model judge response."""
    text = raw_text.strip()
    # Strip markdown code blocks if present
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()
    else:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace : last_brace + 1]

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "rule_adherence" in data:
            return data
    except Exception:
        pass

    # Fallback cleanup: remove trailing commas before closing braces/brackets
    cleaned = re.sub(r",\s*([\}\]])", r"\1", text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return None


def main() -> None:
    overall_start_time = time.time()

    print("=" * 65)
    print("      LLM-as-a-Judge: Llama 4 Scout (FP8 on 4 GPUs)")
    print("=" * 65)
    print(f"[INFO] Judge Model     : {JUDGE_MODEL_NAME}")
    print(f"[INFO] Input Results   : {EVAL_RESULTS_PATH}")
    print(f"[INFO] Output Results  : {JUDGE_RESULTS_PATH}")
    print(f"[INFO] Output Summary  : {JUDGE_SUMMARY_PATH}")
    print(f"[INFO] TP Size         : {TENSOR_PARALLEL_SIZE}")

    if not EVAL_RESULTS_PATH.exists():
        print(f"[ERROR] Input results file '{EVAL_RESULTS_PATH}' does not exist!")
        sys.exit(1)

    with EVAL_RESULTS_PATH.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[INFO] Loaded {len(records)} evaluated samples from '{EVAL_RESULTS_PATH}'.")

    # Exclude non-eval integrity checks if desired or evaluate everything
    eval_records = [r for r in records if r.get("id") not in ("i001", "i002") and r.get("assistant")]
    if not eval_records:
        eval_records = records

    if MAX_EVAL_SAMPLES > 0:
        eval_records = eval_records[:MAX_EVAL_SAMPLES]
        print(f"[INFO] Subsetting to first {len(eval_records)} samples for evaluation (MAX_EVAL_SAMPLES={MAX_EVAL_SAMPLES}).")

    judge_system_prompt = load_file_content(
        JUDGE_SYSTEM_PROMPT_PATH,
        default="Du bist ein unabhängiger Sprachexperte und Evaluator für Leichte Sprache. Antworte ausschließlich in JSON.",
    )
    judge_prompt_template = load_file_content(
        JUDGE_PROMPT_TEMPLATE_PATH,
        default=(
            "### 1. Vorgaben und Regeln für die Übersetzung\n```markdown\n%SYSTEM_PROMPT%\n```\n\n"
            "### 2. Ausgangstext (Standardsprache)\n```text\n%ORIGINAL_TEXT%\n```\n\n"
            "### 3. Menschliche Referenzübersetzung (Ground Truth)\n```text\n%GROUND_TRUTH%\n```\n\n"
            "### 4. Zu bewertende Modellübersetzung (%MODEL_LABEL%)\n```text\n%TRANSLATION%\n```\n\n"
            "### Aufgabe für den Evaluator\nBewerte die Modellübersetzung und antworte ausschließlich im geforderten JSON-Format."
        ),
    )
    translator_system_prompt = load_file_content(
        TRANSLATOR_SYSTEM_PROMPT_PATH,
        default="Du bist ein spezialisierter Übersetzer, der deutsche Sprache in Leichte Sprache übersetzt.",
    )

    # Build pointwise evaluation requests: (sample, candidate_key, candidate_label)
    evaluation_requests = []
    for rec in eval_records:
        user_input = rec.get("user_input", "").strip()
        ground_truth = (rec.get("assistant") or "").strip()

        for cand_key, cand_label in CANDIDATE_MODELS:
            cand_text = (rec.get(cand_key) or "").strip()
            if not cand_text:
                continue

            user_prompt = (
                judge_prompt_template
                .replace("%SYSTEM_PROMPT%", translator_system_prompt)
                .replace("%ORIGINAL_TEXT%", user_input)
                .replace("%GROUND_TRUTH%", ground_truth)
                .replace("%TRANSLATION%", cand_text)
                .replace("%MODEL_LABEL%", cand_label)
            )

            evaluation_requests.append({
                "sample_id": rec["id"],
                "candidate_key": cand_key,
                "candidate_label": cand_label,
                "conversation": [
                    {"role": "system", "content": judge_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            })

    print(f"[INFO] Total pointwise evaluation requests: {len(evaluation_requests)}")

    # Load Model & Tokenizer
    model_path = get_model_snapshot_path(JUDGE_MODEL_NAME)
    print(f"\n[INFO] Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    formatted_prompts = [
        tokenizer.apply_chat_template(
            req["conversation"],
            tokenize=False,
            add_generation_prompt=True,
        )
        for req in evaluation_requests
    ]

    print(f"\n[INFO] Initializing SGLang Engine for Llama 4 Scout (TP={TENSOR_PARALLEL_SIZE})...")
    engine_start = time.time()
    engine = sgl.Engine(
        model_path=model_path,
        tp_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        context_length=MAX_SEQUENCE_LENGTH,
        mem_fraction_static=0.85,
    )
    engine_ready_time = time.time() - engine_start
    print(f"[SUCCESS] SGLang engine ready in {engine_ready_time:.1f}s.")

    # Execute Batch Generation
    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": 2048,
        "skip_special_tokens": True,
    }

    print(f"\n[INFO] Running batch evaluation for {len(formatted_prompts)} items...")
    eval_start = time.time()
    outputs = engine.generate(formatted_prompts, sampling_params)
    eval_elapsed = time.time() - eval_start
    print(f"[SUCCESS] Generation completed in {eval_elapsed:.1f}s ({eval_elapsed / len(formatted_prompts):.2f}s/eval).")

    # Process and Save Detailed Results
    JUDGE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    detailed_results = []
    model_metrics = defaultdict(lambda: {
        "rule_adherence": [],
        "factual_completeness": [],
        "readability": [],
        "comparisons": {"better": 0, "comparable": 0, "worse": 0},
        "count": 0,
    })

    for idx, req in enumerate(evaluation_requests):
        raw_output = outputs[idx].get("text", "") if isinstance(outputs[idx], dict) else getattr(outputs[idx], "text", str(outputs[idx]))
        parsed = parse_judge_json(raw_output)

        entry = {
            "sample_id": req["sample_id"],
            "candidate_key": req["candidate_key"],
            "candidate_label": req["candidate_label"],
            "parsed_judgement": parsed,
            "raw_judge_output": raw_output.strip(),
        }
        detailed_results.append(entry)

        cand_key = req["candidate_key"]
        if parsed:
            m = model_metrics[cand_key]
            m["count"] += 1
            if "rule_adherence" in parsed and isinstance(parsed["rule_adherence"], dict) and "score" in parsed["rule_adherence"]:
                m["rule_adherence"].append(float(parsed["rule_adherence"]["score"]))
            if "factual_completeness" in parsed and isinstance(parsed["factual_completeness"], dict) and "score" in parsed["factual_completeness"]:
                m["factual_completeness"].append(float(parsed["factual_completeness"]["score"]))
            if "readability" in parsed and isinstance(parsed["readability"], dict) and "score" in parsed["readability"]:
                m["readability"].append(float(parsed["readability"]["score"]))

            cmp_cat = parsed.get("ground_truth_comparison", {}).get("relative_to_ground_truth", "").lower()
            if cmp_cat in m["comparisons"]:
                m["comparisons"][cmp_cat] += 1

    with JUDGE_RESULTS_PATH.open("w", encoding="utf-8") as f:
        for item in detailed_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Wrote {len(detailed_results)} detailed judge evaluations to: {JUDGE_RESULTS_PATH}")

    # Build Aggregate Summary
    summary_data: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judge_model": JUDGE_MODEL_NAME,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "total_samples": len(eval_records),
        "total_evaluations": len(evaluation_requests),
        "evaluation_time_seconds": round(eval_elapsed, 2),
        "models": {},
    }

    print("\n" + "=" * 65)
    print("        LLM-as-a-Judge: Aggregated Evaluation Summary")
    print("=" * 65)

    model_score_averages = {}
    for cand_key, cand_label in CANDIDATE_MODELS:
        data = model_metrics[cand_key]
        n = data["count"]
        if n == 0:
            continue

        avg_rule = round(sum(data["rule_adherence"]) / len(data["rule_adherence"]), 2) if data["rule_adherence"] else None
        avg_fact = round(sum(data["factual_completeness"]) / len(data["factual_completeness"]), 2) if data["factual_completeness"] else None
        avg_read = round(sum(data["readability"]) / len(data["readability"]), 2) if data["readability"] else None

        model_score_averages[cand_key] = {
            "rule_adherence": avg_rule or 0.0,
            "factual_completeness": avg_fact or 0.0,
            "readability": avg_read or 0.0,
        }

        total_cmp = sum(data["comparisons"].values()) or 1
        pct_better = round((data["comparisons"]["better"] / total_cmp) * 100, 1)
        pct_comparable = round((data["comparisons"]["comparable"] / total_cmp) * 100, 1)
        pct_worse = round((data["comparisons"]["worse"] / total_cmp) * 100, 1)

        summary_data["models"][cand_key] = {
            "label": cand_label,
            "evaluated_samples": n,
            "average_scores": {
                "rule_adherence": avg_rule,
                "factual_completeness": avg_fact,
                "readability": avg_read,
            },
            "vs_ground_truth_percent": {
                "better": pct_better,
                "comparable": pct_comparable,
                "worse": pct_worse,
            },
        }

        print(f"\n* Model: {cand_label}")
        print(f"  - Rule Adherence      : {avg_rule} / 5.0")
        print(f"  - Factual Completeness: {avg_fact} / 5.0")
        print(f"  - Readability & Tone  : {avg_read} / 5.0")
        print(f"  - vs. Ground Truth    : {pct_better}% Better | {pct_comparable}% Comparable | {pct_worse}% Worse")

    # Dynamic Pareto Frontier computation across the 3 independent axes
    pareto_frontier = []
    for m1, s1 in model_score_averages.items():
        dominated = False
        for m2, s2 in model_score_averages.items():
            if m1 == m2:
                continue
            r_ge = s2["rule_adherence"] >= s1["rule_adherence"]
            f_ge = s2["factual_completeness"] >= s1["factual_completeness"]
            e_ge = s2["readability"] >= s1["readability"]
            strictly_better = (
                s2["rule_adherence"] > s1["rule_adherence"]
                or s2["factual_completeness"] > s1["factual_completeness"]
                or s2["readability"] > s1["readability"]
            )
            if r_ge and f_ge and e_ge and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_frontier.append(m1)

    summary_data["pareto_frontier"] = pareto_frontier
    print(f"\n[INFO] Pareto Frontier Models (Non-dominated): {pareto_frontier}")

    with JUDGE_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"[SUCCESS] Wrote benchmark summary to: {JUDGE_SUMMARY_PATH}")
    print(f"[SUCCESS] Total execution time: {time.time() - overall_start_time:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
