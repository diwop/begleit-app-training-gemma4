#!/usr/bin/env python3
"""
Runs baseline, dynamic few-shot, and fine-tuned merged adapter evaluation for Gemma 4 26B-A4B on data/dataset_eval.jsonl using SGLang.
Evaluates the model across five techniques:
1. Standard zero-shot generation on base model (without thinking: enable_thinking=False)
2. Thinking-enabled generation on base model (with thinking: enable_thinking=True, T=1.0, top_p=0.95, top_k=64)
3. Dynamic Few-Shot generation on base model (2 semantically closest training examples retrieved via multilingual-e5-base)
4. Fine-Tuned Merged Adapter generation WITHOUT thinking (enable_thinking=False)
5. Fine-Tuned Merged Adapter generation WITH thinking (enable_thinking=True, T=1.0, top_p=0.95, top_k=64)

Calculates German readability metrics (FRE and WSTF) via textstat for all I/O texts.
"""

from __future__ import annotations

import gc
import os
import json
from pathlib import Path
import re
import sys
import time

# Enforce offline mode and cluster stability on GPU nodes
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

import torch

try:
    import sglang as sgl
    from transformers import AutoTokenizer
except ImportError:
    print("[ERROR] SGLang is not installed in the current environment.", file=sys.stderr)
    print("[INFO] Please run this script inside the SGLang container (images/sglang_sandbox).", file=sys.stderr)
    sys.exit(1)

try:
    import textstat
    textstat.set_lang("de")
except ImportError:
    textstat = None
    print("[WARNING] 'textstat' is not installed. Text readability metrics will default to 0.0.", file=sys.stderr)

from dynamic_few_shots import (
    DynamicFewShotIndex,
    build_dynamic_few_shot_user_prompt,
    get_fitting_few_shot_examples,
)

# Context length and token budgets for SGLang engine
MAX_SEQUENCE_LENGTH = 32768
MAX_NEW_TOKENS = 8192
MAX_INPUT_TOKENS = MAX_SEQUENCE_LENGTH - MAX_NEW_TOKENS - 512

MAX_EVAL_SAMPLES = int(os.environ.get("MAX_EVAL_SAMPLES", "8"))

BASE_MODEL_NAME = "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
MERGED_MODEL_PATH = Path(os.environ.get("MERGED_MODEL", "local/models/gemma-4-26b-a4b-it-fp8"))
EVAL_DATA_PATH = Path("data/dataset_eval.jsonl")
TRAIN_DATA_PATH = Path("data/dataset_train.jsonl")
RESULTS_OUTPUT_PATH = Path(os.environ.get("EVAL_RESULTS_OUTPUT", "data/results.jsonl"))


def extract_gemma4_reasoning(text: str) -> tuple[str, str]:
    """
    Extracts reasoning trace and final clean text from Gemma 4 thinking output.
    Handles:
      - Standard Gemma 4 channel tokens: <|channel>thought\n...<channel|>\n...
      - Alternative delimiters: <|thought|> ... </thought>
    """
    pattern_gemma4 = r"<\|channel>thought\s*(.*?)(?:<channel\|>|<\|channel\|>|$)"
    match = re.search(pattern_gemma4, text, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        clean_text = text[match.end():].strip()
        clean_text = re.sub(r"<\|?[a-zA-Z0-9_]+\|?>", "", clean_text).strip()
        return reasoning, clean_text

    pattern_general = r"(?:<\|thought\|>|<\|channel\|>thought)\s*(.*?)(?:</thought>|<\|channel\|>|<channel\|>|$)"
    match = re.search(pattern_general, text, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        clean_text = text[match.end():].strip()
        clean_text = re.sub(r"<\|?[a-zA-Z0-9_]+\|?>", "", clean_text).strip()
        return reasoning, clean_text

    clean_text = re.sub(r"<\|?[a-zA-Z0-9_]+\|?>", "", text).strip()
    return "", clean_text


def get_raw_metrics(text: str) -> dict[str, float]:
    """Calculates German textstat metrics and returns rounded raw floats."""
    if not text or not text.strip() or textstat is None:
        return {"fre": 0.0, "wstf": 0.0}

    try:
        fre = round(float(textstat.flesch_reading_ease(text)), 1)
    except Exception:
        fre = 0.0

    try:
        wstf = round(float(textstat.wiener_sachtextformel(text, 1)), 1)
    except Exception:
        wstf = 0.0

    return {"fre": fre, "wstf": wstf}


def get_model_snapshot_path(model_name: str) -> str:
    """Get snapshot directory for model from Hugging Face cache or fail fast."""
    if Path(model_name).exists():
        return model_name

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_folder = "models--" + model_name.replace("/", "--")
    snapshots_dir = hf_home / "hub" / repo_folder / "snapshots"

    if not snapshots_dir.exists():
        print(f"[ERROR] Hugging Face cache directory not found at: {snapshots_dir}", file=sys.stderr)
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

    resolved_path = str(snapshots[0])
    print(f"[INFO] Resolved local model snapshot: {resolved_path}")
    return resolved_path


def load_raw_standardsprache(doc_id: str, raw_dir: Path = Path("data/raw")) -> str:
    """Load raw Standardsprache text from data/raw/{id}_Standardsprache.txt."""
    raw_file = raw_dir / f"{doc_id}_Standardsprache.txt"
    if raw_file.exists():
        return raw_file.read_text(encoding="utf-8").strip()
    return ""


def load_eval_data(path: Path) -> list[dict[str, str]]:
    """Load evaluation samples from JSONL."""
    if not path.exists():
        print(f"[ERROR] Evaluation dataset not found at {path}", file=sys.stderr)
        print("[INFO] Run 'dvc repro' or 'python3 src-train/prepare_data.py' first.", file=sys.stderr)
        sys.exit(1)

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def get_integrity_checks(default_system_prompt: str) -> list[dict[str, str | None]]:
    """Build the two initial integrity check prompts."""
    # i001: General assistant, direct question, no prompt template
    i001 = {
        "id": "i001",
        "system": "Du bist ein hilfreicher Assistent.",
        "user": "Warum ist der Himmel blau?",
        "assistant": None,
    }

    # i002: Standard system prompt, prompt template wrapped
    prompt_template_path = Path("prompts/prompt-template.md")
    if prompt_template_path.exists():
        template = prompt_template_path.read_text(encoding="utf-8").strip()
        i002_user = template.replace("%INPUT%", "Warum ist der Himmel blau und nicht schwarz?")
    else:
        i002_user = (
            "### Text in Standardsprache:\n```input\nWarum ist der Himmel blau und nicht schwarz?\n```\n\n"
            "### Aufgabe:\nÜbersetze den Text in Leichte Sprache. Beachte alle Regeln für Leichte Sprache."
        )

    i002 = {
        "id": "i002",
        "system": default_system_prompt,
        "user": i002_user,
        "assistant": None,
    }
    return [i001, i002]


def extract_output_text(output_obj: object) -> str:
    """Extract string response from SGLang output item."""
    if isinstance(output_obj, dict):
        return output_obj.get("text", "").strip()
    if hasattr(output_obj, "text"):
        return output_obj.text.strip()
    return str(output_obj).strip()


def main() -> None:
    overall_start_time = time.time()

    print("=" * 60)
    print("      Gemma 4 Evaluation (Baseline, Few-Shot & Merged Adapter)")
    print("=" * 60)
    print(f"[INFO] Base Model       : {BASE_MODEL_NAME}")
    print(f"[INFO] Merged Model Path: {MERGED_MODEL_PATH}")
    print(f"[INFO] Input Dataset    : {EVAL_DATA_PATH}")
    print(f"[INFO] Output Results   : {RESULTS_OUTPUT_PATH}")

    gpu_count = torch.cuda.device_count()
    tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    print(f"[INFO] Detected GPUs    : {gpu_count} (Using Tensor Parallel Size: {tensor_parallel_size})")

    records = load_eval_data(EVAL_DATA_PATH)
    default_system_prompt = records[0]["system"] if records else "Du bist ein hilfreicher Assistent für Leichte Sprache."
    integrity_records = get_integrity_checks(default_system_prompt)

    if MAX_EVAL_SAMPLES > 0:
        dataset_subset_count = max(0, MAX_EVAL_SAMPLES - len(integrity_records))
        records = integrity_records + records[:dataset_subset_count]
        print(f"[INFO] Evaluating {len(records)} total samples ({len(integrity_records)} integrity checks + {dataset_subset_count} dataset samples).")
    else:
        records = integrity_records + records
        print(f"[INFO] Loaded {len(records)} total evaluation samples (including {len(integrity_records)} integrity checks).")

    # 1. Resolve local offline model snapshot path & Tokenizer
    model_path = get_model_snapshot_path(BASE_MODEL_NAME)
    print(f"\n[INFO] Loading tokenizer from snapshot: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. Build prompt conversations for all 3 modes
    zero_shot_conversations = [
        [
            {"role": "system", "content": rec["system"]},
            {"role": "user", "content": rec["user"]},
        ]
        for rec in records
    ]

    print(f"\n[INFO] Constructing Dynamic Few-Shot prompts with token budget (max input: {MAX_INPUT_TOKENS} tokens)...")
    few_shot_conversations = []
    for rec in records:
        if rec["id"] in ("i001", "i002"):
            few_shot_user_prompt = rec["user"]
        else:
            raw_user_in = load_raw_standardsprache(rec["id"]) or rec["user"]
            fitting_examples = get_fitting_few_shot_examples(
                query=raw_user_in,
                tokenizer=tokenizer,
                max_input_tokens=MAX_INPUT_TOKENS,
                max_examples=2,
                dataset_path=TRAIN_DATA_PATH,
            )
            few_shot_user_prompt = build_dynamic_few_shot_user_prompt(raw_user_in, fitting_examples)
            token_count = len(tokenizer.encode(few_shot_user_prompt, add_special_tokens=False))
            print(f"[INFO] Sample '{rec['id']}': retrieved {len(fitting_examples)} few-shot demonstrations ({token_count} tokens).")

        few_shot_conversations.append([
            {"role": "system", "content": rec["system"]},
            {"role": "user", "content": few_shot_user_prompt},
        ])

    prompts_no_thinking = [
        tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for conv in zero_shot_conversations
    ]

    prompts_thinking = [
        tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for conv in zero_shot_conversations
    ]

    prompts_few_shots = [
        tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for conv in few_shot_conversations
    ]

    # Initialize SGLang Engine
    print(f"\n[INFO] Initializing SGLang engine and loading model weights from: {model_path}")
    print(f"[INFO] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    model_load_start = time.time()

    engine = sgl.Engine(
        model_path=model_path,
        tp_size=tensor_parallel_size,
        trust_remote_code=True,
        mem_fraction_static=0.85,
        context_length=MAX_SEQUENCE_LENGTH,
    )

    model_load_elapsed = time.time() - model_load_start
    print(f"[SUCCESS] SGLang engine & weights ready in {model_load_elapsed:.1f}s ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")

    # Sampling parameters
    sampling_params_no_thinking = {
        "temperature": 0.0,
        "max_new_tokens": 4096,
        "skip_special_tokens": True,
    }

    sampling_params_thinking = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_new_tokens": 8192,
        "skip_special_tokens": False,
    }

    # STEP 1: Zero-Shot WITHOUT thinking
    print("=" * 60)
    print(f"[STEP 1/4] Running Zero-Shot WITHOUT thinking on Base Model for {len(records)} samples...")
    print(f"[INFO] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    step1_start = time.time()

    no_thinking_outputs = engine.generate(prompts_no_thinking, sampling_params_no_thinking)

    step1_elapsed = time.time() - step1_start
    print(f"[SUCCESS] Step 1 completed in {step1_elapsed:.1f}s ({step1_elapsed/len(records):.2f}s/sample)\n")

    # STEP 2: Zero-Shot WITH thinking
    print("=" * 60)
    print(f"[STEP 2/4] Running Zero-Shot WITH thinking on Base Model for {len(records)} samples...")
    print(f"[INFO] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    step2_start = time.time()

    thinking_outputs = engine.generate(prompts_thinking, sampling_params_thinking)

    step2_elapsed = time.time() - step2_start
    print(f"[SUCCESS] Step 2 completed in {step2_elapsed:.1f}s ({step2_elapsed/len(records):.2f}s/sample)\n")

    # STEP 3: Dynamic Few-Shot WITH thinking (2 semantically closest demonstrations)
    print("=" * 60)
    print(f"[STEP 3/4] Running Dynamic Few-Shot WITH thinking on Base Model for {len(records)} samples...")
    print(f"[INFO] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    step3_start = time.time()

    few_shot_outputs = engine.generate(prompts_few_shots, sampling_params_thinking)

    step3_elapsed = time.time() - step3_start
    print(f"[SUCCESS] Step 3 completed in {step3_elapsed:.1f}s ({step3_elapsed/len(records):.2f}s/sample)\n")

    # STEP 4 & 5: Fine-Tuned Merged Adapter Evaluation
    merged_outputs = None
    merged_thinking_outputs = None
    if MERGED_MODEL_PATH.exists():
        # Release base engine GPU memory before loading merged model
        engine.shutdown()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        print(f"\n[INFO] Initializing SGLang engine with Fine-Tuned Merged Model from: {MERGED_MODEL_PATH}")
        merged_engine = sgl.Engine(
            model_path=str(MERGED_MODEL_PATH),
            tp_size=tensor_parallel_size,
            trust_remote_code=True,
            mem_fraction_static=0.85,
            context_length=MAX_SEQUENCE_LENGTH,
        )

        # STEP 4: Merged Model WITHOUT thinking
        print("=" * 60)
        print(f"[STEP 4/5] Running Fine-Tuned Merged Model WITHOUT thinking for {len(records)} samples...")
        print(f"[INFO] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        step4_start = time.time()

        merged_outputs = merged_engine.generate(prompts_no_thinking, sampling_params_no_thinking)

        step4_elapsed = time.time() - step4_start
        print(f"[SUCCESS] Step 4 completed in {step4_elapsed:.1f}s ({step4_elapsed/len(records):.2f}s/sample)\n")

        # STEP 5: Merged Model WITH thinking
        print("=" * 60)
        print(f"[STEP 5/5] Running Fine-Tuned Merged Model WITH thinking (enable_thinking=True, T=1.0, top_p=0.95) for {len(records)} samples...")
        print(f"[INFO] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        step5_start = time.time()

        merged_thinking_outputs = merged_engine.generate(prompts_thinking, sampling_params_thinking)

        step5_elapsed = time.time() - step5_start
        print(f"[SUCCESS] Step 5 completed in {step5_elapsed:.1f}s ({step5_elapsed/len(records):.2f}s/sample)\n")

        merged_engine.shutdown()
    else:
        print("=" * 60)
        print(f"[INFO] [STEP 4/5 & 5/5] Fine-tuned merged model not found at '{MERGED_MODEL_PATH}'.")
        print("[INFO] Skipping Passes 4 & 5 (run scripts/merge_and_quantize.sh first to produce merged FP8 model).\n")
        engine.shutdown()

    # 5. Calculate metrics and assemble results
    print("=" * 60)
    print(f"[INFO] Calculating German readability metrics (FRE & WSTF)...")
    RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    fre_scores = {
        "input": [],
        "ground_truth": [],
        "gemma4": [],
        "gemma4_thinking": [],
        "gemma4_dynamic_few_shots": [],
        "gemma4_merged": [],
        "gemma4_merged_thinking": [],
    }
    wstf_scores = {
        "input": [],
        "ground_truth": [],
        "gemma4": [],
        "gemma4_thinking": [],
        "gemma4_dynamic_few_shots": [],
        "gemma4_merged": [],
        "gemma4_merged_thinking": [],
    }

    for idx, rec in enumerate(records):
        raw_no_thinking = extract_output_text(no_thinking_outputs[idx])
        out_no_thinking = re.sub(r"<\|?[a-zA-Z0-9_]+\|?>", "", raw_no_thinking).strip()

        raw_thinking_output = extract_output_text(thinking_outputs[idx])
        reasoning_trace, out_thinking = extract_gemma4_reasoning(raw_thinking_output)

        raw_few_shots = extract_output_text(few_shot_outputs[idx])
        few_shots_reasoning, out_few_shots = extract_gemma4_reasoning(raw_few_shots)

        raw_user_input = load_raw_standardsprache(rec["id"]) if rec["id"] not in ("i001", "i002") else rec["user"]
        if not raw_user_input:
            raw_user_input = rec["user"]

        user_metrics = get_raw_metrics(raw_user_input)
        assistant_metrics = get_raw_metrics(rec["assistant"]) if rec["assistant"] is not None else None
        gemma4_metrics = get_raw_metrics(out_no_thinking)
        gemma4_thinking_metrics = get_raw_metrics(out_thinking)
        gemma4_few_shots_metrics = get_raw_metrics(out_few_shots)

        out_merged = None
        merged_reasoning = None
        gemma4_merged_metrics = None

        if merged_outputs is not None:
            raw_merged = extract_output_text(merged_outputs[idx])
            merged_reasoning, out_merged = extract_gemma4_reasoning(raw_merged)
            gemma4_merged_metrics = get_raw_metrics(out_merged)

        out_merged_thinking = None
        merged_thinking_reasoning = None
        gemma4_merged_thinking_metrics = None

        if merged_thinking_outputs is not None:
            raw_merged_thinking = extract_output_text(merged_thinking_outputs[idx])
            merged_thinking_reasoning, out_merged_thinking = extract_gemma4_reasoning(raw_merged_thinking)
            gemma4_merged_thinking_metrics = get_raw_metrics(out_merged_thinking)

        if assistant_metrics is not None:
            fre_scores["input"].append(user_metrics["fre"])
            fre_scores["ground_truth"].append(assistant_metrics["fre"])
            fre_scores["gemma4"].append(gemma4_metrics["fre"])
            fre_scores["gemma4_thinking"].append(gemma4_thinking_metrics["fre"])
            fre_scores["gemma4_dynamic_few_shots"].append(gemma4_few_shots_metrics["fre"])
            if gemma4_merged_metrics is not None:
                fre_scores["gemma4_merged"].append(gemma4_merged_metrics["fre"])
            if gemma4_merged_thinking_metrics is not None:
                fre_scores["gemma4_merged_thinking"].append(gemma4_merged_thinking_metrics["fre"])

            wstf_scores["input"].append(user_metrics["wstf"])
            wstf_scores["ground_truth"].append(assistant_metrics["wstf"])
            wstf_scores["gemma4"].append(gemma4_metrics["wstf"])
            wstf_scores["gemma4_thinking"].append(gemma4_thinking_metrics["wstf"])
            wstf_scores["gemma4_dynamic_few_shots"].append(gemma4_few_shots_metrics["wstf"])
            if gemma4_merged_metrics is not None:
                wstf_scores["gemma4_merged"].append(gemma4_merged_metrics["wstf"])
            if gemma4_merged_thinking_metrics is not None:
                wstf_scores["gemma4_merged_thinking"].append(gemma4_merged_thinking_metrics["wstf"])

        result_entry = {
            "id": rec["id"],
            "system": rec["system"],
            "user_input": raw_user_input,
            "user_input_metrics": user_metrics,
            "user": rec["user"],
            "user_dynamic_few_shots": few_shot_conversations[idx][1]["content"],
            "assistant": rec["assistant"],
            "assistant_metrics": assistant_metrics,
            "assistant_gemma4": out_no_thinking,
            "assistant_gemma4_metrics": gemma4_metrics,
            "assistant_gemma4_thinking_reasoning": reasoning_trace,
            "assistant_gemma4_thinking": out_thinking,
            "assistant_gemma4_thinking_metrics": gemma4_thinking_metrics,
            "assistant_gemma4_dynamic_few_shots_reasoning": few_shots_reasoning,
            "assistant_gemma4_dynamic_few_shots": out_few_shots,
            "assistant_gemma4_dynamic_few_shots_metrics": gemma4_few_shots_metrics,
            "assistant_gemma4_merged_reasoning": merged_reasoning,
            "assistant_gemma4_merged": out_merged,
            "assistant_gemma4_merged_metrics": gemma4_merged_metrics,
            "assistant_gemma4_merged_thinking_reasoning": merged_thinking_reasoning,
            "assistant_gemma4_merged_thinking": out_merged_thinking,
            "assistant_gemma4_merged_thinking_metrics": gemma4_merged_thinking_metrics,
        }
        results.append(result_entry)

    with RESULTS_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    overall_elapsed = time.time() - overall_start_time

    print(f"[SUCCESS] Wrote {len(results)} evaluated results with textstat metrics to: {RESULTS_OUTPUT_PATH}")
    print("=" * 60)
    print("      Evaluation Summary Metrics (Dataset Averages)")
    print("=" * 60)
    num_eval_ds = len(fre_scores["ground_truth"])
    if num_eval_ds > 0:
        avg_in_fre = sum(fre_scores["input"]) / num_eval_ds
        avg_gt_fre = sum(fre_scores["ground_truth"]) / num_eval_ds
        avg_g4_fre = sum(fre_scores["gemma4"]) / num_eval_ds
        avg_g4_think_fre = sum(fre_scores["gemma4_thinking"]) / num_eval_ds
        avg_g4_few_fre = sum(fre_scores["gemma4_dynamic_few_shots"]) / num_eval_ds

        avg_in_wstf = sum(wstf_scores["input"]) / num_eval_ds
        avg_gt_wstf = sum(wstf_scores["ground_truth"]) / num_eval_ds
        avg_g4_wstf = sum(wstf_scores["gemma4"]) / num_eval_ds
        avg_g4_think_wstf = sum(wstf_scores["gemma4_thinking"]) / num_eval_ds
        avg_g4_few_wstf = sum(wstf_scores["gemma4_dynamic_few_shots"]) / num_eval_ds

        print(f"  * Input Standardsprache         : FRE = {avg_in_fre:.1f}  |  WSTF = {avg_in_wstf:.1f}")
        print(f"  * Ground Truth (Target)         : FRE = {avg_gt_fre:.1f}  |  WSTF = {avg_gt_wstf:.1f}")
        print(f"  * Gemma 4 (Zero-Shot)           : FRE = {avg_g4_fre:.1f}  |  WSTF = {avg_g4_wstf:.1f}")
        print(f"  * Gemma 4 (With Thinking)       : FRE = {avg_g4_think_fre:.1f}  |  WSTF = {avg_g4_think_wstf:.1f}")
        print(f"  * Gemma 4 (Few-Shots + Thinking): FRE = {avg_g4_few_fre:.1f}  |  WSTF = {avg_g4_few_wstf:.1f}")

        if fre_scores["gemma4_merged"]:
            avg_g4_merged_fre = sum(fre_scores["gemma4_merged"]) / len(fre_scores["gemma4_merged"])
            avg_g4_merged_wstf = sum(wstf_scores["gemma4_merged"]) / len(wstf_scores["gemma4_merged"])
            print(f"  * Gemma 4 (Fine-Tuned Merged)   : FRE = {avg_g4_merged_fre:.1f}  |  WSTF = {avg_g4_merged_wstf:.1f}")

        if fre_scores["gemma4_merged_thinking"]:
            avg_g4_merged_think_fre = sum(fre_scores["gemma4_merged_thinking"]) / len(fre_scores["gemma4_merged_thinking"])
            avg_g4_merged_think_wstf = sum(wstf_scores["gemma4_merged_thinking"]) / len(wstf_scores["gemma4_merged_thinking"])
            print(f"  * Gemma 4 (Merged + Thinking)   : FRE = {avg_g4_merged_think_fre:.1f}  |  WSTF = {avg_g4_merged_think_wstf:.1f}")

    print(f"  * Total Evaluation Time         : {overall_elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
