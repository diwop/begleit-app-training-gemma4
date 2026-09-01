#!/usr/bin/env python3
"""
Runs baseline, dynamic few-shot, and fine-tuned merged FP8 model evaluation for Gemma 4 26B-A4B on data/dataset_eval.jsonl using SGLang.
Evaluates the model across four techniques:
1. Standard zero-shot generation on base model (without thinking: enable_thinking=False)
2. Thinking-enabled generation on base model (with thinking: enable_thinking=True, T=1.0, top_p=0.95, top_k=64)
3. Dynamic Few-Shot generation on base model (2 semantically closest training examples retrieved via multilingual-e5-base, with thinking: enable_thinking=True)
4. Fine-Tuned Merged 8-bit Adapter generation WITH thinking (enable_thinking=True, T=1.0, top_p=0.95, top_k=64)

Calculates German readability metrics (FRE and WSTF) via textstat for all I/O texts and writes results to data/results.jsonl and data/results-metadata.json.
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
    extract_raw_standardsprache,
    get_fitting_few_shot_examples,
)

# Context length and token budgets for SGLang engine
MAX_SEQUENCE_LENGTH = 32768
MAX_NEW_TOKENS = 8192
MAX_INPUT_TOKENS = MAX_SEQUENCE_LENGTH - MAX_NEW_TOKENS - 512

# Default to 0 for full dataset evaluation. Set MAX_EVAL_SAMPLES=8 for smoke test.
MAX_EVAL_SAMPLES = int(os.environ.get("MAX_EVAL_SAMPLES", "0"))

BASE_MODEL_NAME = "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
MERGED_MODEL_PATH = Path(os.environ.get("MERGED_MODEL", "local/models/gemma-4-26b-a4b-it-fp8"))
EVAL_DATA_PATH = Path("data/dataset_eval.jsonl")
TRAIN_DATA_PATH = Path("data/dataset_train.jsonl")
RESULTS_OUTPUT_PATH = Path(os.environ.get("EVAL_RESULTS_OUTPUT", "data/results.jsonl"))
RESULTS_METADATA_PATH = Path(os.environ.get("EVAL_METADATA_OUTPUT", "data/results-metadata.json"))


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


def calculate_speed(outputs, elapsed_seconds: float, tokenizer) -> float:
    """Calculate total generated tokens per second for a batch of outputs."""
    if not outputs or elapsed_seconds <= 0:
        return 0.0
    total_tokens = 0
    for out in outputs:
        text = extract_output_text(out)
        if text:
            total_tokens += len(tokenizer.encode(text, add_special_tokens=False))
    return round(total_tokens / elapsed_seconds, 2)


def get_raw_metrics(text: str | None) -> dict[str, float]:
    """Calculates German textstat metrics and bounds to standard ranges (FRE 0-100, WSTF 1-15)."""
    if not text or not text.strip() or len(text.strip().split()) < 3 or textstat is None:
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

def get_model_snapshot_path(model_name: str, required: bool = True) -> str:
    """Get snapshot directory for model from Hugging Face cache or fail fast if required."""
    if Path(model_name).exists():
        return model_name

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_folder = "models--" + model_name.replace("/", "--")
    snapshots_dir = hf_home / "hub" / repo_folder / "snapshots"

    if not snapshots_dir.exists():
        if required:
            print(f"[ERROR] Hugging Face cache directory not found at: {snapshots_dir}", file=sys.stderr)
            print("[INFO] Please run 'bash scripts/download_models.sh' on the login node first.", file=sys.stderr)
            sys.exit(1)
        return ""

    snapshots = sorted(
        [p for p in snapshots_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not snapshots:
        if required:
            print(f"[ERROR] No snapshot directories found inside: {snapshots_dir}", file=sys.stderr)
            print("[INFO] Please run 'bash scripts/download_models.sh' on the login node first.", file=sys.stderr)
            sys.exit(1)
        return ""

    resolved_path = str(snapshots[0])
    print(f"[INFO] Resolved local model snapshot: {resolved_path}")
    return resolved_path


def load_eval_data(path: Path) -> list[dict[str, str]]:
    """Load evaluation samples from JSONL."""
    if not path.exists():
        print(f"[ERROR] Evaluation dataset not found at {path}", file=sys.stderr)
        print("[INFO] Run 'dvc repro' or 'python3 src-train/prepare_data.py' first.", file=sys.stderr)
        sys.exit(1)

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sys_prompt = ""
            user_prompt = ""
            assistant_prompt = ""
            for msg in entry.get("messages", []):
                role = msg.get("role")
                if role == "system":
                    sys_prompt = msg.get("content", "")
                elif role == "user":
                    user_prompt = msg.get("content", "")
                elif role == "assistant":
                    assistant_prompt = msg.get("content", "")
            records.append({
                "id": entry.get("id", f"sample_{len(records)}"),
                "system": sys_prompt,
                "user": user_prompt,
                "assistant": assistant_prompt,
            })
    return records


def get_integrity_checks(default_system_prompt: str) -> list[dict[str, str]]:
    """Generates standard integrity test prompts."""
    i001 = {
        "id": "i001",
        "system": default_system_prompt,
        "user": "Erzähle einen Witz.",
        "assistant": None,
    }

    train_path = Path("data/dataset_train.jsonl")
    sample_text = None
    if train_path.exists():
        try:
            with train_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        for msg in entry.get("messages", []):
                            if msg.get("role") == "user":
                                raw_extracted = extract_raw_standardsprache(text=msg.get("content", ""), doc_id=entry.get("id", ""))
                                if raw_extracted and len(raw_extracted) > 40:
                                    sample_text = raw_extracted
                                    break
                        if sample_text:
                            break
        except Exception:
            pass

    if sample_text:
        i002_user = (
            f"### Text in Standardsprache:\n```input\n{sample_text}\n```\n\n"
            "### Aufgabe:\nÜbersetze den Text in Leichte Sprache. Beachte alle Regeln für Leichte Sprache."
        )
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
    print("      Gemma 4 Evaluation: FP8 Base, Few-Shot & Merged 8-bit")
    print("=" * 60)
    print(f"[INFO] Base FP8 Model   : {BASE_MODEL_NAME}")
    print(f"[INFO] Merged Model Path: {MERGED_MODEL_PATH}")
    print(f"[INFO] Input Dataset    : {EVAL_DATA_PATH}")
    print(f"[INFO] Output Results   : {RESULTS_OUTPUT_PATH}")
    print(f"[INFO] Output Metadata  : {RESULTS_METADATA_PATH}")

    gpu_count = torch.cuda.device_count()
    tp_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    print(f"[INFO] Detected GPUs    : {gpu_count} (Using Tensor Parallel Size: {tp_size})")

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

    # Resolve local offline model snapshot path & Tokenizer
    model_path = get_model_snapshot_path(BASE_MODEL_NAME)
    print(f"\n[INFO] Loading tokenizer from snapshot: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Build prompt conversations for all modes
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
            raw_user_in = extract_raw_standardsprache(text=rec.get("user", ""), doc_id=rec["id"])
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

    sampling_params_thinking = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_new_tokens": 8192,
        "skip_special_tokens": False,
    }
    sampling_params_no_thinking = {
        "temperature": 0.0,
        "max_new_tokens": 8192,
        "skip_special_tokens": False,
    }

    # =========================================================================
    # STEPS 1 to 3: Base FP8 Model (Zero-Shot, Thinking, Dynamic Few-Shot)
    # =========================================================================
    print("=" * 60)
    print(f"[INFO] Initializing SGLang engine for Base FP8 Model ({BASE_MODEL_NAME})...")
    engine = sgl.Engine(
        model_path=model_path,
        tp_size=tp_size,
        trust_remote_code=True,
        mem_fraction_static=0.85,
        context_length=MAX_SEQUENCE_LENGTH,
        watchdog_timeout=86400,
        dist_timeout=7200,
    )

    # 1. Base model without thinking
    print("=" * 60)
    print(f"[STEP 1/4] Running Base Model WITHOUT thinking (enable_thinking=False, T=0.0) for {len(records)} samples...")
    print(f"[INFO] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    step1_start = time.time()
    no_thinking_outputs = engine.generate(prompts_no_thinking, sampling_params_no_thinking)
    step1_elapsed = time.time() - step1_start
    print(f"[SUCCESS] Step 1/4 (Zero-Shot) completed in {step1_elapsed:.1f}s ({step1_elapsed/len(records):.2f}s/sample)\n")

    # 2. Base model with thinking
    print("=" * 60)
    print(f"[STEP 2/4] Running Base Model WITH thinking (enable_thinking=True, T=1.0, top_p=0.95) for {len(records)} samples...")
    print(f"[INFO] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    step2_start = time.time()
    thinking_outputs = engine.generate(prompts_thinking, sampling_params_thinking)
    step2_elapsed = time.time() - step2_start
    print(f"[SUCCESS] Step 2/4 (With Thinking) completed in {step2_elapsed:.1f}s ({step2_elapsed/len(records):.2f}s/sample)\n")

    # 3. Base model with dynamic few shots and thinking
    print("=" * 60)
    print(f"[STEP 3/4] Running Base Model WITH Dynamic Few-Shots & thinking for {len(records)} samples...")
    print(f"[INFO] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    step3_start = time.time()
    few_shot_outputs = engine.generate(prompts_few_shots, sampling_params_thinking)
    step3_elapsed = time.time() - step3_start
    print(f"[SUCCESS] Step 3/4 (Few-Shots + Thinking) completed in {step3_elapsed:.1f}s ({step3_elapsed/len(records):.2f}s/sample)\n")

    engine.shutdown()
    del engine
    time.sleep(3)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # =========================================================================
    # STEP 4: Fine-Tuned Merged 8-bit Model Evaluation
    # =========================================================================
    merged_adapter_8bit_outputs = None
    step4_elapsed = None
    if MERGED_MODEL_PATH.exists():
        print("=" * 60)
        print(f"[INFO] Initializing SGLang engine for Fine-Tuned Merged 8-bit Model ({MERGED_MODEL_PATH})...")
        merged_engine = sgl.Engine(
            model_path=str(MERGED_MODEL_PATH),
            tp_size=tp_size,
            trust_remote_code=True,
            mem_fraction_static=0.85,
            context_length=MAX_SEQUENCE_LENGTH,
            watchdog_timeout=86400,
            dist_timeout=7200,
        )

        print("=" * 60)
        print(f"[STEP 4/4] Running Fine-Tuned Merged 8-bit Model WITH thinking (enable_thinking=True, T=1.0, top_p=0.95) for {len(records)} samples...")
        print(f"[INFO] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        step4_start = time.time()

        merged_adapter_8bit_outputs = merged_engine.generate(prompts_thinking, sampling_params_thinking)

        step4_elapsed = time.time() - step4_start
        print(f"[SUCCESS] Step 4/4 (Merged 8-bit + Think) completed in {step4_elapsed:.1f}s ({step4_elapsed/len(records):.2f}s/sample)\n")

        merged_engine.shutdown()
        del merged_engine
        time.sleep(3)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    else:
        print("=" * 60)
        print(f"[INFO] [STEP 4/4] Fine-tuned merged model not found at '{MERGED_MODEL_PATH}'. Skipping Pass 4.\n")

    # =========================================================================
    # Assemble Results & Calculate Readability Metrics
    # =========================================================================
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

    for idx, rec in enumerate(records):
        out_no_thinking = None
        gemma4_metrics = None
        if no_thinking_outputs is not None:
            raw_no_thinking = extract_output_text(no_thinking_outputs[idx])
            out_no_thinking = re.sub(r"<\|?[a-zA-Z0-9_]+\|?>", "", raw_no_thinking).strip()
            gemma4_metrics = get_raw_metrics(out_no_thinking)

        reasoning_trace = None
        out_thinking = None
        gemma4_thinking_metrics = None
        if thinking_outputs is not None:
            raw_thinking_output = extract_output_text(thinking_outputs[idx])
            reasoning_trace, out_thinking = extract_gemma4_reasoning(raw_thinking_output)
            gemma4_thinking_metrics = get_raw_metrics(out_thinking)

        few_shots_reasoning = None
        out_few_shots = None
        gemma4_few_shots_metrics = None
        if few_shot_outputs is not None:
            raw_few_shots = extract_output_text(few_shot_outputs[idx])
            few_shots_reasoning, out_few_shots = extract_gemma4_reasoning(raw_few_shots)
            gemma4_few_shots_metrics = get_raw_metrics(out_few_shots)

        # Extract clean raw Standardsprache input text without prompt template wrapper
        if rec["id"] == "i001":
            raw_user_input = rec["user"]
        elif rec["id"] == "i002":
            raw_user_input = extract_raw_standardsprache(rec["user"])
        else:
            raw_user_input = extract_raw_standardsprache(text=rec.get("user", ""), doc_id=rec["id"])

        user_metrics = get_raw_metrics(raw_user_input)
        assistant_metrics = get_raw_metrics(rec["assistant"]) if rec["assistant"] is not None else None

        out_merged_adapter_8bit = None
        merged_adapter_8bit_reasoning = None
        gemma4_merged_adapter_8bit_metrics = None

        if merged_adapter_8bit_outputs is not None:
            raw_merged_8bit = extract_output_text(merged_adapter_8bit_outputs[idx])
            merged_adapter_8bit_reasoning, out_merged_adapter_8bit = extract_gemma4_reasoning(raw_merged_8bit)
            gemma4_merged_adapter_8bit_metrics = get_raw_metrics(out_merged_adapter_8bit)

        if assistant_metrics is not None:
            fre_scores["input"].append(user_metrics["fre"])
            fre_scores["ground_truth"].append(assistant_metrics["fre"])
            if gemma4_metrics is not None:
                fre_scores["gemma4"].append(gemma4_metrics["fre"])
            if gemma4_thinking_metrics is not None:
                fre_scores["gemma4_thinking"].append(gemma4_thinking_metrics["fre"])
            if gemma4_few_shots_metrics is not None:
                fre_scores["gemma4_dynamic_few_shots"].append(gemma4_few_shots_metrics["fre"])
            if gemma4_merged_adapter_8bit_metrics is not None:
                fre_scores["gemma4_merged_adapter_8bit"].append(gemma4_merged_adapter_8bit_metrics["fre"])

            wstf_scores["input"].append(user_metrics["wstf"])
            wstf_scores["ground_truth"].append(assistant_metrics["wstf"])
            if gemma4_metrics is not None:
                wstf_scores["gemma4"].append(gemma4_metrics["wstf"])
            if gemma4_thinking_metrics is not None:
                wstf_scores["gemma4_thinking"].append(gemma4_thinking_metrics["wstf"])
            if gemma4_few_shots_metrics is not None:
                wstf_scores["gemma4_dynamic_few_shots"].append(gemma4_few_shots_metrics["wstf"])
            if gemma4_merged_adapter_8bit_metrics is not None:
                wstf_scores["gemma4_merged_adapter_8bit"].append(gemma4_merged_adapter_8bit_metrics["wstf"])

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
            "assistant_gemma4_merged_adapter_8bit_reasoning": merged_adapter_8bit_reasoning,
            "assistant_gemma4_merged_adapter_8bit": out_merged_adapter_8bit,
            "assistant_gemma4_merged_adapter_8bit_metrics": gemma4_merged_adapter_8bit_metrics,
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
        avg_in_wstf = sum(wstf_scores["input"]) / num_eval_ds
        avg_gt_wstf = sum(wstf_scores["ground_truth"]) / num_eval_ds

        print(f"  * Input Standardsprache         : FRE = {avg_in_fre:.1f}  |  WSTF = {avg_in_wstf:.1f}")
        print(f"  * Ground Truth (Target)         : FRE = {avg_gt_fre:.1f}  |  WSTF = {avg_gt_wstf:.1f}")

        if fre_scores["gemma4"]:
            avg_g4_fre = sum(fre_scores["gemma4"]) / len(fre_scores["gemma4"])
            avg_g4_wstf = sum(wstf_scores["gemma4"]) / len(wstf_scores["gemma4"])
            print(f"  * Gemma 4 (Zero-Shot)           : FRE = {avg_g4_fre:.1f}  |  WSTF = {avg_g4_wstf:.1f}")

        if fre_scores["gemma4_thinking"]:
            avg_g4_think_fre = sum(fre_scores["gemma4_thinking"]) / len(fre_scores["gemma4_thinking"])
            avg_g4_think_wstf = sum(wstf_scores["gemma4_thinking"]) / len(wstf_scores["gemma4_thinking"])
            print(f"  * Gemma 4 (With Thinking)       : FRE = {avg_g4_think_fre:.1f}  |  WSTF = {avg_g4_think_wstf:.1f}")

        if fre_scores["gemma4_dynamic_few_shots"]:
            avg_g4_few_fre = sum(fre_scores["gemma4_dynamic_few_shots"]) / len(fre_scores["gemma4_dynamic_few_shots"])
            avg_g4_few_wstf = sum(wstf_scores["gemma4_dynamic_few_shots"]) / len(wstf_scores["gemma4_dynamic_few_shots"])
            print(f"  * Gemma 4 (Few-Shots + Thinking): FRE = {avg_g4_few_fre:.1f}  |  WSTF = {avg_g4_few_wstf:.1f}")

        if fre_scores["gemma4_merged_adapter_8bit"]:
            avg_g4_merged_8bit_fre = sum(fre_scores["gemma4_merged_adapter_8bit"]) / len(fre_scores["gemma4_merged_adapter_8bit"])
            avg_g4_merged_8bit_wstf = sum(wstf_scores["gemma4_merged_adapter_8bit"]) / len(wstf_scores["gemma4_merged_adapter_8bit"])
            print(f"  * Gemma 4 (Merged 8-bit + Think): FRE = {avg_g4_merged_8bit_fre:.1f}  |  WSTF = {avg_g4_merged_8bit_wstf:.1f}")

    print(f"  * Total Evaluation Time         : {overall_elapsed:.1f}s")
    print("=" * 60)

    # Calculate throughput speeds and write results-metadata.json
    assistant_gemma4_speed = calculate_speed(no_thinking_outputs, step1_elapsed or 0, tokenizer)
    assistant_gemma4_thinking_speed = calculate_speed(thinking_outputs, step2_elapsed or 0, tokenizer)
    assistant_gemma4_dynamic_few_shots_speed = calculate_speed(few_shot_outputs, step3_elapsed or 0, tokenizer)
    assistant_gemma4_merged_adapter_8bit_speed = calculate_speed(merged_adapter_8bit_outputs, step4_elapsed or 0, tokenizer)

    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": len(results),
        "total_evaluation_time_seconds": round(overall_elapsed, 2),
        "phase_elapsed_seconds": {
            "step1_base_zero_shot": round(step1_elapsed, 2) if step1_elapsed is not None else None,
            "step2_base_thinking": round(step2_elapsed, 2) if step2_elapsed is not None else None,
            "step3_base_few_shots": round(step3_elapsed, 2) if step3_elapsed is not None else None,
            "step4_merged_adapter_8bit": round(step4_elapsed, 2) if step4_elapsed is not None else None,
        },
        "model_speeds_tokens_per_second": {
            "assistant_gemma4_speed": assistant_gemma4_speed,
            "assistant_gemma4_thinking_speed": assistant_gemma4_thinking_speed,
            "assistant_gemma4_dynamic_few_shots_speed": assistant_gemma4_dynamic_few_shots_speed,
            "assistant_gemma4_merged_adapter_8bit_speed": assistant_gemma4_merged_adapter_8bit_speed,
        },
        "average_metrics": {
            "input_standardsprache": {
                "fre": round(avg_in_fre, 1) if num_eval_ds > 0 else None,
                "wstf": round(avg_in_wstf, 1) if num_eval_ds > 0 else None,
            },
            "ground_truth": {
                "fre": round(avg_gt_fre, 1) if num_eval_ds > 0 else None,
                "wstf": round(avg_gt_wstf, 1) if num_eval_ds > 0 else None,
            },
            "assistant_gemma4": {
                "fre": round(avg_g4_fre, 1) if fre_scores["gemma4"] else None,
                "wstf": round(avg_g4_wstf, 1) if wstf_scores["gemma4"] else None,
                "speed_tokens_per_sec": assistant_gemma4_speed,
            },
            "assistant_gemma4_thinking": {
                "fre": round(avg_g4_think_fre, 1) if fre_scores["gemma4_thinking"] else None,
                "wstf": round(avg_g4_think_wstf, 1) if wstf_scores["gemma4_thinking"] else None,
                "speed_tokens_per_sec": assistant_gemma4_thinking_speed,
            },
            "assistant_gemma4_dynamic_few_shots": {
                "fre": round(avg_g4_few_fre, 1) if fre_scores["gemma4_dynamic_few_shots"] else None,
                "wstf": round(avg_g4_few_wstf, 1) if wstf_scores["gemma4_dynamic_few_shots"] else None,
                "speed_tokens_per_sec": assistant_gemma4_dynamic_few_shots_speed,
            },
            "assistant_gemma4_merged_adapter_8bit": {
                "fre": round(avg_g4_merged_8bit_fre, 1) if fre_scores["gemma4_merged_adapter_8bit"] else None,
                "wstf": round(avg_g4_merged_8bit_wstf, 1) if wstf_scores["gemma4_merged_adapter_8bit"] else None,
                "speed_tokens_per_sec": assistant_gemma4_merged_adapter_8bit_speed,
            },
        },
    }

    with RESULTS_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Wrote evaluation metadata and throughput speeds to: {RESULTS_METADATA_PATH}\n")


if __name__ == "__main__":
    main()
