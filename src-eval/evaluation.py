#!/usr/bin/env python3
"""
Runs baseline, dynamic few-shot, fine-tuned merged FP8, and 16-bit unmerged LoRA adapter evaluation for Gemma 4 26B-A4B on data/dataset_eval.jsonl using SGLang.
Evaluates the model across five techniques:
1. Standard zero-shot generation on base model (without thinking: enable_thinking=False)
2. Thinking-enabled generation on base model (with thinking: enable_thinking=True, T=1.0, top_p=0.95, top_k=64)
3. Dynamic Few-Shot generation on base model (2 semantically closest training examples retrieved via multilingual-e5-base, with thinking: enable_thinking=True)
4. Fine-Tuned Merged 8-bit Adapter generation WITH thinking (enable_thinking=True, T=1.0, top_p=0.95, top_k=64)
5. Fine-Tuned 16-bit Base Model + Unmerged LoRA Adapter WITH thinking (enable_thinking=True, T=1.0, top_p=0.95, top_k=64)

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

# Apply SGLang monkey patch for Gemma 4 clippable layers with LoRA
import sglang_clippable_lora_patch

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

# Default to 8 for quick smoke feedback. Set MAX_EVAL_SAMPLES=0 to evaluate entire dataset.
MAX_EVAL_SAMPLES = int(os.environ.get("MAX_EVAL_SAMPLES", "8"))

BASE_MODEL_NAME = "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
BASE_16B_MODEL_NAME = "google/gemma-4-26b-a4b-it"
MERGED_MODEL_PATH = Path(os.environ.get("MERGED_MODEL", "local/models/gemma-4-26b-a4b-it-fp8"))
ADAPTER_DIR = Path(os.environ.get("LORA_ADAPTER", "local/adapters/gemma-4-26b-a4b-it-lora"))
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


def ensure_sglang_compatible_adapter(adapter_path: Path) -> None:
    """Ensure adapter_config.json and weights are fully compatible with SGLang dynamic LoRA loading."""
    config_file = adapter_path / "adapter_config.json"
    if config_file.exists():
        with config_file.open("r", encoding="utf-8") as f:
            config = json.load(f)
        target_modules = config.get("target_modules")
        if isinstance(target_modules, str) and target_modules not in ("all", "all-linear"):
            print(f"[INFO] Converting regex target_modules in {config_file} to standard list for SGLang compatibility.")
            standard_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            config["target_modules"] = standard_modules
            with config_file.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

    # In Gemma 4 architecture, full attention layers (5, 11, 17, 23, 29) have attention_k_eq_v: true and lack v_proj.
    # SGLang normalize_qkv_proj unconditionally expects v_proj when q_proj is present.
    # We supply zero-initialized dummy v_proj weights (which contribute zero to output) to satisfy SGLang concatenation.
    safetensors_file = adapter_path / "adapter_model.safetensors"
    if safetensors_file.exists():
        try:
            from safetensors import safe_open
            from safetensors.torch import save_file

            tensors = {}
            with safe_open(str(safetensors_file), framework="pt") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)

            added = 0
            for layer_num in range(30):
                q_k = f"base_model.model.model.language_model.layers.{layer_num}.self_attn.q_proj.lora_A.weight"
                v_a_k = f"base_model.model.model.language_model.layers.{layer_num}.self_attn.v_proj.lora_A.weight"
                v_b_k = f"base_model.model.model.language_model.layers.{layer_num}.self_attn.v_proj.lora_B.weight"
                k_a_k = f"base_model.model.model.language_model.layers.{layer_num}.self_attn.k_proj.lora_A.weight"
                k_b_k = f"base_model.model.model.language_model.layers.{layer_num}.self_attn.k_proj.lora_B.weight"
                if q_k in tensors and v_a_k not in tensors:
                    tensors[v_a_k] = torch.zeros_like(tensors[k_a_k])
                    tensors[v_b_k] = torch.zeros_like(tensors[k_b_k])
                    added += 2

            if added > 0:
                print(f"[INFO] Injected {added} zero-weight dummy v_proj tensors into {safetensors_file} for SGLang QKV alignment.")
                save_file(tensors, str(safetensors_file))
        except Exception as exc:
            print(f"[WARNING] Could not check/patch safetensors file: {exc}")


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

    # Sampling parameters
    sampling_params_thinking = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_new_tokens": 8192,
        "skip_special_tokens": False,
    }

    # STEP 1 to 4: Temporarily commented out for fast Phase 5 feedback
    no_thinking_outputs = None
    thinking_outputs = None
    few_shot_outputs = None
    merged_adapter_8bit_outputs = None
    print("\n[INFO] Steps 1-4 commented out for quick Phase 5 (16-bit unmerged LoRA) feedback.")

    # STEP 5: Fine-Tuned 16-bit Base Model + Unmerged LoRA Adapter Evaluation
    adapter_16bit_outputs = None
    base_16b_path = get_model_snapshot_path(BASE_16B_MODEL_NAME, required=False)

    if ADAPTER_DIR.exists() and base_16b_path and gpu_count >= 2:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Ensure adapter config and weights are compatible with SGLang dynamic LoRA
        ensure_sglang_compatible_adapter(ADAPTER_DIR)

        # Tensor Parallel size for 16-bit 26B model: requires 2 GPUs (TP=2) for 52 GB model in 96 GB VRAM
        tp_size_16b = 2 if gpu_count in (2, 3) else min(gpu_count, 4)
        print("=" * 60)
        print(f"[INFO] Initializing SGLang engine with 16-bit Base Model ({BASE_16B_MODEL_NAME}) + Unmerged LoRA ({ADAPTER_DIR})")
        print(f"[INFO] Using Tensor Parallel Size: {tp_size_16b} (available GPUs: {gpu_count})")

        engine_16b = sgl.Engine(
            model_path=base_16b_path,
            tp_size=tp_size_16b,
            trust_remote_code=True,
            mem_fraction_static=0.85,
            context_length=MAX_SEQUENCE_LENGTH,
            enable_lora=True,
            lora_paths={"adapter": str(ADAPTER_DIR)},
            max_loras_per_batch=1,
            max_lora_rank=64,
            disable_cuda_graph=True,
            watchdog_timeout=1200,
            dist_timeout=1200,
        )

        print("=" * 60)
        print(f"[STEP 5/5] Running 16-bit Base Model + Unmerged LoRA Adapter WITH thinking (enable_thinking=True, T=1.0, top_p=0.95) for {len(records)} samples...")
        print(f"[INFO] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        step5_start = time.time()

        adapter_16bit_outputs = engine_16b.generate(
            prompts_thinking,
            sampling_params_thinking,
            lora_path="adapter",
        )

        step5_elapsed = time.time() - step5_start
        print(f"[SUCCESS] Step 5 completed in {step5_elapsed:.1f}s ({step5_elapsed/len(records):.2f}s/sample)\n")

        engine_16b.shutdown()
    else:
        print("=" * 60)
        if gpu_count < 2:
            print(f"[INFO] [STEP 5/5] Skipping 16-bit Unmerged LoRA evaluation: requires at least 2 GPUs for 52 GB model (detected {gpu_count} GPU).")
            print("[INFO] Set '#SBATCH --gpus=2' or '--gpus=3' in your submission script to enable 16-bit evaluation.")
        elif not ADAPTER_DIR.exists():
            print(f"[INFO] [STEP 5/5] LoRA adapter not found at '{ADAPTER_DIR}'. Skipping Pass 5.")
        elif not base_16b_path:
            print(f"[INFO] [STEP 5/5] 16-bit base model '{BASE_16B_MODEL_NAME}' snapshot not found in cache. Skipping Pass 5.")
        print()

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
        "gemma4_merged_adapter_8bit": [],
        "gemma4_adapter_16bit": [],
    }
    wstf_scores = {
        "input": [],
        "ground_truth": [],
        "gemma4": [],
        "gemma4_thinking": [],
        "gemma4_dynamic_few_shots": [],
        "gemma4_merged_adapter_8bit": [],
        "gemma4_adapter_16bit": [],
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

        out_adapter_16bit = None
        adapter_16bit_reasoning = None
        gemma4_adapter_16bit_metrics = None

        if adapter_16bit_outputs is not None:
            raw_adapter_16bit = extract_output_text(adapter_16bit_outputs[idx])
            adapter_16bit_reasoning, out_adapter_16bit = extract_gemma4_reasoning(raw_adapter_16bit)
            gemma4_adapter_16bit_metrics = get_raw_metrics(out_adapter_16bit)

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
            if gemma4_adapter_16bit_metrics is not None:
                fre_scores["gemma4_adapter_16bit"].append(gemma4_adapter_16bit_metrics["fre"])

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
            if gemma4_adapter_16bit_metrics is not None:
                wstf_scores["gemma4_adapter_16bit"].append(gemma4_adapter_16bit_metrics["wstf"])

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
            "assistant_gemma4_adapter_16bit_reasoning": adapter_16bit_reasoning,
            "assistant_gemma4_adapter_16bit": out_adapter_16bit,
            "assistant_gemma4_adapter_16bit_metrics": gemma4_adapter_16bit_metrics,
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

        if fre_scores["gemma4_adapter_16bit"]:
            avg_g4_16b_fre = sum(fre_scores["gemma4_adapter_16bit"]) / len(fre_scores["gemma4_adapter_16bit"])
            avg_g4_16b_wstf = sum(wstf_scores["gemma4_adapter_16bit"]) / len(wstf_scores["gemma4_adapter_16bit"])
            print(f"  * Gemma 4 (16-bit LoRA + Think) : FRE = {avg_g4_16b_fre:.1f}  |  WSTF = {avg_g4_16b_wstf:.1f}")

    print(f"  * Total Evaluation Time         : {overall_elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
