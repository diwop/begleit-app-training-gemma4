#!/usr/bin/env python3
"""
Runs 16-bit Base Model + Unmerged LoRA Adapter evaluation for Gemma 4 26B-A4B on data/dataset_eval.jsonl using SGLang.
Merges Step 5 (16-bit LoRA adapter) results into previous data/results.jsonl and data/results-metadata.json produced by evaluation.py.

Evaluates:
5. Fine-Tuned 16-bit Base Model + Unmerged LoRA Adapter WITH thinking (enable_thinking=True, T=1.0, top_p=0.95, top_k=64)
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

from dynamic_few_shots import extract_raw_standardsprache

# Context length and token budgets for SGLang engine
MAX_SEQUENCE_LENGTH = 32768
MAX_NEW_TOKENS = 8192
MAX_INPUT_TOKENS = MAX_SEQUENCE_LENGTH - MAX_NEW_TOKENS - 512

# Default to 0 for full dataset evaluation. Set MAX_EVAL_SAMPLES=8 for smoke test.
MAX_EVAL_SAMPLES = int(os.environ.get("MAX_EVAL_SAMPLES", "0"))

BASE_MODEL_NAME = "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
BASE_16B_MODEL_NAME = "google/gemma-4-26b-a4b-it"
ADAPTER_DIR = Path(os.environ.get("LORA_ADAPTER", "local/adapters/gemma-4-26b-a4b-it-lora"))
EVAL_DATA_PATH = Path("data/dataset_eval.jsonl")
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
    print("      Gemma 4 BF16 / 16-bit LoRA Evaluation (Phase 5)")
    print("=" * 60)
    print(f"[INFO] 16-bit Base Model: {BASE_16B_MODEL_NAME}")
    print(f"[INFO] LoRA Adapter Path: {ADAPTER_DIR}")
    print(f"[INFO] Input Dataset    : {EVAL_DATA_PATH}")
    print(f"[INFO] Target Results   : {RESULTS_OUTPUT_PATH}")
    print(f"[INFO] Target Metadata  : {RESULTS_METADATA_PATH}")

    gpu_count = torch.cuda.device_count()
    default_tp = 4 if gpu_count >= 4 else (2 if gpu_count in (2, 3) else min(gpu_count, 4))
    tp_size_16b = int(os.environ.get("TENSOR_PARALLEL_SIZE", str(default_tp)))
    print(f"[INFO] Detected GPUs    : {gpu_count} (Using Tensor Parallel Size for 16-bit LoRA: {tp_size_16b})")

    if gpu_count < 2:
        print(f"[ERROR] 16-bit model evaluation requires at least 2 GPUs (detected {gpu_count}).", file=sys.stderr)
        sys.exit(1)

    base_16b_path = get_model_snapshot_path(BASE_16B_MODEL_NAME, required=True)
    if not ADAPTER_DIR.exists():
        print(f"[ERROR] LoRA adapter not found at '{ADAPTER_DIR}'.", file=sys.stderr)
        sys.exit(1)

    ensure_sglang_compatible_adapter(ADAPTER_DIR)

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

    # Load tokenizer
    model_path = get_model_snapshot_path(BASE_MODEL_NAME, required=False) or base_16b_path
    print(f"\n[INFO] Loading tokenizer from snapshot: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    zero_shot_conversations = [
        [
            {"role": "system", "content": rec["system"]},
            {"role": "user", "content": rec["user"]},
        ]
        for rec in records
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

    sampling_params_thinking = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_new_tokens": 8192,
        "skip_special_tokens": False,
    }

    # =========================================================================
    # STEP 5: Fine-Tuned 16-bit Base Model + Unmerged LoRA
    # =========================================================================
    print("=" * 60)
    print(f"[INFO] Initializing SGLang engine with 16-bit Base Model ({BASE_16B_MODEL_NAME}) + Unmerged LoRA ({ADAPTER_DIR})")
    print(f"[INFO] Fresh process with clean GPU VRAM (TP={tp_size_16b}, GPUs={gpu_count})")

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
        watchdog_timeout=86400,
        dist_timeout=7200,
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
    print(f"[SUCCESS] Step 5/5 (16-bit LoRA) completed in {step5_elapsed:.1f}s ({step5_elapsed/len(records):.2f}s/sample)\n")

    engine_16b.shutdown()
    del engine_16b
    time.sleep(3)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # =========================================================================
    # Merge Results into existing results.jsonl and results-metadata.json
    # =========================================================================
    print("=" * 60)
    print(f"[INFO] Merging 16-bit LoRA outputs into: {RESULTS_OUTPUT_PATH}")

    existing_entries = {}
    if RESULTS_OUTPUT_PATH.exists():
        with RESULTS_OUTPUT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing_entries[item["id"]] = item

    merged_results = []
    fre_16b_scores = []
    wstf_16b_scores = []

    for idx, rec in enumerate(records):
        raw_output = extract_output_text(adapter_16bit_outputs[idx])
        reasoning_trace, clean_text = extract_gemma4_reasoning(raw_output)
        metrics_16b = get_raw_metrics(clean_text)

        rec_id = rec["id"]
        if rec_id in existing_entries:
            entry = existing_entries[rec_id]
        else:
            raw_user_input = extract_raw_standardsprache(text=rec.get("user", ""), doc_id=rec_id)
            entry = {
                "id": rec_id,
                "system": rec["system"],
                "user_input": raw_user_input,
                "user_input_metrics": get_raw_metrics(raw_user_input),
                "user": rec["user"],
                "assistant": rec["assistant"],
                "assistant_metrics": get_raw_metrics(rec["assistant"]) if rec["assistant"] else None,
            }

        entry["assistant_gemma4_adapter_16bit_reasoning"] = reasoning_trace
        entry["assistant_gemma4_adapter_16bit"] = clean_text
        entry["assistant_gemma4_adapter_16bit_metrics"] = metrics_16b
        merged_results.append(entry)

        if rec["assistant"] is not None:
            fre_16b_scores.append(metrics_16b["fre"])
            wstf_16b_scores.append(metrics_16b["wstf"])

    with RESULTS_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for entry in merged_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Wrote {len(merged_results)} merged evaluation samples to: {RESULTS_OUTPUT_PATH}")

    # Calculate throughput speed
    adapter_16bit_speed = calculate_speed(adapter_16bit_outputs, step5_elapsed, tokenizer)

    # Load and update metadata
    metadata = {}
    if RESULTS_METADATA_PATH.exists():
        try:
            with RESULTS_METADATA_PATH.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    if not metadata:
        metadata = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_samples": len(merged_results),
            "total_evaluation_time_seconds": 0.0,
            "phase_elapsed_seconds": {},
            "model_speeds_tokens_per_second": {},
            "average_metrics": {},
        }

    prev_total_time = metadata.get("total_evaluation_time_seconds", 0.0) or 0.0
    metadata["total_evaluation_time_seconds"] = round(prev_total_time + step5_elapsed, 2)
    metadata["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    phase_elapsed = metadata.setdefault("phase_elapsed_seconds", {})
    phase_elapsed["step5_adapter_16bit"] = round(step5_elapsed, 2)

    model_speeds = metadata.setdefault("model_speeds_tokens_per_second", {})
    model_speeds["assistant_gemma4_adapter_16bit_speed"] = adapter_16bit_speed

    avg_metrics = metadata.setdefault("average_metrics", {})
    avg_fre_16b = round(sum(fre_16b_scores) / len(fre_16b_scores), 1) if fre_16b_scores else None
    avg_wstf_16b = round(sum(wstf_16b_scores) / len(wstf_16b_scores), 1) if wstf_16b_scores else None

    avg_metrics["assistant_gemma4_adapter_16bit"] = {
        "fre": avg_fre_16b,
        "wstf": avg_wstf_16b,
        "speed_tokens_per_sec": adapter_16bit_speed,
    }

    with RESULTS_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Updated evaluation metadata with 16-bit LoRA metrics to: {RESULTS_METADATA_PATH}\n")

    print("=" * 60)
    print("      Evaluation Summary Metrics (Dataset Averages)")
    print("=" * 60)
    for model_key, scores in avg_metrics.items():
        if isinstance(scores, dict):
            fre_val = scores.get("fre")
            wstf_val = scores.get("wstf")
            spd_val = scores.get("speed_tokens_per_sec")
            spd_str = f"  |  Speed = {spd_val:.1f} tok/s" if spd_val is not None else ""
            print(f"  * {model_key:<35}: FRE = {fre_val}  |  WSTF = {wstf_val}{spd_str}")
    print(f"  * Phase 5 (16-bit LoRA) Elapsed Time : {step5_elapsed:.1f}s")
    print(f"  * Cumulative Total Evaluation Time   : {metadata['total_evaluation_time_seconds']:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
