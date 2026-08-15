#!/usr/bin/env python3
"""
Runs baseline evaluation for RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic on data/dataset_eval.jsonl.
Evaluates the model twice:
1. Standard generation (without thinking: chat_template_kwargs={"enable_thinking": False})
2. Thinking-enabled generation (with thinking: chat_template_kwargs={"enable_thinking": True})

Calculates German readability metrics (FRE and WSTF) via textstat for all I/O texts.
"""

from __future__ import annotations

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

# --- 1. Automated on-disk hotfix for vLLM weight loading, MLA & heterogeneous layers ---
def apply_vllm_hotfixes() -> None:
    """Automated hotfix for vLLM weight loading & heterogeneous layer compatibility."""
    try:
        import vllm
        vllm_path = os.path.dirname(vllm.__file__)

        # 1a. Weight loader tensor slice hotfix (Gemma 4 FP8 [512] vs [256])
        weight_utils_file = os.path.join(vllm_path, "model_executor/model_loader/weight_utils.py")
        if os.path.exists(weight_utils_file):
            with open(weight_utils_file, "r", encoding="utf-8") as f:
                code = f.read()
            broken_str = "assert param.size() == loaded_weight.size(), ("
            fixed_str = (
                "if param.size() != loaded_weight.size():\n"
                "        if param.numel() <= loaded_weight.numel():\n"
                "            loaded_weight = loaded_weight.flatten()[:param.numel()].reshape(param.shape)\n"
                "    assert param.size() == loaded_weight.size(), ("
            )
            if broken_str in code and "loaded_weight.flatten()" not in code:
                print("Applying vLLM Gemma 4 weight loader hotfix on disk...", flush=True)
                with open(weight_utils_file, "w", encoding="utf-8") as f:
                    f.write(code.replace(broken_str, fixed_str, 1))

        # 1b. MLA attention hotfix (vLLM Issue #43263)
        mla_file = os.path.join(vllm_path, "model_executor/layers/attention/mla_attention.py")
        if os.path.exists(mla_file):
            with open(mla_file, "r", encoding="utf-8") as f:
                code = f.read()
            broken_mla = "kv_c_normed = kv_c_normed.to(self.kv_b_proj.weight.dtype)"
            fixed_mla = "kv_c_normed = kv_c_normed.to(_kv_b_proj_w_dtype)"
            if broken_mla in code:
                print("🔧 Applying vLLM MLA hotfix on disk...", flush=True)
                with open(mla_file, "w", encoding="utf-8") as f:
                    f.write(code.replace(broken_mla, fixed_mla, 1))
    except Exception as e:
        print(f"[INFO] vLLM hotfix check: {e}")

apply_vllm_hotfixes()

# --- 2. Transformers 5.x RoPE & Tokenizer Hotfixes ---
try:
    import transformers
    from transformers.configuration_utils import PretrainedConfig
    _orig_init = PretrainedConfig.__init__
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.allow_global_per_layer_attribute_access = True
    PretrainedConfig.__init__ = _patched_init
    PretrainedConfig.allow_global_per_layer_attribute_access = True
except Exception:
    pass

try:
    from transformers.models.gemma.configuration_gemma import GemmaConfig
    _orig_gemma_init = GemmaConfig.__init__
    def _patched_gemma_init(self, *args, **kwargs):
        if "rope_scaling" in kwargs and kwargs["rope_scaling"] is not None:
            rs = kwargs["rope_scaling"]
            if isinstance(rs, dict):
                if "rope_type" not in rs and "type" in rs:
                    rs["rope_type"] = rs["type"]
                if "rope_type" not in rs:
                    rs["rope_type"] = "default"
        _orig_gemma_init(self, *args, **kwargs)
    GemmaConfig.__init__ = _patched_gemma_init
except Exception:
    pass

try:
    from transformers import GemmaTokenizer, GemmaTokenizerFast
    for cls in [GemmaTokenizer, GemmaTokenizerFast]:
        if not hasattr(cls, "all_special_tokens_extended"):
            cls.all_special_tokens_extended = property(lambda self: self.all_special_tokens)
except Exception:
    pass

import torch

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("[ERROR] vLLM is not installed in the current environment.", file=sys.stderr)
    print("[INFO] Please run this script inside the vLLM container (images/vllm_sandbox).", file=sys.stderr)
    sys.exit(1)

try:
    import textstat
    textstat.set_lang("de")
except ImportError:
    textstat = None
    print("[WARNING] 'textstat' is not installed. Text readability metrics will default to 0.0.", file=sys.stderr)

# Single source of truth for max sequence/token length
MAX_SEQUENCE_LENGTH = 8192

MODEL_NAME = "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
EVAL_DATA_PATH = Path("data/dataset_eval.jsonl")
RESULTS_OUTPUT_PATH = Path("data/results.jsonl")


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


def main() -> None:
    overall_start_time = time.time()

    print("=" * 60)
    print("      Gemma 4 Baseline Evaluation (vLLM)")
    print("=" * 60)
    print(f"[INFO] Model            : {MODEL_NAME}")
    print(f"[INFO] Input Dataset    : {EVAL_DATA_PATH}")
    print(f"[INFO] Output Results   : {RESULTS_OUTPUT_PATH}")

    gpu_count = torch.cuda.device_count()
    tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    print(f"[INFO] Detected GPUs    : {gpu_count} (Using Tensor Parallel Size: {tensor_parallel_size})")

    records = load_eval_data(EVAL_DATA_PATH)
    print(f"[INFO] Loaded {len(records)} evaluation samples.")

    # Get local offline snapshot path on disk (fail fast if missing)
    model_path = get_model_snapshot_path(MODEL_NAME)

    # Initialize vLLM model with robust Gemma 4 FP8 configuration
    print(f"\n[INFO] Initializing vLLM engine and loading model weights from: {model_path}")
    print(f"[INFO] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    model_load_start = time.time()

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=0.90,
        max_model_len=MAX_SEQUENCE_LENGTH,
        disable_log_stats=False,
    )

    model_load_elapsed = time.time() - model_load_start
    print(f"[SUCCESS] vLLM engine & weights ready in {model_load_elapsed:.1f}s ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")

    # Standard chat conversations for both passes
    conversations = [
        [
            {"role": "system", "content": rec["system"]},
            {"role": "user", "content": rec["user"]},
        ]
        for rec in records
    ]

    # Sampling parameters
    sampling_params_no_thinking = SamplingParams(
        temperature=0.0,
        max_tokens=4096,
    )

    sampling_params_thinking = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
    )

    # STEP 1: Standard Inference
    print("=" * 60)
    print(f"[STEP 1/2] Running inference WITHOUT thinking (enable_thinking=False) for {len(records)} samples...")
    print(f"[INFO] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    step1_start = time.time()

    no_thinking_outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling_params_no_thinking,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=True,
    )

    step1_elapsed = time.time() - step1_start
    print(f"[SUCCESS] Step 1 completed in {step1_elapsed:.1f}s ({step1_elapsed/len(records):.2f}s/sample)\n")

    # STEP 2: Thinking-Enabled Inference
    print("=" * 60)
    print(f"[STEP 2/2] Running inference WITH thinking (enable_thinking=True) for {len(records)} samples...")
    print(f"[INFO] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    step2_start = time.time()

    thinking_outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling_params_thinking,
        chat_template_kwargs={"enable_thinking": True},
        use_tqdm=True,
    )

    step2_elapsed = time.time() - step2_start
    print(f"[SUCCESS] Step 2 completed in {step2_elapsed:.1f}s ({step2_elapsed/len(records):.2f}s/sample)\n")

    # Calculate metrics and assemble results
    print("=" * 60)
    print(f"[INFO] Calculating German readability metrics (FRE & WSTF)...")
    RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    fre_scores = {"input": [], "ground_truth": [], "gemma4": [], "gemma4_thinking": []}
    wstf_scores = {"input": [], "ground_truth": [], "gemma4": [], "gemma4_thinking": []}

    for idx, rec in enumerate(records):
        out_no_thinking = no_thinking_outputs[idx].outputs[0].text.strip()
        out_thinking = thinking_outputs[idx].outputs[0].text.strip()
        raw_user_input = load_raw_standardsprache(rec["id"]) or rec["user"]

        user_metrics = get_raw_metrics(raw_user_input)
        assistant_metrics = get_raw_metrics(rec["assistant"])
        gemma4_metrics = get_raw_metrics(out_no_thinking)
        gemma4_thinking_metrics = get_raw_metrics(out_thinking)

        fre_scores["input"].append(user_metrics["fre"])
        fre_scores["ground_truth"].append(assistant_metrics["fre"])
        fre_scores["gemma4"].append(gemma4_metrics["fre"])
        fre_scores["gemma4_thinking"].append(gemma4_thinking_metrics["fre"])

        wstf_scores["input"].append(user_metrics["wstf"])
        wstf_scores["ground_truth"].append(assistant_metrics["wstf"])
        wstf_scores["gemma4"].append(gemma4_metrics["wstf"])
        wstf_scores["gemma4_thinking"].append(gemma4_thinking_metrics["wstf"])

        result_entry = {
            "id": rec["id"],
            "system": rec["system"],
            "user_input": raw_user_input,
            "user_input_metrics": user_metrics,
            "assistant": rec["assistant"],
            "assistant_metrics": assistant_metrics,
            "assistant_gemma4": out_no_thinking,
            "assistant_gemma4_metrics": gemma4_metrics,
            "assistant_gemma4_thinking": out_thinking,
            "assistant_gemma4_thinking_metrics": gemma4_thinking_metrics,
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
    if fre_scores["input"] and sum(fre_scores["input"]) > 0:
        avg_in_fre = sum(fre_scores["input"]) / len(results)
        avg_gt_fre = sum(fre_scores["ground_truth"]) / len(results)
        avg_g4_fre = sum(fre_scores["gemma4"]) / len(results)
        avg_g4_think_fre = sum(fre_scores["gemma4_thinking"]) / len(results)

        avg_in_wstf = sum(wstf_scores["input"]) / len(results)
        avg_gt_wstf = sum(wstf_scores["ground_truth"]) / len(results)
        avg_g4_wstf = sum(wstf_scores["gemma4"]) / len(results)
        avg_g4_think_wstf = sum(wstf_scores["gemma4_thinking"]) / len(results)

        print(f"  * Input Standardsprache  : FRE = {avg_in_fre:.1f}  |  WSTF = {avg_in_wstf:.1f}")
        print(f"  * Ground Truth (Target)  : FRE = {avg_gt_fre:.1f}  |  WSTF = {avg_gt_wstf:.1f}")
        print(f"  * Gemma 4 (No Thinking)  : FRE = {avg_g4_fre:.1f}  |  WSTF = {avg_g4_wstf:.1f}")
        print(f"  * Gemma 4 (With Thinking): FRE = {avg_g4_think_fre:.1f}  |  WSTF = {avg_g4_think_wstf:.1f}")
    print(f"  * Total Evaluation Time  : {overall_elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
