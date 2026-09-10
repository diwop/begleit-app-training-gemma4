# LLM-as-a-Judge: Llama 4 Scout (FP8 on 4 GPUs)

This document details the automated **LLM-as-a-Judge** evaluation framework for German Leichte Sprache translations.

---

## 1. Why Llama 4 Scout as Judge?

1. **Unbiased Third-Party Architecture**:
   * All translation candidates originate from Google's Gemma 4 family. Evaluating them with Gemma 4 introduces self-preference bias.
   * Meta's **Llama 4 Scout** (`meta-llama/Llama-4-Scout-17B-16E-Instruct`, 109B total MoE / 17B active parameters) was trained independently on a massive multilingual corpus with deep European language reasoning.
2. **High Throughput (17B Active FLOPs)**:
   * Although Llama 4 Scout holds 109B total parameters of knowledge, generating each token only routes through 17B active parameters. Decode latency is comparable to a small model.
3. **Massive Context & Native Multilinguality**:
   * Native multi-million token context window allows comprehensive rubrics without context degradation.

---

## 2. Hardware Architecture & Deployment on HSUper

The evaluation runs on the **HSUper cluster** in partition `small_gpu8` across **4 $\times$ NVIDIA L40S GPUs (192 GB total VRAM)**:

```
4 × NVIDIA L40S (192 GB total VRAM):
├── Model Weights (FP8 109B MoE) : ~110–120 GB
├── KV Cache & Working Memory   : ~40–50 GB
└── Memory Overhead & Buffers   : ~20 GB
Total Allocation               : ~170–180 GB (TP=4)
```

* **Tensor Parallelism**: `TP=4`
* **Inference Engine**: SGLang inside Apptainer (`images/sglang_sandbox`)
* **Scratch Space**: High-speed node-local NVMe SSD via `$SLURM_TMPDIR`

---

## 3. Evaluation Methodology: Pointwise Rubric Scoring

Rather than presenting all 4 candidate translations in a single prompt (which triggers position bias, attention dilution, and verbosity bias), each candidate translation is judged **pointwise** in isolation against:
1. The original Standardsprache source text.
2. The Human Ground Truth reference.
3. The official Leichte Sprache guidelines (from `prompts/judge-rubric.md`).

### Models Evaluated
1. `assistant_gemma4`: Zero-Shot (Base Gemma 4)
2. `assistant_gemma4_thinking`: Zero-Shot + Thinking (Base Gemma 4)
3. `assistant_gemma4_dynamic_few_shots`: Dynamic Few-Shot ($k=2$)
4. `assistant_gemma4_merged_adapter_8bit`: Fine-Tuned Merged 8-bit Adapter
5. `assistant`: Human Reference (Ground Truth Control)

---

## 4. Scoring Criteria (1.0 – 5.0)

| Criterion | Scale | Focus Areas |
| :--- | :---: | :--- |
| **`rule_adherence`** | 1.0 – 5.0 | Sentence length ($\le 10$ words), hyphenation of compound nouns (`Bundes-Tag`), active voice only, no subjunctive/passive, simple vocabulary. |
| **`factual_completeness`** | 1.0 – 5.0 | Preservation of core legal/practical facts, zero hallucinations, no distortive framing. |
| **`readability`** | 1.0 – 5.0 | Natural flow, logical structure with headings/paragraphs, respectful and adult tone. |
| **`overall_score`** | 1.0 – 5.0 | Comprehensive evaluation of the translation quality. |
| **`ground_truth_comparison`** | Categorical | `"better"`, `"comparable"`, or `"worse"` relative to the human reference. |

### Structured Output Schema
The judge outputs valid, parseable JSON:
```json
{
  "rule_adherence": {
    "score": 4.5,
    "critique": "Hervorragende Bindestrich-Trennung und kurze Sätze. Ein Passivsatz im zweiten Absatz hätte aktiv formuliert werden können."
  },
  "factual_completeness": {
    "score": 5.0,
    "critique": "Alle Förderbedingungen und Ansprechpartner wurden vollständig übernommen."
  },
  "readability": {
    "score": 4.8,
    "critique": "Sehr klar und flüssig lesbar, respektvolle Sprache."
  },
  "ground_truth_comparison": {
    "relative_to_ground_truth": "comparable",
    "critique": "Struktur und Vereinfachung sind auf Augenhöhe mit der menschlichen Übersetzung."
  },
  "overall_score": 4.8,
  "summary": "Sehr gelungene Übersetzung mit minimalem Optimierungspotenzial beim Passiv."
}
```

---

## 5. Execution Workflow

### Step 1: Download Model Weights (on Login Node)
Run on `hsuper-login01` where internet access is available:
```bash
bash scripts/download_judge_model.sh
```

### Step 2: Submit Evaluation to Slurm (4 GPUs)
Submit the job to partition `small_gpu8`:
```bash
sbatch scripts/submit_judge_evaluation.sbatch
```

### Step 3: Inspect Results
* **Detailed per-sample evaluations**: [`data/judge_results.jsonl`](file:///Users/christophwulf/github/diwop/begleit-app-training-gemma4/data/judge_results.jsonl)
* **Dataset-wide benchmark summary**: [`data/judge_summary.json`](file:///Users/christophwulf/github/diwop/begleit-app-training-gemma4/data/judge_summary.json)
