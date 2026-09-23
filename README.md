---
train_image: axolotlai/axolotl:0.18.0
eval_image: lmsysorg/sglang:v0.5.17-cu130-runtime
---

# BegleitApp Training (Gemma 4 on HSUper)

This repository contains code to fine-tune and evaluate adapters for
[Google Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it)
on the [HSUper](https://www.hsu-hh.de/hpc/en/hsuper/) GPU infrastructure.  
[Axolotl](https://axolotl.ai) is used to train the adapter for the Mixture of
Experts model, and [SGLang](https://sglang.ai) is used to provide high-performance
inference on the quantized base model merged with the tuned adapter.

Training and evaluation is designed for **NVIDIA L40S GPUs** as provided by HSUper.

The actual training data is stored using [DVC](https://dvc.org).

## Pipeline Architecture

```mermaid
flowchart TD
    subgraph Data["1. Data Preparation (DVC)"]
        D1["Raw Training Data<br><code>data/raw/</code>"] --> D2["Prepared Datasets<br><code>dataset_train.jsonl</code> & <code>dataset_eval.jsonl</code>"]
        P1["System Prompt & Rules<br><code>prompts/system-prompt.md</code>"] --> D2
    end

    subgraph Training["2. Training Pipeline (Axolotl · 4x L40S)"]
        T1["Base Model:<br><b>Gemma 4 26B-A4B-it</b>"] --> T2["LoRA Fine-Tuning<br>(DeepSpeed ZeRO-3)"]
        D2 --> T2
        T2 --> T3["Tuned LoRA Adapter"]
        T3 --> T4["Merge & Quantize<br><code>src-train/merge_and_quantize.py</code>"]
        T1 --> T4
        T4 --> T5["Fine-Tuned Merged 8-bit Model<br><code>local/models/gemma-4-26b-a4b-it-fp8</code>"]
    end

    subgraph Eval["3. Multi-Technique Evaluation (SGLang · 2x L40S)"]
        E_IN["Evaluation Set (77 samples)<br><code>data/dataset_eval.jsonl</code>"]
        E_E5["Semantic Retrieval Index<br><code>multilingual-e5-base</code>"]

        E_IN --> M1["1. Base Zero-Shot<br>(No Thinking, T=0.0)"]
        E_IN --> M2["2. Base Thinking<br>(Thinking Mode, T=1.0)"]
        E_IN & E_E5 --> M3["3. Dynamic Few-Shot<br>(k=2 Demonstrations + Thinking)"]
        E_IN & T5 --> M4["4. Fine-Tuned Adapter<br>(Merged 8-bit + Thinking)"]

        M1 & M2 & M3 & M4 --> E_OUT["Evaluation Results<br><code>data/results.jsonl</code>"]
        E_OUT --> E_STAT["Readability Metrics<br>(FRE & WSTF)"]
        E_STAT --> E_META["Throughput & Metrics Summary<br><code>data/results-metadata.json</code>"]
    end

    subgraph Judge["4. LLM-as-a-Judge Evaluation (Llama 4 Scout FP8 · 4x L40S)"]
        J_MODEL["Judge Model:<br><b>Llama 4 Scout 109B MoE (FP8)</b>"]
        J_PROMPT["Judge Template<br><code>prompts/judge-prompt-template.md</code>"]
        P1 -.->|"Dynamic Rule Embedding<br>(Zero Drift)"| J_PROMPT
        E_OUT --> J_RUN["Pointwise Evaluation<br><code>src-eval/judge_evaluation.py</code>"]
        J_MODEL & J_PROMPT --> J_RUN

        J_RUN --> J_RES["Per-Sample Critiques & Scores<br><code>data/judge_results.jsonl</code>"]
        J_RUN --> J_SUM["Benchmark Summary & Pareto Frontier<br><code>data/judge_summary.json</code>"]

        subgraph Criteria["Scoring Dimensions (1.0 – 5.0)"]
            C1["Regeltreue (Rule Adherence)"]
            C2["Faktentreue (Factual Completeness)"]
            C3["Verständlichkeit (Readability & Tone)"]
            C4["Quality vs. Human Ground Truth"]
        end
        J_RUN -.- Criteria
    end

    classDef data fill:#e1f5fe,stroke:#0288d1,stroke-width:1px;
    classDef train fill:#e8f5e9,stroke:#388e3c,stroke-width:1px;
    classDef eval fill:#fff3e0,stroke:#f57c00,stroke-width:1px;
    classDef judge fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px;

    class D1,D2,P1 data;
    class T1,T2,T3,T4,T5 train;
    class M1,M2,M3,M4,E_OUT,E_STAT,E_META eval;
    class J_MODEL,J_PROMPT,J_RUN,J_RES,J_SUM,C1,C2,C3,C4 judge;
```

---

## Details

See [docs/implementation-details.md](docs/implementation-details.md) for details on training and inference.  
See [docs/judge-evaluation.md](docs/judge-evaluation.md) for details on LLM-as-a-Judge evaluation.  
See [adrs](adrs/) for Architecture Decision Records.