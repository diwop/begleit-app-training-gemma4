# Implementation Details

This document records technical details, setup steps, and execution guidelines for running Axolotl fine-tuning, SGLang evaluation, and DVC data management on the **HSUper** GPU cluster (NVIDIA L40S nodes).

---

## 1. Axolotl & SGLang Cluster GPU Smoke Test

### Overview
The smoke test verifies that:
1. The **Axolotl** Docker container (`train_image` in `README.md`) and **SGLang** Docker container (`eval_image` in `README.md`) execute seamlessly under **Apptainer** on HSUper GPU nodes (`small_gpu8`).
2. Axolotl, SGLang, and essential ML frameworks are installed and functional in their respective container environments.
3. PyTorch detects NVIDIA L40S GPUs with full CUDA capability and reports correct VRAM capacity (~44 GB free per GPU).

---

### How Code & Scripts Are Available on the GPU Node

1. **Shared Network Filesystem (NFS / BeeGFS)**:
   - You clone the git repository on the **login node** into `$HOME/begleit-app-training-gemma4`:
     ```bash
     cd $HOME
     git clone https://github.com/diwop/begleit-app-training-gemma4.git
     ```
   - The login node and GPU compute nodes share `$HOME`.
   - When you `salloc` or `sbatch` onto a GPU node, your files are **already present and accessible** at `$HOME/begleit-app-training-gemma4`—no git or network cloning is needed on the GPU node itself.

2. **Apptainer Volume Bind Mount (`--bind`)**:
   - `scripts/run_smoke_test.sh` executes Apptainer mounting the repo host path to `/repo`:
     ```bash
     apptainer exec --nv --bind "${WORKSPACE_ROOT}:/repo" --pwd /repo ...
     ```
   - Mounting to `/repo` (instead of `/workspace`) ensures that container-specific virtual environments (such as `/workspace/axolotl-venv/bin/python` in Axolotl) remain intact and accessible.
   - For Axolotl: Executes `/workspace/axolotl-venv/bin/python /repo/src-train/smoke_test.py`.
   - For SGLang: Executes `python3 /repo/src-eval/smoke_test.py`.

---

### Step-by-Step Instructions

* **SSH to HSUper login node**:
   ```bash
   ssh <hsu-name>@hsuper-login01.hsu-hh.de
   ```

* **Navigate to the repository in `$HOME`**:
   ```bash
   cd $HOME/begleit-app-training-gemma4
   ```

* **Prepare Apptainer Containers (Login Node only - internet required)**:
   ```bash
   bash scripts/prepare_images.sh
   ```
   *(This extracts `train_image` and `eval_image` strictly from `README.md` frontmatter and builds `images/axolotl_sandbox` and `images/sglang_sandbox` on the shared filesystem).*

#### Option 1: Interactive Node

* **Allocate a GPU node** (e.g. 2 GPUs for 5 min):
   ```bash
   salloc --partition=small_gpu8 --gpus 2 --time=00:05:00
   ```

* **Wait for the node and SSH into it** (e.g. node `gpu08` is ready):
   ```bash
   ssh gpu08
   ```

* **Run the smoke test script on the GPU node**:
   ```bash
   bash scripts/run_smoke_test.sh
   ```

#### Option 2: Queue using sbatch

* **Queue the task**:
   ```bash
   sbatch scripts/submit_smoke_test.sbatch
   ```

* **Monitor the job status**:
   ```bash
   squeue -u $USER -l
   ```

* **View the results**:
   ```bash
   cat logs/smoke_test_*.log
   ```

---

### Expected Output Example

```text
============================================================
 Starting Full Training & Evaluation Cluster Smoke Test
============================================================

[STEP 1/2] Executing Axolotl Training Container Smoke Test...
[INFO] Container: .../images/axolotl_sandbox
============================================================
      Axolotl GPU Cluster Environment Smoke Test
============================================================

[System Info]
  Python Version : 3.12.13
  Platform       : Linux-4.18.0-553.126.1.el8_10.0.1.x86_64-x86_64-with-glibc2.35

[ML Core Libraries]
  Axolotl        : 0.18.0 / 0.19.0.dev0
  Transformers   : 5.x.x
  PEFT           : 0.20.0
  Accelerate     : 1.13.0
  BitsAndBytes   : 0.50.0
  LLMCompressor  : 0.13.0
  Triton         : 3.7.0
  SGLang         : Not Installed

[PyTorch & CUDA Environment]
  PyTorch        : 2.12.0+cu130
  CUDA Available : True
  CUDA Version   : 13.0
  cuDNN Version  : 92000
  GPU Count      : 2

[GPU Details]
  --- GPU 0: NVIDIA L40S ---
      Compute Capability : 8.9
      SM Count           : 142
      Total VRAM         : 44.42 GB
      Free VRAM          : 44.00 GB
  --- GPU 1: NVIDIA L40S ---
      Compute Capability : 8.9
      SM Count           : 142
      Total VRAM         : 44.42 GB
      Free VRAM          : 44.00 GB
============================================================

[STEP 2/2] Executing SGLang Evaluation Container Smoke Test...
[INFO] Container: .../images/sglang_sandbox
[INFO] Using SGLang Container Python: /usr/bin/python3
============================================================
      SGLang Evaluation Cluster Environment Smoke Test
============================================================

[System Info]
  Python Version : 3.12.x
  Platform       : Linux-...

[Inference Core Libraries]
  SGLang         : 0.5.17
  Transformers   : 5.x.x
  Ray            : Installed
  FlashInfer     : Installed

[PyTorch & CUDA Environment]
  PyTorch        : 2.5.x+cu124
  CUDA Available : True
  CUDA Version   : 12.4
  GPU Count      : 2

[GPU Details]
  --- GPU 0: NVIDIA L40S ---
      Compute Capability : 8.9
      SM Count           : 142
      Total VRAM         : 44.42 GB
      Free VRAM          : 44.00 GB
  --- GPU 1: NVIDIA L40S ---
      Compute Capability : 8.9
      SM Count           : 142
      Total VRAM         : 44.42 GB
      Free VRAM          : 44.00 GB
============================================================

============================================================
[SUCCESS] Both Training (Axolotl) and Evaluation (SGLang) smoke tests completed!
============================================================
```

---

## 2. DVC Data Management (AWS S3 Remote)

Raw training and evaluation datasets are stored in AWS S3 and tracked in Git via **DVC (Data Version Control)** pointer files (`data/raw.dvc`).

### Data Directory Structure
- `data/raw/`: Contains raw text dataset files (e.g. `NNNN_Standardsprache.txt`, `NNNN_Leichte_Sprache.txt`).
- `data/raw.dvc`: Small Git-tracked text pointer containing dataset content hashes and metadata.
- `data/.gitignore`: Auto-generated by DVC to prevent large dataset files from being committed to Git directly.

---

### Filename Validation
Before tracking new dataset files in `data/raw/`, run the filename validator script:
```bash
python3 scripts/check_filenames.py
```
This script automatically:
1. Deletes non-text files (e.g. `.docx`).
2. Strips trailing spaces before `.txt`.
3. Unifies naming conventions (e.g. `Leichte Sprache` → `Leichte_Sprache`).
4. Fixes 3-digit prefixes (e.g. `115_` → `0115_`) and resolves duplicates.

---

### Adding New Datasets (`dvc add`)

When new files are added or modified in `data/raw/`:

1. **Add the directory to DVC tracking**:
   ```bash
   dvc add data/raw
   ```

2. **Stage DVC pointer files in Git**:
   ```bash
   git add data/.gitignore data/raw.dvc scripts/check_filenames.py
   git commit -m "add raw training dataset via DVC"
   ```

---

### Pushing Data to AWS S3 (`dvc push`)

To upload raw dataset files from your local machine to the remote S3 bucket:

1. **Authenticate against AWS**:
   - Using AWS CLI configuration:
     ```bash
     aws configure
     ```
   - Or using environment variable / AWS profile:
     ```bash
     export AWS_PROFILE=my-aws-profile
     # OR
     export AWS_ACCESS_KEY_ID="AKIA..."
     export AWS_SECRET_ACCESS_KEY="wJalrX..."
     export AWS_DEFAULT_REGION="eu-central-1"
     ```

2. **Push data to S3**:
   ```bash
   dvc push
   ```

---

### Pulling Data on the Cluster (`dvc pull`)

> **IMPORTANT**: Run `dvc pull` on the **login node** (`hsuper-login01`), as compute nodes do not have outbound internet access.

Use the helper script [`scripts/pull_data.sh`](file:///Users/christophwulf/github/diwop/begleit-app-training-gemma4/scripts/pull_data.sh) which automatically manages the workspace virtual environment and pulls all DVC data from S3:

```bash
# 1. SSH to HSUper login node
ssh <hsu-name>@hsuper-login01.hsu-hh.de

# 2. Navigate to repo and pull Git changes
cd $HOME/begleit-app-training-gemma4
git pull

# 3. Pull dataset files from AWS S3
bash scripts/pull_data.sh
```

Once `pull_data.sh` finishes, `data/raw/` (and generated dataset files) on the shared filesystem are fully populated and immediately ready for offline GPU training on compute nodes!

---

### Data Preparation Pipeline (`dvc repro`)

A reproducible DVC stage in [dvc.yaml](file:///Users/christophwulf/github/diwop/begleit-app-training-gemma4/dvc.yaml) transforms raw paired text documents and prompt templates into training and evaluation JSONL files:

* **Script**: [src-train/prepare_data.py](file:///Users/christophwulf/github/diwop/begleit-app-training-gemma4/src-train/prepare_data.py)
* **Dependencies**:
  - `prompts/system-prompt.md`
  - `prompts/prompt-template.md`
  - `data/raw`
  - `src-train/prepare_data.py`
* **Outputs**:
  - `data/dataset_train.jsonl` (90% split, seed 42)
  - `data/dataset_eval.jsonl` (10% split, seed 42)

To run or reproduce the data preparation stage:
```bash
dvc repro
```

---

## 3. Gemma 4 Evaluation (`src-eval/evaluation.py`)

Runs inference on the 10% evaluation dataset (`data/dataset_eval.jsonl`) using **SGLang** with the base model `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` across two modes via native chat template thinking control:
1. **Standard Zero-Shot Translation** (`enable_thinking=False`, $T=0.0$)
2. **Thinking-Enabled Zero-Shot Translation** (`enable_thinking=True`, $T=1.0, top\_p=0.95, top\_k=64$)
3. **Dynamic Few-Shot Translation with Thinking** ($k=2$ retrieved examples from training set via `multilingual-e5-base`, `enable_thinking=True`, $T=1.0, top\_p=0.95, top\_k=64$)
4. **Fine-Tuned Merged FP8 Model with Thinking** (`enable_thinking=True`, $T=1.0, top\_p=0.95, top\_k=64$)
5. **Fine-Tuned 16-bit Base Model with Unmerged LoRA Adapter** (`enable_thinking=True`, $T=1.0, top\_p=0.95, top\_k=64$, requires $\ge 2$ GPUs)

Produces `data/results.jsonl` with German textstat readability metrics (`fre` = Flesch Reading Ease, `wstf` = Wiener Sachtextformel):
```json
{
  "id": "<id>",
  "system": "<system-prompt>",
  "user_input": "<raw input text without template wrapper>",
  "user_input_metrics": { "fre": 45.2, "wstf": 11.4 },
  "user": "<zero-shot prompt template wrapped text>",
  "user_dynamic_few_shots": "<prompt template with 2 few-shot demonstrations>",
  "assistant": "<ground-truth Leichte_Sprache>",
  "assistant_metrics": { "fre": 88.5, "wstf": 4.1 },
  "assistant_gemma4": "<gemma4 output without thinking>",
  "assistant_gemma4_metrics": { "fre": 82.1, "wstf": 5.2 },
  "assistant_gemma4_thinking_reasoning": "<gemma4 reasoning trace>",
  "assistant_gemma4_thinking": "<gemma4 output with thinking>",
  "assistant_gemma4_thinking_metrics": { "fre": 85.3, "wstf": 4.6 },
  "assistant_gemma4_dynamic_few_shots_reasoning": "<few-shots reasoning trace>",
  "assistant_gemma4_dynamic_few_shots": "<gemma4 output with few-shots and thinking>",
  "assistant_gemma4_dynamic_few_shots_metrics": { "fre": 87.1, "wstf": 4.2 },
  "assistant_gemma4_merged_adapter_8bit_reasoning": "<merged 8-bit adapter reasoning trace>",
  "assistant_gemma4_merged_adapter_8bit": "<merged 8-bit adapter output with thinking>",
  "assistant_gemma4_merged_adapter_8bit_metrics": { "fre": 89.2, "wstf": 3.9 },
  "assistant_gemma4_adapter_16bit_reasoning": "<16-bit unmerged adapter reasoning trace>",
  "assistant_gemma4_adapter_16bit": "<16-bit unmerged adapter output with thinking>",
  "assistant_gemma4_adapter_16bit_metrics": { "fre": 89.8, "wstf": 3.8 }
}
```

### Pre-download Models on Login Node (`hsuper-login01`)

Because GPU compute nodes do not have internet access, download both the LLM and embedding model weights (`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` and `intfloat/multilingual-e5-base`) to the shared Hugging Face cache on the login node first:

```bash
# Optional: Set HF token if accessing gated models
export HF_TOKEN="hf_..."

# Download models and install textstat & sentence-transformers to shared filesystem
bash scripts/download_model.sh
```

---

## 4. Dynamic Few-Shot RAG Retrieval (`src-eval/dynamic_few_shots.py`)

Provides semantic search over training dataset input/output pairs using **`sentence-transformers`** and **`intfloat/multilingual-e5-base`**:

- Indexes all `user_input` texts from `data/raw/{id}_Standardsprache.txt` mapped to target `assistant` Leichte Sprache.
- Pre-encodes corpus with E5 passage prefix (`passage: <text>`).
- Encodes queries with E5 query prefix (`query: <text>`) and computes cosine similarity.
- Returns top-$k$ semantically closest training demonstrations.

### Usage in Python:
```python
from dynamic_few_shots import get_dynamic_few_shots, format_few_shot_prompt

examples = get_dynamic_few_shots("Wie beantrage ich Wohngeld?", k=3)
formatted_prompt = format_few_shot_prompt(examples)
```

---

## 5. Execution on GPU Compute Node

#### Option 1: Interactive Node (`salloc`)
```bash
salloc --partition=small_gpu8 --gpus 1 --time=00:30:00
ssh <assigned-gpu-node>
cd $HOME/begleit-app-training-gemma4
bash scripts/run_evaluation.sh
```

#### Option 2: Queue with Slurm (`sbatch`)
```bash
sbatch scripts/submit_evaluation.sbatch
```

---

## 4. Gemma 4 MoE Adapter Fine-Tuning (`src-train/config.yml`)

Fine-tunes a LoRA adapter on `google/gemma-4-26b-a4b-it` using **Axolotl** with **DeepSpeed ZeRO-3** CPU offloading across 2+ NVIDIA L40S GPUs.

### Key Architecture & Configuration Decisions

1. **MoE LoRA Module Regex Targeting (SGLang & vLLM Compatibility)**:
   Gemma 4 26B-A4B integrates vision components where bare suffixes like `gate_proj` match vision layers wrapped by `Gemma4ClippableLinear`. Targeting only the language model layers:
   ```yaml
   lora_target_modules: 'model\.language_model\.layers\.[\d]+\.(_checkpoint_wrapped_module\.)?(mlp|self_attn)\.(up|down|gate|q|k|v|o)_proj'
   ```
   ensures PEFT cleanly merges/saves the weights and inference engines like SGLang can load the MoE adapter seamlessly during inference.

2. **Gemma 4 Turn Boundaries**:
   Turn endings are marked by `<turn|>` (`id: 106`), not `<end_of_turn>` (Gemma 3). Configured explicitly via:
   ```yaml
   special_tokens:
     eos_token: "<eos>"
   eot_tokens:
     - "<turn|>"
   ```

3. **Multi-GPU, DeepSpeed ZeRO-3 & Local NVMe Scratch (`$SLURM_TMPDIR`)**:
   - Requires at least 2 GPUs (`scripts/run_training.sh` automatically verifies GPU count and fails fast if `< 2`).
   - DeepSpeed ZeRO-3 offloads optimizer states and parameters to CPU RAM (`src-train/deepspeed_zero3.json`), enabling unquantized bfloat16 training of 26B MoE parameters.
   - When running under Slurm, `scripts/run_training.sh` automatically routes high-IOPS temporary caches (`TMPDIR`, `TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`) to the node's local NVMe SSD (`$SLURM_TMPDIR`), preventing network storage bottlenecks while saving the final adapter to `local/adapters/`.

### Pre-download Training & Evaluation Models

On the **login node** (`hsuper-login01`):
```bash
bash scripts/download_models.sh
```

### Run Combined Fine-Tuning, Merge & FP8 Quantization on GPU Node

The runner script `scripts/run_training.sh` automatically performs the end-to-end pipeline:
1. Distributed Axolotl LoRA fine-tuning across 2+ GPUs -> saves adapter to `local/adapters/gemma-4-26b-a4b-it-lora`.
2. Merges LoRA into base model and compresses to **FP8-Dynamic** via `llmcompressor` (`src-train/merge_and_quantize.py`) -> saves production model to `local/models/gemma-4-26b-a4b-it-fp8`.

#### Option 1: Interactive Node (`salloc`)
```bash
salloc --partition=small_gpu8 --gpus 2 --time=00:45:00
ssh <assigned-gpu-node>
cd $HOME/begleit-app-training-gemma4-adapter-tuning
bash scripts/run_training.sh
```

#### Option 2: Queue with Slurm (`sbatch`)
```bash
sbatch scripts/submit_training.sbatch
```

---

## 5. 5-Pass SGLang Evaluation (`src-eval/evaluation.py`)

Run the 5-pass evaluation on the evaluation split using SGLang:
```bash
sbatch scripts/submit_evaluation.sbatch
# Or interactively on multi-GPU node:
# bash scripts/run_evaluation.sh
```

The script evaluates:
1. Base Model Zero-Shot (`enable_thinking=False`)
2. Base Model Zero-Shot WITH Thinking (`enable_thinking=True`)
3. Base Model Dynamic Few-Shot WITH Thinking
4. Fine-Tuned Merged FP8 Model WITH Thinking (`enable_thinking=True`)
5. Fine-Tuned 16-bit Base Model with Unmerged LoRA Adapter WITH Thinking (`enable_thinking=True`, requires $\ge 2$ GPUs)