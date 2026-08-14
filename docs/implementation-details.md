# Implementation Details

This document records technical details, setup steps, and execution guidelines for running Axolotl fine-tuning, vLLM evaluation, and DVC data management on the **HSUper** GPU cluster (NVIDIA L40S nodes).

---

## 1. Axolotl & vLLM Cluster GPU Smoke Test

### Overview
The smoke test verifies that:
1. The **Axolotl** Docker container (`train_image` in `README.md`) and **vLLM** Docker container (`eval_image` in `README.md`) execute seamlessly under **Apptainer** on HSUper GPU nodes (`small_gpu8`).
2. Axolotl, vLLM, and essential ML frameworks are installed and functional in their respective container environments.
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
   - For vLLM: Executes `python3 /repo/src-eval/smoke_test.py`.

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
   *(This extracts `train_image` and `eval_image` strictly from `README.md` frontmatter and builds `images/axolotl_sandbox` and `images/vllm_sandbox` on the shared filesystem).*

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
  Flash-Attn     : Not Installed
  Triton         : 3.7.0
  vLLM           : Not Installed

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

[STEP 2/2] Executing vLLM Evaluation Container Smoke Test...
[INFO] Container: .../images/vllm_sandbox
[INFO] Using vLLM Container Python: /usr/bin/python3
============================================================
      vLLM Evaluation Cluster Environment Smoke Test
============================================================

[System Info]
  Python Version : 3.12.x
  Platform       : Linux-...

[Inference Core Libraries]
  vLLM           : 0.27.1
  Transformers   : 5.15.0
  Ray            : Not Installed
  TrtLLM         : Not Installed

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
[SUCCESS] Both Training (Axolotl) and Evaluation (vLLM) smoke tests completed!
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

Because system Python, AWS CLI, and DVC are not pre-installed on the login node host OS, the **recommended method** is to use **Apptainer** with a workspace virtual environment (`.dvc-venv`).

#### Step-by-Step Execution on `hsuper-login01`:

1. **SSH to HSUper login node**:
   ```bash
   ssh <hsu-name>@hsuper-login01.hsu-hh.de
   ```

2. **Navigate to repository and pull latest Git changes**:
   ```bash
   cd $HOME/begleit-app-training-gemma4
   git pull
   ```

3. **Initialize workspace DVC environment (one-time setup)**:
   ```bash
   apptainer exec --bind "$PWD:/repo" --pwd /repo images/axolotl_sandbox uv venv /repo/.dvc-venv
   apptainer exec --bind "$PWD:/repo" --pwd /repo images/axolotl_sandbox uv pip install --python /repo/.dvc-venv "dvc[s3]"
   ```

4. **Pull dataset files from AWS S3**:
   ```bash
   apptainer exec --bind "$PWD:/repo" --pwd /repo images/axolotl_sandbox /repo/.dvc-venv/bin/dvc pull
   ```

Once `dvc pull` finishes, `data/raw/` on the shared filesystem is fully populated and immediately ready for offline GPU training on compute nodes!

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

## 3. Gemma 4 Baseline Evaluation (`src-eval/evaluation.py`)

Runs inference on the 10% evaluation dataset (`data/dataset_eval.jsonl`) using **vLLM** with the base model `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` across two modes via native chat template thinking control:
1. **Standard Translation** (`chat_template_kwargs={"enable_thinking": False}`)
2. **Thinking-Enabled Translation** (`chat_template_kwargs={"enable_thinking": True}`)

Produces `data/results.jsonl` with:
```json
{
  "id": "<id>",
  "system": "<system-prompt>",
  "user_input": "<raw input text without template wrapper>",
  "assistant": "<ground-truth Leichte_Sprache>",
  "assistant_gemma4": "<gemma4 output without thinking>",
  "assistant_gemma4_thinking": "<gemma4 output with thinking>"
}
```

### Pre-download Model on Login Node (`hsuper-login01`)

Because GPU compute nodes do not have internet access, download the model weights to the shared Hugging Face cache on the login node first:

```bash
# Optional: Set HF token if accessing gated models
export HF_TOKEN="hf_..."

# Download model to ~/.cache/huggingface on the shared filesystem
bash scripts/download_model.sh RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic
```

### Run Evaluation on GPU Compute Node

#### Option 1: Interactive Node (`salloc`)
```bash
salloc --partition=small_gpu8 --gpus 2 --time=00:30:00
ssh <assigned-gpu-node>
cd $HOME/begleit-app-training-gemma4
bash scripts/run_evaluation.sh
```

#### Option 2: Queue with Slurm (`sbatch`)
```bash
sbatch scripts/submit_evaluation.sbatch
```