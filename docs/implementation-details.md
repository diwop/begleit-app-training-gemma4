# Implementation Details

This document records technical details, setup steps, and execution guidelines for running Axolotl fine-tuning and vLLM evaluation on the **HSUper** GPU cluster (NVIDIA L40S nodes).

---

## How to run the Axolotl & vLLM Cluster GPU Smoke Test

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