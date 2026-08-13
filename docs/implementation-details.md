# Implementation Details

This document records technical details, setup steps, and execution guidelines for running Axolotl fine-tuning and vLLM evaluation on the **HSUper** GPU cluster (NVIDIA L40S nodes).

---

## How to run the Axolotl Cluster GPU Smoke Test

### Overview
The smoke test verifies that:
1. The official Axolotl Docker container executes seamlessly under **Apptainer** on HSUper GPU nodes (`small_gpu8`).
2. Axolotl and essential ML frameworks are installed and functional.
3. PyTorch detects NVIDIA L40S GPUs with full CUDA capability and reports correct VRAM capacity (~48 GB).

---

### How Code & Scripts Are Available on the GPU Node

1. **Shared Network Filesystem (NFS / Lustre)**:
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
   - Mounting to `/repo` (instead of `/workspace`) ensures that the container's built-in virtual environment at `/workspace/axolotl-venv/bin/python` is preserved and accessible.
   - When Apptainer runs, it executes `/workspace/axolotl-venv/bin/python /repo/src-train/smoke_test.py`.

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

* **Prepare Apptainer Container (Login Node only - internet required)**:
   ```bash
   bash scripts/prepare_image.sh
   ```
   *(This downloads an Axolotl image and builds a sandbox directory container `images/axolotl_sandbox` on the shared filesystem. Using sandbox format completely bypasses `mksquashfs` kernel/PRoot restrictions on the login node).*

#### Option 1: Interactive Node

* **Allocate a GPU node** (e.g. 2 GPUs for 5 min):
   ```bash
   salloc --partition=small_gpu8 --gpus 2 --time=00:05:00
   ```

* **Wait for the node and ssh into it** (e.g. node `gpu08` is ready)
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
  squeue --me
  ```

* **View the results**:
  ```bash
  cat logs/smoke_test_*.log
  ``` 

---

### Expected Output Example

```text
============================================================
      Axolotl GPU Cluster Environment Smoke Test
============================================================

[System Info]
  Python Version : 3.11.x
  Platform       : Linux-x86_64-...

[ML Core Libraries]
  Axolotl        : 0.4.x / 0.5.x
  Transformers   : 4.x.x
  PEFT           : 0.x.x
  Accelerate     : 0.x.x
  BitsAndBytes   : 0.x.x
  Triton         : 3.x.x
  vLLM           : ...

[PyTorch & CUDA Environment]
  PyTorch        : 2.x.x+cu12x
  CUDA Available : True
  CUDA Version   : 12.x
  cuDNN Version  : 9x / 8x
  GPU Count      : 2

[GPU Details]
  --- GPU 0: NVIDIA L40S ---
      Compute Capability : 8.9
      SM Count           : 142
      Total VRAM         : 47.51 GB
      Free VRAM          : 47.10 GB
  --- GPU 1: NVIDIA L40S ---
      Compute Capability : 8.9
      SM Count           : 142
      Total VRAM         : 47.51 GB
      Free VRAM          : 47.10 GB
============================================================
```