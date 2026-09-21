---
name: cluster-usage
description: Instructions and best practices for interacting with the HSUper GPU cluster, including data pulling, running jobs, environment setup, and monitoring.
---

# Cluster Usage Skill (HSUper GPU Cluster)

This skill documents how to interact with the HSUper GPU cluster (`hsuper-login01.hsu-hh.de`) for training, evaluation, and data management.

## 1. Golden Rule: Use Existing Scripts

**Always use the provided scripts in `scripts/` instead of executing raw, ad-hoc commands on the cluster.**

The repository provides tailored helper scripts that handle Apptainer container binding, environment setup, and batch execution properly:

| Task | Helper Script | Where to Run | Description |
| :--- | :--- | :--- | :--- |
| **Pull Data (Standard)** | `bash scripts/pull_data.sh` | **Login Node** (`hsuper-login01`) | Pulls standard DVC data from S3 using Apptainer and `.dvc-venv`. |
| **Pull Data (Dialogs)** | `bash scripts/pull_data_dialogs.sh` | **Login Node** (`hsuper-login01`) | Pulls raw dialogs and stage outputs (`prepare_data_dialogs`, `prepare_few_shots_dialogs`). |
| **Prepare Images** | `bash scripts/prepare_images.sh` | **Login Node** (`hsuper-login01`) | Pulls and extracts Apptainer sandbox images. |
| **Download Models** | `bash scripts/download_models.sh` | **Login Node** (`hsuper-login01`) | Downloads base HuggingFace models to cluster cache. |
| **Submit Training (Standard)** | `sbatch scripts/submit_training.sbatch` | **Login Node** | Submits Axolotl LoRA training job. |
| **Submit Training (Dialogs)** | `sbatch scripts/submit_training_dialogs.sbatch` | **Login Node** | Submits dialog LoRA training + merge + FP8 quantization. |
| **Submit Evaluation (Standard)** | `sbatch scripts/submit_evaluation.sbatch` | **Login Node** | Submits SGLang evaluation job. |
| **Submit Evaluation (Dialogs)** | `sbatch scripts/submit_evaluation_dialogs.sbatch` | **Login Node** | Submits dialog SGLang evaluation job. |

---

## 2. Cluster Architecture & Internet Access

- **Login Node (`hsuper-login01.hsu-hh.de`)**:
  - Has outbound internet access.
  - Used for `git pull`, DVC data pulling (`scripts/pull_data*.sh`), model downloads (`scripts/download_models.sh`), and job submissions (`sbatch`).
  - **Never** run heavy compute or training jobs directly on the login node.
- **Compute Nodes (`small_gpu8` partition, NVIDIA L40S)**:
  - **No outbound internet access**.
  - All data (`data/*.jsonl`), models, and Apptainer sandboxes must be present on the shared BeeGFS filesystem before job start.
  - Offline flags (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) are enforced.

---

## 3. Workflow: Syncing & Running Jobs

When changes are made locally:

1. **Commit & Push locally**:
   ```bash
   git push origin <branch>
   AWS_PROFILE=diwop-production dvc push  # if new data was produced
   ```

2. **Sync on Cluster Login Node**:
   ```bash
   ssh wulfc@hsuper-login01.hsu-hh.de "cd ~/begleit-app-training-gemma4 && git pull origin <branch>"
   ```

3. **Pull Data on Login Node**:
   ```bash
   # For dialogs:
   ssh wulfc@hsuper-login01.hsu-hh.de "cd ~/begleit-app-training-gemma4 && bash scripts/pull_data_dialogs.sh"
   ```

4. **Submit Slurm Jobs**:
   - Training:
     ```bash
     ssh wulfc@hsuper-login01.hsu-hh.de "cd ~/begleit-app-training-gemma4 && sbatch scripts/submit_training_dialogs.sbatch"
     ```
   - Evaluation with Dependency on Training:
     ```bash
     ssh wulfc@hsuper-login01.hsu-hh.de "cd ~/begleit-app-training-gemma4 && sbatch --dependency=afterok:<TRAIN_JOB_ID> scripts/submit_evaluation_dialogs.sbatch"
     ```

5. **Monitor Jobs**:
   ```bash
   ssh wulfc@hsuper-login01.hsu-hh.de "squeue -u wulfc"
   ```
