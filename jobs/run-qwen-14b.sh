#!/bin/bash
#SBATCH --job-name=qwen14b
#SBATCH --partition=A40devel
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs cache .hf

export MY_VENV="qwen_env_gpu"

module purge
module load Python/3.11.3-GCCcore-12.3.0
source $MY_VENV/bin/activate

# HuggingFace cache
export HF_HOME="$SLURM_SUBMIT_DIR/.hf"

# vLLM multiprocessing safety on HPC
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- Model config (matches your main.py env vars) ---
export MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
export VLLM_DTYPE="bfloat16"
export MAX_MODEL_LEN="4096"          # start conservative; increase after it runs
export GPU_MEM_UTIL="0.90"
export TP="1"                        # MUST match --gres=gpu:x
export ENFORCE_EAGER="0"             # faster on A100

# Optional: allocator hint (new name)
export PYTORCH_ALLOC_CONF=expandable_segments:True

python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
print("GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU0:", torch.cuda.get_device_name(0))
PY

python src/main.py
