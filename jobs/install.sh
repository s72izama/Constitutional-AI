#!/bin/bash
#SBATCH --job-name=install-myvenv
#SBATCH --partition=A40short
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/install-%j.out
#SBATCH --error=logs/install-%j.err

# bash variables
export MY_VENV="qwen_env_gpu"
export MAX_JOBS=8

set -euo pipefail

# 1. Setup NVMe Temp Space
# Using the path provided by your cluster docs
export NVME_TMP="/local/nvme/${USER}_${SLURM_JOB_ID}"
mkdir -p "$NVME_TMP"
export TMPDIR="$NVME_TMP"
export PIP_CACHE_DIR="$NVME_TMP/pip_cache"
mkdir -p "$PIP_CACHE_DIR"

echo "--- Storage & Speed Check ---"
echo "Local NVMe SSD Path: $NVME_TMP"
df -h "$NVME_TMP"
echo "-----------------------------"

cd "$SLURM_SUBMIT_DIR"

module purge
module load Python/3.11.3-GCCcore-12.3.0 CUDA/12.4.0 

# Create venv if missing
if [ ! -d $MY_VENV ]; then
  python -m venv $MY_VENV
fi
source $MY_VENV/bin/activate

pip install --upgrade pip setuptools wheel packaging --no-cache-dir

# vLLM with matching CUDA version 12.4.0
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

pip install transformers datasets accelerate sentencepiece protobuf --no-cache-dir

pip install flash-attn --no-build-isolation --no-cache-dir

# Remove pip download cache (optional)
pip cache purge

# ==========================================================
# VALIDATION SECTION (optional)
# ==========================================================
echo "--- Validating Installation ---"
python test_env.py
echo "--- Validation Complete ---"