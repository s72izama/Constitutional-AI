#!/bin/bash
#SBATCH --job-name=testenv
#SBATCH --partition=A40devel
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:01:00
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

python test_env.py