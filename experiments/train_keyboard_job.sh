#!/bin/bash
# train_keyboard_job.sh — SLURM job for experiments/train_keyboard_denoised.py
#
# Trains RNeNcodec on the denoised keyboard corpus at 12 kbps / 16 codebooks, so
# its output matches what the nac-bwe enhancer models expect (see the driver's
# docstring for why the codebook count matters).
#
# RNeNcodec is NOT pip-installed here. The driver puts the repo root on sys.path
# itself, so this job never mutates the shared conda env — which matters because
# other jobs are running in that same env off shared GPFS home.
#
# Submit from this repo's root on the HPC:
#   sbatch experiments/train_keyboard_job.sh <dataset_root> [out_dir]
# e.g.
#   sbatch experiments/train_keyboard_job.sh \
#          ~/encodec_bandwidth_extension/data/keyboard_denoised
#
# Add --exclude to dodge busy nodes, remembering that a command-line --exclude
# REPLACES the #SBATCH one below, so carry the Pascal nodes forward yourself:
#   sbatch --exclude=node031,node032,node020,node023 experiments/train_keyboard_job.sh <ds>

#SBATCH -J rnenc_kbd
#SBATCH -p medium
# This cluster's torch build has no kernels for Pascal GPUs (sm_60/sm_61); jobs
# landing on node031 (P100) or node032 (GTX 1080 Ti) die with "no kernel image is
# available for execution on the device". Everything else (T4 sm_75, RTX 6000
# sm_75, L40S sm_89) works.
#SBATCH --gres=gpu:1
#SBATCH --exclude=node031,node032
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH -o logs/%J.out
#SBATCH -e logs/%J.err

set -e

DATASET=${1:?usage: sbatch experiments/train_keyboard_job.sh <dataset_root> [out_dir]}
OUT=${2:-runs/keyboard_denoised_12kbps}

module load Miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate encodec_enhancer_baseline

mkdir -p logs

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Dataset: $DATASET"
echo "Output:  $OUT"

# Fail fast and legibly if the env lacks something, rather than dying deep inside
# the training loop after the scheduler has already spent its queue wait.
python - <<'PY'
import importlib.util, sys   # NOT `import importlib`: util is a submodule and is
                             # not imported as a side effect of importing the package
missing = [m for m in ("torch", "transformers", "datasets", "soundfile", "numpy")
           if not importlib.util.find_spec(m)]
if missing:
    sys.exit(f"missing packages in this env: {', '.join(missing)}")
import torch
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
PY

python -u experiments/train_keyboard_denoised.py \
    --dataset "$DATASET" \
    --out "$OUT" \
    --epochs 100 \
    --save-interval 10

echo "=== done: $(date) ==="
