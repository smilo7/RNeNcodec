#!/bin/bash
# setup_hpc_env.sh — build a dedicated conda env for RNeNcodec on the SMC cluster.
#
# WHY A SEPARATE ENV. RNeNcodec needs HuggingFace `datasets` (it loads its corpus
# via load_from_disk); the nac-bwe env `encodec_enhancer_baseline` does not have
# it. Installing into that env would mutate an environment that other jobs are
# actively running out of, on shared GPFS home — so RNeNcodec gets its own.
#
# WHY CLONE RATHER THAN BUILD FRESH. This cluster's torch build is particular:
# it has no kernels for Pascal GPUs, and the project README is explicit that
# letting pip resolve torch can replace the working CUDA build and reproduce the
# "no kernel image is available for execution on the device" failure. Cloning the
# known-good env inherits exactly that working build, so the only new thing is
# `datasets` and its pure-Python dependencies. conda hardlinks identical packages,
# so the clone costs far less than the 7.6 GB the source env reports.
#
# Run once:
#   sbatch experiments/setup_hpc_env.sh
# Then check logs/<jobid>.out for the verification block at the end.

#SBATCH -J rnenc_env
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o logs/%J.out
#SBATCH -e logs/%J.err

set -e

SRC_ENV=${1:-encodec_enhancer_baseline}
NEW_ENV=${2:-rnencodec}

module load Miniconda3/4.9.2
eval "$(conda shell.bash hook)"

mkdir -p logs

if conda env list | awk '{print $1}' | grep -qx "$NEW_ENV"; then
    echo "env '$NEW_ENV' already exists — skipping clone"
else
    echo "=== cloning $SRC_ENV -> $NEW_ENV ==="
    conda create -y --clone "$SRC_ENV" -n "$NEW_ENV"
fi

conda activate "$NEW_ENV"

# --no-deps is NOT used here: datasets legitimately needs pyarrow/dill/xxhash/
# multiprocess/fsspec, which are pure-Python or self-contained wheels. torch is
# already satisfied in the clone, so pip will not try to touch it. Pinning below
# torch's requirement is unnecessary; just do not let anything upgrade torch.
echo "=== installing datasets into $NEW_ENV ==="
pip install -q "datasets" --no-input
python - <<'PY'
import torch
print("torch after install:", torch.__version__, "| cuda build:", torch.version.cuda)
PY

echo "=== verification ==="
python - <<'PY'
import importlib.util as u
need = ["torch", "transformers", "datasets", "soundfile", "numpy",
        "pyarrow", "fsspec", "huggingface_hub"]
missing = [m for m in need if not u.find_spec(m)]
for m in need:
    print(f"  {'OK  ' if u.find_spec(m) else 'MISS'} {m}")
import torch, numpy
print(f"\ntorch {torch.__version__} | numpy {numpy.__version__} | cuda avail {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    torch.zeros(1, device="cuda")          # the real test: are there kernels for this arch?
    print("cuda kernel check: OK")
raise SystemExit(f"MISSING: {missing}" if missing else 0)
PY

echo "=== env '$NEW_ENV' ready ==="
