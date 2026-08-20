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

# RNeNcodec's requirements.txt minus the torch trio (torch/torchaudio/torchvision):
# those are already present from the clone in the cluster's known-good CUDA build,
# and letting pip resolve them is exactly how the "no kernel image" failure gets
# reintroduced. Everything listed here is pure-Python or a self-contained wheel.
#
# librosa is NOT optional even for training: rnencodec/__init__.py imports the
# generator, which imports librosa at module scope, so `from training.loop import
# train_model` fails without it.
#
# Deliberately skipped: sounddevice (needs PortAudio, real-time playback only) and
# jupyterlab/ipywidgets (notebook UI) — neither is reachable from a training run.
echo "=== installing RNeNcodec deps into $NEW_ENV (torch left untouched) ==="
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
pip install -q --no-input \
    datasets \
    librosa lazy_loader soxr resampy audioread soundfile scipy \
    tensorboard matplotlib safetensors huggingface-hub tqdm pyyaml

TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
echo "torch before: $TORCH_BEFORE"
echo "torch after:  $TORCH_AFTER"
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
    echo "!! pip changed the torch build ($TORCH_BEFORE -> $TORCH_AFTER)."
    echo "!! That is the documented route to 'no kernel image is available'. Aborting."
    exit 1
fi

echo "=== verification ==="
python - <<'PY'
import importlib.util as u
need = ["torch", "transformers", "datasets", "soundfile", "numpy", "scipy",
        "pyarrow", "fsspec", "huggingface_hub", "librosa", "soxr", "resampy",
        "lazy_loader", "audioread", "tensorboard", "matplotlib"]
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
