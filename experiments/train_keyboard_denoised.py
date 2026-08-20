"""
Train RNeNcodec on the denoised keyboard corpus, at 12 kbps / 16 codebooks.

Lives in experiments/ rather than training/ so that everything this fork adds is
separated from upstream code, keeping later merges from lonce/RNeNcodec clean.

WHY 16 CODEBOOKS. This model's output feeds the nac-bwe bandwidth-extension
models, and every one of those trained on EnCodec latents at 12 kbps. EnCodec
24 kHz spends 750 bps per codebook, so 12.0 kbps = 16 codebooks while RNeNcodec's
dataprep default of 6.0 kbps = 8. The earlier keyboard model was built at 8, which
silently mismatched the enhancer: the latent is 128-d at either setting, so nothing
raises — the BWE model simply receives latents far coarser than anything it saw in
training. The dataset consumed here was prepared with
`quick_encode(..., bandwidth=12.0)`, and `train_model` auto-detects n_q from the
data, so the cascade builds 16 heads to match.

COST OF THAT CHOICE: the output cascade doubles (8 -> 16 heads), making this model
bigger and slower at inference than the 8-codebook one — and RNeNcodec is already
the real-time bottleneck in the live demo. Worth measuring rather than assuming.

Hyperparameters follow the previous keyboard run (quickstart/output_keyboard_12kbps,
config_v2.json): hidden 128, 3 GRU layers, soft cascade, tau_soft 0.6,
sequence_length 125. Only the corpus and the codebook count differ, so this is a
clean successor to that run.

! TINY CORPUS: four ~68 s takes, ~4.6 minutes, and the dataset carries a single
  'train' split with no validation set. Training loss here is NOT a generalisation
  measure. Whether this model synthesises or merely replays its four takes is an
  open question the downstream evaluation must measure directly — it bounds how
  strongly the enhancer comparison can be stated.

Usage (from the repo root):
    python experiments/train_keyboard_denoised.py \
        --dataset /path/to/data/keyboard_denoised \
        --out runs/keyboard_denoised_12kbps
"""

import argparse
import sys
from pathlib import Path

# Repo root on sys.path so `training` / `rnencodec` import without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _detect_n_q(dataset_root: Path) -> int:
    """Codebooks in the corpus, read from the data rather than from a placeholder.

    Loads one sample with a slice width of 32 -- the most EnCodec 24 kHz can emit
    -- so the slice never truncates and the returned width is the true n_q.
    """
    from rnencodec.audioDataLoader.audio_dataset import (
        EnCodecLatentDataset_dynamic, LatentDatasetConfig,
    )
    import json
    cc = json.loads((dataset_root / "hf_dataset" / "conditioning_config.json").read_text())
    cfg = LatentDatasetConfig(
        dataset_path=str(dataset_root / "hf_dataset"),
        sequence_length=8,
        parameter_specs={n: None for n in cc["feature_names"]},
        add_noise=False, noise_weight=0.0, codebook_size=1024,
        n_q=32, clamp_val=15, filters={}, files_per_sequence=1,
        cond_root=None, cond_suffix=".cond.npy", strict=False,
    )
    ds = EnCodecLatentDataset_dynamic(cfg, "facebook/encodec_24khz", split="train")
    _, target = ds[0]
    return int(target.shape[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="Dataset ROOT (the dir containing hf_dataset/), NOT hf_dataset itself")
    ap.add_argument("--out", default="runs/keyboard_denoised_12kbps",
                    help="Model output dir (checkpoints/, tensorboard/, config_v2.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--save-interval", type=int, default=10)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--sequence-length", type=int, default=125)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--batches-per-epoch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--tau-soft", type=float, default=0.6)
    ap.add_argument("--resume", default=None, help="Previous run dir to resume from")
    ap.add_argument("--expect-n-q", type=int, default=16,
                    help="Fail unless the dataset really has this many codebooks "
                         "(0 disables). Guards the whole point of this run.")
    args = ap.parse_args()

    ds = Path(args.dataset)
    cfg = ds / "hf_dataset" / "conditioning_config.json"
    if not cfg.exists():
        raise SystemExit(
            f"no conditioning_config.json at {cfg}\n"
            f"--dataset must be the dataset ROOT (containing hf_dataset/), "
            f"not hf_dataset itself."
        )

    import torch
    from training.loop import train_model

    # Verify the codebook count BEFORE training, because getting it wrong is
    # silent. `train_model` derives n_q by slicing a sample to a placeholder and
    # reading the shape back, so a too-small placeholder simply reports itself --
    # which is how a 16-codebook corpus was trained as 8 without any warning.
    # A whole run at the wrong n_q is unusable for the enhancer, so check first.
    if args.expect_n_q:
        n_q = _detect_n_q(ds)
        if n_q != args.expect_n_q:
            raise SystemExit(
                f"dataset has n_q={n_q}, expected {args.expect_n_q}.\n"
                f"The BWE enhancer models require 16 (EnCodec 24 kHz @ 12 kbps). "
                f"Re-run the dataprep with quick_encode(..., bandwidth=12.0), or "
                f"pass --expect-n-q {n_q} if you really mean to train at this rate."
            )
        print(f"codebooks: {n_q} (as expected)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"device:  {device}{gpu}", flush=True)
    print(f"dataset: {ds.resolve()}", flush=True)
    print(f"output:  {Path(args.out).resolve()}", flush=True)

    train_model(
        dataset_path      = str(ds),
        model_output_path = args.out,
        num_epochs        = args.epochs,
        batch_size        = args.batch_size,
        sequence_length   = args.sequence_length,
        batches_per_epoch = args.batches_per_epoch,
        learning_rate     = args.lr,
        hidden_size       = args.hidden_size,
        num_layers        = args.num_layers,
        cascade_mode      = "soft",     # deterministic / RNG-free, as in the prior run
        tau_soft          = args.tau_soft,
        save_interval     = args.save_interval,
        resume_checkpoint = args.resume,
        device            = device,
        use_tensorboard   = True,
        use_tqdm          = False,      # tqdm bars flood a SLURM .out file
    )


if __name__ == "__main__":
    main()
