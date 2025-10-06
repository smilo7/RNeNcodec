# scripts/download_artifacts.py
from __future__ import annotations
import argparse
from pathlib import Path
from rnencodec.utils.downloads import fetch

# --- Version pin for artifacts you host ---
ARTIFACT_VERSION = "v0.1"

# --- URLs on YOUR server (replace domain/path) ---
SERVER_BASE = f"https://animatedsound.com/RNeNcodec/{ARTIFACT_VERSION}"

WEIGHTS_URL  = f"{SERVER_BASE}/weights/waterfill_quickstart.pt"
DATASET_URL  = f"{SERVER_BASE}/data/waterfill_quickstart_dataset.tar.gz"

# --- Put the REAL SHA256 you compute for each file here ---
WEIGHTS_SHA256 = "2880edd259b5f7b926ad5c5c825025cfddd206f2107d18c2d1c6de592ac9f04a"
DATASET_SHA256 = "30d3726cf700256b561cacf3a8b9bffb965ddd71a8d2e5b922ffb2ab7b93e309"

def main():
    ap = argparse.ArgumentParser(description="Download model weights / example dataset")
    ap.add_argument("--weights", action="store_true", help="download pretrained weights")
    ap.add_argument("--dataset", action="store_true", help="download example dataset")
    ap.add_argument("--all", action="store_true", help="download both")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    if args.all:
        args.weights = args.dataset = True

    root = Path("artifacts")
    if args.weights:
        fetch(
            url=WEIGHTS_URL,
            dest=(root / "weights" / "rnencodec_quickstart.ckpt"),
            sha256=WEIGHTS_SHA256,
            extract=False,
            force=args.force,
        )

    if args.dataset:
        fetch(
            url=DATASET_URL,
            dest=(root / "data" / "example_hf_dataset.tar.gz"),
            sha256=DATASET_SHA256,
            extract=True,   # will unpack into artifacts/data/
            force=args.force,
        )

if __name__ == "__main__":
    main()
