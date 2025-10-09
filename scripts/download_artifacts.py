#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, sys, tarfile, tempfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# ---- Version pin for your hosted artifacts ----
ARTIFACT_VERSION = "v0.1"


# --- URLs on YOUR server (replace domain/path) ---
SERVER_BASE = f"https://animatedsound.com/RNeNcodec/{ARTIFACT_VERSION}"
WEIGHTS_URL  = f"{SERVER_BASE}/weights/waterfill_quickstart.pt"
CONFIG_URL    = f"{SERVER_BASE}/weights/config_v2.pt" 
DATASET_URL  = f"{SERVER_BASE}/data/waterfill_quickstart_hf_dataset.tar.gz"

# --- Put the REAL SHA256 you compute for each file here ---
WEIGHTS_SHA256 = "2880edd259b5f7b926ad5c5c825025cfddd206f2107d18c2d1c6de592ac9f04a"
CONFIG_SHA256  = "27343fe23169044e377b52aa10ff97a99a53964a7b7169b92c377536e3447cd4"                  # <-- update me
DATASET_SHA256 = "30d3726cf700256b561cacf3a8b9bffb965ddd71a8d2e5b922ffb2ab7b93e309"

# ---------------------------------------------------------------------------

def sha256_file(path: Path, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, dest: Path, *, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[skip] {dest} already exists (use --force to re-download)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        print(f"[get] {url}")
        req = Request(url, headers={"User-Agent": "rnencodec-downloader"})
        with urlopen(req) as r, tmp.open("wb") as f:
            total = int(r.headers.get("Content-Length", "0") or 0)
            read = 0
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk: break
                f.write(chunk); read += len(chunk)
                if total:
                    pct = read * 100 // total
                    print(f"\r  {read/1e6:6.1f}/{total/1e6:.1f} MB ({pct:3d}%)", end="", flush=True)
        if total: print()
        tmp.replace(dest)
        print(f"[ok ] saved to {dest}")
    except (HTTPError, URLError) as e:
        if tmp.exists(): tmp.unlink(missing_ok=True)
        raise SystemExit(f"[err] download failed: {e}")

def verify(path: Path, expect: str, label: str) -> None:
    print(f"[hash] verifying {label} sha256…")
    got = sha256_file(path)
    if got != expect:
        raise SystemExit(f"[err] sha256 mismatch for {label}:\n  expected {expect}\n  got      {got}")
    print(f"[ok ] {label} checksum verified")

def extract_tar_gz(archive: Path, outdir: Path) -> None:
    print(f"[untar] {archive} -> {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        # prevent path traversal
        for m in tar.getmembers():
            dest = (outdir / m.name).resolve()
            if not str(dest).startswith(str(outdir.resolve())):
                raise SystemExit(f"[err] unsafe member in tar: {m.name}")
        tar.extractall(outdir)
    print("[ok ] dataset extracted")

def main() -> None:
    p = argparse.ArgumentParser(description="Download model weights+config and/or example dataset")
    p.add_argument("--weights", action="store_true", help="download pretrained weights (also downloads config_v2.pt)")
    p.add_argument("--dataset", action="store_true", help="download example dataset")
    p.add_argument("--all",     action="store_true", help="download weights+config and dataset")
    p.add_argument("--force",   action="store_true", help="re-download even if present")
    p.add_argument("--dest-root", default="artifacts", help="destination root (default: artifacts/)")
    args = p.parse_args()
    if args.all:
        args.weights = args.dataset = True

    root = Path(args.dest_root)

    if args.weights:
        # weights
        wpath = root / "weights" / "waterfill_quickstart.pt"
        download(WEIGHTS_URL, wpath, force=args.force)
        verify(wpath, WEIGHTS_SHA256, "weights")
        # config sidecar (always with weights)
        cpath = root / "weights" / "config_v2.pt"
        download(CONFIG_URL, cpath, force=args.force)
        verify(cpath, CONFIG_SHA256, "config_v2.pt")

    if args.dataset:
        archive = root / "data" / "waterfill_quickstart_hf_dataset.tar.gz"
        outdir  = root / "data"
        download(DATASET_URL, archive, force=args.force)
        verify(archive, DATASET_SHA256, "dataset")
        extract_tar_gz(archive, outdir)

    if not (args.weights or args.dataset or args.all):
        p.print_help()

if __name__ == "__main__":
    main()