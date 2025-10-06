# rnencodec/utils/downloads.py
from __future__ import annotations
import hashlib, tarfile, zipfile
from pathlib import Path
from urllib.request import urlretrieve

__all__ = ["fetch", "sha256sum"]

def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _maybe_extract(archive: Path, target_dir: Path) -> None:
    # Extract .zip or .tar.gz/.tgz
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target_dir)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(target_dir)

def fetch(
    url: str,
    dest: Path,
    *,
    sha256: str | None = None,
    extract: bool = False,
    force: bool = False,
) -> Path:
    """
    Download to 'dest' (creating parent dirs). Verify sha256 if provided.
    Optionally extract archives into dest.parent.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if force or not dest.exists():
        print(f"Downloading {url} → {dest}")
        urlretrieve(url, dest)

    if sha256:
        got = sha256sum(dest)
        if got != sha256:
            try:
                dest.unlink()
            except FileNotFoundError:
                pass
            raise RuntimeError(
                f"SHA256 mismatch for {dest}\n expected {sha256}\n got      {got}"
            )

    if extract:
        _maybe_extract(dest, dest.parent)
    return dest
