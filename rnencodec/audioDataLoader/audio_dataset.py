# audio_dataset.py
# Refactor: shared base + constant/dynamic subclasses.
# - EnCodecLatentDataset_constant: (params from Arrow, one per file, spread over frames)
# - EnCodecLatentDataset_dynamic:  per-frame params from sidecar <basename>.cond.npy (+ .json metadata)
#
# Back-compat alias:
#   EnCodecLatentDataset = EnCodecLatentDataset_constant
#
# Notes:
# - Dynamic class: uses sidecar metadata "names" to select columns; normalizes to [0,1] using JSON "norm.min/max".
# - Dynamic class: complains if parameter_specs provides non-None (min,max); specs are used only to choose which params.

import os
import re
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Callable, Optional, Any, Set, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from datasets.features import ClassLabel


# ------------------------------- Types / Config -------------------------------
SIDECAR_JSON_SUFFIX = ".json"

# type alias
FilterSpec = Dict[str, Union[Tuple[float, float], Set[Any]]]

@dataclass
class LatentDatasetConfig:
    dataset_path: str                              # Path to HF dataset folder (load_from_disk)
    sequence_length: int                           # Number of frames per training sequence
    parameter_specs: Dict[str, Tuple[float, float]]  # {'param': (min, max), ...} or (None,None) in dynamic
    add_noise: bool = False                        # Whether to add noise to latents
    noise_weight: float = 0.1                      # Weight for noise injection
    codebook_size: int = 1024                      # Number of possible target values per codebook
    n_q: int = 4                                   # Number of codebooks to use
    clamp_val: float = 15
    filters: Optional[FilterSpec] = None           # e.g. {"foo": {4,6,7}, "bar": (0.0, 3.0), "label": {"A","B"}}
    files_per_sequence: int = 2

    # --- Used only by the dynamic (per-frame) dataset ---
    # If cond_root is None, we look for sidecars next to the .ecdc with the given suffix.
    cond_root: Optional[str] = None
    cond_suffix: str = ".cond.npy"
    # If True, dynamic dataset raises on missing sidecar/mismatch; else it skips those rows.
    strict: bool = True


# ------------------------------- EnCodec helpers ------------------------------

def preprocess_latents_for_RNN(latents, clamp_val):
    """Convert latents to [-1,1] range with optional clamping"""
    if clamp_val != 0:
        return torch.clamp(latents, -clamp_val, clamp_val) / clamp_val
    return latents

def efficient_codes_to_latents(model, codes):
    """
    Efficient version for repeated use (e.g. in DataLoaders).
    Input codes: (1, n_q, T)  or (n_q, 1, T) depending on caller; we handle both.
    """
    model.eval()
    with torch.no_grad():
        if codes.shape[0] == 1:  # batch dimension first
            codes_transposed = codes.transpose(0, 1)  # (n_q, batch, time)
        else:
            codes_transposed = codes
        embeddings = model.quantizer.decode(codes_transposed)
        return embeddings

@torch.no_grad()
def latents_to_audio_simple(model, embeddings):
    """
    EnCodec 24kHz mono: 128-D latents -> waveform. No normalization, no PQMF.

    Args:
        model: EnCodec model (24 kHz, mono).
        embeddings: (B, T_frames, 128) or (B, 128, T_frames)

    Returns:
        wave: (B, 1, T_audio) float32 on CPU
        sr:   int (should be 24000 for your model)
    """
    z = embeddings
    if z.dim() != 3:
        raise ValueError(f"Expected (B, T, 128) or (B, 128, T), got {tuple(z.shape)}")
    if z.shape[1] != 128 and z.shape[2] == 128:
        z = z.transpose(1, 2)
    x = model.decoder(z)
    if isinstance(x, (tuple, list)):
        x = x[0]
    elif isinstance(x, dict):
        x = x.get("x", next(iter(x.values())))
    x = x.detach().to(torch.float32).cpu()
    sr = int(getattr(model, "sample_rate", 24000))
    return x, sr


# ------------------------------ Filter utilities -----------------------------

def _normalize_filter_spec(filters):
    """
    Normalize a user filter spec like:
      {"foo": (0.0, 3.0), "bar": {4, 6, 7}, "baz": [1, 2]}
    into:
      {"foo": ("range", 0.0, 3.0), "bar": ("set", {4,6,7}), "baz": ("set", {1,2})}
    """
    if not filters:
        return {}
    norm = {}
    for key, rule in filters.items():
        if isinstance(rule, tuple) and len(rule) == 2:
            lo, hi = rule
            norm[key] = ("range", float(lo), float(hi))
        else:
            try:
                norm[key] = ("set", set(rule))
            except TypeError:
                norm[key] = ("set", {rule})
    return norm

def _apply_hf_filters(ds, filters):
    """
    Apply normalized filters to a Hugging Face Dataset once (batched).
    """
    spec = _normalize_filter_spec(filters)
    if not spec:
        return ds
    import numpy as _np
    def keep(batch):
        length = len(next(iter(batch.values())))
        mask = _np.ones(length, dtype=bool)
        for col, rule in spec.items():
            if col not in batch:
                mask &= False
                continue
            v = _np.asarray(batch[col])
            kind = rule[0]
            if kind == "range":
                _, lo, hi = rule
                try:
                    vf = v.astype(_np.float32)
                except Exception:
                    mask &= False
                    continue
                mask &= (vf >= lo) & (vf <= hi)
            else:
                _, allowed = rule
                mask &= _np.isin(v, list(allowed))
        return mask
    return ds.filter(keep, batched=True)


# ------------------------------- Misc utilities ------------------------------
# used for creating sequences from n different data files

def _split_even(total: int, k: int) -> List[int]:
    """
    Split 'total' into k integers as evenly as possible.
    Earlier segments get the +1 if there's a remainder.
    E.g., total=7, k=3 -> [3, 2, 2]
    """
    k = max(1, int(k))
    base = total // k
    r = total % k
    return [base + (1 if i < r else 0) for i in range(k)]


# ------------------------------ Shared base class ----------------------------

class _BaseEnCodecLatentDataset(Dataset):
    """
    Shared dataset logic:
      - load HF split
      - optional filters
      - build sequence_map with path resolution
      - per-item: slice codes, latents, targets, noise, concat conditioning
    Subclasses implement: _cond_for_segment(...) and (optionally) row validation hook.
    """

    def __init__(self, config: LatentDatasetConfig, encodec_model_path, split='train'):
        # Load EnCodec on CPU (matches your current behavior)
        from transformers import EncodecModel
        self.model = EncodecModel.from_pretrained(encodec_model_path)
        self.model.eval()

        self.config = config
        self.sequence_length = config.sequence_length
        self.parameter_specs = config.parameter_specs or {}
        self.n_q = config.n_q
        self.split = split
        self.clamp_val = self.config.clamp_val

        # HF dataset
        self.dataset = load_from_disk(config.dataset_path)[split]
        self.dataset_root = Path(config.dataset_path)

        self.files_per_sequence = getattr(self.config, "files_per_sequence", 2)

        # Filters
        total_before = len(self.dataset)
        if config.filters:
            self.dataset = _apply_hf_filters(self.dataset, config.filters)
            total_after = len(self.dataset)
            pct = (100.0 * total_after / total_before) if total_before else 0.0
            print(f"[EnCodecLatentDataset] Filters applied: kept {total_after} of {total_before} rows ({pct:.1f}%).")
        else:
            print(f"[EnCodecLatentDataset] No filters provided; using all {total_before} rows.")

        # Build sequence map
        self.sequence_map: List[tuple] = []
        k = max(1, int(self.files_per_sequence))
        longest_seg = (self.sequence_length + k - 1) // k
        needed = longest_seg + 1  # +1 for next-step target shift

        for dataset_idx, row in enumerate(self.dataset):
            # Resolve .ecdc path
            audio_path = row['audio']
            possible_paths = [
                self.dataset_root / audio_path,                   # audio already includes split
                self.dataset_root / self.split / audio_path,      # add split if needed
                Path(audio_path),                                  # absolute path
            ]
            token_file_path = None
            for p in possible_paths:
                if p.exists():
                    token_file_path = p
                    break
            if token_file_path is None:
                print("Warning: Could not find audio file for any of:")
                for p in possible_paths:
                    print(f"  - {p}")
                continue

            # Load codes to get T and optionally validate per subclass
            codes = self._load_ecdc_codes(token_file_path)
            if codes is None:
                continue
            num_frames = codes.shape[-1]  # (1, n_q, T)

            if not self._validate_row_for_subclass(token_file_path, num_frames):
                # Dynamic subclass may skip rows without valid sidecar/matching frames
                continue

            # Create all valid starts
            if num_frames >= needed:
                max_start = num_frames - needed + 1
                for start_frame in range(max_start):
                    self.sequence_map.append((dataset_idx, start_frame, token_file_path))

        print(f"Loaded {len(self.sequence_map)} sequences from {len(self.dataset)} files in '{split}' split")

    # ---- Base: standard __len__/__getitem__ pipeline (identical for both classes) ----

    def __len__(self):
        return len(self.sequence_map)

    def __getitem__(self, idx):
        # How many separate files to mix into one sequence
        k = max(1, int(self.files_per_sequence))

        # Pick k sequences: 1 anchored + (k-1) random distinct if possible
        picks = [(self.sequence_map[idx], idx)]
        if len(self.sequence_map) > 1 and k > 1:
            pool = list(range(len(self.sequence_map)))
            try:
                pool.remove(idx)
            except ValueError:
                pass
            need = min(k - 1, len(pool))
            extra_idxs = random.sample(pool, need)
            picks.extend([(self.sequence_map[j], j) for j in extra_idxs])
            while len(picks) < k:
                j = random.randint(0, len(self.sequence_map) - 1)
                picks.append((self.sequence_map[j], j))

        seg_lens = _split_even(self.sequence_length, k)

        latent_chunks: List[torch.Tensor] = []
        target_chunks: List[torch.Tensor] = []
        cond_chunks:   List[torch.Tensor] = []

        for seg_len, (seq_entry, _seq_idx) in zip(seg_lens, picks):
            dataset_idx, start_frame, token_file_path = seq_entry
            row = self.dataset[dataset_idx]

            # Load codes
            codes = self._load_ecdc_codes(token_file_path)
            if codes is None:
                return self.__getitem__((idx + 1) % len(self.sequence_map))

            # Slice seg_len + 1 frames for input/target shift
            end_frame = start_frame + seg_len + 1
            sequence_codes = codes[:, :self.n_q, start_frame:end_frame]  # (1, n_q, seg_len+1)

            # Input codes (drop last), then to latents
            input_codes = sequence_codes[:, :, :-1]                       # (1, n_q, seg_len)
            latent_in = efficient_codes_to_latents(self.model, input_codes)  # (1, 128, seg_len)
            latent_in = latent_in.squeeze(0).transpose(0, 1)              # (seg_len, 128)

            # Optional noise + preprocess
            if self.config.add_noise:
                latent_in = self._add_noise(latent_in, self.config.noise_weight)
            latent_in = preprocess_latents_for_RNN(latent_in, self.clamp_val)

            # Targets: codes shifted by 1
            targets = sequence_codes[:, :, 1:].squeeze(0).transpose(0, 1)  # (seg_len, n_q)

            # Conditioning for this segment (implemented by subclass)
            cond = self._cond_for_segment(row, token_file_path, start_frame, seg_len)  # (seg_len, P)

            latent_chunks.append(latent_in)
            target_chunks.append(targets)
            cond_chunks.append(cond)

        # Concatenate along time
        latent_all = torch.cat(latent_chunks, dim=0)   # (L, 128)
        targets_all = torch.cat(target_chunks, dim=0)  # (L, n_q)
        cond_all    = torch.cat(cond_chunks,   dim=0)  # (L, P)

        # Final input = latents || cond
        input_tensor = torch.cat([latent_all, cond_all], dim=-1)  # (L, 128+P)
        return input_tensor, targets_all.long()

    # ---- Shared helpers (kept with same names/behavior) ----

    def _load_ecdc_codes(self, token_file_path):
        """
        Load codes from saved torch file
        Returns tensor of shape (1, n_q_total, num_frames)
        """
        try:
            saved_data = torch.load(token_file_path, map_location='cpu', weights_only=False)
            audio_codes = saved_data['audio_codes']

            # Handle different shapes
            if audio_codes.dim() == 4:
                # [1, 1, n_q, time] -> [1, n_q, time]
                audio_codes = audio_codes.squeeze(1)
            elif audio_codes.dim() == 2:
                # [n_q, time] -> [1, n_q, time]
                audio_codes = audio_codes.unsqueeze(0)
            elif audio_codes.dim() == 3 and audio_codes.shape[0] != 1:
                # [n_q, batch, time] -> [batch, n_q, time]
                audio_codes = audio_codes.permute(1, 0, 2)

            if audio_codes.shape[0] != 1:
                audio_codes = audio_codes.unsqueeze(0)
            return audio_codes

        except Exception as e:
            print(f"Error loading {token_file_path}: {e}")
            return None

    def _parse_and_normalize_params_from_row(self, row, filename):
        """
        Get parameters from the dataset row and normalize them (constant params)
        """
        try:
            result = []
            for key, mm in self.parameter_specs.items():
                if key not in row:
                    print(f"Parameter '{key}' not found in dataset columns for {filename}")
                    return None
                # mm is (vmin, vmax)
                vmin, vmax = mm
                raw_val = float(row[key])
                norm_val = (raw_val - vmin) / (vmax - vmin)
                norm_val = max(0.0, min(1.0, norm_val))
                result.append(norm_val)
            return torch.tensor(result, dtype=torch.float32)
        except Exception as e:
            print(f"Error parsing params from row for {filename}: {e}")
            return None

    def _add_noise(self, latent_tensor, weight):
        """
        Add noise to latent vectors
        """
        if weight == 0:
            return latent_tensor
        noise = torch.randn_like(latent_tensor) * weight
        return latent_tensor + noise

    def rand_sample(self, idx=None):
        """
        Return a random sample for debugging
        """
        if idx is None:
            idx = random.randint(0, len(self.sequence_map) - 1)
        return self[idx]

    def getUniqueStrings(self, column: str, sort: bool = True) -> List[str]:
        """
        Return unique class names for a string-like column.

        - If the column is already a ClassLabel feature, return its label names.
        - Otherwise, use Dataset.unique(column), drop Nones, coerce to str, and (optionally) sort.
        """
        if column not in self.dataset.column_names:
            raise ValueError(f"Column '{column}' not found. Available: {self.dataset.column_names}")
        feat = self.dataset.features.get(column)
        if isinstance(feat, ClassLabel):
            names = list(feat.names)
            return sorted(names) if sort else names
        vals = self.dataset.unique(column)
        vals = [v for v in vals if v is not None]
        vals = [v if isinstance(v, str) else str(v) for v in vals]
        return sorted(vals) if sort else vals

    # ---- Subclass hooks ----

    def _cond_for_segment(self, row, token_file_path: Path, start: int, length: int) -> torch.Tensor:
        """Return (length, P) tensor of conditioning. Implemented by subclasses."""
        raise NotImplementedError

    def _validate_row_for_subclass(self, token_file_path: Path, num_frames: int) -> bool:
        """Row-level validation. Dynamic subclass uses this to check sidecar existence/match."""
        return True


# ----------------------------- Constant (per file) parameters ---------------------------

class EnCodecLatentDataset_constant(_BaseEnCodecLatentDataset):
    """
      - parameters from dataset row (Arrow columns)
      - normalized with (min,max) from parameter_specs
      - expanded per frame
    """
    def _cond_for_segment(self, row, token_file_path: Path, start: int, length: int) -> torch.Tensor:
        norm_params = self._parse_and_normalize_params_from_row(row, row['audio'])
        if norm_params is None:
            # Fallback: try next sample (mirrors existing behavior)
            # Note: __getitem__ handles re-try; here we just provide zeros to avoid shape errors.
            P = len(self.parameter_specs)
            return torch.zeros((length, P), dtype=torch.float32)
        return norm_params.unsqueeze(0).expand(length, -1)


# ----------------------------- Dynamic (per frame) parameters ----------------------------

class EnCodecLatentDataset_dynamic(_BaseEnCodecLatentDataset):
    """
    Per-frame conditioning from sidecar:
      <same_basename>.cond.npy : [T, D] float16/32
      <same_basename>.cond.json: {"fps": 75, "names":[...], "norm":{"min":[...],"max":[...], ...}}

    Behavior:
      - Uses parameter_specs KEYS to choose which features to load.
      - Ignores parameter_specs VALUES for min/max; instead, reads min/max from sidecar JSON.
      - If a specs value is not None (or not (None,None)), we raise a clear error.
    """

    def __init__(self, config: LatentDatasetConfig, encodec_model_path, split='train'):
        # Validate parameter_specs values (must be None or (None,None) since we use sidecar stats)
        bad = [k for k, mm in (config.parameter_specs or {}).items() if mm not in (None, (None, None))]
        if bad:
            raise ValueError(
                f"[Dynamic] parameter_specs should provide ONLY the keys (feature names); "
                f"min/max must be None because we normalize using sidecar metadata. Offending keys: {bad}"
            )
        super().__init__(config, encodec_model_path, split)
        # Cache dicts (optional)
        self._json_cache: Dict[Path, dict] = {}

    # ---- path + metadata helpers ----

    def _cond_path_for(self, token_file_path: Path) -> Path:
        if self.config.cond_root is None:
            return token_file_path.with_suffix(self.config.cond_suffix)
        # mirror rel path under cond_root if possible
        try:
            rel = token_file_path.relative_to(Path(self.config.dataset_path))
            return Path(self.config.cond_root) / rel.with_suffix(self.config.cond_suffix)
        except Exception:
            return Path(self.config.cond_root) / token_file_path.name.replace(".ecdc", self.config.cond_suffix)

    def _read_sidecar_meta(self, cpath: Path) -> dict:
        jpath = cpath.with_suffix(SIDECAR_JSON_SUFFIX)
        if jpath in self._json_cache:
            return self._json_cache[jpath]
        meta = {}
        if jpath.exists():
            try:
                meta = json.loads(jpath.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Warning: failed to read sidecar JSON {jpath}: {e}")
        self._json_cache[jpath] = meta
        return meta

    # ---- subclass hooks ----

    def _validate_row_for_subclass(self, token_file_path: Path, num_frames: int) -> bool:
        # Ensure sidecar exists and has matching T
        cpath = self._cond_path_for(token_file_path)
        if not cpath.exists():
            msg = f"[Dynamic] Missing sidecar for {token_file_path} -> {cpath}"
            if self.config.strict:
                raise FileNotFoundError(msg)
            print(msg)
            return False
        try:
            arr = np.load(cpath, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(f"Sidecar {cpath} must be 2D [T,D], got {arr.shape}")
            if arr.shape[0] != num_frames:
                msg = f"[Dynamic] Frame mismatch (codes={num_frames}, cond={arr.shape[0]}) for {token_file_path}"
                if self.config.strict:
                    raise ValueError(msg)
                print(msg)
                return False
        except Exception as e:
            if self.config.strict:
                raise
            print(f"[Dynamic] Unreadable sidecar {cpath}: {e}")
            return False
        return True

    def _cond_for_segment(self, row, token_file_path: Path, start: int, length: int) -> torch.Tensor:
        cpath = self._cond_path_for(token_file_path)
        try:
            arr = np.load(cpath, mmap_mode="r")  # [T, D]
        except Exception as e:
            if self.config.strict:
                raise
            P = len(self.parameter_specs)
            return torch.zeros((length, P), dtype=torch.float32)

        meta = self._read_sidecar_meta(cpath)
        names = [str(n) for n in meta.get("names", [])] if isinstance(meta.get("names", []), (list, tuple)) else []
        norm = meta.get("norm", {}) if isinstance(meta.get("norm", {}), dict) else {}
        mins = norm.get("min", [])
        maxs = norm.get("max", [])

        keys = list(self.parameter_specs.keys())
        P = len(keys)
        out = np.zeros((length, P), dtype=np.float32)

        if names:
            # map requested keys to indices
            missing = [k for k in keys if k not in names]
            if missing:
                msg = f"[Dynamic] Sidecar missing requested features {missing} for {token_file_path}"
                if self.config.strict:
                    raise KeyError(msg)
                print(msg)
                # proceed for the ones that exist; others remain zeros
            name_to_idx = {n: i for i, n in enumerate(names)}
            # pull slice
            for j, k in enumerate(keys):
                if k in name_to_idx:
                    col = name_to_idx[k]
                    x = arr[start:start+length, col].astype(np.float32, copy=False)
                    # normalize with sidecar min/max
                    vmin = float(mins[col]) if col < len(mins) else None
                    vmax = float(maxs[col]) if col < len(maxs) else None
                    if vmin is None or vmax is None or vmax == vmin:
                        # fallback to [0,1] via percentile or no-op; here just clamp to 0..1 if already in range
                        pass
                    else:
                        x = (x - vmin) / (vmax - vmin + 1e-8)
                        np.clip(x, 0.0, 1.0, out=x)
                    out[:, j] = x
                else:
                    # missing feature -> leave zeros
                    pass
        else:
            # No names metadata: assume sidecar columns correspond to parameter_specs order
            cols = min(P, arr.shape[1])
            x = arr[start:start+length, :cols].astype(np.float32, copy=False)
            # Without min/max metadata, we cannot safely normalize—leave as-is or clamp.
            # We'll clamp to [0,1] if it already is in that range; otherwise leave.
            out[:, :cols] = np.clip(x, 0.0, 1.0)

        return torch.from_numpy(out)


#------------------------------------------------------------------------------

class EnCodecLatentDataset_dynamic_v2(_BaseEnCodecLatentDataset):
    """
    Per-frame conditioning from v2 sidecars:
      <basename>.cond.npy  : [T, D] float16/32 (column order matches JSON 'features' insertion order)
      <basename>.json      : {
         "schema_version": 2,
         "fps": <float>, "source_rate": <float>,
         "features": {
            "<name>": {"min": <float>, "max": <float>, "mean": <float>, "std": <float>,
                       "units": "<str>", "doc_string": "<str>"},
            ...
         }
      }

    Behavior:
      - Uses parameter_specs KEYS to choose which features to load.
      - Ignores parameter_specs VALUES (must be None or (None,None)); min/max come from JSON.
      - If strict=True, raises on missing sidecar / feature / frame mismatch; else logs and skips/zeros.
    """
    
    def __init__(self, config: LatentDatasetConfig, encodec_model_path, split='train'):
        # Validate parameter_specs values (must be None since we normalize using sidecar)
        bad = [k for k, mm in (config.parameter_specs or {}).items() if mm not in (None, (None, None))]
        if bad:
            raise ValueError(
                f"[Dynamic v2] parameter_specs should provide ONLY the keys (feature names); "
                f"min/max must be None because we normalize using sidecar metadata. Offending keys: {bad}"
            )
        
        self._json_cache: Dict[Path, dict] = {}
        self._requested_keys = list((config.parameter_specs or {}).keys())      # <— NEW
        self._missing_features_seen: set[str] = set()                           # <— NEW


        super().__init__(config, encodec_model_path, split)

        # One-time summary after building sequence_map
        if self._missing_features_seen:                                          # <— NEW
            print(f"[Dynamic v2] Missing (in at least one file): "
                  f"{sorted(self._missing_features_seen)}")                      # <— NEW
            
    # ---- path + metadata helpers ----

    def _cond_path_for(self, token_file_path: Path) -> Path:
        """Locate .cond.npy next to the .ecdc, or mirror under cond_root if provided."""
        if self.config.cond_root is None:
            return token_file_path.with_suffix(self.config.cond_suffix)
        try:
            rel = token_file_path.relative_to(Path(self.config.dataset_path))
            return Path(self.config.cond_root) / rel.with_suffix(self.config.cond_suffix)
        except Exception:
            return Path(self.config.cond_root) / token_file_path.name.replace(".ecdc", self.config.cond_suffix)

    def _read_sidecar_meta(self, cpath: Path) -> dict:
        jpath = cpath.with_suffix(SIDECAR_JSON_SUFFIX)
        if jpath in self._json_cache:
            return self._json_cache[jpath]
        meta = {}
        if jpath.exists():
            try:
                meta = json.loads(jpath.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[Dynamic v2] Warning: failed to read sidecar JSON {jpath}: {e}")
        self._json_cache[jpath] = meta
        return meta

    # ---- subclass hooks ----

    def _validate_row_for_subclass(self, token_file_path: Path, num_frames: int) -> bool:
        cpath = self._cond_path_for(token_file_path)
        if not cpath.exists():
            msg = f"[Dynamic v2] Missing sidecar for {token_file_path} -> {cpath}"
            if self.config.strict:
                raise FileNotFoundError(msg)
            print(msg)
            return False
        try:
            arr = np.load(cpath, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(f"Sidecar {cpath} must be 2D [T,D], got {arr.shape}")
            if arr.shape[0] != num_frames:
                msg = f"[Dynamic v2] Frame mismatch (codes={num_frames}, cond={arr.shape[0]}) for {token_file_path}"
                if self.config.strict:
                    raise ValueError(msg)
                print(msg)
                return False

            meta = self._read_sidecar_meta(cpath)
            feats = meta.get("features", {})
            if not isinstance(feats, dict) or not feats:
                msg = f"[Dynamic v2] JSON lacks 'features' dict: {cpath.with_suffix(SIDECAR_JSON_SUFFIX)}"
                if self.config.strict:
                    raise ValueError(msg)
                print(msg)
                return False

            # ---- NEW: check requested keys exist in this file ----
            if self._requested_keys:
                missing_here = [k for k in self._requested_keys if k not in feats]
                if missing_here:
                    if self.config.strict:
                        raise KeyError(f"[Dynamic v2] Requested features {missing_here} "
                                       f"not present in {cpath.with_suffix(SIDECAR_JSON_SUFFIX)}")
                    # collect for a single summary later (avoid per-file spam)
                    self._missing_features_seen.update(missing_here)
            # -----------------------------------------

        except Exception as e:
            if self.config.strict:
                raise
            print(f"[Dynamic v2] Unreadable sidecar {cpath}: {e}")
            return False
        return True

    def _cond_for_segment(self, row, token_file_path: Path, start: int, length: int) -> torch.Tensor:
        cpath = self._cond_path_for(token_file_path)
        try:
            arr = np.load(cpath, mmap_mode="r")  # [T, D]
        except Exception as e:
            if self.config.strict:
                raise
            P = len(self.config.parameter_specs or {})
            return torch.zeros((length, P), dtype=torch.float32)

        meta = self._read_sidecar_meta(cpath)
        feats = meta.get("features", {})
        if not isinstance(feats, dict) or not feats:
            if self.config.strict:
                raise ValueError(f"[Dynamic v2] Missing/invalid 'features' in JSON for {cpath}")
            P = len(self.config.parameter_specs or {})
            return torch.zeros((length, P), dtype=torch.float32)

        # Column order = insertion order of feats keys when sidecar was written
        names_in_order = list(feats.keys())
        name_to_idx = {n: i for i, n in enumerate(names_in_order)}

        keys = list(self.config.parameter_specs.keys())
        P = len(keys)
        out = np.zeros((length, P), dtype=np.float32)

        # slice the frame window once
        if start < 0 or start + length > arr.shape[0]:
            if self.config.strict:
                raise IndexError(f"[Dynamic v2] window [{start}:{start+length}) out of bounds for {cpath}")
            start = max(0, min(start, arr.shape[0]))
            end = min(arr.shape[0], start + length)
        else:
            end = start + length
        window = arr[start:end]  # shape (length, D) or smaller if clamped above

        for j, k in enumerate(keys):
            if k not in name_to_idx:
                msg = f"[Dynamic v2] Sidecar missing requested feature '{k}' for {token_file_path}"
                if self.config.strict:
                    raise KeyError(msg)
                # leave zeros
                continue

            col = name_to_idx[k]
            if col >= window.shape[1]:
                if self.config.strict:
                    raise IndexError(f"[Dynamic v2] Column {col} out of range in {cpath}")
                continue

            x = window[:, col].astype(np.float32, copy=False)

            # normalize with v2 per-feature min/max
            fmeta = feats.get(k, {})
            vmin = fmeta.get("min", None)
            vmax = fmeta.get("max", None)
            if isinstance(vmin, (int, float)) and isinstance(vmax, (int, float)) and vmax != vmin:
                x = (x - float(vmin)) / (float(vmax) - float(vmin) + 1e-8)
                np.clip(x, 0.0, 1.0, out=x)
            # else: leave as-is (already scaled) — or you could clamp

            # pad if window was shorter (should be rare if lengths validated)
            if x.shape[0] < length:
                pad = np.zeros((length - x.shape[0],), dtype=np.float32)
                x = np.concatenate([x, pad], axis=0)

            out[:, j] = x

        return torch.from_numpy(out)

# -----------------------------------------------------------------------------


class EnCodecLatentDataset_dynamic_v3(_BaseEnCodecLatentDataset):
    """
    Conditioning from a single global config + per-file .cond.npy:

      Global config (once per dataset):
        <dataset_root>/conditioning_config.json  : {
          "schema_version": 1,
          "fps": 75,
          "feature_names": [...],          # defines column order in .cond.npy
          "features": { name: {min,max,...}, ... },
          ...
        }

      Per item:
        <basename>.cond.npy  : [T, D] float16/32

    Behavior:
      - Uses parameter_specs KEYS to choose which features to load.
      - Ignores parameter_specs VALUES; min/max come from conditioning_config.json.
      - If strict=True, raises on missing sidecar / feature / frame mismatch; else logs and skips/zeros.
    """

    def __init__(self, config: LatentDatasetConfig, encodec_model_path, split="train"):
        # 1) Validate parameter_specs values
        bad = [k for k, mm in (config.parameter_specs or {}).items() if mm not in (None, (None, None))]
        if bad:
            raise ValueError(
                f"[Dynamic v3] parameter_specs should provide ONLY the keys (feature names); "
                f"min/max must be None because we normalize using conditioning_config.json. Offending keys: {bad}"
            )

        # 2) Predefine attributes used by hooks (so base-class calls won't crash)
        self._requested_keys = list((config.parameter_specs or {}).keys())
        self._missing_features_seen: set[str] = set()

        self._cond_cfg_path: Path | None = None
        self._cond_cfg: dict = {}
        self._feature_names: list[str] = []
        self._features_meta: dict = {}
        self._name_to_idx: dict[str, int] = {}

        # 3) Load global conditioning config BEFORE super().__init__()
        self._cond_cfg_path, self._cond_cfg = self._load_conditioning_config_from_dataset_path(
            Path(config.dataset_path),
            split=split,
        )
        self._feature_names = list(self._cond_cfg.get("feature_names", []))
        self._features_meta = self._cond_cfg.get("features", {})

        if not self._feature_names or not isinstance(self._features_meta, dict) or not self._features_meta:
            raise ValueError(
                f"[Dynamic v3] conditioning_config.json missing/invalid 'feature_names' or 'features': "
                f"{self._cond_cfg_path}"
            )

        self._name_to_idx = {n: i for i, n in enumerate(self._feature_names)}

        # 4) Validate requested keys exist in global config (optional but helpful)
        if self._requested_keys:
            missing_global = [k for k in self._requested_keys if k not in self._features_meta]
            if missing_global:
                msg = f"[Dynamic v3] Requested features not present in conditioning_config.json: {missing_global}"
                if config.strict:
                    raise KeyError(msg)
                print(msg)
                self._missing_features_seen.update(missing_global)

        # 5) NOW it's safe to call super; it may call _validate_row_for_subclass()
        super().__init__(config, encodec_model_path, split)

        if self._missing_features_seen:
            print(f"[Dynamic v3] Missing (global or at least one file): {sorted(self._missing_features_seen)}")

    # ---- conditioning config helpers ----

    def _load_conditioning_config_from_dataset_path(self, dataset_path: Path, split: str):
        cfg_path = dataset_path / "conditioning_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"[Dynamic v3] Missing {cfg_path}")
        return cfg_path, json.loads(cfg_path.read_text(encoding="utf-8"))
    

    # ---- path helper (same as v2) ----

    def _cond_path_for(self, token_file_path: Path) -> Path:
        """Locate .cond.npy next to the .ecdc, or mirror under cond_root if provided."""
        if self.config.cond_root is None:
            return token_file_path.with_suffix(self.config.cond_suffix)
        try:
            rel = token_file_path.relative_to(Path(self.config.dataset_path))
            return Path(self.config.cond_root) / rel.with_suffix(self.config.cond_suffix)
        except Exception:
            return Path(self.config.cond_root) / token_file_path.name.replace(".ecdc", self.config.cond_suffix)

    # ---- subclass hooks ----

    def _validate_row_for_subclass(self, token_file_path: Path, num_frames: int) -> bool:
        cpath = self._cond_path_for(token_file_path)
        if not cpath.exists():
            msg = f"[Dynamic v3] Missing sidecar for {token_file_path} -> {cpath}"
            if self.config.strict:
                raise FileNotFoundError(msg)
            print(msg)
            return False
    
        try:
            arr = np.load(cpath, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(f"Sidecar {cpath} must be 2D [T,D], got {arr.shape}")
            if arr.shape[0] != num_frames:
                msg = f"[Dynamic v3] Frame mismatch (codes={num_frames}, cond={arr.shape[0]}) for {token_file_path}"
                if self.config.strict:
                    raise ValueError(msg)
                print(msg)
                return False
    
            D_expected = len(self._feature_names)
            if arr.shape[1] != D_expected:
                msg = (f"[Dynamic v3] Feature dim mismatch for {token_file_path}: "
                       f"cond has D={arr.shape[1]} but conditioning_config expects D={D_expected}")
                if self.config.strict:
                    raise ValueError(msg)
                print(msg)
                return False
    
        except Exception:
            if self.config.strict:
                raise
            return False
    
        return True

    def _cond_for_segment(self, row, token_file_path: Path, start: int, length: int) -> torch.Tensor:
        cpath = self._cond_path_for(token_file_path)
        keys = list(self.config.parameter_specs.keys())
        P = len(keys)
    
        try:
            arr = np.load(cpath, mmap_mode="r")  # [T, D]
        except Exception:
            if self.config.strict:
                raise
            return torch.zeros((length, P), dtype=torch.float32)
    
        # Slice window once
        if start < 0 or start + length > arr.shape[0]:
            if self.config.strict:
                raise IndexError(f"[Dynamic v3] window [{start}:{start+length}) out of bounds for {cpath}")
            start = max(0, min(start, arr.shape[0]))
            end = min(arr.shape[0], start + length)
        else:
            end = start + length
    
        window = arr[start:end]  # (<=length, D)
        out = np.zeros((length, P), dtype=np.float32)
    
        for j, k in enumerate(keys):
            idx = self._name_to_idx.get(k, None)
            if idx is None:
                msg = f"[Dynamic v3] conditioning_config missing requested feature '{k}'"
                if self.config.strict:
                    raise KeyError(msg)
                # leave zeros
                continue
    
            if idx >= window.shape[1]:
                if self.config.strict:
                    raise IndexError(f"[Dynamic v3] Column {idx} out of range in {cpath}")
                continue
    
            x = window[:, idx].astype(np.float32, copy=False)
    
            # Normalize with GLOBAL per-feature min/max (from conditioning_config.json)
            fmeta = self._features_meta.get(k, {})
            vmin = fmeta.get("min", None)
            vmax = fmeta.get("max", None)
            if isinstance(vmin, (int, float)) and isinstance(vmax, (int, float)) and vmax != vmin:
                x = (x - float(vmin)) / (float(vmax) - float(vmin) + 1e-8)
                np.clip(x, 0.0, 1.0, out=x)
    
            # Pad if window shorter (rare if validated)
            if x.shape[0] < length:
                pad = np.zeros((length - x.shape[0],), dtype=np.float32)
                x = np.concatenate([x, pad], axis=0)
    
            out[:, j] = x
    
        return torch.from_numpy(out)
            


# -------------------------- Back-compat class alias --------------------------

# Keep the original class name working exactly as before.
EnCodecLatentDataset = EnCodecLatentDataset_constant