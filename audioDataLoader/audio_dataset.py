import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable, Optional, Any, Set, Union, Any

from datasets.features import ClassLabel

import random
from datasets import load_from_disk
from pathlib import Path



# type alias
FilterSpec = Dict[str, Union[Tuple[float, float], Set[Any]]]

@dataclass
class LatentDatasetConfig:
    dataset_path: str                              # Path to the main dataset folder (contains dataset_dict.json and split subfolders)
    sequence_length: int                           # Number of frames per training sequence
    parameter_specs: Dict[str, Tuple[float, float]]  # {'param': (min, max), ...}
    add_noise: bool = False                        # Whether to add noise to latents
    noise_weight: float = 0.1                      # Weight for noise injection
    codebook_size: int = 1024                      # Number of possible target values per codebook
    n_q: int = 4                                   # Number of codebooks to use
    clamp_val: float = 15
    filters: Optional[FilterSpec] = None           # e.g. {"foo": {4,6,7}, "bar": (0.0, 3.0), "label": {"A","B"}}


def preprocess_latents_for_RNN(latents, clamp_val):
    """Convert latents to [-1,1] range with optional clamping"""
    if clamp_val != 0:
        return torch.clamp(latents, -clamp_val, clamp_val) / clamp_val
    return latents
    
#############             encoded conversions   ###############################################
def efficient_codes_to_latents(model, codes):
    """
    Efficient version for repeated use (e.g., in DataLoaders)
    """
    model.eval()
    
    with torch.no_grad():
        # Direct quantizer access with proper transpose
        if codes.shape[0] == 1:  # batch dimension first
            codes_transposed = codes.transpose(0, 1)  # (n_q, batch, time)
        else:
            codes_transposed = codes
            
        # Direct quantizer decode - this is just lookups + addition
        embeddings = model.quantizer.decode(codes_transposed)
        return embeddings

#----------------------------------------------------------------------------------------------------
@torch.no_grad()
def latents_to_audio_simple(model, embeddings):
    """
    EnCodec 24kHz mono: 128-D latents -> waveform.
    No normalization, no PQMF.

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

    # Ensure (B, 128, T)
    if z.shape[1] != 128 and z.shape[2] == 128:
        z = z.transpose(1, 2)

    # Decode latents to waveform
    x = model.decoder(z)              # usually returns a tensor directly

    # Unwrap if some variant returns (x, ...)
    if isinstance(x, (tuple, list)):
        x = x[0]
    elif isinstance(x, dict):
        x = x.get("x", next(iter(x.values())))

    # Move to CPU float32 for saving/playback
    x = x.detach().to(torch.float32).cpu()

    sr = int(getattr(model, "sample_rate", 24000))
    return x, sr



#################    Datset Filter helpers    ##############################################

def _normalize_filter_spec(filters):
    """
    Normalize a user filter spec like:
      {"foo": (0.0, 3.0), "bar": {4, 6, 7}, "baz": [1, 2]}
    into a canonical form:
      {"foo": ("range", 0.0, 3.0), "bar": ("set", {4,6,7}), "baz": ("set", {1,2})}
    """
    if not filters:
        return {}

    norm = {}
    for key, rule in filters.items():
        # numeric range (lo, hi)
        if isinstance(rule, tuple) and len(rule) == 2:
            lo, hi = rule
            norm[key] = ("range", float(lo), float(hi))
        else:
            # membership in a set/list/iterable
            try:
                norm[key] = ("set", set(rule))
            except TypeError:
                # single scalar -> singleton set
                norm[key] = ("set", {rule})
    return norm


def _apply_hf_filters(ds, filters):
    """
    Apply normalized filters to a Hugging Face Dataset once.
    Uses batched filtering for speed. Returns a new filtered Dataset.
    """
    spec = _normalize_filter_spec(filters)
    if not spec:
        return ds

    import numpy as np

    def keep(batch):
        # Build a boolean mask for this chunk of rows
        length = len(next(iter(batch.values())))
        mask = np.ones(length, dtype=bool)

        for col, rule in spec.items():
            if col not in batch:
                # if the column doesn't exist in this dataset, drop all rows
                mask &= False
                continue

            v = np.asarray(batch[col])

            kind = rule[0]
            if kind == "range":
                _, lo, hi = rule
                # try numeric compare; if strings slipped in, coerce
                try:
                    vf = v.astype(np.float32)
                except Exception:
                    # if coercion fails, reject these rows
                    mask &= False
                    continue
                mask &= (vf >= lo) & (vf <= hi)
            else:
                _, allowed = rule
                mask &= np.isin(v, list(allowed))

        return mask

    return ds.filter(keep, batched=True)


###############################################################################
#################    THE CUSTOM DATASET CLASS ITSELF   ########################
###############################################################################

## Usage:              (member)         (in range)
#    filters = {"foo": {4,6,7}, "bar": (0.0, 3.0), "label": {"A","B"}}
#    ds = EnCodecLatentDataset(config, encodec_model_path, split="train", filters=filters)


class EnCodecLatentDataset(Dataset):
    def __init__(self, config: LatentDatasetConfig, encodec_model_path, split='train'):

        # Load our own copy of EnCodec for CPU operations
        from transformers import EncodecModel
        self.model = EncodecModel.from_pretrained(encodec_model_path)
        self.model.eval()  # Keep on CPU

        
        self.config = config
        self.sequence_length = config.sequence_length
        self.parameter_specs = config.parameter_specs
        self.n_q = config.n_q
        self.split = split
        self.clamp_val = self.config.clamp_val
        
        # Load HuggingFace dataset using load_from_disk
        self.dataset = load_from_disk(config.dataset_path)[split]
        
        # The dataset path contains the .ecdc files
        self.dataset_root = Path(config.dataset_path)

        # >>> NEW: apply subset filters once, before building the sequence map
        total_before = len(self.dataset)

        if config.filters:
            self.dataset = _apply_hf_filters(self.dataset, config.filters)
            total_after = len(self.dataset)
            pct = (100.0 * total_after / total_before) if total_before else 0.0
            print(f"[EnCodecLatentDataset] Filters applied: kept {total_after} of {total_before} rows ({pct:.1f}%).")
        else:
            print(f"[EnCodecLatentDataset] No filters provided; using all {total_before} rows.")
        # <<<
        
        # Build sequence map: (dataset_idx, start_frame)
        self.sequence_map = []
        
        for dataset_idx, row in enumerate(self.dataset):
            # The 'audio' field might already contain the split folder path
            audio_path = row['audio']
            
            # Try different path combinations
            possible_paths = [
                self.dataset_root / audio_path,  # If audio_path already includes split
                self.dataset_root / split / audio_path,  # If we need to add split
                Path(audio_path),  # If it's already an absolute path
            ]
            
            token_file_path = None
            for path in possible_paths:
                if path.exists():
                    token_file_path = path
                    break
            
            if token_file_path is None:
                print(f"Warning: Could not find audio file for any of these paths:")
                for path in possible_paths:
                    print(f"  - {path}")
                continue
            
            # Load codes to get number of frames
            codes = self._load_ecdc_codes(token_file_path)
            
            if codes is None:
                continue
                
            num_frames = codes.shape[-1]  # Shape is (1, n_q, num_frames)
            
            # Create sequence map entries
            if num_frames > self.sequence_length:
                for start_frame in range(num_frames - self.sequence_length):
                    self.sequence_map.append((dataset_idx, start_frame, token_file_path))
        
        print(f"Loaded {len(self.sequence_map)} sequences from {len(self.dataset)} files in '{split}' split")

    def __len__(self):
        return len(self.sequence_map)

    def __getitem__(self, idx):
        dataset_idx, start_frame, token_file_path = self.sequence_map[idx]
        row = self.dataset[dataset_idx]
        
        # Load codes from token file
        codes = self._load_ecdc_codes(token_file_path)  # Shape: (1, total_n_q, num_frames)
        
        # Extract sequence + 1 frame for input/target shift
        end_frame = start_frame + self.sequence_length + 1
        sequence_codes = codes[:, :self.n_q, start_frame:end_frame]  # Use only first n_q codebooks
        
        # Input: convert codes[:-1] to 128D latents
        input_codes = sequence_codes[:, :, :-1]  # Shape: (1, n_q, sequence_length)
        latent_input = efficient_codes_to_latents(self.model, input_codes)  # Shape: (1, 128, sequence_length)
        
        # Remove batch dimension and transpose for sequence-first format
        latent_input = latent_input.squeeze(0).transpose(0, 1)  # Shape: (sequence_length, 128)
        
        # Add noise if requested
        if self.config.add_noise:
            latent_input = self._add_noise(latent_input, self.config.noise_weight)

        latent_input = preprocess_latents_for_RNN(latent_input, self.clamp_val)
        
        # Target: codes shifted by one frame
        target_codes = sequence_codes[:, :, 1:].squeeze(0).transpose(0, 1)  # Shape: (sequence_length, n_q)
        
        # Get conditioning parameters from dataset columns
        norm_params = self._parse_and_normalize_params_from_row(row, row['audio'])
        
        if norm_params is None:
            # Skip this sample if parameters can't be parsed
            return self.__getitem__((idx + 1) % len(self.sequence_map))
        
        # Expand conditioning params across sequence length
        cond_params = norm_params.unsqueeze(0).expand(self.sequence_length, -1)
        
        # Combine latent input and conditioning parameters
        input_tensor = torch.cat([latent_input, cond_params], dim=-1)
        
        return input_tensor, target_codes.long()

    def _load_ecdc_codes(self, token_file_path):
        """
        Load codes from saved torch file
        Returns tensor of shape (1, n_q_total, num_frames)
        """
        try:
            saved_data = torch.load(token_file_path, map_location='cpu')
            audio_codes = saved_data['audio_codes']
            
            # Handle different possible shapes
            if audio_codes.dim() == 4:
                # Shape [1, 1, n_q, time] -> [1, n_q, time]
                audio_codes = audio_codes.squeeze(1)
            elif audio_codes.dim() == 2:
                # Shape [n_q, time] -> add batch dimension [1, n_q, time]  
                audio_codes = audio_codes.unsqueeze(0)
            elif audio_codes.dim() == 3 and audio_codes.shape[0] != 1:
                # Shape [n_q, batch, time] -> [batch, n_q, time]
                audio_codes = audio_codes.permute(1, 0, 2)
            
            # Ensure shape is [1, n_q, time]
            if audio_codes.shape[0] != 1:
                audio_codes = audio_codes.unsqueeze(0)
                
            return audio_codes
            
        except Exception as e:
            print(f"Error loading {token_file_path}: {e}")
            return None

    def _parse_and_normalize_params_from_row(self, row, filename):
        """
        Get parameters from the dataset row and normalize them
        """
        try:
            result = []
            for key, (vmin, vmax) in self.parameter_specs.items():
                if key not in row:
                    print(f"Parameter '{key}' not found in dataset columns for {filename}")
                    return None
                    
                # Get from dataset row
                raw_val = float(row[key])
                norm_val = (raw_val - vmin) / (vmax - vmin)
                # Clamp to [0,1] for safety
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
            raise ValueError(f"Column '{column}' not found. Available: {self.ds.column_names}")
    
        feat = self.dataset.features.get(column)
        if isinstance(feat, ClassLabel):
            names = list(feat.names)
            return sorted(names) if sort else names
    
        # Plain string (or mixed) column
        vals = self.dataset.unique(column)               # list of Python objects
        vals = [v for v in vals if v is not None]   # drop missing
        vals = [v if isinstance(v, str) else str(v) for v in vals]
        return sorted(vals) if sort else vals