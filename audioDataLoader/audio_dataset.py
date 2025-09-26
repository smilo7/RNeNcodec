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
    files_per_sequence: int = 2 

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

#----------------------------
# Helper for creating ~equal length segments in sequences from different files
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

        self.files_per_sequence = getattr(config, "files_per_sequence", 2)

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
        # Build sequence map: (dataset_idx, start_frame, token_file_path)
        self.sequence_map = []
        
        # How many files are combined per training item?
        k = max(1, int(getattr(self.config, "files_per_sequence", 2)))
        # Longest segment we might take from any one file
        longest_seg = (self.sequence_length + k - 1) // k   # ceil(sequence_length / k)
        # Need +1 for next-step target shift during slicing
        needed = longest_seg + 1
        
        for dataset_idx, row in enumerate(self.dataset):
            # The 'audio' field might already contain the split folder path
            audio_path = row['audio']
        
            possible_paths = [
                self.dataset_root / audio_path,            # audio already includes split
                self.dataset_root / self.split / audio_path,  # add split if needed
                Path(audio_path),                          # absolute path
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
        
            num_frames = codes.shape[-1]  # (1, n_q, T)
        
            # Create sequence map entries
            if num_frames >= needed:
                # valid starts are [0 .. num_frames - needed]
                max_start = num_frames - needed + 1
                for start_frame in range(max_start):
                    self.sequence_map.append((dataset_idx, start_frame, token_file_path))
        
        print(f"Loaded {len(self.sequence_map)} sequences from {len(self.dataset)} files in '{split}' split")

    def __len__(self):
        return len(self.sequence_map)

    def __getitem__(self, idx):
        # How many separate files to mix into one sequence
        k = max(1, int(getattr(self.config, "files_per_sequence", 2)))
    
        # Choose k sequences: 1 anchored at idx + (k-1) random distinct picks (when possible)
        picks = [(self.sequence_map[idx], idx)]
        if len(self.sequence_map) > 1 and k > 1:
            # Sample without replacement, avoiding 'idx' if possible
            # Build a pool of indices excluding idx (if there are enough)
            pool = list(range(len(self.sequence_map)))
            try:
                pool.remove(idx)
            except ValueError:
                pass
            need = min(k - 1, len(pool))
            extra_idxs = random.sample(pool, need)
            picks.extend([(self.sequence_map[j], j) for j in extra_idxs])
    
            # If dataset is tiny and we still need more, allow repeats
            while len(picks) < k:
                j = random.randint(0, len(self.sequence_map) - 1)
                picks.append((self.sequence_map[j], j))
    
        # Split total sequence_length across k segments as evenly as possible
        seg_lens = _split_even(self.sequence_length, k)
    
        latent_chunks = []
        target_chunks = []
        cond_chunks = []
    
        for seg_len, (seq_entry, seq_idx) in zip(seg_lens, picks):
            dataset_idx, start_frame, token_file_path = seq_entry
            row = self.dataset[dataset_idx]
    
            # Load codes
            codes = self._load_ecdc_codes(token_file_path)
            if codes is None:
                # Try next index if any, otherwise fallback to another sample
                return self.__getitem__((idx + 1) % len(self.sequence_map))
    
            # Slice seg_len + 1 frames for input/target shift
            end_frame = start_frame + seg_len + 1
            sequence_codes = codes[:, :self.n_q, start_frame:end_frame]  # (1, n_q, seg_len+1)
    
            # Input codes (drop last frame), then to latents
            input_codes = sequence_codes[:, :, :-1]                       # (1, n_q, seg_len)
            latent_in = efficient_codes_to_latents(self.model, input_codes)  # (1, 128, seg_len)
    
            # (T, 128)
            latent_in = latent_in.squeeze(0).transpose(0, 1)
    
            # Optional noise + preprocess
            if self.config.add_noise:
                latent_in = self._add_noise(latent_in, self.config.noise_weight)
            latent_in = preprocess_latents_for_RNN(latent_in, self.clamp_val)
    
            # Targets: codes shifted by 1
            targets = sequence_codes[:, :, 1:].squeeze(0).transpose(0, 1)  # (seg_len, n_q)
    
            # Conditioning for this segment
            norm_params = self._parse_and_normalize_params_from_row(row, row['audio'])
            if norm_params is None:
                return self.__getitem__((idx + 1) % len(self.sequence_map))
            cond = norm_params.unsqueeze(0).expand(seg_len, -1)            # (seg_len, P)
    
            latent_chunks.append(latent_in)
            target_chunks.append(targets)
            cond_chunks.append(cond)
    
        # Concatenate along time
        latent_all = torch.cat(latent_chunks, dim=0)         # (L, 128)
        targets_all = torch.cat(target_chunks, dim=0)        # (L, n_q)
        cond_all = torch.cat(cond_chunks, dim=0)             # (L, P)
    
        # Final input = latents || cond
        input_tensor = torch.cat([latent_all, cond_all], dim=-1)  # (L, 128+P)
    
        return input_tensor, targets_all.long()
    
    
    # def __getitem__(self, idx):
    #     # Get the first sequence (first half)
    #     dataset_idx1, start_frame1, token_file_path1 = self.sequence_map[idx]
    #     row1 = self.dataset[dataset_idx1]
        
    #     # Get a random second sequence (second half)
    #     random_idx = random.randint(0, len(self.sequence_map) - 1)
    #     dataset_idx2, start_frame2, token_file_path2 = self.sequence_map[random_idx]
    #     row2 = self.dataset[dataset_idx2]
        
    #     # Calculate half sequence length
    #     half_seq_len = self.sequence_length // 2
    #     remainder = self.sequence_length % 2  # Handle odd sequence lengths
        
    #     # First half length (gets the extra frame if sequence_length is odd)
    #     first_half_len = half_seq_len + remainder
    #     second_half_len = half_seq_len
        
    #     # Load and process first sequence
    #     codes1 = self._load_ecdc_codes(token_file_path1)
    #     if codes1 is None:
    #         return self.__getitem__((idx + 1) % len(self.sequence_map))
        
    #     # Extract first half + 1 frame for input/target shift
    #     end_frame1 = start_frame1 + first_half_len + 1
    #     sequence_codes1 = codes1[:, :self.n_q, start_frame1:end_frame1]
        
    #     # Load and process second sequence  
    #     codes2 = self._load_ecdc_codes(token_file_path2)
    #     if codes2 is None:
    #         return self.__getitem__((idx + 1) % len(self.sequence_map))
        
    #     # Extract second half + 1 frame for input/target shift
    #     end_frame2 = start_frame2 + second_half_len + 1
    #     sequence_codes2 = codes2[:, :self.n_q, start_frame2:end_frame2]
        
    #     # Convert codes to latents for both sequences
    #     input_codes1 = sequence_codes1[:, :, :-1]  # Remove last frame for input
    #     input_codes2 = sequence_codes2[:, :, :-1]  # Remove last frame for input
        
    #     latent_input1 = efficient_codes_to_latents(self.model, input_codes1)
    #     latent_input2 = efficient_codes_to_latents(self.model, input_codes2)
        
    #     # Remove batch dimension and transpose for sequence-first format
    #     latent_input1 = latent_input1.squeeze(0).transpose(0, 1)  # (first_half_len, 128)
    #     latent_input2 = latent_input2.squeeze(0).transpose(0, 1)  # (second_half_len, 128)
        
    #     # Add noise if requested
    #     if self.config.add_noise:
    #         latent_input1 = self._add_noise(latent_input1, self.config.noise_weight)
    #         latent_input2 = self._add_noise(latent_input2, self.config.noise_weight)
        
    #     # Preprocess latents
    #     latent_input1 = preprocess_latents_for_RNN(latent_input1, self.clamp_val)
    #     latent_input2 = preprocess_latents_for_RNN(latent_input2, self.clamp_val)
        
    #     # Combine latent inputs
    #     combined_latent_input = torch.cat([latent_input1, latent_input2], dim=0)
        
    #     # Process target codes (shifted by one frame)
    #     target_codes1 = sequence_codes1[:, :, 1:].squeeze(0).transpose(0, 1)  # (first_half_len, n_q)
    #     target_codes2 = sequence_codes2[:, :, 1:].squeeze(0).transpose(0, 1)  # (second_half_len, n_q)
        
    #     # Combine target codes
    #     combined_target_codes = torch.cat([target_codes1, target_codes2], dim=0)
        
    #     # Get conditioning parameters from both dataset rows
    #     norm_params1 = self._parse_and_normalize_params_from_row(row1, row1['audio'])
    #     norm_params2 = self._parse_and_normalize_params_from_row(row2, row2['audio'])
        
    #     if norm_params1 is None or norm_params2 is None:
    #         # Skip this sample if parameters can't be parsed
    #         return self.__getitem__((idx + 1) % len(self.sequence_map))
        
    #     # Create conditioning parameters for each half
    #     cond_params1 = norm_params1.unsqueeze(0).expand(first_half_len, -1)
    #     cond_params2 = norm_params2.unsqueeze(0).expand(second_half_len, -1)
        
    #     # Combine conditioning parameters
    #     combined_cond_params = torch.cat([cond_params1, cond_params2], dim=0)
        
    #     # Combine latent input and conditioning parameters
    #     input_tensor = torch.cat([combined_latent_input, combined_cond_params], dim=-1)
        
    #     return input_tensor, combined_target_codes.long()

    ###########################################################################################
    # This getitem() works perfectly well, but data files only have a single constant parameter
    # The replacement (above) splits each item into a sequence with two halves constructed from two data files
    ###########################################################################################
    def __get___ONE____item__(self, idx):
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
                print(f'({key} Mapping raw val = {raw_val} to norm val = {norm_val}')
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