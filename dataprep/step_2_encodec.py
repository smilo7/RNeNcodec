"""
EnCodec Encoding for Dataset Pipeline

This script encodes normalized WAV files into EnCodec token files (.ecdc)
using the facebook/encodec_24khz model.

The encoding process:
1. Loads normalized 24kHz mono WAV files
2. Encodes using EnCodec at specified bandwidth (default 6.0 kbps)
3. Saves as .ecdc files maintaining folder structure
4. Output goes to tokens/ subfolder for dataset compatibility

Based on: lonce/EncHF_Dataset/workflow/s2_wav2encodec24.py
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import argparse
import math
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import librosa
from transformers import EncodecModel, AutoProcessor
from tqdm import tqdm

from dataprep.auxiliary_functions import find_audio_files


def iter_token_files(root: Path, suffix: str, recursive: bool) -> Iterable[Path]:
    """Iterate over token files (.ecdc) in a directory."""
    patterns = (f"*{suffix}",)
    if recursive:
        for pat in patterns:
            yield from root.rglob(pat)
    else:
        for pat in patterns:
            yield from root.glob(pat)


def cpuify(obj):
    """Move torch tensors to CPU, recursively handle lists/dicts."""
    if hasattr(obj, "cpu"):
        return obj.cpu()
    if isinstance(obj, (list, tuple)):
        return [cpuify(x) for x in obj]
    if isinstance(obj, dict):
        return {k: cpuify(v) for k, v in obj.items()}
    return obj


def encode_file(wav_path: Path, out_path: Path, model, processor, device, overwrite: bool):
    """
    Encode a single WAV file to EnCodec tokens.
    
    Args:
        wav_path: Input WAV file path
        out_path: Output .ecdc file path
        model: EnCodec model
        processor: EnCodec processor
        device: torch device
        overwrite: Whether to overwrite existing files
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        if out_path.exists() and not overwrite:
            return True, "exists"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Load audio at 24kHz mono (should already be normalized)
        audio, sr_og = librosa.load(str(wav_path), sr=None, mono=True)
        audio, _ = librosa.load(str(wav_path), sr=24000, mono=True)
        if _ != sr_og:
            audio = audio[:-1]
        
        # Ensure audio is C-contiguous numpy array
        if not audio.flags.c_contiguous:
            audio = np.ascontiguousarray(audio)

        inputs = processor(
            raw_audio=audio,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        )
        # Ensure tensors are contiguous before moving to device to avoid cuDNN errors
        inputs = {k: v.contiguous().to(device) for k, v in inputs.items()}

        with torch.no_grad():
            enc = model.encode(inputs["input_values"], inputs.get("padding_mask", None))

        save_data = {
            "audio_codes": cpuify(enc.audio_codes),
            "audio_scales": cpuify(enc.audio_scales),
            "audio_length": int(audio.shape[-1]),
        }

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save(save_data, tmp_path)
        os.replace(tmp_path, out_path)

        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def expected_out_path(in_dir: Path, out_dir: Path, wav_path: Path, suffix: str) -> Path:
    """Calculate expected output path maintaining relative structure."""
    rel = wav_path.relative_to(in_dir)  # keep relative structure
    return (out_dir / rel).with_suffix(suffix)


def verify_tokens(out_dir: Path, recursive: bool, suffix: str, max_samples: int):
    """
    Verify encoded token files by sampling and checking structure.
    
    Args:
        out_dir: Directory containing token files
        recursive: Search recursively
        suffix: Token file suffix (.ecdc)
        max_samples: Maximum number of files to check
        
    Returns:
        Tuple of (checked_count, errors_list)
    """
    toks = list(iter_token_files(out_dir, suffix, recursive))
    if not toks:
        return 0, [(out_dir, "No token files found")]

    sample = toks if len(toks) <= max_samples else random.sample(toks, max_samples)

    errors = []
    checked = 0
    for tp in sample:
        try:
            obj = torch.load(tp, map_location="cpu")
            if not isinstance(obj, dict):
                errors.append((tp, "Not a dict"))
                continue
            for key in ("audio_codes", "audio_scales", "audio_length"):
                if key not in obj:
                    errors.append((tp, f"Missing key {key}"))
            if isinstance(obj.get("audio_codes"), (list, tuple)):
                if len(obj["audio_codes"]) == 0:
                    errors.append((tp, "audio_codes empty"))
            else:
                errors.append((tp, "audio_codes not list/tuple"))
            if not isinstance(obj.get("audio_length"), int) or obj["audio_length"] <= 0:
                errors.append((tp, "audio_length invalid"))
            checked += 1
        except Exception as e:
            errors.append((tp, f"{type(e).__name__}: {e}"))
    return checked, errors


def encode_dataset(
    input_folder: Path,
    output_folder: Path,
    bandwidth: float = 6.0,
    device: str = "auto",
    overwrite: bool = False,
    verify: bool = False
) -> dict:
    """
    Encode all WAV files in a dataset directory to EnCodec tokens.
    
    Args:
        input_folder: Input directory containing normalized WAV files
        output_folder: Output directory for .ecdc token files
        bandwidth: EnCodec bandwidth in kbps (default 6.0)
        device: Device to use ("auto", "cuda", or "cpu")
        overwrite: Whether to overwrite existing files
        verify: Whether to verify outputs after encoding
        
    Returns:
        Dictionary with processing statistics
    """
    print(f"\n\033[1mDATA PROCESSING:\033[0m\n")


    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Collect WAV files (only wav extension since they should be normalized)
    wav_files = find_audio_files(input_folder, extensions=['wav', 'WAV'])
    
    if not wav_files:
        print(f"No WAV files found in {input_folder}")
        return {'total': 0, 'success': 0, 'skipped': 0, 'failed': 0}
    
    print(f"Found {len(wav_files)} WAV files under {input_folder}")
    
    # Setup device
    if device == "cuda":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cpu":
        torch_device = torch.device("cpu")
    else:  # auto
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch_device}")
    
    # Load EnCodec model
    print("Loading EnCodec model (facebook/encodec_24khz)...")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    model.config.target_bandwidths = [bandwidth]
    model.to(torch_device).eval()
    
    # Encode files
    ok, skipped, failed = 0, 0, 0
    errors = []
    
    for wav_file in tqdm(wav_files, desc="Encoding", unit="file"):
        out_path = expected_out_path(input_folder, output_folder, wav_file, ".ecdc")
        success, msg = encode_file(wav_file, out_path, model, processor, torch_device, overwrite)
        
        if success:
            if msg == "exists":
                skipped += 1
            else:
                ok += 1
        else:
            failed += 1
            errors.append((wav_file, msg))
    
    # print("\n" + "=" * 60)
    # print("Encoding Summary")
    # print("=" * 60)
    # print(f"Encoded : {ok}")
    # print(f"Skipped : {skipped} (already exist)")
    # print(f"Failed  : {failed}")
    
    if errors:
        print(f"\nShowing first {min(10, len(errors))} errors:")
        for wav_path, msg in errors[:10]:
            print(f"  ERR {wav_path.name}: {msg}")
    
    # Verify if requested
    if verify and (ok > 0 or skipped > 0):
        print("\nVerifying encoded files...")
        checked, errs = verify_tokens(output_folder, recursive=True, suffix=".ecdc", max_samples=20)
        print(f"Verified: {checked} files checked, {len(errs)} errors found")
        if errs:
            print(f"Showing first {min(10, len(errs))} verification errors:")
            for path, msg in errs[:10]:
                print(f"  ERR {path.name}: {msg}")
    
    return {
        'total': len(wav_files),
        'success': ok,
        'skipped': skipped,
        'failed': failed,
        'errors': errors
    }


def inspect_ecdc_files(tokens_folder: Path) -> dict:
    """
    Inspect .ecdc files and report their shapes and properties.
    
    Args:
        tokens_folder: Path to folder containing .ecdc files
        
    Returns:
        Dictionary with inspection results
    """
    tokens_folder = Path(tokens_folder)
    
    if not tokens_folder.exists():
        print(f"❌ Tokens folder not found: {tokens_folder}")
        return {'files': [], 'error': 'folder_not_found'}
    
    # Find all .ecdc files
    ecdc_files = list(tokens_folder.rglob("*.ecdc"))
    
    if not ecdc_files:
        print(f"❌ No .ecdc files found in {tokens_folder}")
        return {'files': [], 'error': 'no_files'}
    
    # print(f"\n📊 Inspecting {len(ecdc_files)} .ecdc files in {tokens_folder}")
    # print("=" * 70)
    
    results = []
    for ecdc_file in sorted(ecdc_files):
        try:
            # Load the .ecdc file
            data = torch.load(ecdc_file, map_location="cpu")
            
            # Extract information
            audio_codes = data.get("audio_codes")
            audio_scales = data.get("audio_scales") 
            audio_length = data.get("audio_length")
            
            if audio_codes is not None:
                if isinstance(audio_codes, torch.Tensor):
                    codes_shape = tuple(audio_codes.shape)
                    codes_dtype = str(audio_codes.dtype)
                else:
                    codes_shape = "Not a tensor"
                    codes_dtype = str(type(audio_codes))
                
                # Calculate EnCodec frames and duration
                if len(codes_shape) >= 2:
                    if len(codes_shape) == 3:  # [1, Cb, T] or [Cb, 1, T]
                        frames = codes_shape[2] if codes_shape[1] > codes_shape[2] else codes_shape[1]
                        codebooks = codes_shape[1] if codes_shape[1] > codes_shape[2] else codes_shape[2]
                    elif len(codes_shape) == 2:  # [Cb, T]
                        codebooks = codes_shape[0]
                        frames = codes_shape[1]
                    else:
                        frames = "unknown"
                        codebooks = "unknown"
                else:
                    frames = "unknown"
                    codebooks = "unknown"
                
                duration = frames / 75 if isinstance(frames, int) else "unknown"
            else:
                codes_shape = "Missing"
                codes_dtype = "Missing"
                frames = "unknown"
                codebooks = "unknown"
                duration = "unknown"
            
            file_info = {
                'name': ecdc_file.name,
                'path': ecdc_file,
                'audio_codes_shape': codes_shape,
                'audio_codes_dtype': codes_dtype,
                'frames': frames,
                'codebooks': codebooks,
                'duration_seconds': duration,
                'audio_length': audio_length,
                'has_scales': audio_scales is not None
            }
            
            results.append(file_info)
            
            # Print file info
            print(f"📁 {ecdc_file.name}")
            print(f"   Audio codes shape: {codes_shape}")
            print(f"   Frames: {frames} ({duration:.2f}s @ 75Hz)" if isinstance(duration, (int, float)) else f"   Frames: {frames}")
            print(f"   Codebooks: {codebooks}")
            print(f"   Original length: {audio_length} samples" if audio_length else "   Original length: not recorded")
            print()
            
        except Exception as e:
            error_info = {
                'name': ecdc_file.name,
                'path': ecdc_file,
                'error': str(e)
            }
            results.append(error_info)
            print(f"❌ Error reading {ecdc_file.name}: {e}")
            print()
    
    # Summary statistics
    valid_files = [r for r in results if 'error' not in r]
    if valid_files:
        total_duration = sum(r['duration_seconds'] for r in valid_files if isinstance(r['duration_seconds'], (int, float)))
        unique_shapes = set(str(r['audio_codes_shape']) for r in valid_files)
        unique_codebooks = set(r['codebooks'] for r in valid_files if r['codebooks'] != 'unknown')
        
        # print("=" * 70)
        # print("📈 Summary:")
        # print(f"   Valid files: {len(valid_files)}")
        # print(f"   Total duration: {total_duration:.1f} seconds" if total_duration > 0 else "   Total duration: unknown")
        # print(f"   Unique shapes: {', '.join(unique_shapes)}")
        # print(f"   Codebooks used: {', '.join(map(str, unique_codebooks))}" if unique_codebooks else "   Codebooks: unknown")
    
    return {'files': results, 'summary': {'valid': len(valid_files), 'total': len(ecdc_files)}}


def quick_inspect_ecdc(dataset_path):
    """
    Quick inspection for notebook - inspect .ecdc files in tokens folder.
    
    Args:
        dataset_path: Path to dataset directory (e.g., "./dataset_01")
        
    Example:
        quick_inspect_ecdc('./data/dataset_01')
    """
    print(f"\n\033[1mDATA SUMMARY:\033[0m\n")


    dataset_path = Path(dataset_path)
    tokens_dir = dataset_path / 'tokens'
    return inspect_ecdc_files(tokens_dir)

def quick_encode(dataset_path, bandwidth=6.0, device="cpu", overwrite=False):  # Force CPU for testing
    """
    Quick encoding for notebook - encodes normalized audio to tokens.
    
    This function follows the EncHF_Dataset convention:
    - Input: {dataset_path}/normalized/*.wav
    - Output: {dataset_path}/tokens/*.ecdc
    
    Args:
        dataset_path: Path to dataset directory (e.g., "./dataset_01")
        bandwidth: EnCodec bandwidth in kbps (default 6.0)
        device: Device to use ("auto", "cuda", or "cpu")
        overwrite: Whether to overwrite existing files
        
    Example:
        quick_encode('./data/dataset_01')
    """
    dataset_path = Path(dataset_path)
    input_dir = dataset_path / 'normalized'
    output_dir = dataset_path / 'tokens'
    
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Normalized folder not found: {input_dir}\n"
            f"Please run step_1_normalization first."
        )
    
    result = encode_dataset(
        input_folder=input_dir,
        output_folder=output_dir,
        bandwidth=bandwidth,
        device=device,
        overwrite=overwrite,
        verify=False
    )
    
    # If encoding was successful, inspect the created files
    # print("\n" + "=" * 70)
    # print("🔍 INSPECTING CREATED .ECDC FILES")
    # print("=" * 70)
    quick_inspect_ecdc(dataset_path)
    
    return
