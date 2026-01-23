"""
Audio Normalization for Dataset Pipeline

This script normalizes audio files to a consistent peak windowed RMS level
and resamples to 24kHz mono, matching the EncHF_Dataset approach.

The normalization process:
1. Loads audio at 24kHz mono using librosa (supports mp3, flac, wav, etc.)
2. Calculates peak windowed RMS (250ms windows with 75% overlap)
3. Applies gain to normalize to target RMS level
4. Saves normalized audio as WAV maintaining folder structure

Based on: lonce/EncHF_Dataset/workflow/s1_audio_normalize.py
"""

import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import math
from pathlib import Path
import argparse
from typing import Optional

from dataprep.auxiliary_functions import find_audio_files


def calculate_windowed_rms(audio: np.ndarray, sample_rate: int, window_ms: int = 250) -> np.ndarray:
    """
    Calculate RMS values over sliding windows.
    
    Uses 75% overlap (hop = window_size / 4) for better precision.
    
    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate in Hz
        window_ms: Window size in milliseconds
        
    Returns:
        Array of RMS values for each window
    """
    window_samples = int(sample_rate * window_ms / 1000)
    hop_samples = window_samples // 4  # 75% overlap for better precision
    
    if len(audio) < window_samples:
        # If audio is shorter than window, return single RMS
        return np.array([np.sqrt(np.mean(audio**2))])
    
    rms_values = []
    for i in range(0, len(audio) - window_samples + 1, hop_samples):
        window = audio[i:i + window_samples]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    
    return np.array(rms_values)


def get_peak_windowed_rms(filepath: Path, window_ms: int = 250) -> Optional[float]:
    """
    Get the peak RMS value from windowed analysis after converting to 24kHz mono.
    
    This function:
    1. Uses librosa to load audio at 24kHz mono (supports any format)
    2. Calculates windowed RMS values
    3. Returns the peak (maximum) RMS value
    
    Args:
        filepath: Path to input audio file
        window_ms: Window size in milliseconds
        
    Returns:
        Peak RMS value, or None if error
    """
    try:
        # Load audio at 24kHz mono using librosa (handles mp3, flac, wav, etc.)
        audio, sr_og = librosa.load(str(filepath), sr=None, mono=True)
        audio, sr = librosa.load(str(filepath), sr=24000, mono=True)
        if sr != sr_og:
            audio = audio[:-12]

        
        # Calculate windowed RMS
        rms_values = calculate_windowed_rms(audio, sr, window_ms)
        peak_rms = np.max(rms_values)
        
        return peak_rms
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def process_file(input_path: Path, output_path: Path, target_peak_rms: float, window_ms: int = 250, apply_rms_normalization: bool = True) -> bool:
    """
    Process a single audio file: normalize RMS and convert to 24kHz mono WAV.
    
    This function also ensures the audio length matches CSV annotations:
    - If CSV has fewer frames than audio, audio is trimmed to match CSV
    - If CSV has more frames than audio, audio length is preserved (extra CSV rows ignored later)
    
    Args:
        input_path: Input audio file path (any format supported by librosa)
        output_path: Output audio file path (will be saved as WAV)
        target_peak_rms: Target peak RMS value
        window_ms: Window size for RMS calculation
        apply_rms_normalization: If True, apply RMS normalization. If False, only resample to 24kHz
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Processing: {input_path.name}")
    
    if apply_rms_normalization:
        # Get current peak windowed RMS
        current_peak_rms = get_peak_windowed_rms(input_path, window_ms)
        
        if current_peak_rms is None or current_peak_rms == 0:
            print(f"    ✗ Skipping - could not calculate RMS")
            return False
        
        # Calculate gain needed
        gain_linear = target_peak_rms / current_peak_rms
        gain_db = 20 * np.log10(gain_linear)
        
        print(f"    Current peak RMS: {current_peak_rms:.6f}")
        print(f"    Target peak RMS: {target_peak_rms:.6f}")
        print(f"    Applying gain: {gain_db:.2f} dB")
    else:
        print(f"    RMS normalization disabled - resampling to 24kHz only")
    
    try:
        # Load audio at 24kHz mono
        audio, sr_og = librosa.load(str(input_path), sr=None, mono=True)
        audio, sr = librosa.load(str(input_path), sr=24000, mono=True)
        if sr != sr_og:
            audio = audio[:-12]

        # Apply gain only if RMS normalization is enabled
        if apply_rms_normalization:
            audio_normalized = audio * gain_linear
        else:
            audio_normalized = audio
        
        # Check for corresponding CSV file to align lengths
        csv_path = input_path.with_suffix('.csv')
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                csv_frames = len(df)
                
                # Calculate expected EnCodec frames using ceiling division (same logic as auxiliary_functions)
                samples_per_frame = sr // 75  # 320 samples per frame at 24kHz
                expected_frames = math.ceil(len(audio_normalized) / samples_per_frame)
                
                # Take minimum to ensure alignment (same logic as visualization)
                aligned_frames = min(expected_frames, csv_frames)
                
                # If we need to trim audio to match CSV
                if aligned_frames < expected_frames:
                    aligned_duration = aligned_frames / 75  # Convert frames to seconds
                    aligned_samples = int(aligned_duration * sr)
                    audio_normalized = audio_normalized[:aligned_samples]
                    
                    extra_audio_frames = expected_frames - aligned_frames
                    print(f"    🔄 Trimmed audio: {expected_frames} → {aligned_frames} frames ({extra_audio_frames} extra audio frames in audio file)")
                elif csv_frames > expected_frames:
                    extra_csv_frames = csv_frames - expected_frames
                    print(f"    ℹ️  Alignment: {csv_frames} CSV frames vs {expected_frames} audio frames ({extra_csv_frames} extra annotation frames in CSV)")
                    print(f"    (Extra CSV rows will be ignored in later steps)")
                    
            except Exception as e:
                print(f"    ⚠️  Warning: Could not check CSV alignment: {e}")
                print(f"    Proceeding with full audio length")
        
        # Save as WAV at 24kHz
        sf.write(str(output_path), audio_normalized, sr, subtype='PCM_16')
        
        print(f"    ✓ Saved: {output_path.name}")
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def normalize_dataset(
    input_folder: Path,
    output_folder: Path,
    target_rms: float = 0.1,
    window_ms: int = 250,
    apply_rms_normalization: bool = True
) -> dict:
    """
    Normalize all audio-CSV pairs in a dataset directory.
    Only processes audio files that have corresponding CSV annotation files.
    
    Args:
        input_folder: Input directory containing audio-CSV pairs
        output_folder: Output directory for normalized files
        target_rms: Target peak windowed RMS value
        window_ms: Window size for RMS calculation in milliseconds
        apply_rms_normalization: If True, apply RMS normalization. If False, only resample to 24kHz
        
    Returns:
        Dictionary with processing statistics
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder {input_folder} does not exist")
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find audio-CSV pairs (only process files with both audio and CSV)
    from dataprep.auxiliary_functions import find_audio_csv_pairs
    
    audio_csv_pairs = find_audio_csv_pairs(input_folder)
    
    if not audio_csv_pairs:
        print(f"No audio-CSV pairs found in {input_folder}")
        return {'total': 0, 'success': 0, 'failed': 0}
    
    print(f"\n\033[1mDATA SUMMARY:\033[0m\n")
    print(f"Found {len(audio_csv_pairs)} audio-CSV pairs")
    if apply_rms_normalization:
        print(f"Target peak windowed RMS: {target_rms}")
        print(f"Window size: {window_ms}ms")
    else:
        print(f"RMS normalization: DISABLED (resampling to 24kHz only)")
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    # print("=" * 60)
    
    print(f"\n\033[1mDATA PROCESSING:\033[0m\n")

    success_count = 0
    
    for audio_file, csv_file in sorted(audio_csv_pairs):
        # Calculate relative path from input folder
        rel_path = audio_file.relative_to(input_folder)
        
        # Create output paths maintaining folder structure
        output_audio = output_folder / rel_path.parent / f"{audio_file.stem}.wav"
        output_csv = output_folder / rel_path.parent / f"{audio_file.stem}.csv"
        
        # Create output subdirectory if needed
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        
        if process_file(audio_file, output_audio, target_rms, window_ms, apply_rms_normalization):
            # Copy the corresponding CSV file to output directory
            import shutil
            try:
                shutil.copy2(csv_file, output_csv)
                print(f"    ✅ Copied CSV: {csv_file.name} → {output_csv}")
                success_count += 1
            except Exception as e:
                print(f"    ❌ Error copying CSV {csv_file}: {e}")
        
        print()  # Empty line for readability
    
    # print("=" * 60)
    print(f"Successfully processed {success_count}/{len(audio_csv_pairs)} pairs")
    
    if success_count < len(audio_csv_pairs):
        print(f"Failed to process {len(audio_csv_pairs) - success_count} pairs")
    
    return {
        'total': len(audio_csv_pairs),
        'success': success_count,
        'failed': len(audio_csv_pairs) - success_count
    }


# Convenience functions for notebook use
def quick_normalize(dataset_dir, target_rms=0.1, window_ms=250, apply_rms_normalization=True):
    """
    Quick normalization for notebook - just call this function!
    
    Args:
        dataset_dir: Root dataset directory containing 'raw' folder
        target_rms: Target peak windowed RMS value (default: 0.1)
        window_ms: Window size for RMS calculation in milliseconds (default: 250)
        apply_rms_normalization: If True, apply RMS normalization. If False, only resample to 24kHz (default: True)
    
    Example:
        # With RMS normalization (default):
        quick_normalize('./data')
        
        # Without RMS normalization (only resampling to 24kHz):
        quick_normalize('./data', apply_rms_normalization=False)
    """
    input_dir = dataset_dir + "/raw"
    output_dir = dataset_dir + "/normalized"
    normalize_dataset(
        Path(input_dir),
        Path(output_dir),
        target_rms=target_rms,
        window_ms=window_ms,
        apply_rms_normalization=apply_rms_normalization
    )
    return
