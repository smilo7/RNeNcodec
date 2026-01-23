"""
Shared utility functions for dataset processing pipeline.

These functions are used across multiple dataset processing steps
(visualization, normalization, encoding, etc.) to ensure consistency.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import librosa


def load_parameter_config(config_path: Path) -> Dict:
    """
    Load and parse the parameters.json configuration file.
    
    Args:
        config_path: Path to parameters.json
        
    Returns:
        Dictionary containing parameter configurations
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def find_audio_csv_pairs(raw_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find all audio-CSV pairs in the raw directory, including nested folders.
    
    Uses recursive search (rglob) to discover files in any subdirectory level.
    
    Args:
        raw_dir: Root directory to search for audio files and their CSV annotations
        
    Returns:
        List of tuples (audio_path, csv_path) for matching pairs
    """
    raw_dir = Path(raw_dir)
    
    # Supported audio extensions
    audio_exts = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    
    # Find all audio files recursively
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(raw_dir.rglob(f'*{ext}'))  # rglob for recursive search
    
    # Find matching CSV files
    pairs = []
    for audio_file in audio_files:
        csv_file = audio_file.with_suffix('.csv')
        if csv_file.exists():
            pairs.append((audio_file, csv_file))
        else:
            print(f"⚠️  Warning: No CSV found for {audio_file.name}")
    
    return pairs


def find_audio_files(input_dir: Path, extensions: List[str] = None) -> List[Path]:
    """
    Find all audio files in a directory, including nested folders.
    
    Args:
        input_dir: Root directory to search
        extensions: List of file extensions to search for (default: common audio formats)
        
    Returns:
        List of audio file paths
    """
    input_dir = Path(input_dir)
    
    if extensions is None:
        # Support same formats as find_audio_csv_pairs
        extensions = ['wav', 'WAV', 'mp3', 'MP3', 'flac', 'FLAC', 'm4a', 'M4A', 'ogg', 'OGG']
    
    # Find all audio files recursively
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_dir.rglob(f'*.{ext}'))
    
    return sorted(audio_files)


def get_audio_info(audio_path: Path) -> Dict:
    """
    Get audio file information using the same approach as EncHF_Dataset.
    
    This function loads audio at 24kHz mono to match the EnCodec processing
    pipeline used in the EncHF_Dataset repository.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio metadata:
        - duration: Duration in seconds
        - samplerate: Sample rate (always 24000 after resampling)
        - frames: Number of audio samples
        - channels: Number of channels (always 1 after mono conversion)
        - expected_encodec_frames: Expected number of EnCodec frames at 75Hz
    """
    audio_path_str = str(audio_path)
    
    try:
        # Load audio at 24kHz mono to match EnCodec processing
        # This follows the EncHF_Dataset approach in s2_wav2encodec24.py
        audio, sr_og = librosa.load(audio_path_str, sr=None, mono=True)
        audio, sr = librosa.load(audio_path_str, sr=24000, mono=True)
        if sr != sr_og:
            audio = audio[:-12]
        
        duration = len(audio) / sr
        
        # Calculate EnCodec frames using ceiling division
        # EnCodec uses exactly 320 samples per frame at 24kHz (24000/75=320)
        # If there's even 1 extra sample, it creates an additional frame
        samples_per_frame = sr // 75  # 320 samples per frame at 24kHz
        expected_encodec_frames = math.ceil(len(audio) / samples_per_frame)
        
        return {
            'duration': duration,
            'samplerate': sr,  # Always 24000 after resampling
            'frames': len(audio),
            'channels': 1,  # Always mono after conversion
            'expected_encodec_frames': expected_encodec_frames  # EnCodec frames using ceil division
        }
    except Exception as e:
        print(f"❌ Error reading {audio_path}: {e}")
        return None


def get_parameter_names(config: Dict) -> List[str]:
    """
    Extract names of continuous (non-class) parameters from config.
    
    Args:
        config: Parameter configuration dictionary from parameters.json
        
    Returns:
        List of continuous parameter names
    """
    params = []
    for param_info in config.values():
        params.append(param_info['name'])
    return params


def get_parameter_unit(config: Dict, param_name: str) -> str:
    """
    Get the unit string for a parameter from config.
    
    Args:
        config: Parameter configuration dictionary
        param_name: Name of the parameter
        
    Returns:
        Unit string or None if not found
    """
    for param_info in config.values():
        if param_info['name'] == param_name:
            return param_info.get('unit', None)
    return None
