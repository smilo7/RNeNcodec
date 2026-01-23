"""
Step 4: Create HuggingFace Dataset

This script converts the processed tokens and sidecar files into a HuggingFace dataset
that can be used for training. Based on s5_build_dataset_from_folder.py from EncHF_Dataset.
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from datasets import Dataset, DatasetDict

# Disable datasets progress bars
import datasets
datasets.disable_progress_bar()


def expand_parameters_config(parameters_path: Path) -> Dict:
    """
    Expand the parameters.json configuration to include individual feature names
    for one-hot encoded categorical parameters.
    
    This creates a smarter representation where categorical parameters are expanded
    to show all the individual binary features they generate.
    
    Example:
        Input:  {"instrument": {"type": "class", "classes": ["sinusoid", "piano"]}}
        Output: {
            "tempo": {...},
            "reverb": {...},
            "instrument_sinusoid": {"type": "binary", "source_parameter": "instrument", ...},
            "instrument_piano": {"type": "binary", "source_parameter": "instrument", ...}
        }
    
    Args:
        parameters_path: Path to original parameters.json file
        
    Returns:
        Expanded dictionary with individual feature definitions
    """
    with open(parameters_path, 'r') as f:
        original_config = json.load(f)
    
    expanded_config = {}
    feature_list = []  # Ordered list of feature names
    
    for param_key, param_info in original_config.items():
        param_name = param_info['name']
        param_type = param_info.get('type', 'continuous')
        
        if param_type == 'continuous':
            # Continuous parameters remain as single features
            expanded_config[param_name] = {
                "type": "continuous",
                "min": param_info.get('min', 0.0),
                "max": param_info.get('max', 1.0),
                "unit": param_info.get('unit', ''),
                "description": f"Continuous parameter: {param_name}"
            }
            feature_list.append(param_name)
            
        elif param_type == 'class':
            # Categorical parameters expand to multiple binary features (one-hot encoding)
            classes = param_info.get('classes', [])
            num_classes = len(classes) if classes else param_info.get('num_classes', 2)
            
            if classes:
                # Use actual class names
                for class_name in classes:
                    feature_name = f"{param_name}_{class_name}"
                    expanded_config[feature_name] = {
                        "type": "binary",
                        "source_parameter": param_name,
                        "class_name": class_name,
                        "unit": "probability",
                        "description": f"One-hot encoding for {param_name} class '{class_name}'"
                    }
                    feature_list.append(feature_name)
            else:
                # Fallback to indexed names
                for i in range(num_classes):
                    feature_name = f"{param_name}_{i}"
                    expanded_config[feature_name] = {
                        "type": "binary",
                        "source_parameter": param_name,
                        "class_index": i,
                        "unit": "probability",
                        "description": f"One-hot encoding for {param_name} class {i}"
                    }
                    feature_list.append(feature_name)
    
    # Add metadata
    result = {
        "schema_version": 1,
        "fps": 75,  # EnCodec frame rate
        "feature_names": feature_list,
        "features": expanded_config,
        "num_features": len(feature_list)
    }
    
    return result


def detect_split_structure(tokens_dir: Path) -> Optional[List[str]]:
     """
     Detect if tokens directory has train/validation/test split structure.

     Strict requirements:
     - Folders must be directly inside tokens_dir (not nested deeper)
     - 'train' folder MUST exist
     - Other splits are optional
     - Recognizes any folder names as potential splits

     Args:
         tokens_dir: Directory to check for split structure

     Returns:
         List of detected split names (with 'train' first) or None if no
valid structure
     """
     tokens_dir = Path(tokens_dir)

     # Check for train folder - required
     if not (tokens_dir / 'train').is_dir():
         return None

     # Find all directories directly inside tokens_dir
     all_splits = [d.name for d in tokens_dir.iterdir() if d.is_dir()]

     # Build list with 'train' first, then others in sorted order
     other_splits = sorted([s for s in all_splits if s != 'train'])
     splits = ['train'] + other_splits

     return splits


def collect_token_files(tokens_dir: Path, suffix: str = ".ecdc", recursive: bool = True) -> List[Path]:
    """
    Find all token files in the tokens directory.
    
    Args:
        tokens_dir: Directory containing .ecdc files
        suffix: File extension to search for
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        Sorted list of token file paths
    """
    pattern = f"*{suffix}"
    
    if recursive:
        paths = sorted(tokens_dir.rglob(pattern), key=lambda p: p.as_posix().lower())
    else:
        paths = sorted(tokens_dir.glob(pattern), key=lambda p: p.as_posix().lower())
    
    return [p for p in paths if p.is_file()]


def verify_sidecar_files(ecdc_path: Path) -> Dict[str, bool]:
    """
    Check if sidecar files exist for a given .ecdc file.
    Note: JSON files are no longer created per file (only .npy files).
    
    Args:
        ecdc_path: Path to .ecdc file
        
    Returns:
        Dictionary indicating which sidecar files exist
    """
    cond_npy_path = ecdc_path.with_suffix('.cond.npy')
    
    return {
        'npy_exists': cond_npy_path.exists(),
        'both_exist': cond_npy_path.exists()  # Only .npy is required now
    }


def materialize_files(src_ecdc: Path, dst_ecdc: Path, materialize_mode: str = "link"):
    """
    Create links or copies of .ecdc and sidecar files in the dataset directory.
    Note: Only .npy sidecar files are materialized (JSON no longer created per file).
    
    Args:
        src_ecdc: Source .ecdc file path
        dst_ecdc: Destination .ecdc file path in dataset
        materialize_mode: How to materialize files ("link", "copy", or "none")
    """
    # Create parent directory
    dst_ecdc.parent.mkdir(parents=True, exist_ok=True)
    
    # List of files to materialize (only .ecdc and .cond.npy)
    files_to_materialize = [
        (src_ecdc, dst_ecdc),
        (src_ecdc.with_suffix('.cond.npy'), dst_ecdc.with_suffix('.cond.npy'))
    ]
    
    for src_file, dst_file in files_to_materialize:
        if not src_file.exists():
            continue
            
        if materialize_mode == "none":
            continue
            
        # Remove existing file/link before creating new one
        if dst_file.exists() or dst_file.is_symlink():
            dst_file.unlink()
            
        # Ensure parent directory exists
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        
        if materialize_mode == "link":
            # Create relative symlink - calculate path from dst to src
            try:
                target = os.path.relpath(src_file.resolve(), start=dst_file.parent.resolve())
                os.symlink(target, dst_file)
            except OSError as e:
                print(f"⚠️ Failed to create symlink for {dst_file.name}: {e}")
                print(f"   Falling back to copy mode for this file")
                shutil.copy2(src_file, dst_file)
        elif materialize_mode == "copy":
            # Copy the file
            shutil.copy2(src_file, dst_file)
        else:
            raise ValueError(f"Unknown materialize mode: {materialize_mode}")


def verify_dataset_files(df: pd.DataFrame, output_dir: Path) -> int:
    """
    Verify that all referenced audio files exist in the dataset.
    
    Args:
        df: DataFrame with 'audio' column containing relative paths
        output_dir: Root directory of the dataset
        
    Returns:
        Number of missing files
    """
    missing_files = []
    
    for _, row in df.iterrows():
        audio_path = output_dir / row["audio"]
        if not audio_path.exists():
            missing_files.append(str(audio_path))
    
    if missing_files:
        print(f"⚠️ Found {len(missing_files)} missing files:")
        for missing in missing_files[:10]:  # Show first 10
            print(f"   • {missing}")
        if len(missing_files) > 10:
            print(f"   ... and {len(missing_files) - 10} more")
    else:
        print("✅ All audio files verified successfully")
    
    return len(missing_files)


def cleanup_dataset_duplicates(output_dir: Path, tokens_subdir: str = "tokens"):
    """
    Clean up any duplicate files in the dataset directory that might exist
    from previous runs or incorrect materialization.
    
    Args:
        output_dir: Root directory of the HuggingFace dataset
        tokens_subdir: Subdirectory name for tokens within the dataset
    """
    tokens_dir = output_dir / tokens_subdir
    
    if not tokens_dir.exists():
        return
    
    print(f"🧹 Cleaning up dataset directory: {tokens_dir}")
    
    # Remove the entire tokens subdirectory and recreate it
    if tokens_dir.exists():
        shutil.rmtree(tokens_dir)
    
    tokens_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Cleaned up tokens directory")


def create_single_split_dataset(
    tokens_split_dir: Path,
    output_dir: Path,
    split_name: str,
    tokens_subdir: str,
    materialize_mode: str
) -> tuple[Dataset, Dict]:
    """
    Create dataset for a single split.
    
    Args:
        tokens_split_dir: Directory containing .ecdc files for this split
        output_dir: Output directory for the dataset
        split_name: Name of this split
        tokens_subdir: Subdirectory name for tokens
        materialize_mode: How to handle files
        
    Returns:
        Tuple of (Dataset object, statistics dict)
    """
    dst_tokens_root = output_dir / tokens_subdir
    
    # Find all token files in this split
    token_paths = collect_token_files(tokens_split_dir, suffix=".ecdc", recursive=True)
    
    if not token_paths:
        print(f"   ⚠️  No .ecdc files found in '{split_name}' split")
        return Dataset.from_pandas(pd.DataFrame(columns=["audio"])), {
            'total_files': 0,
            'valid_files': 0,
            'missing_sidecars': 0
        }
    
    print(f"   📂 {split_name}: {len(token_paths)} .ecdc files found")
    
    # Verify sidecar files and create dataset rows
    rows = []
    valid_files = 0
    missing_sidecars = 0
    
    for ecdc_path in token_paths:
        # Check sidecar files
        sidecar_status = verify_sidecar_files(ecdc_path)
        
        if not sidecar_status['both_exist']:
            missing_sidecars += 1
            continue
        
        # Calculate relative path within dataset (preserve split folder structure)
        rel_within_split = ecdc_path.relative_to(tokens_split_dir)
        rel_path_in_dataset = Path(tokens_subdir) / split_name / rel_within_split
        
        # Materialize files (create links/copies)
        dst_ecdc_path = output_dir / rel_path_in_dataset
        materialize_files(ecdc_path, dst_ecdc_path, materialize_mode)
        
        # Add row to dataset
        rows.append({"audio": str(rel_path_in_dataset)})
        valid_files += 1
    
    if missing_sidecars > 0:
        print(f"      ⚠️  {missing_sidecars} files missing sidecars")
    
    # Create dataset
    if rows:
        df = pd.DataFrame(rows)
        dataset = Dataset.from_pandas(df, preserve_index=False)
    else:
        dataset = Dataset.from_pandas(pd.DataFrame(columns=["audio"]))
    
    stats = {
        'total_files': len(token_paths),
        'valid_files': valid_files,
        'missing_sidecars': missing_sidecars
    }
    
    return dataset, stats


def create_huggingface_dataset(
    tokens_dir: Path, 
    output_dir: Path,
    raw_dir: Optional[Path] = None,
    split_name: str = "train",
    tokens_subdir: str = "tokens",
    materialize_mode: str = "link",
    verify_files: bool = True
) -> Dict:
    """
    Create a HuggingFace dataset from tokens and sidecar files.
    Automatically detects and handles train/validation/test split structure.
    
    Args:
        tokens_dir: Directory containing .ecdc and sidecar files
        output_dir: Output directory for the HuggingFace dataset
        raw_dir: Directory containing raw data and parameters.json (optional)
        split_name: Default split name if no split structure detected
        tokens_subdir: Subdirectory name for tokens within the dataset
        materialize_mode: How to handle files ("link", "copy", "none")
        verify_files: Whether to verify all files exist after creation
        
    Returns:
        Dictionary with creation statistics
    """
    tokens_dir = Path(tokens_dir)
    output_dir = Path(output_dir)
    if raw_dir:
        raw_dir = Path(raw_dir)
    
    print(f"\n\033[1mHUGGINGFACE DATASET CREATION:\033[0m\n")
    print(f"📁 Tokens directory: {tokens_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔗 Materialize mode: {materialize_mode}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up any existing files from previous runs
    cleanup_dataset_duplicates(output_dir, tokens_subdir)
    
    # Detect split structure
    detected_splits = detect_split_structure(tokens_dir)
    
    if detected_splits:
        # Multi-split mode: train/validation/test structure detected
        print(f"\n🎯 Split structure detected: {', '.join(detected_splits)}")
        print(f"� Creating separate HuggingFace splits for each folder\n")
        
        datasets = {}
        all_stats = {}
        
        for split in detected_splits:
            split_dir = tokens_dir / split
            dataset, stats = create_single_split_dataset(
                tokens_split_dir=split_dir,
                output_dir=output_dir,
                split_name=split,
                tokens_subdir=tokens_subdir,
                materialize_mode=materialize_mode
            )
            datasets[split] = dataset
            all_stats[split] = stats
        
        # Create DatasetDict with all splits
        dataset_dict = DatasetDict(datasets)
        
        # Calculate totals
        total_files = sum(s['total_files'] for s in all_stats.values())
        total_valid = sum(s['valid_files'] for s in all_stats.values())
        total_missing = sum(s['missing_sidecars'] for s in all_stats.values())
        
    else:
        # Single-split mode: no split structure, use all files as single split
        print(f"\n📊 No split structure detected")
        print(f"📌 Using all data as '{split_name}' split\n")
        
        dataset, stats = create_single_split_dataset(
            tokens_split_dir=tokens_dir,
            output_dir=output_dir,
            split_name=split_name,
            tokens_subdir=tokens_subdir,
            materialize_mode=materialize_mode
        )
        
        # Create DatasetDict with single split
        datasets = {split_name: dataset}
        dataset_dict = DatasetDict(datasets)
        
        total_files = stats['total_files']
        total_valid = stats['valid_files']
        total_missing = stats['missing_sidecars']
    
    # Save to disk
    dataset_dict.save_to_disk(str(output_dir))
    
    # Save expanded parameters configuration if raw_dir is provided
    if raw_dir:
        parameters_path = raw_dir / 'parameters.json'
        if parameters_path.exists():
            try:
                expanded_params = expand_parameters_config(parameters_path)
                
                # Save to conditioning_config.json in the dataset root
                config_output_path = output_dir / 'conditioning_config.json'
                with open(config_output_path, 'w') as f:
                    json.dump(expanded_params, f, indent=2)
                
                print(f"\n📋 Saved expanded conditioning config to: {config_output_path.name}")
                print(f"   • Features: {', '.join(expanded_params['feature_names'])}")
                print(f"   • Total features: {expanded_params['num_features']}")
            except Exception as e:
                print(f"\n⚠️  Warning: Could not create conditioning config: {e}")
        else:
            print(f"\n⚠️  Warning: parameters.json not found in {raw_dir}")
    
    print(f"\n💾 Saved DatasetDict to: {output_dir}")
    print(f"📈 Dataset summary:")
    for split, dataset in dataset_dict.items():
        print(f"   • {split}: {len(dataset)} samples")
    
    # Verify files if requested
    missing_count = 0
    if verify_files and total_valid > 0:
        print(f"\n🔍 Verifying dataset files...")
        # Combine all rows for verification
        all_rows = []
        for dataset in dataset_dict.values():
            if len(dataset) > 0:
                all_rows.extend(dataset.to_pandas()['audio'].tolist())
        
        if all_rows:
            df_verify = pd.DataFrame({'audio': all_rows})
            missing_count = verify_dataset_files(df_verify, output_dir)
    
    return {
        'splits_detected': detected_splits if detected_splits else [split_name],
        'total_ecdc_files': total_files,
        'valid_files': total_valid,
        'missing_sidecars': total_missing,
        'missing_files': missing_count,
        'output_dir': str(output_dir)
    }


# Convenience function for notebook use
def quick_create_dataset(
    dataset_dir: str, 
    split_name: str = "train",
    materialize_mode: str = "link"
) -> Dict:
    """
    Quick HuggingFace dataset creation for notebook - just call this function!
    
    Automatically detects split structure:
    - If tokens/ has train + (validation or test) folders → creates multi-split dataset
    - Otherwise → creates single split with all data
    
    Args:
        dataset_dir: Root dataset directory (e.g., './data/dataset_01')
        split_name: Default split name if no structure detected (default: "train")
        materialize_mode: How to handle files ("link", "copy", "none")
        
    Returns:
        Dictionary with creation statistics
    
    Example:
        results = quick_create_dataset('./data/dataset_01')
    """
    dataset_path = Path(dataset_dir)
    tokens_dir = dataset_path / 'tokens'
    output_dir = dataset_path / 'hf_dataset'
    raw_dir = dataset_path / 'raw'
    
    if not tokens_dir.exists():
        raise FileNotFoundError(f"Tokens directory not found: {tokens_dir}")
    
    return create_huggingface_dataset(
        tokens_dir=tokens_dir,
        output_dir=output_dir,
        raw_dir=raw_dir if raw_dir.exists() else None,
        split_name=split_name,
        tokens_subdir="tokens",
        materialize_mode=materialize_mode,
        verify_files=True
    )


def quick_load_dataset(dataset_dir: str):
    """
    Quick dataset loading helper for notebook.
    
    Args:
        dataset_dir: Root dataset directory (e.g., './data/dataset_01')
        
    Returns:
        Loaded HuggingFace DatasetDict
    
    Example:
        dataset_dict = quick_load_dataset('./data/dataset_01')
        train_data = dataset_dict['train']
    """
    try:
        from datasets import load_from_disk
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    dataset_path = Path(dataset_dir)
    hf_dataset_dir = dataset_path / 'hf_dataset'
    
    if not hf_dataset_dir.exists():
        raise FileNotFoundError(f"HuggingFace dataset not found: {hf_dataset_dir}")
    
    return load_from_disk(str(hf_dataset_dir))
