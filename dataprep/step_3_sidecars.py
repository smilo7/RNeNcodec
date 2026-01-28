"""
Step 3: Create Sidecar Files

This script creates sidecar files (.cond.npy and .cond.json) for each .ecdc file,
reading conditioning parameters from CSV files and handling normalization and 
one-hot encoding as specified.

Based on the EncHF_Dataset sidecar format specification.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from dataprep.auxiliary_functions import (
    load_parameter_config,
    find_audio_csv_pairs,
    get_parameter_names
)


def infer_frames_from_ecdc(ecdc_path: Path) -> int:
    """Get the number of frames T from an .ecdc file."""
    ckpt = torch.load(ecdc_path, map_location="cpu",weights_only=False)
    codes = ckpt["audio_codes"]
    
    # Handle different formats - codes might be a list of tensors or a single tensor
    if isinstance(codes, list):
        # If it's a list, take the first tensor
        codes = codes[0] if codes else torch.zeros(1, 1, 1, 0)
    
    # Convert to tensor if needed
    if not isinstance(codes, torch.Tensor):
        codes = torch.tensor(codes)
    
    # Handle different tensor shapes
    if codes.ndim == 4:      # [1,1,Cb,T] or [B,1,Cb,T]
        codes = codes.squeeze(1)  # Remove the second dimension
    if codes.ndim == 2:      # [Cb,T]
        codes = codes[None, ...]  # Add batch dimension
    
    T = int(codes.shape[-1])
    return T


def normalize_parameter_values(values: np.ndarray, param_info: Dict, param_name: str) -> np.ndarray:
    """
    Normalize parameter values to [0, 1] range based on parameter configuration.
    
    Args:
        values: Raw parameter values
        param_info: Parameter configuration from parameters.json
        param_name: Name of the parameter for error reporting
        
    Returns:
        Normalized values in [0, 1] range
    """
    param_type = param_info.get('type', 'continuous')
    
    if param_type == 'continuous':
        min_val = param_info.get('min', 0.0)
        max_val = param_info.get('max', 1.0)
        
        # Clip values to the specified range
        clipped_values = np.clip(values, min_val, max_val)
        
        # Normalize to [0, 1]
        if max_val != min_val:
            normalized = (clipped_values - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(clipped_values)
            
        return normalized.astype(np.float32)
    
    elif param_type == 'class':
        # Handle class parameters - convert class names to indices
        classes = param_info.get('classes', [])
        if classes:
            # Create a mapping from class names to indices
            class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
            num_classes = len(classes)
            
            # Convert class names to indices
            class_indices = []
            warning_count = 0
            for value in values:
                if isinstance(value, str) and value in class_to_idx:
                    class_indices.append(class_to_idx[value])
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    # Handle case where values are already indices
                    idx = int(value)
                    if 0 <= idx < num_classes:
                        class_indices.append(idx)
                    else:
                        # Out of range index
                        warning_count += 1
                        class_indices.append(0)
                else:
                    # Unknown type or value
                    warning_count += 1
                    class_indices.append(0)
            
            if warning_count > 0:
                print(f"   • ⚠️  {warning_count} invalid class values for '{param_name}', defaulted to '{classes[0]}'")
            
            class_values = np.array(class_indices, dtype=int)
        else:
            # Fallback to num_classes if no class names provided
            num_classes = param_info.get('num_classes', 2)
            class_values = values.astype(int)
            # Clip class indices to valid range
            class_values = np.clip(class_values, 0, num_classes - 1)
        
        return class_values
    
    else:
        raise ValueError(f"Unknown parameter type '{param_type}' for parameter '{param_name}'")


def create_one_hot_encoding(class_values: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert class indices to one-hot encoding.
    
    Args:
        class_values: Integer class indices
        num_classes: Total number of classes
        
    Returns:
        One-hot encoded array of shape [T, num_classes]
    """
    T = len(class_values)
    one_hot = np.zeros((T, num_classes), dtype=np.float32)
    
    for i, class_idx in enumerate(class_values):
        if 0 <= class_idx < num_classes:
            one_hot[i, class_idx] = 1.0
    
    return one_hot


def create_sidecar_files(ecdc_path: Path, csv_path: Path, config: Dict, tokens_dir: Path) -> bool:
    """
    Create sidecar files (.cond.npy and .cond.json) for a single .ecdc file.
    
    Args:
        ecdc_path: Path to .ecdc file
        csv_path: Path to corresponding CSV file with annotations
        config: Parameter configuration from parameters.json
        tokens_dir: Root directory containing .ecdc files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get number of frames from .ecdc file
        T_ecdc = infer_frames_from_ecdc(ecdc_path)
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        T_csv = len(df)
        
        # Print file processing header
        file_name = ecdc_path.stem
        print(f"📄 {file_name} sidecar:")
        print(f"    EnCodec frames: {T_ecdc}")
        print(f"    CSV annotations: {T_csv}")
        
        # Handle alignment - use minimum frames and trim both CSV and EnCodec if necessary
        T_aligned = min(T_ecdc, T_csv)
        
        if T_csv > T_ecdc:
            extra_annotations = T_csv - T_ecdc
            print(f"    ⚠️  Trimming {extra_annotations} extra annotations from CSV")
            df = df.iloc[:T_aligned].copy()
        elif T_csv < T_ecdc:
            extra_frames = T_ecdc - T_csv
            print(f"    ⚠️  Trimming {extra_frames} extra frames from EnCodec file")
            
            # Load, trim, and save EnCodec file
            ckpt = torch.load(ecdc_path, map_location="cpu", weights_only=False)
            codes = ckpt["audio_codes"]
            
            # Handle different formats - codes might be a list of tensors or a single tensor
            if isinstance(codes, list):
                # Trim each tensor in the list
                trimmed_codes = []
                for code_tensor in codes:
                    if isinstance(code_tensor, torch.Tensor):
                        # Handle different tensor shapes and trim to T_aligned frames
                        if code_tensor.ndim == 4:      # [1,1,Cb,T]
                            trimmed_codes.append(code_tensor[:, :, :, :T_aligned])
                        elif code_tensor.ndim == 3:    # [1,Cb,T] 
                            trimmed_codes.append(code_tensor[:, :, :T_aligned])
                        elif code_tensor.ndim == 2:    # [Cb,T]
                            trimmed_codes.append(code_tensor[:, :T_aligned])
                        else:
                            trimmed_codes.append(code_tensor)
                    else:
                        trimmed_codes.append(code_tensor)
                ckpt["audio_codes"] = trimmed_codes
            else:
                # Single tensor case
                if isinstance(codes, torch.Tensor):
                    if codes.ndim == 4:      # [1,1,Cb,T]
                        ckpt["audio_codes"] = codes[:, :, :, :T_aligned]
                    elif codes.ndim == 3:    # [1,Cb,T] 
                        ckpt["audio_codes"] = codes[:, :, :T_aligned]
                    elif codes.ndim == 2:    # [Cb,T]
                        ckpt["audio_codes"] = codes[:, :T_aligned]
            
            # Also trim other audio-related tensors if they exist
            if "audio_scales" in ckpt and ckpt["audio_scales"] is not None:
                scales = ckpt["audio_scales"]
                if isinstance(scales, torch.Tensor) and scales.numel() > 0:
                    if scales.ndim >= 1 and scales.shape[-1] == T_ecdc:
                        if scales.ndim == 3:    # [1,1,T]
                            ckpt["audio_scales"] = scales[:, :, :T_aligned]
                        elif scales.ndim == 2:  # [1,T]
                            ckpt["audio_scales"] = scales[:, :T_aligned]
                        elif scales.ndim == 1:  # [T]
                            ckpt["audio_scales"] = scales[:T_aligned]
            
            # Save the trimmed EnCodec file
            torch.save(ckpt, ecdc_path)
            
            # Update T_ecdc to reflect the new length
            T_ecdc = T_aligned
            
        else:
            print(f"    ✅ Perfect alignment!")
        
        # Process each parameter
        feature_columns = []
        feature_names = []
        feature_metadata = {}
        
        for param_key, param_info in config.items():
            param_name = param_info['name']
            param_type = param_info.get('type', 'continuous')
            
            if param_name not in df.columns:
                print(f"    ⚠️ Parameter '{param_name}' not found in CSV, skipping")
                continue
            
            values = df[param_name].to_numpy()
            
            if param_type == 'continuous':
                # Normalize continuous parameters to [0, 1]
                normalized_values = normalize_parameter_values(values, param_info, param_name)
                
                feature_columns.append(normalized_values.reshape(-1, 1))
                feature_names.append(param_name)
                
                # Store metadata
                feature_metadata[param_name] = {
                    "min": float(np.min(normalized_values)),
                    "max": float(np.max(normalized_values)),
                    "mean": float(np.mean(normalized_values)),
                    "std": float(np.std(normalized_values)),
                    "units": param_info.get('unit', ''),  # Fixed: use 'unit' (singular) from parameters.json
                    "doc_string": f"Normalized {param_name} parameter"
                }
                
            elif param_type == 'class':
                # Convert class indices to one-hot encoding
                classes = param_info.get('classes', [])
                num_classes = len(classes) if classes else param_info.get('num_classes', 2)
                class_values = normalize_parameter_values(values, param_info, param_name)
                one_hot = create_one_hot_encoding(class_values, num_classes)
                
                feature_columns.append(one_hot)
                
                # Create feature names using actual class names if available
                if classes:
                    class_names = [f"{param_name}_{class_name}" for class_name in classes]
                else:
                    class_names = [f"{param_name}_{i}" for i in range(num_classes)]
                
                feature_names.extend(class_names)
                
                # Store metadata for each class dimension
                for i, class_name in enumerate(class_names):
                    class_col = one_hot[:, i]
                    # Extract the actual class name from the feature name (e.g., "instrument_piano" -> "piano")
                    actual_class_name = class_name.replace(f"{param_name}_", "") if "_" in class_name else str(i)
                    feature_metadata[class_name] = {
                        "min": float(np.min(class_col)),
                        "max": float(np.max(class_col)),
                        "mean": float(np.mean(class_col)),
                        "std": float(np.std(class_col)),
                        "units": "probability",
                        "doc_string": f"One-hot encoding for {param_name} class {actual_class_name}"
                    }
        
        # Concatenate all features
        if feature_columns:
            conditioning_matrix = np.concatenate(feature_columns, axis=1)
        else:
            conditioning_matrix = np.zeros((T_aligned, 0), dtype=np.float32)
        
        # Ensure correct shape and dtype
        conditioning_matrix = conditioning_matrix.astype(np.float32)
        
        # Create output paths
        base_name = ecdc_path.stem
        cond_npy_path = ecdc_path.with_suffix('.cond.npy')
        cond_json_path = ecdc_path.with_suffix('.cond.json')
        
        # Create metadata JSON
        metadata = {
            "schema_version": 1,
            "fps": 75,
            "source_rate": 75,
            "names": feature_names,
            "features": feature_metadata,
            "norm": {
                "min": [feature_metadata[name]["min"] for name in feature_names],
                "max": [feature_metadata[name]["max"] for name in feature_names],
                "mean": [feature_metadata[name]["mean"] for name in feature_names],
                "std": [feature_metadata[name]["std"] for name in feature_names]
            }
        }
        
        # Save files atomically
        # Save .npy file
        temp_npy = cond_npy_path.with_suffix('.tmp.npy')
        np.save(temp_npy, conditioning_matrix)
        temp_npy.rename(cond_npy_path)
        
        # Save .json file (COMMENTED OUT - metadata now saved centrally in HF dataset)
        # This metadata (feature_names, feature_metadata) is still needed in step 4
        # temp_json = cond_json_path.with_suffix('.tmp.json')
        # with open(temp_json, 'w') as f:
        #     json.dump(metadata, f, indent=2)
        # temp_json.rename(cond_json_path)
        
        print(f"    ✅ Sidecar created ({T_aligned} frames, {len(feature_names)} features)\n")
        return True
        
    except Exception as e:
        file_name = ecdc_path.stem
        print(f"\n📄 {file_name} sidecar:")
        print(f"   • ❌ Error: {e}")
        return False


def create_sidecars_dataset(tokens_dir: Path, raw_dir: Path) -> Dict:
    """
    Create sidecar files for all .ecdc files in the tokens directory.
    
    Args:
        tokens_dir: Directory containing .ecdc files
        raw_dir: Directory containing raw CSV files and parameters.json
        
    Returns:
        Dictionary with processing statistics
    """
    print(f"\n\033[1mSIDECAR CREATION:\033[0m\n")

    tokens_dir = Path(tokens_dir)
    raw_dir = Path(raw_dir)
    
    # Load parameter configuration
    config_path = raw_dir / 'parameters.json'
    if not config_path.exists():
        raise FileNotFoundError(f"parameters.json not found in {raw_dir}")
    
    config = load_parameter_config(config_path)
    
    # Find all .ecdc files
    ecdc_files = list(tokens_dir.rglob('*.ecdc'))
    if not ecdc_files:
        raise ValueError(f"No .ecdc files found in {tokens_dir}")
    
    # Find corresponding CSV files in raw directory
    ecdc_csv_pairs = []
    for ecdc_path in ecdc_files:
        # Get relative path and construct CSV path
        rel_path = ecdc_path.relative_to(tokens_dir)
        csv_name = rel_path.stem + '.csv'
        csv_path = raw_dir / rel_path.with_suffix('.csv')  # Preserves structure: "raw/train/piano.csv" ✅        
        
        if csv_path.exists():
            ecdc_csv_pairs.append((ecdc_path, csv_path))
        else:
            print(f"⚠️ No CSV found for {ecdc_path.name}, skipping")
    
    if not ecdc_csv_pairs:
        raise ValueError(f"No matching CSV files found for .ecdc files")
    
    # print("=" * 70)
    # print("🔧 CREATING SIDECAR FILES")
    # print("=" * 70)
    print(f"📁 Tokens directory: {tokens_dir}")
    print(f"📁 Raw directory: {raw_dir}")
    print(f"📊 Found {len(ecdc_csv_pairs)} .ecdc-CSV pairs")
    print(f"🎛️ Parameters: {', '.join([info['name'] for info in config.values()])}")
    # print("=" * 70)
    
    # Process each pair
    success_count = 0
    
    print(f"\n\033[1mSIDECARS SUMMARY:\033[0m\n")

    for ecdc_path, csv_path in ecdc_csv_pairs:
        if create_sidecar_files(ecdc_path, csv_path, config, tokens_dir):
            success_count += 1
    
    # print("=" * 70)
    print(f"✅ Successfully created {success_count}/{len(ecdc_csv_pairs)} sidecar pairs")
    
    if success_count < len(ecdc_csv_pairs):
        failed_count = len(ecdc_csv_pairs) - success_count
        print(f"❌ Failed to create {failed_count} sidecar pairs")
    
    return {
        'total': len(ecdc_csv_pairs),
        'success': success_count,
        'failed': len(ecdc_csv_pairs) - success_count
    }


def validate_sidecars(tokens_dir: Path) -> Dict:
    """
    Validate that all .ecdc files have properly aligned sidecar files.
    
    Args:
        tokens_dir: Directory containing .ecdc and sidecar files
        
    Returns:
        Dictionary with validation results
    """
    tokens_dir = Path(tokens_dir)
    
    ecdc_files = list(tokens_dir.rglob('*.ecdc'))
    validation_results = {
        'total_ecdc': len(ecdc_files),
        'valid_sidecars': 0,
        'missing_sidecars': 0,
        'alignment_errors': 0,
        'issues': []
    }
    
    print("\n🔍 VALIDATING SIDECAR ALIGNMENT")
    print("=" * 50)
    
    for ecdc_path in ecdc_files:
        cond_npy_path = ecdc_path.with_suffix('.cond.npy')
        
        # Check if sidecar .npy file exists (JSON no longer created per file)
        if not cond_npy_path.exists():
            validation_results['missing_sidecars'] += 1
            validation_results['issues'].append(f"Missing sidecar .npy file for {ecdc_path.name}")
            continue
        
        try:
            # Check frame alignment
            T_ecdc = infer_frames_from_ecdc(ecdc_path)
            cond_array = np.load(cond_npy_path)
            
            if cond_array.shape[0] != T_ecdc:
                validation_results['alignment_errors'] += 1
                validation_results['issues'].append(
                    f"Frame mismatch in {ecdc_path.name}: "
                    f"ecdc={T_ecdc}, sidecar={cond_array.shape[0]}"
                )
                continue
            
            validation_results['valid_sidecars'] += 1
            
        except Exception as e:
            validation_results['alignment_errors'] += 1
            validation_results['issues'].append(f"Error validating {ecdc_path.name}: {e}")
    
    # # Print validation summary
    # print(f"✅ Valid sidecars: {validation_results['valid_sidecars']}")
    # print(f"❌ Missing sidecars: {validation_results['missing_sidecars']}")
    # print(f"⚠️ Alignment errors: {validation_results['alignment_errors']}")
    
    # if validation_results['issues']:
    #     print("\n⚠️ Issues found:")
    #     for issue in validation_results['issues'][:10]:  # Show first 10 issues
    #         print(f"  - {issue}")
    #     if len(validation_results['issues']) > 10:
    #         print(f"  ... and {len(validation_results['issues']) - 10} more issues")
    
    return validation_results

# Convenience functions for notebook use
def quick_create_sidecars(dataset_dir: str) -> Dict:
    """
    Quick sidecar creation for notebook - just call this function!
    
    Args:
        dataset_dir: Root dataset directory (e.g., './data/dataset_01')
        
    Returns:
        Dictionary with processing statistics
    
    Example:
        results = quick_create_sidecars('./data/dataset_01')
    """
    dataset_path = Path(dataset_dir)
    tokens_dir = dataset_path / 'tokens'
    raw_dir = dataset_path / 'raw'
    
    # Create sidecar files
    results = create_sidecars_dataset(tokens_dir, raw_dir)
    
    # # Validate the created sidecars
    # validation = validate_sidecars(tokens_dir)
    # results['validation'] = validation
    
    # return results
    return