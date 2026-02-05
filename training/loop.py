"""
Training Loop Module for RNeNcodec

This module provides high-level functions for training RNN models on EnCodec latent datasets.
It automatically reads dataset configuration (conditioning_config.json) and handles the complete
training pipeline with minimal boilerplate.

Main Functions:
    - load_dataset_config(): Load conditioning parameters from dataset
    - create_dataloaders(): Create train/validation data loaders
    - create_model(): Initialize the RNN model with proper configuration
    - train_model(): Complete training loop with checkpointing and logging
"""

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import EncodecModel

from rnencodec.audioDataLoader.audio_dataset import (
    LatentDatasetConfig,
    EnCodecLatentDataset_dynamic
)
from rnencodec.model.gru_audio_model import GRUModelConfig, RNN
from rnencodec.utils.io import save_run_config


import sys
import os

_PRINT_STATE = {"on": True, "stdout": None, "stderr": None,}

# ⚠️ [!]
# ✅ [OK}

import platform 
# Detect OS and set num_workers 
num_workers = 0 if platform.system() in ("Windows", "Darwin") else 4 #windows and mac 

def print_switch(on: bool | None = None):
    global _PRINT_STATE
    if on is None:
        on = not _PRINT_STATE["on"]
    if on and not _PRINT_STATE["on"]:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = _PRINT_STATE["stdout"]
        sys.stderr = _PRINT_STATE["stderr"]
        _PRINT_STATE["on"] = True
    elif not on and _PRINT_STATE["on"]:
        _PRINT_STATE["stdout"] = sys.stdout
        _PRINT_STATE["stderr"] = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        _PRINT_STATE["on"] = False

# ============================================================================
# Dataset Configuration
# ============================================================================

def load_dataset_config(dataset_path: str) -> Dict:
    """
    Load conditioning configuration from dataset's conditioning_config.json.
    
    Args:
        dataset_path: Path to dataset root (containing raw/ and hf_dataset/)
        
    Returns:
        Dictionary with conditioning configuration including feature names,
        number of features, and parameter specifications
        
    Example:
        config = load_dataset_config("../../datasets/dataset_01")
        # Returns: {"feature_names": ["tempo", "reverb", ...], "num_features": 4, ...}
    """
    dataset_path = Path(dataset_path)
    config_file = dataset_path / 'hf_dataset' / 'conditioning_config.json'
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"conditioning_config.json not found at {config_file}.\n"
            f"Make sure you've run the dataset creation pipeline (step 4) first."
        )
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"📋 Conditioning configuration:")
    print(f"    Features: {', '.join(config['feature_names'])}")
    print(f"    Total features: {config['num_features']}")
    print(f"    FPS: {config['fps']}")
    
    return config


def get_available_splits(dataset_path: str) -> List[str]:
    """
    Detect available data splits in the HuggingFace dataset.
    
    Args:
        dataset_path: Path to dataset root
        
    Returns:
        List of available split names (e.g., ['train', 'validation', 'test'])
    """
    from datasets import load_from_disk
    
    dataset_path = Path(dataset_path)
    hf_dataset_path = dataset_path / 'hf_dataset'
    
    if not hf_dataset_path.exists():
        raise FileNotFoundError(f"HuggingFace dataset not found at {hf_dataset_path}")
    
    dataset_dict = load_from_disk(str(hf_dataset_path))
    splits = list(dataset_dict.keys())
    
    print(f"📊 Available splits: {', '.join(splits)}")
    return splits


# ============================================================================
# Data Loaders
# ============================================================================

def create_dataloaders(
    dataset_path: str,
    conditioning_config: Dict,
    sequence_length: int = 125,
    batch_size: int = 100,
    train_splits: Optional[Union[str, List[str]]] = None,
    val_splits: Optional[Union[str, List[str]]] = None,
    num_workers: int = num_workers,
    add_noise: bool = True,
    noise_weight: float = 0.05,
    files_per_sequence: int = 4,
    codebook_size: int = 1024,
    clamp_val: int = 15
) -> Tuple[DataLoader, Optional[DataLoader], object, int]:
    """
    Create training and validation data loaders.
    Auto-detects n_q (number of codebooks) from the dataset.
    Supports multiple splits for both training and validation.
    
    Args:
        dataset_path: Path to dataset root
        conditioning_config: Conditioning configuration from load_dataset_config()
        sequence_length: Length of training sequences (frames)
        batch_size: Batch size for training
        train_splits: List of split names to use for training (e.g., ['train'] or ['train', 'test'])
                     Can also be a single string. None defaults to ['train'].
        val_splits: List of split names to use for validation (e.g., ['validation'])
                   Can also be a single string. None means no validation.
        num_workers: Number of data loading workers
        add_noise: Whether to add noise during training
        noise_weight: Noise weight for augmentation
        files_per_sequence: Number of files to sample per sequence
        codebook_size: EnCodec codebook size
        clamp_val: Clamping value for latents
        
    Returns:
        Tuple of (train_loader, val_loader, enc_model, n_q). 
        val_loader is None if val_splits is None. n_q is detected from data.
    """
    dataset_path = Path(dataset_path)
    hf_dataset_path = str(dataset_path / 'hf_dataset')
    
    # Normalize split inputs to lists
    if train_splits is None:
        train_splits = ['train']
    elif isinstance(train_splits, str):
        train_splits = [train_splits]
    
    if val_splits is not None and isinstance(val_splits, str):
        val_splits = [val_splits]
    
    # Create parameter specs dictionary from conditioning config
    props = {name: None for name in conditioning_config['feature_names']}
    
    # Load EnCodec model
    enc_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    enc_model.eval()
    
    # Verify that requested splits exist
    from datasets import load_from_disk
    dataset_dict = load_from_disk(hf_dataset_path)
    available_splits = list(dataset_dict.keys())
    
    # Check training splits
    for split in train_splits:
        if split not in available_splits:
            raise ValueError(f"Training split '{split}' not found. Available splits: {available_splits}")
    
    # Check validation splits
    if val_splits:
        for split in val_splits:
            if split not in available_splits:
                raise ValueError(f"Validation split '{split}' not found. Available splits: {available_splits}")
    
    # Create a temporary dataset to detect n_q from the first sample
    # Use first training split for detection
    temp_config = LatentDatasetConfig(
        dataset_path=hf_dataset_path,
        sequence_length=sequence_length,
        parameter_specs=props,
        add_noise=False,
        noise_weight=0.0,
        codebook_size=codebook_size,
        n_q=8,  # Temporary placeholder
        clamp_val=clamp_val,
        filters={},
        files_per_sequence=1,
        cond_root=None,
        cond_suffix=".cond.npy",
        strict=False
    )
    temp_dataset = EnCodecLatentDataset_dynamic(temp_config, "facebook/encodec_24khz", split=train_splits[0])
    
    # Get first sample to detect n_q
    try:
        sample_inp, sample_target = temp_dataset[0]
        # sample_target shape is [T, n_q] where T is sequence length
        # print("sample_target_shape:", sample_target.shape)
        n_q = sample_target.shape[1]
        # print(f"   • Detected n_q = {n_q} codebooks")
    except Exception as e:
        print(f"    [!] Could not auto-detect n_q: {e}. Using default n_q = 8")
        n_q = 8
    
    # Create training datasets from all requested splits and concatenate them
    train_datasets = []
    for split in train_splits:
        train_config = LatentDatasetConfig(
            dataset_path=hf_dataset_path,
            sequence_length=sequence_length,
            parameter_specs=props,
            add_noise=add_noise,
            noise_weight=noise_weight,
            codebook_size=codebook_size,
            n_q=n_q,
            clamp_val=clamp_val,
            filters={},
            files_per_sequence=files_per_sequence,
            cond_root=None,  # Co-located sidecars
            cond_suffix=".cond.npy",
            strict=False
        )
        dataset = EnCodecLatentDataset_dynamic(train_config, "facebook/encodec_24khz", split=split)
        train_datasets.append(dataset)
    
    # Concatenate all training datasets
    from torch.utils.data import ConcatDataset
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        train_dataset = ConcatDataset(train_datasets)
    
    # Create training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    print(f"[OK!] Training dataset loaded:")
    print(f"    Splits: {', '.join(train_splits)}")
    print(f"    Size: {len(train_dataset)} sequences")
    print(f"    Batch size: {batch_size}")
    
    # Create validation dataset and loader if requested
    val_loader = None
    if val_splits:
        # Create validation datasets from all requested splits and concatenate them
        val_datasets = []
        for split in val_splits:
            val_config = LatentDatasetConfig(
                dataset_path=hf_dataset_path,
                sequence_length=sequence_length,
                parameter_specs=props,
                add_noise=False,  # No noise for validation
                noise_weight=0.0,
                codebook_size=codebook_size,
                n_q=n_q,
                clamp_val=clamp_val,
                filters={},
                files_per_sequence=files_per_sequence,
                cond_root=None,
                cond_suffix=".cond.npy",
                strict=False
            )
            dataset = EnCodecLatentDataset_dynamic(val_config, "facebook/encodec_24khz", split=split)
            val_datasets.append(dataset)
        
        # Concatenate all validation datasets
        if len(val_datasets) == 1:
            val_dataset = val_datasets[0]
        else:
            val_dataset = ConcatDataset(val_datasets)
        
        # Create validation loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation
            num_workers=num_workers,
            drop_last=False
        )
        
        print(f"[OK!] Validation dataset loaded:")
        print(f"    Splits: {', '.join(val_splits)}")
        print(f"    Size: {len(val_dataset)} sequences")
        print(f"    Batch size: {batch_size}")
    
    return train_loader, val_loader, enc_model, n_q


# ============================================================================
# Model Creation
# ============================================================================

def create_model(
    conditioning_config: Dict,
    n_q: int,
    hidden_size: int = 128,
    num_layers: int = 3,
    input_size: int = 128,
    codebook_size: int = 1024,
    dropout: float = 0.1,
    cascade_mode: str = "soft",
    temperature: float = 1.0,
    top_n: Optional[int] = None,
    tau_soft: float = 0.6,
    device: Optional[torch.device] = None
) -> Tuple[RNN, GRUModelConfig]:
    """
    Create and initialize the RNN model.
    
    Args:
        conditioning_config: Conditioning configuration from load_dataset_config()
        n_q: Number of codebooks (auto-detected from dataset)
        hidden_size: Size of hidden state
        num_layers: Number of GRU layers
        input_size: Size of input latent vectors
        codebook_size: EnCodec codebook size
        dropout: Dropout probability
        cascade_mode: "soft" (deterministic) or "hard" (sampling)
        temperature: Temperature for hard cascade sampling
        top_n: Top-k restriction for sampling (None = no restriction)
        tau_soft: Temperature for soft cascade mode
        device: Device to place model on (None for auto-detect)
        
    Returns:
        Tuple of (model, model_config)
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create model configuration with cascade parameters
    model_config = GRUModelConfig(
        input_size=input_size,
        cond_size=conditioning_config['num_features'],
        hidden_size=hidden_size,
        num_layers=num_layers,
        codebook_size=codebook_size,
        dropout=dropout,
        n_q=n_q,
        # Cascade configuration
        cascade=cascade_mode,
        temperature_hard=temperature,
        top_n_hard=top_n,
        tau_soft=tau_soft
    )
    
    # Load EnCodec for model initialization
    enc_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    enc_model.eval()
    
    # Create model
    model = RNN(model_config, enc_model).to(device)
    
    print(f"🧠 Model created:")
    print(f"    Device: {device}")
    print(f"    Conditioning features: {conditioning_config['num_features']}")
    print(f"    Hidden size: {hidden_size}")
    print(f"    Layers: {num_layers}")
    print(f"    Codebooks: {n_q}")
    print(f"    Cascade mode: {cascade_mode}")
    if cascade_mode == "hard":
        print(f"    Temperature: {temperature}")
        print(f"    Top-k: {top_n if top_n else 'None (full distribution)'}")
    else:
        print(f"    Tau (soft): {tau_soft}")
    
    return model, model_config


# ============================================================================
# Training Utilities
# ============================================================================

def prepare_target_codebook_latents(rnn_model, target_codes, scales_bq=None):
    """
    Prepare target latents for teacher forcing.
    
    Args:
        rnn_model: The RNN model
        target_codes: Target codes (B, n_q)
        scales_bq: Optional scales
        
    Returns:
        List of per-codebook latents
    """
    dev = rnn_model._E_eff.device
    codes_bq = target_codes.to(dev, dtype=torch.long, non_blocking=True)
    if scales_bq is not None:
        scales_bq = scales_bq.to(dev, non_blocking=True)

    out = []
    for q in range(rnn_model.n_q):
        E_q = rnn_model._E_eff[q]
        e_q = F.embedding(codes_bq[:, q], E_q)
        if scales_bq is not None:
            e_q = e_q * scales_bq[:, q].unsqueeze(-1)
        out.append(e_q)
    return out


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    params: Dict,
    epoch: int,
    use_tqdm: bool = True
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: The RNN model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss criterion
        device: Device to train on
        params: Training parameters dictionary
        epoch: Current epoch number
        use_tqdm: Whether to show progress bar
        
    Returns:
        Dictionary with loss statistics
    """
    model.train()
    
    # Extract parameters
    batch_size = params['batch_size']
    seq_len = params['sequence_length']
    n_q = params['n_q']
    batches_per_epoch = params.get('batches_per_epoch', len(train_loader))
    
    # Teacher forcing schedule
    TF_cycle = params['TF_schedule'][0] + params['TF_schedule'][1]
    use_tf = (epoch % TF_cycle) < params['TF_schedule'][0]
    
    # Quantizer weights
    raw_weights = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], dtype=torch.float)
    quantizer_weights = (raw_weights * (len(raw_weights) / raw_weights.sum()))[:n_q].to(device)
    
    # Tracking
    epoch_loss = 0.0
    epoch_quantizer_losses = [0.0] * n_q
    num_batches = 0
    
    # Setup iterator with optional progress bar
    total_batches = min(batches_per_epoch, len(train_loader))
    iterator = iter(train_loader)
    
    if use_tqdm:
        from tqdm import tqdm
        pbar = tqdm(range(total_batches), desc=f"Epoch {epoch+1}", leave=False)
    else:
        pbar = range(total_batches)
    
    for batch_num in pbar:
        if batch_num >= batches_per_epoch:
            break
        
        try:
            inp, target = next(iterator)
        except StopIteration:
            break
            
        inp, target = inp.to(device), target.to(device)
        hidden = model.init_hidden(batch_size)
        optimizer.zero_grad()
        
        batch_loss = 0
        batch_quantizer_losses = [0.0] * n_q
        
        # Process sequence
        for i in range(seq_len):
            # Teacher forcing
            if params.get('simulate_parallel', False):
                use_teacher_forcing = True
                tflatents = [torch.zeros(batch_size, params['input_size'], device=device) for _ in range(n_q)]
            else:
                use_teacher_forcing = use_tf
                if use_teacher_forcing:
                    tflatents = prepare_target_codebook_latents(model, target[:, i, :])
                else:
                    tflatents = None
            
            # Forward pass
            logits_list, hidden, sampled_indices, step_latent = model(
                inp[:, i, :],
                hidden,
                target_codebook_latents=tflatents,
                use_teacher_forcing=use_teacher_forcing,
                return_step_latent=False
            )
            
            # Compute loss per quantizer
            for j in range(n_q):
                quantizer_loss = criterion(logits_list[j], target[:, i, j])
                batch_quantizer_losses[j] += quantizer_loss.item()
                batch_loss = batch_loss + quantizer_weights[j] * quantizer_loss
        
        # Average over sequence
        batch_loss = batch_loss / seq_len
        batch_quantizer_losses = [ql / seq_len for ql in batch_quantizer_losses]
        
        # Backward pass
        batch_loss.backward()
        optimizer.step()
        
        # Accumulate
        epoch_loss += batch_loss.item()
        for j in range(n_q):
            epoch_quantizer_losses[j] += batch_quantizer_losses[j]
        num_batches += 1
        
        # Update progress bar
        if use_tqdm:
            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}', 'tf': use_tf})
    
    # Average over batches
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    avg_quantizer_losses = [ql / num_batches for ql in epoch_quantizer_losses]
    
    return {
        'loss': avg_loss,
        'quantizer_losses': avg_quantizer_losses,
        'teacher_forcing': use_tf
    }


def validate_epoch(
    model: RNN,
    val_loader: DataLoader,
    quantizer_weights: Optional[List[float]],
    n_q: int,
    batches_per_epoch: Optional[int] = None,
    device: Optional[torch.device] = None,
    use_tqdm: bool = True
) -> Dict:
    """
    Run one validation epoch (no gradient computation).
    
    Args:
        model: The RNN model
        val_loader: Validation data loader
        quantizer_weights: Per-codebook loss weights (None for equal weighting)
        n_q: Number of codebooks
        batches_per_epoch: Optional limit on number of batches (None = full epoch)
        device: Device to run on
        use_tqdm: Whether to show progress bar
        
    Returns:
        Dictionary with validation metrics
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.eval()  # Set to evaluation mode
    
    # Initialize metrics
    epoch_loss = 0.0
    epoch_quantizer_losses = [0.0] * n_q
    num_batches = 0
    
    # Use same loss criterion as training
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    # Setup iterator
    iterator = iter(val_loader)
    total_batches = batches_per_epoch if batches_per_epoch else len(val_loader)
    
    if use_tqdm:
        from tqdm import tqdm
        pbar = tqdm(range(total_batches), desc="Validation", leave=False)
    else:
        pbar = range(total_batches)
    
    with torch.no_grad():  # No gradients during validation
        for batch_idx in pbar:
            try:
                inp, target = next(iterator)
            except StopIteration:
                break
            
            # Move to device
            inp, target = inp.to(device), target.to(device)
            B, T, n_q_actual = target.shape
            
            # Initialize hidden state
            hidden = model.init_hidden(B)
            
            # Forward pass through sequence
            batch_quantizer_losses = [0.0] * n_q
            for i in range(T):
                # Always use teacher forcing for validation
                tflatents = prepare_target_codebook_latents(model, target[:, i, :])
                
                # Forward pass
                logits_list, hidden, sampled_indices, step_latent = model(
                    inp[:, i, :],
                    hidden,
                    target_codebook_latents=tflatents,
                    use_teacher_forcing=True,
                    return_step_latent=False
                )
                
                # Compute loss per quantizer
                for j in range(n_q):
                    quantizer_loss = criterion(logits_list[j], target[:, i, j])
                    batch_quantizer_losses[j] += quantizer_loss.item()
            
            # Average over time and apply quantizer weights
            batch_quantizer_losses = [ql / T for ql in batch_quantizer_losses]
            
            if quantizer_weights is not None:
                batch_loss = sum(w * ql for w, ql in zip(quantizer_weights, batch_quantizer_losses))
            else:
                batch_loss = sum(batch_quantizer_losses) / n_q
            
            # Accumulate
            epoch_loss += batch_loss
            for j in range(n_q):
                epoch_quantizer_losses[j] += batch_quantizer_losses[j]
            num_batches += 1
            
            # Update progress bar
            if use_tqdm:
                pbar.set_postfix({'val_loss': f'{batch_loss:.4f}'})
    
    # Average over batches
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    avg_quantizer_losses = [ql / num_batches for ql in epoch_quantizer_losses]
    
    return {
        'loss': avg_loss,
        'quantizer_losses': avg_quantizer_losses
    }


# ============================================================================
# Main Training Function
# ============================================================================

def train_model(
    dataset_path: str,
    model_output_path: str,
    num_epochs: int = 75,
    batch_size: int = 100,
    sequence_length: int = 125,
    batches_per_epoch: int = 100,
    learning_rate: float = 0.005,
    hidden_size: int = 128,
    num_layers: int = 3,
    cascade_mode: str = "soft",
    temperature: float = 1.0,
    top_n: Optional[int] = None,
    tau_soft: float = 0.6,
    save_dir: Optional[str] = None,
    save_interval: int = 25,
    resume_checkpoint: Optional[str] = None,
    device: Optional[torch.device] = None,
    # New parameters
    train_splits: Optional[Union[str, List[str]]] = None,
    val_splits: Optional[Union[str, List[str]]] = None,
    TF_schedule: Optional[List[int]] = None,
    quantizer_weights: Optional[List[float]] = None,
    simulate_parallel: bool = False,
    use_tensorboard: bool = True,
    use_tqdm: bool = True,
    val_interval: int = 1,
    **kwargs
) -> Dict:
    """
    Complete training pipeline for RNeNcodec model.
    Automatically detects and uses available splits from the dataset.
    Automatically detects n_q (number of codebooks) from the data.
    
    Args:
        dataset_path: Path to dataset root (containing raw/ and hf_dataset/)
        model_output_path: Full path where the model will be saved
        num_epochs: Number of training epochs
        batch_size: Batch size
        sequence_length: Sequence length (frames)
        batches_per_epoch: Number of batches per epoch
        learning_rate: Learning rate
        hidden_size: Hidden layer size
        num_layers: Number of GRU layers
        cascade_mode: "soft" (deterministic, RNG-free) or "hard" (sampling-based)
        temperature: Temperature for hard cascade sampling (only used if cascade_mode="hard")
        top_n: Top-k restriction for hard sampling (None = no restriction)
        tau_soft: Temperature for soft cascade mode (only used if cascade_mode="soft")
        save_dir: DEPRECATED - use model_output_path instead
        save_interval: Save checkpoint every N epochs
        resume_checkpoint: Path to checkpoint to resume from
        device: Device to train on (None for auto)
        train_splits: Training split(s) - single string or list of strings (default: auto-detect)
        val_splits: Validation split(s) - single string or list of strings (default: auto-detect)
        TF_schedule: Teacher forcing schedule [epochs_on, epochs_off] (default: [25, 25])
        quantizer_weights: Per-codebook loss weights (default: [3.0, 2.0, 1.5, 1.0, ...])
        simulate_parallel: If True, always use teacher forcing (parallel training)
        use_tensorboard: If True, log to TensorBoard
        use_tqdm: If True, show progress bars
        val_interval: Run validation every N epochs (default: 1 = every epoch)
        **kwargs: Additional parameters (noise_weight, dropout, add_noise, etc.)
        
    Returns:
        Dictionary with training statistics and paths
        
    Example:
        # Simple usage with soft cascade (deterministic)
        results = train_model(
            dataset_path="../../datasets/dataset_01",
            model_output_path="../models/my_model_v1",
            num_epochs=75,
            batch_size=100,
            cascade_mode="soft"
        )
        
        # Hard cascade with sampling
        results = train_model(
            dataset_path="../../datasets/dataset_01",
            model_output_path="../models/my_model_v2",
            num_epochs=75,
            batch_size=100,
            cascade_mode="hard",
            temperature=0.8,
            top_n=10
        )
    """
    from pathlib import Path

    p = Path(model_output_path).resolve()
    save_dir   = p.parent
    model_name = p.name

    print("\n\033[1m🚀 Training Configurations\033[0m\n")
    
    # Setup device
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    # Load dataset configuration
    # print("📋 Loading dataset configuration...")
    conditioning_config = load_dataset_config(dataset_path)
    # print()
    
    # Auto-detect available splits in the dataset
    # print("🔍 Detecting available splits...")
    available_splits = get_available_splits(dataset_path)
    
    # Determine which splits to use (allow user override)
    if train_splits is None:
        train_splits = 'train' if 'train' in available_splits else available_splits[0]
    
    if val_splits is None:
        # Use validation or test split if available (prefer validation)
        if 'validation' in available_splits:
            val_splits = 'validation'
        elif 'test' in available_splits:
            val_splits = 'test'
        else:
            val_splits = None
        val_split = 'test'
    
    # print(f"   • Using train split: '{train_split}'")
    # if val_split:
    #     print(f"   • Using validation split: '{val_split}'")
    # else:
    #     print(f"   • No validation split (generative mode)")
    # print()
    
    # Create data loaders (also detects n_q from dataset)
    print("📦 Creating data loaders...")
    print_switch(False)
    train_loader, val_loader, enc_model, n_q = create_dataloaders(
        dataset_path=dataset_path,
        conditioning_config=conditioning_config,
        sequence_length=sequence_length,
        batch_size=batch_size,
        train_splits=train_splits,
        val_splits=val_splits,
        add_noise=kwargs.get('add_noise', True),
        noise_weight=kwargs.get('noise_weight', 0.05),
        files_per_sequence=kwargs.get('files_per_sequence', 4)
    )
    print_switch(True)
    
    # Create model
    # print("🏗️ Creating model...")
    model, model_config = create_model(
        conditioning_config=conditioning_config,
        n_q=n_q,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=kwargs.get('dropout', 0.1),
        cascade_mode=cascade_mode,
        temperature=temperature,
        top_n=top_n,
        tau_soft=tau_soft,
        device=device
    )
    enc_model.to(device)
    # print()
    
    # Setup optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    # Setup output directory
    if save_dir is None:
        save_dir = "./models"
    
    if model_name is None:
        model_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    out_dir = Path(save_dir) / model_name
    
    start_epoch = 0
    if resume_checkpoint:
        out_dir = Path(resume_checkpoint)
        checkpoint_path = out_dir / "checkpoints" / "last_checkpoint.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"📂 Resumed from checkpoint at epoch {start_epoch}\n")
        else:
            print(f"[!]  Checkpoint not found at {checkpoint_path}, starting fresh\n")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(exist_ok=True)
        (out_dir / "tensorboard").mkdir(exist_ok=True)
    
    # Copy conditioning_config.json to model directory (human-readable reference)
    conditioning_config_source = Path(dataset_path) / 'hf_dataset' / 'conditioning_config.json'
    if conditioning_config_source.exists():
        conditioning_config_dest = out_dir / 'conditioning_config.json'
        shutil.copy2(conditioning_config_source, conditioning_config_dest)
        print(f"📋 Copied conditioning_config.json to model directory")
    else:
        print(f"[!]  Warning: conditioning_config.json not found at {conditioning_config_source}")
    
    # Save standard configuration
    data_config = train_loader.dataset.config
    save_run_config(
        str(out_dir / "config_v2.pt"), 
        params=None, 
        model_config=model_config, 
        data_config=data_config
    )
    
    # Also save conditioning_config separately in a .pt file for programmatic access
    if conditioning_config_source.exists():
        conditioning_config_pt = out_dir / 'conditioning_config.pt'
        torch.save(conditioning_config, conditioning_config_pt)
        print(f"💾 Saved conditioning config to: conditioning_config.pt")
    
    print(f"💾 Output directory: {out_dir}")
    print(f"💾 Saved configuration to: {out_dir / 'config_v2.pt'}")
    
    # Setup tensorboard (optional)
    writer = None
    if use_tensorboard:
        writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
    
    # Use provided TF_schedule or default
    if TF_schedule is None:
        TF_schedule = [25, 25]
    
    # Use provided quantizer_weights or default
    if quantizer_weights is None:
        quantizer_weights = [3.0, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4]
    
    # Training parameters
    params = {
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'n_q': n_q,
        'batches_per_epoch': batches_per_epoch,
        'input_size': 128,
        'TF_schedule': TF_schedule,
        'simulate_parallel': simulate_parallel,
        'quantizer_weights': quantizer_weights
    }
    
    # Training loop
    # print("="*70)
    print("\n\033[1m🏋️ Training Started\033[0m\n")
    # print("="*70)
    print(f"Epochs: {start_epoch + 1} → {start_epoch + num_epochs}")
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Batch size: {batch_size}")
    print(f"TF schedule: {TF_schedule}")
    print(f"Simulate parallel: {simulate_parallel}")
    if val_loader:
        print(f"Validation: Enabled ({len(val_loader)} batches, every {val_interval} epoch(s))")
    else:
        print(f"Validation: Disabled")
    
    # TensorBoard instructions
    if use_tensorboard:
        # print(f"    Log directory: {out_dir / 'tensorboard'}")
        print(f"Tensorboard: Enabled")
        print(f"    run in a new terminal: \033[1mtensorboard --logdir={out_dir / 'tensorboard'}\033[0m")
    
    print("="*70)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Train for one epoch
        stats = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            params=params,
            epoch=epoch,
            use_tqdm=use_tqdm
        )
        
        # Validation (if validation data available and at the right interval)
        val_stats = None
        if val_loader is not None and (epoch + 1) % val_interval == 0:
            val_stats = validate_epoch(
                model=model,
                val_loader=val_loader,
                quantizer_weights=quantizer_weights,
                n_q=n_q,
                batches_per_epoch=batches_per_epoch,
                device=device,
                use_tqdm=use_tqdm
            )
        
        # Print progress
        tf_status = "TF=ON" if stats['teacher_forcing'] else "TF=OFF"
        if val_stats:
            print(f"Epoch {epoch+1:3d}/{start_epoch + num_epochs} | {tf_status} | Training loss: {stats['loss']:.4f} | Validation loss: {val_stats['loss']:.4f}")
        else:
            print(f"Epoch {epoch+1:3d}/{start_epoch + num_epochs} | {tf_status} | Training loss: {stats['loss']:.4f}")
        
        # Log to tensorboard
        if writer:
            writer.add_scalar("Loss/train", stats['loss'], epoch + 1)
            if val_stats:
                writer.add_scalar("Loss/val", val_stats['loss'], epoch + 1)
            for j, ql in enumerate(stats['quantizer_losses']):
                writer.add_scalar(f"Loss/train_quantizer_{j}", ql, epoch + 1)
            if val_stats:
                for j, ql in enumerate(val_stats['quantizer_losses']):
                    writer.add_scalar(f"Loss/val_quantizer_{j}", ql, epoch + 1)
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            
            numbered_path = out_dir / "checkpoints" / f"checkpoint_{epoch+1}.pt"
            torch.save(checkpoint_data, numbered_path)
            
            last_path = out_dir / "checkpoints" / "last_checkpoint.pt"
            shutil.copy2(numbered_path, last_path)
            
            print(f"   💾 Checkpoint saved at epoch {epoch+1}")
    
    # Save final checkpoint
    checkpoint_data = {
        'epoch': start_epoch + num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    final_path = out_dir / "checkpoints" / f"checkpoint_{start_epoch + num_epochs}.pt"
    torch.save(checkpoint_data, final_path)
    last_path = out_dir / "checkpoints" / "last_checkpoint.pt"
    shutil.copy2(final_path, last_path)
    
    if writer:
        writer.close()
    
    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n\033[1m🏋️ Training Completed\033[0m\n")
    print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Model saved to: {out_dir}")
    print("="*70 + "\n")
    
    return {
        'output_dir': str(out_dir),
        'final_epoch': start_epoch + num_epochs,
        'elapsed_time': elapsed,
        'device': str(device)
    }


# ============================================================================
# Exported Functions (for notebook use)
# ============================================================================

__all__ = [
    'load_dataset_config',
    'get_available_splits',
    'create_dataloaders',
    'create_model',
    'train_model'
]