"""
Real-time and offline inference for RNeNcodec models.

This module provides a clean interface for loading trained models and generating audio
both in real-time (with GUI) and offline. It automatically reads conditioning metadata
from the model directory to provide proper parameter ranges and units in the GUI.
"""

import os
import json
import torch
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from IPython.display import Audio, display

from transformers import EncodecModel
from rnencodec.generator import RNNGeneratorSoft, EncodecRTPlayer
from rnencodec.model.gru_audio_model import GRUModelConfig
from rnencodec.audioDataLoader.audio_dataset import LatentDatasetConfig

# Import real-time synth components
try:
    from ipywidgets import FloatSlider, ToggleButton, VBox, HBox, Label, Layout, HTML
    from IPython.display import display
    from realtime_synth.engine import RealtimeSynth
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    print("Warning: realtime_synth not available. Real-time GUI will not work.")


# System constants
SR = 24000  # Sample rate
FRAME_RATE = 75  # EnCodec frame rate
DEVICE = 'cpu'  # Best for inference
BUFFERSIZE = 320  # Frame length in samples (24000/75)


def build_rnencodec_ui(rt_player, samplerate=48000, blocksize=640, channels=1):
    """
    Build a custom UI for RNeNcodec without generator selector and with proper readouts.
    
    Args:
        rt_player: EncodecRTPlayer instance
        samplerate: Output sample rate
        blocksize: Audio block size
        channels: Number of audio channels
    
    Returns:
        Tuple of (synth, ui) - the synthesizer engine and UI container
    """
    synth = RealtimeSynth(generator=rt_player, samplerate=samplerate, blocksize=blocksize, channels=channels)
    
    # UI widgets
    sliders_box = VBox()
    readouts_box = VBox()  # Changed to VBox for vertical stacking
    play_toggle = ToggleButton(value=False, description='▶️ Play', layout=Layout(width='150px'))
    status_line = HTML(value="")
    
    def refresh_readouts():
        """Update the readout labels with formatted values."""
        formatted = synth.gen.formatted_readouts()
        labels = list(readouts_box.children)
        for i, lbl in enumerate(labels):
            lbl.value = formatted[i] if i < len(formatted) else ""
    
    def build_param_ui():
        """Build sliders and readout labels."""
        sliders = []
        readout_labels = []
        n = synth.gen.num_params()
        
        for i in range(n):
            # Get label (just the feature name, no range)
            lab = synth.gen.param_labels[i] if i < len(synth.gen.param_labels) else f"Param {i+1}"
            init = synth.gen.norm_params[i] if i < len(synth.gen.norm_params) else 0.5
            
            # Create slider WITHOUT readout display
            s = FloatSlider(
                value=float(init), 
                min=0.0, 
                max=1.0, 
                step=0.001,
                description=lab, 
                readout=False,  # Hide the 0-1 value display
                layout=Layout(width='500px')  # Wider for longer feature names
            )
            
            def on_change(change):
                if change['name'] == 'value':
                    values = [w.value for w in sliders_box.children]
                    synth.set_params(values)
                    refresh_readouts()
            
            s.observe(on_change, names='value')
            sliders.append(s)
            
            # Create readout label (wider to show full values with units)
            readout_labels.append(Label("", layout=Layout(width='300px')))
        
        sliders_box.children = sliders
        readouts_box.children = readout_labels
        synth.set_params([w.value for w in sliders])
        refresh_readouts()
    
    def on_play(change):
        """Handle play/stop button clicks."""
        if change['name'] == 'value':
            if change['new']:
                play_toggle.description = '⏹ Stop'
                synth.start()
                status_line.value = "<i>Audio running…</i>"
            else:
                play_toggle.description = '▶️ Play'
                synth.stop()
                status_line.value = "<i>Audio stopped.</i>"
    
    play_toggle.observe(on_play, names='value')
    
    # Build the UI layout - sliders and readouts side by side
    controls = VBox([
        HBox([sliders_box, readouts_box]),  # Side by side layout
        HBox([play_toggle]),
        HTML(
            "<small>Adjust parameters in real-time. Values update as you move the sliders.<br>"
            "Tip: decrease blocksize for lower latency (higher xrun risk); increase if you hear clicks.</small>"
        )
    ])
    
    build_param_ui()
    display(controls, status_line)
    return synth, controls


class ParameterScaler:
    """Handles scaling between normalized [0,1] values and real parameter ranges."""
    
    def __init__(self, conditioning_config: Dict):
        """
        Initialize scaler from conditioning config.
        
        Args:
            conditioning_config: Dict with 'features' containing parameter metadata
        """
        self.config = conditioning_config
        self.features = conditioning_config.get('features', {})
        self.feature_names = conditioning_config.get('feature_names', [])
    
    def normalize(self, param_name: str, real_value: float) -> float:
        """Convert real value to normalized [0,1] range."""
        if param_name not in self.features:
            return real_value
        
        feature = self.features[param_name]
        if feature['type'] == 'binary':
            # Binary features are already 0 or 1
            return float(real_value)
        elif feature['type'] == 'continuous':
            min_val = feature['min']
            max_val = feature['max']
            return (real_value - min_val) / (max_val - min_val)
        return real_value
    
    def denormalize(self, param_name: str, norm_value: float) -> float:
        """Convert normalized [0,1] value to real range."""
        if param_name not in self.features:
            return norm_value
        
        feature = self.features[param_name]
        if feature['type'] == 'binary':
            return float(norm_value)
        elif feature['type'] == 'continuous':
            min_val = feature['min']
            max_val = feature['max']
            return min_val + norm_value * (max_val - min_val)
        return norm_value
    
    def get_range(self, param_name: str) -> Tuple[float, float]:
        """Get the real value range for a parameter."""
        if param_name not in self.features:
            return (0.0, 1.0)
        
        feature = self.features[param_name]
        if feature['type'] == 'binary':
            return (0.0, 1.0)
        elif feature['type'] == 'continuous':
            return (float(feature['min']), float(feature['max']))
        return (0.0, 1.0)
    
    def get_unit(self, param_name: str) -> str:
        """Get the unit for a parameter."""
        if param_name not in self.features:
            return ""
        return self.features[param_name].get('unit', '')
    
    def get_label(self, param_name: str) -> str:
        """Get a formatted label with name, range, and unit."""
        if param_name not in self.features:
            return param_name
        
        feature = self.features[param_name]
        unit = feature.get('unit', '')
        
        if feature['type'] == 'binary':
            # For binary: "instrument_piano (0-1)"
            return f"{param_name} (0-1)"
        elif feature['type'] == 'continuous':
            min_val = feature['min']
            max_val = feature['max']
            if unit:
                return f"{param_name} ({min_val}-{max_val} {unit})"
            else:
                return f"{param_name} ({min_val}-{max_val})"
        return param_name


def load_conditioning_config(model_dir: str) -> Dict:
    """
    Load conditioning configuration from model directory.
    
    Args:
        model_dir: Path to model directory containing conditioning_config.json
    
    Returns:
        Dict with conditioning configuration
    """
    config_path = Path(model_dir) / 'conditioning_config.json'
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"conditioning_config.json not found at {config_path}.\n"
            "This file should have been saved during training."
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded conditioning config with {config['num_features']} features:")
    for name in config['feature_names']:
        feature = config['features'][name]
        if feature['type'] == 'continuous':
            print(f"  - {name}: {feature['min']}-{feature['max']} {feature.get('unit', '')}")
        else:
            print(f"  - {name}: {feature['type']}")
    
    return config


def load_model(
    model_dir: str,
    checkpoint_name: Optional[str] = None,
    chunksize: int = 20,
    hopsize: int = 8,
    override_cascade_mode: Optional[str] = None,
    override_temperature: Optional[float] = None,
    override_top_n: Optional[int] = None,
    override_tau_soft: Optional[float] = None,
    sample_mode: str = "sample",
    top_k_outside: Optional[int] = None,
    temperature_outside: float = 1.0
) -> Tuple[RNNGeneratorSoft, EncodecModel, Dict, GRUModelConfig]:
    """
    Load trained RNN model and EnCodec model from checkpoint.
    
    Args:
        model_dir: Path to model directory containing checkpoints and configs
        checkpoint_name: Name of checkpoint file (e.g., 'checkpoint_75.pt'). 
                        If None, uses the last checkpoint.
        chunksize: Chunk size for generation (default: 20)
        hopsize: Hop size for generation (default: 8)
        override_cascade_mode: Override the model's cascade mode ("soft" or "hard")
        override_temperature: Override temperature for hard cascade
        override_top_n: Override top-k for hard cascade
        override_tau_soft: Override tau for soft cascade
        sample_mode: Sampling mode for soft cascade ("argmax", "gumbel", "sample")
        top_k_outside: Top-k filtering for soft cascade sampling
        temperature_outside: Temperature for soft cascade sampling
    
    Returns:
        Tuple of (rnn_generator, encodec_model, conditioning_config, model_config)
    """
    model_dir = Path(model_dir)
    
    # Load conditioning config
    conditioning_config = load_conditioning_config(model_dir)
    
    # Load EnCodec model
    print("\nLoading EnCodec model...")
    enc_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    enc_model.eval()
    
    # Find checkpoint
    if checkpoint_name is None:
        # Find the last checkpoint
        checkpoint_dir = model_dir / 'checkpoints'
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pt'))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        checkpoint_path = checkpoints[-1]
        print(f"Using last checkpoint: {checkpoint_path.name}")
    else:
        checkpoint_path = model_dir / 'checkpoints' / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Using checkpoint: {checkpoint_name}")
    
    # Load model configs
    config_path = model_dir / "config_v2.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    saved_configs = torch.load(config_path, weights_only=False)
    model_config = GRUModelConfig(**saved_configs["model_config"])
    data_config = LatentDatasetConfig(**saved_configs["data_config"])
    
    # Apply overrides to model config if requested
    original_cascade = model_config.cascade
    if override_cascade_mode is not None:
        if override_cascade_mode not in ["soft", "hard"]:
            raise ValueError(f"override_cascade_mode must be 'soft' or 'hard', got: {override_cascade_mode}")
        model_config.cascade = override_cascade_mode
        print(f"⚠️  Overriding cascade mode: {original_cascade} → {override_cascade_mode}")
    
    if override_temperature is not None:
        model_config.temperature_hard = override_temperature
        print(f"⚠️  Overriding temperature: {override_temperature}")
    
    if override_top_n is not None:
        model_config.top_n_hard = override_top_n
        print(f"⚠️  Overriding top-k: {override_top_n}")
    
    if override_tau_soft is not None:
        model_config.tau_soft = override_tau_soft
        print(f"⚠️  Overriding tau (soft): {override_tau_soft}")
    
    # Create RNN generator
    print(f"Loading RNN model from {checkpoint_path.name}...")
    rnngen = RNNGeneratorSoft.from_checkpoint(
        checkpoint_path,
        model_config,
        data_config,
        enc_model,
        chunksize,
        hopsize,
        sample_mode_outside=sample_mode,
        top_k_outside=top_k_outside,
        temperature_outside=temperature_outside
    )
    
    print(f"Model loaded successfully!")
    print(f"  - Hidden size: {model_config.hidden_size}")
    print(f"  - Num layers: {model_config.num_layers}")
    print(f"  - Conditioning size: {model_config.cond_size}")
    print(f"  - Cascade mode: {model_config.cascade}")
    if model_config.cascade == "hard":
        print(f"  - Hard sampling mode: {model_config.hard_sample_mode}")
        print(f"  - Temperature: {model_config.temperature_hard}")
        print(f"  - Top-k: {model_config.top_n_hard if model_config.top_n_hard else 'None'}")
    else:
        print(f"  - Tau (soft): {model_config.tau_soft}")
    print(f"  - Device: {DEVICE}")
    
    return rnngen, enc_model, conditioning_config, model_config


def create_realtime_synth(
    rnngen: RNNGeneratorSoft,
    conditioning_config: Dict,
    scaler: ParameterScaler,
    initial_values: Optional[Dict[str, float]] = None,
    chunksize: int = 20,
    hopsize: int = 8,
    output_samplerate: int = 48000
):
    """
    Create real-time synthesizer with GUI.
    
    Args:
        rnngen: Loaded RNN generator
        conditioning_config: Conditioning configuration dict
        scaler: ParameterScaler instance for converting between normalized and real values
        initial_values: Dict mapping parameter names to initial real values.
                       If None, uses midpoint for continuous, 0.5 for binary.
        chunksize: Chunk size for generation (default: 20)
        hopsize: Hop size for generation (default: 8)
        output_samplerate: Output sample rate for audio hardware (default: 48000)
    
    Returns:
        Tuple of (synth, ui) - the synthesizer and UI objects
    """
    if not REALTIME_AVAILABLE:
        raise RuntimeError(
            "Real-time synthesis requires 'rtpysynth[ui]' package.\n"
            "Install with: pip install 'rtpysynth[ui] @ git+https://github.com/lonce/RTPySynth@v0.1.4'"
        )
    
    # Use provided scaler
    feature_names = conditioning_config['feature_names']
    
    # Prepare initial normalized values
    norm_param_vals = []
    for name in feature_names:
        if initial_values and name in initial_values:
            # User provided real value - normalize it
            real_val = initial_values[name]
            norm_val = scaler.normalize(name, real_val)
        else:
            # Default: midpoint for continuous, 0.5 for binary
            norm_val = 0.5
        norm_param_vals.append(norm_val)
    
    # Create parameter labels (feature names only, ranges removed since they'll be in the readout)
    param_labels = feature_names  # Just use the feature names
    
    print("\nInitializing real-time synthesizer...")
    print("Parameter configuration:")
    for name, norm_val in zip(feature_names, norm_param_vals):
        real_val = scaler.denormalize(name, norm_val)
        unit = scaler.get_unit(name)
        min_val, max_val = scaler.get_range(name)
        print(f"  - {name}: {real_val:.2f} {unit} (range: {min_val}-{max_val} {unit})")
    
    # Create RT player with scaler
    rt_player = EncodecRTPlayer(
        rnngen,
        SR,
        FRAME_RATE,
        BUFFERSIZE,
        chunksize,
        hopsize,
        norm_param_vals,
        param_labels,
        param_scaler=scaler  # Pass the scaler
    )
    
    # Create custom UI (no generator selector, proper readouts)
    synth, ui = build_rnencodec_ui(
        rt_player,
        samplerate=output_samplerate,
        blocksize=BUFFERSIZE * 2,
        channels=1
    )
    
    print("\n✓ Real-time synthesizer ready!")
    print("  Use the GUI to control parameters and generate audio in real-time.")
    print("  Note: Check synth.gen._last_error after stopping to see any runtime issues.")
    
    return synth, ui


def generate_offline(
    rnngen: RNNGeneratorSoft,
    conditioning_config: Dict,
    conditioning_sequence: Optional[torch.Tensor] = None,
    duration: float = 10.0,
    param_values: Optional[Dict[str, float]] = None,
    device: str = DEVICE
) -> np.ndarray:
    """
    Generate audio offline (non-real-time) with custom conditioning.
    
    Args:
        rnngen: Loaded RNN generator
        conditioning_config: Conditioning configuration dict
        conditioning_sequence: Pre-defined conditioning tensor of shape (num_frames, num_features).
                              If provided, overrides duration and param_values.
        duration: Duration in seconds (default: 10.0). Ignored if conditioning_sequence provided.
        param_values: Dict mapping parameter names to constant real values for generation.
                     Only used if conditioning_sequence is None. If None, uses 0.5 for all params.
        device: Device to use for generation (default: 'cpu')
    
    Returns:
        Generated audio as numpy array
    """
    scaler = ParameterScaler(conditioning_config)
    feature_names = conditioning_config['feature_names']
    num_features = len(feature_names)
    
    if conditioning_sequence is not None:
        # Use provided sequence
        print(f"Using provided conditioning sequence: {conditioning_sequence.shape}")
        cond_seq = conditioning_sequence
    else:
        # Create constant conditioning from param_values
        num_frames = int(duration * FRAME_RATE)
        cond_seq = torch.zeros(num_frames, num_features)
        
        print(f"Generating {duration}s of audio ({num_frames} frames)...")
        print("Conditioning values:")
        
        for i, name in enumerate(feature_names):
            if param_values and name in param_values:
                real_val = param_values[name]
                norm_val = scaler.normalize(name, real_val)
            else:
                norm_val = 0.5
                real_val = scaler.denormalize(name, norm_val)
            
            cond_seq[:, i] = norm_val
            unit = scaler.get_unit(name)
            print(f"  - {name}: {real_val:.2f} {unit}")
    
    # Generate audio
    start = time.perf_counter()
    generated_audio = rnngen.getNextAudioHop(cond_seq.to(device))
    elapsed = time.perf_counter() - start
    
    duration_sec = generated_audio.shape[0] / SR
    print(f"\nGeneration complete!")
    print(f"  - Audio duration: {duration_sec:.2f}s")
    print(f"  - Generation time: {elapsed:.2f}s")
    print(f"  - Real-time factor: {duration_sec/elapsed:.2f}x")
    
    return generated_audio


def play_audio(audio: np.ndarray, sample_rate: int = SR):
    """
    Display audio player in notebook.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate (default: 24000)
    """
    display(Audio(audio, rate=sample_rate))


def run_inference(
    model_dir: str,
    checkpoint_name: Optional[str] = None,
    mode: str = "realtime",
    cascade_mode: Optional[str] = None,
    temperature: Optional[float] = None,
    top_n: Optional[int] = None,
    tau_soft: Optional[float] = None,
    sample_mode: str = "sample",
    top_k_outside: Optional[int] = None,
    temperature_outside: float = 1.0,
    initial_values: Optional[Dict[str, float]] = None,
    offline_duration: float = 10.0,
    offline_params: Optional[Dict[str, float]] = None,
    chunksize: int = 20,
    hopsize: int = 8
):
    """
    Run inference - either real-time GUI or offline generation.
    
    This is the main entry point for inference. It loads the model and either:
    - Creates a real-time GUI for interactive control (mode='realtime')
    - Generates audio offline and returns it (mode='offline')
    
    You can override the model's trained cascade mode and parameters at inference time
    for experimentation, though the model may perform best with its original settings.
    
    Args:
        model_dir: Path to model directory
        checkpoint_name: Checkpoint filename (None = use last checkpoint)
        mode: 'realtime' for GUI, 'offline' for non-interactive generation
        cascade_mode: Override cascade mode - "soft" (deterministic) or "hard" (sampling).
                     If None, uses the model's trained setting.
        temperature: Override temperature for hard cascade (only used if cascade_mode="hard")
        top_n: Override top-k for hard cascade (None = no restriction)
        tau_soft: Override tau for soft cascade (only used if cascade_mode="soft")
        sample_mode: Sampling mode for soft cascade ("argmax", "gumbel", "sample") - default: "sample"
        top_k_outside: Top-k filtering for soft cascade sampling (None = no restriction)
        temperature_outside: Temperature for soft cascade sampling - default: 1.0
        initial_values: For realtime: dict of parameter_name -> initial_real_value
        offline_duration: For offline: duration in seconds
        offline_params: For offline: dict of parameter_name -> constant_real_value
        chunksize: Chunk size for generation (default: 20)
        hopsize: Hop size for generation (default: 8)
    
    Returns:
        - If mode='realtime': (synth, ui) tuple for GUI control
        - If mode='offline': generated audio as numpy array
        
    Example:
        # Use model's trained cascade mode
        run_inference(
            model_dir="../../models/my_model",
            checkpoint_name="checkpoint_75.pt"
        )
        
        # Override to soft cascade
        run_inference(
            model_dir="../../models/my_model",
            checkpoint_name="checkpoint_75.pt",
            cascade_mode="soft",
            tau_soft=0.5
        )
        
        # Override to hard cascade with sampling
        run_inference(
            model_dir="../../models/my_model",
            checkpoint_name="checkpoint_75.pt",
            cascade_mode="hard",
            temperature=0.8,
            top_n=10
        )
    """

    if cascade_mode=="hard":
        if temperature is None:
            temperature=0.8
        if top_n is None:
            top_n=8
    elif cascade_mode=="soft":
        if tau_soft is None:
            tau_soft=0.8
    
    # Load model with optional overrides
    rnngen, enc_model, conditioning_config, model_config = load_model(
        model_dir=model_dir,
        checkpoint_name=checkpoint_name,
        chunksize=chunksize,
        hopsize=hopsize,
        override_cascade_mode=cascade_mode,
        override_temperature=temperature,
        override_top_n=top_n,
        override_tau_soft=tau_soft,
        sample_mode=sample_mode,
        top_k_outside=top_k_outside,
        temperature_outside=temperature_outside
    )
    
    # Create parameter scaler
    scaler = ParameterScaler(conditioning_config)
    
    if mode == "realtime":
        # Create and return real-time synth
        return create_realtime_synth(
            rnngen=rnngen,
            conditioning_config=conditioning_config,
            scaler=scaler,
            initial_values=initial_values,
            chunksize=chunksize,
            hopsize=hopsize
        )
    
    
    elif mode == "offline":
        # Generate offline and return audio
        audio = generate_offline(
            rnngen=rnngen,
            conditioning_config=conditioning_config,
            duration=offline_duration,
            param_values=offline_params
        )
        return audio
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'realtime' or 'offline'.")
