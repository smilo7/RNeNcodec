import random
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

from dataprep.auxiliary_functions import (
    load_parameter_config,
    find_audio_csv_pairs,
    get_audio_info,
    get_parameter_names,
    get_parameter_unit
)


def validate_csv_alignment(csv_path: Path, expected_frames: int) -> Dict:
    """Validate CSV annotation alignment with audio."""
    try:
        df = pd.read_csv(csv_path)
        csv_rows = len(df)
        
        return {
            'csv_rows': csv_rows,
            'expected_frames': expected_frames,
            'extra_rows': csv_rows - expected_frames,
            'aligned': abs(csv_rows - expected_frames) <= 0  # Allow 0 frame tolerance
        }
    except Exception as e:
        print(f"❌ Error reading {csv_path}: {e}")
        return None


def summarize_dataset(raw_dir: Path) -> Dict:
    """Summarize entire dataset: files, alignment, parameters."""
    raw_dir = Path(raw_dir)
    
    # Load parameter config
    config_path = raw_dir / 'parameters.json'
    if not config_path.exists():
        raise FileNotFoundError(f"parameters.json not found in {raw_dir}")
    
    config = load_parameter_config(config_path)
    
    # Get all parameter names (both continuous and class types for summary)
    param_names = []
    for param_info in config.values():
        param_names.append(param_info['name'])
    
    # Find audio-CSV pairs
    pairs = find_audio_csv_pairs(raw_dir)
    
    if not pairs:
        raise ValueError(f"No audio-CSV pairs found in {raw_dir}")
    
    # Analyze each pair
    summary = {
        'total_files': len(pairs),
        'parameters': param_names,
        'files': [],
        'alignment_issues': [],
        'total_audio_duration': 0,
        'total_csv_frames': 0
    }
    
    for audio_path, csv_path in pairs:
        # Audio info
        audio_info = get_audio_info(audio_path)
        if audio_info is None:
            continue
            
        # CSV validation
        csv_info = validate_csv_alignment(csv_path, audio_info['expected_encodec_frames'])
        if csv_info is None:
            continue
        
        file_info = {
            'name': audio_path.stem,
            'audio_duration': audio_info['duration'],
            'audio_samplerate': audio_info['samplerate'],
            'expected_frames': audio_info['expected_encodec_frames'],
            'csv_frames': csv_info['csv_rows'],
            'aligned': csv_info['aligned'],
            'extra_frames': csv_info['extra_rows'],
            'audio_samples': audio_info['frames']  # Add for debugging
        }
        
        summary['files'].append(file_info)
        summary['total_audio_duration'] += audio_info['duration']
        summary['total_csv_frames'] += csv_info['csv_rows']
        
        if not csv_info['aligned']:
            summary['alignment_issues'].append(file_info)
    
    return summary


def plot_parameter_patterns(raw_dir: Path, file_name: Optional[str] = None):
    """Plot parameter trajectories for all parameters in a file."""
    raw_dir = Path(raw_dir)
    
    # Load config
    config = load_parameter_config(raw_dir / 'parameters.json')
    
    # Filter out class-type parameters and get continuous parameters only
    param_names = get_parameter_names(config)
    
    # Select file
    if file_name is None:
        pairs = find_audio_csv_pairs(raw_dir)
        if not pairs:
            raise ValueError("No audio-CSV pairs found")
        audio_path, csv_path = random.choice(pairs)
        file_name = audio_path.stem
    else:
        csv_path = raw_dir / f'{file_name}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Plot each parameter
    n_params = len(param_names)
    cols = min(3, n_params)
    rows = (n_params + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    if n_params == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'Parameter Trajectories: {file_name}', fontsize=16, fontweight='bold')
    
    for i, param_name in enumerate(param_names):
        if param_name in df.columns:
            values = df[param_name].to_numpy()
            
            axes[i].plot(values, linewidth=1.5)
            axes[i].set_title(f'{param_name}')
            axes[i].set_xlabel('Frame Index')
            
            # Get units from config for y-axis label
            units = get_parameter_unit(config, param_name)
            
            y_label = f'Value ({units})' if units else 'Value'
            axes[i].set_ylabel(y_label)
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            axes[i].axhline(mean_val, color='red', linestyle='--', alpha=0.7, 
                           label=f'Mean: {mean_val:.2f}')
            axes[i].legend()
        else:
            axes[i].text(0.5, 0.5, f'Parameter "{param_name}"\nnot found in CSV', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{param_name} (Missing)')
    
    # Hide extra subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_sample(raw_dir: Path, file_name: Optional[str] = None):
    """Plot entire file: audio waveform + parameter trajectories."""
    raw_dir = Path(raw_dir)
    
    # Select random file if not specified
    pairs = find_audio_csv_pairs(raw_dir)
    if not pairs:
        raise ValueError("No audio-CSV pairs found")
    
    if file_name is None:
        audio_path, csv_path = random.choice(pairs)
        file_name = audio_path.stem
        print(f"📊 Randomly selected file: {file_name}")
    else:
        audio_path = None
        for ap, cp in pairs:
            if ap.stem == file_name:
                audio_path, csv_path = ap, cp
                break
        if audio_path is None:
            raise FileNotFoundError(f"File '{file_name}' not found in dataset")
    
    # Load audio using EncHF_Dataset approach: 24kHz mono
    audio_path_str = str(audio_path)
    try:
        # Load audio at 24kHz mono to match EnCodec processing
        audio, sr_og = librosa.load(audio_path_str, sr=None, mono=True)
        audio, sr = librosa.load(audio_path_str, sr=24000, mono=True)
        if sr != sr_og:
            audio = audio[:-12]
        
    except Exception as e:
        print(f"❌ Error loading audio {audio_path}: {e}")
        return
    
    audio_duration = len(audio) / sr
    
    # Load CSV
    df = pd.read_csv(csv_path)
    config = load_parameter_config(raw_dir / 'parameters.json')
    
    # Filter out class-type parameters and get continuous parameters only
    param_names = get_parameter_names(config)
        
    # Align audio and parameters by taking the minimum length
    # Calculate EnCodec frames using ceiling division (same logic as auxiliary_functions)
    samples_per_frame = sr // 75  # 320 samples per frame at 24kHz
    expected_frames = math.ceil(len(audio) / samples_per_frame)
    csv_frames = len(df)
    
    # Use the shorter of the two to ensure alignment
    aligned_frames = min(expected_frames, csv_frames)
    aligned_duration = aligned_frames / 75  # Convert back to seconds
    
    # Trim audio to match aligned duration
    aligned_audio_samples = int(aligned_duration * sr)
    audio_aligned = audio[:aligned_audio_samples]
    
    # Trim CSV to aligned frames
    df_aligned = df.iloc[:aligned_frames].copy()
    
    print(f"🔄 Aligned data: {aligned_frames} frames ({aligned_duration:.1f}s)")
    if expected_frames != csv_frames:
        print(f"   Original: {expected_frames} audio frames vs {csv_frames} CSV frames")
    
    # Create time axes for entire aligned data
    audio_time = np.linspace(0, aligned_duration, len(audio_aligned))
    param_time = np.linspace(0, aligned_duration, aligned_frames)
    
    # Plot
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params + 1, 1, figsize=(15, 3*(n_params + 1)))
    
    fig.suptitle(f'File: {file_name} ({aligned_duration:.1f}s, {aligned_frames} frames)', 
                 fontsize=16, fontweight='bold')
    
    # Audio waveform
    axes[0].plot(audio_time, audio_aligned, color='blue', alpha=0.7, linewidth=0.5)
    axes[0].set_title('Audio Waveform (Entire File)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Parameters
    for i, param_name in enumerate(param_names):
        if param_name in df_aligned.columns:
            values = df_aligned[param_name].to_numpy()
            
            axes[i+1].plot(param_time, values, color='red', linewidth=1.5, alpha=0.8)
            axes[i+1].set_title(f'Parameter: {param_name}')
            
            # Get units from config for y-axis label
            units = get_parameter_unit(config, param_name)
            
            y_label = f'{units}' if units else 'Value'
            axes[i+1].set_ylabel(y_label)
            axes[i+1].grid(True, alpha=0.3)
            
        else:
            axes[i+1].text(0.5, 0.5, f'Parameter "{param_name}" not found', 
                          ha='center', va='center', transform=axes[i+1].transAxes)
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.show()


def analyze_dataset(raw_dir: Path):
    """
    Complete dataset analysis: summarize, validate, and optionally plot a sample.
    
    This is the main function to call from a notebook cell.
    """
    import matplotlib
    # Check if we're in a notebook or have a display
    try:
        import IPython
        in_notebook = True
    except ImportError:
        in_notebook = False
        
    if not in_notebook:
        matplotlib.use('Agg')  # Use non-interactive backend for headless mode
        plot_random = False
    
    raw_dir = Path(raw_dir)
    
    # print("=" * 70)
    print("\033[1mDATASET SUMMARY:\033[0m\n")
    # print("=" * 70)
    
    # 1. Summarize dataset
    # print("📊 Loading and summarizing dataset...")
    try:
        summary = summarize_dataset(raw_dir)
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return
    
    # 2. Print summary
    print(f"✅ Found {summary['total_files']} audio-CSV pairs")
    print(f"📁 Parameters: {', '.join(summary['parameters'])}")
    print(f"⏱️ Total audio duration: {summary['total_audio_duration']:.1f} seconds")
    # print(f"📏 Total CSV frames: {summary['total_csv_frames']:,}")
    
    # 3. Print file details
    print(f"\n\033[1mFILE DETAILS:\033[0m\n")
    for file_info in summary['files']:
        status = "✅" if file_info['aligned'] else "⚠️"
        
        # Build alignment message
        base_msg = f"{status} {file_info['name']}: {file_info['audio_duration']:.1f}s, {file_info['csv_frames']} frames"
        
        if not file_info['aligned']:
            extra_frames = file_info['extra_frames']
            # # Debug info
            # print(f"DEBUG {file_info['name']}: {file_info['audio_samples']} samples → {file_info['expected_frames']} expected frames, {file_info['csv_frames']} CSV frames")
            if extra_frames > 0:
                base_msg += f" ({extra_frames} extra annotation frames in CSV)"
            else:
                base_msg += f" ({abs(extra_frames)} extra audio frames in audio file)"
        
        print(base_msg)
    
    if summary['alignment_issues']:
        print("\n   ⚠️ Some files have alignment issues (see above). The dataloader will automatically trim extra frames.")
    else:
        print("\n   ✅ All files are properly aligned!")
    return summary

def interactive_file_selector(raw_dir):
    """
    Create an interactive widget to select and plot files from the dataset.
    Returns a widget that can be displayed in a Jupyter notebook.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError:
        print("❌ This function requires ipywidgets. Install with: pip install ipywidgets")
        return

    print(f"\n\033[1mFILE PLOTTER:\033[0m\n")

    raw_dir = Path(raw_dir)
    
    # Get all audio-CSV pairs
    pairs = find_audio_csv_pairs(raw_dir)
    if not pairs:
        print("❌ No audio-CSV pairs found in dataset")
        return
    
    # Extract file names (without extension)
    file_names = [audio_path.stem for audio_path, _ in pairs]
    
    # Create widgets
    file_dropdown = widgets.Dropdown(
        options=file_names,
        description='Select File:',
        disabled=False,
        style={'description_width': '100px'}
    )
    
    plot_button = widgets.Button(
        description='📊 Plot File',
        button_style='success',
        tooltip='Click to plot the selected file',
        icon='chart-line'
    )
    
    output = widgets.Output()
    
    # Button click handler
    def on_plot_button_clicked(b):
        with output:
            clear_output(wait=True)
            selected_file = file_dropdown.value
            print(f"📈 Plotting file: {selected_file}")
            try:
                plot_sample(raw_dir, file_name=selected_file)
            except Exception as e:
                print(f"❌ Error plotting file: {e}")
    
    plot_button.on_click(on_plot_button_clicked)
    
    # Layout
    controls = widgets.HBox([file_dropdown, plot_button])
    ui = widgets.VBox([controls, output])
    
    return ui


# Convenience functions for notebook use
def quick_analyze(dataset_path, individual_plot_selector=False):
    raw_dir = dataset_path + "/raw"
    """Quick analysis for notebook - just call this function!"""
    analyze_dataset(Path(raw_dir))
    if individual_plot_selector:
        from IPython.display import display
        display(interactive_file_selector(Path(raw_dir)))
    return

# def quick_file_selector(dataset_path):
#     raw_dir = dataset_path + "/raw"
#     """Quickly display the interactive file selector in a notebook."""
#     from IPython.display import display
#     display(interactive_file_selector(Path(raw_dir)))
#     return