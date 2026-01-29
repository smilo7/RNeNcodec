import numpy as np
import torch
import matplotlib.pyplot as plt

##############################################################################################
#########          These are for computing and displaying the parameter contours  ############
##############################################################################################
def multi_linspace(breakpoints, num_points):
    """
    Generate linearly interpolated values across multiple segments.
    
    Parameters:
    -----------
    breakpoints : list of tuples
        Each tuple contains (proportion, value) where:
        - proportion: position along the sequence (0 to 1)
        - value: the value at that position
    num_points : int
        Total number of points to generate
    
    Returns:
    --------
    numpy.ndarray
        Array of interpolated values
    
    Example:
    --------
    >>> multi_linspace([(0, 0), (0.25, 0), (0.75, 1), (1, 1)], 9)
    array([0.   , 0.   , 0.   , 0.25 , 0.5  , 0.75 , 1.   , 1.   , 1.   ])
    """
    # Sort breakpoints by proportion to ensure correct order
    #breakpoints = sorted(breakpoints, key=lambda x: x[0])
    
    # Extract proportions and values
    proportions = np.array([bp[0] for bp in breakpoints])
    values = np.array([bp[1] for bp in breakpoints])
    
    # Create the index array (0 to num_points-1)
    indices = np.arange(num_points)
    
    # Convert indices to proportions (0 to 1)
    if num_points == 1:
        index_proportions = np.array([0.0])
    else:
        index_proportions = indices / (num_points - 1)
    
    # Interpolate values at each index proportion
    interpolated_values = np.interp(index_proportions, proportions, values)
    
    return interpolated_values

# -------------------------------------------------
def steps(values, num_points):
    """
    Create horizontal line segments of equal length for each value.
    
    Parameters:
    -----------
    values : list or array-like
        Values for each horizontal segment
    num_points : int
        Total number of points to generate
    
    Returns:
    --------
    numpy.ndarray
        Array with horizontal segments at each value
    
    Example:
    --------
    >>> steps([1, 3, 2, 4], 12)
    array([1., 1., 1., 3., 3., 3., 2., 2., 2., 4., 4., 4.])
    """
    if len(values) == 0:
        return np.array([])
    
    if len(values) == 1:
        return np.full(num_points, values[0])
    
    # Create breakpoints for step function
    breakpoints = []
    n_segments = len(values)
    
    for i, value in enumerate(values):
        # Start proportion for this segment
        start_prop = i / n_segments
        # End proportion for this segment  
        end_prop = (i + 1) / n_segments
        
        if i == 0:
            # First segment: start at 0
            breakpoints.append((start_prop, value))
        else:
            # Add step transition: duplicate the proportion with new value
            breakpoints.append((start_prop, value))
        
        if i == len(values) - 1:
            # Last segment: end at 1
            breakpoints.append((end_prop, value))
        else:
            # Add end of current segment
            breakpoints.append((end_prop, value))
    
    return multi_linspace(breakpoints, num_points)

# ----------------------------------------------------------------------------
def plot_condition_tensor(cond_tensor: torch.FloatTensor, sr: int):
    """
    Plot conditioning tensor of shape [T, p] with time in seconds on the x-axis.
    """
    T, p = cond_tensor.shape
    time = torch.arange(T) / sr

    plt.figure(figsize=(20, 5))
    for i in range(p):
        plt.plot(time, cond_tensor[:, i], label=f'Param {i+1}')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized value")
    plt.title("Conditioning Parameters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



############################################################################################
##################         model parameter counting        #################################
############################################################################################

from collections import defaultdict
import torch.nn as nn

def count_params(module: nn.Module, trainable_only=True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)

def param_breakdown(model, trainable_only=True):
    out = {}

    # 1) Simple linear projections
    out['latent_proj'] = count_params(model.latent_proj, trainable_only)
    out['cond_proj']   = count_params(model.cond_proj, trainable_only)

    # 2) GRU – total and per-layer/per-direction
    gru_total = count_params(model.gru, trainable_only)
    out['gru_total'] = gru_total

    gru_layers = defaultdict(int)
    for name, p in model.gru.named_parameters():
        if not (p.requires_grad or not trainable_only):
            continue
        # names look like: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
        # bidirectional adds suffix like _reverse
        # extract layer and direction
        layer = 'unknown'
        direction = 'fwd'
        if '_l' in name:
            # e.g., weight_ih_l2 or weight_ih_l1_reverse
            base, after = name.split('_l', 1)
            # after might be '2' or '1_reverse'
            parts = after.split('_', 1)
            layer_idx = parts[0]
            if len(parts) > 1 and parts[1] == 'reverse':
                direction = 'rev'
            layer = f"l{layer_idx}"
        gru_layers[(layer, direction)] += p.numel()

    # format per-layer breakdown
    out['gru_layers'] = {
        f"{layer}/{direction}": n for (layer, direction), n in sorted(gru_layers.items())
    }

    # 3) Decoders – per head and total
    dec_counts = []
    for i, head in enumerate(model.decoders):
        dec_counts.append(count_params(head, trainable_only))
    out['decoders'] = dec_counts
    out['decoders_total'] = sum(dec_counts)

    # 4) Model total (sanity check)
    out['model_total'] = sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)

    return out

# # ---- Example usage ----
# bd = param_breakdown(rnn, trainable_only=True)

# # pretty print
# print(f"latent_proj: {bd['latent_proj']:,}")
# print(f"cond_proj:   {bd['cond_proj']:,}")
# print(f"GRU total:   {bd['gru_total']:,}")
# for k, v in bd['gru_layers'].items():
#     print(f"  {k}: {v:,}")
# print("Decoders (per head):")
# for i, n in enumerate(bd['decoders']):
#     print(f"  decoder[{i}]: {n:,}")
# print(f"Decoders total: {bd['decoders_total']:,}")
# print(f"Model total:    {bd['model_total']:,}")



############################################################################################
##################         DISPLAY utils        #################################
############################################################################################

from pathlib import Path

def load_sidecar(
    basepath: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load a pre-normalized sidecar (.cond.npy) and return a (T, D) tensor.

    Assumes:
      - Values are already in [0,1]
      - No per-file or global JSON metadata is required at load time

    Parameters
    ----------
    basepath : str | Path
        Path WITHOUT extension, e.g.:
        'data/foo/bar' for file:
        'data/foo/bar.cond.npy'

    dtype : torch.dtype
        Output tensor dtype (default: float32)

    Returns
    -------
    torch.Tensor
        Shape (T, D)
    """
    basepath = Path(basepath)
    cond_path = basepath.with_suffix(".cond.npy")

    if not cond_path.exists():
        raise FileNotFoundError(f"Missing sidecar file: {cond_path}")

    arr = np.load(cond_path)

    if arr.ndim != 2:
        raise ValueError(f"{cond_path}: expected 2D array (T, D), got shape {arr.shape}")

    return torch.from_numpy(arr).to(dtype=dtype)
############################################################################################
from transformers import EncodecModel

def read_ecdc_reconstruct_audio(
    ecdcpath: str,
    model: EncodecModel | None = None,
    device: torch.device | str | None = None,
    model_name: str = "facebook/encodec_24khz",
    sample_rate: int = 24000,
    target_bandwidths: list[float] | None = None,  # e.g. [6.0] like your notebook
):
    """
    Read a .ecdc file saved as a torch dict with keys:
      - 'audio_codes'
      - 'audio_scales'
      - 'audio_length'
    and return reconstructed audio (numpy) using ALL stored codebooks.

    Returns:
      audio: np.ndarray float32, shape (N,) for mono or (C, N) for multi-channel (depending on decode)
      sr: int
    """
    saved = torch.load(ecdcpath, map_location="cpu", weights_only=False)

    audio_codes = saved["audio_codes"]          # typically (C, K, T) or (K, T)
    audio_scales = saved.get("audio_scales")    # None / tensor / list of tensors
    audio_length = int(saved["audio_length"])   # used to build padding_mask

    # Build padding mask like your notebook
    padding_mask = torch.zeros(1, audio_length, dtype=torch.bool)

    # Load / configure model (same as notebook)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if model is None:
        model = EncodecModel.from_pretrained(model_name)

    if target_bandwidths is not None:
        model.config.target_bandwidths = list(target_bandwidths)

    model = model.to(device)
    model.eval()

    # --- Normalize audio_codes shape to what decode expects ---
    # Notebook logic assumes you can do: audio_codes[:, :n_codebooks, :].unsqueeze(0)
    # so we want audio_codes shaped like (C, K, T) before adding batch dim.
    if not torch.is_tensor(audio_codes):
        audio_codes = torch.tensor(audio_codes)

    if audio_codes.dim() == 2:
        # (K, T) -> (C=1, K, T)
        audio_codes = audio_codes.unsqueeze(0)
    elif audio_codes.dim() == 3:
        # already (C, K, T)
        pass
    elif audio_codes.dim() == 4:
        # already has batch? (B, C, K, T) -- keep it
        pass
    else:
        raise ValueError(f"Unexpected audio_codes shape: {tuple(audio_codes.shape)}")

    # Infer number of codebooks K (works for (C,K,T) or (B,C,K,T))
    if audio_codes.dim() == 3:
        K = audio_codes.shape[1]
        codes_for_decode = audio_codes[:, :K, :].unsqueeze(0)  # -> (1, C, K, T)
    else:  # dim == 4
        K = audio_codes.shape[2]
        codes_for_decode = audio_codes[:, :, :K, :]           # -> (B, C, K, T)

    codes_for_decode = codes_for_decode.to(device)
    padding_mask = padding_mask.to(device)

    # Move scales to device if present
    if isinstance(audio_scales, list):
        audio_scales_device = [
            s.to(device) if hasattr(s, "to") else s for s in audio_scales
        ]
    else:
        audio_scales_device = audio_scales.to(device) if hasattr(audio_scales, "to") else audio_scales

    # --- Decode ---
    with torch.no_grad():
        audio_values = model.decode(codes_for_decode, audio_scales_device, padding_mask)[0]
        # audio_values is typically (C, N) or (N,)
        audio_np = audio_values.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)

    return audio_np, sample_rate
    

############################################################################################
# for nice audio + param plots 
############################################################################################


import numpy as np
import matplotlib.pyplot as plt

def plot_audio_with_params_two_yaxes(
    trialaudio,
    whole_param_seq,
    audio_sr=24000,
    param_sr=75,
    param_names=None,
    figsize=(14, 4),
    audio_pad_frac=0.08,
    audio_pad_abs=1e-3,
    param_pad_frac=0.05,   # NEW
    title="RNeNcodec parameter-driven synthesis",
    subtitle=""
):
    color1 = '#AAAAAA'
    colors = ['#AA0022', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown',
              'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    if hasattr(whole_param_seq, "detach"):
        whole_param_seq = whole_param_seq.detach().cpu().numpy()

    trialaudio = np.asarray(trialaudio).squeeze()
    whole_param_seq = np.asarray(whole_param_seq)

    N = trialaudio.shape[0]
    T, D = whole_param_seq.shape

    t_audio = np.arange(N) / audio_sr
    t_param = np.arange(T) / param_sr

    fig, ax_audio = plt.subplots(figsize=figsize)

    # ---- Audio axis ----
    ax_audio.plot(t_audio, trialaudio, color=color1, linewidth=0.6, alpha=0.85, label="audio")
    ax_audio.set_xlabel("Time (seconds)")
    ax_audio.set_ylabel("Audio amplitude")
    ax_audio.grid(True, alpha=0.25)

    a_min = float(np.min(trialaudio))
    a_max = float(np.max(trialaudio))
    a_rng = a_max - a_min

    pad = max(audio_pad_abs, audio_pad_frac * (a_rng if a_rng > 0 else 1.0))
    y0 = a_min - pad
    y1 = a_max + pad

    if not np.isfinite(y0) or not np.isfinite(y1) or abs(y1 - y0) < 1e-6:
        y0, y1 = -0.1, 0.1

    ax_audio.set_ylim(y0, y1)

    # ---- Parameter mapping with visual padding ----
    # Parameter *visual* range
    p_lo = -param_pad_frac
    p_hi = 1.0 + param_pad_frac
    p_span = p_hi - p_lo

    y_span = y1 - y0

    # Map param values into audio y-range using padded param space
    params_in_audio_units = y0 + (whole_param_seq - p_lo) / p_span * y_span

    for d in range(D):
        c = colors[d % len(colors)]
        label = param_names[d] if (param_names is not None and d < len(param_names)) else f"param {d}"
        ax_audio.plot(
            t_param,
            params_in_audio_units[:, d],
            color=c,
            linewidth=2.0,
            alpha=0.95,
            label=label
        )

    # ---- Right axis: padded parameter scale ----
    ax_param = ax_audio.twinx()
    ax_param.set_ylabel("Parameter value")

    # Nice ticks in parameter space (still centered on [0,1])
    p_ticks = np.linspace(0, 1, 6)
    y_ticks = y0 + (p_ticks - p_lo) / p_span * y_span

    ax_param.set_ylim(y0, y1)
    ax_param.set_yticks(y_ticks)
    ax_param.set_yticklabels([f"{p:.1f}" for p in p_ticks])

    ax_audio.legend(loc="upper right", frameon=False)

    if title is not None:
        ax_audio.set_title(title + "\n" + subtitle)

    plt.tight_layout()
    plt.show()


####     Simple audio plot

def plot_audio(audio, sr=24000, channel=0, figsize=(14, 4), title="Audio Waveform", subtitle=None):
    """
    Plot audio waveform (amplitude vs time).
    
    Args:
        audio: torch.Tensor or numpy array of shape (channels, samples) or (samples,)
        sample_rate: Sample rate in Hz
        channel: Which channel to plot (default 0 for mono/left)
        figsize: Figure size tuple
    """
    # Convert to numpy if needed
    if hasattr(audio, 'numpy'):
        audio = audio.cpu().numpy()
    
    # Handle different shapes
    if audio.ndim == 1:
        audio_data = audio
    else:
        audio_data = audio[channel]
    
    # Create time axis
    duration = len(audio_data) / sr
    time = np.linspace(0, duration, len(audio_data))

    color1 = '#AA88AA'

    if subtitle is not None :
        title = title + "\n" + subtitle
        
    # Plot
    plt.figure(figsize=figsize)
    plt.plot(time, audio_data, color=color1,)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Create parameter contour with breakpoints for driving synthesis
import torch

def interpolate_breakpoints(breakpoints, frames_per_second):
    """
    Create interpolated tensor from breakpoint lists.
    
    Args:
        breakpoints: List of lists, each containing (time, value) tuples
        frames_per_second: Number of frames per second
    
    Returns:
        torch.Tensor of shape (T, D) where T=frames, D=num_parameters
    """
    # Validate inputs
    if not breakpoints:
        raise ValueError("breakpoints cannot be empty")
    
    D = len(breakpoints)  # Number of parameters
    
    # Check that all parameters have breakpoint at time=0
    for i, param_breakpoints in enumerate(breakpoints):
        if not param_breakpoints or param_breakpoints[0][0] != 0:
            raise ValueError(f"Parameter {i} must have a breakpoint at time=0")
    
    # Find maximum time across all breakpoints
    max_time = 0
    for param_breakpoints in breakpoints:
        for time, _ in param_breakpoints:
            max_time = max(max_time, time)
    
    T = int(frames_per_second * max_time)
    if T == 0:
        T = 1  # At least one frame
    
    # Create time array for each frame
    frame_times = torch.arange(T, dtype=torch.float32) / frames_per_second
    
    # Initialize result tensor
    result = torch.zeros((T, D))
    
    # For each parameter
    for param_idx, param_breakpoints in enumerate(breakpoints):
        # Sort breakpoints by time
        param_breakpoints = sorted(param_breakpoints, key=lambda x: x[0])
        times = [t for t, v in param_breakpoints]
        values = [v for t, v in param_breakpoints]
        
        # Manual linear interpolation
        for frame_idx, frame_time in enumerate(frame_times):
            # Find the right segment
            if frame_time <= times[0]:
                result[frame_idx, param_idx] = values[0]
            elif frame_time >= times[-1]:
                result[frame_idx, param_idx] = values[-1]
            else:
                # Find the two breakpoints to interpolate between
                for i in range(len(times) - 1):
                    if times[i] <= frame_time <= times[i + 1]:
                        # Linear interpolation
                        t0, t1 = times[i], times[i + 1]
                        v0, v1 = values[i], values[i + 1]
                        alpha = (frame_time - t0) / (t1 - t0)
                        result[frame_idx, param_idx] = v0 + alpha * (v1 - v0)
                        break
    
    return result