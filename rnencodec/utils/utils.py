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

