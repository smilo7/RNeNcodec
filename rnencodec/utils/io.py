# Config I/O (no pickled classes)

from dataclasses import is_dataclass, asdict
from pathlib import Path
import json
import torch

def _as_plain_dict(x):
    if is_dataclass(x): return asdict(x)
    if isinstance(x, dict): return x
    raise TypeError("config must be a dataclass or dict")

def save_run_config(path, *, params: dict, model_config, data_config, write_json_sidecar=True):
    """
    Save configs as plain data (refactor-proof).
    - params: dict (may include tensors)
    - model_config, data_config: dataclass OR dict
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "run_config_v2",
        "params": params,
        "model_config": _as_plain_dict(model_config),
        "data_config": _as_plain_dict(data_config),
    }
    torch.save(payload, p)
    print(f'saved to {p}')

    if write_json_sidecar:
        def _jsonify(v):
            if isinstance(v, torch.Tensor): return v.tolist()
            if isinstance(v, dict): return {k: _jsonify(x) for k, x in v.items()}
            return v
        with open(p.with_suffix(".json"), "w") as f:
            json.dump(_jsonify(payload), f, indent=2)
        print(f'wrote json param file')

def load_run_config(path):
    """
    Load plain-data configs (returns dicts).
    You reconstruct your config classes yourself:
        GRUModelConfig(**cfg['model_config'])
    """
    return torch.load(Path(path), map_location="cpu")