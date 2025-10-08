#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

try:
    from safetensors.torch import save_file as save_safetensors
    HAVE_ST = True
except Exception:
    HAVE_ST = False

DEFAULT_KEEP_FP32 = ("norm", "bn", "bias", "running_mean", "running_var")

def _tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def _report_size(sd: dict[str, torch.Tensor], label: str) -> None:
    total = sum(_tensor_bytes(v) for v in sd.values() if torch.is_tensor(v))
    print(f"{label}: ~{total/1024/1024:.2f} MB of tensor data")

def _extract_state_dict(obj: object) -> dict:
    """
    Accepts:
      - full training checkpoint dict with 'model_state_dict' or 'model'
      - a raw state_dict (mapping of tensor name -> tensor)
      - a whole-model object saved via torch.save(model)
    Returns a plain state_dict mapping.
    """
    # whole model object?
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        # Might already be a state_dict:
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise SystemExit("Could not find a state_dict in the input. "
                     "Expecting a dict with 'model_state_dict'/'state_dict', "
                     "a whole model, or a raw state_dict.")

def _cast_for_inference(sd: dict[str, torch.Tensor], half: bool,
                        keep_fp32_keywords: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        t = v.detach().cpu()
        if half and t.is_floating_point():
            # keep numerically sensitive params in fp32
            if any(w in k.lower() for w in keep_fp32_keywords):
                t = t.float()
            else:
                t = t.half()
        else:
            # ensure standard fp32 for floats (some people accidentally save fp64)
            if t.is_floating_point():
                t = t.float()
        out[k] = t
    return out

def main():
    ap = argparse.ArgumentParser(
        description="Export a minimal inference checkpoint (weights only).")
    ap.add_argument("input", type=Path, help="Path to full checkpoint (.pt)")
    ap.add_argument("output", type=Path, help="Path to write export (.pt or .safetensors)")
    ap.add_argument("--half", action="store_true",
                    help="Downcast most floating weights to fp16 (keeps norms/bias/etc in fp32).")
    ap.add_argument("--keep-fp32", nargs="*", default=list(DEFAULT_KEEP_FP32),
                    help=f"Extra substrings to keep in fp32 (default: {', '.join(DEFAULT_KEEP_FP32)}).")
    ap.add_argument("--safetensors", action="store_true",
                    help="Write .safetensors (requires safetensors).")
    args = ap.parse_args()

    if args.safetensors and not HAVE_ST:
        sys.exit("safetensors not installed. pip install safetensors or omit --safetensors.")

    print(f"Loading: {args.input}")
    obj = torch.load(str(args.input), map_location="cpu")
    sd = _extract_state_dict(obj)
    _report_size(sd, "Input (as saved)")

    export_sd = _cast_for_inference(sd, half=args.half,
                                    keep_fp32_keywords=tuple(s.lower() for s in args.keep_fp32))
    _report_size(export_sd, "Export (tensors)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.safetensors or args.output.suffix == ".safetensors":
        save_safetensors(export_sd, str(args.output))
    else:
        torch.save(export_sd, str(args.output))
    print(f"✓ Wrote: {args.output}")

if __name__ == "__main__":
    main()
