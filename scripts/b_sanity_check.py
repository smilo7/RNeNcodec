#!/usr/bin/env python3
import os
import torch
import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------
# USER EDIT THESE
# ---------------------------------------------------------------------

checkpoint_path = "/home/lonce/working/RNeNcodec/notebooks/output/20251028_141417_waterfill_softcascade_outsidesamples/checkpoints/last_checkpoint.pt"
onnx_path       = "./rnn_step.onnx"
encodec_model_name = "facebook/encodec_24khz"

# Your model config must match how you trained.
# Fill in the correct numbers.
from dataclasses import dataclass
from typing import Literal, Optional

CascadeMode = Literal["hard", "soft"]
HardSampleMode = Literal["argmax","gumbel","sample"]

@dataclass
class GRUModelConfig:
    input_size: int = 128          # latent dim
    cond_size: int = 1             # how many conditioning scalars per frame
    hidden_size: int = 128         # GRU hidden dim
    num_layers: int = 3            # GRU layers
    n_q: int = 8                   # codebooks
    codebook_size: int = 1024      # entries per codebook
    dropout: float = 0.1
    inp_proportion: int = 5
    cond_proportion: int = 1

    # cascade / sampling knobs (must match training export mode)
    cascade: CascadeMode = "soft"          # <-- IMPORTANT for ONNX export path
    hard_sample_mode: HardSampleMode = "sample"
    top_n_hard: Optional[int] = None
    temperature_hard: float = 1.0
    tau_soft: float = 0.6
    top_n_soft: Optional[int] = None
    gumbel_tau_start: float = 1.0
    gumbel_tau_end: float = 0.5
    straight_through: bool = True

model_config = GRUModelConfig(
    input_size=128,
    cond_size=1,
    hidden_size=128,
    num_layers=3,
    n_q=8,
    codebook_size=1024,
    dropout=0.1,
    cascade="soft",        # MUST be "soft" for ONNX export/check
)

# ---------------------------------------------------------------------
# Imports of your model code
# ---------------------------------------------------------------------

# rnencodec-style imports (adjust if paths differ in your repo)
from transformers import EncodecModel
from rnencodec.model.gru_audio_model import RNN


# This MUST match what you used in export_rnn_step.py (or whatever you named it)
class RNNExportWrapper(torch.nn.Module):
    """
    Thin wrapper used for ONNX export:
    - takes (latent_in, cond_in, hidden_in)
    - runs ONE step of the RNN in cascade='soft' mode
    - returns:
        logits_cat: (n_q, K)
        hidden_out: (num_layers, 1, hidden_size)
        step_latent: (1, 128)
    """
    def __init__(self, rnn_model: RNN):
        super().__init__()
        self.rnn = rnn_model

    def forward(self, latent_in, cond_in, hidden_in):
        """
        latent_in: (1, 128)
        cond_in:   (1, cond_size)
        hidden_in: (num_layers, 1, hidden_size)
        """
        # concatenate latent + cond to match how you normally call forward()
        step_in = torch.cat([latent_in, cond_in], dim=-1)  # (1, 128+cond)

        # run one step of the model
        logits_list, hidden_out, sampled_indices, step_latent = self.rnn(
            step_in,
            hidden_in,
            use_teacher_forcing=False,
            return_step_latent=True,
        )

        # logits_list is list length n_q, each shape (1, K)
        # stack to (n_q, K)
        logits_cat = torch.stack([lq[0] for lq in logits_list], dim=0)

        return logits_cat, hidden_out, step_latent


def main():
    # -----------------------------------------------------------------
    # 1. Load Encodec and your trained RNN weights (PyTorch)
    # -----------------------------------------------------------------
    print("[check] Loading Encodec + RNN checkpoint (PyTorch)…")
    device = torch.device("cpu")  # sanity check on CPU
    enc = EncodecModel.from_pretrained(encodec_model_name).to(device)
    enc.eval()

    rnn_model = RNN(model_config, enc).to(device)
    rnn_model.eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = rnn_model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] Missing keys when loading checkpoint:", missing)
    if unexpected:
        print("[warn] Unexpected keys when loading checkpoint:", unexpected)

    # Force eval() and no grad
    rnn_model.eval()
    for p in rnn_model.parameters():
        p.requires_grad_(False)

    # -----------------------------------------------------------------
    # 2. Build wrapper module exactly like export time
    # -----------------------------------------------------------------
    wrapper = RNNExportWrapper(rnn_model).to(device)
    wrapper.eval()

    # -----------------------------------------------------------------
    # 3. Create dummy inputs for a single step
    #    Use real sizes from config
    # -----------------------------------------------------------------
    B = 1
    latent_in = torch.randn(B, model_config.input_size,  device=device, dtype=torch.float32)
    cond_in   = torch.randn(B, model_config.cond_size,    device=device, dtype=torch.float32)

    # hidden shape: (num_layers, B, hidden_size)
    hidden_in = torch.randn(
        model_config.num_layers,
        B,
        model_config.hidden_size,
        device=device,
        dtype=torch.float32,
    )

    print("[check] Input shapes:")
    print(" latent_in:", tuple(latent_in.shape))
    print(" cond_in:  ", tuple(cond_in.shape))
    print(" hidden_in:", tuple(hidden_in.shape))

    # -----------------------------------------------------------------
    # 4. Run PyTorch wrapper forward
    # -----------------------------------------------------------------
    with torch.no_grad():
        logits_pt, hidden_pt, step_latent_pt = wrapper(latent_in, cond_in, hidden_in)

    print("[check] PyTorch outputs:")
    print(" logits_pt:", tuple(logits_pt.shape), " (should be (n_q, codebook_size))")
    print(" hidden_pt:", tuple(hidden_pt.shape), " (should be (num_layers,1,hidden_size))")
    print(" step_latent_pt:", tuple(step_latent_pt.shape), " (should be (1,128))")

    # -----------------------------------------------------------------
    # 5. Run ONNX session on same inputs
    #    NOTE: requires 'pip install onnxruntime'
    # -----------------------------------------------------------------
    print("[check] Loading ONNX InferenceSession…")
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    # ONNX expects plain numpy, float32
    latent_np = latent_in.cpu().numpy().astype(np.float32)
    cond_np   = cond_in.cpu().numpy().astype(np.float32)
    hidden_np = hidden_in.cpu().numpy().astype(np.float32)

    # The input names MUST match what you used in export
    ort_inputs = {
        "latent_in": latent_np,
        "cond_in": cond_np,
        "hidden_in": hidden_np,
    }

    ort_outs = sess.run(
        ["logits_out", "hidden_out", "step_latent"],
        ort_inputs,
    )
    logits_onnx, hidden_onnx, step_latent_onnx = ort_outs

    print("[check] ONNX outputs:")
    print(" logits_onnx:", tuple(logits_onnx.shape))
    print(" hidden_onnx:", tuple(hidden_onnx.shape))
    print(" step_latent_onnx:", tuple(step_latent_onnx.shape))

    # -----------------------------------------------------------------
    # 6. Compare numerically
    # -----------------------------------------------------------------
    def max_abs_diff(a, b):
        a_np = a if isinstance(a, np.ndarray) else a.detach().cpu().numpy()
        b_np = b if isinstance(b, np.ndarray) else b.detach().cpu().numpy()
        return float(np.max(np.abs(a_np - b_np)))

    d_logits = max_abs_diff(logits_pt, logits_onnx)
    d_hidden = max_abs_diff(hidden_pt, hidden_onnx)
    d_latent = max_abs_diff(step_latent_pt, step_latent_onnx)

    print("\n[check] max |PyTorch - ONNX| diffs:")
    print(f" logits: {d_logits:.6f}")
    print(f" hidden: {d_hidden:.6f}")
    print(f" step_latent: {d_latent:.6f}")

    print("\nIf these diffs are on the order of 1e-5 or so, you're golden.")
    print("If they're huge (like >1e-2), something in export vs wrapper doesn't match.")


if __name__ == "__main__":
    main()
