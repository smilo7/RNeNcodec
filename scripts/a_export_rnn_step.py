import torch
import torch.nn as nn
from pathlib import Path

# You may need to edit these imports to match your project layout:
from rnencodec.model.gru_audio_model import RNN, GRUModelConfig
from transformers import EncodecModel  # HF Encodec

###############################################################################
# 1. Small wrapper module for ONNX
###############################################################################

class RNNStepWrapper(nn.Module):
    """
    Wrap one timestep of your RNN model into a clean, ONNX-friendly forward().
    Inputs:
        latent_in      (1, 128)         float32
        cond_in        (1, cond_size)   float32
        hidden_in      (num_layers, 1, hidden_size) float32
    Outputs:
        logits_out     (n_q, codebook_size) float32
        hidden_out     (num_layers, 1, hidden_size) float32
        step_latent    (1, 128) float32

    Notes:
    - Assumes model.config.cascade == "soft"
      so that forward() is deterministic and returns no sampled_indices.
    - We call model.forward() with:
        use_teacher_forcing=False
        return_step_latent=True
      and we do NOT pass target_codebook_latents.
    """

    def __init__(self, model: RNN):
        super().__init__()
        self.model = model.eval()

        # convenience for sanity checks / shape metadata
        self.n_q = model.config.n_q
        self.codebook_size = model.config.codebook_size
        self.hidden_size = model.config.hidden_size
        self.num_layers = model.config.num_layers
        self.cond_size = model.config.cond_size
        self.input_size = model.config.input_size  # should be 128

        # Optional: assert we're in soft cascade mode for export.
        cascade_mode = getattr(model.config, "cascade", None)
        if cascade_mode != "soft":
            print(f"[WARN] model.config.cascade is {cascade_mode}, "
                  f"expected 'soft' for deterministic ONNX export.")

    def forward(self,
                latent_in: torch.Tensor,        # (1,128)
                cond_in: torch.Tensor,          # (1,cond_size)
                hidden_in: torch.Tensor         # (num_layers,1,hidden_size)
                ):
        """
        Run one step of the GRU.
        We'll hand in latent+cond concatenated, and reuse hidden_in as self.model.hidden.
        We'll grab logits_list, new_hidden, and step_latent from model.forward().
        """

        # The RNN object in training code stores its GRU hidden internally
        # when you call .forward(...). For ONNX we do this manually:
        self.model.hidden = hidden_in

        # Build the per-step input the same way RNNGeneratorSoft did:
        step_in = torch.cat([latent_in, cond_in], dim=-1)  # (1, 128+cond)

        # Forward through your model
        # We request return_step_latent=True so we also get the frame latent.
        with torch.no_grad():
            logits_list, new_hidden, sampled_indices, step_latent = self.model(
                step_in,
                hidden_in,
                use_teacher_forcing=False,
                return_step_latent=True,
            )

        # logits_list is a Python list of length n_q,
        #   each element is shape (1, codebook_size).
        # Stack -> (n_q, codebook_size)
        logits_stacked = torch.stack([logit[0] for logit in logits_list], dim=0)

        # Enforce float32 on outputs that JS will consume
        logits_stacked = logits_stacked.to(torch.float32)
        next_hidden    = new_hidden.to(torch.float32)
        step_latent    = step_latent.to(torch.float32)

        return logits_stacked, next_hidden, step_latent


###############################################################################
# 2. Build the model exactly like you do for inference and wrap it
###############################################################################

def load_trained_rnn(
    checkpoint_path: str,
    model_config: GRUModelConfig,
    encodec_model_name: str = "facebook/encodec_24khz",
    map_location: str | torch.device = "cpu",
) -> RNN:
    """
    Rebuilds your RNN with the correct EncodecModel so it can generate E_eff,
    then loads the trained state_dict.
    """

    # Load Encodec from HF just like you do at runtime so _E_eff is built
    enc = EncodecModel.from_pretrained(encodec_model_name)
    enc = enc.to(map_location)
    enc.eval()

    # Construct the RNN with this enc_model (this builds its codebook table etc.)
    model = RNN(model_config, enc).to(map_location)
    model.eval()

    # Load checkpoint weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(map_location).eval()

    return model


###############################################################################
# 3. Actually export to ONNX
###############################################################################

def export_rnn_step_onnx(
    checkpoint_path: str,
    model_config: GRUModelConfig,
    outfile: str = "rnn_step.onnx",
    encodec_model_name: str = "facebook/encodec_24khz",
    device: str = "cpu",
):
    """
    Creates rnn_step.onnx with clean inputs/outputs for a single timestep.
    """

    dev = torch.device(device)
    model = load_trained_rnn(
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        encodec_model_name=encodec_model_name,
        map_location=dev,
    )

    wrapper = RNNStepWrapper(model).to(dev).eval()

    # ----- Make example (dummy) inputs with correct shapes -----
    # latent_in: (1,128)
    latent_dummy = torch.zeros(1, model.config.input_size, dtype=torch.float32, device=dev)

    # cond_in: (1,cond_size)
    cond_dim = model.config.cond_size
    if cond_dim > 0:
        cond_dummy = torch.zeros(1, cond_dim, dtype=torch.float32, device=dev)
    else:
        # If you ever truly have cond_size = 0, you'd need to adjust the wrapper
        # to not concat, but let's assume cond_size>0 for now.
        raise RuntimeError("cond_size == 0 not currently handled in export dummy.")

    # hidden_in: (num_layers,1,hidden_size)
    hidden_dummy = torch.zeros(
        model.config.num_layers, 1, model.config.hidden_size,
        dtype=torch.float32, device=dev
    )

    # ----- Names for inputs/outputs in the ONNX graph -----
    input_names  = ["latent_in", "cond_in", "hidden_in"]
    output_names = ["logits_out", "hidden_out", "step_latent"]

    # No dynamic axes necessary if we keep batch=1, seq_len=1 fixed.
    # If later you want variable batch or whatever, we can add dynamic_axes.

    torch.onnx.export(
        wrapper,
        (latent_dummy, cond_dummy, hidden_dummy),
        outfile,
        export_params=True,
        opset_version=17,  # good modern default
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,  # all shapes fixed for browser simplicity
    )

    print(f"[OK] Exported {outfile}")
    print(f"  latent_in shape:      {tuple(latent_dummy.shape)}")
    print(f"  cond_in shape:        {tuple(cond_dummy.shape)}")
    print(f"  hidden_in shape:      {tuple(hidden_dummy.shape)}")
    print("Outputs will be:")
    print(f"  logits_out:    (n_q, codebook_size) = "
          f"({model.config.n_q}, {model.config.codebook_size})")
    print(f"  hidden_out:    (num_layers,1,hidden_size) = "
          f"({model.config.num_layers},1,{model.config.hidden_size})")
    print(f"  step_latent:   (1,128)")


###############################################################################
# 4. If you want to run this script directly:
###############################################################################

if __name__ == "__main__":
    # You will fill these in with your actual values.
    #
    # Example:
    #   checkpoint_path = "checkpoints/my_rnn_latest.pt"
    #   model_config = GRUModelConfig(
    #       input_size=128,
    #       cond_size=1,
    #       hidden_size=128,
    #       num_layers=3,
    #       n_q=8,
    #       codebook_size=1024,
    #       dropout=0.1,
    #       inp_proportion=5,
    #       cond_proportion=1,
    #       cascade="soft",          # <-- important: export in soft mode
    #       tau_soft=0.6,
    #       top_n_soft=None,
    #       hard_sample_mode="sample",
    #       top_n_hard=None,
    #       temperature_hard=1.0,
    #       gumbel_tau_start=1.0,
    #       gumbel_tau_end=0.5,
    #       straight_through=True,
    #   )
    #
    # Then run:
    #   python export_rnn_step.py
    #
    checkpoint_path = "/home/lonce/working/RNeNcodec/notebooks/output/20251028_141417_waterfill_softcascade_outsidesamples/checkpoints/last_checkpoint.pt"

    model_config = GRUModelConfig(
        input_size=128,
        cond_size=1,
        hidden_size=128,
        num_layers=3,
        n_q=8,
        codebook_size=1024,
        dropout=0.1,
        inp_proportion=5,
        cond_proportion=1,
        # ---- SOFT cascade defaults for export ----
        # make sure this matches how you trained / want to run in browser
        cascade="soft",
        tau_soft=0.6,
        top_n_soft=None,
        # ---- the "hard" knobs stay in the config but won't matter for export ----
        hard_sample_mode="sample",
        top_n_hard=None,
        temperature_hard=1.0,
        gumbel_tau_start=1.0,
        gumbel_tau_end=0.5,
        straight_through=True,
    )

    export_rnn_step_onnx(
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        outfile="rnn_step.onnx",
        encodec_model_name="facebook/encodec_24khz",
        device="cpu",
    )
