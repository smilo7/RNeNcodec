import torch
import torch.nn as nn
from transformers import EncodecModel


class EncodecDecodeWrapper(nn.Module):
    """
    Wraps just the decode() path of Encodec so we can export to ONNX.

    Forward signature:
        codes_bnt : (1, n_q, T)  int64 or int32
    Returns:
        audio_out : (1, T_audio) float32  (mono waveform)

    Notes:
    - We inline the minimal logic to match what you currently do by hand:
        enc_model.decode([codes_bnt], audio_scales=[None])[0]
      and then squeeze batch/channel dims.
    """

    def __init__(self, enc_model: EncodecModel):
        super().__init__()
        # We keep the full model inside, but in forward() we only call decode()
        self.enc_model = enc_model
        self.sample_rate = int(getattr(enc_model, "sample_rate", 24000))

    def forward(self, codes_bnt: torch.Tensor) -> torch.Tensor:
        """
        codes_bnt: LongTensor (1, n_q, T) – batch=1, mono encodec24 style
        returns:  (1, T_audio) float32
        """
        # Hugging Face Encodec expects a *list* of code tensors, each shaped (B, n_q, T)
        # and audio_scales list. We'll just wrap/unwrap.
        decoded_list = self.enc_model.decode([codes_bnt], audio_scales=[None])
        # decoded_list[0] should be shape (B, 1, T_audio) or (B, T_audio)
        audio = decoded_list[0]

        if audio.dim() == 3:
            # (B, C, T) → assume mono C=1
            audio = audio[:, 0, :]
        elif audio.dim() == 2:
            # already (B, T)
            pass
        else:
            raise RuntimeError(f"Unexpected decoded audio shape {tuple(audio.shape)}")

        # ensure float32
        return audio.to(dtype=torch.float32)


@torch.no_grad()
def export_encodec_decoder_onnx(
    encodec_model_name: str,
    out_onnx: str,
    n_q: int = 8,
    T_frames: int = 64,
):
    """
    1. Load HF Encodec on CPU
    2. Wrap decode() in EncodecDecodeWrapper
    3. Export ONNX with dummy (1,n_q,T_frames) input

    encodec_model_name: e.g. "facebook/encodec_24khz"
    out_onnx:           path to write ONNX model (e.g. "encodec_decode.onnx")
    n_q:                number of codebooks (8 for Encodec 24kHz)
    T_frames:           dummy frame length for export. We'll mark T as dynamic.

    Returns:
        sample_rate for reference.
    """
    device = torch.device("cpu")
    print(f"[export-decode] loading {encodec_model_name} on {device} …")
    enc = EncodecModel.from_pretrained(encodec_model_name).to(device)
    enc.eval()

    wrapper = EncodecDecodeWrapper(enc).to(device).eval()

    # dummy input: (1, n_q, T_frames)
    dummy_codes = torch.randint(
        low=0,
        high=int(enc.config.codebook_size),
        size=(1, n_q, T_frames),
        dtype=torch.long,
        device=device,
    )

    # ONNX export
    print(f"[export-decode] exporting to {out_onnx} …")
    torch.onnx.export(
        wrapper,
        dummy_codes,
        out_onnx,
        input_names=["codes_bnt"],
        output_names=["audio_out"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "codes_bnt": {2: "T_frames"},   # allow varying T at runtime
            "audio_out": {1: "T_audio"}     # decoded waveform length changes with T
        },
    )

    sr = int(getattr(enc, "sample_rate", 24000))
    print(f"[export-decode] done. sample_rate={sr}  n_q={n_q}")
    print(f"  dummy_codes: {tuple(dummy_codes.shape)}  -> audio_out[0] shape depends on Encodec hop.")
    return sr


if __name__ == "__main__":
    # tweak these as you wish:
    encodec_model_name = "facebook/encodec_24khz"
    out_onnx = "encodec_decode.onnx"
    n_q = 8
    T_frames = 64  # doesn't matter much, just a representative chunk

    sr = export_encodec_decoder_onnx(
        encodec_model_name=encodec_model_name,
        out_onnx=out_onnx,
        n_q=n_q,
        T_frames=T_frames,
    )
    print(f"[export-decode] Wrote {out_onnx} with sr={sr} Hz.")
