import torch
import numpy as np
import onnxruntime as ort
from transformers import EncodecModel
from c_export_encodec_decoder import EncodecDecodeWrapper  # from the file above


def main():
    encodec_model_name = "facebook/encodec_24khz"
    onnx_path = "encodec_decode.onnx"
    n_q = 8
    T_frames = 50  # arbitrary test length

    device = torch.device("cpu")

    # Load torch ref
    enc = EncodecModel.from_pretrained(encodec_model_name).to(device).eval()
    wrap = EncodecDecodeWrapper(enc).to(device).eval()

    # Make random fake codes
    codes = torch.randint(
        low=0,
        high=int(enc.config.codebook_size),
        size=(1, n_q, T_frames),
        dtype=torch.long,
        device=device,
    )

    # Torch reference
    with torch.no_grad():
        audio_pt = wrap(codes)  # (1, T_audio)
    print("[check-decode] torch audio:", tuple(audio_pt.shape))

    # ONNX run
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    ort_outs = sess.run(
        ["audio_out"],
        {"codes_bnt": codes.cpu().numpy().astype(np.int64)},
    )
    audio_onnx = ort_outs[0]  # (1, T_audio)
    print("[check-decode] onnx audio:", audio_onnx.shape)

    # Compare a short slice
    min_len = min(audio_pt.shape[1], audio_onnx.shape[1])
    diff = np.max(np.abs(audio_pt.cpu().numpy()[0, :min_len] - audio_onnx[0, :min_len]))
    print(f"[check-decode] max|pt - onnx| over first {min_len} samples:", diff)


if __name__ == "__main__":
    main()
