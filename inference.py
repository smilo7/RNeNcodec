import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import soundfile as sf

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import soundfile as sf
import matplotlib.pyplot as plt

# Local imports from your project structure
from model.gru_audio_model import RNN, GRUModelConfig
from audioDataLoader.audio_dataset import latents_to_audio_simple, efficient_codes_to_latents, preprocess_latents_for_RNN

import time

# def transform_outputs_to_inputs(logits_list, encodec_model, clamp_val, top_n=3, temperature=1.0, codebook_size=1024, n_q=4):
#     """
#     Transform model outputs (logits) into the next input (128D latent).
    
#     Args:
#         logits_list: List of logit tensors, one per quantizer
#         encodec_model: EnCodec model for code->latent conversion
#         clamp_val (float) - clamp latents (produced by encodec token decoding) in [-clamp_val, clampval], the map to [-1,1] for input to model next step
#         top_n: Number of top predictions to sample from
#         temperature: Sampling temperature
#         codebook_size: Size of each codebook
#         n_q: Number of quantizers
    
#     Returns:
#         torch.Tensor: Next input latent of shape (1, 128)
#     """
#     device = logits_list[0].device
#     encodec_model.to(device)
#     sampled_codes = []
    
#     for j in range(n_q):
#         # Apply temperature and get top-k
#         logits_j = logits_list[j].div(temperature).squeeze()  # (codebook_size,)
#         top_n_logits, top_n_indices = torch.topk(logits_j, top_n)
#         top_n_probs = F.softmax(top_n_logits, dim=-1)
        
#         # Sample from top-k
#         try:
#             sampled_relative_idx = torch.multinomial(top_n_probs, 1).squeeze()
#             sampled_code = top_n_indices[sampled_relative_idx]
#             sampled_codes.append(sampled_code.item())
#         except Exception as e:
#             print(f"Sampling error for quantizer {j}: {e}")
#             # Fallback to random sampling
#             sampled_codes.append(torch.randint(0, codebook_size, (1,)).item())
    
#     # Convert sampled codes back to latent - CREATE TENSOR ON CORRECT DEVICE
#     codes_tensor = torch.tensor(sampled_codes, device=device).unsqueeze(0).unsqueeze(-1)  # (1, n_q, 1)
#     next_latent = efficient_codes_to_latents(encodec_model, codes_tensor).squeeze(0).squeeze(-1).unsqueeze(0)  # (1, 128)
#     next_latent = preprocess_latents_for_RNN(next_latent, clamp_val)
    
#     return next_latent, sampled_codes


def run_inference(model, encodec_model, cond_seq, warmup_latents, clamp_val, top_n=3, temperature=1.0, include_warmup_audio=False) :
    """
    Generates audio sequence based on conditioning sequence using EnCodec latents.

    Args:
        model: The trained RNNLatent model.
        encodec_model: The EnCodec model for latent/audio conversion.
        cond_seq (torch.Tensor): The sequence of conditioning parameters. Shape: (seq_len, cond_size) or None if no conditioning.
        warmup_latents (torch.Tensor): Latent sequence to warm up the model's hidden state. Shape: (warmup_len, 128), in [-1,1]
        clamp_val (float) - clamp latents (produced by encodec token decoding) in [-clamp_val, clampval], the map to [-1,1] for input to model next step
        top_n (int): The number of top predictions to sample from.
        temperature (float): Controls the randomness of predictions. Higher is more random.

    Returns:
        np.array: The generated audio waveform.
        int: Sample rate
    """
    device = next(model.parameters()).device
    codebook_size = model.config.codebook_size
    n_q = model.config.n_q
    cond_size = model.config.cond_size
    
    print(f"Starting latent inference... with codebook_size={codebook_size}, n_q={n_q}, cond_size={cond_size}")

    # --- 1. Warm-up Phase ---
    print("Warming up model hidden state with provided latents...")
    
    # Ensure warmup_latents is on the right device and has correct shape
    warmup_latents = warmup_latents.to(device)  # Shape: (warmup_len, 128)
    warmup_len = warmup_latents.shape[0]
    
    # Handle conditioning
    if cond_size > 0 and cond_seq is not None:
        # Use first conditioning vector for entire warmup
        first_cond_vec = cond_seq[0].unsqueeze(0).repeat(warmup_len, 1).to(device)  # (warmup_len, cond_size)
        warmup_full_input = torch.cat([warmup_latents, first_cond_vec], dim=-1)  # (warmup_len, 128 + cond_size)
    else:
        # No conditioning
        warmup_full_input = warmup_latents  # (warmup_len, 128)

    hidden = model.init_hidden(batch_size=1)
    for i in range(len(warmup_full_input)):
        #_, hidden = model(warmup_full_input[i].unsqueeze(0), hidden,  batch_size=1)
        _, hidden, _, _ = model(
            warmup_full_input[i].unsqueeze(0), 
            hidden,  
            batch_size=1)
        

    # Get the last latent for starting generation
    current_latent = warmup_latents[-1].unsqueeze(0)  # (1, 128)

    # --- 2. Generation Phase ---
    generation_length = len(cond_seq) if cond_seq is not None else 150  # Default length if no conditioning
    print(f"Generating {generation_length} latent frames...")
    generated_codes = []

    start_time = time.monotonic()
    
    with torch.no_grad():
        model.eval()
    
        #-------------------------------------------------------------------------------
        for i in range(generation_length):
            # Build input (latent + cond)
            if cond_size > 0 and cond_seq is not None:
                current_cond_vec = cond_seq[i].unsqueeze(0).to(device)      # (1, cond_size)
                next_input_full   = torch.cat([current_latent, current_cond_vec], dim=-1)  # (1, 128 + cond)
            else:
                next_input_full   = current_latent                          # (1, 128)
    
            # One call -> one set of tokens (no resampling)
            logits_list, hidden, sampled_indices, step_latent = model(
                next_input_full,
                hidden,
                use_teacher_forcing=False,
                temperature=temperature,
                batch_size=1,
                sample_mode="sample" if top_n and top_n > 0 else "argmax",  # keep your top-k/temperature behavior
                top_n=top_n,
                return_step_latent=True
            )
    
            # Use the exact latent produced by those sampled tokens
            current_latent = preprocess_latents_for_RNN(step_latent, clamp_val)  # (1, 128)
    
            # Save the exact tokens used this step
            generated_codes.append(sampled_indices.squeeze(0).tolist())  # (n_q,)
    #-------------------------------------------------------------------------------

    rnnelapsed_time = time.monotonic() - start_time
    print(f"Latent generation complete. RNN time to generate: {rnnelapsed_time:.2f}. Converting to audio...")

    # Convert generated codes to audio
    generated_codes_tensor = torch.tensor(generated_codes).transpose(0, 1).unsqueeze(0).to(device)  # (1, n_q, seq_len)
    generated_latents = efficient_codes_to_latents(encodec_model, generated_codes_tensor)  # (1, 128, seq_len)
    generated_audio, sample_rate = latents_to_audio_simple(encodec_model, generated_latents)

    if include_warmup_audio:
        # Convert warmup latents to audio
        warmup_latents_for_audio = warmup_latents.unsqueeze(0).transpose(1, 2).to(device)  # (1, 128, warmup_len)
        warmup_audio, _ = latents_to_audio_simple(encodec_model, warmup_latents_for_audio*clamp_val)
        
        # Concatenate warmup + generated audio
        full_audio = torch.cat([warmup_audio, generated_audio], dim=-1)
    else:
        full_audio = generated_audio

    encelapsed_time = time.monotonic() - start_time - rnnelapsed_time
    print(f"Inference complete - encoding time = {encelapsed_time:.2f}.")
    return full_audio.squeeze().cpu().numpy(), sample_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate audio from a trained EnCodec latent GRU model.')
    parser.add_argument('--run_directory', type=str, required=True, help='Path to the directory of the saved run.')
    parser.add_argument('--encodec_model_path', type=str, default='facebook/encodec_24khz', help='Path/name of EnCodec model.')
    parser.add_argument('--top_n', type=int, default=5, help='Sample from the top N most likely outputs.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Controls the randomness of predictions.')
    parser.add_argument('--length_seconds', type=float, default=2.0, help='Length of the audio to generate in seconds.')
    parser.add_argument('--output_wav_path', type=str, default='generated_audio.wav', help='Path to save the output WAV file.')
    parser.add_argument('--output_plot_path', type=str, default='generated_waveform.png', help='Path to save the output plot.')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    config_path = os.path.join(args.run_directory, "config.pt")
    checkpoint_path = os.path.join(args.run_directory, "checkpoints", "last_checkpoint.pt")

    assert os.path.exists(args.run_directory), f"Run directory not found: {args.run_directory}"
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    assert os.path.exists(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"

    saved_configs = torch.load(config_path)
    model_config = saved_configs["model_config"]

    model = RNNLatent(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load EnCodec model
    from transformers import EncodecModel
    encodec_model = EncodecModel.from_pretrained(args.encodec_model_path).to(device)
    encodec_model.eval()

    print("Models successfully loaded from checkpoint.")

    # Generate conditioning sequence
    frame_rate = 75  # EnCodec frames per second at 24kHz
    generation_length_frames = int(args.length_seconds * frame_rate)

    num_cond_params = model_config.cond_size
    cond_size = model_config.cond_size
    if cond_size > 0:
        cond_seq = torch.zeros(generation_length_frames, cond_size)
        # Only set values if dimensions exist
        if cond_size >= 1:
            cond_seq[:, 0] = 0.0  # first parameter
        if cond_size >= 2:
            cond_seq[:, 1] = 0.8  # second parameter
        if cond_size >= 3:
            cond_seq[:, 2] = torch.linspace(0, 1, generation_length_frames)  # third parameter sweep
        # Additional parameters remain zero
    else:
        cond_seq = None

        
    # Create warmup latents
    warmup_frames = 50
    warmup_latents = torch.randn(warmup_frames, 128)

    generated_audio, sample_rate = run_inference(
        model=model,
        encodec_model=encodec_model,
        cond_seq=cond_seq,
        warmup_latents=warmup_latents,
        top_n=args.top_n,
        temperature=args.temperature
    )

    generated_audio, sample_rate = run_inference(
        model=model,
        encodec_model=encodec_model,
        cond_seq=cond_seq,
        warmup_codes=warmup_codes,
        top_n=args.top_n,
        temperature=args.temperature
    )

    print(f"Saving generated audio to {args.output_wav_path}")
    sf.write(args.output_wav_path, generated_audio, sample_rate)

    print(f"Saving waveform plot to {args.output_plot_path}")
    plt.figure(figsize=(20, 5))
    plt.plot(generated_audio)
    plt.title("Generated Audio Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.savefig(args.output_plot_path)
    plt.close()

    print("Done.")