import torch
import torch.nn.functional as F
import numpy as np
import time
from collections import deque
from audioDataLoader.audio_dataset import latents_to_audio_simple, efficient_codes_to_latents, preprocess_latents_for_RNN


def transform_outputs_to_inputs(logits_list, encodec_model, clamp_val, top_n=3, temperature=1.0, codebook_size=1024, n_q=4):
    """
    Transform model outputs (logits) into the next input (128D latent).
    
    Args:
        logits_list: List of logit tensors, one per quantizer
        encodec_model: EnCodec model for code->latent conversion
        clamp_val (float): clamp latents in [-clamp_val, clamp_val], then map to [-1,1] for model input
        top_n: Number of top predictions to sample from
        temperature: Sampling temperature
        codebook_size: Size of each codebook
        n_q: Number of quantizers
    
    Returns:
        torch.Tensor: Next input latent of shape (1, 128)
    """
    device = logits_list[0].device
    sampled_codes = []
    
    for j in range(n_q):
        # Apply temperature and get top-k
        logits_j = logits_list[j].div(temperature).squeeze()  # (codebook_size,)
        top_n_logits, top_n_indices = torch.topk(logits_j, top_n)
        top_n_probs = F.softmax(top_n_logits, dim=-1)
        
        # Sample from top-k
        try:
            sampled_relative_idx = torch.multinomial(top_n_probs, 1).squeeze()
            sampled_code = top_n_indices[sampled_relative_idx]
            sampled_codes.append(sampled_code.item())
        except Exception as e:
            print(f"Sampling error for quantizer {j}: {e}")
            # Fallback to random sampling
            sampled_codes.append(torch.randint(0, codebook_size, (1,)).item())
    
    # Convert sampled codes back to latent - CREATE TENSOR ON CORRECT DEVICE
    codes_tensor = torch.tensor(sampled_codes, device=device).unsqueeze(0).unsqueeze(-1)  # (1, n_q, 1)
    next_latent = efficient_codes_to_latents(encodec_model, codes_tensor).squeeze(0).squeeze(-1).unsqueeze(0)  # (1, 128)
    next_latent = preprocess_latents_for_RNN(next_latent, clamp_val)
    
    return next_latent, sampled_codes


class RealTimeLatentSynthesizer:
    """
    Real-time audio synthesizer using RNN latent generation with sliding window approach.
    
    The synthesizer maintains a sliding window of latent frames and generates audio
    on-demand using overlapping EnCodec decoding windows.
    """
    
    def __init__(self, model, encodec_model, buffer_size, sample_rate=24000, 
                 latent_window_size=100, clamp_val=15.0):
        """
        Initialize the real-time synthesizer.
        
        Args:
            model: Trained RNN model
            encodec_model: EnCodec model for latent->audio conversion
            buffer_size (int): Number of audio samples to return per generate() call
            sample_rate (int): Audio sample rate (24000 for EnCodec 24kHz)
            latent_window_size (int): Size of sliding latent frame window for EnCodec context
            clamp_val (float): Clamp value for latent preprocessing
        """
        self.model = model
        self.encodec_model = encodec_model
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.latent_window_size = latent_window_size
        self.clamp_val = clamp_val
        
        # EnCodec frame rate (75 fps at 24kHz)
        self.frame_rate = 75
        self.samples_per_frame = sample_rate // self.frame_rate  # ~320 samples per frame
        
        # Model configuration
        self.codebook_size = model.config.codebook_size
        self.n_q = model.config.n_q
        self.cond_size = model.config.cond_size
        self.device = next(model.parameters()).device
        
        # State variables
        self.hidden_state = None
        self.latent_buffer = deque(maxlen=latent_window_size)  # Sliding window of latent frames
        self.current_audio_position = 0  # Track position in audio timeline (in samples)
        self.frames_generated_post_warmup = 0  # Track frames generated AFTER warmup
        
        # Audio caching
        self.last_decoded_audio = None  # Cache last EnCodec decode result
        self.last_decode_start_frame = -1  # Track which frame the cached audio starts from
        
        # Conditioning parameters (can be updated externally)
        self.conditioning_params = torch.zeros(self.cond_size) if self.cond_size > 0 else None
        
        # Performance tracking
        self.generation_times = deque(maxlen=100)
        
        print(f"Synthesizer initialized:")
        print(f"  Buffer size: {buffer_size} samples ({buffer_size/sample_rate*1000:.1f}ms)")
        print(f"  Latent window: {latent_window_size} frames ({latent_window_size/self.frame_rate*1000:.1f}ms)")
        print(f"  Samples per frame: {self.samples_per_frame}")
        
        # Initialize with warmup
        self._initialize_warmup()
    
    def _initialize_warmup(self):
        """Initialize the synthesizer. Warmup frames must equal latent_window_size for proper timing."""
        warmup_frames = self.latent_window_size
        print(f"Initializing with {warmup_frames} warmup frames (= latent_window_size)...")
        
        # Generate random warmup latents in [-1,1] range (already preprocessed)
        warmup_latents = torch.clamp(torch.randn(warmup_frames, 128), -3, 3) / 3  # Normal to [-1,1]
        warmup_latents = warmup_latents.to(self.device)
        
        # Initialize hidden state
        self.hidden_state = self.model.init_hidden(batch_size=1)
        
        # Run warmup through the model and fill the latent buffer
        with torch.no_grad():
            for i in range(warmup_frames):
                current_latent = warmup_latents[i].unsqueeze(0)  # (1, 128)
                
                # Create input with conditioning
                if self.cond_size > 0:
                    cond_input = self.conditioning_params.unsqueeze(0).to(self.device)
                    full_input = torch.cat([current_latent, cond_input], dim=-1)
                else:
                    full_input = current_latent
                
                # Update hidden state and store latent for EnCodec context
                _, self.hidden_state = self.model(full_input, self.hidden_state, batch_size=1)
                self.latent_buffer.append(current_latent.squeeze(0).cpu())
        
        # After warmup, frames_generated_post_warmup starts at 0
        self.frames_generated_post_warmup = 0
        print(f"Warmup complete. Ready to generate new frames.")
    
    def set_conditioning_params(self, **kwargs):
        """Update conditioning parameters. Accepts param names as kwargs."""
        if self.cond_size == 0:
            return
        
        # Map parameter names to indices based on your parameter_specs order
        param_names = list(kwargs.keys())
        for i, (param_name, value) in enumerate(kwargs.items()):
            if i < self.cond_size:
                self.conditioning_params[i] = float(value)
        
        print(f"Updated conditioning: {dict(zip(param_names, self.conditioning_params[:len(param_names)]))}")
    
    def _generate_next_frame(self):
        """Generate the next latent frame using the RNN."""
        start_time = time.monotonic()
        
        with torch.no_grad():
            # Get current latent (last in buffer)
            current_latent = self.latent_buffer[-1].unsqueeze(0).to(self.device)  # (1, 128)
            
            # Create input with current conditioning
            if self.cond_size > 0:
                cond_input = self.conditioning_params.unsqueeze(0).to(self.device)
                full_input = torch.cat([current_latent, cond_input], dim=-1)
            else:
                full_input = current_latent
            
            # Forward pass
            logits_list, self.hidden_state = self.model(full_input, self.hidden_state, batch_size=1)
            
            # Sample codes and convert back to latent
            next_latent, sampled_codes = self._sample_and_convert(logits_list)
            
            # Add to buffer (store on CPU)
            self.latent_buffer.append(next_latent.squeeze(0).cpu())
            self.frames_generated_post_warmup += 1
        
        elapsed = time.monotonic() - start_time
        self.generation_times.append(elapsed)
        
        return sampled_codes
    
    def _sample_and_convert(self, logits_list, top_n=3, temperature=1.0):
        """Sample codes from logits and convert back to latent."""
        return transform_outputs_to_inputs(
            logits_list, self.encodec_model, self.clamp_val, top_n, temperature, 
            self.codebook_size, self.n_q
        )
    
    def _decode_latent_window_to_audio(self):
        """Decode current latent window to audio."""
        # Stack latent buffer into tensor
        latent_window = torch.stack(list(self.latent_buffer))  # (window_size, 128)
        latent_window = latent_window.unsqueeze(0).transpose(1, 2).to(self.device)  # (1, 128, window_size)
        
        # Decode to audio
        audio_tensor, _ = latents_to_audio_simple(self.encodec_model, latent_window)
        return audio_tensor.squeeze().cpu().numpy()
    
    def generate(self, requested_samples=None):
        """
        Generate the next buffer of audio samples with precise temporal alignment.
        
        Args:
            requested_samples: Number of samples requested (uses self.buffer_size if None)
            
        Returns:
            np.array: Audio buffer of requested length
        """
        if requested_samples is None:
            requested_samples = self.buffer_size
        
        # Calculate what frame position this request corresponds to
        request_start_frame = self.current_audio_position // self.samples_per_frame
        request_end_frame = (self.current_audio_position + requested_samples) // self.samples_per_frame
        
        # Ensure RNN has generated enough POST-WARMUP frames to cover this request + margin
        required_post_warmup_frames = request_end_frame + 2  # 2-frame safety margin
        while self.frames_generated_post_warmup < required_post_warmup_frames:
            self._generate_next_frame()
        
        # The decode window is always the most recent latent_window_size frames
        # This includes warmup frames + generated frames, but we track them separately
        total_frames = self.latent_window_size + self.frames_generated_post_warmup
        decode_start_frame = total_frames - self.latent_window_size
        decode_end_frame = total_frames
        
        # Check if we need to decode new audio
        if (self.last_decoded_audio is None or 
            decode_start_frame != self.last_decode_start_frame):
            
            # Decode current latent window to audio
            self.last_decoded_audio = self._decode_latent_window_to_audio()
            self.last_decode_start_frame = decode_start_frame
        
        # Calculate precise sample extraction from decoded audio
        # The decoded audio corresponds to frames [decode_start_frame : decode_end_frame]
        # We want samples corresponding to [current_audio_position : current_audio_position + requested_samples]
        
        # Frame offset within the decoded window
        request_frame_offset = request_start_frame - decode_start_frame
        sample_offset_in_decode = max(0, request_frame_offset * self.samples_per_frame)
        
        # Add the sub-frame sample offset
        sub_frame_offset = self.current_audio_position % self.samples_per_frame
        sample_offset_in_decode += sub_frame_offset
        
        # Extract the exact samples
        extract_end = min(len(self.last_decoded_audio), 
                         sample_offset_in_decode + requested_samples)
        
        if sample_offset_in_decode < len(self.last_decoded_audio):
            output_buffer = self.last_decoded_audio[sample_offset_in_decode:extract_end]
        else:
            output_buffer = np.array([])
        
        # Handle insufficient audio (should never happen with correct logic)
        if len(output_buffer) < requested_samples:
            padding_needed = requested_samples - len(output_buffer)
            output_buffer = np.pad(output_buffer, (0, padding_needed))
            print(f"ERROR: Audio starvation! Post-warmup frames: {self.frames_generated_post_warmup}, "
                  f"request spans frames {request_start_frame}-{request_end_frame}, "
                  f"decode window: {decode_start_frame}-{decode_end_frame}. "
                  f"Sample offset: {sample_offset_in_decode}, decode length: {len(self.last_decoded_audio)}. "
                  f"Padded {padding_needed} samples.")
        
        # Update position
        self.current_audio_position += requested_samples
        
        return output_buffer
    
    def get_stats(self):
        """Get performance statistics."""
        stats = {
            'buffer_length_ms': self.buffer_size / self.sample_rate * 1000,
            'latent_buffer_size': len(self.latent_buffer),
            'current_position_seconds': self.current_audio_position / self.sample_rate,
            'frames_generated_post_warmup': self.frames_generated_post_warmup,
            'generation_calls': len(self.generation_times)
        }
        
        if self.generation_times:
            avg_gen_time = np.mean(self.generation_times)
            max_gen_time = np.max(self.generation_times)
            stats.update({
                'avg_generation_time_ms': avg_gen_time * 1000,
                'max_generation_time_ms': max_gen_time * 1000,
            })
        else:
            stats.update({
                'avg_generation_time_ms': 0.0,
                'max_generation_time_ms': 0.0,
            })
        
        return stats
    
    def reset(self):
        """Reset the synthesizer state."""
        self.latent_buffer.clear()
        self.current_audio_position = 0
        self.frames_generated_post_warmup = 0
        self.hidden_state = None
        self.last_decoded_audio = None
        self.last_decode_start_frame = -1
        self._initialize_warmup()


# Example usage class for testing
class InteractiveSynthTest:
    def __init__(self, model, encodec_model, clamp_val):
        self.synth = RealTimeLatentSynthesizer(
            model=model,
            encodec_model=encodec_model,
            buffer_size=24000, #1024,  # ~43ms at 24kHz
            latent_window_size=75,  # ~1 second context
            clamp_val=clamp_val
        )
    
    def simulate_real_time(self, duration_seconds=5.0, param_change_interval=0.5):
        """Simulate real-time synthesis with changing parameters."""
        total_samples = int(duration_seconds * 24000)
        buffer_size = self.synth.buffer_size
        audio_output = []
        
        # Parameter animation
        param_values = np.linspace(0, 1, int(duration_seconds / param_change_interval))
        
        print(f"Simulating {duration_seconds}s of real-time synthesis...")
        
        for i in range(0, total_samples, buffer_size):
            # Update conditioning parameters periodically
            if i % int(param_change_interval * 24000) == 0:
                param_idx = min(len(param_values) - 1, i // int(param_change_interval * 24000))
                self.synth.set_conditioning_params(
                    param0=0.0,  # class stays constant
                    param1=param_values[param_idx],  # animated parameter
                    param2=0.5   # static parameter
                )
            
            # Generate next buffer
            buffer = self.synth.generate(buffer_size)
            audio_output.append(buffer)
        
        # Combine all buffers
        full_audio = np.concatenate(audio_output)
        
        # Print stats
        stats = self.synth.get_stats()
        print(f"\nPerformance Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
        
        return full_audio[:total_samples]  # Trim to exact length