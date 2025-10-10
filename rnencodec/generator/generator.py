import librosa
import torch
import torch.nn.functional as F
import os
import numpy as np
from pathlib import Path
import copy # for deepcopy

import warnings
from typing import Optional, Literal
from typing import Literal



from rnencodec.model.gru_audio_model import RNN, GRUModelConfig
#from rnencodec.audioDataLoader.audio_dataset import  efficient_codes_to_latents, preprocess_latents_for_RNN # , latents_to_audio_simple,
from rnencodec.audioDataLoader.audio_dataset import  preprocess_latents_for_RNN # , latents_to_audio_simple,
ValidSampleMode = Literal["argmax", "gumbel", "sample"]
spf = 320

class RNNGenerator():
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_config: GRUModelConfig, data_config, enc_model, chunksize: int, hopsize: int, sample_mode: str, top_n: int, temperature: float,
        *,
        strict: bool = True,
        map_location: Optional[torch.device | str] = None,
    ) -> "RNNGenerator":

        device = getattr(enc_model, "device", None)

        print(f'Initializing the RNNGenerator on device = {device}')
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model, load weights
        model = RNN(model_config, enc_model).to(device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state, strict=False)  # False if your export is fp16; True if fp32
        model.to(device).eval()

        return cls(model=model, model_config=model_config, data_config=data_config, enc_model=enc_model, chunksize=chunksize, hopsize=hopsize, sample_mode=sample_mode, top_n=top_n, temperature=temperature)
    
    def __init__(self, model, model_config, data_config, enc_model, chunksize, hopsize, sample_mode: str, top_n, temperature) : 

        
        self.model=model
        self.model.eval()
        self.enc_model=enc_model
        self.dev = next(self.enc_model.parameters()).device 

        
        self.codebook_size = self.model.config.codebook_size
        self.n_q = self.model.config.n_q
        self.cond_size = self.model.config.cond_size
        self.clamp_val = data_config.clamp_val # need this to map between encodec latents and model input ranges


        self.chunksize=chunksize
        self.hopsize=hopsize

        self.top_n=top_n
        self.temperature=temperature
        self.sample_mode = sample_mode

        #state between call to generate steps
        self.hidden = None # updated on every sequence step in warmup and in run_inference

        sd=.33 # to create data in [-1,1]
        self.current_latent = torch.clamp(torch.randn(1, 128) * sd, -3*sd, 3*sd).to(self.dev)  # need one from the "previous step" if we are running pure inference with parameters only
        self.codebuf = torch.zeros(self.n_q, self.chunksize, dtype=torch.long, device=self.dev) #if users "warm up", this will be filled naturally before post-warmup calls


    # same parmvect used across all T
    def run_inference(self, params_seq, *, hop: int | None = None, latent_seq=None):
        """
        params_seq: None, a single 1D vector (cond_size,), or a sequence (T, cond_size).
        latent_seq: optional sequence (T, latent_size). If provided, must have same T as params_seq.
        Returns LongTensor (n_q, T) on self.dev.
        """
        T=hop
        
        dev = self.dev
        n_q = self.n_q
        latent_size = self.current_latent.shape[-1]
    
        # ---- Normalize params (keeps your existing behavior) ----
        if params_seq is None or self.cond_size == 0:
            cond_mat = None
        else:
            cond = torch.as_tensor(params_seq, device=dev, dtype=torch.float32)
            if cond.dim() == 1:                   # single vector -> your old broadcast path
                T = int(T) if T is not None else 1
                cond_mat = cond.view(1, -1).expand(T, -1).contiguous()
            elif cond.dim() == 2:                 # (T, cond_size)
                T = cond.size(0) if T is None else int(T)
                cond_mat = cond
                assert cond_mat.size(0) >= T, "params_seq shorter than T"
            else:
                raise ValueError("params_seq must be (cond_size,) or (T, cond_size)")
    
        # ---- Optional external latent sequence ----
        lat_mat = None
        if latent_seq is not None:
            lat_mat = torch.as_tensor(latent_seq, device=dev, dtype=torch.float32).to(self.dev)
            assert lat_mat.dim() == 2, "latent_seq must be 2D (T, latent_size)"
            assert lat_mat.size(1) == latent_size, f"latent_seq latent_size {lat_mat.size(1)} != {latent_size}"
            if T is None:
                T = lat_mat.size(0)
            else:
                assert lat_mat.size(0) == T, "latent_seq and params_seq must have same T"
    
        # ---- Finalize T ----
        if T is None:
            raise ValueError("T could not be inferred; provide params_seq or latent_seq, or pass T explicitly.")
        T = int(T)
    
        codes_nt = torch.empty(n_q, T, dtype=torch.long, device=dev)
    
        with torch.inference_mode():
            for t in range(T):
                # choose input latent for this step
                in_latent = lat_mat[t:t+1] if lat_mat is not None else self.current_latent
    
                # concat conditioning if present
                if cond_mat is not None:
                    cond_vec = cond_mat[t:t+1]
                    next_input_full = torch.cat([in_latent, cond_vec], dim=-1)
                else:
                    next_input_full = in_latent
    
                logits_list, self.hidden, sampled_indices, step_latent = self.model(
                    next_input_full,
                    self.hidden,
                    use_teacher_forcing=False,
                    temperature=self.temperature,
                    batch_size=1,
                    sample_mode=("sample" if (self.sample_mode=="sample" and self.top_n and self.top_n > 0) else "argmax"),
                    top_n=self.top_n,
                    return_step_latent=True,
                )

                codes_nt[:, t] = sampled_indices[0]
                # keep computing current_latent from model output for possible later use
                self.current_latent = preprocess_latents_for_RNN(step_latent, self.clamp_val)
    
        return codes_nt



    # same parmvect used across all T
    def getNextCodeChunk(self, params, *, hop: int | None = None, latent_seq=None):
        with torch.inference_mode():
            new_codes = self.run_inference(params, hop=hop, latent_seq=latent_seq)  # (n_q, h) long on self.dev
    
            buf = self.codebuf           # (n_q, T)
            T   = self.chunksize
            h   = new_codes.size(1)

            if h >= T:
                buf.copy_(new_codes[:, -T:])        # replace with newest T
                warnings.warn(f"Warning: ....... chunk size {T} is <= hop size {h}. RETURNING full hop and continuing")
                return new_codes
            else:
                buf[:, :-h] = buf[:, h:]            # shift left
                buf[:, -h:] = new_codes             # append tail
    
            return buf


    # same parmvect used across all T
    def getNextAudioHop(self, params, *, hop: int | None = None, latent_seq=None) :
        with torch.inference_mode():
            FOO = self.getNextCodeChunk(params, hop=hop, latent_seq=latent_seq)  # (n_q, T) long
            codes_bnt = FOO.unsqueeze(0)  # (B=1, n_q, T)
            
            # decode; some HF builds return (B,C,S), others (C,S)
            audio_t = self.enc_model.decode([codes_bnt], audio_scales=[None])[0]
    
            # normalize to (S,) torch tensor BEFORE converting to numpy
            if audio_t.ndim == 3:      # (B, C, S)
                audio_t = audio_t[0, 0]
            elif audio_t.ndim == 2:    # (C, S)
                audio_t = audio_t[0]
            elif audio_t.ndim == 1:    # (S,)
                pass
            else:
                raise RuntimeError(f"unexpected audio shape: {tuple(audio_t.shape)}")

            alen = hop * spf if hop is not None else params.shape[0] * spf
            return audio_t[-alen:] \
                                        .to("cpu", non_blocking=True) \
                                        .contiguous() \
                                        .numpy()


    def warmup(self, params, hop: int, sigma: float = 0.1):
        """
        Prime the RNN by teacher-forcing `hop` steps with:
          - params repeated each step
          - mean-0 noisy latent sequence (128-D) per step
        Returns whatever your downstream call returns (e.g., audio or codes).
        """
        dev = self.dev
        cond_size = getattr(self, "cond_size", None)

        assert cond_size is None or len(params) == cond_size, \
            f"params length {len(params)} != cond_size {cond_size}"
    
        # (T, cond_size): repeat the single params vector hop times (zero-copy view)
        p = torch.as_tensor(params, dtype=torch.float32, device=dev)
        params_seq = p.view(1, -1).expand(hop, -1)

        print(f'params_seq.shape = {params_seq.shape}')
    
        # (T, latent_size): mean-0 Gaussian noise around zero-latent
        latent_size = self.current_latent.shape[-1]   # e.g., 128
        latent_seq  = torch.randn(hop, latent_size, device=dev) * sigma
    
        # # Drive the model once to warm the hidden state; discard or return as you like
        # # If you want *codes*, call run_inference; if you want *audio*, call your audio hop method.
        # codes = self.run_inference(params_seq=params_seq, hop=hop, latent_seq=latent_seq)
        # return codes
        # # or: return self.getNextAudioHop(params_seq, latent_seq)

        return self.getNextAudioHop(params_seq,  latent_seq=latent_seq)






        
