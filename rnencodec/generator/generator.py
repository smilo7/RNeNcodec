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

spf = 320

class RNNGenerator():
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_config: GRUModelConfig, data_config, enc_model, chunksize: int, hopsize: int,  
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

        return cls(model=model, model_config=model_config, data_config=data_config, enc_model=enc_model, chunksize=chunksize, hopsize=hopsize)
    
    def __init__(self, model, model_config, data_config, enc_model, chunksize, hopsize) : 

        
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
    
                #  CALL THE RNN ------------------------------------------------------
                logits_list, self.hidden, sampled_indices, step_latent = self.model(
                    next_input_full,
                    self.hidden,
                    use_teacher_forcing=False,
                    return_step_latent=True,
                )

                # Handle soft cascade mode (sampled_indices is None)
                if sampled_indices is None:
                    # Soft cascade: no discrete tokens, use argmax of logits to get codes
                    codes_nt[:, t] = torch.stack([logits.argmax(dim=-1)[0] for logits in logits_list])
                else:
                    # Hard cascade: use the sampled indices
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



#################################################################################
#################################################################################
#################################################################################

# frames-per-second of Encodec latent stream
# (you already have this constant elsewhere, but we need it here for hop->samples)
spf = 320  # 24kHz / 75fps ≈ 320 samples per frame

# same normalization helper you use in training/data
def preprocess_latents_for_RNN(latent_in, clamp_val):
    """
    Clamp latent vectors to [-clamp_val, clamp_val] and scale to [-1,1].
    latent_in: (B, 128)
    returns: (B, 128) float32 on same device
    """
    if clamp_val != 0:
        x = torch.clamp(latent_in, -clamp_val, clamp_val) / clamp_val
    else:
        x = latent_in
    return x.to(dtype=torch.float32, non_blocking=True)


class RNNGeneratorSoft:
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_config,
        data_config,
        enc_model,
        chunksize: int,
        hopsize: int,
        *,
        # outside-model sampling knobs (used in soft mode post-model sampling):
        sample_mode_outside: Literal["argmax", "gumbel", "sample"] = "sample",
        top_k_outside: Optional[int] = None,
        temperature_outside: float = 1.0,
        strict: bool = True,
        map_location: Optional[torch.device | str] = None,
    ) -> "RNNGeneratorSoft":
        """
        Build the RNN model, load checkpoint weights, create generator.
        """

        device = getattr(enc_model, "device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing the RNNGeneratorSoft on device = {device}")

        # Build model + load weights
        from rnencodec.model.gru_audio_model import RNN  # adjust import path if different
        model = RNN(model_config, enc_model).to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state, strict=False)
        model.to(device).eval()

        return cls(
            model=model,
            model_config=model_config,
            data_config=data_config,
            enc_model=enc_model,
            chunksize=chunksize,
            hopsize=hopsize,
            sample_mode_outside=sample_mode_outside,
            top_k_outside=top_k_outside,
            temperature_outside=temperature_outside,
        )

    def __init__(
        self,
        model,
        model_config,
        data_config,
        enc_model,
        chunksize,
        hopsize,
        *,
        sample_mode_outside: Literal["argmax", "gumbel", "sample"] = "sample",
        top_k_outside: Optional[int] = None,
        temperature_outside: float = 1.0,
    ):
        """
        model_config: config that controls INSIDE-model behavior (cascade='hard'|'soft',
                      hard_sample_mode, tau_soft, etc.)
        sample_mode_outside/top_k_outside/temperature_outside:
            how WE (the generator) sample logits AFTER the model returns them
            in the soft pipeline.
        """
        self.model = model.eval()
        self.model_config = model_config
        self.data_config = data_config
        self.enc_model = enc_model
        self.dev = next(self.enc_model.parameters()).device

        self.codebook_size = self.model.config.codebook_size
        self.n_q = self.model.config.n_q
        self.cond_size = self.model.config.cond_size
        self.clamp_val = data_config.clamp_val

        self.chunksize = int(chunksize)
        self.hopsize = int(hopsize)

        self.hidden = None  # GRU hidden state

        # current_latent is the feedback latent we feed at each new time step
        # start with a random-ish seed latent
        sd = 0.33
        seed = torch.clamp(torch.randn(1, 128) * sd, -3*sd, 3*sd).to(self.dev)
        self.current_latent = seed.to(dtype=torch.float32)

        # FIFO buffer of codes to decode. Shape (n_q, chunksize)
        self.codebuf = torch.zeros(
            self.n_q, self.chunksize,
            dtype=torch.long,
            device=self.dev,
        )

        # How we post-sample in "soft mode" after forward()
        self.sample_mode_outside = sample_mode_outside
        self.top_k_outside = top_k_outside
        self.temperature_outside = temperature_outside

    # ---------------------------------------------------------------------
    # low-level helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _select_from_logits(
        logits: torch.Tensor,  # (B, K)
        mode: Literal["argmax", "gumbel", "sample"] = "sample",
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Vectorized sampler: given a (B,K) logits tensor, choose integer indices.
        Used in SOFT mode *outside* the model forward, where we want to
        inject stochasticity into what we decode / feed forward.
        """
        B, K = logits.shape

        # Optional top-k mask
        if top_k is not None and 1 <= top_k < K:
            vals, inds = logits.topk(top_k, dim=-1)  # (B,top_k)
            masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
            logits_eff = masked.scatter(-1, inds, vals)
        else:
            logits_eff = logits

        # Deterministic path
        if mode == "argmax" or temperature <= 0:
            return logits_eff.argmax(dim=-1)

        t = max(float(temperature), 1e-6)

        if mode == "gumbel":
            # gumbel noise ~ -log(-log(U))
            g = torch.empty_like(logits_eff).exponential_().log_().neg_()
            return (logits_eff / t + g).argmax(dim=-1)

        # "sample": multinomial from softmax
        probs = torch.softmax(logits_eff / t, dim=-1)  # (B,K)
        return torch.multinomial(probs, 1).squeeze(-1)

    def _tokens_to_latent_sum(self, codes_nq: torch.Tensor) -> torch.Tensor:
        """
        Convert a set of code indices for this frame, shape (n_q,),
        into a (1,128) latent = sum of each codebook's embedding.
        """
        dev = self.dev
        n_q = self.n_q
        out = torch.zeros(1, 128, device=dev, dtype=torch.float32)

        for q in range(n_q):
            idx_q = codes_nq[q].view(1)  # (1,)
            # model._code_to_latent_level returns (B,128)
            e_q = self.model._code_to_latent_level(q, idx_q.to(dev), out_device=dev)
            out = out + e_q.to(dtype=torch.float32)

        # final clamp+scale just like training
        return preprocess_latents_for_RNN(out, self.clamp_val)

    def _prepare_step_inputs(
        self,
        params_step: Optional[torch.Tensor],
        latent_step: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the single-step input vector for the model:
        concat [latent_step (1,128), cond_step (1,cond_size)]
        """
        if params_step is None or self.cond_size == 0:
            return latent_step
        else:
            return torch.cat([latent_step, params_step], dim=-1)

    # ---------------------------------------------------------------------
    # 1 STEP of model.forward()
    # returns: (logits_per_q, codes_from_model, step_latent_model)
    #   logits_per_q: list length n_q, each (1,K)
    #   codes_from_model: (n_q,) LongTensor or None (model returns sampled_indices)
    #   step_latent_model: (1,128) after model's own cascade logic
    # ---------------------------------------------------------------------
    def _run_single_step(
        self,
        params_step: Optional[torch.Tensor],
        latent_step: Optional[torch.Tensor],
    ):
        """
        Runs model.forward() for exactly one time step.
        Uses either the provided latent_step (forced mode) or self.current_latent.
        Keeps self.hidden updated.

        Returns:
          logits_list      [len n_q] each (1,K)
          model_codes_nq   (n_q,) long OR None
          step_latent_model (1,128) float
        """
        dev = self.dev

        in_latent = latent_step if latent_step is not None else self.current_latent
        assert in_latent.shape == (1, 128), f"expected (1,128) latent, got {in_latent.shape}"

        step_in = self._prepare_step_inputs(params_step, in_latent).to(dev)

        with torch.inference_mode():
            logits_list, self.hidden, sampled_indices, step_latent = self.model(
                step_in,
                self.hidden,
                use_teacher_forcing=False,
                return_step_latent=True,
            )
            # logits_list: list of length n_q, each (1,K)
            # sampled_indices: (1, n_q) or None
            # step_latent: (1,128)

        # normalize the latent we got back from the model
        step_latent_model = preprocess_latents_for_RNN(step_latent, self.clamp_val)

        # flatten sampled_indices to (n_q,), or None
        if sampled_indices is not None:
            model_codes_nq = sampled_indices[0].to(torch.long)  # (n_q,)
        else:
            model_codes_nq = None

        return logits_list, model_codes_nq, step_latent_model

    # ---------------------------------------------------------------------
    # HARD MODE:
    # - We trust the model's own sampling (cascade="hard" in the config).
    # - We take the model-produced codes at each step.
    # - We update current_latent using the model's step_latent_model.
    #
    # Returns a FIFO view of codes with shape (n_q, chunksize) like before.
    # ---------------------------------------------------------------------
    def getNextCodeChunkHard(
        self,
        params,
        *,
        hop: Optional[int] = None,
        latent_seq: Optional[torch.Tensor] = None,
    ):
        """
        params:
          - either shape (cond_size,) for static conditioning
          - or shape (T, cond_size) to vary per step
        hop:
          - number of inference steps to run now (defaults to self.hopsize)
        latent_seq:
          - optional external latents to force for each step (T,128)
            instead of feeding back model.step_latent or our running current_latent.

        Returns:
          self.codebuf  (n_q, chunksize) after FIFO update
        """
        dev = self.dev
        T = int(hop if hop is not None else self.hopsize)

        # massage params into per-step rows: a list of (1,cond_size) or [None]s
        if params is None or self.cond_size == 0:
            param_steps = [None] * T
        else:
            p = torch.as_tensor(params, dtype=torch.float32, device=dev)
            if p.dim() == 1:
                p = p.view(1, -1).expand(T, -1).contiguous()
            assert p.shape == (T, self.cond_size), f"cond must be (T,{self.cond_size})"
            param_steps = [p[t:t+1] for t in range(T)]

        # latent forcing?
        if latent_seq is not None:
            l = torch.as_tensor(latent_seq, dtype=torch.float32, device=dev)
            assert l.shape == (T, 128), "latent_seq must be (T,128)"
            latent_steps = [l[t:t+1] for t in range(T)]
        else:
            latent_steps = [None] * T

        # We'll collect the per-step codes from the model
        codes_list = []  # each (n_q,)

        for t in range(T):
            logits_list, model_codes_nq, step_latent_model = self._run_single_step(
                param_steps[t],
                latent_steps[t],
            )

            if model_codes_nq is None:
                # This "shouldn't" happen in cascade="hard", but guard anyway
                raise RuntimeError(
                    "Model did not return sampled_indices in HARD mode. "
                    "Did you set cascade='hard' in model.config?"
                )

            # store codes
            codes_list.append(model_codes_nq.clone())  # (n_q,)

            # update current_latent for next step if we are *not* forcing latent_seq
            if latent_seq is None:
                self.current_latent = step_latent_model  # (1,128)

        # stack -> (T, n_q) -> (n_q, T)
        new_codes = torch.stack(codes_list, dim=0).transpose(0, 1).contiguous()  # (n_q, T)

        # FIFO buffer update
        buf = self.codebuf
        h = new_codes.size(1)
        if h >= self.chunksize:
            buf.copy_(new_codes[:, -self.chunksize:])
            warnings.warn(
                f"Warning: chunk size {self.chunksize} <= hop size {h}. Returning full hop."
            )
            return buf
        else:
            buf[:, :-h] = buf[:, h:]
            buf[:, -h:] = new_codes
            return buf

    # ---------------------------------------------------------------------
    # SOFT MODE:
    # - Inside model.forward(), cascade="soft" means it doesn't sample tokens;
    #   it returns logits_list and a "soft" step_latent_model.
    # - We, OUTSIDE the model, sample from those logits with our own knobs
    #   (sample_mode_outside, top_k_outside, temperature_outside).
    # - We convert those sampled tokens to a latent sum and feed THAT forward
    #   on the NEXT step, so randomness still shapes future predictions.
    #
    # Returns FIFO code buffer (n_q, chunksize) just like hard.
    # ---------------------------------------------------------------------
    def getNextCodeChunkSoft(
        self,
        params,
        *,
        hop: Optional[int] = None,
        latent_seq: Optional[torch.Tensor] = None,
    ):
        dev = self.dev
        T = int(hop if hop is not None else self.hopsize)

        # massage params into per-step rows
        if params is None or self.cond_size == 0:
            param_steps = [None] * T
        else:
            p = torch.as_tensor(params, dtype=torch.float32, device=dev)
            if p.dim() == 1:
                p = p.view(1, -1).expand(T, -1).contiguous()
            assert p.shape == (T, self.cond_size), f"cond must be (T,{self.cond_size})"
            param_steps = [p[t:t+1] for t in range(T)]

        # external latent override?
        if latent_seq is not None:
            l = torch.as_tensor(latent_seq, dtype=torch.float32, device=dev)
            assert l.shape == (T, 128), "latent_seq must be (T,128)"
            latent_steps = [l[t:t+1] for t in range(T)]
        else:
            latent_steps = [None] * T

        codes_list = []  # (n_q,) per step

        for t in range(T):
            # run one model step
            logits_list, model_codes_nq, step_latent_model = self._run_single_step(
                param_steps[t],
                latent_steps[t],
            )
            # logits_list is len n_q, each (1,K)

            # Post-model sampling (outside) from logits_list
            sampled_tokens = []
            for q, lq in enumerate(logits_list):
                # lq: (1,K) -> sample index
                idx_q = self._select_from_logits(
                    lq,  # (1,K)
                    mode=self.sample_mode_outside,
                    temperature=self.temperature_outside,
                    top_k=self.top_k_outside,
                )[0]  # scalar
                sampled_tokens.append(idx_q)
            sampled_tokens_nq = torch.stack(sampled_tokens, dim=0).to(torch.long)  # (n_q,)

            codes_list.append(sampled_tokens_nq.clone())

            # Unless user provided latent_seq for this step,
            # update self.current_latent based on *our* sampled tokens,
            # not the model's step_latent_model (which is "soft expectation").
            if latent_seq is None:
                self.current_latent = self._tokens_to_latent_sum(sampled_tokens_nq)  # (1,128)

        # Now we have T steps of codes -> (T, n_q) -> (n_q, T)
        new_codes = torch.stack(codes_list, dim=0).transpose(0, 1).contiguous()  # (n_q, T)

        # FIFO update
        buf = self.codebuf
        h = new_codes.size(1)
        if h >= self.chunksize:
            buf.copy_(new_codes[:, -self.chunksize:])
            warnings.warn(
                f"Warning: chunk size {self.chunksize} <= hop size {h}. Returning full hop."
            )
            return buf
        else:
            buf[:, :-h] = buf[:, h:]
            buf[:, -h:] = new_codes
            return buf

    # ---------------------------------------------------------------------
    # Public dispatcher: choose hard vs soft based on model.config.cascade
    # and return the updated rolling buffer of codes.
    # ---------------------------------------------------------------------
    def getNextCodeChunk(
        self,
        params,
        *,
        hop: Optional[int] = None,
        latent_seq: Optional[torch.Tensor] = None,
    ):
        mode = getattr(self.model.config, "cascade", "hard")
        if mode == "soft":
            return self.getNextCodeChunkSoft(
                params,
                hop=hop,
                latent_seq=latent_seq,
            )
        else:
            # default / legacy behavior
            return self.getNextCodeChunkHard(
                params,
                hop=hop,
                latent_seq=latent_seq,
            )

    # ---------------------------------------------------------------------
    # Turn the newest codes in the rolling buffer into PCM audio,
    # just like your old getNextAudioHop().
    #
    # We decode the whole buffer of codes, then return ONLY the last hop's
    # worth of PCM (~ hopsize * spf samples).
    # ---------------------------------------------------------------------
    def getNextAudioHop(
        self,
        params,
        *,
        hop: Optional[int] = None,
        latent_seq: Optional[torch.Tensor] = None,
    ):
        with torch.inference_mode():
            codes_buf = self.getNextCodeChunk(
                params,
                hop=hop,
                latent_seq=latent_seq,
            )  # (n_q, chunksize)

            # Encodec expects (B, n_q, T)
            codes_bnt = codes_buf.unsqueeze(0)  # (1, n_q, chunksize)

            audio_t = self.enc_model.decode(
                [codes_bnt], audio_scales=[None]
            )[0]
            # HF Encodec returns weird batch/channel dims sometimes
            if audio_t.ndim == 3:
                # (B,C,T)
                audio_t = audio_t[0, 0]
            elif audio_t.ndim == 2:
                # (B,T)
                audio_t = audio_t[0]

            # pull just the latest hop worth of samples
            hop_frames = int(hop if hop is not None else self.hopsize)
            alen = hop_frames * spf
            return (
                audio_t[-alen:]
                .to("cpu", non_blocking=True)
                .contiguous()
                .numpy()
            )

    # ---------------------------------------------------------------------
    # Convenience warmup: fill hidden state and FIFO before first playback.
    # Same idea you had before.
    # ---------------------------------------------------------------------
    def warmup(self, params, hop: int, sigma: float = 0.1):
        dev = self.dev
        hop = int(hop)

        # broadcast params to length hop if needed
        if self.cond_size:
            p = torch.as_tensor(params, dtype=torch.float32, device=dev)
            if p.dim() == 1:
                p = p.view(1, -1).expand(hop, -1).contiguous()
        else:
            p = None

        latent_size = self.current_latent.shape[-1]  # 128
        latent_seq = torch.randn(hop, latent_size, device=dev) * sigma

        # just generate audio to "prime the pump"
        return self.getNextAudioHop(
            p,
            hop=hop,
            latent_seq=latent_seq,
        )


        
