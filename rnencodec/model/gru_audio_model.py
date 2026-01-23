import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Literal

CascadeMode = Literal["hard", "soft"]
HardSampleMode = Literal["argmax","gumbel","sample"]

@dataclass
class GRUModelConfig:
    # core
    input_size: int = 128
    cond_size: int = 3
    hidden_size: int = 48
    num_layers: int = 4
    n_q: int = 8
    codebook_size: int = 1024
    dropout: float = 0.1

    
    
    inp_proportion: int = 1
    cond_proportion: int = 1

    # training-time knobs you already have
    gumbel_tau_start: float = 1.0
    gumbel_tau_end: float = 0.5
    straight_through: bool = True

    # cascade selection
    cascade: CascadeMode = "soft"  # "hard" | "soft"


    # HARD cascade knobs
    hard_sample_mode: HardSampleMode = "sample"
    top_n_hard: Optional[int] = None
    temperature_hard: float = 1.0

    # SOFT cascade knobs (RNG-free unless you later add gumbel-soft)
    tau_soft: float = 0.6
    top_n_soft: Optional[int] = None  # optional sparse softmax (still deterministic)


class RNN(nn.Module):
   def __init__(self, config: GRUModelConfig, encodec_model):
       super(RNN, self).__init__()
       self.config = config

       self.input_size = config.input_size
       self.cond_size = config.cond_size
       self.hidden_size = config.hidden_size
       self.n_q = config.n_q
       self.codebook_size = config.codebook_size
       self.num_layers = config.num_layers

       # --- cascade & knobs from config (single source of truth) ---
       self.cascade = config.cascade

        # hard
       self.hard_sample_mode = config.hard_sample_mode
       self.top_n_hard = config.top_n_hard
       self.temperature_hard = config.temperature_hard

        # soft
       self.tau_soft = config.tau_soft
       self.top_n_soft = config.top_n_soft


       # input projection to RNN model size, split btween content and conditioning parameters
       lpn = config.inp_proportion * self.hidden_size // (config.inp_proportion + config.cond_proportion)
       lcn = self.hidden_size - lpn
       print(f"Latents embedded in {lpn} of the GRU input size of {self.hidden_size}")
       print(f"Conditioning parameters embedded in {lcn} of the GRU input size of {self.hidden_size}")
       self.latent_proj = nn.Linear(self.input_size, lpn)
       self.cond_proj = nn.Linear(self.cond_size, lcn)

       # GRU backbone
       self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, 
                         batch_first=True, 
                         dropout=config.dropout if config.num_layers > 1 else 0.0)
       
       # Sequential decoder heads - each takes RNN output + lower codebook latents
       # Input size: hidden_size (RNN) + self.input_size (sum of lower codebook latents)
       decoder_input_size = self.hidden_size + self.input_size
       self.decoders = nn.ModuleList([
           nn.Linear(decoder_input_size, self.codebook_size) 
           for _ in range(self.n_q)
       ])

       self._initialize_weights()


       # Build effective tables now, on THIS model's device
       dev = next(self.parameters()).device
       encodec_model = encodec_model.to(dev)
       E_eff = self._build_effective_codebooks(encodec_model)   # (n_q, K, D) on dev
       # Non-persistent: won’t be saved in checkpoints
       self.register_buffer("_E_eff", E_eff, persistent=False)


   def _initialize_weights(self):
       for name, param in self.named_parameters():
           if "weight" in name:
               nn.init.xavier_uniform_(param)
           elif "bias" in name:
               nn.init.constant_(param, 0.0)


   
#---------------------           Unified sampling   ---------------------------------

   def forward(
    self,
    input: torch.Tensor,
    hidden: torch.Tensor,
    target_codebook_latents: Optional[List[torch.Tensor]] = None,
    use_teacher_forcing: bool = False,
    *,
    return_step_latent: bool = True,
    ):  
        """
        Args:
            input:  (B, input_size + cond_size)
            hidden: GRU hidden state
            target_codebook_latents: optional [n_q x (B,128)] for teacher forcing
            use_teacher_forcing: bool
            return_step_latent: include per-step latent sum in outputs

        Returns:
            logits_list: [n_q x (B, K)]
            hidden:      updated hidden
            sampled_indices: (B, n_q) LongTensor or None (soft/TF)
            step_latent: (B, 128) or None
        """

        B = input.size(0)
        device = input.device

        # --- project + GRU ---
        latent_part = input[:, :self.input_size]
        cond_part   = input[:, self.input_size:]
        h_in = torch.cat([self.latent_proj(latent_part), self.cond_proj(cond_part)], dim=-1)
        h_out, hidden = self.gru(h_in.view(B, 1, -1), hidden)
        h_out = h_out.view(B, -1)

        logits_list: List[torch.Tensor] = []
        sampled_tokens_list: List[Optional[torch.Tensor]] = []
        cumulative_latent = torch.zeros(B, self.input_size, device=device)
        step_latent_sum   = torch.zeros(B, self.input_size, device=device)

        for q in range(self.n_q):
            # decoder head
            dec_in = torch.cat([h_out, cumulative_latent], dim=-1)
            logits_q = self.decoders[q](dec_in)        # (B, K)
            logits_list.append(logits_q)

            if use_teacher_forcing and target_codebook_latents is not None:
                # teacher-forced: drive next level with GT latents
                e_q = target_codebook_latents[q]       # (B, 128)
                sampled_tokens_list.append(None)
            elif self.cascade == "soft":
                # SOFT (deterministic) — optional sparse top-k before softmax
                logits_soft = logits_q
                if self.top_n_soft is not None and self.top_n_soft < logits_soft.size(-1):
                    vals, inds = logits_soft.topk(self.top_n_soft, dim=-1)           # (B, top_k)
                    masked = torch.full_like(logits_soft, torch.finfo(logits_soft.dtype).min)
                    logits_soft = masked.scatter(-1, inds, vals)

                weights = torch.softmax(logits_soft / max(self.tau_soft, 1e-6), dim=-1)  # (B, K)
                E_q = self._E_eff[q]                    # (K, D)
                e_q = weights @ E_q                     # (B, D)
                sampled_tokens_list.append(None)        # no hard sample in soft mode
            else:
                # HARD: choose token then embed it
                idx_q = self._select_tokens(
                    logits_q,
                    mode=self.hard_sample_mode,
                    temperature=self.temperature_hard,
                    top_n_hard=self.top_n_hard,
                )  # (B,)
                e_q = self._code_to_latent_level(q, idx_q, out_device=device)   # (B,128)
                sampled_tokens_list.append(idx_q)

            # drive next level + accumulate
            if q < self.n_q - 1:
                cumulative_latent = cumulative_latent + e_q
            step_latent_sum = step_latent_sum + e_q

        sampled_indices = (
            torch.stack(
                [t if t is not None else torch.full((B,), -1, device=device, dtype=torch.long)
                for t in sampled_tokens_list],
                dim=1,
            )
            if any(t is not None for t in sampled_tokens_list) else None
        )

        step_latent = step_latent_sum if return_step_latent else None
        return logits_list, hidden, sampled_indices, step_latent


####################################################################
#  Helpers
####################################################################

   def _select_tokens(
        self,
        logits_k: torch.Tensor, *,            # (..., K)
        mode: str = "gumbel",                 # "argmax" | "gumbel" | "sample"
        temperature: float = 1.0,
        top_n_hard: int | None = None,
    ) -> torch.LongTensor:
        K = logits_k.size(-1)
    
        # Fast path
        if mode == "argmax" or temperature <= 0:
            return logits_k.argmax(dim=-1)
    
        # Sanitize top_n_hard
        if top_n_hard is not None:
            top_n_hard = int(top_n_hard)
            if top_n_hard < 1:
                raise ValueError("top_n_hard must be >= 1")
            if top_n_hard >= K:
                top_n_hard = None  # full set
    
        t = 1e-6 if temperature <= 0 else temperature
    
        if top_n_hard is None:
            if mode == "gumbel":
                g = torch.empty_like(logits_k).exponential_().log_().neg_()  # ~Gumbel(0,1)
                return (logits_k / t + g).argmax(dim=-1)
            elif mode == "sample":
                probs = torch.softmax(logits_k / t, dim=-1)
                flat  = probs.view(-1, K)
                idx   = torch.multinomial(flat, 1).squeeze(-1)
                return idx.view(probs.shape[:-1])
            else:
                raise ValueError(f"unknown mode: {mode!r}")
    
        # Restrict to top-k slice
        vals, inds = logits_k.topk(top_n_hard, dim=-1)  # inds: (..., top_n_hard)
    
        if mode == "gumbel":
            g   = torch.empty_like(vals).exponential_().log_().neg_()
            sel = (vals / t + g).argmax(dim=-1)                # (...,)
        elif mode == "sample":
            probs = torch.softmax(vals / t, dim=-1)
            flat  = probs.view(-1, top_n_hard)
            sel   = torch.multinomial(flat, 1).view(*probs.shape[:-1]).squeeze(-1)  # (...,)
        else:
            raise ValueError(f"unknown mode: {mode!r}")
    
        # Map back to original indices — handle 1D and batched cases
        if inds.dim() == 1:
            # Unbatched: inds (top_n_hard,), sel scalar
            return inds[sel]
        else:
            # Batched: make sel shape (..., 1) to match inds (..., top_n_hard)
            sel_exp = sel.view(*inds.shape[:-1], 1).long()
            return inds.gather(-1, sel_exp).squeeze(-1)


    
   def _build_effective_codebooks(self, encodec_model: nn.Module) -> torch.Tensor:
        """
        This is the lookup table for use in going from tokens to latent space. 
        We create the table by actually decoding each token index since we couldn't find how they are stored in the Encodec model!

        Conventions:
            K - codebook size (eg. 1024)
            D - latent dimension (e.g. 128)
        """
        device = next(encodec_model.parameters()).device
        q = getattr(encodec_model, "quantizer", None)
        layers = getattr(q, "layers", None) or getattr(getattr(q, "vq", None), "layers", None)
        if layers is None:
            raise RuntimeError("encodec_model.quantizer.layers not found")

        K = getattr(getattr(encodec_model, "config", None), "codebook_size", None)
        if K is None: K = getattr(encodec_model.quantizer, "codebook_size", None)
        if K is None: K = getattr(self, "codebook_size", None)
        if K is None: raise RuntimeError("Could not determine codebook_size (K)")

        E_list = []
        with torch.no_grad():
            idx_all = torch.arange(K, device=device, dtype=torch.long).unsqueeze(1)  # (K,1)
            for qidx in range(self.n_q):
                z_kD1 = layers[qidx].decode(idx_all)     # (K,D,1)
                E_list.append(z_kD1.squeeze(-1).contiguous())  # (K,D)
        return torch.stack(E_list, dim=0)
   


   def _code_to_latent_level(self,  level_q: int, tokens: torch.Tensor, out_device=None):
        """
        Decode ONE level using the cached effective table (matches Encodec exactly).
        tokens: (B,) or (B,1) LongTensor
        returns: (B, 128) float

        Note, this is essentially a drop in replacement for Hugging Face enc_model.quantizer.decode() except for arg order:
            input codes_btq is (B, T, n_q) → output (B, T, D).
            HF: input codes is (n_q, B, T) → output (B, D, T).
            So
                z_bDt = enc_model.quantizer.decode(codes_ntb)  # (B, D, T)
                is equal to: 
                z_btD = model._codes_to_latent_sum(codes_btq, scales_btq=None)  # (B, T, D)
                z_bDt = z_btD.permute(0, 2, 1)  # if you want Encodec decoder layout
        """
        E_q = self._E_eff[level_q]                                 # (K, D)
        idx = tokens.view(-1).to(E_q.device).long()                # (B,)
        lat = F.embedding(idx, E_q)                                # (B, D)
        if out_device is not None and lat.device != out_device:
            lat = lat.to(out_device)
        return lat

   def _codes_to_latent_sum(
        self,
        codes_btq: torch.Tensor,                 # (B, T, n_q) long/int
        scales_btq: torch.Tensor | None = None,  # (B, T, n_q) float, optional
        out_device=None):
        """
        Sum per-level latents using cached codebook tables.
        Returns: (B, T, D)
        """
        B, T, n_q = codes_btq.shape
        assert n_q == self.n_q, f"codes last dim {n_q} != n_q {self.n_q}"
    
        dev = self._E_eff.device
        D = self._E_eff.size(-1)
    
        # Ensure dtype/device ONCE
        if codes_btq.dtype != torch.long or codes_btq.device != dev:
            codes_btq = codes_btq.to(dev, dtype=torch.long, non_blocking=True)
    
        if scales_btq is not None and scales_btq.device != dev:
            scales_btq = scales_btq.to(dev, non_blocking=True)
    
        # Preallocate accumulator
        z = torch.zeros(B, T, D, device=dev, dtype=self._E_eff.dtype)
    
        # Loop levels; _E_eff[q] is a view (safe after .to())
        for q in range(n_q):
            E_q = self._E_eff[q]  # (K, D)
            idx = codes_btq[..., q].reshape(-1)  # (B*T,)
            e_q = F.embedding(idx, E_q).view(B, T, D)  # (B, T, D)
            if scales_btq is not None:
                e_q = e_q * scales_btq[..., q].unsqueeze(-1)  # broadcast
            z.add_(e_q)  # in-place accumulate
    
        if out_device is not None and out_device != dev:
            z = z.to(out_device, non_blocking=True)
        return z
       

    # NEW: expected latent for one codebook
   def _expected_latent_from_logits(
        self,
        logits: torch.Tensor,       # (B, K)
        E_q: torch.Tensor,          # (K, D)
        *, 
        tau: float = 1.0,
        top_n_hard: Optional[int] = None
    ) -> torch.Tensor:              # (B, D)
        # Optional top-k mask before softmax
        if top_n_hard is not None and top_n_hard < logits.size(-1):
            vals, inds = logits.topk(top_n_hard, dim=-1)           # (B, top_n_hard)
            masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
            logits = masked.scatter(-1, inds, vals)

        probs = torch.softmax(logits / max(tau, 1e-6), dim=-1)  # (B, K)
        # (B,K) @ (K,D) -> (B,D)
        return probs @ E_q



   def _soft_and_hard_from_logits(self, logits_btnk: torch.Tensor, tau: float = 0.5, use_gumbel: bool = False):
        """
        Prepare for soft/ST training (not used yet).
        logits: (B, T, n_q, K)
        returns:
        indices      (B, T, n_q)   hard choices (argmax or gumbel-argmax)
        e_soft_sum   (B, T, D)     sum_q soft latents (weighted by softmax/tau)
        e_hard_sum   (B, T, D)     sum_q hard latents (embedding of indices)
        e_st_sum     (B, T, D)     straight-through sum (forward=hard, backward=soft)
        """

        B, T, n_q, K = logits_btnk.shape
        assert n_q == self.n_q

        # Soft weights
        w = torch.softmax(logits_btnk / tau, dim=-1)                       # (B,T,n_q,K)
        # Soft latents via einsum with precomputed E_eff
        e_soft = torch.einsum("btnk,nkd->btnd", w, self._E_eff)            # (B,T,n_q,D)

        # Hard indices (single draw, reuse everywhere)
        if use_gumbel:
            g = -torch.log(-torch.log(torch.rand_like(logits_btnk)))
            idx = (logits_btnk + g).argmax(dim=-1)                          # (B,T,n_q)
        else:
            idx = logits_btnk.argmax(dim=-1)

        # Hard latents
        e_hard_levels = []
        for q in range(n_q):
            e_q = F.embedding(idx[..., q].reshape(-1), self._E_eff[q]).view(B, T, -1)
            e_hard_levels.append(e_q)                                       # (B,T,D)
        e_hard = torch.stack(e_hard_levels, dim=2)                          # (B,T,n_q,D)

        # Sums
        e_soft_sum = e_soft.sum(dim=2)                                      # (B,T,D)
        e_hard_sum = e_hard.sum(dim=2)                                      # (B,T,D)

        # Straight-through
        e_st_sum = e_hard_sum + (e_soft_sum - e_hard_sum).detach()
        return idx, e_soft_sum, e_hard_sum, e_st_sum



   def init_hidden(self, batch_size=1):
       """Initialize hidden state for each minibatch"""
       return .1 * torch.rand(self.num_layers, batch_size, self.hidden_size, 
                             dtype=torch.float, 
                             device=self.gru.weight_hh_l0.device) - .05