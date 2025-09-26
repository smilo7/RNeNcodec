import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class GRUModelConfig:
   input_size: int = 128  # 128D latent vectors
   cond_size: int = 3
   hidden_size: int = 48
   num_layers: int = 4
   n_q: int = 8  # Number of quantization levels (codebooks)
   codebook_size: int = 1024  # Size of each codebook
   dropout: float = 0.1
   inp_proportion = 1
   cond_proportion = 1


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

   def forward(self,
            input,
            hidden,
            target_codebook_latents=None,
            use_teacher_forcing=False,
            temperature=1.0,
            batch_size=1,
            *,
            sample_mode: str = "sample",      # "argmax" | "gumbel" | "sample"
            top_n: int | None = None,         # optional top-k restriction
            return_step_latent: bool = True   # return sum of per-level latents this step
            ):
        """
        Args:
            input: (batch_size, input_size + cond_size) - 128D latent + conditioning
            hidden: GRU hidden state
            target_codebook_latents: Optional[List[Tensor]] - 128D latents per codebook (teacher forcing)
            use_teacher_forcing: bool - whether to use teacher forcing
            temperature: float - sampling temperature (used for gumbel/sample)
            batch_size: int
            sample_mode: "argmax" | "gumbel" | "sample"
            top_n: if set, restrict sampling to top_n logits (top-k)
            return_step_latent: also return (batch, 128) sum of all codebook latents for this step

        Returns:
            logits_list: List[Tensor], each (batch_size, codebook_size)
            hidden: updated GRU hidden
            sampled_indices: (batch_size, n_q) LongTensor of the ONE set of tokens used (None if pure TF)
            step_latent: (batch_size, 128) sum of per-level latents for this step (or None if disabled)
        """
        # Split the input and process through GRU
        latent_part = input[:, :self.input_size]           # (batch, 128)
        cond_part   = input[:, self.input_size:]           # (batch, cond_size)

        assert latent_part.abs().max().item() < 1.05, f"Max absolute value {latent_part.abs().max().item():.3f} >= 1.05"

        latent_h = self.latent_proj(latent_part)
        cond_h   = self.cond_proj(cond_part)
        h1 = torch.cat([latent_h, cond_h], dim=-1)

        h_out, hidden = self.gru(h1.view(batch_size, 1, -1), hidden)
        h_out = h_out.view(batch_size, -1)

        # Sequential codebook prediction with unified sampling
        logits = []
        device = h_out.device
        cumulative_latent = torch.zeros(batch_size, self.input_size, device=device)  # running 128D sum
        sampled_tokens_list = []   # collect per-q sampled indices (B,)

        for codebook_idx in range(self.n_q):
            decoder_input = torch.cat([h_out, cumulative_latent], dim=-1)
            codebook_logits = self.decoders[codebook_idx](decoder_input)   # (B, K)
            logits.append(codebook_logits)

            if use_teacher_forcing and target_codebook_latents is not None:
                sampled_tokens_list.append(None)
                if codebook_idx < self.n_q - 1:
                    cumulative_latent = cumulative_latent + target_codebook_latents[codebook_idx]  # (B,128)
            else:
                # --- sample ONCE here and reuse it everywhere else ---
                idx_q = self._select_tokens(
                    codebook_logits,
                    mode=sample_mode,
                    temperature=temperature,
                    top_n=top_n
                )  # (B,)
                sampled_tokens_list.append(idx_q)

                if codebook_idx < self.n_q - 1:
                    decoded_latent = self._code_to_latent_level(
                        codebook_idx,
                        idx_q,
                        out_device=device
                    )  # (B,128)
                    cumulative_latent = cumulative_latent + decoded_latent

        # Package sampled indices (B, n_q) or None if TF
        sampled_indices = None
        if any(t is not None for t in sampled_tokens_list):
            sampled_indices = torch.stack([t if t is not None else torch.full((batch_size,), -1, device=device, dtype=torch.long)
                                        for t in sampled_tokens_list], dim=1)  # (B, n_q)

        # Compute per-step latent sum if requested
        step_latent = None
        if return_step_latent:
            if use_teacher_forcing and target_codebook_latents is not None:
                step_latent = torch.stack(target_codebook_latents, dim=0).sum(dim=0)  # (B,128)
            else:
                if sampled_indices is None:
                    step_latent = torch.zeros(batch_size, self.input_size, device=device)
                else:
                    step_latent = torch.zeros(batch_size, self.input_size, device=device)
                    for q in range(self.n_q):
                        idx_q = sampled_indices[:, q]  # (B,)
                        if (idx_q >= 0).any():
                            e_q = self._code_to_latent_level(q, idx_q.clamp_min(0), out_device=device)  # (B,128)
                            if (idx_q < 0).any():
                                mask = (idx_q >= 0).float().unsqueeze(-1)
                                e_q = e_q * mask
                            step_latent = step_latent + e_q

        return logits, hidden, sampled_indices, step_latent

####################################################################
#  Helpers
####################################################################

   # def _select_tokens(self, logits_k: torch.Tensor, *, mode: str = "gumbel",
   #                 temperature: float = 1.0, top_n: int | None = None) -> torch.LongTensor:
   #      """
   #      Select hard token indices from logits (..., K) once.
   #      mode: "argmax" | "gumbel" | "sample"
   #      top_n: if set, restrict choice to top_n logits (top-k sampling).
   #      returns: indices with shape logits_k.shape[:-1]
   #      """
   #      K = logits_k.size(-1)
   #      if mode == "argmax":
   #          return logits_k.argmax(dim=-1)

   #      # optional top-k mask
   #      if top_n is not None and 1 <= top_n < K:
   #          topv, topi = torch.topk(logits_k, k=top_n, dim=-1)
   #          masked = torch.full_like(logits_k, float("-inf"))
   #          logits_k = masked.scatter(-1, topi, topv)

   #      if mode == "gumbel":
   #          # Gumbel(0,1) noise
   #          u = torch.rand_like(logits_k).clamp_(1e-6, 1 - 1e-6)
   #          g = -torch.log(-torch.log(u))
   #          return ((logits_k + g) / max(temperature, 1e-6)).argmax(dim=-1)

   #      if mode == "sample":
   #          probs = F.softmax(logits_k / max(temperature, 1e-6), dim=-1)
   #          flat = probs.reshape(-1, probs.size(-1))
   #          idx = torch.multinomial(flat, num_samples=1).squeeze(-1)
   #          return idx.view(probs.shape[:-1])

   #      raise ValueError(f"Unknown sample_mode={mode!r}")

   def _select_tokens(
        self,
        logits_k: torch.Tensor, *,            # (..., K)
        mode: str = "gumbel",                 # "argmax" | "gumbel" | "sample"
        temperature: float = 1.0,
        top_n: int | None = None,
    ) -> torch.LongTensor:
        K = logits_k.size(-1)
    
        # Fast path
        if mode == "argmax" or temperature <= 0:
            return logits_k.argmax(dim=-1)
    
        # Sanitize top_n
        if top_n is not None:
            top_n = int(top_n)
            if top_n < 1:
                raise ValueError("top_n must be >= 1")
            if top_n >= K:
                top_n = None  # full set
    
        t = 1e-6 if temperature <= 0 else temperature
    
        if top_n is None:
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
        vals, inds = logits_k.topk(top_n, dim=-1)  # inds: (..., top_n)
    
        if mode == "gumbel":
            g   = torch.empty_like(vals).exponential_().log_().neg_()
            sel = (vals / t + g).argmax(dim=-1)                # (...,)
        elif mode == "sample":
            probs = torch.softmax(vals / t, dim=-1)
            flat  = probs.view(-1, top_n)
            sel   = torch.multinomial(flat, 1).view(*probs.shape[:-1]).squeeze(-1)  # (...,)
        else:
            raise ValueError(f"unknown mode: {mode!r}")
    
        # Map back to original indices — handle 1D and batched cases
        if inds.dim() == 1:
            # Unbatched: inds (top_n,), sel scalar
            return inds[sel]
        else:
            # Batched: make sel shape (..., 1) to match inds (..., top_n)
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