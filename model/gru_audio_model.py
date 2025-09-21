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
       print(f' debug 1')
       # for Encodec codebook tables (one per RVQ level)
       #self._E_eff = None   # (n_q, K, D) effective tables
       self._E_list = None  # filled on first forward() when encodec_model is provided

       print(f' debug 2')


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

       print(f' debug 4')
       self._initialize_weights()


       # Build effective tables now, on THIS model's device
       dev = next(self.parameters()).device
       encodec_model = encodec_model.to(dev)
       E_eff = self._build_effective_codebooks(encodec_model)   # (n_q, K, D) on dev
       # Non-persistent: won’t be saved in checkpoints
       self.register_buffer("_E_eff", E_eff, persistent=False)
       self._E_list = [self._E_eff[i] for i in range(self.n_q)]



   def _initialize_weights(self):
       for name, param in self.named_parameters():
           if "weight" in name:
               nn.init.xavier_uniform_(param)
           elif "bias" in name:
               nn.init.constant_(param, 0.0)

   def forward(self, input, hidden, target_codebook_latents=None, use_teacher_forcing=False, 
               temperature=1.0, batch_size=1):
       """
       Args:
           input: (batch_size, input_size + cond_size) - 128D latent + conditioning
           hidden: GRU hidden state
           target_codebook_latents: Optional[List[Tensor]] - 128D latents for each codebook for teacher forcing
           use_teacher_forcing: bool - whether to use teacher forcing or autoregressive prediction
           encodec_model: EnCodec model for decoding predicted tokens to latents (needed for autoregressive mode)
           temperature: float - sampling temperature for autoregressive mode
           batch_size: batch size
           
       Returns:
           logits: List of tensors, each (batch_size, codebook_size)
           hidden: Updated GRU hidden state
       """
       
       # Split the input and process through GRU (same as before)
       latent_part = input[:, :self.input_size]           # (batch, 128)
       cond_part = input[:, self.input_size:]             # (batch, cond_size)

       assert latent_part.abs().max().item() < 1.05, f"Max absolute value {latent_part.abs().max().item():.3f} >= {1.05}"
       
       # Embed each separately to a different segment of the GRU input
       latent_h = self.latent_proj(latent_part)
       cond_h = self.cond_proj(cond_part)
       
       # Combine
       h1 = torch.cat([latent_h, cond_h], dim=-1)
       
       # GRU processing of the combined input
       h_out, hidden = self.gru(h1.view(batch_size, 1, -1), hidden)
       h_out = h_out.view(batch_size, -1)
       
       # Sequential codebook prediction
       logits = []
       cumulative_latent = torch.zeros(batch_size, self.input_size, device=h_out.device)  # Sum of lower codebook latents
       
       for codebook_idx in range(self.n_q):
           # Prepare input for this codebook predictor
           decoder_input = torch.cat([h_out, cumulative_latent], dim=-1)
           
           # Predict logits for this codebook
           codebook_logits = self.decoders[codebook_idx](decoder_input)
           logits.append(codebook_logits)
           
           # Update cumulative latent for next codebook
           if codebook_idx < self.n_q - 1:  # Don't need to update after last codebook
               if use_teacher_forcing and target_codebook_latents is not None:
                   # Teacher forcing: use ground truth latent
                   cumulative_latent = cumulative_latent + target_codebook_latents[codebook_idx]
               else:
                   
                   # Sample from the predicted distribution
                   probs = torch.softmax(codebook_logits / temperature, dim=-1)
                   sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)  # (batch_size,)


                   decoded_latent = self._code_to_latent_level(
                            codebook_idx,
                            sampled_tokens,
                            out_device=h_out.device
                   )  # (batch_size, 128)



                   cumulative_latent = cumulative_latent + decoded_latent
       
       return logits, hidden
   

####################################################################
#  Helpers
####################################################################

   def _select_tokens(self, logits_k: torch.Tensor, *, mode: str = "gumbel",
                   temperature: float = 1.0, top_n: int | None = None) -> torch.LongTensor:
        """
        Pick hard token indices from logits (…, K) once.
        mode: "argmax" | "gumbel" | "sample"
        top_n: if set, restricts sampling to top_n logits (nucleus-like top-k).
        returns: indices with shape logits_k.shape[:-1]
        """
        K = logits_k.size(-1)
        if mode == "argmax":
            return logits_k.argmax(dim=-1)
    
        if top_n is not None and top_n < K:
            # mask everything but top_n
            topv, topi = torch.topk(logits_k, k=top_n, dim=-1)
            mask = torch.full_like(logits_k, float("-inf"))
            logits_k = mask.scatter(-1, topi, topv)
    
        if mode == "gumbel":
            g = -torch.log(-torch.rand_like(logits_k).clamp_min_(1e-9)).clamp_min_(1e-9)
            return ((logits_k + g) / max(temperature, 1e-6)).argmax(dim=-1)
    
        if mode == "sample":
            probs = F.softmax(logits_k / max(temperature, 1e-6), dim=-1)
            # multinomial expects 2D; flatten then unflatten
            flat = probs.reshape(-1, probs.size(-1))
            idx = torch.multinomial(flat, num_samples=1).squeeze(-1)
            return idx.view(probs.shape[:-1])
    
        raise ValueError(f"Unknown mode={mode}")
        
   def _build_effective_codebooks(self, encodec_model: nn.Module) -> torch.Tensor:
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
        """
        E_q = self._E_eff[level_q]                                 # (K, D)
        idx = tokens.view(-1).to(E_q.device).long()                # (B,)
        lat = F.embedding(idx, E_q)                                # (B, D)
        if out_device is not None and lat.device != out_device:
            lat = lat.to(out_device)
        return lat

   def _codes_to_latent_sum(self, codes_btq: torch.Tensor, scales_btq: torch.Tensor | None = None, out_device=None):
        """
        Manual sum across levels using cached effective tables.
        codes_btq: (B, T, n_q) long
        returns: (B, T, D) float
        """
        B, T, n_q = codes_btq.shape
        assert n_q == self.n_q, f"codes last dim {n_q} != n_q {self.n_q}"
        z = 0.0
        for q in range(n_q):
            E_q = self._E_eff[q]                                    # (K, D)
            idx = codes_btq[..., q].reshape(-1).to(E_q.device).long()
            e_q = F.embedding(idx, E_q).view(B, T, -1)              # (B, T, D)
            z = z + e_q
        if out_device is not None and isinstance(z, torch.Tensor) and z.device != out_device:
            z = z.to(out_device)
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