import torch
import torch.nn as nn
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
   def __init__(self, config: GRUModelConfig):
       super(RNN, self).__init__()
       self.config = config
       
       self.input_size = config.input_size
       self.cond_size = config.cond_size
       self.hidden_size = config.hidden_size
       self.n_q = config.n_q
       self.codebook_size = config.codebook_size
       self.num_layers = config.num_layers
       
       # Same input projection as before
       lpn = config.inp_proportion * self.hidden_size // (config.inp_proportion + config.cond_proportion)
       lcn = self.hidden_size - lpn
       print(f"Latents embedded in {lpn} of the GRU input size of {self.hidden_size}")
       print(f"Conditioning parameters embedded in {lcn} of the GRU input size of {self.hidden_size}")
       self.latent_proj = nn.Linear(self.input_size, lpn)
       self.cond_proj = nn.Linear(self.cond_size, lcn)

       # Same GRU backbone
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

   def _initialize_weights(self):
       for name, param in self.named_parameters():
           if "weight" in name:
               nn.init.xavier_uniform_(param)
           elif "bias" in name:
               nn.init.constant_(param, 0.0)

   def forward(self, input, hidden, target_codebook_latents=None, use_teacher_forcing=False, 
               encodec_model=None, temperature=1.0, batch_size=1):
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
                   # Autoregressive: sample from predicted logits and decode
                   if encodec_model is None:
                       raise ValueError("encodec_model is required for autoregressive prediction")
                   
                   # Sample from the predicted distribution
                   probs = torch.softmax(codebook_logits / temperature, dim=-1)
                   sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)  # (batch_size,)
                   
                   # Convert to the format expected by efficient_codes_to_latents
                   # We need (1, batch_size, 1) for single codebook, single timestep
                   codes_for_decode = sampled_tokens.unsqueeze(0).unsqueeze(-1)  # (1, batch_size, 1)
                   
                   # Decode to latent (this will be 128D)
                   decoded_latent = self._codes_to_latents(encodec_model, codes_for_decode)
                   decoded_latent = decoded_latent.squeeze(-1)  # Remove time dimension -> (batch_size, 128)
                   
                   cumulative_latent = cumulative_latent + decoded_latent
       
       return logits, hidden

   def _codes_to_latents(self, encodec_model, codes):
       """
       Helper function to decode codes to latents using the provided EnCodec model
       """
       encodec_model.eval()
       with torch.no_grad():
           # codes shape: (n_q, batch, time) - in our case (1, batch_size, 1)
           embeddings = encodec_model.quantizer.decode(codes)
           return embeddings  # (batch_size, 128, time)

   def init_hidden(self, batch_size=1):
       """Initialize hidden state for each minibatch"""
       return .1 * torch.rand(self.num_layers, batch_size, self.hidden_size, 
                             dtype=torch.float, 
                             device=self.gru.weight_hh_l0.device) - .05