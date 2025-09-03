import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List

@dataclass
class GRUModelConfig:
   input_size: int = 128  # 128D latent vectors
   cond_size: int = 3
   hidden_size: int = 48
   num_layers: int = 4
   n_q: int = 8  # Number of quantization levels (codebooks)
   codebook_size: int = 1024  # Size of each codebook
   dropout: float = 0.1


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
       ########## self.i2h = nn.Linear(self.input_size + self.cond_size, self.hidden_size)
       
       # Separate projections for data and conditional values to balance their influence and learning capacity
       self.latent_proj = nn.Linear(self.input_size, self.hidden_size // 2)
       self.cond_proj = nn.Linear(self.cond_size, self.hidden_size // 2)

       
       # Same GRU backbone
       self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, 
                         batch_first=True, 
                         dropout=config.dropout if config.num_layers > 1 else 0.0)
       
       # Multiple decoder heads - one for each quantization level
       self.decoders = nn.ModuleList([
           nn.Linear(self.hidden_size, self.codebook_size) 
           for _ in range(self.n_q)
       ])

       self._initialize_weights()

   def _initialize_weights(self):
       for name, param in self.named_parameters():
           if "weight" in name:
               nn.init.xavier_uniform_(param)
           elif "bias" in name:
               nn.init.constant_(param, 0.0)

   def forward(self, input, hidden, batch_size=1):
       """
       Args:
           input: (batch_size, input_size + cond_size) - 128D latent + conditioning
           hidden: GRU hidden state
           batch_size: batch size
           
       Returns:
           logits: List of tensors, each (batch_size, codebook_size)
           hidden: Updated GRU hidden state
       """
       # Same projection as before
       ################h1 = self.i2h(input)
       
       # Split the input
       latent_part = input[:, :self.input_size]           # (batch, 128)
       cond_part = input[:, self.input_size:]             # (batch, cond_size)

       assert latent_part.abs().max().item() < 1.05, f"Max absolute value {latent_part.abs().max().item():.3f} >= {1.05}"
       
       # Process separately
       latent_h = self.latent_proj(latent_part)
       cond_h = self.cond_proj(cond_part)
       
       # Combine
       h1 = torch.cat([latent_h, cond_h], dim=-1)

       
       
       # Same GRU processing
       h_out, hidden = self.gru(h1.view(batch_size, 1, -1), hidden)
       h_out = h_out.view(batch_size, -1)
       
       # Multiple decoder heads - one for each quantization level
       logits = []
       for decoder in self.decoders:
           logits.append(decoder(h_out))
       
       return logits, hidden

   def init_hidden(self, batch_size=1):
       """Initialize hidden state for each minibatch"""
       return .1 * torch.rand(self.num_layers, batch_size, self.hidden_size, 
                             dtype=torch.float, 
                             device=self.gru.weight_hh_l0.device) - .05