import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
  def __init__(self, d_model):
      super().__init__()
      self.d_model = d_model

      # Q, K, V
      self.W_q = nn.Linear(d_model, d_model, bias=False)
      self.W_k = nn.Linear(d_model, d_model, bias=False)
      self.W_v = nn.Linear(d_model, d_model, bias=False)

      # Output
      self.W_o = nn.Linear(d_model, d_model)

  def forward(self, x, mask=None, return_attention=False):
    
    Q = self.W_q(x)  # (B, T, D)
    K = self.W_k(x)
    V = self.W_v(x)
  
