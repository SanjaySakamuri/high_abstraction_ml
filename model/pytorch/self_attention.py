import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
  def __init__(self, d_model):
      super().__init__()
      self.d_model = d_model
