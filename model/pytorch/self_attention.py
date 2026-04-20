import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None, return_attention=False):

        
        Q = self.W_q(x) 
        K = self.W_k(x)
        V = self.W_v(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5) 
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)  
        
        out = torch.matmul(attn_weights, V) 
        out = self.W_o(out)
        
        if return_attention:
            return out, attn_weights
        
        return out
