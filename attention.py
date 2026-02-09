import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, output_dim, context_length, dropout_rate=0.1):
        super().__init__()
        self.output_dim = output_dim

        self.q_weights = nn.Linear(embedding_dim, output_dim, bias=False)
        self.k_weights = nn.Linear(embedding_dim, output_dim, bias=False)
        self.v_weights = nn.Linear(embedding_dim, output_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):

        B, T, C = x.shape 
        
        q = self.q_weights(x) 
        k = self.k_weights(x) 
        v = self.v_weights(x) 

        attn_scores = q @ k.transpose(-2, -1) * (self.output_dim ** -0.5)

        attn_scores = attn_scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
   
        attn_weights = torch.softmax(attn_scores, dim=-1) 
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ v
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embedding_dim, context_length, dropout_rate=0.1):
        super().__init__()

        self.heads = nn.ModuleList([
            SelfAttention(embedding_dim, head_size, context_length, dropout_rate) 
            for _ in range(num_heads)
        ])
        
        self.proj = nn.Linear(num_heads * head_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
    
        out = self.dropout(self.proj(out))
        return out
    

