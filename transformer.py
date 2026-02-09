import torch
import math
import torch.nn as nn
from attention import MultiHeadAttention

class LayerNormalization(nn.Module):
    def __init__(self, embedding_dim, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.weights = nn.Parameter(torch.ones(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        normalized_x = (x - mean) / torch.sqrt(variance + self.eps)

        return self.weights * normalized_x + self.bias
    

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (
          1 + torch.tanh(
           torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        )
    )


class MLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.gate_proj = nn.Linear(embedding_dim,hidden_dim)
        self.up_proj = nn.Linear(embedding_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, embedding_dim)
        self.gelu = GeLU()

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = self.gelu(gate)
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)

        return outputs


class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length):
        super().__init__()
        head_size = embedding_dim // num_heads

        self.self_attention = MultiHeadAttention(num_heads, head_size, embedding_dim, context_length, dropout_rate=0.1)
        self.mlp = MLP(embedding_dim, 4 * embedding_dim)
        self.norm1 = LayerNormalization(embedding_dim)
        self.norm2 = LayerNormalization(embedding_dim)
    
    def forward(self, x):
        x = x + self.self_attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

