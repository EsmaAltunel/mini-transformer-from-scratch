import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def get_position_embedding(context_length, embedding_dim, base = 10000, device = "cpu"):
    p_embeddings = torch.zeros(context_length, embedding_dim, device = device)
    for pos in range(context_length):
        for i in range(embedding_dim//2):
            p_embeddings[pos, 2 * i] = math.sin(pos / (base ** (2 * i / embedding_dim)))
            if i + 1 < embedding_dim:
                p_embeddings[pos, 2 * i + 1] = math.cos(pos / (base ** (2 * i / embedding_dim)))

    return p_embeddings.unsqueeze(0)

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.register_buffer('pos_encoding', get_position_embedding(context_length, embedding_dim))

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        x = tok_emb + self.pos_encoding[:, :T, :]
        
        return x
    
