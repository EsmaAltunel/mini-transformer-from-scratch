import torch
import torch.nn as nn

from embedding import Embedding
from transformer import Decoder
from transformer import LayerNormalization

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, num_heads, num_layers, dropout_rate = 0.1):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, context_length)
        self.blocks = nn.Sequential(*[Decoder(embedding_dim, num_heads,context_length)
        for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(embedding_dim,vocab_size)
    
    def forward(self, idx):
        x = self.embedding(idx)
        x = self.blocks(x)
        x = self.lm_head(x)

        return x
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.embedding.pos_encoding.size(1):]

            logits = self.forward(idx_cond)

            logits = logits[:, -1, :] / temperature 

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, next_token), dim=1)
            
        return idx
