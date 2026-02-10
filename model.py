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
    
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            # Sadece modelin görebileceği kadar (context_length) veriyi kesip alıyoruz
            idx_cond = idx[:, -self.embedding.pos_encoding.size(1):]
            
            # İleri besleme (Forward)
            logits = self.forward(idx_cond)
            
            # Sadece en son zaman adımındaki tahmine odaklanıyoruz
            logits = logits[:, -1, :] # (Batch, Vocab_Size)
            
            # Olasılıkları hesapla ve en yüksek olanı seç (Greedy Search)
            probs = torch.softmax(logits, dim=-1)
            _, next_token = torch.max(probs, dim=-1, keepdim=True)
            
            # Yeni kelimeyi mevcut kelimelerin yanına ekle
            idx = torch.cat((idx, next_token), dim=1)
            
        return idx

