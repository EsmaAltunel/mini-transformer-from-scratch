import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import TextDataset
from model import Model
from tokenizer import Tokenizer
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"
context_length = 128
batch_size = 4
embedding_dim = 64
num_heads = 4
num_layers = 6 
epochs = 150

my_tokenizer = Tokenizer("vocab.json")
vocab_size = len(my_tokenizer.vocab) 

dataset = TextDataset("data.txt", my_tokenizer, context_length)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Model(vocab_size, embedding_dim, context_length, num_heads, num_layers).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x) 
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Model Eğitim Kaybı")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show() #

if __name__ == "__main__":
    train()


# Modeli değerlendirme moduna al
model.eval()

# 1. Test senaryoları (Farklı şarkılardan başlangıçlar)
test_prompts = [
    "i remember it",           # All Too Well
    "this love is",             # This Love
    "champagne",                # Champagne Problems
    "i walked through the door" # All Too Well
]

print("-" * 30)
print("🎸 VICTUS SWIFTIE MODEL TEST 🎸")
print("-" * 30)

with torch.no_grad(): # Gradyan hesaplamayı kapat, VRAM tasarrufu yap [cite: 2026-02-05]
    for soru in test_prompts:
        # Senin yeni tokenizer'ınla encode ediyoruz
        input_ids = my_tokenizer.encode(soru).unsqueeze(0).to(device)
        
        # max_new_tokens'ı 40 yapalım ki şarkı sözü akıp gitsin
        output_ids = model.generate(input_ids, max_new_tokens=40)
        
        cevap = my_tokenizer.decode(output_ids[0])
        
        print(f"\nPrompt: '{soru}'")
        print(f"AI: {cevap}")
        print("-" * 20)