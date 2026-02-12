import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import TextDataset
from model import Model
from tokenizer import Tokenizer
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
context_length = 128
batch_size = 16
embedding_dim = 128
num_heads = 4
num_layers = 6 
epochs = 100

my_tokenizer = Tokenizer("vocab.json")
vocab_size = len(my_tokenizer.vocab) 


dataset = TextDataset("data.txt", my_tokenizer, context_length)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Model(vocab_size, embedding_dim, context_length, num_heads, num_layers).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
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
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='tab:red')
    plt.title("Swiftie AI - Model Eğitim Kaybı (Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show() 

if __name__ == "__main__":
    print(f"Eğitim başlıyor... Cihaz: {device}")
    train()

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': my_tokenizer.vocab,
        'config': {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'context_length': context_length,
            'num_heads': num_heads,
            'num_layers': num_layers
        }
    }
    torch.save(checkpoint, "model.pth")

    model.eval()
    print("\n--- Model Hazır! Taylor modunda konuşmaya başlayabilirsin ---")
    print("(Çıkmak için 'q' yaz)")
    
    while True:
      prompt = input("\nSen: ")
      if prompt.lower() == 'q': break

      input_ids = my_tokenizer.encode(prompt).unsqueeze(0).to(device)
    
      with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=60, temperature=0.7, top_k=30)
 
      cevap = my_tokenizer.decode(output_ids[0])
      print(f"Swiftie AI: {cevap}")