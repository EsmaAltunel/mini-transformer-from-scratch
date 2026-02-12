import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer

my_tokenizer = Tokenizer("vocab.json")

class TextDataset(Dataset):

    def __init__(self, file, my_tokenizer, context_length):
        super().__init__()
        
        with open(file, "r") as f:
            text = f.read()
        
        self.data = my_tokenizer.encode(text).detach().clone().long()
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length
    
    def __getitem__(self, index):
        x = self.data[index : index + self.context_length]
        y = self.data[index + 1 : index + self.context_length + 1]
        return x, y
    

dataset = TextDataset("data.txt", my_tokenizer, context_length=16)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

inputs, targets = next(iter(train_loader))

