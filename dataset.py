import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer

my_tokenizer = Tokenizer("vocab.json")

class TextDataset(Dataset):

    def __init__(self, file, my_tokenizer, context_length):
        super().__init__()
        
        with open(file, "r") as f:
            text = f.read()
        
        data = my_tokenizer.encode(text)
        
        self.inputs = []
        self.targets = []

        for i in range(0, len(data) - context_length):
            input = data[i : i + context_length]
            target = data[i + 1 : i + 1 + context_length]

            self.inputs.append(input)
            self.targets.append(target) 

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
    

dataset = TextDataset("science_mini.txt", my_tokenizer, context_length=16)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

inputs, targets = next(iter(train_loader))

print("Girdi Paketi Boyutu:", inputs.shape)  
print("Hedef Paketi Boyutu:", targets.shape)