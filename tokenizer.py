import re
import json
import torch

with open("science_mini.txt", "r") as f:
    text = f.read().lower()

words = sorted(list(set(re.findall(r"[\w']+|[?!,.]|\s", text))))
vocab = {word: i for i, word in enumerate(words)}
vocab["<unk>"] = len(vocab) 

with open("vocab.json", "w") as f:
    json.dump(vocab, f)

class Tokenizer():
    def __init__(self, vocab_file):
        with open(vocab_file, "r") as vocab:
           self.vocab = json.load(vocab)
           self.reverse_vocab = {v : k for k, v in self.vocab.items()}

    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            found = False
            for j in range(len(text), i, -1):
                word = text[i:j]
                if word in self.vocab:
                    tokens.append(self.vocab[word])
                    i = j
                    found = True
                    break
            if not found:
                tokens.append(self.vocab["<unk>"])
                i+=1
            
        return torch.tensor(tokens)
    
    
    def decode(self, ids):
        text = ""
        
        for id in ids:
          val = id.item() if torch.is_tensor(id) else id
          text += self.reverse_vocab[val]
        
        return text
