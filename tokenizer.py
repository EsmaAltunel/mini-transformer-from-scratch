import re
import json
import torch

with open("data.txt", "r") as f:
    text = f.read().lower()

words = sorted(list(set(re.findall(r"[\w']+|[?!,.]|\s", text))))
vocab = {word: i for i, word in enumerate(words)}
vocab["<unk>"] = len(vocab) 

with open("vocab.json", "w") as f:
    json.dump(vocab, f)

class Tokenizer():
    def __init__(self, vocab_file):
        with open(vocab_file, "r") as f:
            self.vocab = json.load(f)
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        words_in_text = re.findall(r"[\w']+|[?!,.]|\s", text.lower())
        
        tokens = []
        for word in words_in_text:
            tokens.append(self.vocab.get(word, self.vocab["<unk>"]))
            
        return torch.tensor(tokens)
    
    def decode(self, ids):
        res = []
        for id in ids:
            val = id.item() if torch.is_tensor(id) else id
            res.append(self.reverse_vocab.get(val, "<unk>"))
        return "".join(res)