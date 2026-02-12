import re
import json
import torch
import string


with open("data.txt", "r") as f:
    text = f.read().lower()

tokens = re.findall(r"[a-zA-Z']+|[0-9]+|[?!,.]| ", text)
words = sorted(list(set(tokens)))

vocab = {word: i for i, word in enumerate(words)}

for ch in string.ascii_lowercase:
    if ch not in vocab:
        vocab[ch] = len(vocab)

for ch in string.digits:
    if ch not in vocab:
        vocab[ch] = len(vocab)

basic_tokens = [" ", ".", ",", "?", "!", "'", "\n"] 
for tok in basic_tokens:
    if tok not in vocab:
        vocab[tok] = len(vocab)

special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
for tok in special_tokens:
    vocab[tok] = len(vocab)

with open("vocab.json", "w") as f:
    json.dump(vocab, f)

class Tokenizer():
    def __init__(self, vocab_file):
        with open(vocab_file, "r") as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):

        words = re.findall(r"[a-zA-Z']+|[0-9]+|[?!,.]|\s+", text.lower())
        
        tokens = []
        for word in words:
            i = 0
            while i < len(word):
                found_match = False
                for j in range(len(word), i, -1):
                    sub_word = word[i:j]
                    if sub_word in self.vocab:
                        tokens.append(self.vocab[sub_word])
                        i = j
                        found_match = True
                        break
                if not found_match:
                    ch = word[i]
                    tokens.append(self.vocab.get(ch, self.vocab["<unk>"]))
                    i += 1
        return torch.tensor(tokens)

    def decode(self, ids):
        res = []
        for id in ids:
            val = id.item() if torch.is_tensor(id) else id
            res.append(self.reverse_vocab.get(val, "<unk>"))
        return "".join(res)
