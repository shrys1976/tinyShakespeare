import os
from turtle import forward
import urllib.request

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, functional as F

from main import head_size


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "input.txt"

batch_size = 32
block_size = 8
max_iters = 10000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)


def download_dataset():
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def load_text():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_tokenizer(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    return chars, encode, decode

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads =nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj

    def forward(self,x):

        return torch.cat([h(x) for h in self.heads], dim = -1)



class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(

            nn.Linear(n_embd,n_embd),
            nn.Relu*(),
        )

    def forward(self, x):
        return self.net(x)



class Block(nn.Module):

    # transformer block

    def __init__(self,n_embd, n_head):
        # multi head attentions
        # n_head -> number of embedding dims
        # n_head -> number of attention heads

        super().__init__()
        head_size = n_embd // n_head
        self.sa  =MultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)


    def forward(self, x):
        # residual connection
        x = x+self.sa(x)
        x =x+self.ffwd(x)
        return x



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def get_batch(split, train_data, val_data):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


def main():
    download_dataset()
    text = load_text()
    chars, encode, decode = build_tokenizer(text)
    vocab_size = len(chars)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        if step % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch("train", train_data, val_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
