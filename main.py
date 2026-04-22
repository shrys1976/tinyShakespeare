#!/usr/bin/env python
# coding: utf-8

import os
import urllib.request

if not os.path.exists('input.txt'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', 'input.txt')


with open('input.txt', 'r',encoding = 'utf-8') as f:
    text = f.read()


print(text[:1000])


# all the unique chars occuring int the corpus
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)


#tokenizing
# encoder and decoder just like makemore architecture

stoi = {ch:i for i , ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

#print(stoi)

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
#print(encode("hello"))


import torch
data = torch.tensor(encode(text), dtype = torch.long)
print(data.shape, data.dtype)
print(data[:100])



n = int(0.9*(len(data)))
train_data = data[:n]
val_data = data[n:]




torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 10000
eval_interval = 300
eval_iters = 200
batch_size = 4 # number of independent sequences
block_size = 8

def get_batch(split):
    # generates a small btach of data
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)- block_size,(batch_size,)) # random offsets 
    x = torch.stack([data[i:i+block_size]for i in ix] )
    y = torch.stack([data[i+1:i+block_size+1] for i in ix] )
    x, y = x.to(device), y.to(device)
    return x,y 


# average out loss over multiple batches 

@torch.no_grad()
def estimate_loss():
    out ={}
    model.eval() # setting model to eval phase
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    model.train() # setting model to training phase
    return out     

xb, yb = get_batch('train')

#print('inputs:')
#print(xb.shape)
#print('targets')
#print(yb.shape)
#print('---------')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b:t+1]
        target = yb[b:t]


import torch 
import torch.nn as nn
from torch.nn import  functional as F

torch.manual_seed(42)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.lm_head = nn.Linear(n_emb,vocab_size)

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T,device = device))
        x = tok_emb + pos_emb
        logits=  self.lm_head(tok_emb) # batch, time, channel
        
        if targets is None:
            loss = None

        else:    
            # B -> batch size, T-> context length(number of tokens)
            B,T,C = logits.shape
            logits = logits.view(B*T, C) # torch expects logits to have C as the second parameter
            targets = targets.view(B*T) # same reason to change the view as above
            loss = F.cross_entropy(logits,targets)
        return logits, loss

# idx is some intial token sequence, for starting the prediction
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)# forward pass 
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim =1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim =1) # (B, T+1)

        return idx   

m = BigramLanguageModel()     
m = m.to(device)
model = m
logits, loss = m(xb,yb)
print(logits.shape)   
print(loss)

print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long, device=device),max_new_tokens = 100)[0].tolist()))


optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)
batch_size = 32

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']: .4f}, val loss {losses['val']: .4f}")


    xb,yb = get_batch('train')


    # evaluate the loss    
    logits , loss =m(xb,yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()


#print(loss.item())

# generate from the model
context = torch.zeros((1,1), dtype = torch.long, device =device)
print(decode(m.generate(context,max_new_tokens = 500)[0].tolist()))

