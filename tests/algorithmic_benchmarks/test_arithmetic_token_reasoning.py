import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_domain_benchmark import generate_multi_domain_suite

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Let's inspect: Why does 400 samples fail on raw byte addition?
# 400 samples is EXTREMELY SMALL for raw byte addition (there are 1,000,000 pairs of 3-digit numbers).
# In standard arithmetic benchmarks (e.g. Naurzbayeva et al., Charton 2021, Lee et al. 2023),
# addition training requires at least 5,000 - 10,000 examples, OR a specialized scratchpad.
# 
# What if we give 2,000 training samples for addition?
# Let's test the scaling law of sample efficiency!

def generate_addition_dataset(num_samples=2000, seed=42):
    random.seed(seed)
    samples = []
    for _ in range(num_samples):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        samples.append((f"{a} + {b} = ", str(a + b), "addition"))
    return samples

train_samples = generate_addition_dataset(num_samples=2500, seed=42)
test_samples = generate_addition_dataset(num_samples=400, seed=999)

def encode_task(samples):
    max_p = max(len(p) for p, _, _ in samples)
    max_a = max(len(a) for _, a, _ in samples) + 1
    P = torch.full((len(samples), max_p), 256, dtype=torch.long, device=device)
    A = torch.full((len(samples), max_a), 256, dtype=torch.long, device=device)
    for i, (p, a, _) in enumerate(samples):
        p_b = [ord(c) for c in p]
        a_b = [ord(c) for c in a] + [ord('\n')]
        P[i, :len(p_b)] = torch.tensor(p_b, dtype=torch.long, device=device)
        A[i, :len(a_b)] = torch.tensor(a_b, dtype=torch.long, device=device)
    return P, A

P_tr, A_tr = encode_task(train_samples)

# A compact 4-layer Transformer with Rotary / Learned Positional Embeddings
class ArithmeticTransformer(nn.Module):
    def __init__(self, vocab=258, dim=128, num_layers=4):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim)
        self.pos_p = nn.Parameter(torch.randn(1, 32, dim) * 0.02)
        self.pos_a = nn.Parameter(torch.randn(1, 16, dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=512, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=8, dim_feedforward=512, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(dim, vocab)

    def forward(self, p, a_in):
        B, Sp = p.shape
        Sa = a_in.shape[1]
        
        h_p = self.emb(p) + self.pos_p[:, :Sp, :]
        p_mask = (p == 256)
        
        mem = self.encoder(h_p, src_key_padding_mask=p_mask)
        h_a = self.emb(a_in) + self.pos_a[:, :Sa, :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(Sa, device=p.device)
        
        out = self.decoder(h_a, mem, tgt_mask=tgt_mask, memory_key_padding_mask=p_mask)
        return self.head(out)

m = ArithmeticTransformer().to(device)
opt = torch.optim.AdamW(m.parameters(), lr=0.001)

print(f"Training on 2,500 samples (50 Epochs)...")
for ep in range(1, 51):
    m.train()
    idx = torch.randperm(len(P_tr))[:256]
    p_b = P_tr[idx]
    a_b = A_tr[idx]
    B = p_b.shape[0]
    bos = torch.full((B, 1), 257, dtype=torch.long, device=device)
    dec_in = torch.cat([bos, a_b[:, :-1]], dim=1)
    
    opt.zero_grad()
    logits = m(p_b, dec_in)
    loss = F.cross_entropy(logits.reshape(-1, 258), a_b.reshape(-1), ignore_index=256)
    loss.backward()
    opt.step()
    if ep % 10 == 0 or ep == 1:
        print(f"Epoch {ep:02d} | Loss: {loss.item():.4f}")

m.eval()
correct = 0
for p, exp, _ in test_samples:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    gen = []
    curr_tok = torch.tensor([[257]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(len(exp) + 2):
            logits = m(p_t, curr_tok)
            nxt = torch.argmax(logits[0, -1, :]).item()
            if nxt in (ord('\n'), 256, 257):
                break
            gen.append(nxt)
            curr_tok = torch.cat([curr_tok, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            
    got = bytes(gen).decode('utf-8', errors='ignore').strip()
    if got == exp.strip():
        correct += 1

print(f"\n🎯 Arithmetic Accuracy with 2,500 samples: {correct}/{len(test_samples)} = {correct/len(test_samples)*100:.2f}%")
