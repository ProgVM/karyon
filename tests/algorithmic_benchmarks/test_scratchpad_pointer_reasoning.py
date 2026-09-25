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

# ==============================================================================
# DISCOVERY & REVOLUTION: The Power of Intermediate Scratchpad Steps
#
# Look at what we tested so far:
# Single-pass direct query: "a=2; b=8; c=4; d=9; c=d; a=c; b=a; c=" -> "9" (28.75% max)
#
# WHY?
# Because resolving 3 hops of indirection (c -> d -> 9) in a single feedforward pass
# requires 3 composition layers working in perfect numerical sync.
#
# BUT if the model generates an INTERMEDIATE LATENT SCRATCHPAD TRACE:
# Step 1: "c=d" -> resolves c to 9
# Step 2: "a=c" -> resolves a to 9
# Step 3: "b=a" -> resolves b to 9
# Query c -> "9"!
#
# What if we give the Transformer a 4-step Latent Recurrent Loop (Recurrent Thinking Time),
# where the internal hidden states update 4 times before emitting the answer?
# ==============================================================================

suite_train = generate_multi_domain_suite(seed=42)
suite_test = generate_multi_domain_suite(seed=999)

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

ptr_train_P, ptr_train_A = encode_task(suite_train['pointer'])

class RecurrentThinkingTransformer(nn.Module):
    def __init__(self, vocab=258, dim=192, heads=6, thinking_cycles=4):
        super().__init__()
        self.dim = dim
        self.thinking_cycles = thinking_cycles
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, 64, dim) * 0.02)
        
        # 1-layer Recurrent Transformer Block applied recursively K times!
        self.block = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=768, batch_first=True, norm_first=True)
        self.norm  = nn.LayerNorm(dim)
        
        self.head = nn.Linear(dim, vocab)

    def forward(self, p):
        B, Sp = p.shape
        h = self.emb(p) + self.pos[:, :Sp, :]
        p_mask = (p == 256)
        
        # Recurrent Latent Thinking Cycles:
        # Instead of stacking 12 separate layers, we recirculate internal representations through 1 block K times!
        # This allows variable depth (K=1, K=2, K=4, K=8) depending on problem complexity!
        state = h
        for cycle in range(self.thinking_cycles):
            state = self.block(state, src_key_padding_mask=p_mask)
            
        state = self.norm(state)
        q_rep = state[:, -2, :]
        return self.head(q_rep)

print("--- Testing Recurrent Thinking Cycles (K = 1, 2, 4, 8) ---")
for k_cycles in [1, 2, 4, 6]:
    m = RecurrentThinkingTransformer(thinking_cycles=k_cycles).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001)
    
    for ep in range(1, 81):
        m.train()
        idx = torch.randperm(len(ptr_train_P))[:128]
        p_b = ptr_train_P[idx]
        a_b = ptr_train_A[idx]
        
        opt.zero_grad()
        logits = m(p_b)
        loss = F.cross_entropy(logits, a_b[:, 0])
        loss.backward()
        opt.step()
        
    m.eval()
    correct = 0
    for p, exp, _ in suite_test['pointer']:
        p_b = [ord(c) for c in p]
        p_t = torch.tensor([p_b], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = m(p_t)
            got = chr(torch.argmax(logits[0]).item())
            if got == exp.strip():
                correct += 1
                
    acc = (correct / len(suite_test['pointer'])) * 100.0
    print(f"Cycles K = {k_cycles} | Pointer Accuracy: {correct}/400 = {acc:.2f}% | Final Train Loss: {loss.item():.4f}")

