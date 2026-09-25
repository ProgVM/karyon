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
# RIGOROUS MATHEMATICAL AUDIT: How should pointers be resolved?
#
# A program consists of tokens.
# Let tokens be: a, =, 2, ;, b, =, 8, ...
# In ANY programming language, an interpreter maintains:
# env = {}
# For statement 'X = Y':
#   if Y is in env:
#     env[X] = env[Y]
#   else:
#     env[X] = Y
#
# How does a biological or digital neural substrate do this natively?
# A 2D Matrix of Synaptic Weights W in R^{N x N}:
# Variables: 'a', 'b', 'c', 'd' (indices 0..3)
# Values: '0'..'9' (indices 4..13)
#
# Each statement:
# 1. Reads LHS node index i
# 2. Reads RHS node index j
# 3. Sets: W[i, :] = W[j, :] (if j is a variable), OR W[i, j] = 1 (if j is a digit)
#
# Can an end-to-end neural network learn this mapping without any manual hardcoding?
# Let's test a Slot-Based Key-Value Program Memory with Differentiable Hard Gating (Gumbel/Straight-Through)!
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

# Let's test a 2-Layer Transformer with Rotary Position Embeddings and Multi-Head Cross Attention
class RelationalTransformerProgrammer(nn.Module):
    def __init__(self, vocab=258, dim=192, heads=6, layers=4):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, 64, dim) * 0.02)
        
        # We use standard Pre-LayerNorm Transformer with high expressive feedforward capacity
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=768, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        
        # Query readout
        self.head = nn.Linear(dim, vocab)

    def forward(self, p):
        B, Sp = p.shape
        h = self.emb(p) + self.pos[:, :Sp, :]
        p_mask = (p == 256)
        
        # Self-attention across entire program context
        out = self.transformer(h, src_key_padding_mask=p_mask) # [B, Sp, D]
        
        # Read from query position (Sp-2: 'c')
        q_rep = out[:, -2, :]
        return self.head(q_rep)

m = RelationalTransformerProgrammer(layers=4).to(device)
opt = torch.optim.AdamW(m.parameters(), lr=0.001)

print("Training 4-Layer Relational Transformer on Pointers (100 Epochs)...")
for ep in range(1, 101):
    m.train()
    idx = torch.randperm(len(ptr_train_P))[:128]
    p_b = ptr_train_P[idx]
    a_b = ptr_train_A[idx]
    
    opt.zero_grad()
    logits = m(p_b)
    loss = F.cross_entropy(logits, a_b[:, 0])
    loss.backward()
    opt.step()
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:03d} | Loss: {loss.item():.4f}")

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
print(f"\n🎯 4-Layer Relational Transformer Accuracy: {correct}/{len(suite_test['pointer'])} = {acc:.2f}%")
for p, exp, _ in suite_test['pointer'][:8]:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m(p_t)
        got = chr(torch.argmax(logits[0]).item())
        ok = '✅' if got == exp.strip() else '❌'
        print(f"  {ok} '{p}' -> Target: '{exp.strip()}' | Got: '{got}'")
