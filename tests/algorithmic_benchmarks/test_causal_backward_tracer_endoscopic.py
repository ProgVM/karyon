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
# DEEP ENDOSCOPIC MECHANISTIC BREAKTHROUGH:
# String:
# Pos 00: 'a', Pos 01: '=', Pos 02: '2'
# Pos 05: 'b', Pos 06: '=', Pos 07: '8'
# Pos 10: 'c', Pos 11: '=', Pos 12: '4'
# Pos 15: 'd', Pos 16: '=', Pos 17: '9'
# Pos 20: 'c', Pos 21: '=', Pos 22: 'd'
# Pos 25: 'a', Pos 26: '=', Pos 27: 'c'
# Pos 30: 'b', Pos 31: '=', Pos 32: 'a'
# Pos 35: 'c', Pos 36: '='
#
# Query is at Pos 35 ('c').
# What needs to happen?
# At Pos 35, the target variable is 'c'.
# We need to find the LATEST assignment to 'c' BEFORE Pos 35.
# Looking backwards, at Pos 20: 'c' = 'd'.
# The value is 'd' (at Pos 22).
# Now the target variable is 'd'.
# We look backwards before Pos 20 for 'd'.
# At Pos 15: 'd' = '9'.
# The value is '9' (at Pos 17).
# '9' is a digit! Done! Emitted '9'.
#
# Why did standard attention fail?
# Because standard bidirectional GRU blends ALL futures and pasts into every token!
# At Pos 10 ('c=4'), the bidirectional GRU sees Pos 35 ('c='), creating an artificial shortcut loop!
# What happens if we use a STRICT CAUSAL (forward) representation or LOCAL representation,
# and query backwards?
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

# Clean Causal Hopfield Dereferencer
class CausalHopfieldDereferencer(nn.Module):
    def __init__(self, vocab=258, dim=128):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim)
        # 1-layer unidirectional or shallow CNN to avoid future leakage
        self.enc = nn.GRU(dim, dim, batch_first=True, bidirectional=False)
        
        # Identity variable embeddings: 4 variables (a, b, c, d)
        self.var_emb = nn.Embedding(4, dim) # a->0, b->1, c->2, d->3
        
        # Key: identifies variable assigned on LHS
        self.lhs_key = nn.Linear(dim, dim, bias=False)
        # Value: representation of RHS
        self.rhs_val = nn.Linear(dim, dim, bias=False)
        self.head    = nn.Linear(dim, vocab)

    def forward(self, p):
        B, Sp = p.shape
        h_p = self.emb(p)
        mem, _ = self.enc(h_p) # [B, Sp, D]
        
        k_lhs = F.normalize(self.lhs_key(mem), dim=-1) # [B, Sp, D]
        v_rhs = self.rhs_val(mem)                      # [B, Sp, D]
        
        # Query: from position Sp-2 (e.g. 'c' at pos 35)
        # Find token at Sp-2:
        q = F.normalize(mem[:, -2, :], dim=-1).unsqueeze(1) # [B, 1, D]
        
        # Recency bias: monotonically increasing across sequence
        recency = torch.linspace(0.0, 5.0, Sp, device=p.device).unsqueeze(0) # [1, Sp]
        
        p_mask = (p == 256)
        # Also mask out the query position itself (cannot attend to self)
        causal_mask = torch.zeros(B, Sp, dtype=torch.bool, device=p.device)
        causal_mask[:, -2:] = True
        total_mask = p_mask | causal_mask
        
        # Iterative dereferencing (up to 4 hops)
        for _ in range(4):
            sim = torch.bmm(q, k_lhs.transpose(1, 2)).squeeze(1) # [B, Sp]
            # Sharp modern Hopfield temperature (beta=16) + recency bias
            sim = (sim * 16.0) + recency
            sim = sim.masked_fill(total_mask, -1e9)
            
            attn = F.softmax(sim, dim=-1) # [B, Sp]
            
            # Read RHS
            read_v = torch.bmm(attn.unsqueeze(1), v_rhs).squeeze(1) # [B, D]
            q = F.normalize(read_v, dim=-1).unsqueeze(1)
            
        logits = self.head(q.squeeze(1))
        return logits

m = CausalHopfieldDereferencer().to(device)
opt = torch.optim.AdamW(m.parameters(), lr=0.003)

print("Training Causal Hopfield Dereferencer on Pointers (60 Epochs)...")
for ep in range(1, 61):
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
        print(f"Epoch {ep:02d} | Loss: {loss.item():.4f}")

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
print(f"\n🎯 Causal Hopfield Dereferencer Accuracy: {correct}/{len(suite_test['pointer'])} = {acc:.2f}%")
for p, exp, _ in suite_test['pointer'][:8]:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m(p_t)
        got = chr(torch.argmax(logits[0]).item())
        ok = '✅' if got == exp.strip() else '❌'
        print(f"  {ok} '{p}' -> Target: '{exp.strip()}' | Got: '{got}'")
