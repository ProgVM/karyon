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
# DISCOVERY: The Hopfield Attractor Dereferencing Basin
# Why did Softmax Attention fail (22%)?
# Because Softmax is a "mushy" continuous average.
# In "a=2; b=8; c=4; d=9; c=d; a=c; b=a; c=":
# If attention distributes 40% on 'd=9' and 60% on 'b=8', the linear combination
# is an uninterpretable blend in embedding space!
#
# But Continuous Modern Hopfield Attractors (Demircigil / Ramsauer / Krotov 2020)
# possess an Inverse Temperature beta -> inf that produces SHARP ATTRACTOR SNAPPING!
#
# Moreover, causal recency matters: the LATEST assignment must take precedence!
# Let's test a Sharp Modern Hopfield Dereferencing Core with Recency Bias!
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

class SharpHopfieldProgramCortex(nn.Module):
    def __init__(self, vocab=258, dim=128, beta=32.0, hops=5):
        super().__init__()
        self.dim = dim
        self.beta = beta
        self.hops = hops
        self.emb = nn.Embedding(vocab, dim)
        self.enc = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        
        self.lhs_key = nn.Linear(dim * 2, dim, bias=False)
        self.rhs_val = nn.Linear(dim * 2, dim, bias=False)
        self.q_proj  = nn.Linear(dim * 2, dim, bias=False)
        self.head    = nn.Linear(dim, vocab)

    def forward(self, p):
        B, Sp = p.shape
        h_p = self.emb(p)
        mem, _ = self.enc(h_p) # [B, Sp, 2*D]
        
        k_lhs = F.normalize(self.lhs_key(mem), dim=-1) # [B, Sp, D]
        v_rhs = self.rhs_val(mem)                      # [B, Sp, D]
        
        # Position recency bias: later positions have monotonically higher base energy
        # recency: [1, Sp], scaled between 0.0 and 2.0
        recency = torch.linspace(0.0, 3.0, Sp, device=p.device).unsqueeze(0) # [1, Sp]
        
        # Initial query from prompt end (e.g. 'c' in 'c=')
        q = F.normalize(self.q_proj(mem[:, -2, :]), dim=-1).unsqueeze(1) # [B, 1, D]
        
        p_mask = (p == 256)
        
        # Hopfield Iterative Relaxation:
        for _ in range(self.hops):
            # Energy calculation with Sharp Beta and Recency Prior
            sim = torch.bmm(q, k_lhs.transpose(1, 2)).squeeze(1) # [B, Sp] in [-1, 1]
            sim = (sim + 0.5 * recency) * self.beta
            sim = sim.masked_fill(p_mask, -1e9)
            
            # Sharp modern Hopfield retrieval
            attn = F.softmax(sim, dim=-1) # [B, Sp]
            
            # Update state into attractor basin
            read_v = torch.bmm(attn.unsqueeze(1), v_rhs).squeeze(1) # [B, D]
            q = F.normalize(read_v, dim=-1).unsqueeze(1)
            
        logits = self.head(q.squeeze(1))
        return logits

m = SharpHopfieldProgramCortex(beta=24.0, hops=4).to(device)
opt = torch.optim.AdamW(m.parameters(), lr=0.003)

print("Training Sharp Hopfield Program Cortex on Pointers (60 Epochs)...")
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
print(f"\n🎯 Sharp Hopfield Program Cortex Accuracy: {correct}/{len(suite_test['pointer'])} = {acc:.2f}%")
for p, exp, _ in suite_test['pointer'][:8]:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m(p_t)
        got = chr(torch.argmax(logits[0]).item())
        ok = '✅' if got == exp.strip() else '❌'
        print(f"  {ok} '{p}' -> Target: '{exp.strip()}' | Got: '{got}'")
