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

suite_train = generate_multi_domain_suite(seed=42)
suite_test = generate_multi_domain_suite(seed=999)

par_train = suite_train['parity']
par_test = suite_test['parity']

print(f"Parity Train Samples: {len(par_train)} | Test Samples: {len(par_test)}")
for p, a, _ in par_train[:5]:
    print(f"Prompt: '{p}' -> Answer: '{a}'")

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

P_tr, A_tr = encode_task(par_train)

# Test 1: Complex Phasor / Rotator Recurrent Unit
# A recurrent cell that models hidden state on the 2D unit circle:
# z_t = z_{t-1} * exp(i * theta_t)
# For '1', theta = pi -> flip sign!
# For '0' or spaces, theta = 0 -> identity!
class PhasorRotatorCell(nn.Module):
    def __init__(self, vocab=258, dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.theta_proj = nn.Linear(dim, 1) # scalar angle rotation per token
        self.head = nn.Linear(2, vocab)     # from (cos, sin) to character '0' or '1'

    def forward(self, p):
        B, S = p.shape
        x = self.emb(p)
        theta = self.theta_proj(x).squeeze(-1) # [B, S]
        
        # Cumulative phase sum: phi_t = sum_{k=1}^t theta_k
        phi = torch.cumsum(theta, dim=1) # [B, S]
        
        # Unit phasor at the end of the sequence
        cos_phi = torch.cos(phi[:, -1:]) # [B, 1]
        sin_phi = torch.sin(phi[:, -1:]) # [B, 1]
        phasor = torch.cat([cos_phi, sin_phi], dim=-1) # [B, 2]
        
        logits = self.head(phasor) # [B, vocab]
        return logits

m = PhasorRotatorCell().to(device)
opt = torch.optim.AdamW(m.parameters(), lr=0.01)

print("\n--- Training Phasor Rotator on Parity (30 Epochs) ---")
for ep in range(1, 31):
    m.train()
    opt.zero_grad()
    logits = m(P_tr)
    loss = F.cross_entropy(logits, A_tr[:, 0])
    loss.backward()
    opt.step()
    if ep % 10 == 0 or ep == 1:
        print(f"Epoch {ep:02d} | Loss: {loss.item():.4f}")

m.eval()
correct = 0
for p, exp, _ in par_test:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m(p_t)
        got = chr(torch.argmax(logits[0]).item())
        if got == exp.strip():
            correct += 1

acc = (correct / len(par_test)) * 100.0
print(f"🎯 Phasor Rotator Parity Test Accuracy: {correct}/{len(par_test)} = {acc:.2f}%")
