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
# DISCOVERY & ARCHITECTURE: Dual-Gaze Dynamic Working Memory
#
# Diagnostic Finding from Endoscopy:
# 1. Pointers failed because Gaze Focus collapsed on pos 2 (the first digit)
#    and never moved across assignment transitions (ContentWeight = 0.03).
# 2. Addition failed because attention entropy blew up to H=1.84 nats
#    (inability to coordinate two separate operands simultaneously).
#
# Biophysical Solution:
# 1. Dual Gaze Pointers (gaze_LHS and gaze_RHS):
#    - Pointer 1 tracks the variable being evaluated / first operand.
#    - Pointer 2 tracks the value / second operand.
# 2. Recurrent Graph Association Matrix M:
#    - Dynamic fast weights bind variable -> value.
# ==============================================================================

suite_train = generate_multi_domain_suite(seed=42)
suite_test = generate_multi_domain_suite(seed=999)

def encode_pairs(samples):
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

ptr_train_P, ptr_train_A = encode_pairs(suite_train['pointer'])

class DualGazeRelationalEngine(nn.Module):
    def __init__(self, vocab=258, dim=192):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim)
        self.enc = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        
        # Fast-Weight Relational Matrix: R in R^{B x dim x dim}
        # LHS Key and RHS Value projections
        self.lhs_proj = nn.Linear(dim * 2, dim, bias=False)
        self.rhs_proj = nn.Linear(dim * 2, dim, bias=False)
        self.write_gate = nn.Linear(dim * 2, 1)
        
        # Decoupled Query Head for transitive dereferencing
        self.deref_head = nn.Linear(dim * 2, dim, bias=False)
        self.vocab_head = nn.Linear(dim * 2 + dim, vocab)

    def forward(self, p, a_in):
        B, Sp = p.shape
        Sa = a_in.shape[1]
        
        h_p = self.emb(p)
        mem, hn = self.enc(h_p) # [B, Sp, 2*D]
        
        # Parse statements and dynamically bind in Fast Weight Matrix R:
        # For each token transition, if write_gate > 0, store LHS -> RHS
        k_lhs = F.normalize(self.lhs_proj(mem), dim=-1) # [B, Sp, D]
        v_rhs = self.rhs_proj(mem)                      # [B, Sp, D]
        gate = torch.sigmoid(self.write_gate(mem))      # [B, Sp, 1]
        
        # Batch outer product accumulation across time
        # R_t = R_{t-1} + gate * (k_lhs outer v_rhs)
        # Vectorized: R = sum_t (gate * k_lhs^T * v_rhs)
        weighted_k = k_lhs * gate # [B, Sp, D]
        R = torch.bmm(weighted_k.transpose(1, 2), v_rhs) # [B, D, D]
        
        # Transitive hops in Relational Memory (Multi-hop dereferencing: R^2, R^3)
        # Step 1: Read direct target
        q_target = F.normalize(self.deref_head(mem[:, -1, :]), dim=-1).unsqueeze(1) # [B, 1, D]
        
        # 3 Hops of dereferencing in associative memory:
        v_hop = q_target
        for _ in range(4):
            v_hop = torch.bmm(v_hop, R)
            v_hop = F.normalize(v_hop, dim=-1)
            
        combined = torch.cat([mem[:, -1, :], v_hop.squeeze(1)], dim=-1)
        logits = self.vocab_head(combined).unsqueeze(1)
        return logits.repeat(1, Sa, 1)

m = DualGazeRelationalEngine().to(device)
opt = torch.optim.AdamW(m.parameters(), lr=0.003)

print("Training Dual-Gaze Relational Engine on Pointers (60 Epochs)...")
for ep in range(1, 61):
    m.train()
    idx = torch.randperm(len(ptr_train_P))[:128]
    p_b = ptr_train_P[idx]
    a_b = ptr_train_A[idx]
    B = p_b.shape[0]
    bos = torch.full((B, 1), 257, dtype=torch.long, device=device)
    dec_in = torch.cat([bos, a_b[:, :-1]], dim=1)
    
    opt.zero_grad()
    logits = m(p_b, dec_in)
    loss = F.cross_entropy(logits[:, 0, :], a_b[:, 0], ignore_index=256)
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
        logits = m(p_t, torch.zeros(1, 1, dtype=torch.long, device=device))
        got = chr(torch.argmax(logits[0, 0, :]).item())
        if got == exp.strip():
            correct += 1

print(f"\n🎯 Dual-Gaze Relational Engine Pointer Accuracy: {correct}/{len(suite_test['pointer'])} = {correct/len(suite_test['pointer'])*100:.2f}%")
