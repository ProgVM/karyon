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
# DEEP MECHANISTIC AUDIT (Principle 23): Why do neural nets struggle with Pointers?
#
# Consider: "a=2; b=8; c=4; d=9; c=d; a=c; b=a; c="
# A symbolic program has 4 variables: a, b, c, d.
# When a statement "c=d" occurs:
# Memory must OVERWRITE register 'c' with the CURRENT value of 'd'.
#
# What happens in a standard continuous neural net?
# 1. Outer products (sum_t k^T v) ACCUMULATE, they DO NOT OVERWRITE.
#    Old values (c=4) remain in memory and blur with new values (c=d=9)!
# 2. To OVERWRITE, a neural network needs an Explicit Erasure Gate:
#    M_t = M_{t-1} * (I - e_t outer k_t) + (w_t outer v_t)
#    This is exactly the Fast-Weight Programmers (Schmidhuber 1992) / Neural Turing Machines (Graves 2014) / Mamba-2 / Titans!
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

# Let's test a Differentiable Neural Turing Register with explicit ERASURE GATE:
class DifferentiableOverwritingCortex(nn.Module):
    def __init__(self, vocab=258, dim=128, num_slots=4):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots # 4 variables: a, b, c, d
        self.emb = nn.Embedding(vocab, dim)
        self.enc = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        
        # Variable slot routing: maps char to slot index (a->0, b->1, c->2, d->3)
        self.slot_write = nn.Linear(dim * 2, num_slots)
        self.erase_gate = nn.Linear(dim * 2, 1)
        self.write_vec  = nn.Linear(dim * 2, dim)
        
        # Read from existing slots to resolve RHS (e.g. if RHS is variable 'd')
        self.slot_read = nn.Linear(dim * 2, num_slots)
        self.rhs_mix   = nn.Linear(dim * 2, 1) # decides whether to write literal or read value
        
        self.query_slot = nn.Linear(dim * 2, num_slots)
        self.head = nn.Linear(dim, vocab)

    def forward(self, p):
        B, Sp = p.shape
        h_p = self.emb(p)
        mem, _ = self.enc(h_p) # [B, Sp, 2*D]
        
        # Explicit Memory Matrix M: [B, num_slots, dim]
        # Initialized to zeros
        M = torch.zeros(B, self.num_slots, self.dim, device=p.device)
        
        for t in range(Sp):
            x_t = mem[:, t, :]
            
            # Read from memory (resolving RHS if it is a variable)
            r_weights = F.softmax(self.slot_read(x_t), dim=-1).unsqueeze(1) # [B, 1, num_slots]
            val_from_mem = torch.bmm(r_weights, M).squeeze(1) # [B, dim]
            
            # Literal value from token
            val_from_token = self.write_vec(x_t)
            mix = torch.sigmoid(self.rhs_mix(x_t))
            new_val = mix * val_from_token + (1.0 - mix) * val_from_mem
            
            # Write to slot with ERASURE:
            # M_new[slot] = (1 - erase) * M_old[slot] + write_val
            w_weights = F.softmax(self.slot_write(x_t), dim=-1).unsqueeze(-1) # [B, num_slots, 1]
            erase = torch.sigmoid(self.erase_gate(x_t)).unsqueeze(-1) # [B, 1, 1]
            
            # Erase target slot and add new representation
            M = (1.0 - w_weights * erase) * M + (w_weights * erase) * new_val.unsqueeze(1)
            
        # At query time, read from queried slot
        q_weights = F.softmax(self.query_slot(mem[:, -1, :]), dim=-1).unsqueeze(1) # [B, 1, num_slots]
        res = torch.bmm(q_weights, M).squeeze(1) # [B, dim]
        logits = self.head(res)
        return logits

m = DifferentiableOverwritingCortex().to(device)
opt = torch.optim.AdamW(m.parameters(), lr=0.003)

print("Training Differentiable Overwriting Cortex on Pointers (60 Epochs)...")
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

print(f"\n🎯 Differentiable Overwriting Cortex Accuracy: {correct}/{len(suite_test['pointer'])} = {correct/len(suite_test['pointer'])*100:.2f}%")
for p, exp, _ in suite_test['pointer'][:8]:
    p_b = [ord(c) for c in p]
    p_t = torch.tensor([p_b], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = m(p_t)
        got = chr(torch.argmax(logits[0]).item())
        ok = '✅' if got == exp.strip() else '❌'
        print(f"  {ok} '{p}' -> Target: '{exp.strip()}' | Got: '{got}'")
