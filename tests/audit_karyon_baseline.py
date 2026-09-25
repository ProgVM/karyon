import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import karyon_core
from multi_domain_benchmark import generate_multi_domain_suite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==============================================================================
# AUDITING NATIVE KARYON-CORE ARCHITECTURE ON MULTI-DOMAIN SUITE
# Native substrate:
# 1. Byte Embedding (V=258, D=128)
# 2. C++20 CausalParallelSSD (Continuous linear state space time-mixing)
# 3. DynamicMorphicGraph (Recurrent thinking cycles across dynamic mathematical operators)
# ==============================================================================

suite_train = generate_multi_domain_suite(seed=42)
suite_test = generate_multi_domain_suite(seed=999)

all_train = []
for d, s in suite_train.items():
    all_train.extend(s)
random.seed(42)
random.shuffle(all_train)

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

P_train, A_train = encode_pairs(all_train)

class NativeKaryonEngine(nn.Module):
    def __init__(self, vocab_size=258, dim=128, thinking_steps=4):
        super().__init__()
        self.dim = dim
        self.thinking_steps = thinking_steps
        self.emb = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.emb.weight, 0.0, 0.02)
        
        # Native C++20 CausalParallelSSD
        self.ssd = karyon_core.CausalParallelSSD(dim, str(device))
        
        # Native C++20 DynamicMorphicGraph
        self.graph = karyon_core.DynamicMorphicGraph(dim, str(device))
        self.graph.add_node("acc", "LinearAccumulator", True, 1.0)
        self.graph.add_node("sat", "SaturatedAttractor", True, 1.0)
        self.graph.add_node("bilinear", "BilinearMultiplicative", False, 1.0)
        self.graph.add_node("hopfield", "ContinuousHopfield", False, 1.0)
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, input_seq):
        # input_seq: [B, S]
        B, S = input_seq.shape
        x = self.emb(input_seq)
        
        # 1. State-Space Temporal Mixing
        h_seq = self.ssd.forward(x) # [B, S, D]
        
        # 2. Recurrent Thinking across Dynamic Operators
        h_flat = h_seq.reshape(B * S, self.dim)
        h_delib = self.graph.forward(h_flat, self.thinking_steps).reshape(B, S, self.dim)
        
        # 3. Readout
        logits = self.head(self.norm(h_delib))
        return logits

m = NativeKaryonEngine().to(device)
opt = torch.optim.AdamW(m.parameters(), lr=0.003)

print("\n--- Training Native Karyon Architecture (80 Epochs) ---")
# Autoregressive sequence training
for ep in range(1, 81):
    m.train()
    idx = torch.randperm(len(P_train))[:128]
    p_b = P_train[idx]
    a_b = A_train[idx]
    
    # Concatenate prompt + answer
    full_seq = torch.cat([p_b, a_b], dim=1) # [B, Sp + Sa]
    targets = full_seq[:, 1:].clone()
    targets[:, :p_b.shape[1]-1] = 256 # Mask out prompt loss
    
    opt.zero_grad()
    logits = m(full_seq[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, 258), targets.reshape(-1), ignore_index=256)
    loss.backward()
    opt.step()
    
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:02d} | CrossEntropy: {loss.item():.4f}")

# Evaluate across all 5 domains natively
m.eval()
print("\n" + "=" * 80)
print("=== NATIVE KARYON EVALUATION ACROSS ALL DOMAINS ===")
print("=" * 80)

domain_results = {}
for domain, samples in suite_test.items():
    correct = 0
    total = len(samples)
    for p, exp, _ in samples:
        p_b = [ord(c) for c in p]
        curr_seq = torch.tensor([p_b], dtype=torch.long, device=device)
        gen = []
        with torch.no_grad():
            for _ in range(len(exp) + 3):
                logits = m(curr_seq)
                nxt = torch.argmax(logits[0, -1, :]).item()
                if nxt in (ord('\n'), 256, 257):
                    break
                gen.append(nxt)
                curr_seq = torch.cat([curr_seq, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
        got = bytes(gen).decode('utf-8', errors='ignore')
        if got.strip() == exp.strip():
            correct += 1
    acc = (correct / total) * 100.0
    domain_results[domain] = acc
    print(f"Domain [{domain:10s}]: {correct:3d}/{total} ({acc:.2f}%)")

print("=" * 80)
