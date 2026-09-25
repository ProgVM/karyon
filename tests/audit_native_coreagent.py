import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import karyon_core
from karyon_agent import CoREAgent
from multi_domain_benchmark import generate_multi_domain_suite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Production CoREAgent v35.0 (C++20 Causal SSD + DynamicMorphicGraph Spatiotemporal Engine)
agent = CoREAgent(vocab_size=258, embed_dim=128, device=str(device))
# Add native primitives for rich latent deliberation
agent.add_node("bilinear", "BilinearMultiplicative", False, 1.0)
agent.add_node("hopfield", "ContinuousHopfield", False, 1.0)

print(f"Agent initialized with Spatiotemporal Engine. Total nodes: {agent.graph.k_nodes}")
print(f"Total optimizable parameters: {len(list(agent.parameters()))}")

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

opt = torch.optim.AdamW(agent.parameters(), lr=0.003)

print("\n--- Training CoREAgent v35.0 (80 Epochs) ---")
for ep in range(1, 81):
    agent.train()
    idx = torch.randperm(len(P_train))[:128]
    p_b = P_train[idx]
    a_b = A_train[idx]
    
    full_seq = torch.cat([p_b, a_b], dim=1) # [B, S]
    targets = full_seq[:, 1:].clone()
    targets[:, :p_b.shape[1]-1] = 256 # Mask out prompt
    
    opt.zero_grad()
    logits = agent(full_seq[:, :-1], thinking_steps=3) # [B, S-1, 258]
    loss = F.cross_entropy(logits.reshape(-1, 258), targets.reshape(-1), ignore_index=256)
    loss.backward()
    opt.step()
    
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:02d} | CrossEntropy: {loss.item():.4f}")

# Evaluation across all domains
agent.eval()
print("\n" + "=" * 80)
print("=== PRODUCTION CoREAgent MULTI-DOMAIN EVALUATION ===")
print("=" * 80)

for domain, samples in suite_test.items():
    correct = 0
    total = len(samples)
    for p, exp, _ in samples:
        p_b = [ord(c) for c in p]
        curr_seq = torch.tensor([p_b], dtype=torch.long, device=device)
        gen = []
        with torch.no_grad():
            for _ in range(len(exp) + 3):
                logits = agent(curr_seq, thinking_steps=3)
                nxt = torch.argmax(logits[0, -1, :]).item()
                if nxt in (ord('\n'), 256, 257):
                    break
                gen.append(nxt)
                curr_seq = torch.cat([curr_seq, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
        got = bytes(gen).decode('utf-8', errors='ignore')
        if got.strip() == exp.strip():
            correct += 1
    acc = (correct / total) * 100.0
    print(f"Domain [{domain:10s}]: {correct:3d}/{total} ({acc:.2f}%)")

print("=" * 80)
