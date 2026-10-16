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

suite = generate_multi_domain_suite(seed=42)
pointer_data = suite['pointer'][:100]

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

P_data, A_data = encode_pairs(pointer_data)

agent = CoREAgent(vocab_size=258, embed_dim=128, device=str(device))
opt = torch.optim.AdamW(agent.parameters(), lr=0.01)

print("\n--- Overfitting Single Domain (Pointer) on 100 samples ---")
for ep in range(1, 101):
    agent.train()
    full_seq = torch.cat([P_data, A_data], dim=1)
    targets = full_seq[:, 1:].clone()
    targets[:, :P_data.shape[1]-1] = 256
    
    opt.zero_grad()
    logits = agent(full_seq[:, :-1], thinking_steps=3)
    loss = F.cross_entropy(logits.reshape(-1, 258), targets.reshape(-1), ignore_index=256)
    loss.backward()
    opt.step()
    
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:03d} | Loss: {loss.item():.4f}")

agent.eval()
correct = 0
for p, exp, _ in pointer_data[:20]:
    p_b = [ord(c) for c in p]
    curr_seq = torch.tensor([p_b], dtype=torch.long, device=device)
    gen = []
    with torch.no_grad():
        for _ in range(len(exp) + 2):
            logits = agent(curr_seq, thinking_steps=3)
            nxt = torch.argmax(logits[0, -1, :]).item()
            if nxt in (ord('\n'), 256, 257):
                break
            gen.append(nxt)
            curr_seq = torch.cat([curr_seq, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    got = bytes(gen).decode('utf-8', errors='ignore')
    status = "✅" if got.strip() == exp.strip() else "❌"
    if got.strip() == exp.strip():
        correct += 1
    print(f"  {status} Prompt: {p!r:30s} | Exp: {exp!r:5s} | Got: {got!r:5s}")

print(f"\nAccuracy: {correct}/20 ({correct*5.0}%)")
