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

class SpatiotemporalCoREAgent(nn.Module):
    def __init__(self, vocab_size=258, embed_dim=128, device='cpu'):
        super().__init__()
        self.dim = embed_dim
        self.vocab_size = vocab_size
        self.device = device
        
        # 1. Universal Byte Manifold
        self.emb = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.emb.weight, 0.0, 0.02)
        
        # 2. C++20 Causal SSD for temporal memory mixing across sequence (Spatiotemporal flow)
        self.ssd = karyon_core.CausalParallelSSD(embed_dim, str(device))
        
        # 3. Dynamic Morphic Graph for multi-step recurrent thinking (Spatial/Latent depth)
        self.graph = karyon_core.DynamicMorphicGraph(embed_dim, str(device))
        self.graph.add_node("acc", "LinearAccumulator", True, 1.0)
        self.graph.add_node("sat", "SaturatedAttractor", True, 1.0)
        self.graph.add_node("bilinear", "BilinearMultiplicative", False, 1.0)
        self.graph.add_node("hopfield", "ContinuousHopfield", False, 1.0)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight # Tied embeddings (biophysical resonance)

    def forward(self, input_ids: torch.Tensor, thinking_steps: int = 4) -> torch.Tensor:
        # input_ids: [B, S]
        B, S = input_ids.shape
        x = self.emb(input_ids) # [B, S, D]
        
        # Level 1: Temporal Causal State-Space Mixing (Mamba-2 / SSD)
        h_temporal = self.ssd.forward(x) # [B, S, D]
        
        # Level 2: Spatial Thinking Recurrence through Dynamic Graph
        # Each position now possesses full causal history!
        h_flat = h_temporal.reshape(B * S, self.dim)
        h_delib = self.graph.forward(h_flat, thinking_steps).reshape(B, S, self.dim)
        
        # Residual skip connection from temporal stream + thinking output
        h_out = h_temporal + h_delib
        logits = self.head(self.norm(h_out))
        return logits

    def parameters(self, recurse: bool = True):
        for p in self.get_complete_state_dict().values():
            if isinstance(p, torch.Tensor) and p.requires_grad:
                yield p

    def get_complete_state_dict(self):
        state = {}
        for k, v in self.graph.named_parameters_map().items():
            state[f"graph.{k}"] = v
        for k, v in self.ssd.named_parameters().items():
            state[f"ssd.{k}"] = v
        state["emb.weight"] = self.emb.weight
        state["norm.weight"] = self.norm.weight
        state["norm.bias"] = self.norm.bias
        return state

agent = SpatiotemporalCoREAgent(vocab_size=258, embed_dim=128, device=str(device)).to(device)
print(f"Agent instantiated. Total optimizable parameters: {len(list(agent.parameters()))}")

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

print("\n--- Training Spatiotemporal CoREAgent (80 Epochs) ---")
for ep in range(1, 81):
    agent.train()
    idx = torch.randperm(len(P_train))[:128]
    p_b = P_train[idx]
    a_b = A_train[idx]
    
    full_seq = torch.cat([p_b, a_b], dim=1) # [B, S]
    targets = full_seq[:, 1:].clone()
    targets[:, :p_b.shape[1]-1] = 256 # Mask out prompt
    
    opt.zero_grad()
    logits = agent(full_seq[:, :-1], thinking_steps=3)
    loss = F.cross_entropy(logits.reshape(-1, 258), targets.reshape(-1), ignore_index=256)
    loss.backward()
    opt.step()
    
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:02d} | CrossEntropy: {loss.item():.4f}")

agent.eval()
print("\n" + "=" * 80)
print("=== SPATIOTEMPORAL CoREAgent MULTI-DOMAIN EVALUATION ===")
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
