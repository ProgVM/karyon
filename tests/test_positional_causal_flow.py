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
# Testing Positional-Aware Spatiotemporal Cognitive Flow:
# 1. Byte Embedding + Rotary Positional Encoding (RoPE / Explicit Causal Positions)
# 2. Multi-Layer Dynamic Morphic Graph (Spatial/Latent Routing)
# 3. Autoregressive Training on Reversal & Dyck
# ==============================================================================

class RoPEEmbedding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", torch.cos(freqs))
        self.register_buffer("sin", torch.sin(freqs))

    def forward(self, x):
        # x: [B, S, D]
        S = x.shape[1]
        cos = self.cos[:S].unsqueeze(0).to(x.device)
        sin = self.sin[:S].unsqueeze(0).to(x.device)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rot

class PositionalSpatiotemporalEngine(nn.Module):
    def __init__(self, vocab_size=258, dim=128, num_layers=2):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab_size, dim)
        self.rope = RoPEEmbedding(dim)
        
        # Self-Attention + Graph Stack
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'q': nn.Linear(dim, dim, bias=False),
                'k': nn.Linear(dim, dim, bias=False),
                'v': nn.Linear(dim, dim, bias=False),
                'out': nn.Linear(dim, dim, bias=False),
                'norm1': nn.LayerNorm(dim),
                'norm2': nn.LayerNorm(dim),
                'swi_g': nn.Linear(dim, dim * 2, bias=False),
                'swi_v': nn.Linear(dim, dim * 2, bias=False),
                'swi_o': nn.Linear(dim * 2, dim, bias=False),
            }) for _ in range(num_layers)
        ])
        
        # Dynamic Morphic Graph for Thinking Cycles
        self.graph = karyon_core.DynamicMorphicGraph(dim, str(device))
        self.graph.add_node("acc", "LinearAccumulator", True, 1.0)
        self.graph.add_node("sat", "SaturatedAttractor", True, 1.0)
        self.graph.add_node("bilinear", "BilinearMultiplicative", False, 1.0)
        self.graph.add_node("hopfield", "ContinuousHopfield", False, 1.0)
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, input_seq, thinking_steps=3):
        B, S = input_seq.shape
        x = self.emb(input_seq)
        
        causal_mask = torch.tril(torch.ones(S, S, device=input_seq.device)).view(1, 1, S, S)
        
        for layer in self.layers:
            # Self-Attention with RoPE
            q = self.rope(layer['q'](x))
            k = self.rope(layer['k'](x))
            v = layer['v'](x)
            
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim)
            scores = scores.masked_fill(causal_mask.squeeze(1) == 0, -1e9)
            attn = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v)
            
            x = layer['norm1'](x + layer['out'](attn_out))
            
            # SwiGLU FFN
            swi = layer['swi_o'](F.silu(layer['swi_g'](x)) * layer['swi_v'](x))
            x = layer['norm2'](x + swi)
            
        # Recurrent Morphic Deliberation across graph
        h_flat = x.reshape(B * S, self.dim)
        h_delib = self.graph.forward(h_flat, thinking_steps).reshape(B, S, self.dim)
        
        return self.head(self.norm(x + h_delib))

    def parameters(self, recurse=True):
        for p in self.get_complete_state_dict().values():
            if isinstance(p, torch.Tensor) and p.requires_grad:
                yield p

    def get_complete_state_dict(self):
        state = {}
        for k, v in self.graph.named_parameters_map().items():
            state[f"graph.{k}"] = v
        for idx, layer in enumerate(self.layers):
            for k, p in layer.named_parameters():
                state[f"layers.{idx}.{k}"] = p
        state["emb.weight"] = self.emb.weight
        state["norm.weight"] = self.norm.weight
        state["norm.bias"] = self.norm.bias
        return state

agent = PositionalSpatiotemporalEngine(vocab_size=258, dim=128, num_layers=2).to(device)
print(f"Instantiated PositionalSpatiotemporalEngine. Total parameters: {len(list(agent.parameters()))}")

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

print("\n--- Training PositionalSpatiotemporalEngine (100 Epochs) ---")
for ep in range(1, 101):
    agent.train()
    idx = torch.randperm(len(P_train))[:128]
    p_b = P_train[idx]
    a_b = A_train[idx]
    
    full_seq = torch.cat([p_b, a_b], dim=1)
    targets = full_seq[:, 1:].clone()
    targets[:, :p_b.shape[1]-1] = 256
    
    opt.zero_grad()
    logits = agent(full_seq[:, :-1], thinking_steps=3)
    loss = F.cross_entropy(logits.reshape(-1, 258), targets.reshape(-1), ignore_index=256)
    loss.backward()
    opt.step()
    
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:03d} | CrossEntropy: {loss.item():.4f}")

agent.eval()
print("\n" + "=" * 80)
print("=== POSITIONAL SPATIOTEMPORAL MULTI-DOMAIN EVALUATION ===")
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
