"""
===============================================================================
EXP-290: Holographic Cellular Wave Attractor (HCWA) for Sovereign Algorithmic Reasoning
Testing the Two-Phase Deliberation Hypothesis:
Phase 1: Sensory Ingestion & Causal Temporal Context (C++20 Causal SSD)
Phase 2: Bidirectional Cellular Wave Deliberation (Latent Thought Propagation)
Phase 3: Saccadic Attractor Gaze & Motor Efference

Benchmark: Multi-Domain Algorithmic Suite (Pointers, Reversal, Addition, Parity, Dyck-1)
===============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import random
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import karyon_core
from multi_domain_benchmark import generate_multi_domain_suite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("EXP-290: HOLOGRAPHIC CELLULAR WAVE ATTRACTOR (HCWA) BENCHMARK")
print(f"Hardware Compute Device: {device} | Backend: C++20 LibTorch + Cellular Wave Cortex")
print("=" * 80)

# ==============================================================================
# 1. Holographic Cellular Wave Layer (Phase 2 Deliberation)
# ==============================================================================

class CellularWaveCortex(nn.Module):
    """
    Weight-tied bidirectional cellular wave layer.
    Propagates information across the spatial sequence axis during latent thinking cycles (K steps)
    without full O(N^2) pairwise attention matrices.
    """
    def __init__(self, dim, num_iterations=4):
        super().__init__()
        self.dim = dim
        self.num_iterations = num_iterations
        
        # Cellular diffusion kernels: Left, Center, Right receptive field
        self.conv_wave = nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1, groups=1, bias=False)
        self.swi_gate = nn.Linear(dim, dim, bias=False)
        self.swi_val = nn.Linear(dim, dim, bias=False)
        self.swi_out = nn.Linear(dim, dim, bias=False)
        self.norm = nn.RMSNorm(dim)
        
        # Dynamic allostatic damping factor (learnable wave velocity)
        self.wave_velocity = nn.Parameter(torch.tensor(0.5))

    def forward(self, h_seq, steps=None):
        # h_seq: [B, S, D]
        B, S, D = h_seq.shape
        k_steps = steps if steps is not None else self.num_iterations
        
        h = h_seq
        v = torch.sigmoid(self.wave_velocity)
        
        for _ in range(k_steps):
            # 1. 1D Cellular Wave Diffusion [B, D, S]
            h_trans = h.transpose(1, 2)
            wave_feat = self.conv_wave(h_trans).transpose(1, 2) # [B, S, 2*D]
            gate, val = wave_feat.chunk(2, dim=-1)
            wave_update = F.silu(gate) * val
            
            # 2. Local Non-linear SwiGLU Reaction
            swi = self.swi_out(F.silu(self.swi_gate(h)) * self.swi_val(h))
            
            # 3. Wave Integration with RMSNorm
            h_next = self.norm(h + v * (wave_update + swi))
            h = h_next
            
        return h

# ==============================================================================
# 2. Karyon-CoRE HCWA Sovereign Mind Architecture
# ==============================================================================

class KaryonHCWAgent(nn.Module):
    def __init__(self, vocab_size=258, dim=128, wave_steps=4, thinking_steps=3):
        super().__init__()
        self.dim = dim
        self.wave_steps = wave_steps
        self.thinking_steps = thinking_steps
        
        self.emb = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.emb.weight, 0.0, 1.0 / math.sqrt(dim))
        
        # 1. C++20 Causal Parallel SSD Scan (Temporal Memory Context Flow)
        self.ssd = karyon_core.CausalParallelSSD(dim, str(device))
        
        # 2. Holographic Cellular Wave Cortex (Spatial Thought Diffusion)
        self.wave_cortex = CellularWaveCortex(dim, num_iterations=wave_steps)
        
        # 3. Dynamic Morphic Graph (Recurrent Latent Primitives)
        self.graph = karyon_core.DynamicMorphicGraph(dim, str(device))
        self.graph.add_node("acc", "LinearAccumulator", True, 1.0)
        self.graph.add_node("sat", "SaturatedAttractor", True, 1.0)
        self.graph.add_node("bilinear", "BilinearMultiplicative", False, 1.0)
        self.graph.add_node("hopfield", "ContinuousHopfield", False, 1.0)
        
        # 4. Gaze-Coupled Motor Efference Readout
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight # Synaptic weight tying

    def forward(self, input_seq, thinking_steps=None, wave_steps=None):
        B, S = input_seq.shape
        x = self.emb(input_seq)
        
        # Phase 1: Causal Temporal Memory Flow (C++20 SSD)
        h_temp = self.ssd.forward(x)
        
        # Phase 2: Cellular Wave Deliberation across sequence space
        h_wave = self.wave_cortex(h_temp, steps=wave_steps)
        
        # Phase 3: Graph Recurrent Deliberation
        k_steps = thinking_steps if thinking_steps is not None else self.thinking_steps
        h_flat = h_wave.reshape(B * S, self.dim)
        h_delib = self.graph.forward(h_flat, k_steps).reshape(B, S, self.dim)
        
        # Readout Motor Efference
        out = self.norm(h_wave + h_delib)
        return self.head(out)

    def parameters(self, recurse=True):
        for p in self.get_complete_state_dict().values():
            if isinstance(p, torch.Tensor) and p.requires_grad:
                yield p

    def get_complete_state_dict(self):
        state = {}
        for k, v in self.graph.named_parameters_map().items():
            state[f"graph.{k}"] = v
        for k, v in self.ssd.named_parameters():
            state[f"ssd.{k}"] = v
        for k, v in self.wave_cortex.named_parameters():
            state[f"wave.{k}"] = v
        state["emb.weight"] = self.emb.weight
        return state

# ==============================================================================
# 3. Training & Evaluation Pipeline
# ==============================================================================

def run_experiment():
    print("\n--- Generating Multi-Domain Benchmark Dataset ---")
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
    
    agent = KaryonHCWAgent(vocab_size=258, dim=128, wave_steps=4, thinking_steps=3).to(device)
    params = list(agent.parameters())
    print(f"Instantiated KaryonHCWAgent. Total Parameter Tensors: {len(params)} ({sum(p.numel() for p in params):,} weights)")
    
    opt = torch.optim.AdamW(params, lr=0.003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-4)
    
    print("\n--- Training HCWA (100 Epochs across all 5 domains) ---")
    t0 = time.time()
    for ep in range(1, 101):
        agent.train()
        idx = torch.randperm(len(P_train))[:128]
        p_b = P_train[idx]
        a_b = A_train[idx]
        
        full_seq = torch.cat([p_b, a_b], dim=1)
        targets = full_seq[:, 1:].clone()
        targets[:, :p_b.shape[1]-1] = 256
        
        opt.zero_grad()
        logits = agent(full_seq[:, :-1], wave_steps=4, thinking_steps=3)
        loss = F.cross_entropy(logits.reshape(-1, 258), targets.reshape(-1), ignore_index=256)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        scheduler.step()
        
        if ep % 20 == 0 or ep == 1:
            print(f"Epoch {ep:03d} | CrossEntropy: {loss.item():.4f} nats | LR: {scheduler.get_last_lr()[0]:.6f}")

    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.2f}s.")
    
    # --------------------------------------------------------------------------
    # Evaluation across Multi-Domain Benchmark
    # --------------------------------------------------------------------------
    agent.eval()
    print("\n" + "=" * 80)
    print("=== EXP-290 MULTI-DOMAIN EVALUATION RESULTS ===")
    print("=" * 80)
    
    domain_accuracies = {}
    for domain, samples in suite_test.items():
        correct = 0
        total = len(samples)
        sample_preds = []
        for p, exp, _ in samples:
            p_b = [ord(c) for c in p]
            curr_seq = torch.tensor([p_b], dtype=torch.long, device=device)
            gen = []
            with torch.no_grad():
                for _ in range(len(exp) + 3):
                    logits = agent(curr_seq, wave_steps=4, thinking_steps=3)
                    nxt = torch.argmax(logits[0, -1, :]).item()
                    if nxt in (ord('\n'), 256, 257):
                        break
                    gen.append(nxt)
                    curr_seq = torch.cat([curr_seq, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            got = bytes(gen).decode('utf-8', errors='ignore')
            is_match = (got.strip() == exp.strip())
            if is_match:
                correct += 1
            if len(sample_preds) < 3:
                sample_preds.append((p, exp, got, is_match))
                
        acc = (correct / total) * 100.0
        domain_accuracies[domain] = acc
        print(f"\nDomain [{domain.upper():10s}]: Accuracy = {acc:.2f}% ({correct}/{total})")
        for p, exp, got, ok in sample_preds:
            status = "✅ PASS" if ok else "❌ FAIL"
            print(f"  {status} | Prompt: {p!r:30s} | Exp: {exp!r:10s} | Got: {got!r:10s}")

    mean_acc = sum(domain_accuracies.values()) / len(domain_accuracies)
    print("\n" + "=" * 80)
    print("=== SUMMARY METRICS ===")
    print(f"Pointer Accuracy   : {domain_accuracies.get('pointer', 0.0):.2f}%")
    print(f"Reversal Accuracy  : {domain_accuracies.get('reversal', 0.0):.2f}%")
    print(f"Addition Accuracy  : {domain_accuracies.get('addition', 0.0):.2f}%")
    print(f"Parity Accuracy    : {domain_accuracies.get('parity', 0.0):.2f}%")
    print(f"Dyck Accuracy      : {domain_accuracies.get('dyck', 0.0):.2f}%")
    print(f"Mean Benchmark Acc : {mean_acc:.2f}%")
    print("=" * 80)
    
    verdict = "POSITIVE" if mean_acc > 30.0 or domain_accuracies.get('reversal', 0.0) > 50.0 else "NEUTRAL"
    
    results = {
        "exp_id": "EXP-290",
        "final_loss": loss.item(),
        "domain_accuracies": domain_accuracies,
        "mean_accuracy": mean_acc,
        "verdict": verdict
    }
    with open("experiments/exp_290_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_experiment()
