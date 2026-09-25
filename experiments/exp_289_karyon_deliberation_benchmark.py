"""
===============================================================================
EXP-289: Turing-Complete Karyon-CoRE Recurrent Deliberation Benchmark
Evaluating Turing-Complete primitives inside DynamicMorphicGraph:
1. IndexShiftOp (Pointer register simulations)
2. ConditionalBranchOp (Dynamic runtime routing)
3. CausalParallelSSD (Temporal memory context)

Evaluating Out-of-Distribution (OOD) sequence reversal (Length 7..9 vs training 3..5)
Comparing Shallow (K=1) vs Deep Deliberative (K=6) internal thinking steps.
===============================================================================
"""

import os
import sys
import json
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import karyon_core

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("EXP-289: TURING-COMPLETE KARYON RECURRENT DELIBERATION (ZERO TRANSFORMER)")
print(f"Hardware Compute Device: {device} | Backend: C++20 LibTorch DynamicMorphicGraph")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Dataset: Algorithmic Logic & Out-of-Distribution Evaluation
# ---------------------------------------------------------------------------

def make_reverse_sample(length, vocab_max=8):
    digits = [str(random.randint(1, vocab_max)) for _ in range(length)]
    inp = " ".join(digits) + " | "
    out = " ".join(reversed(digits))
    return inp, out

random.seed(42)
# In-distribution: short sequences (3 to 5 elements)
train_samples = [make_reverse_sample(random.randint(3, 5)) for _ in range(1200)]

# OOD 1: Unseen length (7 to 9 elements - 150% longer than any training instance!)
ood_length = [make_reverse_sample(random.randint(7, 9)) for _ in range(50)]

# OOD 2: Unseen alphabet (letters instead of digits)
def make_symbol_sample(length):
    letters = [chr(ord('a') + random.randint(0, 7)) for _ in range(length)]
    inp = " ".join(letters) + " | "
    out = " ".join(reversed(letters))
    return inp, out

ood_symbols = [make_symbol_sample(random.randint(3, 5)) for _ in range(50)]

# ---------------------------------------------------------------------------
# 2. Pure Karyon-CoRE Deliberative Architecture with Turing-Complete Ops
# ---------------------------------------------------------------------------

class KaryonTuringAgent(nn.Module):
    def __init__(self, vocab_size=258, dim=128):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=1.0 / math.sqrt(dim))
        
        # 1. Zero-Loop Parallel C++20 Causal State-Space Duality (Time-Mixing)
        self.ssd = karyon_core.CausalParallelSSD(dim, str(device))
        
        # 2. Dynamic Morphic Recirculation Graph with Turing-Complete Primitives
        self.graph = karyon_core.DynamicMorphicGraph(dim, str(device))
        
        # Core & Turing-complete computational primitives:
        self.graph.add_node("sensory_core", "LinearAccumulator", True, 10.0)
        self.graph.add_node("ptr_shift", "IndexShift", False, 1.0)
        self.graph.add_node("cond_branch", "ConditionalBranch", False, 1.0)
        self.graph.add_node("bilinear_synergy", "BilinearMultiplicative", False, 1.0)
        self.graph.add_node("hopfield_attractor", "ContinuousHopfield", False, 1.0)
        self.graph.add_node("saturated_memory", "SaturatedAttractor", False, 1.0)
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight # Synaptic weight-tying

    def forward(self, input_ids, thinking_steps=4):
        B, S = input_ids.shape
        x = self.emb(input_ids)
        
        # Step A: Temporal Causal Integration via C++20 SSD (Linear causal memory)
        h_temporal = self.ssd.forward(x)
        
        # Step B: Latent Deliberation (Thinking BEFORE speaking)
        # Recirculate internal representations across graph nodes for K cycles
        flat_temporal = h_temporal.reshape(B * S, self.dim)
        h_thought = self.graph.forward(flat_temporal, thinking_steps).reshape(B, S, self.dim)
        
        # Step C: Motor Efference Readout
        return self.head(self.norm(h_thought))

# ---------------------------------------------------------------------------
# 3. Training & Evaluation Routines
# ---------------------------------------------------------------------------

def encode_batch(samples):
    enc = []
    for p, a in samples:
        full = f"{p}{a}\n"
        enc.append([ord(c) for c in full])
    max_len = max(len(s) for s in enc)
    t = torch.full((len(enc), max_len), 256, dtype=torch.long, device=device)
    for i, s in enumerate(enc):
        t[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return t

train_tensor = encode_batch(train_samples)

def evaluate_suite(model, dataset, name, thinking_steps=4):
    model.eval()
    correct = 0
    total = len(dataset)
    samples = []
    
    with torch.no_grad():
        for p, exp in dataset:
            p_bytes = [ord(c) for c in p]
            gen = list(p_bytes)
            # Autoregressive decoding
            for _ in range(len(exp) + 2):
                inp = torch.tensor([gen], dtype=torch.long, device=device)
                logits = model(inp, thinking_steps=thinking_steps)
                next_b = torch.argmax(logits[0, -1, :]).item()
                if next_b in (ord('\n'), 256, 257):
                    break
                gen.append(next_b)
            got = bytes(gen[len(p_bytes):]).decode("utf-8", errors="ignore").strip()
            is_ok = (got == exp)
            if is_ok:
                correct += 1
            if len(samples) < 4:
                samples.append((p, exp, got, is_ok))
                
    acc = (correct / total) * 100.0
    print(f"\n[{name} | Thinking Steps K={thinking_steps}] Accuracy: {correct}/{total} = {acc:.2f}%")
    for p, exp, got, ok in samples:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status} | Prompt: '{p}' | Expected: '{exp}' | Model: '{got}'")
    return acc

# ---------------------------------------------------------------------------
# 4. Execution Pipeline
# ---------------------------------------------------------------------------

def run():
    model = KaryonTuringAgent(dim=128).to(device)
    # Collect parameters including C++ registered parameters
    named_params = dict(model.named_parameters())
    params = list(named_params.values())
    print(f"Model instantiated with {len(params)} parameter tensors ({sum(p.numel() for p in params):,} total parameters)")
    
    optimizer = torch.optim.AdamW(params, lr=0.003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-4)
    
    print("\n[PHASE 1] Training with Recurrent Latent Deliberation (K=4 thinking steps)...")
    t0 = time.time()
    
    for epoch in range(1, 201):
        model.train()
        idx = torch.randperm(len(train_tensor))[:128]
        batch = train_tensor[idx]
        optimizer.zero_grad()
        
        logits = model(batch, thinking_steps=4)[:, :-1, :]
        targets = batch[:, 1:]
        
        loss = F.cross_entropy(
            logits.reshape(-1, 258),
            targets.reshape(-1),
            ignore_index=256
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 40 == 0 or epoch == 1 or loss.item() < 0.1:
            print(f"  Epoch {epoch:03d} | Loss: {loss.item():.6f} nats | LR: {scheduler.get_last_lr()[0]:.6f}")
            if loss.item() < 0.05:
                print(f"  🎯 Convergence reached at epoch {epoch}!")
                break
                
    train_duration = time.time() - t0
    print(f"\nTraining completed in {train_duration:.2f}s.")
    
    # -----------------------------------------------------------------------
    # PHASE 2: In-Distribution Evaluation
    # -----------------------------------------------------------------------
    id_acc = evaluate_suite(model, train_samples[:40], "IN-DISTRIBUTION", thinking_steps=4)
    
    # -----------------------------------------------------------------------
    # PHASE 3: Out-of-Distribution Evaluation (Length & Symbols)
    # -----------------------------------------------------------------------
    ood_len_acc = evaluate_suite(model, ood_length, "OOD 1 (UNSEEN LENGTH)", thinking_steps=4)
    ood_sym_acc = evaluate_suite(model, ood_symbols, "OOD 2 (UNSEEN SYMBOLS)", thinking_steps=4)
    
    # -----------------------------------------------------------------------
    # PHASE 4: Thinking Ablation: No Thinking (K=1) vs Deep Thinking (K=6)
    # -----------------------------------------------------------------------
    print("\n[PHASE 4] Thinking Ablation: Evaluating impact of Deliberation Time on Logic...")
    acc_k1 = evaluate_suite(model, ood_length[:30], "SHALLOW THINKING (K=1)", thinking_steps=1)
    acc_k6 = evaluate_suite(model, ood_length[:30], "DEEP DELIBERATION (K=6)", thinking_steps=6)
    
    print("\n" + "=" * 80)
    print("EXP-289 BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Architecture                    : 100% C++20 Causal SSD + Turing-Complete Morphic Graph")
    print(f"Transformers Used               : ZERO (0)")
    print(f"In-Distribution Accuracy        : {id_acc:.2f}%")
    print(f"OOD Length Generalization       : {ood_len_acc:.2f}%")
    print(f"OOD Symbol Generalization       : {ood_sym_acc:.2f}%")
    print(f"Shallow Thinking (K=1)          : {acc_k1:.2f}%")
    print(f"Deep Deliberation (K=6)         : {acc_k6:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    run()
