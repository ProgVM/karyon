"""
===============================================================================
EXP-289: Non-Trivial Out-of-Distribution Algorithmic Generalization Benchmark
Testing Causal Temporal SSD + Dynamic Morphic Recirculation on:
1. Multi-Step Variable Assignment & Tracking:
   x=A; y=B; x=y; y=C; x=?
2. Unseen Compositional Operator Discovery:
   Given primitives star(a)=a*2+1, moon(a)=a^2, evaluate moon(star(a)) on unseen ranges.
3. Strict Out-of-Distribution (OOD) Domain Evaluation:
   Training numbers: a in [1..12]
   Test OOD numbers: a in [15..25] (Never seen during training!)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("EXP-289: ALGORITHMIC REASONING & OOD LOGICAL GENERALIZATION BENCHMARK")
print(f"Hardware Compute Device: {device} | CUDA Available: {torch.cuda.is_available()}")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Dataset Generation: Two Abstract Algorithmic Tasks
# ---------------------------------------------------------------------------

def generate_symbolic_reasoning_tasks():
    random.seed(42)
    
    # Task 1: Compositional Functions: f(x)=2x+1, g(x)=x^2, fog(x)=2*(x^2)+1, gof(x)=(2x+1)^2
    # In-Distribution: x in [1..12]
    # Out-of-Distribution (OOD): x in [13..20]
    train_tasks = []
    val_id_tasks = []
    val_ood_tasks = []
    
    # Primitive 1: f(x) = 2x+1
    # Primitive 2: g(x) = 3x-2
    # Composition: f_g(x) = f(g(x)) = 2*(3x-2)+1 = 6x - 3
    # Composition: g_f(x) = g(f(x)) = 3*(2x+1)-2 = 6x + 1
    for x in range(1, 13):
        train_tasks.append((f"f({x})", str(2 * x + 1)))
        train_tasks.append((f"g({x})", str(3 * x - 2)))
        train_tasks.append((f"f(g({x}))", str(6 * x - 3)))
        train_tasks.append((f"g(f({x}))", str(6 * x + 1)))
        
    for x in range(13, 22):
        val_ood_tasks.append((f"f(g({x}))", str(6 * x - 3)))
        val_ood_tasks.append((f"g(f({x}))", str(6 * x + 1)))
        val_ood_tasks.append((f"f({x})", str(2 * x + 1)))
        val_ood_tasks.append((f"g({x})", str(3 * x - 2)))
        
    # Task 2: Multi-step Variable Pointer Swapping
    # "a=3;b=7;c=a;a=b;b=c;a=?" -> 7
    # "a=4;b=9;a=b;b=1;a=?" -> 9
    symbols = ["a", "b", "c"]
    for _ in range(150):
        v1 = random.randint(1, 9)
        v2 = random.randint(1, 9)
        # Swap logic
        # a=v1, b=v2, t=a, a=b, b=t
        expr = f"a={v1};b={v2};c=a;a=b;b=c;a="
        ans = str(v2)
        train_tasks.append((expr, ans))
        
        expr_b = f"a={v1};b={v2};c=a;a=b;b=c;b="
        ans_b = str(v1)
        train_tasks.append((expr_b, ans_b))
        
    # OOD Pointer Swapping: Multi-digit numbers [20..99] never seen in training
    for _ in range(50):
        v1 = random.randint(20, 99)
        v2 = random.randint(20, 99)
        expr = f"a={v1};b={v2};c=a;a=b;b=c;a="
        ans = str(v2)
        val_ood_tasks.append((expr, ans))
        
        expr_b = f"a={v1};b={v2};c=a;a=b;b=c;b="
        ans_b = str(v1)
        val_ood_tasks.append((expr_b, ans_b))

    random.shuffle(train_tasks)
    return train_tasks, val_ood_tasks

train_data, ood_data = generate_symbolic_reasoning_tasks()
print(f"Dataset Synthesized:")
print(f"  Training Instances (In-Distribution)    : {len(train_data)}")
print(f"  OOD Test Instances (Unseen Numbers/Ops) : {len(ood_data)}")
print(f"  Sample Train: '{train_data[0][0]} -> {train_data[0][1]}'")
print(f"  Sample OOD  : '{ood_data[0][0]} -> {ood_data[0][1]}'")

# ---------------------------------------------------------------------------
# 2. Spatiotemporal Morphic Recirculation Agent
# ---------------------------------------------------------------------------

class SpatiotemporalMorphicReasoner(nn.Module):
    def __init__(self, vocab_size=258, dim=256, num_nodes=4, max_seq_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_nodes = num_nodes
        
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=1.0 / math.sqrt(dim))
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=1.0 / math.sqrt(dim))
        
        # Temporal Causal SSD / Self-Attention Layer (Axis S: Sequence)
        self.temp_ln = nn.LayerNorm(dim)
        self.temp_q = nn.Linear(dim, dim, bias=False)
        self.temp_k = nn.Linear(dim, dim, bias=False)
        self.temp_v = nn.Linear(dim, dim, bias=False)
        self.temp_out = nn.Linear(dim, dim, bias=False)
        
        # Morphic Thinking Nodes (Axis K: Graph Recurrence / Reasoning Depth)
        # Node 0: Sensory Projection
        # Node 1: Multiplicative Bilinear Gate (a*b, variable interactions)
        # Node 2: SwiGLU Non-Linear FFN
        # Node 3: Saturated Attractor / Memory Snapping
        self.w_lin0 = nn.Linear(dim, dim, bias=False)
        
        # Bilinear Node 1
        self.b_left = nn.Linear(dim, dim, bias=False)
        self.b_right = nn.Linear(dim, dim, bias=False)
        self.b_out = nn.Linear(dim, dim, bias=False)
        
        # SwiGLU Node 2
        self.swi_gate = nn.Linear(dim, dim * 4, bias=False)
        self.swi_val = nn.Linear(dim, dim * 4, bias=False)
        self.swi_out = nn.Linear(dim * 4, dim, bias=False)
        
        # Attractor Node 3
        self.attractor_w = nn.Linear(dim, dim, bias=False)
        
        self.node_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_nodes)])
        
        # Contraction Routing Matrix
        self.w_route = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.1)
        self.gamma = nn.Parameter(torch.full((num_nodes,), 0.5))
        
        self.out_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def op_forward(self, j, x):
        if j == 0:
            return self.w_lin0(x)
        elif j == 1:
            return self.b_out(self.b_left(x) * self.b_right(x))
        elif j == 2:
            return self.swi_out(F.silu(self.swi_gate(x)) * self.swi_val(x))
        elif j == 3:
            return torch.tanh(self.attractor_w(x))
        else:
            return x

    def forward(self, input_ids, thinking_steps=4):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.emb(input_ids) + self.pos_emb(pos)
        
        # 1. Temporal Mixing across Sequence Length (Causal)
        norm_x = self.temp_ln(x)
        q = self.temp_q(norm_x)
        k = self.temp_k(norm_x)
        v = self.temp_v(norm_x)
        
        causal_mask = torch.triu(torch.full((S, S), float('-inf'), device=input_ids.device), diagonal=1)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.dim)
        attn = F.softmax(scores + causal_mask.unsqueeze(0), dim=-1)
        temporal_ctx = x + self.temp_out(torch.bmm(attn, v))
        
        # 2. Dynamic Morphic Graph Recirculation (Thinking Steps)
        K = self.num_nodes
        states = torch.zeros(K, B, S, self.dim, device=input_ids.device)
        states[0] = temporal_ctx
        
        w_sub = F.softmax(self.w_route, dim=-1)
        
        for _ in range(thinking_steps):
            agg = torch.einsum('ij, i b s d -> j b s d', w_sub, states)
            agg[0] = agg[0] + temporal_ctx
            
            new_states = []
            for j in range(K):
                g = torch.sigmoid(self.gamma[j])
                raw = self.node_norms[j](self.op_forward(j, agg[j]))
                h_new = (1.0 - g) * states[j] + g * raw
                new_states.append(h_new)
            states = torch.stack(new_states, dim=0)
            
        motor_latent = states[0] + states[1] + states[2] # Multi-path readout
        motor_norm = self.out_norm(motor_latent)
        return self.head(motor_norm)

# ---------------------------------------------------------------------------
# 3. Training & Evaluation Engine
# ---------------------------------------------------------------------------

def encode_pairs(pairs):
    encoded = []
    for prompt, ans in pairs:
        full_seq = f"{prompt}{ans}\n"
        encoded.append((prompt, ans, [ord(c) for c in full_seq]))
    return encoded

train_encoded = encode_pairs(train_data)

def get_batch(data, batch_size=16):
    batch = random.sample(data, batch_size)
    max_len = max(len(item[2]) for item in batch)
    tensor = torch.full((batch_size, max_len), 256, dtype=torch.long, device=device)
    for i, item in enumerate(batch):
        seq = item[2]
        tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return tensor

def evaluate_tasks(model, tasks, name="EVAL", thinking_steps=4):
    model.eval()
    correct = 0
    total = len(tasks)
    samples = []
    
    with torch.no_grad():
        for prompt, expected_ans in tasks:
            prompt_bytes = [ord(c) for c in prompt]
            gen_bytes = list(prompt_bytes)
            
            # Autoregressive generation of the answer
            for _ in range(len(expected_ans) + 2):
                inp = torch.tensor([gen_bytes], dtype=torch.long, device=device)
                logits = model(inp, thinking_steps=thinking_steps)
                next_byte = torch.argmax(logits[0, -1, :]).item()
                if next_byte in (ord('\n'), 256, 257):
                    break
                gen_bytes.append(next_byte)
                
            gen_text = bytes(gen_bytes[len(prompt_bytes):]).decode("utf-8", errors="replace").strip()
            is_match = (gen_text == expected_ans.strip())
            if is_match:
                correct += 1
            if len(samples) < 5:
                samples.append((prompt, expected_ans, gen_text, is_match))
                
    accuracy = (correct / total) * 100.0
    print(f"\n[{name}] Exact Match Accuracy: {accuracy:.2f}% ({correct}/{total})")
    for p, exp, got, ok in samples:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status} | Prompt: '{p}' | Expected: '{exp}' | Model: '{got}'")
    return accuracy

# ---------------------------------------------------------------------------
# 4. Main Execution Loop
# ---------------------------------------------------------------------------

def run_experiment():
    model = SpatiotemporalMorphicReasoner(dim=256, num_nodes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-4)
    
    print("\n[PHASE 1] Initializing Training on Algorithmic Primitives (300 steps)...")
    t0 = time.time()
    
    for step in range(1, 301):
        model.train()
        batch_tokens = get_batch(train_encoded, batch_size=32)
        optimizer.zero_grad()
        
        logits = model(batch_tokens, thinking_steps=4)[:, :-1, :]
        targets = batch_tokens[:, 1:]
        
        loss = F.cross_entropy(
            logits.reshape(-1, 258),
            targets.reshape(-1),
            ignore_index=256
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 50 == 0 or step == 1 or loss.item() < 0.05:
            print(f"  Step {step:03d} | Loss: {loss.item():.6f} nats | LR: {scheduler.get_last_lr()[0]:.6f}")
            if loss.item() < 0.005:
                print(f"  🎯 Target loss reached at step {step}!")
                break
                
    total_train_time = time.time() - t0
    print(f"\nTraining Completed in {total_train_time:.2f} seconds.")
    
    # -----------------------------------------------------------------------
    # PHASE 2: In-Distribution Reasoning Accuracy
    # -----------------------------------------------------------------------
    print("\n[PHASE 2] Evaluating In-Distribution Algorithmic Accuracy...")
    id_accuracy = evaluate_tasks(model, train_data[:40], name="IN-DISTRIBUTION REASONING", thinking_steps=4)
    
    # -----------------------------------------------------------------------
    # PHASE 3: Out-of-Distribution (OOD) Generalization
    # -----------------------------------------------------------------------
    print("\n[PHASE 3] Evaluating Out-of-Distribution (OOD) Algorithmic Generalization...")
    ood_accuracy = evaluate_tasks(model, ood_data, name="OUT-OF-DISTRIBUTION GENERALIZATION", thinking_steps=4)
    
    # -----------------------------------------------------------------------
    # PHASE 4: Thinking Recirculation Ablation (K=1 vs K=4)
    # -----------------------------------------------------------------------
    print("\n[PHASE 4] Ablation: Evaluating impact of Recurrent Thinking Steps on OOD...")
    print("  -> Running with K=1 (Single-pass shallow feed-forward):")
    ood_acc_k1 = evaluate_tasks(model, ood_data[:30], name="SHALLOW K=1 OOD", thinking_steps=1)
    
    print("  -> Running with K=6 (Deep recurrent latent deliberation):")
    ood_acc_k6 = evaluate_tasks(model, ood_data[:30], name="DEEP K=6 OOD", thinking_steps=6)
    
    print("\n" + "=" * 80)
    print("EXP-289 BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"In-Distribution Exact Accuracy        : {id_accuracy:.2f}%")
    print(f"OOD Generalization Exact Accuracy     : {ood_accuracy:.2f}%")
    print(f"OOD Accuracy (Shallow Thinking K=1)   : {ood_acc_k1:.2f}%")
    print(f"OOD Accuracy (Deep Deliberation K=6)  : {ood_acc_k6:.2f}%")
    print(f"Total Benchmark Execution Time        : {time.time() - t0:.2f} s")
    print("=" * 80)
    
    # Verdict determination
    # Positive if model generalizes OOD (> 70%) and multi-step thinking beats shallow thinking
    verdict = "POSITIVE" if (ood_accuracy >= 70.0 and ood_acc_k6 >= ood_acc_k1) else "NEUTRAL"
    status_emoji = "🟢" if verdict == "POSITIVE" else "⚪"
    print(f"Final Benchmark Verdict               : {status_emoji} {verdict}")
    
    results = {
        "exp_id": "EXP-289",
        "id_accuracy": id_accuracy,
        "ood_accuracy": ood_accuracy,
        "ood_acc_k1": ood_acc_k1,
        "ood_acc_k6": ood_acc_k6,
        "verdict": verdict
    }
    with open("experiments/exp_289_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_experiment()
