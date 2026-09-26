"""
EXP-298: Direct Execution of Production C++20 CoREAgent on Multi-Domain Algorithmic Suite.
Testing the full native pipeline:
- C++20 CausalParallelSSD (Temporal State Space)
- C++20 ContinuousSaccadicDrift (Continuous Amari Neural Field Saccades)
- C++20 DynamicMorphicGraph (Latent Deliberation Graph)
- Epigenetic Methylation Optimizer & Sleep Consolidation
"""

import sys, os
sys.path.insert(0, '.')
import math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

import karyon_agent
import karyon_core
from multi_domain_benchmark import generate_multi_domain_suite
from experiments.exp_297_epigenetic_single_pass_benchmark import EpigeneticMethylationOptimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_production_cpp_benchmark():
    print("=" * 80)
    print("⚡ EXP-298: BENCHMARKING PRODUCTION C++20 COREAGENT ON 5-DOMAIN SUITE")
    print("=" * 80)
    
    suite = generate_multi_domain_suite(seed=42)
    dim = 128
    
    agent = karyon_agent.CoREAgent(vocab_size=258, embed_dim=dim, device=str(device))
    agent.add_node("stack_integrator", "LinearAccumulator", True, 1.0)
    agent.add_node("carry_latch", "SaturatedAttractor", True, 1.0)
    agent.add_node("conjunctive_gate", "BilinearMultiplicative", True, 1.0)
    agent.add_node("attractor_snap", "ContinuousHopfield", True, 1.0)
    
    params = list(agent.parameters())
    for p in agent.ssd.parameters():
        params.append(p)
    for p in agent.graph.parameters():
        params.append(p)
    for p in agent.saccadic_drift.parameters():
        params.append(p)
        
    print(f"Total Native C++20 + PyTorch Parameters: {sum(p.numel() for p in params):,}")
    
    optimizer = EpigeneticMethylationOptimizer(params, base_lr=1e-3, lambda_methyl=20.0, weight_decay=1e-4)
    
    # We will test in both modes:
    # 1. First, benchmark Reversal domain directly to verify native C++20 C-SSD Amari drift mechanics!
    print("\n--- Phase 1: Micro-Training on Reversal Domain using C++20 C-SSD Engine ---")
    rev_samples = suite['reversal'][:200]
    
    for epoch in range(40):
        agent.train()
        total_loss = 0.0
        random.shuffle(rev_samples)
        
        for i in range(0, len(rev_samples), 16):
            batch = rev_samples[i:i+16]
            prompts = [list(item[0].encode('utf-8')) for item in batch]
            targets = [list(item[2][len(item[0]):].encode('utf-8')) for item in batch]
            
            max_p = max(len(p) for p in prompts)
            max_t = max(len(t) for t in targets)
            
            p_pad = torch.full((len(batch), max_p), 256, dtype=torch.long, device=device)
            t_pad = torch.full((len(batch), max_t), 256, dtype=torch.long, device=device)
            
            for b_idx, (p, t) in enumerate(zip(prompts, targets)):
                p_pad[b_idx, -len(p):] = torch.tensor(p, dtype=torch.long, device=device)
                t_pad[b_idx, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
                
            # Settle prompt field using CausalParallelSSD
            p_emb = agent.emb(p_pad)
            p_field = agent.ssd.forward(p_emb)
            
            B, P_len, _ = p_field.shape
            # Initialize bump on the last SIGNIFICANT non-delimiter token before '='
            bump = torch.zeros(B, P_len, device=device)
            for b_i in range(B):
                # Find last index where token is not '=' (61) and not space (32)
                row_tokens = p_pad[b_i].tolist()
                init_idx = P_len - 1
                for idx in range(P_len - 1, -1, -1):
                    tok = row_tokens[idx]
                    if tok not in (61, 32, 0): # '=' is 61, ' ' is 32, pad is 0
                        init_idx = idx
                        break
                bump[b_i, init_idx] = 1.0
            bump = F.softmax(bump * 10.0, dim=-1)
            
            agent.graph.reset_state()
            h_core = p_field[:, -1, :]
            inputs = torch.cat([p_pad[:, -1:], t_pad[:, :-1]], dim=1)
            
            all_logits = []
            for t in range(max_t):
                x_t = agent.emb(inputs[:, t])
                h_fused, h_core, bump, p_copy, copy_logits = agent.forward_autoregressive_step(
                    x_t, h_core, p_field, p_pad, bump, thinking_steps=3
                )
                gen_logits = agent.head(h_fused)
                fused_probs = (1.0 - p_copy) * F.softmax(gen_logits, dim=-1) + p_copy * copy_logits
                all_logits.append(torch.log(fused_probs + 1e-8))
                
            log_probs = torch.stack(all_logits, dim=1)
            loss = F.nll_loss(log_probs.reshape(-1, 258), t_pad.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.update_methylation(loss.item(), threshold=1.0)
            optimizer.step(lr_mult=1.0)
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d} | C++20 Native Reversal Loss: {total_loss / (len(rev_samples)//16):.4f} nats")
            
    # Evaluate Reversal
    agent.eval()
    test_samples = suite['reversal'][200:300]
    correct = 0
    with torch.no_grad():
        for expr, expected_ans, full in test_samples:
            p_bytes = list(expr.encode('utf-8'))
            p_t = torch.tensor([p_bytes], dtype=torch.long, device=device)
            p_emb = agent.emb(p_t)
            p_field = agent.ssd.forward(p_emb)
            
            bump = torch.zeros(1, len(p_bytes), device=device)
            init_idx = len(p_bytes) - 1
            for idx in range(len(p_bytes) - 1, -1, -1):
                if p_bytes[idx] not in (61, 32, 0):
                    init_idx = idx
                    break
            bump[:, init_idx] = 1.0
            bump = F.softmax(bump * 10.0, dim=-1)
            
            agent.graph.reset_state()
            h_core = p_field[:, -1, :]
            curr_token = p_t[:, -1]
            
            gen_bytes = []
            for _ in range(len(expected_ans) + 4):
                x_t = agent.emb(curr_token)
                h_fused, h_core, bump, p_copy, copy_logits = agent.forward_autoregressive_step(
                    x_t, h_core, p_field, p_t, bump, thinking_steps=3
                )
                gen_logits = agent.head(h_fused)
                fused_probs = (1.0 - p_copy) * F.softmax(gen_logits, dim=-1) + p_copy * copy_logits
                next_token = fused_probs.argmax(dim=-1)
                gen_bytes.append(next_token.item())
                curr_token = next_token
                if next_token.item() == 10:
                    break
            gen_str = bytes(gen_bytes).decode('utf-8', errors='ignore').strip()
            if expected_ans.strip() in gen_str:
                correct += 1
            if _s_idx < 5:
                print(f"Sample {_s_idx+1} | Expr: {expr.strip()} | Expected: {expected_ans.strip()} | Generated: {repr(gen_str)}")
                
    print(f"\n🎯 Native C++20 C-SSD Reversal Accuracy: {correct / len(test_samples) * 100:.2f}% ({correct}/{len(test_samples)})\n")
    print("=" * 80)

if __name__ == '__main__':
    run_production_cpp_benchmark()
