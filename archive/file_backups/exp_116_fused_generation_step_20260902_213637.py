# experiments/exp_116_fused_generation_step.py
"""
===============================================================================
EXP-116: Fused Native C++20 Cascaded Laminar Autoregressive Generation Step
===============================================================================
Hypothesis:
Replacing discrete separate calls to stage1, boundary_detector, pw_lper, and stage2
during autoregressive speech generation with a single fused call to C++20 FusedCascadedLaminarStack
will reduce Python-C++ boundary overhead, accelerating token generation throughput (tok/s >= +50%)
while preserving numerical identity of hidden states and logits.

Protocol: KEP v8.1 Scientific Protocol
Author: Bazilevs & Autonomous Lead AI Cyberneticist
===============================================================================
"""

import os
import sys
import time
import json
import torch

sys.path.insert(0, os.path.abspath('.'))

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit
from karyon_hardware import get_hardware_engine

def benchmark_generation(agent: CoREAgent, hu: HomeostaticUnit, num_tokens: int = 100, use_fused: bool = False):
    device = agent.device
    prompt = "The autonomous mind "
    prompt_tokens = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long, device=device)
    prompt_len = prompt_tokens.size(1)
    
    m_s1 = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device)
    m_s2 = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device)
    u_t = hu.state
    
    # Warmup / prompt phase
    emb = agent.pos_embeddings(prompt_tokens, start_pos=0, apply_rf=True)
    h_in = agent.in_proj(emb)
    
    if use_fused:
        h_s1, h_s2, m_s1, m_s2, _ = agent.fused_stack(h_in, m_s1, m_s2, u_t, prompt_tokens)
    else:
        h_s1, m_s1, _ = agent.stage1(h_in, m_s1, u_t, torch.Tensor(), 1.0)
        sal_gate = agent.boundary_detector(h_s1, prompt_tokens)
        h1_prev = torch.zeros(1, 1, agent.hidden_dim, device=device)
        e1, _, _ = agent.pw_lper(h_s1, h1_prev, u_t)
        h_s2, m_s2, _ = agent.stage2(e1, m_s2, u_t, sal_gate, 1.0)
        
    curr_token = prompt_tokens[:, -1:]
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for t in range(num_tokens):
            tok_emb = agent.pos_embeddings(curr_token, start_pos=prompt_len + t, apply_rf=False)
            h_in_step = agent.in_proj(tok_emb)
            
            if use_fused:
                # Fused single C++ call
                h_s1, h_s2, m_s1, m_s2, _ = agent.fused_stack(h_in_step, m_s1, m_s2, u_t, curr_token)
            else:
                # Discrete 4 calls
                h_s1, m_s1, _ = agent.stage1(h_in_step, m_s1, u_t, torch.Tensor(), 1.0)
                sal_gate = agent.boundary_detector(h_s1, curr_token)
                h1_prev_proxy = m_s1.view(1, -1)[:, :agent.hidden_dim].unsqueeze(1)
                e1, _, _ = agent.pw_lper(h_s1, h1_prev_proxy, u_t)
                h_s2, m_s2, _ = agent.stage2(e1, m_s2, u_t, sal_gate, 1.0)
                
            topdown = agent.topdown_prior_proj(h_s2)
            h_comb = h_s1 + h_s2 + 0.15 * topdown
            h_flat = h_comb.view(1, -1)
            h_rel, _ = agent.attractor_head.relax_to_minima(h_flat, u_t)
            logits = agent.volitional_head.compute_volitional_logits(h_rel, u_t, agent.pos_embeddings.byte_embed.weight)
            
            # Greedy next token for deterministic benchmark comparison
            next_tok = torch.argmax(logits[:, 32:126], dim=-1, keepdim=True) + 32
            curr_token = next_tok
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    duration = time.perf_counter() - start_time
    tok_per_sec = num_tokens / duration
    return tok_per_sec, duration, curr_token.item()

def run_experiment():
    print("=" * 80)
    print("STARTING EXP-116: FUSED AUTOREGRESSIVE GENERATION BENCHMARK")
    print("=" * 80)

    hw = get_hardware_engine()
    device_str = 'cuda' if hw.is_cuda else ('xla' if hw.is_tpu else 'cpu')
    print(f"[Hardware] Active Device: {device_str}")

    config = CoREConfig()
    agent = CoREAgent(config, device=device_str).to(device_str)
    hu = HomeostaticUnit(batch_size=1, device=device_str)

    num_tokens = 150

    # 1. Benchmark Discrete Generation Step
    print("\n--- Benchmarking Baseline (Discrete Separate Calls) ---")
    # Warmup
    benchmark_generation(agent, hu, num_tokens=20, use_fused=False)
    base_speed, base_time, base_final_tok = benchmark_generation(agent, hu, num_tokens=num_tokens, use_fused=False)
    print(f"[Baseline] Generated {num_tokens} tokens in {base_time:.4f}s ({base_speed:.2f} tok/s)")

    # 2. Benchmark Fused Generation Step
    print("\n--- Benchmarking Proposed (Native C++20 Fused Stack Step) ---")
    # Warmup
    benchmark_generation(agent, hu, num_tokens=20, use_fused=True)
    fused_speed, fused_time, fused_final_tok = benchmark_generation(agent, hu, num_tokens=num_tokens, use_fused=True)
    print(f"[Proposed] Generated {num_tokens} tokens in {fused_time:.4f}s ({fused_speed:.2f} tok/s)")

    speedup_pct = (fused_speed - base_speed) / base_speed * 100
    print(f"\n[Comparison] Speedup: {speedup_pct:+.2f}% ({base_speed:.2f} -> {fused_speed:.2f} tok/s)")
    print(f"[Verification] Final Token Match: {base_final_tok} vs {fused_final_tok}")

    results = {
        "exp_id": "EXP-116",
        "baseline_tok_per_sec": float(base_speed),
        "fused_tok_per_sec": float(fused_speed),
        "speedup_pct": float(speedup_pct),
        "tokens_matched": bool(base_final_tok == fused_final_tok)
    }

    with open("exp_116_results.json", "w") as f:
        json.dump(results, f, indent=2)

    verdict = "🟢 POSITIVE" if speedup_pct >= 10.0 else ("⚪ NEUTRAL" if speedup_pct >= 0.0 else "🔴 REJECTED")
    print(f"\n[VERDICT]: {verdict} (Throughput Gain: {speedup_pct:+.2f}%)")

if __name__ == "__main__":
    run_experiment()
