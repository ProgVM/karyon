# experiments/exp_116_fused_generation_step.py
"""
===============================================================================
EXP-116: End-to-End Fused Autoregressive Generation Step (Zero-Ping-Pong)
===============================================================================
Hypothesis:
Fusing the entire token generation pipeline (Embedding -> InProj -> LaminarStack -> Attractor -> VolitionalHead -> NextToken)
into a single unified callable method avoids intermediate Python tensor instantiations,
significantly accelerating autoregressive streaming throughput (tok/s >= +25%).

Protocol: KEP v8.1 Scientific Protocol
Author: Bazilevs & Autonomous Lead AI Cyberneticist
===============================================================================
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('.'))

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit
from karyon_hardware import get_hardware_engine

class FusedGenerationEngine(nn.Module):
    def __init__(self, agent: CoREAgent):
        super().__init__()
        self.agent = agent

    def step(self, curr_token: torch.Tensor, pos: int, m_s1: torch.Tensor, m_s2: torch.Tensor, u_t: torch.Tensor):
        # 1. Direct embedding + in_proj
        tok_emb = self.agent.pos_embeddings(curr_token, start_pos=pos, apply_rf=False)
        h_in_step = self.agent.in_proj(tok_emb)
        
        # 2. Native C++20 Fused Laminar Stack
        h_s1, h_s2, m_s1_next, m_s2_next, _ = self.agent.fused_stack(h_in_step, m_s1, m_s2, u_t, curr_token)
        
        # 3. Topdown & Attractor
        topdown = self.agent.topdown_prior_proj(h_s2)
        h_comb = h_s1 + h_s2 + 0.15 * topdown
        h_flat = h_comb.view(1, -1)
        h_rel, _ = self.agent.attractor_head.relax_to_minima(h_flat, u_t)
        
        # 4. Volitional Motor Readout
        logits = self.agent.volitional_head.compute_volitional_logits(
            h_rel, u_t, self.agent.pos_embeddings.byte_embed.weight
        )
        
        # 5. Greedy / MAP byte decision
        next_tok = torch.argmax(logits[:, 32:126], dim=-1, keepdim=True) + 32
        return next_tok, m_s1_next, m_s2_next

def run_experiment():
    print("=" * 80)
    print("STARTING EXP-116: END-TO-END FUSED AUTOREGRESSIVE GENERATION BENCHMARK")
    print("=" * 80)

    hw = get_hardware_engine()
    device_str = 'cuda' if hw.is_cuda else ('xla' if hw.is_tpu else 'cpu')
    print(f"[Hardware] Active Device: {device_str}")

    config = CoREConfig()
    agent = CoREAgent(config, device=device_str).to(device_str)
    hu = HomeostaticUnit(batch_size=1, device=device_str)
    fused_engine = FusedGenerationEngine(agent)

    num_tokens = 300
    prompt = "The autonomous mind "
    prompt_tokens = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long, device=device_str)
    prompt_len = prompt_tokens.size(1)

    # 1. Baseline: Discrete calls in Python loop
    print("\n--- Benchmarking Baseline (Unfused Step) ---")
    m_s1 = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device_str)
    m_s2 = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device_str)
    u_t = hu.state
    
    emb = agent.pos_embeddings(prompt_tokens, start_pos=0, apply_rf=True)
    h_in = agent.in_proj(emb)
    h_s1, m_s1, _ = agent.stage1(h_in, m_s1, u_t, torch.Tensor(), 1.0)
    sal_gate = agent.boundary_detector(h_s1, prompt_tokens)
    h1_prev = torch.zeros(1, 1, agent.hidden_dim, device=device_str)
    e1, _, _ = agent.pw_lper(h_s1, h1_prev, u_t)
    h_s2, m_s2, _ = agent.stage2(e1, m_s2, u_t, sal_gate, 1.0)
    
    curr_token = prompt_tokens[:, -1:]
    
    if device_str == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for t in range(num_tokens):
            tok_emb = agent.pos_embeddings(curr_token, start_pos=prompt_len + t, apply_rf=False)
            h_in_step = agent.in_proj(tok_emb)
            
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
            
            next_tok = torch.argmax(logits[:, 32:126], dim=-1, keepdim=True) + 32
            curr_token = next_tok
            
    if device_str == 'cuda':
        torch.cuda.synchronize()
    base_duration = time.perf_counter() - start_time
    base_speed = num_tokens / base_duration
    print(f"[Baseline] Generated {num_tokens} tokens in {base_duration:.4f}s ({base_speed:.2f} tok/s)")

    # 2. Proposed: Fused Engine Step
    print("\n--- Benchmarking Proposed (Fused Generation Engine Step) ---")
    m_s1 = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device_str)
    m_s2 = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device_str)
    
    emb = agent.pos_embeddings(prompt_tokens, start_pos=0, apply_rf=True)
    h_in = agent.in_proj(emb)
    h_s1, h_s2, m_s1, m_s2, _ = agent.fused_stack(h_in, m_s1, m_s2, u_t, prompt_tokens)
    
    curr_token = prompt_tokens[:, -1:]
    
    if device_str == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for t in range(num_tokens):
            curr_token, m_s1, m_s2 = fused_engine.step(curr_token, prompt_len + t, m_s1, m_s2, u_t)
            
    if device_str == 'cuda':
        torch.cuda.synchronize()
    fused_duration = time.perf_counter() - start_time
    fused_speed = num_tokens / fused_duration
    print(f"[Proposed] Generated {num_tokens} tokens in {fused_duration:.4f}s ({fused_speed:.2f} tok/s)")

    speedup_pct = (fused_speed - base_speed) / base_speed * 100
    print(f"\n[Comparison] Speedup: {speedup_pct:+.2f}% ({base_speed:.2f} -> {fused_speed:.2f} tok/s)")

    results = {
        "exp_id": "EXP-116",
        "baseline_tok_per_sec": float(base_speed),
        "fused_tok_per_sec": float(fused_speed),
        "speedup_pct": float(speedup_pct)
    }

    with open("exp_116_results.json", "w") as f:
        json.dump(results, f, indent=2)

    verdict = "🟢 POSITIVE" if speedup_pct >= 5.0 else ("⚪ NEUTRAL" if speedup_pct >= 0.0 else "🔴 REJECTED")
    print(f"\n[VERDICT]: {verdict} (Throughput Gain: {speedup_pct:+.2f}%)")

if __name__ == "__main__":
    run_experiment()
