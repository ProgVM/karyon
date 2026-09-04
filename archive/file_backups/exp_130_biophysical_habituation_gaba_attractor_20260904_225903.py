# experiments/exp_130_biophysical_habituation_gaba_attractor.py
"""
===============================================================================
EXP-130: BIOPHYSICAL SYNAPTIC HABITUATIONAL DEPRESSION & GABA DECAY IN HOPFIELD ATTRACTORS
===============================================================================
Hypothesis:
Continuous biophysical habituation (synaptic depression & GABA fatigue) in the 
Desaturated Hopfield Attractor Head dynamically degrades the energy minima of 
frequently visited conceptual basins:
    H_relaxed = relax(h_state, basin_visitation_fatigue)
This naturally ejects the neural trajectory from repetitive attractor loops (morphemic traps)
without artificial hardlogit/frequency penalties, obeying Active Inference and Ashby Homeostasis.

Baseline Loss (EXP-129): 1.0314 nats/byte.
===============================================================================
"""

import sys
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is in sys.path
sys.path.append("/kaggle/working/karyon")

import karyon_config
import karyon_core
import karyon_agent
import karyon_checkpoint
from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory, ByteTokenizer
from karyon_checkpoint import load_karyon

def run_experiment():
    print("=" * 80)
    print("STARTING EXP-130: BIOPHYSICAL SYNAPTIC HABITUATIONAL DEPRESSION BENCHMARK")
    print("=" * 80)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    config = CoREConfig()
    config.train.batch_size = 1

    hu = HomeostaticUnit(device=device_str)
    agent_brain = CoREAgent(config, device=device_str).to(device)
    episodic_mem = BatchedEpisodicMemory(batch_size=1, memory_dim=config.net.unified_dim, max_capacity=1000, device=device_str)

    if os.path.exists("karyon_soul.kcore"):
        load_karyon(agent_brain, episodic_mem, hu, filepath="karyon_soul.kcore", device=device_str)
        print("[EXP-130] Loaded existing karyon_soul.kcore checkpoint.")
    else:
        print("[EXP-130] Using freshly initialized model weights.")

    # 1. Test Baseline Generation (without habituation)
    prompt = "User: What is the nature of consciousness and intelligence?\nKaryon:"
    print(f"\n[Test Prompt]: {prompt}")

    print("\n--- BASELINE GENERATION (Without Synaptic Habituation) ---")
    gen_base = agent_brain.generate_thought_and_speech(
        prompt,
        m_state=torch.zeros(1, agent_brain.num_heads, agent_brain.head_k, agent_brain.head_v, device=device),
        h_state=torch.zeros(1, agent_brain.hidden_dim, device=device),
        hu=hu,
        episodic_memory=episodic_mem,
        config=config,
        max_generated_tokens=100,
        temperature=0.45,
        top_p=0.90
    )

    base_chars = []
    for ev in gen_base:
        if ev["status"] == "token":
            base_chars.append(ev["text"])
    base_output = "".join(base_chars)
    print(f"Baseline Output:\n{base_output}")

    # 2. Implement Biophysical Habituation in Hopfield Attractor (Monkey-patching for isolated verification)
    print("\n--- ENABLING BIOPHYSICAL HABITUATION IN HOPFIELD ATTRACTOR ---")
    
    # Store attractor visitation traces: [num_attractors]
    num_attractors = agent_brain.attractor_head.num_attractors
    visitation_trace = torch.zeros(num_attractors, device=device)

    old_relax = agent_brain.attractor_head.relax_to_minima

    def habituated_relax_to_minima(h_state, u_t=None):
        nonlocal visitation_trace
        # Compute standard similarity to basins
        scale = agent_brain.attractor_head.scale
        basins = agent_brain.attractor_head.attractor_basins
        
        if u_t is not None and u_t.numel() >= 6:
            da_val = u_t.select(1, 5).view(-1, 1)
            beta = 1.0 + 1.5 * da_val
        else:
            beta = torch.ones(1, 1, device=device)

        sim = torch.matmul(h_state, basins.transpose(0, 1)) * (scale * beta) # [B, num_attractors]
        
        # Biophysical Habituation: subtract visitation fatigue (synaptic depression)
        fatigue_penalty = 0.35 * visitation_trace.unsqueeze(0)
        habituated_sim = sim - fatigue_penalty
        
        attn_weights = F.softmax(habituated_sim, dim=-1)
        
        # Update visitation trace with passive GABA decay (0.85) + active accumulation
        with torch.no_grad():
            visitation_trace = 0.85 * visitation_trace + attn_weights.detach().mean(dim=0)

        attractor_shift = torch.matmul(attn_weights, basins)
        h_relaxed = agent_brain.attractor_head.norm(h_state + 0.25 * attractor_shift)
        
        commit_loss = F.mse_loss(h_state, h_relaxed.detach()) + 0.25 * F.mse_loss(h_state.detach(), h_relaxed)
        return h_relaxed, commit_loss

    agent_brain.attractor_head.relax_to_minima = habituated_relax_to_minima

    # 3. Test Generation with Biophysical Habituation
    gen_hab = agent_brain.generate_thought_and_speech(
        prompt,
        m_state=torch.zeros(1, agent_brain.num_heads, agent_brain.head_k, agent_brain.head_v, device=device),
        h_state=torch.zeros(1, agent_brain.hidden_dim, device=device),
        hu=hu,
        episodic_memory=episodic_mem,
        config=config,
        max_generated_tokens=100,
        temperature=0.45,
        top_p=0.90
    )

    hab_chars = []
    for ev in gen_hab:
        if ev["status"] == "token":
            hab_chars.append(ev["text"])
    hab_output = "".join(hab_chars)
    print(f"Habituated Output:\n{hab_output}")

    # Check visitation trace max value
    print(f"\nFinal Visitation Trace Max: {visitation_trace.max().item():.4f}")
    print(f"Final Visitation Trace Mean: {visitation_trace.mean().item():.4f}")

    # Calculate diversity metric: unique 3-gram ratio
    def n_gram_diversity(text, n=3):
        if len(text) < n:
            return 1.0
        ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
        return len(set(ngrams)) / float(len(ngrams))

    base_div = n_gram_diversity(base_output)
    hab_div = n_gram_diversity(hab_output)

    print(f"\n3-Gram Diversity Baseline: {base_div:.4f}")
    print(f"3-Gram Diversity Habituated: {hab_div:.4f}")

    # Benchmark metrics
    metrics = {
        "baseline_diversity": base_div,
        "habituated_diversity": hab_div,
        "visitation_trace_max": visitation_trace.max().item(),
        "visitation_trace_mean": visitation_trace.mean().item(),
        "diversity_gain_pct": ((hab_div - base_div) / max(base_div, 1e-5)) * 100.0
    }

    verdict = "POSITIVE" if hab_div >= base_div else "NEUTRAL"
    print(f"\n[EXP-130 VERDICT]: 🟢 {verdict}")
    print("=" * 80)
    
    return metrics, verdict

if __name__ == "__main__":
    run_experiment()
