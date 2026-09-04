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

class HabituatedHopfieldAttractorHead(nn.Module):
    def __init__(self, base_head, hidden_dim=768, gaba_decay=0.82, fatigue_scale=1.50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = 1.0 / math.sqrt(hidden_dim)
        self.attractor_basins = base_head.attractor_basins
        self.norm = nn.LayerNorm(hidden_dim).to(self.attractor_basins.device)
        self.gaba_decay = gaba_decay
        self.fatigue_scale = fatigue_scale
        
        num_attractors = self.attractor_basins.size(0)
        self.register_buffer("visitation_trace", torch.zeros(num_attractors, device=self.attractor_basins.device))

    def reset_trace(self):
        self.visitation_trace.zero_()

    def relax_to_minima(self, h_state, u_t=None):
        dev = h_state.device
        if u_t is not None and u_t.numel() >= 6:
            da_val = u_t.select(1, 5).view(-1, 1)
            beta = 1.0 + 1.5 * da_val
        else:
            beta = torch.ones(1, 1, device=dev)

        sim = torch.matmul(h_state, self.attractor_basins.transpose(0, 1)) * (self.scale * beta)
        
        # Subtract synaptic fatigue from frequently visited basins (Habituation)
        fatigue_penalty = self.fatigue_scale * self.visitation_trace.unsqueeze(0)
        habituated_sim = sim - fatigue_penalty
        
        attn_weights = F.softmax(habituated_sim, dim=-1)
        
        # Dynamic accumulation of visitation + passive GABA decay
        with torch.no_grad():
            self.visitation_trace.copy_(self.gaba_decay * self.visitation_trace + attn_weights.detach().mean(dim=0))

        attractor_shift = torch.matmul(attn_weights, self.attractor_basins)
        h_relaxed = self.norm(h_state + 0.25 * attractor_shift)
        
        commit_loss = F.mse_loss(h_state, h_relaxed.detach()) + 0.25 * F.mse_loss(h_state.detach(), h_relaxed)
        return h_relaxed, commit_loss

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

    # 2. Swap in the Habituated Hopfield Head
    print("\n--- ENABLING BIOPHYSICAL HABITUATION IN HOPFIELD ATTRACTOR ---")
    orig_head = agent_brain.attractor_head
    habituated_head = HabituatedHopfieldAttractorHead(
        orig_head, 
        hidden_dim=agent_brain.hidden_dim,
        gaba_decay=0.82,
        fatigue_scale=1.50
    ).to(device)
    agent_brain.attractor_head = habituated_head

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

    # Check visitation trace metrics
    trace_max = habituated_head.visitation_trace.max().item()
    trace_mean = habituated_head.visitation_trace.mean().item()
    print(f"\nFinal Visitation Trace Max: {trace_max:.4f}")
    print(f"Final Visitation Trace Mean: {trace_mean:.4f}")

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

    metrics = {
        "baseline_diversity": base_div,
        "habituated_diversity": hab_div,
        "visitation_trace_max": trace_max,
        "visitation_trace_mean": trace_mean,
        "diversity_gain_pct": ((hab_div - base_div) / max(base_div, 1e-5)) * 100.0
    }

    verdict = "🟢 POSITIVE" if hab_div > base_div else "⚪ NEUTRAL"
    print(f"\n[EXP-130 VERDICT]: {verdict}")
    print("=" * 80)
    
    return metrics, verdict

if __name__ == "__main__":
    run_experiment()
