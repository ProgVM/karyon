#!/usr/bin/env python3
"""
EXP-318: Tripartite Continuous Valence & Neutral Anchor Memory Benchmark.

Hypothesis:
Replacing binary reward/punishment with continuous valence V in [-1.0, +1.0] guarantees:
1. Attractor pull (V = +1.0) strictly converges the trajectory toward the beacon (Cosine Similarity >= +0.85).
2. Repulsor barrier (V = -1.0) strictly expels the trajectory away from the aversive target (Cosine Similarity <= -0.85).
3. Neutral topographic anchor (V = 0.0) exerts zero force (|Delta a| = 0.00000000) while registering the context key.
"""

import os
import sys
import math
import torch
import torch.nn.functional as F

import karyon_core as kcore
from karyon_agent import CoREAgent

def run_exp_318_benchmark():
    print("=" * 70)
    print("=== EXP-318: TRIPARTITE CONTINUOUS VALENCE & NEUTRAL ANCHOR BENCHMARK ===")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 256
    num_basins = 32
    max_episodes = 256

    print(f"Hardware Acceleration: {device}")

    # Initialize ContinuousHopfieldMemory
    hopfield_mem = kcore.ContinuousHopfieldMemory(dim, num_basins, device, max_episodes)

    # 1. Setup Context and Action Targets
    torch.manual_seed(42)
    ctx_A = F.normalize(torch.randn(1, dim, device=device), dim=-1)
    act_target = F.normalize(torch.randn(1, dim, device=device), dim=-1)

    # Initial probe action aligned with target
    a_init = act_target.clone()

    print("\n--- PHASE 1: POSITIVE EXPERIENCE / ATTRACTOR (Valence V = +1.0) ---")
    hopfield_mem.record_somatic_episode(ctx_A, act_target, 1.0)
    
    # Action perturbed slightly away from target
    perturbation = F.normalize(torch.randn(1, dim, device=device), dim=-1)
    a_probe_att = F.normalize(a_init + 0.5 * perturbation, dim=-1)
    initial_cos_att = F.cosine_similarity(a_probe_att, act_target).item()
    
    # Relax trajectory
    a_relaxed_att = hopfield_mem.relax_with_repulsion(ctx_A, a_probe_att)
    final_cos_att = F.cosine_similarity(a_relaxed_att, act_target).item()
    
    print(f"Initial Cosine Similarity: {initial_cos_att:.6f}")
    print(f"Relaxed Cosine Similarity: {final_cos_att:.6f}")
    attractor_passed = final_cos_att >= 0.85 and (final_cos_att > initial_cos_att)
    print(f"Attractor Beacon Pull Test: {'PASSED (CosSim >= +0.85)' if attractor_passed else 'FAILED'}")

    print("\n--- PHASE 2: NEGATIVE EXPERIENCE / REPULSOR (Valence V = -1.0) ---")
    hopfield_mem_rep = kcore.ContinuousHopfieldMemory(dim, num_basins, device, max_episodes)
    hopfield_mem_rep.record_somatic_episode(ctx_A, act_target, -1.0)
    
    a_probe_rep = a_init.clone()
    a_relaxed_rep = hopfield_mem_rep.relax_with_repulsion(ctx_A, a_probe_rep)
    final_cos_rep = F.cosine_similarity(a_relaxed_rep, act_target).item()
    
    print(f"Relaxed Cosine Similarity against Repulsor: {final_cos_rep:.6f}")
    repulsor_passed = final_cos_rep <= -0.85
    print(f"Repulsor Barrier Expulsion Test: {'PASSED (CosSim <= -0.85)' if repulsor_passed else 'FAILED'}")

    print("\n--- PHASE 3: NEUTRAL EXPERIENCE / TOPOGRAPHIC ANCHOR (Valence V = 0.0) ---")
    hopfield_mem_neu = kcore.ContinuousHopfieldMemory(dim, num_basins, device, max_episodes)
    hopfield_mem_neu.record_somatic_episode(ctx_A, act_target, 0.0)
    
    a_probe_neu = a_init.clone()
    a_relaxed_neu = hopfield_mem_neu.relax_with_repulsion(ctx_A, a_probe_neu)
    
    delta_force_norm = torch.norm(a_relaxed_neu - a_probe_neu).item()
    cos_sim_neu = F.cosine_similarity(a_relaxed_neu, a_probe_neu).item()
    
    print(f"Delta Trajectory Force Norm (|Delta a|): {delta_force_norm:.10f}")
    print(f"Trajectory Cosine Invariance: {cos_sim_neu:.10f}")
    neutral_passed = (delta_force_norm == 0.0) and (abs(cos_sim_neu - 1.0) < 1e-7)
    print(f"Neutral Topographic Invariance Test: {'PASSED (|Delta a| = 0.0)' if neutral_passed else 'FAILED'}")

    print("\n--- PHASE 4: AGENT SOMATIC STEP FEEDBACK CONTINUOUS CALIBRATION ---")
    agent = CoREAgent(vocab_size=258, embed_dim=dim, device=device).to(device)
    
    # 4.1 Success step (loss significantly below baseline mean)
    v_success = agent.record_somatic_step_feedback(
        context_t=ctx_A,
        action_t=act_target,
        free_energy_surprise=1.5,
        mean_loss=3.0,
        std_loss=1.0
    )
    # 4.2 Error step (loss significantly above baseline mean)
    v_error = agent.record_somatic_step_feedback(
        context_t=ctx_A,
        action_t=act_target,
        free_energy_surprise=4.5,
        mean_loss=3.0,
        std_loss=1.0
    )
    # 4.3 Neutral step (loss near baseline mean)
    v_neutral = agent.record_somatic_step_feedback(
        context_t=ctx_A,
        action_t=act_target,
        free_energy_surprise=3.05,
        mean_loss=3.0,
        std_loss=1.0
    )
    
    print(f"V_success (Free Energy 1.5 vs Mean 3.0): {v_success:+.4f} (Expected > +0.4)")
    print(f"V_error   (Free Energy 4.5 vs Mean 3.0): {v_error:+.4f} (Expected < -0.6)")
    print(f"V_neutral (Free Energy 3.05 vs Mean 3.0): {v_neutral:+.4f} (Expected |V| <= 0.2)")

    calibration_passed = (v_success > 0.4) and (v_error < -0.6) and (abs(v_neutral) <= 0.2)
    print(f"Agent Calibration Test: {'PASSED' if calibration_passed else 'FAILED'}")

    # Final Verdict Assessment
    all_passed = attractor_passed and repulsor_passed and neutral_passed and calibration_passed
    print("\n" + "=" * 70)
    print(f"EXP-318 OVERALL BENCHMARK VERDICT: {'🟢 POSITIVE' if all_passed else '🔴 REJECTED'}")
    print("=" * 70)

    # Return key metrics for scientific pipeline
    return {
        "attractor_cossim": final_cos_att,
        "repulsor_cossim": final_cos_rep,
        "neutral_delta_force": delta_force_norm,
        "v_success": v_success,
        "v_error": v_error,
        "v_neutral": v_neutral,
        "verdict": "POSITIVE" if all_passed else "REJECTED"
    }

if __name__ == "__main__":
    run_exp_318_benchmark()
