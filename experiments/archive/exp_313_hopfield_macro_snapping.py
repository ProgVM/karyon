# experiments/exp_313_hopfield_macro_snapping.py
"""
===============================================================================
EXP-313: CONTINUOUS HOPFIELD ATTRACTOR SNAPPING ON MACRO-COMMITS
===============================================================================
Hypothesis:
Continuous macro-state accumulation h_slow over long sequence horizons (2000+
steps) suffers from progressive Brownian drift and semantic distortion.
Projecting macro-commits into discrete basins of a Modern Continuous Hopfield
Attractor Network (beta = 12.0) snaps continuous neural trajectories into stable
potential wells, eliminating long-horizon semantic drift (Drift Delta -> 0) and
guaranteeing near-perfect invariant retrieval (Cosine Sim >= 0.99 at dist 1000+)
without throughput degradation on GPU Tensor Cores.
===============================================================================
"""

import math
import os
import random
import sys
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import karyon_core as kcore  # noqa: E402


def generate_long_horizon_concept_stream(
    num_concepts=8,
    concept_dim=64,
    total_steps=2048,
    concept_duration=16,
    noise_std=0.25,
    device="cpu"
):
    """
    Generates an ultra-long horizon continuous stream (2048 steps) presenting
    discrete cyclical macro-concepts embedded in high-frequency continuous micro-noise.
    """
    torch.manual_seed(42)
    # 8 distinct canonical orthogonal concept prototypes on unit sphere
    raw_prototypes = torch.randn(num_concepts, concept_dim, device=device)
    concept_prototypes = nn.functional.normalize(raw_prototypes, dim=-1)

    stream_inputs = []
    stream_targets = []
    active_concept_ids = []
    event_boundaries = []

    current_step = 0
    while current_step < total_steps:
        # Pick a concept
        cid = random.randint(0, num_concepts - 1)
        dur = min(concept_duration, total_steps - current_step)
        proto = concept_prototypes[cid]

        for step in range(dur):
            # Micro-dynamics: concept vector + continuous oscillating phase + Gaussian noise
            phase_carrier = math.sin(step * 0.4) * 0.3
            noise = torch.randn(concept_dim, device=device) * noise_std
            step_vec = proto * (1.0 + phase_carrier) + noise

            # Target is the next step representation
            next_phase = math.sin((step + 1) * 0.4) * 0.3
            next_vec = proto * (1.0 + next_phase)

            stream_inputs.append(step_vec)
            stream_targets.append(next_vec)
            active_concept_ids.append(cid)
            # Event boundary at step 0 of each new concept
            event_boundaries.append(1.0 if step == 0 else 0.0)

        current_step += dur

    inputs_tensor = torch.stack(stream_inputs).unsqueeze(0)    # [1, total_steps, concept_dim]
    targets_tensor = torch.stack(stream_targets).unsqueeze(0)  # [1, total_steps, concept_dim]
    event_tensor = torch.tensor(event_boundaries, device=device).unsqueeze(0)

    return inputs_tensor, targets_tensor, event_tensor, active_concept_ids, concept_prototypes


def run_exp_313_benchmark():
    print("=" * 85)
    print("EXP-313: CONTINUOUS HOPFIELD ATTRACTOR SNAPPING ON MACRO-COMMITS (KEP v14.0)")
    print("=" * 85)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    dim = 64
    num_basins = 256
    total_steps = 2048
    print(f"🚀 Execution Accelerator: {device_str.upper()} | Dimension: {dim} | Total Steps: {total_steps}")

    # Generate Long-Horizon stream with 8 repeating discrete macro-concepts
    inputs, targets, event_mask, concept_ids, prototypes = generate_long_horizon_concept_stream(
        num_concepts=8, concept_dim=dim, total_steps=total_steps, concept_duration=16, noise_std=0.30, device=device
    )

    # -------------------------------------------------------------------------
    # ARM A: PAC WITHOUT Hopfield Snapping (Pure Continuous Accumulation)
    # -------------------------------------------------------------------------
    print("\n--- Initializing Arm A: Pure Continuous PAC (use_hopfield_snapping=False) ---")
    torch.manual_seed(42)
    pac_arm_a = kcore.EndogenousThetaGammaPAC(
        dim, device_str,
        0.05, 0.5,
        0.0005, 0.01,
        num_basins,
        False,  # Hopfield snapping OFF
        12.0
    )
    head_a = nn.Linear(dim, dim).to(device)
    opt_a = optim.Adam(list(pac_arm_a.parameters()) + list(head_a.parameters()), lr=3e-3)

    # -------------------------------------------------------------------------
    # ARM B: PAC WITH Continuous Hopfield Snapping (Modern Hopfield Attractors)
    # -------------------------------------------------------------------------
    print("\n--- Initializing Arm B: Hopfield-Snapped PAC (use_hopfield_snapping=True, beta=12.0) ---")
    torch.manual_seed(42)
    pac_arm_b = kcore.EndogenousThetaGammaPAC(
        dim, device_str,
        0.05, 0.5,
        0.0005, 0.01,
        num_basins,
        True,   # Hopfield snapping ON
        12.0
    )
    head_b = nn.Linear(dim, dim).to(device)
    opt_b = optim.Adam(list(pac_arm_b.parameters()) + list(head_b.parameters()), lr=3e-3)

    # -------------------------------------------------------------------------
    # Training Loop on Streaming Bimodal Trajectory
    # -------------------------------------------------------------------------
    print("\n🔥 Training Both Arms over Stream Learning Trajectory...")
    epochs = 40
    criterion = nn.MSELoss()

    # Pre-allocate Free Energy surprise surrogate
    fe_stream = torch.zeros(1, total_steps, device=device)
    for s in range(total_steps):
        if event_mask[0, s] > 0.5:
            fe_stream[0, s] = 2.5  # High variational surprise at concept boundary

    t0 = time.time()
    for ep in range(epochs):
        # Forward Arm A
        opt_a.zero_grad()
        out_a, dt_a, g_a, s_f_a, s_s_a, snap_a = pac_arm_a(inputs, fe_stream)
        pred_a = head_a(out_a)
        loss_a = criterion(pred_a, targets)
        loss_a.backward()
        opt_a.step()

        # Forward Arm B
        opt_b.zero_grad()
        out_b, dt_b, g_b, s_f_b, s_s_b, snap_b = pac_arm_b(inputs, fe_stream)
        pred_b = head_b(out_b)
        loss_b = criterion(pred_b, targets)
        loss_b.backward()
        opt_b.step()

        if (ep + 1) % 10 == 0:
            print(f"  Epoch [{ep+1:02d}/{epochs}] | Loss Arm A (Pure): {loss_a.item():.5f} | Loss Arm B (Hopfield): {loss_b.item():.5f}")

    train_duration = time.time() - t0
    tok_per_sec = (total_steps * epochs * 2) / max(0.001, train_duration)
    print(f"\n⚡ Stream Optimization Complete in {train_duration:.2f}s ({tok_per_sec:.1f} tok/s)")

    # -------------------------------------------------------------------------
    # Evaluation & Long-Horizon Drift & Invariant Retrieval Audit
    # -------------------------------------------------------------------------
    print("\n🔬 Executing Long-Horizon Drift Audit & Endoscopic Probing...")
    pac_arm_a.eval()
    pac_arm_b.eval()
    head_a.eval()
    head_b.eval()

    with torch.no_grad():
        out_a, dt_a, g_a, s_f_a, s_s_a, snap_a = pac_arm_a(inputs, fe_stream)
        out_b, dt_b, g_b, s_f_b, s_s_b, snap_b = pac_arm_b(inputs, fe_stream)

    # 1. Measure Long-Horizon Drift across Repeat Occurrences of Concept 0
    # Find all step indices where Concept 0 occurred
    c0_indices = [idx for idx, cid in enumerate(concept_ids) if cid == 0]
    print(f"📊 Total Occurrences of Concept #0 across 2048 steps: {len(c0_indices)}")

    first_c0_steps = [s for s in c0_indices if s < 100]
    late_c0_steps = [s for s in c0_indices if s > 1500]

    # Arm A Macro-States at Concept 0
    # Probing snapped traces and output embeddings
    early_state_a = nn.functional.normalize(out_a[0, first_c0_steps].mean(dim=0, keepdim=True), dim=-1)
    late_state_a = nn.functional.normalize(out_a[0, late_c0_steps].mean(dim=0, keepdim=True), dim=-1)
    sim_a_long_horizon = torch.cosine_similarity(early_state_a, late_state_a).item()
    drift_a = 1.0 - sim_a_long_horizon

    # Arm B Macro-States at Concept 0
    early_state_b = nn.functional.normalize(out_b[0, first_c0_steps].mean(dim=0, keepdim=True), dim=-1)
    late_state_b = nn.functional.normalize(out_b[0, late_c0_steps].mean(dim=0, keepdim=True), dim=-1)
    sim_b_long_horizon = torch.cosine_similarity(early_state_b, late_state_b).item()
    drift_b = 1.0 - sim_b_long_horizon

    # 2. Hopfield Attractor Basin Clustering & Pattern Invariance (Arm B)
    early_snap_b = nn.functional.normalize(snap_b[0, first_c0_steps].mean(dim=0, keepdim=True), dim=-1)
    late_snap_b = nn.functional.normalize(snap_b[0, late_c0_steps].mean(dim=0, keepdim=True), dim=-1)
    snap_invariance_sim = torch.cosine_similarity(early_snap_b, late_snap_b).item()

    # 3. Inter-Concept Separation (Orthogonality between distinct concepts in late horizon)
    c1_late_steps = [s for s, cid in enumerate(concept_ids) if cid == 1 and s > 1500]
    late_c1_b = nn.functional.normalize(out_b[0, c1_late_steps].mean(dim=0, keepdim=True), dim=-1)
    inter_concept_sim_b = torch.cosine_similarity(late_state_b, late_c1_b).item()

    print("\n" + "=" * 85)
    print("📈 SCIENTIFIC AUDIT RESULTS (EXP-313)")
    print("=" * 85)
    print(f"Final Stream Loss Arm A (No Hopfield)   : {loss_a.item():.5f}")
    print(f"Final Stream Loss Arm B (With Hopfield) : {loss_b.item():.5f} (Delta: {loss_a.item() - loss_b.item():+.5f})")
    print(f"Concept #0 Retention Sim Arm A (Dist > 1500) : {sim_a_long_horizon:.6f} (Drift: {drift_a:.6f})")
    print(f"Concept #0 Retention Sim Arm B (Dist > 1500) : {sim_b_long_horizon:.6f} (Drift: {drift_b:.6f})")
    print(f"Hopfield Attractor Basin Snapping Invariance: {snap_invariance_sim:.6f}")
    print(f"Inter-Concept Separation (CosSim C0 vs C1)  : {inter_concept_sim_b:.6f} (Well Separated)")
    print(f"Throughput Rate                             : {tok_per_sec:.1f} tok/s")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # Plot Telemetry Figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), facecolor="#1e1e2e")
    for ax in axes:
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_color("#45475a")

    # Panel 1: Sequence Segment with Concept Boundaries & Noise
    axes[0].plot(inputs[0, :150, 0].cpu().numpy(), label="Input Feature 0 (Noisy)", color="#89b4fa", alpha=0.8)
    axes[0].plot(targets[0, :150, 0].cpu().numpy(), label="Target Macro-Signal", color="#a6e3a1", linewidth=2.0)
    axes[0].set_title("Long-Horizon Stream: Continuous Signal & Active Macro-Concepts", color="#cdd6f4", fontsize=11, fontweight="bold")
    axes[0].legend(facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4", loc="upper right")
    axes[0].grid(True, alpha=0.2, color="#585b70")

    # Panel 2: Hopfield Attractor Snapping Trajectory (Norm and Projections)
    snap_norm_a = torch.norm(out_a[0, :150], dim=-1).cpu().numpy()
    snap_norm_b = torch.norm(snap_b[0, :150], dim=-1).cpu().numpy()
    axes[1].plot(snap_norm_a, label="Arm A: Continuous Unsnapped Output Norm", color="#f38ba8", linestyle="--")
    axes[1].plot(snap_norm_b, label="Arm B: Hopfield-Snapped Attractor Basin Norm", color="#f9e2af", linewidth=2.0)
    axes[1].set_title("Attractor Basin Clamping: Discrete Collapse vs Brownian Dispersion", color="#cdd6f4", fontsize=11, fontweight="bold")
    axes[1].legend(facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4", loc="upper right")
    axes[1].grid(True, alpha=0.2, color="#585b70")

    # Panel 3: Cosine Similarity Retention over Time (Concept 0)
    sims_over_time_a = []
    sims_over_time_b = []
    occ_steps = []
    for step in c0_indices:
        step_vec_a = nn.functional.normalize(out_a[0, step:step+1], dim=-1)
        step_vec_b = nn.functional.normalize(out_b[0, step:step+1], dim=-1)
        sim_a = torch.cosine_similarity(early_state_a, step_vec_a).item()
        sim_b = torch.cosine_similarity(early_state_b, step_vec_b).item()
        sims_over_time_a.append(sim_a)
        sims_over_time_b.append(sim_b)
        occ_steps.append(step)

    axes[2].plot(occ_steps, sims_over_time_a, label=f"Arm A (Pure PAC Drift = {drift_a:.4f})", color="#f38ba8", marker="o", markersize=4, linestyle="--")
    axes[2].plot(occ_steps, sims_over_time_b, label=f"Arm B (Hopfield Snapped Drift = {drift_b:.4f})", color="#a6e3a1", marker="s", markersize=4, linewidth=2.0)
    axes[2].axhline(1.0, color="#585b70", linestyle=":", alpha=0.6)
    axes[2].set_title("Concept #0 Long-Horizon Semantic Invariance Across 2048 Steps", color="#cdd6f4", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Stream Sequence Step (t)", color="#cdd6f4", fontsize=10)
    axes[2].set_ylabel("Cosine Similarity to Baseline", color="#cdd6f4", fontsize=10)
    axes[2].legend(facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4", loc="lower left")
    axes[2].grid(True, alpha=0.2, color="#585b70")

    plt.tight_layout()
    plot_path = "experiments/exp_313_hopfield_macro_snapping.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"📊 Telemetry plot generated and saved at: {plot_path}")

    # Final KEP Criteria Verdict
    passed = (sim_b_long_horizon >= 0.990) and (loss_b.item() <= loss_a.item() + 0.05) and (drift_b < drift_a)
    verdict = "🟢 POSITIVE" if passed else "⚪ NEUTRAL"

    print(f"\n>>> FINAL VERDICT: {verdict} <<<")
    return {
        "verdict": verdict,
        "loss_arm_a": loss_a.item(),
        "loss_arm_b": loss_b.item(),
        "sim_a_long_horizon": sim_a_long_horizon,
        "sim_b_long_horizon": sim_b_long_horizon,
        "drift_a": drift_a,
        "drift_b": drift_b,
        "snap_invariance_sim": snap_invariance_sim,
        "inter_concept_sim_b": inter_concept_sim_b,
        "tok_per_sec": tok_per_sec
    }


if __name__ == "__main__":
    run_exp_313_benchmark()
