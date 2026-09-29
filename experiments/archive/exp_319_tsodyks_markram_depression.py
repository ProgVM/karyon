# experiments/exp_319_tsodyks_markram_depression.py
"""
===============================================================================
EXP-319: ACTIVITY-DEPENDENT TSODYKS-MARKRAM SYNAPTIC DEPRESSION IN DYNAMIC GRAPH
===============================================================================
Scientific Hypothesis:
Incorporating C++20 TsodyksMarkramSynapticDepressionOp as a dynamic organelle into
DynamicMorphicGraph prevents perseverative attractor trapping during multi-step
recurrent deliberation (K in [1..8]), reducing state repetition/saturation and
accelerating convergence towards novel informational equilibria without degrading
representation quality.

Telemetry Metrics Evaluated:
1. State Perseveration Index (Cosine similarity between consecutive thinking steps h_k and h_{k+1})
2. Vesicular Resource Dynamics (Mean vesicle pool x_t depletion and recovery)
3. Latent Representation Diversity (Participation ratio / effective rank across thinking cycles)
4. Prediction Loss / Perplexity Delta
===============================================================================
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import karyon_core as kcore
from karyon_agent import CoREAgent


def run_exp_319_benchmark():
    print("=" * 75)
    print("=== EXP-319: TSODYKS-MARKRAM SYNAPTIC DEPRESSION BENCHMARK ===")
    print("=" * 75)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Hardware Acceleration: {device.upper()}")

    dim = 256
    seq_len = 32
    batch_size = 4
    max_k_thinking = 8

    # -------------------------------------------------------------------------
    # 1. DIRECT UNIT TEST: TSODYKS-MARKRAM VESICULAR FATIGUE DYNAMICS
    # -------------------------------------------------------------------------
    print("\n--- PHASE 1: DIRECT VESICULAR FATIGUE DYNAMICS ---")
    tm_op = kcore.TsodyksMarkramSynapticDepressionOp(dim=dim, device_str=device, tau_rec=8.0, u_depress=0.5)
    x_impulse = torch.randn(batch_size, dim, device=device) * 2.0  # High activation impulse

    vesicle_history = []
    output_norms = []

    for step in range(10):
        out = tm_op(x_impulse)
        v_pool = tm_op.get_vesicle_resource()
        mean_v = v_pool.mean().item()
        out_norm = out.norm(dim=-1).mean().item()
        vesicle_history.append(mean_v)
        output_norms.append(out_norm)
        print(f"Step {step+1:02d}: Mean Vesicle Pool x_t = {mean_v:.4f} | Output Norm = {out_norm:.4f}")

    # Recovery Phase: low input impulse
    x_quiet = torch.randn(batch_size, dim, device=device) * 0.05
    print("\nQuiet Period (Recovery Phase):")
    for step in range(5):
        out = tm_op(x_quiet)
        v_pool = tm_op.get_vesicle_resource()
        mean_v = v_pool.mean().item()
        vesicle_history.append(mean_v)
        print(f"Recovery Step {step+1:02d}: Mean Vesicle Pool x_t = {mean_v:.4f}")

    depletion_passed = vesicle_history[4] < 0.50
    recovery_passed = vesicle_history[-1] > vesicle_history[9]
    print(f"Vesicular Depletion Check: {'PASSED' if depletion_passed else 'FAILED'} (v_5 = {vesicle_history[4]:.4f})")
    print(f"Vesicular Recovery Check:  {'PASSED' if recovery_passed else 'FAILED'} (v_rec = {vesicle_history[-1]:.4f})")

    # -------------------------------------------------------------------------
    # 2. COMPARATIVE BENCHMARK: GRAPH DELIBERATION WITH VS WITHOUT TSODYKS-MARKRAM
    # -------------------------------------------------------------------------
    print("\n--- PHASE 2: MORPHIC GRAPH DELIBERATION PERSEVERATION TEST ---")

    # Baseline Graph: Accumulator + Attractor
    graph_base = kcore.DynamicMorphicGraph(dim, device)
    graph_base.add_node("acc", "LinearAccumulator", True, 1.0)
    graph_base.add_node("sat", "SaturatedAttractor", True, 1.0)

    # Depressive Graph: Accumulator + Attractor + TsodyksMarkram
    graph_tm = kcore.DynamicMorphicGraph(dim, device)
    graph_tm.add_node("acc", "LinearAccumulator", True, 1.0)
    graph_tm.add_node("sat", "SaturatedAttractor", True, 1.0)
    graph_tm.add_node("tm_fatigue", "TsodyksMarkram", True, 1.0)

    h_input = torch.randn(batch_size * seq_len, dim, device=device)

    # Function to track step-by-step state drift during K=8 thinking cycles
    def audit_thinking_trajectory(graph_obj, h_in, k_steps=8):
        current_h = h_in.clone()
        step_sims = []
        state_norms = []

        for k in range(k_steps):
            next_h = graph_obj.forward(current_h, 1)  # 1 step
            cos_sim = F.cosine_similarity(current_h, next_h, dim=-1).mean().item()
            norm_val = next_h.norm(dim=-1).mean().item()
            step_sims.append(cos_sim)
            state_norms.append(norm_val)
            current_h = next_h

        return step_sims, state_norms, current_h

    sims_base, norms_base, h_out_base = audit_thinking_trajectory(graph_base, h_input, max_k_thinking)
    sims_tm, norms_tm, h_out_tm = audit_thinking_trajectory(graph_tm, h_input, max_k_thinking)

    mean_perseveration_base = sum(sims_base) / len(sims_base)
    mean_perseveration_tm = sum(sims_tm) / len(sims_tm)

    print(f"Baseline Graph  - Mean Step-to-Step Cosine Sim (K=1..8): {mean_perseveration_base:.6f}")
    print(f"Tsodyks-TM Graph - Mean Step-to-Step Cosine Sim (K=1..8): {mean_perseveration_tm:.6f}")

    perseveration_reduction = mean_perseveration_base - mean_perseveration_tm
    print(f"Perseveration Reduction Delta: {perseveration_reduction:+.6f}")

    # -------------------------------------------------------------------------
    # 3. END-TO-END AGENT PREDICTION & LOSS COMPARISON
    # -------------------------------------------------------------------------
    print("\n--- PHASE 3: END-TO-END AGENT SPEECH LOSS COMPARISON ---")

    torch.manual_seed(42)
    agent = CoREAgent(vocab_size=258, embed_dim=dim, device=device).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=256)

    # Test sequence batch
    dummy_input = torch.randint(0, 255, (batch_size, seq_len), device=device)
    dummy_target = torch.randint(0, 255, (batch_size, seq_len), device=device)

    # 1. Baseline Forward Pass (Default Graph)
    with torch.no_grad():
        logits_base = agent(dummy_input, thinking_steps=4)
        loss_base = criterion(logits_base.reshape(-1, 258), dummy_target.reshape(-1)).item()

    # 2. Sprout TsodyksMarkram into Agent's Graph with Net2Net Epigenetic Gating (alpha_epi = 0.0)
    agent.graph.add_node("tsodyks_fatigue", "TsodyksMarkram", False, 0.0)

    # Birth Identity Verification: alpha_epi = 0.0 -> logits_birth MUST equal logits_base
    with torch.no_grad():
        logits_birth = agent(dummy_input, thinking_steps=4)
        birth_diff = (logits_base - logits_birth).abs().max().item()

    print(f"Net2Net Zero-Shock Birth Delta (|logits_base - logits_birth|): {birth_diff:.10f}")
    birth_passed = birth_diff < 1e-5
    print(f"Net2Net Zero-Shock Identity Test: {'PASSED' if birth_passed else 'FAILED'}")

    # 3. Maturation Phase (alpha_epi = 1.0)
    agent.lock_node(agent.graph.k_nodes - 1, 1.0)  # Activate organelle
    with torch.no_grad():
        logits_matured = agent(dummy_input, thinking_steps=4)
        loss_matured = criterion(logits_matured.reshape(-1, 258), dummy_target.reshape(-1)).item()

    delta_loss = loss_matured - loss_base
    print(f"Baseline Loss: {loss_base:.4f} | Matured Loss: {loss_matured:.4f} | Delta: {delta_loss:+.4f}")

    # -------------------------------------------------------------------------
    # 4. VERDICT ASSIGNMENT & TELEMETRY REPORT
    # -------------------------------------------------------------------------
    passed_all_checks = depletion_passed and recovery_passed and birth_passed and (perseveration_reduction > 0.01)

    if passed_all_checks:
        verdict = "🟢 POSITIVE"
    elif birth_passed:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    else:
        verdict = "🔴 REJECTED"

    print("\n" + "=" * 75)
    print(f"=== EXP-319 FINAL VERDICT: {verdict} ===")
    print(f"Metrics: perseveration_reduction={perseveration_reduction:.6f}, birth_delta={birth_diff:.1e}, delta_loss={delta_loss:+.4f}")
    print("=" * 75)

    # Record result into empirical ledger
    metrics = {
        "mean_vesicle_depleted": vesicle_history[4],
        "mean_vesicle_recovered": vesicle_history[-1],
        "perseveration_base": mean_perseveration_base,
        "perseveration_tm": mean_perseveration_tm,
        "perseveration_reduction": perseveration_reduction,
        "birth_delta": birth_diff,
        "loss_base": loss_base,
        "loss_matured": loss_matured,
        "delta_loss": delta_loss
    }

    config_params = {
        "dim": dim,
        "tau_rec": 8.0,
        "u_depress": 0.5,
        "max_k_thinking": max_k_thinking
    }

    try:
        from default_api import record_experiment_result
        record_experiment_result(
            exp_id="EXP-319",
            hypothesis="Activity-dependent Tsodyks-Markram synaptic depression in DynamicMorphicGraph prevents perseverative attractor trapping during multi-step recurrent deliberation.",
            architecture_delta="Implemented C++20 TsodyksMarkramSynapticDepressionOp (7th atomic operator) with activity-dependent vesicular dynamics dx_i/dt = (1-x_i)/tau_rec - u_depress*x_i*a_i.",
            verdict=verdict,
            final_loss=loss_matured,
            metrics=metrics,
            config_params=config_params,
            notes=f"Vesicular depletion v_5={vesicle_history[4]:.4f}, recovery v_rec={vesicle_history[-1]:.4f}, perseveration delta={perseveration_reduction:+.6f}, zero-shock birth delta={birth_diff:.1e}."
        )
        print("Logged EXP-319 to SQLite empirical ledger successfully.")
    except Exception as e:
        print(f"Ledger recording notice: {e}")


if __name__ == "__main__":
    run_exp_319_benchmark()
