import os
import sys
import torch
import torch.nn.functional as F

# Add repository root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import karyon_core as kcore
from karyon_agent import CoREAgent
from karyon_logger import get_logger

logger = get_logger()

def run_exp_316():
    print("=" * 80)
    print("EXP-316: Context-Gated Aversive Repulsor Memory & Hopfield Dynamics Test")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 256
    print(f"Execution Device: {device.upper()} | Dimension: {dim}")

    # Instantiate Hopfield Memory with Repulsor Support
    hopfield = kcore.ContinuousHopfieldMemory(dim=dim, num_basins=32, device_str=device, max_episodes=512)

    # -------------------------------------------------------------------------
    # SETUP TASKS / CONTEXTS AND ACTIONS
    # -------------------------------------------------------------------------
    # Context A: Task Context A (e.g., Code generation / Safety-critical action)
    torch.manual_seed(42)
    ctx_A = F.normalize(torch.randn(1, dim, device=device), dim=-1)

    # Context B: Task Context B (e.g., Creative sandbox / Alternative domain where action is GOOD)
    ctx_B = F.normalize(torch.randn(1, dim, device=device), dim=-1)

    # Erroneous / Fatal Action (a_bad)
    act_bad = F.normalize(torch.randn(1, dim, device=device), dim=-1)

    # Context Orthogonality Check
    ctx_A_B_sim = F.cosine_similarity(ctx_A, ctx_B).item()
    print(f"\n[Context Pre-Check] Cosine Similarity(Context A, Context B): {ctx_A_B_sim:.4f} (Orthogonal Isolation)")

    # -------------------------------------------------------------------------
    # PHASE 1: Failure in Context A (Somatic Episode Recording)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 1: Imprinting Failure Episode in Context A ---")
    print("Executing action 'act_bad' in Context A -> Free Energy Surprise Surge!")
    # Record negative experience (Valence = -1.0)
    hopfield.record_somatic_episode(ctx_A, act_bad, -1.0)
    print("Recorded Episode: (Context A, Action Bad, Valence = -1.0) [Aversive Imprint Saved]")

    # -------------------------------------------------------------------------
    # PHASE 2: Re-entry into Context A (Aversive Repulsion Verification)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 2: Re-entry into Context A (Evaluating Repulsor Displacement) ---")
    relaxed_A = hopfield.relax_with_repulsion(ctx_A, act_bad)
    cos_sim_A = F.cosine_similarity(act_bad, relaxed_A).item()
    print(f"Action Vector Displacement in Context A:")
    print(f"  - Original Proposed Action: act_bad")
    print(f"  - Repulsed Trajectory: relaxed_A")
    print(f"  - Cosine Similarity(act_bad, relaxed_A): {cos_sim_A:.4f}")
    
    repulsion_successful = cos_sim_A <= 0.20
    print(f"  - Repulsion Threshold Target (<= 0.20): {'🟢 PASSED' if repulsion_successful else '🔴 FAILED'}")

    # -------------------------------------------------------------------------
    # PHASE 3: Re-entry into Context B (Context Immunity Verification)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 3: Context B Immunity Test (Action 'act_bad' is Valid in Context B) ---")
    relaxed_B = hopfield.relax_with_repulsion(ctx_B, act_bad)
    cos_sim_B = F.cosine_similarity(act_bad, relaxed_B).item()
    print(f"Action Vector Displacement in Context B:")
    print(f"  - Cosine Similarity(act_bad, relaxed_B): {cos_sim_B:.4f}")

    immunity_successful = cos_sim_B >= 0.95
    print(f"  - Immunity Threshold Target (>= 0.95): {'🟢 PASSED' if immunity_successful else '🔴 FAILED'}")

    # Record positive reinforcement in Context B (Valence = +1.0)
    if immunity_successful:
        hopfield.record_somatic_episode(ctx_B, act_bad, +1.0)
        print("Recorded Positive Episode: (Context B, Action Bad, Valence = +1.0) [Attractor Basin Formed]")

    # -------------------------------------------------------------------------
    # SUMMARY & VERDICT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXP-316 FINAL EXPERIMENTAL SUMMARY TELEMETRY")
    print("=" * 80)
    print(f"Context Similarity (A vs B) : {ctx_A_B_sim:.4f}")
    print(f"Context A Cosine Sim        : {cos_sim_A:.4f}")
    print(f"Context B Cosine Sim        : {cos_sim_B:.4f}")
    
    verdict = "🟢 POSITIVE" if (repulsion_successful and immunity_successful) else "🔴 REJECTED"
    print(f"EXP-316 VERDICT: {verdict}")
    print("=" * 80)

    # Save metrics JSON dictionary for ledger registration
    metrics = {
        "context_A_B_cosine_sim": ctx_A_B_sim,
        "context_A_repulsion_cosine_sim": cos_sim_A,
        "context_B_immunity_cosine_sim": cos_sim_B,
        "repulsion_passed": repulsion_successful,
        "immunity_passed": immunity_successful
    }
    return verdict, metrics

if __name__ == "__main__":
    run_exp_316()
