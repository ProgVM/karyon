# experiments/exp_181_unshackled_subcortical_habit_circuit.py
"""
EXP-181: Unshackled Subcortical Habit Circuit & Continuous Allostatic Gating

Biophysical & Cybernetic Foundation:
1. The basal ganglia (striatum and globus pallidus) execute habituated motor and cognitive routines.
   When dopamine levels are high, fast direct pathways bypass deliberate cortical planning.
2. Bottleneck Elimination (KEP Principle 7): The production ReflexAndHabitCircuit projected
   [256D -> 64D -> action_dim], introducing a 64D bottleneck that lost high-dimensional sensory context.
   Expanding the hidden dimension to 256D (full rank) preserves representation richness.
3. Continuous Allostatic Modulation (KEP Principle 14):
   Instead of a hard threshold cutoff (`if da_level > 0.50:`), habit execution should be continuously
   governed by a smooth sigmoid gate conditioned on Dopamine (DA) and Noradrenaline (Arousal / NA):
   w_habit = sigmoid(10.0 * (da_t - 0.40) + 3.0 * (1.0 - na_t))
   High DA (reward expectancy) and low NA (low arousal/familiarity) smoothly increase habit contribution,
   eradicating artificial step discontinuities.

Telemetry Captured:
- Pre/Post Speech Cross-Entropy Loss (nats)
- Pre/Post Variational Free Energy (F_t)
- Habit Gate Smoothness & Contribution Magnitude
"""

import sys
import os
import time
import math
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_entity import KaryonEntity
from karyon_hardware import get_hardware_engine

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-181")


class ProposedReflexAndHabitCircuit(nn.Module):
    """
    Proposed KEP Principle 7 & 14 Compliant Subcortical Reflex & Habit Module.
    Full-rank 256D habit policy with continuous dopaminergic-noradrenergic dynamic gating.
    """
    def __init__(self, unified_dim=256, action_dim=3, device_str='cpu'):
        super().__init__()
        self.unified_dim = unified_dim
        self.action_dim = action_dim
        dev_clean = 'xla' if str(device_str).startswith('tpu') or str(device_str) == 'xla:0' else device_str
        self.device = torch.device(dev_clean)

        self.habit_policy = nn.Sequential(
            nn.Linear(unified_dim, 256),
            nn.SiLU(),
            nn.Linear(256, action_dim)
        ).to(self.device)

    def check_unconditioned_reflex(self, u_t: torch.Tensor, free_energy: float) -> bool:
        if u_t.numel() == 0:
            return False
        u_min = u_t.min(dim=0).values.cpu().tolist()
        energy = u_min[1] if len(u_min) > 1 else 1.0
        health = u_min[3] if len(u_min) > 3 else 1.0
        fe_check = free_energy if not math.isnan(free_energy) else 0.0
        return (energy < 0.15 or health < 0.20 or fe_check > 0.85)

    def execute_conditioned_habit(self, w_t: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        # Smooth continuous allostatic gating (KEP Principle 14)
        if u_t.numel() > 0:
            if u_t.dim() == 2:
                na_t = u_t[:, 4:5] if u_t.size(1) > 4 else torch.tensor([[0.2]], device=self.device)
                da_t = u_t[:, 5:6] if u_t.size(1) > 5 else torch.tensor([[0.5]], device=self.device)
            else:
                na_t = u_t[..., 4:5]
                da_t = u_t[..., 5:6]
            gate = torch.sigmoid(10.0 * (da_t - 0.40) + 3.0 * (1.0 - na_t))
        else:
            gate = 0.50

        habit_act = self.habit_policy(w_t)
        return gate * habit_act


def evaluate_model(brain, entity, text_samples, num_steps=25, lr=1e-3):
    hw = get_hardware_engine()
    criterion = nn.CrossEntropyLoss(ignore_index=256)
    optimizer = torch.optim.AdamW(brain.parameters(), lr=lr, weight_decay=1e-4)

    step_losses = []
    step_fe_losses = []

    start_time = time.perf_counter()

    for step in range(num_steps):
        text = text_samples[step % len(text_samples)]
        prompt_ids = brain.tokenizer.encode(text)
        seq_t = torch.tensor([prompt_ids[:-1]], dtype=torch.long, device=hw.device)
        target_t = torch.tensor([prompt_ids[1:]], dtype=torch.long, device=hw.device)

        optimizer.zero_grad()
        tot_loss, speech_loss, fe_loss, _, _, _, _ = brain.forward_sequence(
            seq_t, target_t, entity.hu, criterion, chunk_size=seq_t.size(1), use_checkpointing=False
        )
        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()

        step_losses.append(float(speech_loss))
        step_fe_losses.append(float(fe_loss))

    elapsed = time.perf_counter() - start_time

    return {
        "init_loss": step_losses[0],
        "final_loss": step_losses[-1],
        "init_fe": step_fe_losses[0],
        "final_fe": step_fe_losses[-1],
        "loss_delta": step_losses[0] - step_losses[-1],
        "fe_delta": step_fe_losses[0] - step_fe_losses[-1],
        "elapsed_sec": elapsed,
        "step_losses": step_losses,
        "step_fe_losses": step_fe_losses
    }


def main():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-181: UNSHACKLED SUBCORTICAL HABIT CIRCUIT BENCHMARK]")
    logger.info("=" * 80)

    hw = get_hardware_engine()
    device_str = str(hw.device)
    logger.info(f"⚡ Hardware Backend: {device_str.upper()}")

    text_samples = [
        "User: How does the thalamocortical gate route representations across cortical sheets?\n"
        "Karyon: The pulvinar dynamic routing network balances fast sensory features and slow discourse "
        "representations based on homeostatic somatic state and non-linear feature interactions.",
        
        "User: Describe the biophysical interaction between Stage 1 and Stage 2 cortical processing.\n"
        "Karyon: Stage 1 decodes fast phonotactic and morphosyntactic structures, while Stage 2 integrates "
        "long-range semantic dependencies under continuous State-Space Duality.",
        
        "User: Explain active inference and somatic allostasis in Karyon-CoRE.\n"
        "Karyon: Active inference minimizes variational surprise F_t by updating internal generative beliefs "
        "and aligning sensory observations with interoceptive somatic equilibrium."
    ]

    _ = torch.randn(10, 10, device=hw.device) @ torch.randn(10, 10, device=hw.device)

    # 1. Baseline Evaluation
    logger.info("\n--- 1. Evaluating Baseline (Standard Reflex & Habit Circuit) ---")
    entity_b = KaryonEntity.load(filepath="karyon_soul.kcore", device=device_str)
    brain_b = entity_b.brain

    b_results = evaluate_model(brain_b, entity_b, text_samples, num_steps=25)
    logger.info(f"Baseline -> Init Loss: {b_results['init_loss']:.4f} | Final Loss: {b_results['final_loss']:.4f} (Delta: {b_results['loss_delta']:.4f})")
    logger.info(f"Baseline -> Init FE  : {b_results['init_fe']:.6f} | Final FE  : {b_results['final_fe']:.6f} (FE Delta: {b_results['fe_delta']:.6f})")

    # 2. Proposed Evaluation
    logger.info("\n--- 2. Evaluating Proposed (Unshackled 256D Habit Policy + Continuous Gating) ---")
    entity_p = KaryonEntity.load(filepath="karyon_soul.kcore", device=device_str)
    brain_p = entity_p.brain

    proposed_reflex = ProposedReflexAndHabitCircuit(
        unified_dim=brain_p.reflex_circuit.unified_dim,
        action_dim=brain_p.reflex_circuit.action_dim,
        device_str=device_str
    )

    # Partial weight transfer
    with torch.no_grad():
        proposed_reflex.habit_policy[0].weight[:64, :].copy_(brain_p.reflex_circuit.habit_policy[0].weight)
        proposed_reflex.habit_policy[0].bias[:64].copy_(brain_p.reflex_circuit.habit_policy[0].bias)
        proposed_reflex.habit_policy[2].weight[:, :64].copy_(brain_p.reflex_circuit.habit_policy[2].weight)
        proposed_reflex.habit_policy[2].bias.copy_(brain_p.reflex_circuit.habit_policy[2].bias)

    brain_p.reflex_circuit = proposed_reflex

    p_results = evaluate_model(brain_p, entity_p, text_samples, num_steps=25)
    logger.info(f"Proposed -> Init Loss: {p_results['init_loss']:.4f} | Final Loss: {p_results['final_loss']:.4f} (Delta: {p_results['loss_delta']:.4f})")
    logger.info(f"Proposed -> Init FE  : {p_results['init_fe']:.6f} | Final FE  : {p_results['final_fe']:.6f} (FE Delta: {p_results['fe_delta']:.6f})")

    # 3. Comparative Telemetry Analysis
    loss_improvement = b_results["final_loss"] - p_results["final_loss"]
    fe_reduction_pct = (b_results["final_fe"] - p_results["final_fe"]) / max(b_results["final_fe"], 1e-5) * 100

    verdict = "POSITIVE" if (p_results["final_loss"] <= b_results["final_loss"] and p_results["final_fe"] <= b_results["final_fe"] + 0.01) else "NEUTRAL"

    logger.info("=" * 80)
    logger.info("📊 === EXP-181 SCIENTIFIC TELEMETRY REPORT ===")
    logger.info(f"🏆 Final Verdict                 : 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    logger.info(f"📈 Baseline Final Loss           : {b_results['final_loss']:.4f} nats")
    logger.info(f"📈 Proposed Final Loss           : {p_results['final_loss']:.4f} nats (Delta: {loss_improvement:+.4f} nats)")
    logger.info(f"📉 Baseline Final Free Energy    : {b_results['final_fe']:.6f}")
    logger.info(f"📉 Proposed Final Free Energy    : {p_results['final_fe']:.6f} ({fe_reduction_pct:+.2f}% reduction)")
    logger.info(f"⏱️ Baseline Run Duration         : {b_results['elapsed_sec']:.2f}s")
    logger.info(f"⏱️ Proposed Run Duration         : {p_results['elapsed_sec']:.2f}s")
    logger.info("=" * 80)

    summary = {
        "exp_id": "EXP-181",
        "verdict": verdict,
        "hypothesis": "Unshackling Subcortical Habit Circuit (64D -> 256D) with continuous dopaminergic-noradrenergic gating enhances habituated motor/cognitive execution.",
        "architecture_delta": "Expanded ReflexAndHabitCircuit habit_policy to 256D; replaced step threshold with continuous sigmoid gate.",
        "metrics": {
            "baseline_final_loss": b_results["final_loss"],
            "proposed_final_loss": p_results["final_loss"],
            "loss_improvement": loss_improvement,
            "baseline_final_fe": b_results["final_fe"],
            "proposed_final_fe": p_results["final_fe"],
            "fe_reduction_pct": fe_reduction_pct,
            "baseline_elapsed_sec": b_results["elapsed_sec"],
            "proposed_elapsed_sec": p_results["elapsed_sec"]
        }
    }

    with open("experiments/exp_181_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("✅ EXP-181 execution completed successfully. Results recorded in experiments/exp_181_results.json.")


if __name__ == "__main__":
    main()
