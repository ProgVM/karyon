# experiments/exp_216_allostatic_affective_core_synthesis.py
"""
===============================================================================
EXP-216: Differentiable Allostatically-Modulated Affective Core Unit (AffectiveCore v2)
Grounding: KEP Principle 2 (Living AGI & Biological Realism - NON-NEGOTIABLE),
           KEP Principle 14 (Axiom of Allostatic Dynamic Forces — No Static Constants),
           KEP Principle 15 (Net2Net Smooth Grafting: Strict Identity at Birth t_0).
===============================================================================
Hypothesis:
In `karyon_agent.py` lines 208-251, `AffectiveCoreUnit` computes Russell's Affective Coordinates
(Valence, Arousal, Dominance) and Panksepp Primary Affective Drives (SEEKING, FEAR, RAGE, PANIC)
using CPU float conversions (`u_t.mean(dim=0).cpu().tolist()`) and un-differentiable `max(0.0, ...)` ops:
  valence   = DA - (1 - E) - (1 - H)
  arousal   = NA + min(1.0, max(0.0, FE))
  seeking   = max(0.0, Curiosity + DA - max(0.0, FE))

This non-differentiable CPU extraction blocks gradient propagation from intrinsic affective drives
back into the cortical representations, and fails to dynamically learn subcortical emotional weighting.

Replacing the non-differentiable CPU `AffectiveCoreUnit` with a Differentiable Allostatically-Modulated
Affective Core Unit (AffectiveCore v2) that:
  - Operates strictly on GPU tensors without `.cpu().tolist()` GPU stalls.
  - Computes continuous differentiable Russell & Panksepp emotional coordinates using smooth `F.softplus`:
      seeking_tensor = F.softplus(curiosity + da - free_energy)
  - Incorporates a zero-initialized Net2Net neural projection network (`allo_affect_proj`) to dynamically
    refine emotional valence/arousal/SEEKING modulation:
      seeking_modulated = seeking_base * (1.0 + 0.20 * tanh(allo_affect_proj(u_t)))
      (Zero-initialized at birth t_0 -> 100% Zero-Delta Identity)
will eliminate GPU-CPU pipeline stalls, enable gradient flow through intrinsic motivation, and drive
Loss Delta >= 0.08 while preserving 100% stability.
"""

import sys
import os
import time
import math
import json
import logging
from typing import Tuple, List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_entity import KaryonEntity
from karyon_hardware import get_hardware_engine

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-216")


class DifferentiableAffectiveCoreUnit(nn.Module):
    """
    Differentiable Allostatically-Modulated Affective Core Unit (AffectiveCore v2).
    Computes Russell Circumplex & Panksepp Primary Drives entirely on GPU with smooth operations
    and zero-initialized Net2Net allostatic modulation.
    """
    def __init__(self, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')

        # Net2Net Allostatic Neural Affective Modulator (KEP Principle 15 Compliant)
        self.allo_affect_proj = nn.Sequential(
            nn.Linear(6, 16),
            nn.SiLU(),
            nn.Linear(16, 4), # 4 Panksepp drives
            nn.Tanh()
        ).to(self.device)

        nn.init.zeros_(self.allo_affect_proj[0].weight)
        nn.init.zeros_(self.allo_affect_proj[0].bias)
        nn.init.zeros_(self.allo_affect_proj[2].weight)
        nn.init.zeros_(self.allo_affect_proj[2].bias)

    def compute_affective_state_tensor(
        self,
        u_t: torch.Tensor,
        free_energy: torch.Tensor = None,
        value_est: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        if u_t.dim() == 1:
            u_t = u_t.unsqueeze(0)

        curiosity = u_t[:, 0:1]
        energy    = u_t[:, 1:2]
        stability = u_t[:, 2:3]
        health    = u_t[:, 3:4]
        na        = u_t[:, 4:5]
        da        = u_t[:, 5:6]

        fe_t = free_energy if free_energy is not None else torch.tensor(0.0, device=self.device)
        if not isinstance(fe_t, torch.Tensor):
            fe_t = torch.tensor(fe_t, device=self.device)
        if fe_t.dim() == 0:
            fe_t = fe_t.unsqueeze(0).unsqueeze(0)
        if fe_t.size(0) != u_t.size(0):
            fe_t = fe_t.expand(u_t.size(0), -1)

        val_t = value_est if value_est is not None else torch.tensor(0.0, device=self.device)
        if not isinstance(val_t, torch.Tensor):
            val_t = torch.tensor(val_t, device=self.device)
        if val_t.dim() == 0:
            val_t = val_t.unsqueeze(0).unsqueeze(0)
        if val_t.size(0) != u_t.size(0):
            val_t = val_t.expand(u_t.size(0), -1)

        # Continuous Differentiable Russell Affective Coordinates
        valence   = da - (1.0 - energy) - (1.0 - health)
        arousal   = na + torch.clamp(fe_t, 0.0, 1.0)
        dominance = stability + torch.clamp(val_t, -1.0, 1.0)

        # Differentiable Panksepp Primary Drives via Softplus
        seeking_base = F.softplus(curiosity + da - fe_t)
        fear_base    = F.softplus(arousal * (1.0 - stability))
        rage_base    = F.softplus((1.0 - energy) * (1.0 - dominance))
        panic_base   = F.softplus((1.0 - health) * (1.0 - stability))

        panksepp_base = torch.cat([seeking_base, fear_base, rage_base, panic_base], dim=-1)

        # Net2Net Allostatic Neural Modulation (Identity at Birth t_0)
        allo_factors = 1.0 + 0.20 * self.allo_affect_proj(u_t)
        panksepp_modulated = panksepp_base * allo_factors

        seeking_drive = panksepp_modulated[:, 0:1]
        fear_drive    = panksepp_modulated[:, 1:2]
        rage_drive    = panksepp_modulated[:, 2:3]
        panic_drive   = panksepp_modulated[:, 3:4]

        return {
            "valence": valence.mean().item(),
            "arousal": arousal.mean().item(),
            "dominance": dominance.mean().item(),
            "seeking_tensor": seeking_drive,
            "panksepp": {
                "SEEKING": seeking_drive.mean().item(),
                "FEAR": fear_drive.mean().item(),
                "RAGE": rage_drive.mean().item(),
                "PANIC": panic_drive.mean().item()
            }
        }

    def compute_affective_state(self, u_t: torch.Tensor, free_energy: float = 0.0, value_est: float = 0.0) -> dict:
        fe_t = torch.tensor([[free_energy]], device=self.device)
        val_t = torch.tensor([[value_est]], device=self.device)
        return self.compute_affective_state_tensor(u_t, fe_t, val_t)


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
    total_tokens = sum(len(brain.tokenizer.encode(t)) - 1 for t in text_samples[:num_steps])
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "final_loss": step_losses[-1],
        "initial_loss": step_losses[0],
        "mean_loss": sum(step_losses) / len(step_losses),
        "final_fe": step_fe_losses[-1],
        "initial_fe": step_fe_losses[0],
        "tok_per_sec": tok_per_sec,
        "elapsed": elapsed
    }


def run_benchmark():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-216: DIFFERENTIABLE ALLOSTIC AFFECTIVE CORE BENCHMARK]")
    logger.info("=" * 80)

    hw = get_hardware_engine()
    logger.info(f"Target Accelerator: {hw.device_str.upper()}")

    text_samples = [
        "The quick brown fox jumps over the lazy dog near the riverbank with high agility.",
        "Active Inference formulates brain dynamics as continuous minimization of variational free energy.",
        "Homeostasis and allostasis regulate physiological variables through predictive bodily setpoints.",
        "Neural state space duality enables zero-loop associative parallel scans across deep cortical layers.",
        "Continuous Hopfield attractors snap neural trajectories into discrete conceptual semantic basins.",
        "Cortical laminar hierarchy routes top-down predictions and bottom-up precision-weighted error residuals."
    ] * 5

    # 1. Baseline Evaluation
    entity_base = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_base = entity_base.brain
    logger.info("Running Baseline Evaluation...")
    b_results = evaluate_model(brain_base, entity_base, text_samples, num_steps=25)

    # 2. Proposed Evaluation (EXP-216 Differentiable Affective Core Unit)
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain

    diff_affective = DifferentiableAffectiveCoreUnit(device_str=hw.device_str).to(hw.device)
    brain_prop.affective_core = diff_affective

    logger.info("Running Proposed Differentiable Affective Core Evaluation...")
    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    loss_delta = b_results['final_loss'] - p_results['final_loss']
    fe_delta = b_results['final_fe'] - p_results['final_fe']

    logger.info("=" * 80)
    logger.info("📊 === EXP-216 TELEMETRY REPORT ===")
    logger.info(f"  - Baseline Final Loss : {b_results['final_loss']:.4f} nats | Throughput: {b_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Proposed Final Loss : {p_results['final_loss']:.4f} nats | Throughput: {p_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Loss Delta (B - P)  : {loss_delta:.4f} nats")
    logger.info(f"  - Free Energy Delta   : {fe_delta:.6f}")
    logger.info(f"  - Baseline Duration   : {b_results['elapsed']:.3f} s")
    logger.info(f"  - Proposed Duration   : {p_results['elapsed']:.3f} s")

    verdict = "POSITIVE" if (loss_delta >= 0.08 or (loss_delta >= 0.02 and p_results['tok_per_sec'] >= 0.90 * b_results['tok_per_sec'])) else "NEUTRAL"

    results = {
        "exp_id": "EXP-216",
        "verdict": verdict,
        "base_initial_loss": b_results['initial_loss'],
        "base_final_loss": b_results['final_loss'],
        "proposed_initial_loss": p_results['initial_loss'],
        "proposed_final_loss": p_results['final_loss'],
        "loss_delta": loss_delta,
        "fe_delta": fe_delta,
        "throughput_tok_per_sec": p_results['tok_per_sec'],
        "execution_time_s": p_results['elapsed']
    }

    with open("experiments/exp_216_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"🏆 Final Verdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    return results


if __name__ == "__main__":
    run_benchmark()
