# experiments/exp_218_allostatic_fast_weight_hebbian_ssd.py
"""
===============================================================================
EXP-218: Allostatically-Gated Fast-Weight Hebbian SSD State Modulation
Grounding: KEP Principle 2 (Living AGI & Biological Realism - NON-NEGOTIABLE),
           Principle 14 (Axiom of Allostatic Dynamic Forces — No Static Constants),
           Principle 15 (Net2Net Smooth Grafting: Strict Zero-Delta Identity at Birth t_0).
===============================================================================
Hypothesis:
In `karyon_agent.py`, State-Space Duality (SSD) state transitions update chunk states linearly.
Integrating an Allostatically-Gated Fast-Weight Hebbian Trace to the SSD state matrix:
  M_fast = M_t + eta_hebb * tanh(W_allo * u_t) * (K^T @ V)
with 100% Net2Net zero-initialization on W_allo at birth t_0 (KEP Principle 15) allows the
cortical sheets to dynamically adjust associative state persistence based on 6D homeostatic
arousal/energy, accelerating short-term contextual memory retrieval and driving Loss Delta >= 0.08.
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
logger = logging.getLogger("EXP-218")


class AllostaticFastWeightHebbianSSD(nn.Module):
    """
    Net2Net Zero-Initialized Allostatically-Gated Fast-Weight Hebbian Trace for SSD Memory.
    Features strict Zero-Delta Identity at birth t_0 (KEP Principle 15).
    """
    def __init__(self, homeo_dim: int = 6, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.hebb_proj = nn.Sequential(
            nn.Linear(homeo_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        ).to(self.device)

        nn.init.zeros_(self.hebb_proj[0].weight)
        nn.init.zeros_(self.hebb_proj[0].bias)
        nn.init.zeros_(self.hebb_proj[2].weight)
        nn.init.zeros_(self.hebb_proj[2].bias)

    def forward(self, memory_state: torch.Tensor, u_t: torch.Tensor, k_vec: torch.Tensor, v_vec: torch.Tensor) -> torch.Tensor:
        if u_t.dim() == 2:
            u_curr = u_t
        else:
            u_curr = u_t.mean(dim=1) if u_t.dim() == 3 else u_t

        hebb_gain = 0.10 * self.hebb_proj(u_curr).unsqueeze(-1) # [batch, 1, 1]
        hebb_update = torch.matmul(k_vec.transpose(-1, -2), v_vec)
        return memory_state + hebb_gain * hebb_update


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


def run_experiment_218():
    logger.info("=" * 80)
    logger.info("STARTING EXP-218: ALLOSTATICALLY-GATED FAST-WEIGHT HEBBIAN SSD MODULATION")
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

    # 2. Proposed Evaluation (EXP-218)
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain

    hebb_mod = AllostaticFastWeightHebbianSSD(homeo_dim=6, device_str=hw.device_str).to(hw.device)

    # Verify Zero-Delta Identity at Birth (t_0)
    logger.info("Verifying KEP Principle 15 (Zero-Delta Identity at Birth t_0)...")
    dummy_mem = torch.randn(2, 64, 64, device=hw.device)
    dummy_u   = torch.randn(2, 6, device=hw.device)
    dummy_k   = torch.randn(2, 16, 64, device=hw.device)
    dummy_v   = torch.randn(2, 16, 64, device=hw.device)

    with torch.no_grad():
        mod_mem = hebb_mod(dummy_mem, dummy_u, dummy_k, dummy_v)

    delta_mem = torch.abs(dummy_mem - mod_mem).max().item()
    logger.info(f"Zero-Delta Verification: Max |dummy_mem - mod_mem| = {delta_mem:.8f}")

    assert delta_mem < 1e-5, f"KEP Principle 15 Violation: Delta mem = {delta_mem}"
    logger.info("🟢 KEP Principle 15 Zero-Delta Identity VERIFIED AT Step t_0!")

    brain_prop.hebb_ssd_mod = hebb_mod

    logger.info("Running Proposed Fast-Weight Hebbian Evaluation...")
    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    loss_delta = b_results['final_loss'] - p_results['final_loss']
    fe_delta = b_results['final_fe'] - p_results['final_fe']

    logger.info("=" * 80)
    logger.info("📊 === EXP-218 TELEMETRY REPORT ===")
    logger.info(f"  - Baseline Final Loss : {b_results['final_loss']:.4f} nats | Throughput: {b_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Proposed Final Loss : {p_results['final_loss']:.4f} nats | Throughput: {p_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Loss Delta (B - P)  : {loss_delta:.4f} nats")
    logger.info(f"  - Free Energy Delta   : {fe_delta:.6f}")

    verdict = "POSITIVE" if (loss_delta >= 0.08 or (loss_delta >= -0.01 and p_results['tok_per_sec'] >= 1.10 * b_results['tok_per_sec'])) else "NEUTRAL"

    metrics = {
        "baseline_loss": float(b_results['final_loss']),
        "final_loss": float(p_results['final_loss']),
        "loss_delta": float(loss_delta),
        "tok_per_sec": float(p_results['tok_per_sec']),
        "verdict": verdict
    }
    print(f"KEY_METRICS_JSON: {json.dumps(metrics)}")
    return metrics


if __name__ == "__main__":
    run_experiment_218()
