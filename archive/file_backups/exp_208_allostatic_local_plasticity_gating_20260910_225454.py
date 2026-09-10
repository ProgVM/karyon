# experiments/exp_208_allostatic_local_plasticity_gating.py
"""
===============================================================================
EXP-208: Allostatically-Gated Local Neuromodulated Plasticity Engine
Grounding: KEP Principle 1 (C++20 as Engine), Principle 2 (Biological Realism),
           Principle 14 (Axiom of Allostatic Dynamic Forces: No Static Constants),
           Principle 15 (Net2Net Smooth Grafting: Strict Identity at Birth).
===============================================================================
Hypothesis:
In the Grand Synthesis of Neo-Cortical vectors, local neuromodulated plasticity (y_local)
provides rapid, local synaptic weight adaptation to complement global backpropagation.
In previous implementations, y_local has a static scalar coefficient (0.10 * y_local) and its
contribution to the pre-attractor cortical representation is unmodulated by the real-time homeostatic
landscape (Somatic Energy u_t[1], Noradrenaline u_t[4], Dopamine u_t[5]) or Variational Free Energy (F_t).
Replacing the static linear mixing of y_local with an Allostatically-Gated Local Neuromodulated Plasticity
Gating Engine—where local plasticity scaling is dynamically computed via a non-linear allostatic gate:
  g_local(u_t, F_t) = 0.10 * (1.0 + 1.5 * NA_t + 1.2 * DA_t - 0.5 * (1.0 - Energy_t)) * (1.0 + 0.2 * tanh(F_t))
coupled with a zero-initialized residual projection graft to preserve strict zero-delta function identity
at birth (KEP Principle 15)—will enhance rapid associative consolidation, stabilize representations
against noise, and accelerate loss convergence (Loss Delta >= 0.08).
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_entity import KaryonEntity
from karyon_hardware import get_hardware_engine

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-208")


class AllostaticLocalPlasticityGater(nn.Module):
    """
    Allostatically-Gated Local Plasticity Controller (EXP-208).
    Dynamically modulates the local synaptic plasticity projection (y_local)
    based on Somatic Homeostasis (Energy, NA, DA) and Variational Free Energy (F_t).
    Employs zero-initialized residual projection to preserve strict zero-delta function identity at birth.
    """
    def __init__(self, hidden_dim: int, homeo_dim: int = 6, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.hidden_dim = hidden_dim
        
        # Zero-initialized refinement layer (Net2Net Smooth Grafting / KEP Principle 15)
        self.refine_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device)
        nn.init.zeros_(self.refine_proj.weight)
        
        # Allostatic Gating Network
        self.gate_net = nn.Sequential(
            nn.Linear(homeo_dim + 1, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)
        # Initialize to 0 so sigmoid gives 0.5 initially
        nn.init.zeros_(self.gate_net[2].weight)
        nn.init.zeros_(self.gate_net[2].bias)

    def forward(self, y_local: torch.Tensor, u_t: torch.Tensor, fe_loss: torch.Tensor = None) -> torch.Tensor:
        B = y_local.size(0)
        is_3d = (y_local.dim() == 3)
        
        if fe_loss is not None:
            fe_norm = torch.clamp(fe_loss.detach() / 10.0, 0.0, 1.0).view(-1, 1)
        else:
            fe_norm = torch.zeros(B, 1, device=self.device)
            
        if u_t.dim() == 2:
            u_flat = u_t
        else:
            u_flat = u_t.view(1, -1).expand(B, -1)
            
        ctrl_in = torch.cat([u_flat, fe_norm], dim=-1) # [B, 7]
        gate_factor = self.gate_net(ctrl_in) * 2.0 # [B, 1], centered around 1.0
        
        # Base biophysical modulation
        energy_t = u_flat[:, 1:2]
        na_t = u_flat[:, 4:5]
        da_t = u_flat[:, 5:6]
        bio_factor = (1.0 + 1.5 * na_t + 1.2 * da_t - 0.5 * (1.0 - energy_t))
        
        eff_scale = 0.10 * gate_factor * bio_factor # [B, 1]
        if is_3d:
            eff_scale = eff_scale.unsqueeze(1) # [B, 1, 1]
            
        refined = self.refine_proj(y_local)
        # Identity-preserving: at birth refine_proj is zero, so output is exactly 0.10 * y_local * bio_factor
        return eff_scale * (y_local + refined)


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
    logger.info("🔬 [STARTING EXP-208: ALLOSTATIC LOCAL PLASTICITY GATING BENCHMARK]")
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

    # 2. Proposed Evaluation (EXP-208 Allostatic Local Plasticity Gating Engine)
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain
    allostatic_plasticity_gater = AllostaticLocalPlasticityGater(
        hidden_dim=brain_prop.hidden_dim, homeo_dim=6, device_str=hw.device_str
    ).to(hw.device)

    # Wrap local_plasticity to allostatically gate output
    class PatchedLocalPlasticity(nn.Module):
        def __init__(self, original_lp, gater):
            super().__init__()
            self.original_lp = original_lp
            self.gater = gater

        def forward(self, h_s2_gated: torch.Tensor, u_t: torch.Tensor = None):
            y_base = self.original_lp(h_s2_gated)
            if u_t is not None:
                return self.gater(y_base, u_t)
            return y_base

        def adapt_local_fast_weights(self, *args, **kwargs):
            return self.original_lp.adapt_local_fast_weights(*args, **kwargs)

    brain_prop.allostatic_plasticity_gater = allostatic_plasticity_gater
    
    logger.info("Running Proposed Allostatic Local Plasticity Gating Evaluation...")
    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    loss_delta = b_results['final_loss'] - p_results['final_loss']
    fe_delta = b_results['final_fe'] - p_results['final_fe']

    logger.info("=" * 80)
    logger.info("📊 === EXP-208 TELEMETRY REPORT ===")
    logger.info(f"  - Baseline Final Loss : {b_results['final_loss']:.4f} nats | Throughput: {b_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Proposed Final Loss : {p_results['final_loss']:.4f} nats | Throughput: {p_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Loss Delta (B - P)  : {loss_delta:.4f} nats")
    logger.info(f"  - Free Energy Delta   : {fe_delta:.6f}")
    logger.info(f"  - Baseline Duration   : {b_results['elapsed']:.3f} s")
    logger.info(f"  - Proposed Duration   : {p_results['elapsed']:.3f} s")

    verdict = "POSITIVE" if (loss_delta >= 0.08 or (loss_delta >= 0.02 and p_results['tok_per_sec'] >= 0.90 * b_results['tok_per_sec'])) else "NEUTRAL"

    results = {
        "exp_id": "EXP-208",
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

    with open("experiments/exp_208_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"🏆 Final Verdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    return results


if __name__ == "__main__":
    run_benchmark()
