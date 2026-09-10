# experiments/exp_215_allostatic_locus_coeruleus_phasic_gain.py
"""
===============================================================================
EXP-215: Allostatically-Gated Locus Coeruleus (LC-NE) Multi-Signal Phasic Gain Controller
Grounding: KEP Principle 2 (Living AGI & Biological Realism - NON-NEGOTIABLE),
           KEP Principle 14 (Axiom of Allostatic Dynamic Forces — No Static Constants),
           KEP Principle 15 (Net2Net Smooth Grafting: Strict Identity at Birth t_0).
===============================================================================
Hypothesis:
In `karyon_agent.py` lines 341-380, `LocusCoeruleusGainController` computes phasic noradrenergic gain
strictly as a 1D function of scalar Noradrenaline (NA_t):
  z_score = (NA_t - mu_NA) / (sigma_NA + eps)
  phasic_gain = sigmoid(gain_scale * z_score + gain_bias)

However, biophysical neurobiology (Aston-Jones & Cohen 2005; Yu & Dayan 2005; Dayan & Angela 2005)
proves that Locus Coeruleus firing mode switches between tonic and phasic states depending not only
on noradrenergic surprise (NA_t), but on the wider multi-signal allostatic context:
  - Energy (E_t): Low metabolic energy suppresses high-frequency phasic LC bursts to conserve ATP.
  - Stability (S_t): High homeostasis stability stabilizes baseline tonic firing.
  - Dopamine (DA_t): Dopaminergic reward signals cross-inhibit excessive noradrenergic arousal.

Replacing the single-factor 1D LC gain controller with an Allostatically-Gated Multi-Signal
Locus Coeruleus Phasic Gain Controller (LC-Gain v2) that:
  - Preserves exact 1D noradrenergic z-score gain as the baseline signal:
      gain_base = sigmoid(gain_scale * z_score + gain_bias)
  - Modulates gain via a 6D allostatic net initialized with zero weights (KEP Principle 15):
      gain_allostatic = gain_base * (1.0 + 0.20 * tanh(W_allo @ u_t + b_allo))
      (W_allo = 0, b_allo = 0 at birth t_0 -> gain_allostatic == gain_base, 100% Zero-Delta Identity)
will dynamically balance arousal during high uncertainty and conserve energy under metabolic stress,
driving Loss Delta >= 0.08 while preserving 100% numerical stability.
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
logger = logging.getLogger("EXP-215")


class AllostaticLocusCoeruleusGainController(nn.Module):
    """
    Allostatically-Gated Multi-Signal Locus Coeruleus Phasic Gain Controller (LC-Gain v2).
    Modulates LC phasic arousal using the full 6D homeostatic state u_t with zero-initialized Net2Net grafting.
    """
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device else 'cpu')
        
        self.gain_scale = nn.Parameter(torch.tensor(4.0, device=self.device))
        self.gain_bias = nn.Parameter(torch.tensor(0.0, device=self.device))

        self.register_buffer("na_running_mean", torch.tensor(0.10, device=self.device))
        self.register_buffer("na_running_var", torch.tensor(0.01, device=self.device))
        self.register_buffer("momentum", torch.tensor(0.05, device=self.device))

        # Allostatic 6D Modulator Net initialized to ZERO (KEP Principle 15 Compliant)
        self.allo_modulator = nn.Sequential(
            nn.Linear(6, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        ).to(self.device)

        # Initialize linear layers to zero to guarantee zero-delta function identity at birth t_0
        nn.init.zeros_(self.allo_modulator[0].weight)
        nn.init.zeros_(self.allo_modulator[0].bias)
        nn.init.zeros_(self.allo_modulator[2].weight)
        nn.init.zeros_(self.allo_modulator[2].bias)

    def forward(self, na_t: torch.Tensor, u_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes continuous neural gain in (0, 1) based on NA surprise and 6D allostatic state u_t.
        """
        if torch.isnan(self.na_running_mean) or torch.isinf(self.na_running_mean):
            self.na_running_mean.fill_(0.10)
        if torch.isnan(self.na_running_var) or torch.isinf(self.na_running_var) or self.na_running_var < 1e-5:
            self.na_running_var.fill_(0.01)

        if self.training:
            with torch.no_grad():
                batch_mean = na_t.mean()
                batch_var = na_t.var(unbiased=False) if na_t.numel() > 1 else torch.tensor(1e-4, device=self.device)
                self.na_running_mean.copy_((1.0 - self.momentum) * self.na_running_mean + self.momentum * batch_mean)
                self.na_running_var.copy_((1.0 - self.momentum) * self.na_running_var + self.momentum * batch_var)

        sigma_na = torch.sqrt(torch.clamp(self.na_running_var, min=1e-5))
        z_score = (na_t - self.na_running_mean) / (sigma_na + 1e-5)
        base_gain = torch.sigmoid(self.gain_scale * z_score + self.gain_bias)

        if u_t is not None:
            if u_t.dim() == 1:
                u_in = u_t.unsqueeze(0)
            else:
                u_in = u_t
            if u_in.size(0) != base_gain.size(0):
                u_in = u_in[0:1].expand(base_gain.size(0), -1)

            allo_factor = 1.0 + 0.20 * self.allo_modulator(u_in)
            phasic_gain = torch.clamp(base_gain * allo_factor, 0.0, 1.0)
        else:
            phasic_gain = base_gain

        return phasic_gain


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
    logger.info("🔬 [STARTING EXP-215: ALLOSTIC LOCUS COERULEUS PHASIC GAIN BENCHMARK]")
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

    # 2. Proposed Evaluation (EXP-215 Allostatic LC Phasic Gain Controller)
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain

    allo_lc = AllostaticLocusCoeruleusGainController(device=hw.device_str).to(hw.device)
    # Copy baseline running buffers and gain params
    with torch.no_grad():
        allo_lc.gain_scale.copy_(brain_prop.lc_gain.gain_scale)
        allo_lc.gain_bias.copy_(brain_prop.lc_gain.gain_bias)
        allo_lc.na_running_mean.copy_(brain_prop.lc_gain.na_running_mean)
        allo_lc.na_running_var.copy_(brain_prop.lc_gain.na_running_var)
        allo_lc.momentum.copy_(brain_prop.lc_gain.momentum)

    brain_prop.lc_gain = allo_lc

    logger.info("Running Proposed Allostatic Locus Coeruleus Phasic Gain Evaluation...")
    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    loss_delta = b_results['final_loss'] - p_results['final_loss']
    fe_delta = b_results['final_fe'] - p_results['final_fe']

    logger.info("=" * 80)
    logger.info("📊 === EXP-215 TELEMETRY REPORT ===")
    logger.info(f"  - Baseline Final Loss : {b_results['final_loss']:.4f} nats | Throughput: {b_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Proposed Final Loss : {p_results['final_loss']:.4f} nats | Throughput: {p_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Loss Delta (B - P)  : {loss_delta:.4f} nats")
    logger.info(f"  - Free Energy Delta   : {fe_delta:.6f}")
    logger.info(f"  - Baseline Duration   : {b_results['elapsed']:.3f} s")
    logger.info(f"  - Proposed Duration   : {p_results['elapsed']:.3f} s")

    verdict = "POSITIVE" if (loss_delta >= 0.08 or (loss_delta >= 0.02 and p_results['tok_per_sec'] >= 0.90 * b_results['tok_per_sec'])) else "NEUTRAL"

    results = {
        "exp_id": "EXP-215",
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

    with open("experiments/exp_215_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"🏆 Final Verdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    return results


if __name__ == "__main__":
    run_benchmark()
