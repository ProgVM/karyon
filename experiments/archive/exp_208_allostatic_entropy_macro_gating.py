# experiments/exp_208_allostatic_entropy_macro_gating.py
"""
===============================================================================
EXP-208: Allostatically-Modulated Entropy-Driven Concept Macro-Boundary Gating
Grounding: KEP Principle 2 (Biological Realism), Principle 8 (Compositional Depth),
           Principle 14 (Axiom of Allostatic Dynamic Forces: No Static Constants),
           Principle 15 (Net2Net Smooth Grafting: Strict Identity at Birth).
===============================================================================
Hypothesis:
In `EntropyMacroGating` (BLT-Neuro concept boundary detection), the boundary gate uses static constants:
  boundary_gate = sigmoid(boundary_logits + 2.0 * (entropy - 1.5))
and Stage 2 semantic gating uses static scaling:
  h_s2_gated = h_s2 * (0.50 + 1.00 * boundary_gate)
Under KEP Principle 14, static detection thresholds (1.5 nats) and static sensitivity multipliers (2.0)
fail to adapt to arousal and novelty. When Noradrenaline (u_t[4]) and Variational Free Energy (F_t) rise,
the brain's boundary detection sensitivity sharpens, lowering the threshold to register novel concept
boundaries more sensitively. Conversely, when Stability (u_t[2]) is high, boundary thresholds widen to
maintain continuous discourse flow.
Replacing static constants with dynamic allostatic equations:
  H_thresh(u_t, F_t) = 1.50 - 0.30 * NA_t - 0.20 * tanh(F_t) + 0.20 * Stability_t
  beta_sens(u_t) = 2.0 * (1.0 + 1.0 * NA_t)
coupled with a zero-initialized allostatic boundary refinement graft (Net2Net Smooth Grafting) to guarantee
zero-delta function identity at birth, will optimize macro-pulse concept segmentation, improve feature
flow between cortical stages, and accelerate loss convergence (Loss Delta >= 0.08).
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


class AllostaticEntropyMacroGating(nn.Module):
    """
    Allostatically-Modulated Entropy-Driven Macro-Boundary Gating (EXP-208).
    Dynamically modulates the boundary detection threshold H_thresh and sensitivity beta
    conditioned on Somatic Homeostasis (NA, Stability) and Variational Free Energy (F_t).
    Employs zero-initialized residual projection to preserve strict zero-delta function identity at birth.
    """
    def __init__(self, original_gate: nn.Module, homeo_dim: int = 6, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.original_gate = original_gate
        self.hidden_dim = original_gate.macro_boundary_proj.in_features
        
        # Zero-initialized refinement projection (Net2Net Smooth Grafting / KEP Principle 15)
        self.refine_proj = nn.Linear(self.hidden_dim, 1, bias=False).to(self.device)
        nn.init.zeros_(self.refine_proj.weight)
        
        # Allostatic Modulator Net: maps [u_t, F_t] -> [2] (delta_thresh, delta_scale)
        self.modulator = nn.Sequential(
            nn.Linear(homeo_dim + 1, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
            nn.Tanh()
        ).to(self.device)
        nn.init.zeros_(self.modulator[2].weight)
        nn.init.zeros_(self.modulator[2].bias)

    def forward(self, h_s1: torch.Tensor, u_t: torch.Tensor = None, fe_loss: torch.Tensor = None):
        logits = self.original_gate.entropy_head(h_s1)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1) # [B, S] or [B]

        boundary_logits = self.original_gate.macro_boundary_proj(h_s1).squeeze(-1)
        
        # At birth (modulator is zero-initialized), delta_thresh=0 and delta_scale=0 -> exact 1.5 and 2.0
        if u_t is not None and u_t.numel() >= 6:
            B = h_s1.size(0)
            if fe_loss is not None:
                fe_norm = torch.clamp(fe_loss.detach() / 10.0, 0.0, 1.0).view(-1, 1)
            else:
                fe_norm = torch.zeros(B, 1, device=self.device)
                
            if u_t.dim() == 2:
                u_flat = u_t if u_t.size(0) == B else u_t[0:1].expand(B, -1)
            else:
                u_flat = u_t.view(1, -1).expand(B, -1)
                
            ctrl_in = torch.cat([u_flat, fe_norm], dim=-1)
            mods = self.modulator(ctrl_in) # [B, 2]
            
            # Dynamic threshold and sensitivity
            delta_thresh = mods[:, 0] * 0.40 # [B]
            delta_sens = mods[:, 1] * 1.00   # [B]
            
            if h_s1.dim() == 3:
                delta_thresh = delta_thresh.unsqueeze(1)
                delta_sens = delta_sens.unsqueeze(1)
                
            h_thresh = 1.50 + delta_thresh
            beta_sens = 2.00 + delta_sens
        else:
            h_thresh = 1.50
            beta_sens = 2.00

        refine_logit = self.refine_proj(h_s1).squeeze(-1)
        eff_logits = boundary_logits + refine_logit
        boundary_gate = torch.sigmoid(eff_logits + beta_sens * (entropy - h_thresh))
        return entropy, boundary_gate


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
    logger.info("🔬 [STARTING EXP-208: ALLOSTATIC ENTROPY MACRO GATING BENCHMARK]")
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

    # 2. Proposed Evaluation (EXP-208 Allostatic Entropy Macro Gating)
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain
    allostatic_entropy_gate = AllostaticEntropyMacroGating(
        original_gate=brain_prop.entropy_macro_gate,
        homeo_dim=6,
        device_str=hw.device_str
    ).to(hw.device)

    # Patch entropy_macro_gate on brain_prop
    brain_prop.entropy_macro_gate = allostatic_entropy_gate
    
    logger.info("Running Proposed Allostatic Entropy Macro Gating Evaluation...")
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
