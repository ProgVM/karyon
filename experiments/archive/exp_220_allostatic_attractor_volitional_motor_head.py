# experiments/exp_220_allostatic_attractor_volitional_motor_head.py
"""
===============================================================================
EXP-220: Net2Net Allostatically-Modulated Volitional Motor Precision Readout
Grounding: KEP Principle 2 (Living AGI & Biological Realism - NON-NEGOTIABLE),
           Principle 14 (Axiom of Allostatic Dynamic Forces — No Static Constants),
           Principle 15 (Net2Net Smooth Grafting: Strict Zero-Delta Identity at Birth t_0).
===============================================================================
Hypothesis:
In `VolitionalActiveInferenceMotorHead`, the motor readout gain relies on a single 1D
dopamine factor: `motor_gain = (1.0 + 1.0 * DA_t)`. Replacing this with a unified 6D
allostatic gain modulation network:
  gain_allostatic = (1.0 + 1.0 * DA_t) * (1.0 + 0.20 * tanh(W_allo * u_t + b_allo))
with 100% Net2Net zero-initialization on W_allo and b_allo at birth t_0 guarantees
strict zero-delta function identity. This dynamic 6D homeostatic modulation allows the
motor readout to jointly account for curiosity, energy, stability, health, and noradrenaline,
refining generation precision and driving Loss Delta >= 0.08 while maintaining 100% stability.
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
logger = logging.getLogger("EXP-220")


class AllostaticVolitionalMotorGain(nn.Module):
    """
    Allostatically-Gated Volitional Motor Gain with 100% Net2Net Zero-Delta Identity.
    """
    def __init__(self, homeo_dim: int = 6, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.allo_proj = nn.Sequential(
            nn.Linear(homeo_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        ).to(self.device)

        nn.init.zeros_(self.allo_proj[0].weight)
        nn.init.zeros_(self.allo_proj[0].bias)
        nn.init.zeros_(self.allo_proj[2].weight)
        nn.init.zeros_(self.allo_proj[2].bias)

    def forward(self, da_level: torch.Tensor, u_t_exp: torch.Tensor) -> torch.Tensor:
        base_gain = (1.0 + 1.0 * da_level)
        allo_mod = 1.0 + 0.20 * self.allo_proj(u_t_exp)
        return base_gain * allo_mod


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


def run_experiment_220():
    logger.info("=" * 80)
    logger.info("STARTING EXP-220: NET2NET ALLOSTATIC VOLITIONAL MOTOR READOUT")
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

    # 2. Proposed Evaluation (EXP-220)
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain

    allo_motor_gain = AllostaticVolitionalMotorGain(homeo_dim=6, device_str=hw.device_str).to(hw.device)

    # Verify Zero-Delta Identity at Birth (t_0)
    logger.info("Verifying KEP Principle 15 (Zero-Delta Identity at Birth t_0)...")
    dummy_da = torch.randn(2, 1, device=hw.device)
    dummy_u  = torch.randn(2, 6, device=hw.device)

    with torch.no_grad():
        out_base = (1.0 + 1.0 * dummy_da)
        out_prop = allo_motor_gain(dummy_da, dummy_u)

    delta_out = torch.abs(out_base - out_prop).max().item()
    logger.info(f"Zero-Delta Verification: Max |out_base - out_prop| = {delta_out:.8f}")

    assert delta_out < 1e-5, f"KEP Principle 15 Violation: Delta out = {delta_out}"
    logger.info("🟢 KEP Principle 15 Zero-Delta Identity VERIFIED AT Step t_0!")

    brain_prop.volitional_head.allo_gain_mod = allo_motor_gain
    orig_compute_logits = brain_prop.volitional_head.compute_volitional_logits

    def patched_compute_volitional_logits(h_relaxed: torch.Tensor, u_t: torch.Tensor, byte_embed_weights: torch.Tensor) -> torch.Tensor:
        total_tokens = h_relaxed.size(0)
        if u_t.dim() == 2 and u_t.size(0) != total_tokens:
            batch_size = u_t.size(0)
            seq_len = total_tokens // batch_size
            u_t_exp = u_t.unsqueeze(1).expand(batch_size, seq_len, 6).reshape(total_tokens, 6)
        else:
            u_t_exp = u_t

        da_level = u_t_exp[:, 5:6]
        motor_gain = brain_prop.volitional_head.allo_gain_mod(da_level, u_t_exp)

        # 1. Project relaxed state to sensory manifold
        h_proj = brain_prop.volitional_head.motor_text_proj(h_relaxed) # [S, D]
        
        # 2. Apply CPG Causal Motor Receptive Field
        h_proj_seq = h_proj.unsqueeze(0).transpose(1, 2) # [1, D, S]
        try:
            h_cpg_seq = brain_prop.volitional_head.cpg_motor[0](h_proj_seq)
        except RuntimeError:
            with torch.backends.cudnn.flags(enabled=False):
                h_cpg_seq = brain_prop.volitional_head.cpg_motor[0](h_proj_seq)
        h_cpg_seq = h_cpg_seq[:, :, :total_tokens]
        h_cpg = h_cpg_seq.transpose(1, 2).squeeze(0) # [S, D]
        h_cpg_out = brain_prop.volitional_head.cpg_motor[2](brain_prop.volitional_head.cpg_motor[1](h_cpg) + h_proj)

        # 3. Apply Dopaminergic/Allostatic Precision Gain
        h_proj_gain = h_cpg_out * motor_gain
        raw_logits = F.linear(h_proj_gain, byte_embed_weights)

        # 4. Unshackled Full-Rank EFE Manifold Evaluation
        v_emb_proj = brain_prop.volitional_head.efe_motor_proj(byte_embed_weights)
        u_t_proj = brain_prop.volitional_head.efe_homeo_proj(u_t_exp)
        u_t_expanded = u_t_proj.unsqueeze(1)
        efe_combined = v_emb_proj.unsqueeze(0) + u_t_expanded
        efe_scores = brain_prop.volitional_head.efe_evaluator(efe_combined).squeeze(-1)

        gamma_volition = 0.15 * (1.0 + u_t_exp[:, 0:1] + u_t_exp[:, 4:5] - 0.5 * (1.0 - u_t_exp[:, 1:2]))
        volitional_logits = raw_logits - gamma_volition * efe_scores
        return volitional_logits

    brain_prop.volitional_head.compute_volitional_logits = patched_compute_volitional_logits

    logger.info("Running Proposed Allostatic Volitional Motor Head Evaluation...")
    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    loss_delta = b_results['final_loss'] - p_results['final_loss']
    fe_delta = b_results['final_fe'] - p_results['final_fe']

    logger.info("=" * 80)
    logger.info("📊 === EXP-220 TELEMETRY REPORT ===")
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
    run_experiment_220()
