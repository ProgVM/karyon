# experiments/exp_222_allostatic_latent_precision_selector.py
"""
===============================================================================
EXP-222: Allostatically-Gated Latent Precision Selector into Active Inference World Model
Grounding: KEP Principle 2 (Living AGI & Biological Realism - NON-NEGOTIABLE),
           KEP Principle 14 (Axiom of Allostatic Dynamic Forces — No Static Constants),
           KEP Principle 15 (Net2Net Smooth Grafting: Strict Zero-Delta Identity at Birth t_0).
===============================================================================
Hypothesis:
In `LatentPredictor` (Active Inference World Model), replacing static KL divergence weighting
with an Allostatically-Gated Latent Precision Selector:
  beta_KL(u_t) = beta_base * (1.0 + tanh(W_allo_kl * u_t + b_allo_kl))
with Net2Net Zero-Initialization (W_allo_kl = 0, b_allo_kl = 0) at birth t_0 will dynamically
modulate latent precision based on somatic stress (NE, DA, Glu), yielding faster convergence
and lower overall variational Free Energy under continuous stream learning (Target Delta Loss >= 0.08).
===============================================================================
"""

import sys
import os
import time
import math
import torch
import torch.nn as nn
import logging

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_config import CoREConfig
from karyon_entity import KaryonEntity
from karyon_hardware import get_hardware_engine
from karyon_core import LatentPredictor

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-222")


class AllostaticallyGatedLatentPredictor(nn.Module):
    """
    Wrapper / Graft for LatentPredictor incorporating Allostatically-Gated Latent Precision Selection.
    Applies Net2Net Zero-Initialization:
      W_allo_kl = 0, b_allo_kl = 0
    guaranteeing beta_KL(u_t) = beta_base at birth t_0 (Zero-Delta Identity).
    """
    def __init__(self, base_latent_predictor, hidden_dim=512, unified_dim=256, allo_dim=6, beta_base=1.0, device_str="cpu"):
        super().__init__()
        self.base_predictor = base_latent_predictor
        self.beta_base = beta_base

        self.allo_kl_proj = nn.Linear(allo_dim, 1).to(device_str)
        nn.init.zeros_(self.allo_kl_proj.weight)
        nn.init.zeros_(self.allo_kl_proj.bias)

    def forward(self, h_fast_prev: torch.Tensor, h_slow_curr: torch.Tensor, w_t: torch.Tensor, u_t: torch.Tensor = None):
        w_pred, kl_div, free_energy_base, z_t = self.base_predictor(h_fast_prev, h_slow_curr, w_t)

        if u_t is not None:
            # Allostatic dynamic precision factor: 1.0 + tanh(W * u_t + b)
            # At t_0 (W=0, b=0), factor = 1.0 + 0.0 = 1.0 -> Zero-Delta Identity!
            precision_factor = 1.0 + torch.tanh(self.allo_kl_proj(u_t)) # [B, 1]
            if precision_factor.dim() > kl_div.dim():
                precision_factor = precision_factor.squeeze(-1)
            
            beta_effective = self.beta_base * precision_factor
            
            # Recalculate Free Energy with allostatically modulated KL weight
            rec_loss = torch.mean(torch.pow(w_t - w_pred, 2), dim=-1, keepdim=True)
            free_energy = beta_effective * kl_div + rec_loss
        else:
            free_energy = free_energy_base

        return w_pred, kl_div, free_energy, z_t


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
        "tok_per_sec": tok_per_sec
    }


def run_benchmark():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-222: ALLOSTATICALLY-GATED LATENT PRECISION SELECTOR BENCHMARK]")
    logger.info("=" * 80)

    hw = get_hardware_engine()
    logger.info(f"Hardware Acceleration Engine: {hw.device_str} (Device: {hw.device})")

    # Load corpus sample
    text_samples = [
        "The fundamental principle of Active Inference is that self-organizing systems minimize variational free energy.",
        "Karyon-CoRE operates at raw UTF-8 byte level V=258 with zero-loop parallel state-space duality.",
        "Ashby somatic homeostasis tracks curiosity, energy, stability, health, noradrenaline, and dopamine continuously.",
        "Mamba-2 GroupNorm head equalization bounds exponential state dynamics to maintain numeric stability.",
        "Theta-Gamma phase-amplitude coupling coordinates cortical micro-circuits for precise motor speech synthesis."
    ]

    # 1. Baseline Evaluation (Standard Model in karyon_soul.kcore)
    logger.info("📊 Evaluating Baseline World Model...")
    entity_base = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_base = entity_base.brain
    b_results = evaluate_model(brain_base, entity_base, text_samples, num_steps=25)

    loss_base = b_results["final_loss"]
    logger.info(f"Baseline -> Final Loss: {loss_base:.4f} | Initial FE: {b_results['initial_fe']:.4f} | Final FE: {b_results['final_fe']:.4f} | Speed: {b_results['tok_per_sec']:.1f} tok/s")

    # 2. Verify Net2Net Zero-Delta Identity at Birth t_0
    logger.info("Grafting Allostatically-Gated Latent Precision Selector into World Model...")
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain

    base_wm = brain_prop.world_model
    grafted_wm = AllostaticallyGatedLatentPredictor(
        base_wm,
        hidden_dim=brain_prop.hidden_dim,
        unified_dim=brain_prop.unified_dim,
        allo_dim=6,
        beta_base=1.0,
        device_str=hw.device_str
    ).to(hw.device)

    # Test zero-delta identity at t_0 using fixed random seed
    h_fast = torch.randn(1, brain_prop.hidden_dim, device=hw.device)
    h_slow = torch.randn(1, brain_prop.hidden_dim, device=hw.device)
    w_t = torch.randn(1, brain_prop.unified_dim, device=hw.device)
    u_t = torch.rand(1, 6, device=hw.device)

    torch.manual_seed(42)
    with torch.no_grad():
        w_p1, kl1, fe1, z1 = base_wm(h_fast, h_slow, w_t)
    
    torch.manual_seed(42)
    with torch.no_grad():
        w_p2, kl2, fe2, z2 = grafted_wm(h_fast, h_slow, w_t, u_t)

    delta_fe_birth = torch.abs(fe1 - fe2).max().item()
    logger.info(f"Net2Net Birth t_0 Max FE Delta: {delta_fe_birth:.8f}")
    assert delta_fe_birth < 1e-6, f"FATAL: Net2Net Zero-Delta Identity violated at birth t_0! Delta: {delta_fe_birth}"
    logger.info("✅ KEP Principle 15 (Net2Net Zero-Delta Identity at Birth) VERIFIED!")

    # 3. Attach Grafted World Model and Evaluate Proposed Adaptation
    brain_prop.world_model = grafted_wm
    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    final_grafted_loss = p_results["final_loss"]
    loss_delta = loss_base - final_grafted_loss
    fe_delta = b_results["final_fe"] - p_results["final_fe"]

    logger.info("=" * 80)
    logger.info(f"📊 [EXP-222 FINAL TELEMETRY SUMMARY]")
    logger.info(f"   - Baseline Loss    : {loss_base:.4f}")
    logger.info(f"   - Grafted Loss     : {final_grafted_loss:.4f}")
    logger.info(f"   - Loss Delta (Gain): {loss_delta:.4f}")
    logger.info(f"   - Free Energy Delta: {fe_delta:.4f}")
    logger.info(f"   - Throughput       : {p_results['tok_per_sec']:.1f} tok/s")
    logger.info("=" * 80)

    # 4. KEP Rule #2 Verdict Decision
    if loss_delta >= 0.08:
        verdict = "🟢 POSITIVE"
        logger.info(f"VERDICT: {verdict} (Loss Delta {loss_delta:.4f} >= 0.08 KEP Threshold)")
    elif abs(loss_delta) < 0.08:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"
        logger.info(f"VERDICT: {verdict} (Loss Delta {loss_delta:.4f} within noise margin)")
    else:
        verdict = "🔴 REJECTED"
        logger.info(f"VERDICT: {verdict} (Loss degraded by {loss_delta:.4f})")

    return {
        "exp_id": "EXP-222",
        "verdict": verdict,
        "loss_base": loss_base,
        "final_loss": final_grafted_loss,
        "loss_delta": loss_delta,
        "tok_per_sec": p_results["tok_per_sec"]
    }


if __name__ == "__main__":
    run_benchmark()
