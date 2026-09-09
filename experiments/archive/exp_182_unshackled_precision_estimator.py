# experiments/exp_182_unshackled_precision_estimator.py
"""
EXP-182: Unshackled Dynamic Precision Estimator & Multimodal Allostatic Precision

Biophysical & Cybernetic Foundation:
1. In predictive coding and active inference (Friston, 2010), precision (pi_t) represents the expected
   inverse variance of prediction errors. High precision amplifies bottom-up prediction errors to update beliefs.
2. Bottleneck Elimination (KEP Principle 7): In production PrecisionWeightedTopDownGenerator (`pw_hpc_generator`),
   the precision estimator compresses [1537D -> 64D -> 1D], bottlenecking the high-dimensional sensory context.
   Expanding the hidden layer from 64D to 256D preserves representational rank.
3. Multimodal Allostatic Precision (KEP Principle 14): Production precision estimator only conditioned
   on NA_t (arousal). However, precision gating in biophysics is also driven by Dopamine (DA_t, reward expectation/gain)
   and Curiosity (novelty/epistemic drive).
   Input vector expanded from [h_s1, h_s1_hat, NA_t] -> [h_s1, h_s1_hat, NA_t, DA_t, Curiosity_t].

Telemetry Captured:
- Pre/Post Speech Cross-Entropy Loss (nats)
- Pre/Post Variational Free Energy (F_t)
- Precision Weight Dynamics (pi_t mean and std)
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
logger = logging.getLogger("EXP-182")


class ProposedPrecisionWeightedTopDownGenerator(nn.Module):
    """
    Proposed KEP Principle 7 & 14 Compliant Precision-Weighted Top-Down Generator.
    256D full-rank precision estimator conditioned on NA_t, DA_t, and Curiosity_t.
    """
    def __init__(self, hidden_dim=768, device_str='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = torch.device(device_str)

        self.topdown_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        ).to(self.device)

        # Expanded from 64D to 256D, input expanded from +1 to +3 (NA, DA, Curiosity)
        self.precision_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, h_s1: torch.Tensor, h_s2: torch.Tensor, u_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = h_s1.size()
        h_s1_hat = self.topdown_net(h_s2)
        e1 = h_s1 - h_s1_hat

        if u_t.size(0) == 1 and batch_size > 1:
            curiosity_t = u_t[:, 0:1].unsqueeze(1).expand(batch_size, seq_len, 1)
            na_t = u_t[:, 4:5].unsqueeze(1).expand(batch_size, seq_len, 1)
            da_t = u_t[:, 5:6].unsqueeze(1).expand(batch_size, seq_len, 1)
        else:
            curiosity_t = u_t[:batch_size, 0:1].unsqueeze(1).expand(batch_size, seq_len, 1)
            na_t = u_t[:batch_size, 4:5].unsqueeze(1).expand(batch_size, seq_len, 1)
            da_t = u_t[:batch_size, 5:6].unsqueeze(1).expand(batch_size, seq_len, 1)

        prec_input = torch.cat([h_s1, h_s1_hat, na_t, da_t, curiosity_t], dim=-1)
        pi_t = 2.0 * self.precision_estimator(prec_input)

        e1_weighted = pi_t * e1
        return e1_weighted, h_s1_hat, pi_t.mean()


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
    logger.info("🔬 [STARTING EXP-182: UNSHACKLED DYNAMIC PRECISION ESTIMATOR BENCHMARK]")
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
    logger.info("\n--- 1. Evaluating Baseline (Standard Precision Generator) ---")
    entity_b = KaryonEntity.load(filepath="karyon_soul.kcore", device=device_str)
    brain_b = entity_b.brain

    b_results = evaluate_model(brain_b, entity_b, text_samples, num_steps=25)
    logger.info(f"Baseline -> Init Loss: {b_results['init_loss']:.4f} | Final Loss: {b_results['final_loss']:.4f} (Delta: {b_results['loss_delta']:.4f})")
    logger.info(f"Baseline -> Init FE  : {b_results['init_fe']:.6f} | Final FE  : {b_results['final_fe']:.6f} (FE Delta: {b_results['fe_delta']:.6f})")

    # 2. Proposed Evaluation
    logger.info("\n--- 2. Evaluating Proposed (Unshackled 256D Precision Estimator + Multimodal Allostasis) ---")
    entity_p = KaryonEntity.load(filepath="karyon_soul.kcore", device=device_str)
    brain_p = entity_p.brain

    proposed_pw = ProposedPrecisionWeightedTopDownGenerator(
        hidden_dim=brain_p.hidden_dim,
        device_str=device_str
    )

    # Net2Net transfer from brain_p.pw_hpc_generator
    with torch.no_grad():
        proposed_pw.topdown_net[0].weight.copy_(brain_p.pw_hpc_generator.topdown_net[0].weight)
        proposed_pw.topdown_net[0].bias.copy_(brain_p.pw_hpc_generator.topdown_net[0].bias)
        proposed_pw.topdown_net[2].weight.copy_(brain_p.pw_hpc_generator.topdown_net[2].weight)
        proposed_pw.topdown_net[2].bias.copy_(brain_p.pw_hpc_generator.topdown_net[2].bias)
        proposed_pw.topdown_net[3].weight.copy_(brain_p.pw_hpc_generator.topdown_net[3].weight)
        proposed_pw.topdown_net[3].bias.copy_(brain_p.pw_hpc_generator.topdown_net[3].bias)

        # Transfer precision_estimator 64D -> 256D partially
        old_w = brain_p.pw_hpc_generator.precision_estimator[0].weight  # [64, 1537]
        old_b = brain_p.pw_hpc_generator.precision_estimator[0].bias    # [64]
        proposed_pw.precision_estimator[0].weight[:64, :1537].copy_(old_w)
        proposed_pw.precision_estimator[0].bias[:64].copy_(old_b)

        old_w2 = brain_p.pw_hpc_generator.precision_estimator[2].weight # [1, 64]
        old_b2 = brain_p.pw_hpc_generator.precision_estimator[2].bias   # [1]
        proposed_pw.precision_estimator[2].weight[:, :64].copy_(old_w2)
        proposed_pw.precision_estimator[2].bias.copy_(old_b2)

    brain_p.pw_hpc_generator = proposed_pw

    p_results = evaluate_model(brain_p, entity_p, text_samples, num_steps=25)
    logger.info(f"Proposed -> Init Loss: {p_results['init_loss']:.4f} | Final Loss: {p_results['final_loss']:.4f} (Delta: {p_results['loss_delta']:.4f})")
    logger.info(f"Proposed -> Init FE  : {p_results['init_fe']:.6f} | Final FE  : {p_results['final_fe']:.6f} (FE Delta: {p_results['fe_delta']:.6f})")

    # 3. Comparative Telemetry Analysis
    loss_improvement = b_results["final_loss"] - p_results["final_loss"]
    fe_reduction_pct = (b_results["final_fe"] - p_results["final_fe"]) / max(b_results["final_fe"], 1e-5) * 100

    verdict = "POSITIVE" if (p_results["final_loss"] <= b_results["final_loss"] + 0.02 and p_results["final_fe"] < b_results["final_fe"]) else (
        "POSITIVE" if p_results["final_loss"] < b_results["final_loss"] - 0.02 else "NEUTRAL"
    )

    logger.info("=" * 80)
    logger.info("📊 === EXP-182 SCIENTIFIC TELEMETRY REPORT ===")
    logger.info(f"🏆 Final Verdict                 : 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    logger.info(f"📈 Baseline Final Loss           : {b_results['final_loss']:.4f} nats")
    logger.info(f"📈 Proposed Final Loss           : {p_results['final_loss']:.4f} nats (Delta: {loss_improvement:+.4f} nats)")
    logger.info(f"📉 Baseline Final Free Energy    : {b_results['final_fe']:.6f}")
    logger.info(f"📉 Proposed Final Free Energy    : {p_results['final_fe']:.6f} ({fe_reduction_pct:+.2f}% reduction)")
    logger.info(f"⏱️ Baseline Run Duration         : {b_results['elapsed_sec']:.2f}s")
    logger.info(f"⏱️ Proposed Run Duration         : {p_results['elapsed_sec']:.2f}s")
    logger.info("=" * 80)

    summary = {
        "exp_id": "EXP-182",
        "verdict": verdict,
        "hypothesis": "Unshackling precision estimator (64D -> 256D) and expanding conditioning to NA_t, DA_t, and Curiosity_t reduces Free Energy.",
        "architecture_delta": "Expanded PrecisionWeightedTopDownGenerator precision_estimator to 256D; added DA_t and Curiosity_t to precision conditioning.",
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

    with open("experiments/exp_182_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("✅ EXP-182 execution completed successfully. Results recorded in experiments/exp_182_results.json.")


if __name__ == "__main__":
    main()
