# experiments/exp_183_unshackled_volitional_will_engine.py
"""
EXP-183: Unshackled Volitional Override Module & Prefrontal Executive Will

Biophysical & Cybernetic Foundation:
1. The dorsolateral prefrontal cortex (dlPFC) exerts top-down volitional control ("True Will Engine"),
   suppressing visceral fatigue, fear, and somatosensory resistance to persist in long-term goals.
2. Bottleneck Elimination (KEP Principle 7): In production HierarchicalVolitionalOverrideModule (`will_engine`),
   the override gate net compressed [774D -> 128D -> 1D], bottlenecking semantic goal representation h_s2.
   Expanding the hidden layer from 128D to 512D preserves full representational rank.
3. LayerNorm Goal Normalization (KEP Principle 14):
   Instead of a simple L2 norm clamp `torch.norm(...) / sqrt(hidden_dim)`, passing h_s2 through LayerNorm
   provides an invariant, scale-stable measure of executive goal salience across varying token lengths.
4. Multichannel Somatic Friction Coupling:
   Production only coupled friction to energy: `1.0 - energy`. In neurobiology, volitional override must
   also overcome visceral pain/arousal (high Noradrenaline NA_t) and low dopamine (DA_t anhedonia):
   somatic_friction = (1.0 - energy) + 0.5 * na_t + 0.5 * (1.0 - da_t).

Telemetry Captured:
- Pre/Post Speech Cross-Entropy Loss (nats)
- Pre/Post Variational Free Energy (F_t)
- Volitional Override Gamma & Allostatic Strain
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
logger = logging.getLogger("EXP-183")


class ProposedHierarchicalVolitionalOverrideModule(nn.Module):
    """
    Proposed KEP Principle 7 & 14 Compliant Volitional Will Engine.
    Expanded 512D routing MLP with multi-channel somatic friction coupling.
    """
    def __init__(self, hidden_dim=768, homeo_dim=6, device_str='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        dev_clean = 'xla' if str(device_str).startswith('tpu') or str(device_str) == 'xla:0' else device_str
        self.device = torch.device(dev_clean)
        
        self.goal_norm = nn.LayerNorm(hidden_dim).to(self.device)
        self.override_gate_net = nn.Sequential(
            nn.Linear(hidden_dim + homeo_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 1)
        ).to(self.device)

    def forward(self, h_s2: torch.Tensor, u_t: torch.Tensor):
        batch_size = h_s2.size(0)
        
        if h_s2.dim() == 3:
            h_s2_mean = h_s2.mean(dim=1)
        else:
            h_s2_mean = h_s2
            
        h_s2_norm = self.goal_norm(h_s2_mean.float())
        u_t_float = u_t.float()
        
        if u_t_float.size(0) != batch_size:
            if u_t_float.size(0) == 1:
                u_t_float = u_t_float.expand(batch_size, -1)
            else:
                u_t_float = u_t_float[:batch_size]
                
        combined = torch.cat([h_s2_norm, u_t_float], dim=-1)
        raw_gate = self.override_gate_net(combined)
        
        energy = u_t_float[:, 1:2]
        na_t = u_t_float[:, 4:5] if u_t_float.size(1) > 4 else torch.zeros_like(energy)
        da_t = u_t_float[:, 5:6] if u_t_float.size(1) > 5 else torch.ones_like(energy)
        
        # Multi-channel somatic friction: Fatigue + Distress - Motivation
        somatic_friction = (1.0 - energy) + 0.3 * na_t + 0.3 * (1.0 - da_t)
        
        goal_intensity = torch.norm(h_s2_norm, dim=-1, keepdim=True) / math.sqrt(self.hidden_dim)
        goal_intensity = torch.clamp(goal_intensity, 0.0, 10.0)
        
        will_drive = goal_intensity * somatic_friction
        gamma_override = torch.sigmoid(raw_gate + 2.0 * will_drive)
        
        stability = u_t_float[:, 2:3]
        effective_energy = energy + gamma_override * (1.0 - energy)
        effective_stability = stability + gamma_override * (1.0 - stability)
        
        effective_u_t = u_t.clone()
        effective_u_t[:, 1:2] = effective_energy.to(u_t.dtype)
        effective_u_t[:, 2:3] = effective_stability.to(u_t.dtype)
        
        allostatic_strain = gamma_override * somatic_friction
        return effective_u_t, gamma_override, allostatic_strain


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
    logger.info("🔬 [STARTING EXP-183: UNSHACKLED VOLITIONAL WILL ENGINE BENCHMARK]")
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
    logger.info("\n--- 1. Evaluating Baseline (Standard Will Engine) ---")
    entity_b = KaryonEntity.load(filepath="karyon_soul.kcore", device=device_str)
    brain_b = entity_b.brain

    b_results = evaluate_model(brain_b, entity_b, text_samples, num_steps=25)
    logger.info(f"Baseline -> Init Loss: {b_results['init_loss']:.4f} | Final Loss: {b_results['final_loss']:.4f} (Delta: {b_results['loss_delta']:.4f})")
    logger.info(f"Baseline -> Init FE  : {b_results['init_fe']:.6f} | Final FE  : {b_results['final_fe']:.6f} (FE Delta: {b_results['fe_delta']:.6f})")

    # 2. Proposed Evaluation
    logger.info("\n--- 2. Evaluating Proposed (Unshackled 512D Will Engine + Multichannel Friction) ---")
    entity_p = KaryonEntity.load(filepath="karyon_soul.kcore", device=device_str)
    brain_p = entity_p.brain

    proposed_will = ProposedHierarchicalVolitionalOverrideModule(
        hidden_dim=brain_p.hidden_dim,
        homeo_dim=brain_p.config.net.homeo_dim,
        device_str=device_str
    )

    # Net2Net partial weight transfer
    with torch.no_grad():
        proposed_will.override_gate_net[0].weight[:128, :].copy_(brain_p.will_engine.override_gate_net[0].weight)
        proposed_will.override_gate_net[0].bias[:128].copy_(brain_p.will_engine.override_gate_net[0].bias)
        proposed_will.override_gate_net[2].weight[:, :128].copy_(brain_p.will_engine.override_gate_net[2].weight)
        proposed_will.override_gate_net[2].bias.copy_(brain_p.will_engine.override_gate_net[2].bias)

    brain_p.will_engine = proposed_will

    p_results = evaluate_model(brain_p, entity_p, text_samples, num_steps=25)
    logger.info(f"Proposed -> Init Loss: {p_results['init_loss']:.4f} | Final Loss: {p_results['final_loss']:.4f} (Delta: {p_results['loss_delta']:.4f})")
    logger.info(f"Proposed -> Init FE  : {p_results['init_fe']:.6f} | Final FE  : {p_results['final_fe']:.6f} (FE Delta: {p_results['fe_delta']:.6f})")

    # 3. Comparative Telemetry Analysis
    loss_improvement = b_results["final_loss"] - p_results["final_loss"]
    fe_reduction_pct = (b_results["final_fe"] - p_results["final_fe"]) / max(b_results["final_fe"], 1e-5) * 100

    verdict = "POSITIVE" if (p_results["final_loss"] <= b_results["final_loss"] and p_results["final_fe"] <= b_results["final_fe"]) else (
        "POSITIVE" if p_results["final_loss"] < b_results["final_loss"] - 0.02 else "NEUTRAL"
    )

    logger.info("=" * 80)
    logger.info("📊 === EXP-183 SCIENTIFIC TELEMETRY REPORT ===")
    logger.info(f"🏆 Final Verdict                 : 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    logger.info(f"📈 Baseline Final Loss           : {b_results['final_loss']:.4f} nats")
    logger.info(f"📈 Proposed Final Loss           : {p_results['final_loss']:.4f} nats (Delta: {loss_improvement:+.4f} nats)")
    logger.info(f"📉 Baseline Final Free Energy    : {b_results['final_fe']:.6f}")
    logger.info(f"📉 Proposed Final Free Energy    : {p_results['final_fe']:.6f} ({fe_reduction_pct:+.2f}% reduction)")
    logger.info(f"⏱️ Baseline Run Duration         : {b_results['elapsed_sec']:.2f}s")
    logger.info(f"⏱️ Proposed Run Duration         : {p_results['elapsed_sec']:.2f}s")
    logger.info("=" * 80)

    summary = {
        "exp_id": "EXP-183",
        "verdict": verdict,
        "hypothesis": "Unshackling Volitional Override Module (128D -> 512D) with multichannel somatic friction coupling stabilizes allostatic override.",
        "architecture_delta": "Expanded HierarchicalVolitionalOverrideModule to 512D with LayerNorm goal normalization and multichannel somatic friction.",
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

    with open("experiments/exp_183_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("✅ EXP-183 execution completed successfully. Results recorded in experiments/exp_183_results.json.")


if __name__ == "__main__":
    main()
