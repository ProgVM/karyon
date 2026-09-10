# experiments/exp_211_allostatic_cortical_boundary_gating.py
"""
===============================================================================
EXP-211: Allostatically-Modulated Cortical Boundary Gating Engine
Grounding: KEP Principle 1 (C++20 as Engine), Principle 2 (Biological Realism),
           Principle 8 (Compositional Depth), Principle 14 (Axiom of Allostatic Dynamic Forces),
           Principle 15 (Net2Net Smooth Grafting: Strict Identity at Birth).
===============================================================================
Hypothesis:
In the Grand Synthesis of Neo-Cortical vectors, the Stage 2 representation `h_s2` is gated
by the entropy macro-boundary signal before entering the thalamocortical router and predictive residual router:
  `h_s2_gated = h_s2 * (0.50 + 1.00 * boundary_gate.unsqueeze(-1))`
Here, `0.50` (tonic baseline transmission) and `1.00` (phasic boundary surge) are static numerical constants.
Under KEP Principle 14 (Axiom of Allostatic Dynamic Forces), the baseline permeability and boundary sensitivity
of cortical laminar transmission cannot be invariant static scalars:
- When Noradrenaline (u_t[4]) and Curiosity (u_t[0]) are elevated (high cognitive engagement / exploratory drive),
  the sensitivity to syntactic and conceptual boundaries must increase to capture fine-grained transitions.
- Conversely, under high Stability (u_t[2]), baseline tonic transmission should be higher to maintain steady discourse flow.
Replacing the static `0.50 + 1.00 * boundary_gate` with an Allostatically-Modulated Cortical Boundary Gating Engine:
  alpha_tonic(u_t) = 0.50 * (1.0 + 0.15 * tanh(W_tonic * u_t))
  beta_phasic(u_t) = 1.00 * (1.0 + 0.20 * tanh(W_phasic * u_t))
  h_s2_gated = h_s2 * (alpha_tonic(u_t) + beta_phasic(u_t) * boundary_gate.unsqueeze(-1)) + Refine(h_s2)
coupled with a zero-initialized Net2Net refinement projection (strictly preserving identity at birth t_0,
KEP Principle 15) will dynamically adapt cortical laminar routing, optimize semantic flow, and accelerate loss convergence (Loss Delta >= 0.08).
"""

import sys
import os
import time
import math
import json
import logging
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_entity import KaryonEntity
from karyon_hardware import get_hardware_engine

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-211")


class AllostaticCorticalBoundaryGate(nn.Module):
    """
    Allostatically-Modulated Cortical Boundary Gating Engine (EXP-211).
    Dynamically scales tonic and phasic boundary permeability of Stage 2 cortical processing
    based on Somatic Homeostasis (NA, DA, Curiosity, Energy, Stability).
    Preserves exact mathematical identity at birth via zero-initialized weights (KEP Principle 15).
    """
    def __init__(self, hidden_dim: int, homeo_dim: int = 6, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.hidden_dim = hidden_dim

        # Dynamic allostatic modulation network: predicts tonic and phasic scaling deltas
        self.allostatic_net = nn.Sequential(
            nn.Linear(homeo_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
            nn.Tanh()
        ).to(self.device)
        nn.init.zeros_(self.allostatic_net[2].weight)
        nn.init.zeros_(self.allostatic_net[2].bias)

        # Zero-initialized refinement layer (Net2Net Smooth Grafting / KEP Principle 15)
        self.refine_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device)
        nn.init.zeros_(self.refine_proj.weight)

    def forward(self, h_s2: torch.Tensor, boundary_gate: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        B = h_s2.size(0)
        is_3d = (h_s2.dim() == 3)

        if u_t.dim() == 2:
            u_flat = u_t if u_t.size(0) == B else u_t[0:1].expand(B, -1)
        else:
            u_flat = u_t.view(1, -1).expand(B, -1)

        mod_deltas = self.allostatic_net(u_flat) # [B, 2]
        delta_tonic = mod_deltas[:, 0:1] * 0.15   # [B, 1]
        delta_phasic = mod_deltas[:, 1:2] * 0.20  # [B, 1]

        # Base parameters: 0.50 tonic, 1.00 phasic (exact match at birth t_0)
        eff_tonic = 0.50 * (1.0 + delta_tonic)
        eff_phasic = 1.00 * (1.0 + delta_phasic)

        if is_3d:
            eff_tonic = eff_tonic.unsqueeze(1)    # [B, 1, 1]
            eff_phasic = eff_phasic.unsqueeze(1)  # [B, 1, 1]
            gate_expanded = boundary_gate.unsqueeze(-1) # [B, S, 1]
        else:
            gate_expanded = boundary_gate.unsqueeze(-1) # [B, 1]

        scale = eff_tonic + eff_phasic * gate_expanded
        gated_h_s2 = h_s2 * scale
        refined = self.refine_proj(gated_h_s2)
        # Identity-preserving: at birth refine_proj is zero, scale is exactly (0.50 + 1.00 * boundary_gate)
        return gated_h_s2 + refined


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
    logger.info("🔬 [STARTING EXP-211: ALLOSTATIC CORTICAL BOUNDARY GATING BENCHMARK]")
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

    # 2. Proposed Evaluation (EXP-211 Allostatic Cortical Boundary Gating)
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain
    boundary_gate_mod = AllostaticCorticalBoundaryGate(
        hidden_dim=brain_prop.hidden_dim, homeo_dim=6, device_str=hw.device_str
    ).to(hw.device)

    # Patch forward_sequence to use boundary_gate_mod
    orig_forward_seq = brain_prop.forward_sequence

    def patched_forward_sequence(x_seq, target_seq, hu_state, criterion, chunk_size=1024, use_checkpointing=False, episodic_memory=None):
        # We hook into forward_sequence by modifying the boundary gate scaling line dynamically
        # Or running with the patched brain
        return orig_forward_seq(x_seq, target_seq, hu_state, criterion, chunk_size, use_checkpointing, episodic_memory)

    # Instead of monkeypatching the entire forward_sequence, let's wrap entropy_macro_gate
    # and provide a custom boundary gate representation or hook
    orig_entropy_gate = brain_prop.entropy_macro_gate

    class EntropyMacroGateWrapper(nn.Module):
        def __init__(self, base_entropy, gate_mod, agent):
            super().__init__()
            self.base_entropy = base_entropy
            self.gate_mod = gate_mod
            self.agent = agent

        def forward(self, h_s1: torch.Tensor):
            entropy, boundary_gate = self.base_entropy(h_s1)
            # In forward_sequence line 1817:
            # h_s2_gated = h_s2 * (0.50 + 1.00 * boundary_gate.unsqueeze(-1))
            # If we modulate boundary_gate such that:
            # 0.50 + 1.00 * boundary_gate_effective = eff_tonic + eff_phasic * boundary_gate
            # => boundary_gate_effective = ((eff_tonic - 0.50) + eff_phasic * boundary_gate) / 1.00
            u_t = getattr(self.agent, '_current_effective_u_t', None)
            if u_t is not None:
                B = h_s1.size(0)
                u_flat = u_t if u_t.size(0) == B else u_t[0:1].expand(B, -1)
                mod_deltas = self.gate_mod.allostatic_net(u_flat)
                delta_tonic = mod_deltas[:, 0:1] * 0.15
                delta_phasic = mod_deltas[:, 1:2] * 0.20
                eff_tonic = 0.50 * (1.0 + delta_tonic)
                eff_phasic = 1.00 * (1.0 + delta_phasic)
                if h_s1.dim() == 3:
                    eff_tonic = eff_tonic.squeeze(1)
                    eff_phasic = eff_phasic.squeeze(1)
                effective_boundary_gate = (eff_tonic - 0.50) + eff_phasic * boundary_gate
                return entropy, effective_boundary_gate
            return entropy, boundary_gate

        @property
        def entropy_head(self):
            return self.base_entropy.entropy_head

        @property
        def macro_boundary_proj(self):
            return self.base_entropy.macro_boundary_proj

    orig_will = brain_prop.will_engine
    class WillWrapper(nn.Module):
        def __init__(self, will_eng, agent):
            super().__init__()
            self.will_eng = will_eng
            self.agent = agent

        def forward(self, h_s2, u_t):
            eff_u, gamma, strain = self.will_eng(h_s2, u_t)
            self.agent._current_effective_u_t = eff_u
            return eff_u, gamma, strain

    brain_prop.will_engine = WillWrapper(orig_will, brain_prop)
    brain_prop.entropy_macro_gate = EntropyMacroGateWrapper(orig_entropy_gate, boundary_gate_mod, brain_prop)

    logger.info("Running Proposed Allostatic Cortical Boundary Gating Evaluation...")
    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    loss_delta = b_results['final_loss'] - p_results['final_loss']
    fe_delta = b_results['final_fe'] - p_results['final_fe']

    logger.info("=" * 80)
    logger.info("📊 === EXP-211 TELEMETRY REPORT ===")
    logger.info(f"  - Baseline Final Loss : {b_results['final_loss']:.4f} nats | Throughput: {b_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Proposed Final Loss : {p_results['final_loss']:.4f} nats | Throughput: {p_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Loss Delta (B - P)  : {loss_delta:.4f} nats")
    logger.info(f"  - Free Energy Delta   : {fe_delta:.6f}")
    logger.info(f"  - Baseline Duration   : {b_results['elapsed']:.3f} s")
    logger.info(f"  - Proposed Duration   : {p_results['elapsed']:.3f} s")

    verdict = "POSITIVE" if (loss_delta >= 0.08 or (loss_delta >= 0.02 and p_results['tok_per_sec'] >= 0.90 * b_results['tok_per_sec'])) else "NEUTRAL"

    results = {
        "exp_id": "EXP-211",
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

    with open("experiments/exp_211_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"🏆 Final Verdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    return results


if __name__ == "__main__":
    run_benchmark()
