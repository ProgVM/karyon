# experiments/exp_209_allostatic_topdown_prior_synthesis.py
"""
===============================================================================
EXP-209: Allostatically-Modulated Top-Down Cortical Prior Synthesis Engine
Grounding: KEP Principle 1 (C++20 as Engine), Principle 2 (Biological Realism),
           Principle 8 (Compositional Depth), Principle 14 (Axiom of Allostatic Dynamic Forces),
           Principle 15 (Net2Net Smooth Grafting: Strict Identity at Birth).
===============================================================================
Hypothesis:
In the Grand Synthesis of Neo-Cortical vectors, the top-down cortical predictive prior
(`topdown_prior = topdown_prior_proj(h_s2_gated)`) provides high-level semantic regularization
and hierarchical predictive constraints to pre-attractor cortical dynamics (`h_combined`).
In `forward()` and `forward_cached_step()`, `topdown_prior` is weighted by a static scalar (0.15 * topdown_prior),
while in `forward_sequence()` it uses a semi-dynamic gain `(0.10 + 0.15 * phasic_gain) * topdown_prior`.
Under KEP Principle 14 (Axiom of Allostatic Dynamic Forces), top-down prior belief weighting must be
dynamically governed by the allostatic landscape: when Noradrenaline (u_t[4]) and Variational Free Energy (F_t)
are high (sensory surprise/urgency), bottom-up sensory evidence dominates and top-down priors should soften
to allow rapid learning of new patterns; conversely, when Stability (u_t[2]) is high, top-down priors should
strengthen to maintain discourse coherence and syntactic integrity.
Replacing the static/semi-static top-down weighting with an Allostatically-Gated Top-Down Prior Synthesis Engine:
  g_prior(u_t, F_t) = 0.15 + 0.08 * tanh(W_allostatic * [u_t, F_t])
coupled with a zero-initialized residual refinement graft (Net2Net Smooth Grafting / KEP Principle 15)
that strictly preserves zero-delta function identity at birth t_0, will enhance contextual stability,
balance top-down belief with bottom-up surprise, and accelerate loss convergence (Loss Delta >= 0.08).
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
logger = logging.getLogger("EXP-209")


class AllostaticTopdownPriorSynthesizer(nn.Module):
    """
    Allostatically-Modulated Top-Down Prior Synthesizer (EXP-209).
    Dynamically scales the top-down predictive prior (topdown_prior)
    based on Somatic Homeostasis (Stability, NA, Energy) and Variational Free Energy (F_t).
    Employs zero-initialized residual projection to preserve strict zero-delta function identity at birth.
    """
    def __init__(self, hidden_dim: int, homeo_dim: int = 6, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.hidden_dim = hidden_dim
        
        # Zero-initialized refinement layer (Net2Net Smooth Grafting / KEP Principle 15)
        self.refine_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device)
        nn.init.zeros_(self.refine_proj.weight)
        
        # Allostatic Gating Network with zero initialization for exact birth identity
        self.gate_net = nn.Sequential(
            nn.Linear(homeo_dim + 1, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        ).to(self.device)
        nn.init.zeros_(self.gate_net[2].weight)
        nn.init.zeros_(self.gate_net[2].bias)

    def forward(self, topdown_prior: torch.Tensor, u_t: torch.Tensor, fe_loss: torch.Tensor = None) -> torch.Tensor:
        B = topdown_prior.size(0)
        is_3d = (topdown_prior.dim() == 3)
        
        if fe_loss is not None:
            fe_norm = torch.clamp(fe_loss.detach() / 10.0, 0.0, 1.0).view(-1, 1)
        else:
            fe_norm = torch.zeros(B, 1, device=self.device)
            
        if u_t.dim() == 2:
            u_flat = u_t if u_t.size(0) == B else u_t[0:1].expand(B, -1)
        else:
            u_flat = u_t.view(1, -1).expand(B, -1)
            
        ctrl_in = torch.cat([u_flat, fe_norm], dim=-1) # [B, 7]
        gate_delta = self.gate_net(ctrl_in) * 0.08      # [B, 1]
        
        eff_scale = 1.0 + gate_delta # Exact 1.0 multiplier at birth t_0
        if is_3d:
            eff_scale = eff_scale.unsqueeze(1) # [B, 1, 1]
            
        refined = self.refine_proj(topdown_prior)
        # Identity-preserving: at birth refine_proj is zero, eff_scale is 1.0 -> exactly topdown_prior
        return eff_scale * topdown_prior + refined


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
    logger.info("🔬 [STARTING EXP-209: ALLOSTATIC TOP-DOWN PRIOR SYNTHESIS BENCHMARK]")
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

    # 2. Proposed Evaluation (EXP-209 Allostatic Top-Down Prior Synthesis)
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain
    synthesizer = AllostaticTopdownPriorSynthesizer(
        hidden_dim=brain_prop.hidden_dim, homeo_dim=6, device_str=hw.device_str
    ).to(hw.device)

    # Patch topdown_prior_proj to incorporate allostatic synthesis
    class PatchedTopdownPriorProj(nn.Module):
        def __init__(self, original_proj, synth):
            super().__init__()
            self.original_proj = original_proj
            self.synth = synth

        def forward(self, h_s2_gated: torch.Tensor, u_t: torch.Tensor = None):
            prior_base = self.original_proj(h_s2_gated)
            if u_t is not None:
                return self.synth(prior_base, u_t)
            return prior_base

    # Wrap the forward call inside brain_prop by patching topdown_prior_proj
    # To also pass u_t seamlessly, let's wrap the sequential layer to check context
    original_topdown_proj = brain_prop.topdown_prior_proj
    class TopdownPriorWrapper(nn.Module):
        def __init__(self, base_proj, synth, agent):
            super().__init__()
            self.base_proj = base_proj
            self.synth = synth
            self.agent = agent

        def forward(self, h_s2_gated: torch.Tensor):
            prior = self.base_proj(h_s2_gated)
            # Retrieve current effective_u_t from agent if available
            u_t = getattr(self.agent, '_current_effective_u_t', None)
            if u_t is not None:
                return self.synth(prior, u_t)
            return prior

    # In forward_sequence, effective_u_t is calculated at line 1813:
    # effective_u_t, gamma_override, allostatic_strain = self.will_engine(h_s2, u_t)
    # We hook into will_engine to store _current_effective_u_t
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
    brain_prop.topdown_prior_proj = TopdownPriorWrapper(original_topdown_proj, synthesizer, brain_prop)
    
    logger.info("Running Proposed Allostatic Top-Down Prior Synthesis Evaluation...")
    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    loss_delta = b_results['final_loss'] - p_results['final_loss']
    fe_delta = b_results['final_fe'] - p_results['final_fe']

    logger.info("=" * 80)
    logger.info("📊 === EXP-209 TELEMETRY REPORT ===")
    logger.info(f"  - Baseline Final Loss : {b_results['final_loss']:.4f} nats | Throughput: {b_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Proposed Final Loss : {p_results['final_loss']:.4f} nats | Throughput: {p_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Loss Delta (B - P)  : {loss_delta:.4f} nats")
    logger.info(f"  - Free Energy Delta   : {fe_delta:.6f}")
    logger.info(f"  - Baseline Duration   : {b_results['elapsed']:.3f} s")
    logger.info(f"  - Proposed Duration   : {p_results['elapsed']:.3f} s")

    verdict = "POSITIVE" if (loss_delta >= 0.08 or (loss_delta >= 0.02 and p_results['tok_per_sec'] >= 0.90 * b_results['tok_per_sec'])) else "NEUTRAL"

    results = {
        "exp_id": "EXP-209",
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

    with open("experiments/exp_209_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"🏆 Final Verdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    return results


if __name__ == "__main__":
    run_benchmark()
