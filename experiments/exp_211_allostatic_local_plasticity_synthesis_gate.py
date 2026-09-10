# experiments/exp_211_allostatic_local_plasticity_synthesis_gate.py
"""
===============================================================================
EXP-211: Allostatically-Modulated Local Neuromodulated Plasticity Synthesis Engine
Grounding: KEP Principle 1 (C++20 as Engine), Principle 2 (Biological Realism),
           Principle 8 (Compositional Depth), Principle 14 (Axiom of Allostatic Dynamic Forces),
           Principle 15 (Net2Net Smooth Grafting: Strict Identity at Birth).
===============================================================================
Hypothesis:
In the Grand Synthesis of Neo-Cortical vectors, the local 3-factor neuromodulated plasticity
vector `y_local` (computed via native C++20 `LocalNeuromodulatedPlasticity(h_s2_gated)`) provides
immediate synaptic fast-weight adaptation driven by pre-synaptic activation, post-synaptic error,
and noradrenergic/dopaminergic signals.
Currently in `forward_sequence()` (line 1833), `y_local` is integrated into the pre-attractor
state `h_combined` via a static scalar:
  `h_combined = ... + 0.10 * y_local + ...`
Under KEP Principle 14 (Axiom of Allostatic Dynamic Forces), local neuromodulated plasticity
efficacy must not be a frozen constant. In biological cortical circuits, local synaptic transmission
is modulated by the homeostatic state: elevated Noradrenaline (u_t[4]) and Dopamine (u_t[5])
signify high cognitive arousal and reinforcement, demanding amplified local synaptic plasticity,
whereas high somatic stress or low stability requires damping to avoid signal saturation.
Replacing the static `0.10 * y_local` with an Allostatically-Modulated Local Plasticity Synthesizer:
  gamma_local(u_t) = 0.10 * (1.0 + 0.20 * tanh(W_gate * u_t))
coupled with a zero-initialized Net2Net projection refinement graft (strictly preserving zero-delta
function identity at birth t_0, KEP Principle 15) will dynamically adapt local plastic contributions,
enhance cortical representation fidelity, and accelerate loss convergence (Loss Delta >= 0.08).
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
logger = logging.getLogger("EXP-211")


class AllostaticLocalPlasticitySynthesizer(nn.Module):
    """
    Allostatically-Modulated Local Plasticity Synthesis Engine (EXP-211).
    Dynamically modulates and refines the local neuromodulated plasticity output y_local
    based on Somatic Homeostasis (NA, DA, Curiosity, Energy, Stability).
    Zero-initialized Net2Net projection guarantees exact mathematical identity at birth.
    """
    def __init__(self, hidden_dim: int, homeo_dim: int = 6, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.hidden_dim = hidden_dim

        # Dynamic allostatic gating network
        self.gate_net = nn.Sequential(
            nn.Linear(homeo_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        ).to(self.device)
        nn.init.zeros_(self.gate_net[2].weight)
        nn.init.zeros_(self.gate_net[2].bias)

        # Zero-initialized refinement layer (KEP Principle 15 Net2Net Smooth Grafting)
        self.refine_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device)
        nn.init.zeros_(self.refine_proj.weight)

    def forward(self, y_local: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        B = y_local.size(0)
        is_3d = (y_local.dim() == 3)

        if u_t.dim() == 2:
            u_flat = u_t if u_t.size(0) == B else u_t[0:1].expand(B, -1)
        else:
            u_flat = u_t.view(1, -1).expand(B, -1)

        # Dynamic allostatic modulation factor centered at 1.0 (exact 1.0 at birth)
        allostatic_mod = 1.0 + 0.20 * self.gate_net(u_flat) # [B, 1]
        if is_3d:
            allostatic_mod = allostatic_mod.unsqueeze(1) # [B, 1, 1]

        refined = self.refine_proj(y_local)
        # Identity-preserving: at birth refine_proj is zero, allostatic_mod is 1.0 -> exactly y_local
        return allostatic_mod * y_local + refined


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
    logger.info("🔬 [STARTING EXP-211: ALLOSTATIC LOCAL PLASTICITY SYNTHESIS BENCHMARK]")
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

    # 2. Proposed Evaluation (EXP-211 Allostatic Local Plasticity Synthesis)
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain
    synthesizer = AllostaticLocalPlasticitySynthesizer(
        hidden_dim=brain_prop.hidden_dim, homeo_dim=6, device_str=hw.device_str
    ).to(hw.device)

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

    # Wrap local_plasticity to integrate AllostaticLocalPlasticitySynthesizer seamlessly
    orig_local_plasticity = brain_prop.local_plasticity
    class PatchedLocalPlasticity(nn.Module):
        def __init__(self, base_lp, synth, agent):
            super().__init__()
            self.base_lp = base_lp
            self.synth = synth
            self.agent = agent

        def forward(self, h_s2_gated: torch.Tensor) -> torch.Tensor:
            y_raw = self.base_lp(h_s2_gated)
            u_t = getattr(self.agent, '_current_effective_u_t', None)
            if u_t is not None:
                return self.synth(y_raw, u_t)
            return y_raw

        def adapt_local_fast_weights(self, pre_act, post_err, na_t, da_t):
            return self.base_lp.adapt_local_fast_weights(pre_act, post_err, na_t, da_t)

        @property
        def W_base(self):
            return self.base_lp.W_base

        @property
        def W_fast(self):
            return self.base_lp.W_fast

    brain_prop.local_plasticity = PatchedLocalPlasticity(orig_local_plasticity, synthesizer, brain_prop)

    logger.info("Running Proposed Allostatic Local Plasticity Synthesis Evaluation...")
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
