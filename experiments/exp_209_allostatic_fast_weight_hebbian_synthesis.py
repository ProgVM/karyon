# experiments/exp_209_allostatic_fast_weight_hebbian_synthesis.py
"""
===============================================================================
EXP-209: Allostatically-Modulated Synaptic Fast-Weight Hebbian Synthesis Engine
Grounding: KEP Principle 1 (C++20 as Engine), Principle 2 (Biological Realism),
           Principle 7 (Axiom of Unshackled Flow), Principle 8 (Compositional Depth),
           Principle 14 (Axiom of Allostatic Dynamic Forces: No Static Constants),
           Principle 15 (Net2Net Smooth Grafting: Strict Identity at Birth).
===============================================================================
Hypothesis:
In the Grand Synthesis of Neo-Cortical vectors (`h_combined`), the Synaptic Fast-Weight
Hebbian Plasticity pathway (`y_fast = fast_weight_hebbian(h_s1, effective_u_t)`) is currently
linearly merged into `h_combined` with a fixed, static scalar multiplier (0.20 * y_fast):
  h_combined = h_thalamic + 0.20 * y_fast + weighted_error + ...
Under KEP Principle 14 (Axiom of Allostatic Dynamic Forces), biological synaptic memory
traces and fast associative recall are never statically scaled. During high arousal and novelty
(elevated Noradrenaline u_t[4] and Variational Free Energy F_t), rapid episodic/associative binding
via fast weights is prioritized to quickly anchor unexpected transitions. Conversely, under high
Stability (u_t[2]), slow cortical integration (h_thalamic) dominates to preserve long-range discourse.
Replacing the static scalar 0.20 with an Allostatically-Gated Fast-Weight Synthesis Engine:
  g_fast(u_t, F_t) = 0.20 + 0.10 * tanh(W_allostatic * [u_t, F_t])
coupled with a zero-initialized allostatic residual projection graft (Net2Net Smooth Grafting / KEP Principle 15)
that strictly preserves zero-delta function identity at birth t_0, will dynamically tune rapid associative
context injection, eliminate representational interference, and accelerate loss convergence (Loss Delta >= 0.08).
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


class AllostaticFastWeightSynthesizer(nn.Module):
    """
    Allostatically-Modulated Synaptic Fast-Weight Synthesizer (EXP-209).
    Dynamically scales the fast-weight Hebbian memory representation (y_fast)
    based on Somatic Homeostasis (NA, Stability, Curiosity) and Variational Free Energy (F_t).
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

    def forward(self, y_fast: torch.Tensor, u_t: torch.Tensor, fe_loss: torch.Tensor = None) -> torch.Tensor:
        B = y_fast.size(0)
        is_3d = (y_fast.dim() == 3)
        
        if fe_loss is not None:
            fe_norm = torch.clamp(fe_loss.detach() / 10.0, 0.0, 1.0).view(-1, 1)
        else:
            fe_norm = torch.zeros(B, 1, device=self.device)
            
        if u_t.dim() == 2:
            u_flat = u_t if u_t.size(0) == B else u_t[0:1].expand(B, -1)
        else:
            u_flat = u_t.view(1, -1).expand(B, -1)
            
        ctrl_in = torch.cat([u_flat, fe_norm], dim=-1) # [B, 7]
        gate_delta = self.gate_net(ctrl_in) * 0.10      # [B, 1]
        
        eff_scale = 0.20 + gate_delta # Exact 0.20 at birth t_0
        if is_3d:
            eff_scale = eff_scale.unsqueeze(1) # [B, 1, 1]
            
        refined = self.refine_proj(y_fast)
        # Identity-preserving: at birth refine_proj is zero, eff_scale is 0.20 -> exactly 0.20 * y_fast
        return eff_scale * y_fast + 0.20 * refined


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
    logger.info("🔬 [STARTING EXP-209: ALLOSTATIC FAST-WEIGHT HEBBIAN SYNTHESIS BENCHMARK]")
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

    # 2. Proposed Evaluation (EXP-209 Allostatic Fast-Weight Hebbian Synthesis)
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain
    allostatic_fast_weight_synth = AllostaticFastWeightSynthesizer(
        hidden_dim=brain_prop.hidden_dim, homeo_dim=6, device_str=hw.device_str
    ).to(hw.device)

    # Patch fast_weight_hebbian to wrap output with allostatic synthesis
    class PatchedFastWeightHebbian(nn.Module):
        def __init__(self, original_fwh, synthesizer):
            super().__init__()
            self.original_fwh = original_fwh
            self.synthesizer = synthesizer

        def forward(self, h_seq: torch.Tensor, u_t: torch.Tensor):
            y_base = self.original_fwh(h_seq, u_t)
            # In forward_sequence: y_fast is multiplied by 0.20.
            # To provide seamless allostatic synthesis, g_fast dynamically scales y_fast
            # (normalizing by 0.20 so that eff_scale directly replaces 0.20).
            scaled = self.synthesizer(y_base, u_t) * (1.0 / 0.20)
            return scaled

    brain_prop.fast_weight_hebbian = PatchedFastWeightHebbian(brain_prop.fast_weight_hebbian, allostatic_fast_weight_synth)
    
    logger.info("Running Proposed Allostatic Fast-Weight Hebbian Synthesis Evaluation...")
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
