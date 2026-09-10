# experiments/exp_207_allostatic_thalamocortical_gating.py
"""
===============================================================================
EXP-207: Allostatically-Gated Thalamocortical Dynamic Routing Engine
Grounding: KEP Principle 2 (Biological Realism), Principle 7 (Axiom of Unshackled Flow),
           Principle 8 (Compositional Depth), Principle 14 (Axiom of Allostatic Dynamic Forces: No Static Constants),
           Principle 15 (Net2Net Smooth Grafting: Strict Identity at Birth).
===============================================================================
Hypothesis:
In the neo-cortical routing architecture, the Thalamocortical dynamic router (TRN / Pulvinar gate)
interactively routes Stage 1 morpho-syntactic features, Stage 2 semantic-discourse representations,
and their non-linear interactive term (LN(h_s1 * h_s2)).
Replacing static internal routing MLP weights and unmodulated interactive projections with an
Allostatically-Gated Thalamocortical Dynamic Routing Engine—where routing logits, non-linear interaction
gain, and precision gating are dynamically scaled as a continuous function of Somatic Homeostasis
(Energy u_t[1], Noradrenaline u_t[4], Dopamine u_t[5]) and Variational Free Energy (F_t)—coupled with
a zero-initialized residual interaction graft (Net2Net Smooth Grafting / KEP Principle 15) to guarantee
zero-delta function identity at birth, will optimize cross-cortical information integration, prevent
representational bleed, and accelerate loss convergence (Loss Delta >= 0.08).
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
logger = logging.getLogger("EXP-207")


class AllostaticThalamocorticalRouter(nn.Module):
    """
    Allostatically-Gated Thalamocortical Dynamic Routing Engine (EXP-207).
    Couples Pulvinar/TRN dynamic routing between Stage 1, Stage 2, and their interactive
    manifold to Somatic Homeostasis (Energy, Noradrenaline, Dopamine) and Free Energy (F_t).
    Employs zero-initialized residual projection to preserve strict zero-delta function identity at birth.
    """
    def __init__(self, hidden_dim: int, homeo_dim: int = 6, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.hidden_dim = hidden_dim
        
        # Zero-initialized interactive refinement projection (Net2Net Smooth Grafting)
        self.interaction_refine = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device)
        nn.init.zeros_(self.interaction_refine.weight)
        
        # Allostatic Routing Modulator Net: maps homeo state [6] + Free Energy [1] -> [4] factors
        self.allostatic_modulator = nn.Sequential(
            nn.Linear(homeo_dim + 1, 32),
            nn.SiLU(),
            nn.Linear(32, 4),
            nn.Tanh()
        ).to(self.device)
        # Initialize final layer to zero to ensure zero-delta birth
        nn.init.zeros_(self.allostatic_modulator[2].weight)
        nn.init.zeros_(self.allostatic_modulator[2].bias)

    def forward(self, h_thalamic: torch.Tensor, h_s1: torch.Tensor, h_s2: torch.Tensor, u_t: torch.Tensor, fe_loss: torch.Tensor = None) -> torch.Tensor:
        B = h_thalamic.size(0)
        is_3d = (h_thalamic.dim() == 3)
        S = h_thalamic.size(1) if is_3d else 1
        
        # Format Free Energy
        if fe_loss is not None:
            fe_norm = torch.clamp(fe_loss.detach() / 10.0, 0.0, 1.0).view(-1, 1)
        else:
            fe_norm = torch.zeros(B, 1, device=self.device)
            
        if u_t.dim() == 2:
            u_flat = u_t
        else:
            u_flat = u_t.view(1, -1).expand(B, -1)
            
        ctrl_in = torch.cat([u_flat, fe_norm], dim=-1) # [B, 7]
        allostatic_factors = self.allostatic_modulator(ctrl_in) # [B, 4]
        
        # Dynamic interaction gain
        inter_gain = allostatic_factors[:, 3:4] * 0.10 # [B, 1]
        
        if is_3d:
            inter_gain = inter_gain.unsqueeze(1) # [B, 1, 1]
            
        # Non-linear interaction between Stage 1 and Stage 2
        h_inter = F.silu(h_s1 * h_s2)
        refined_inter = self.interaction_refine(h_inter)
        
        # Identity-preserving residual routing
        h_out = h_thalamic + inter_gain * refined_inter
        return h_out


class PatchedThalamocorticalGate(nn.Module):
    """
    Wraps existing thalamic_router with AllostaticThalamocorticalRouter.
    """
    def __init__(self, original_router, allostatic_router):
        super().__init__()
        self.original_router = original_router
        self.allostatic_router = allostatic_router

    def forward(self, h_s1: torch.Tensor, h_s2: torch.Tensor, u_t: torch.Tensor):
        h_thalamic, routing_weights = self.original_router(h_s1, h_s2, u_t)
        h_gated = self.allostatic_router(h_thalamic, h_s1, h_s2, u_t)
        return h_gated, routing_weights


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
    logger.info("🔬 [STARTING EXP-207: ALLOSTATIC THALAMOCORTICAL GATING BENCHMARK]")
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

    # 2. Proposed Evaluation (EXP-207 Allostatic Thalamocortical Dynamic Router)
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain
    allostatic_router = AllostaticThalamocorticalRouter(
        hidden_dim=brain_prop.hidden_dim, homeo_dim=6, device_str=hw.device_str
    ).to(hw.device)

    # Patch brain_prop thalamic router
    brain_prop.thalamic_router = PatchedThalamocorticalGate(brain_prop.thalamic_router, allostatic_router)
    
    logger.info("Running Proposed Allostatic Thalamocortical Dynamic Routing Evaluation...")
    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    loss_delta = b_results['final_loss'] - p_results['final_loss']
    fe_delta = b_results['final_fe'] - p_results['final_fe']

    logger.info("=" * 80)
    logger.info("📊 === EXP-207 TELEMETRY REPORT ===")
    logger.info(f"  - Baseline Final Loss : {b_results['final_loss']:.4f} nats | Throughput: {b_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Proposed Final Loss : {p_results['final_loss']:.4f} nats | Throughput: {p_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Loss Delta (B - P)  : {loss_delta:.4f} nats")
    logger.info(f"  - Free Energy Delta   : {fe_delta:.6f}")
    logger.info(f"  - Baseline Duration   : {b_results['elapsed']:.3f} s")
    logger.info(f"  - Proposed Duration   : {p_results['elapsed']:.3f} s")

    verdict = "POSITIVE" if (loss_delta >= 0.08 or (loss_delta >= 0.02 and p_results['tok_per_sec'] >= 0.90 * b_results['tok_per_sec'])) else "NEUTRAL"

    results = {
        "exp_id": "EXP-207",
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

    with open("experiments/exp_207_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"🏆 Final Verdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    return results


if __name__ == "__main__":
    run_benchmark()
