# experiments/exp_205_anchored_hopfield_motor_relaxation.py
"""
===============================================================================
EXP-205: Anchored Allostatic Hopfield Motor Relaxation with Zero-Initialized Residual Basin Projections
Grounding: KEP Principle 2 (Biological Realism), Principle 7 (Unshackled 256D Space),
           Principle 14 (Allostatic Forces: No Static Constants), Principle 15 (Net2Net Smooth Grafting).
===============================================================================
Hypothesis:
In EXP-204, unaligned randomly initialized Hopfield basins disturbed pre-trained motor hidden trajectories.
Anchoring the Hopfield attractor basins to the pre-trained embedding space (or initializing basin projections to zero)
and employing allostatically-gated precision scaling will allow smooth attractor snapping without perturbing birth representations,
yielding a positive loss delta (Loss Delta >= 0.08).
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
logger = logging.getLogger("EXP-205")


class AnchoredAllostaticHopfieldMotorRelaxation(nn.Module):
    """
    Anchored Allostatically-Gated Hopfield Attractor Energy Relaxation Module.
    Anchors basins to pre-trained token embedding centroids and uses zero-initialized
    residual projections to ensure exact zero-delta identity at birth.
    """
    def __init__(self, hidden_dim: int, embed_weights: torch.Tensor, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.hidden_dim = hidden_dim
        
        # Project embedding weights [V, text_dim] to hidden_dim if needed, or sample V centroids
        V, text_dim = embed_weights.shape
        if text_dim != hidden_dim:
            # Linear projection from text_dim to hidden_dim for basin initialization
            proj = nn.Linear(text_dim, hidden_dim, bias=False).to(self.device)
            nn.init.orthogonal_(proj.weight)
            anchored_basins = proj(embed_weights.detach().to(self.device))
        else:
            anchored_basins = embed_weights.detach().to(self.device)
            
        self.basins = nn.Parameter(F.normalize(anchored_basins, p=2, dim=-1)) # [V, hidden_dim]
        
        # Zero-initialized residual output projection (Net2Net Smooth Grafting / KEP Principle 15)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device)
        nn.init.zeros_(self.out_proj.weight)
        
        # Allostatic Precision & Step Controller Net
        self.allostatic_controller = nn.Sequential(
            nn.Linear(6 + 1, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, h_relaxed: torch.Tensor, u_t: torch.Tensor, fe_loss: torch.Tensor) -> torch.Tensor:
        B, S, D = h_relaxed.shape
        
        fe_norm = torch.clamp(fe_loss.detach() / 10.0, 0.0, 1.0).view(-1, 1)
        if u_t.dim() == 2:
            u_exp = u_t.unsqueeze(1).expand(B, S, -1)
        else:
            u_exp = u_t.view(1, 1, -1).expand(B, S, -1)
            
        fe_exp = fe_norm.view(1, 1, 1).expand(B, S, -1)
        
        ctrl_in = torch.cat([u_exp, fe_exp], dim=-1) # [B, S, 7]
        ctrl_out = self.allostatic_controller(ctrl_in) # [B, S, 2]
        
        # Soft beta precision scale [4.0, 12.0]
        beta_precision = 4.0 + ctrl_out[..., 0:1] * 8.0
        # Dynamic step gain scaled to 0.05
        step_gain = ctrl_out[..., 1:2] * 0.05
        
        norm_h = F.normalize(h_relaxed, p=2, dim=-1)
        sims = torch.einsum('bsd,vd->bsv', norm_h, F.normalize(self.basins, p=2, dim=-1))
        attn_weights = F.softmax(sims * beta_precision, dim=-1)
        snapped_state = torch.einsum('bsv,vd->bsd', attn_weights, self.basins)
        
        # Identity-preserving residual addition
        return h_relaxed + step_gain * self.out_proj(snapped_state)


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
    logger.info("🔬 [STARTING EXP-205: ANCHORED ALLOSTATIC HOPFIELD MOTOR RELAXATION BENCHMARK]")
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

    # 2. Proposed Evaluation
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain
    
    embed_w = brain_prop.embed.embed.weight.data # [V, text_dim]
    brain_prop.anchored_hopfield_motor_relaxation = AnchoredAllostaticHopfieldMotorRelaxation(
        hidden_dim=brain_prop.hidden_dim, embed_weights=embed_w, device_str=hw.device_str
    ).to(hw.device)

    logger.info("Running Proposed Anchored Allostatic Hopfield Motor Relaxation Evaluation...")
    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    loss_delta = b_results['final_loss'] - p_results['final_loss']
    fe_delta = b_results['final_fe'] - p_results['final_fe']

    logger.info("=" * 80)
    logger.info("📊 === EXP-205 TELEMETRY REPORT ===")
    logger.info(f"  - Baseline Final Loss : {b_results['final_loss']:.4f} nats | Throughput: {b_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Proposed Final Loss : {p_results['final_loss']:.4f} nats | Throughput: {p_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Loss Delta (B - P)  : {loss_delta:.4f} nats")
    logger.info(f"  - Free Energy Delta   : {fe_delta:.6f}")
    logger.info(f"  - Baseline Duration   : {b_results['elapsed']:.3f} s")
    logger.info(f"  - Proposed Duration   : {p_results['elapsed']:.3f} s")

    verdict = "POSITIVE" if (loss_delta >= 0.08 or (loss_delta >= 0.02 and p_results['tok_per_sec'] >= 0.90 * b_results['tok_per_sec'])) else "NEUTRAL"

    results = {
        "exp_id": "EXP-205",
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

    with open("experiments/exp_205_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"🏆 Final Verdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    return results


if __name__ == "__main__":
    run_benchmark()
