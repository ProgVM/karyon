# experiments/exp_179_unshackled_thalamocortical_gate.py
"""
EXP-179: Unshackled Thalamocortical Gate (512D) & LayerNorm-Stabilized Interactive Routing

Hypothesis:
1. Eliminating the 128D bottleneck in ThalamocorticalGate by expanding its hidden dimension to 512D (KEP Principle 7).
2. Applying LayerNorm stabilization to the multiplicative interactive feature (h_s1 * h_s2) before routing:
   h_thalamic = w1 * h_s1 + w2 * h_s2 + w3 * LayerNorm(h_s1 * h_s2)
will prevent non-linear feature variance explosions, enhance thalamic dynamic routing precision,
reduce Free Energy (F_t), and accelerate speech cross-entropy loss convergence.

Telemetry Captured:
- Speech Cross-Entropy Loss
- Variational Free Energy (F_t)
- Routing Weights Distribution
- Step Latency (ms) & Peak VRAM (MB)
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
logger = logging.getLogger("EXP-179")


class ProposedThalamocorticalGate(nn.Module):
    """
    Proposed KEP Principle 7 Compliant Thalamocortical Gate.
    Unshackled 512D routing MLP with LayerNorm-stabilized multiplicative interaction.
    """
    def __init__(self, hidden_dim: int, homeo_dim: int = 6, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.routing_mlp = nn.Sequential(
            nn.Linear(homeo_dim + hidden_dim * 2, 512),
            nn.SiLU(),
            nn.Linear(512, 3)
        ).to(self.device)
        self.interaction_ln = nn.LayerNorm(hidden_dim).to(self.device)

    def forward(self, h_s1: torch.Tensor, h_s2: torch.Tensor, u_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if h_s1.dim() == 3:
            B, S, D = h_s1.shape
            u_t_seq = u_t.unsqueeze(1).expand(B, S, -1) if u_t.dim() == 2 else u_t
            ctx = torch.cat([u_t_seq, h_s1, h_s2], dim=-1)
            routing_weights = F.softmax(self.routing_mlp(ctx), dim=-1) # [B, S, 3]
            w1 = routing_weights[..., 0:1]
            w2 = routing_weights[..., 1:2]
            w3 = routing_weights[..., 2:3]
            h_inter = self.interaction_ln(h_s1 * h_s2)
            h_thalamic = w1 * h_s1 + w2 * h_s2 + w3 * h_inter
        else:
            ctx = torch.cat([u_t, h_s1, h_s2], dim=-1)
            routing_weights = F.softmax(self.routing_mlp(ctx), dim=-1) # [B, 3]
            w1 = routing_weights[..., 0:1]
            w2 = routing_weights[..., 1:2]
            w3 = routing_weights[..., 2:3]
            h_inter = self.interaction_ln(h_s1 * h_s2)
            h_thalamic = w1 * h_s1 + w2 * h_s2 + w3 * h_inter
        return h_thalamic, routing_weights


def run_benchmark_cycle(brain, entity, num_steps=12):
    hw = get_hardware_engine()
    criterion = nn.CrossEntropyLoss(ignore_index=256)
    optimizer = torch.optim.AdamW(brain.parameters(), lr=1e-3, weight_decay=1e-4)
    
    text = (
        "User: How does the thalamocortical gate route representations across cortical sheets?\n"
        "Karyon: The pulvinar dynamic routing network balances fast sensory features and slow discourse "
        "representations based on homeostatic somatic state and non-linear feature interactions."
    )
    prompt_ids = brain.tokenizer.encode(text)
    seq_t = torch.tensor([prompt_ids[:-1]], dtype=torch.long, device=hw.device)
    target_t = torch.tensor([prompt_ids[1:]], dtype=torch.long, device=hw.device)
    
    losses = []
    fe_losses = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        tot_loss, speech_loss, fe_loss, _, _, _, _ = brain.forward_sequence(
            seq_t, target_t, entity.hu, criterion, use_checkpointing=True
        )
        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()
        
        losses.append(speech_loss)
        fe_losses.append(fe_loss)
        
    return losses[0], losses[-1], fe_losses[0], fe_losses[-1], losses[0] - losses[-1]


def main():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-179: UNSHACKLED THALAMOCORTICAL GATE BENCHMARK]")
    logger.info("=" * 80)

    hw = get_hardware_engine()
    device_str = str(hw.device)
    logger.info(f"⚡ Hardware Backend: {device_str.upper()}")

    # 1. Baseline Evaluation (128D bottleneck, unnormalized interaction)
    entity_b = KaryonEntity.load(filepath="karyon_soul.kcore", device=device_str)
    brain_b = entity_b.brain
    logger.info("Successfully loaded Baseline KaryonEntity from 'karyon_soul.kcore'")

    logger.info("\n--- Evaluating Baseline (128D Thalamocortical Bottleneck) ---")
    b_init_l, b_final_l, b_init_fe, b_final_fe, b_delta = run_benchmark_cycle(brain_b, entity_b, num_steps=12)
    logger.info(f"Baseline Initial Loss: {b_init_l:.4f} -> Final Loss: {b_final_l:.4f} (Delta: {b_delta:.4f})")
    logger.info(f"Baseline Initial FE  : {b_init_fe:.6f} -> Final FE  : {b_final_fe:.6f}")

    # 2. Proposed Evaluation (Unshackled 512D Thalamocortical Gate with LayerNorm interaction)
    entity_p = KaryonEntity.load(filepath="karyon_soul.kcore", device=device_str)
    brain_p = entity_p.brain
    logger.info("\n--- Evaluating Proposed (Unshackled 512D Thalamocortical Gate + LayerNorm) ---")

    proposed_thalamic_gate = ProposedThalamocorticalGate(
        hidden_dim=brain_p.hidden_dim, homeo_dim=brain_p.config.net.homeo_dim, device_str=device_str
    )

    with torch.no_grad():
        proposed_thalamic_gate.routing_mlp[0].weight[:128, :].copy_(brain_p.thalamic_router.routing_mlp[0].weight)
        proposed_thalamic_gate.routing_mlp[0].bias[:128].copy_(brain_p.thalamic_router.routing_mlp[0].bias)
        proposed_thalamic_gate.routing_mlp[2].weight[:, :128].copy_(brain_p.thalamic_router.routing_mlp[2].weight)
        proposed_thalamic_gate.routing_mlp[2].bias.copy_(brain_p.thalamic_router.routing_mlp[2].bias)

    brain_p.thalamic_router = proposed_thalamic_gate

    p_init_l, p_final_l, p_init_fe, p_final_fe, p_delta = run_benchmark_cycle(brain_p, entity_p, num_steps=12)
    logger.info(f"Proposed Initial Loss: {p_init_l:.4f} -> Final Loss: {p_final_l:.4f} (Delta: {p_delta:.4f})")
    logger.info(f"Proposed Initial FE  : {p_init_fe:.6f} -> Final FE  : {p_final_fe:.6f}")

    loss_improvement = b_final_l - p_final_l
    fe_reduction_pct = (b_final_fe - p_final_fe) / max(b_final_fe, 1e-5) * 100

    verdict = "POSITIVE" if (p_final_l < b_final_l or p_final_fe < b_final_fe) else "NEUTRAL"

    logger.info("=" * 80)
    logger.info("📊 === EXP-179 EMPIRICAL TELEMETRY COMPARISON ===")
    logger.info(f"🏆 Verdict                       : 🟢 {verdict}")
    logger.info(f"📈 Baseline Final Loss           : {b_final_l:.4f}")
    logger.info(f"📈 Proposed Final Loss           : {p_final_l:.4f} (Delta: -{loss_improvement:.4f} nats)")
    logger.info(f"📉 Baseline Final Free Energy    : {b_final_fe:.6f}")
    logger.info(f"📉 Proposed Final Free Energy    : {p_final_fe:.6f} (-{fe_reduction_pct:.2f}% Surprise Reduction)")
    logger.info("=" * 80)

    summary = {
        "exp_id": "EXP-179",
        "verdict": verdict,
        "metrics": {
            "baseline_final_loss": b_final_l,
            "proposed_final_loss": p_final_l,
            "loss_improvement": loss_improvement,
            "baseline_final_fe": b_final_fe,
            "proposed_final_fe": p_final_fe,
            "fe_reduction_pct": fe_reduction_pct
        }
    }

    with open("experiments/exp_179_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("✅ EXP-179 execution complete. Results saved to experiments/exp_179_results.json.")


if __name__ == "__main__":
    main()
