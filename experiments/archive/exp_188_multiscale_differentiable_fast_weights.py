# experiments/exp_188_multiscale_differentiable_fast_weights.py
"""
===============================================================================
EXP-188: Multi-Timescale Differentiable Fast-Weight Hebbian Plasticity
Grounding: KEP Principle 1 (C++/GPU Matrix Ops), Principle 2 (Biological Realism),
           Principle 7 (Axiom of Unshackled Flow), Principle 8 (Compositional Depth),
           Principle 14 (Allostatic Forces: No Static Constants).
===============================================================================
Hypothesis:
Replacing single-decay fast weights with multi-head multi-timescale decay
(fast phasic decay lambda_1 in [0.40, 0.80] for immediate sub-word binding,
and slow tonic decay lambda_2 in [0.85, 0.99] for cross-word concept persistence),
combined with precision-weighted homeostatic gating, will reduce sequence Free Energy
and improve perplexity by capturing both local morphemic bindings and long-span working memory
without GPU synchronization stalls.
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
logger = logging.getLogger("EXP-188")


class MultiTimescaleFastWeightHebbian(nn.Module):
    """
    Multi-Timescale Differentiable Fast-Weight Programmers (EXP-188).
    Decomposes fast memory into multiple timescale heads:
    - Head 1 (Phasic): Rapid decay for local syllabic/morphemic boundings (lambda ~ 0.50 - 0.75).
    - Head 2 (Tonic): Sustained retention for episodic working memory (lambda ~ 0.88 - 0.98).
    100% differentiable matrix exponentiation, zero-sync tensorization.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, head_dim: int = 64, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim

        self.k_proj = nn.Linear(hidden_dim, self.total_dim, bias=False).to(self.device)
        self.v_proj = nn.Linear(hidden_dim, self.total_dim, bias=False).to(self.device)
        self.q_proj = nn.Linear(hidden_dim, self.total_dim, bias=False).to(self.device)
        self.out_proj = nn.Linear(self.total_dim, hidden_dim, bias=False).to(self.device)
        self.norm = nn.LayerNorm(hidden_dim).to(self.device)

        # Multi-timescale base decay exponents: log-spaced across heads
        base_decays = torch.tensor([0.60, 0.78, 0.90, 0.96], device=self.device)
        self.register_buffer("base_decays", base_decays.view(1, num_heads, 1, 1))

    def forward(self, h_seq: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        is_2d = (h_seq.dim() == 2)
        if is_2d:
            h_seq = h_seq.unsqueeze(1)

        B, S, D = h_seq.shape
        K = self.k_proj(h_seq).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D_h]
        V = self.v_proj(h_seq).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D_h]
        Q = self.q_proj(h_seq).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D_h]

        # Dynamic Allostatic Forces per batch
        if u_t.dim() == 2:
            curiosity_t = u_t[:, 0:1].view(B, 1, 1, 1)
            stability_t = u_t[:, 2:3].view(B, 1, 1, 1)
            na_t = u_t[:, 4:5].view(B, 1, 1, 1)
            da_t = u_t[:, 5:6].view(B, 1, 1, 1)
        else:
            curiosity_t = u_t[..., 0:1].view(1, 1, 1, 1)
            stability_t = u_t[..., 2:3].view(1, 1, 1, 1)
            na_t = u_t[..., 4:5].view(1, 1, 1, 1)
            da_t = u_t[..., 5:6].view(1, 1, 1, 1)

        # Dynamic allostatic modulation of per-head decay
        head_decays = torch.clamp(
            self.base_decays + 0.05 * stability_t - 0.04 * curiosity_t + 0.03 * da_t,
            0.40, 0.99
        )  # [B, H, 1, 1]

        eta = 0.10 * (1.0 + 1.8 * na_t + 1.0 * curiosity_t)  # [B, 1, 1, 1]

        if S > 1:
            idx = torch.arange(S, device=h_seq.device)
            decay_powers = (idx.unsqueeze(1) - idx.unsqueeze(0)).view(1, 1, S, S)  # [1, 1, S, S]
            decay_powers = torch.clamp(decay_powers, min=0.0)

            log_lambda = torch.log(head_decays)  # [B, H, 1, 1]
            decay_mask = torch.exp(decay_powers * log_lambda)  # [B, H, S, S]
            causal_mask = torch.tril(torch.ones(S, S, device=h_seq.device)).view(1, 1, S, S)
            causal_decay_mask = decay_mask * causal_mask

            attn_sim = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [B, H, S, S]
            attn_decayed = torch.clamp(attn_sim * causal_decay_mask * eta, min=-10.0, max=10.0)
            y_heads = torch.matmul(attn_decayed, V)  # [B, H, S, D_h]
        else:
            attn_sim = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [B, H, 1, 1]
            attn_decayed = torch.clamp(attn_sim * eta, min=-10.0, max=10.0)
            y_heads = torch.matmul(attn_decayed, V)

        y_flat = y_heads.transpose(1, 2).contiguous().view(B, S, self.total_dim)  # [B, S, Total_dim]
        out = self.norm(self.out_proj(y_flat) + h_seq)
        return out.squeeze(1) if is_2d else out


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
    logger.info("🔬 [STARTING EXP-188: MULTI-TIMESCALE DIFFERENTIABLE FAST WEIGHTS BENCHMARK]")
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

    # 1. Evaluate Baseline
    entity_base = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_base = entity_base.brain
    logger.info("Running Baseline Evaluation...")
    b_results = evaluate_model(brain_base, entity_base, text_samples, num_steps=25)

    # 2. Evaluate Proposed
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain
    # Inject Multi-Timescale Hebbian module
    brain_prop.fast_weight_hebbian = MultiTimescaleFastWeightHebbian(
        hidden_dim=brain_prop.hidden_dim, num_heads=4, head_dim=64, device_str=hw.device_str
    ).to(hw.device)
    logger.info("Running Proposed Multi-Timescale Model Evaluation...")
    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    loss_delta = b_results['final_loss'] - p_results['final_loss']
    fe_delta = b_results['final_fe'] - p_results['final_fe']

    logger.info("=" * 80)
    logger.info("📊 === EXP-188 TELEMETRY REPORT ===")
    logger.info(f"  - Baseline Final Loss : {b_results['final_loss']:.4f} nats | Throughput: {b_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Proposed Final Loss : {p_results['final_loss']:.4f} nats | Throughput: {p_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Loss Delta (B - P)  : {loss_delta:.4f} nats")
    logger.info(f"  - Free Energy Delta   : {fe_delta:.6f}")
    logger.info(f"  - Baseline Duration   : {b_results['elapsed']:.3f} s")
    logger.info(f"  - Proposed Duration   : {p_results['elapsed']:.3f} s")

    # KEP Rule #2 Verdict
    verdict = "POSITIVE" if (loss_delta >= 0.08 or (loss_delta >= 0.02 and p_results['tok_per_sec'] >= 0.90 * b_results['tok_per_sec'])) else "NEUTRAL"

    results = {
        "exp_id": "EXP-188",
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

    with open("experiments/exp_188_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"🏆 Final Verdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    return results


if __name__ == "__main__":
    run_benchmark()
