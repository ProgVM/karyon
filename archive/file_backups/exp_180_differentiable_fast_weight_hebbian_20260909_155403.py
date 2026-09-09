# experiments/exp_180_differentiable_fast_weight_hebbian.py
"""
EXP-180: Differentiable Fast-Weight Hebbian Plasticity & Zero-Sync Allostatic Decay

Biophysical & Cybernetic Foundation:
1. Fast-weight Hebbian plasticity models rapidly decaying synaptic associations in the hippocampus
   and prefrontal cortex, complementing slower gradient-based weight updates.
2. Autograd & Differentiability (KEP Principle 14): In the production FastWeightHebbianPlasticity,
   the decay rate lambda_decay was converted to a Python float via .item() inside forward().
   This broke the computational graph, preventing gradients from flowing back to somatic state u_t,
   and forced a global batch-average decay rate.
3. Zero-Sync GPU Execution (KEP Principle 1 / Axis A): The .item() call forces a host-device
   synchronization stall, blocking the GPU pipeline and lowering throughput.
4. Proposed Solution:
   Implement a 100% differentiable, tensorized decay matrix using element-wise exponentiation:
   decay_matrix = torch.exp(decay_powers * torch.log(lambda_decay))
   This preserves individual batch item decay rates, enables end-to-end autograd optimization
   of homeostatic decay parameters, and eliminates all GPU sync stalls.

Telemetry Captured:
- Pre/Post Speech Cross-Entropy Loss (nats)
- Pre/Post Variational Free Energy (F_t)
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
logger = logging.getLogger("EXP-180")


class ProposedFastWeightHebbianPlasticity(nn.Module):
    """
    Proposed 100% Differentiable, Zero-Sync Fast-Weight Hebbian Plasticity.
    Eliminates CPU-GPU sync stalls and enables backpropagation to somatic states.
    """
    def __init__(self, hidden_dim: int, key_dim: int = 256, value_dim: int = 256, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        self.k_proj = nn.Linear(hidden_dim, key_dim, bias=False).to(self.device)
        self.v_proj = nn.Linear(hidden_dim, value_dim, bias=False).to(self.device)
        self.q_proj = nn.Linear(hidden_dim, key_dim, bias=False).to(self.device)
        self.out_proj = nn.Linear(value_dim, hidden_dim, bias=False).to(self.device)

    def forward(self, h_seq: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        is_2d = (h_seq.dim() == 2)
        if is_2d:
            h_seq = h_seq.unsqueeze(1)
            
        B, S, D = h_seq.shape
        K = self.k_proj(h_seq)
        V = self.v_proj(h_seq)
        Q = self.q_proj(h_seq)
        
        # Dynamic Allostatic Forces (KEP Principle 14)
        if u_t.dim() == 2:
            curiosity_t = u_t[:, 0:1].unsqueeze(1) if u_t.size(0) == B else u_t[0, 0].view(1, 1, 1)
            stability_t = u_t[:, 2:3].unsqueeze(1) if u_t.size(0) == B else u_t[0, 2].view(1, 1, 1)
            na_t = u_t[:, 4:5].unsqueeze(1) if u_t.size(0) == B else u_t[0, 4].view(1, 1, 1)
            da_t = u_t[:, 5:6].unsqueeze(1) if u_t.size(0) == B else u_t[0, 5].view(1, 1, 1)
        else:
            curiosity_t = u_t[..., 0:1]
            stability_t = u_t[..., 2:3]
            na_t = u_t[..., 4:5]
            da_t = u_t[..., 5:6]

        # Dynamic Decay & Write Gain (100% differentiable)
        lambda_decay = torch.clamp(0.85 + 0.12 * stability_t - 0.08 * curiosity_t + 0.05 * da_t, 0.70, 0.98)
        eta = 0.10 * (1.0 + 2.0 * na_t + 1.2 * curiosity_t)
        
        if S > 1:
            idx = torch.arange(S, device=h_seq.device)
            decay_powers = (idx.unsqueeze(1) - idx.unsqueeze(0)).unsqueeze(0)  # [1, S, S]
            decay_powers = torch.clamp(decay_powers, min=0.0)
            
            # Differentiable per-element exponentiation: lambda^power = exp(power * log(lambda))
            log_lambda = torch.log(torch.clamp(lambda_decay, min=1e-5, max=0.999))  # [B, 1, 1] or [B, S, 1]
            decay_mask = torch.exp(decay_powers * log_lambda)  # [B, S, S]
            causal_decay_mask = decay_mask * torch.tril(torch.ones(S, S, device=h_seq.device)).unsqueeze(0)
            
            attn_sim = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.key_dim)
            attn_decayed = attn_sim * causal_decay_mask * eta
            attn_decayed = torch.clamp(attn_decayed, min=-10.0, max=10.0)
            y_fast = torch.bmm(attn_decayed, V)
        else:
            attn_sim = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.key_dim)
            attn_decayed = torch.clamp(attn_sim * eta, min=-10.0, max=10.0)
            y_fast = torch.bmm(attn_decayed, V)
            
        out = self.out_proj(y_fast)
        return out.squeeze(1) if is_2d else out


def evaluate_model(brain, entity, text_samples, num_steps=20, lr=1e-3):
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
    logger.info("🔬 [STARTING EXP-180: DIFFERENTIABLE FAST-WEIGHT HEBBIAN BENCHMARK]")
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

    # 1. Baseline Evaluation (Standard FastWeightHebbianPlasticity with .item())
    logger.info("\n--- 1. Evaluating Baseline (Standard Fast-Weight Hebbian with .item() Sync) ---")
    entity_b = KaryonEntity.load(filepath="karyon_soul.kcore", device=device_str)
    brain_b = entity_b.brain

    b_results = evaluate_model(brain_b, entity_b, text_samples, num_steps=20)
    logger.info(f"Baseline -> Init Loss: {b_results['init_loss']:.4f} | Final Loss: {b_results['final_loss']:.4f} (Delta: {b_results['loss_delta']:.4f})")
    logger.info(f"Baseline -> Init FE  : {b_results['init_fe']:.6f} | Final FE  : {b_results['final_fe']:.6f} (FE Delta: {b_results['fe_delta']:.6f})")

    # 2. Proposed Evaluation (Proposed Differentiable Fast-Weight Hebbian)
    logger.info("\n--- 2. Evaluating Proposed (Differentiable Fast-Weight Hebbian, Zero-Sync) ---")
    entity_p = KaryonEntity.load(filepath="karyon_soul.kcore", device=device_str)
    brain_p = entity_p.brain

    proposed_hebbian = ProposedFastWeightHebbianPlasticity(
        hidden_dim=brain_p.hidden_dim,
        device_str=device_str
    )

    # Copy pretrained weights
    with torch.no_grad():
        proposed_hebbian.k_proj.weight.copy_(brain_p.fast_weight_hebbian.k_proj.weight)
        proposed_hebbian.v_proj.weight.copy_(brain_p.fast_weight_hebbian.v_proj.weight)
        proposed_hebbian.q_proj.weight.copy_(brain_p.fast_weight_hebbian.q_proj.weight)
        proposed_hebbian.out_proj.weight.copy_(brain_p.fast_weight_hebbian.out_proj.weight)

    brain_p.fast_weight_hebbian = proposed_hebbian

    p_results = evaluate_model(brain_p, entity_p, text_samples, num_steps=20)
    logger.info(f"Proposed -> Init Loss: {p_results['init_loss']:.4f} | Final Loss: {p_results['final_loss']:.4f} (Delta: {p_results['loss_delta']:.4f})")
    logger.info(f"Proposed -> Init FE  : {p_results['init_fe']:.6f} | Final FE  : {p_results['final_fe']:.6f} (FE Delta: {p_results['fe_delta']:.6f})")

    # 3. Comparative Telemetry Analysis
    loss_improvement = b_results["final_loss"] - p_results["final_loss"]
    fe_reduction_pct = (b_results["final_fe"] - p_results["final_fe"]) / max(b_results["final_fe"], 1e-5) * 100
    speedup_pct = (b_results["elapsed_sec"] - p_results["elapsed_sec"]) / max(b_results["elapsed_sec"], 1e-5) * 100

    verdict = "POSITIVE" if (p_results["final_loss"] <= b_results["final_loss"] + 0.05 and p_results["final_fe"] <= b_results["final_fe"] and p_results["elapsed_sec"] < b_results["elapsed_sec"]) else "NEUTRAL"

    logger.info("=" * 80)
    logger.info("📊 === EXP-180 SCIENTIFIC TELEMETRY REPORT ===")
    logger.info(f"🏆 Final Verdict                 : 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    logger.info(f"📈 Baseline Final Loss           : {b_results['final_loss']:.4f} nats")
    logger.info(f"📈 Proposed Final Loss           : {p_results['final_loss']:.4f} nats (Delta: {loss_improvement:+.4f} nats)")
    logger.info(f"📉 Baseline Final Free Energy    : {b_results['final_fe']:.6f}")
    logger.info(f"📉 Proposed Final Free Energy    : {p_results['final_fe']:.6f} ({fe_reduction_pct:+.2f}% reduction)")
    logger.info(f"⏱️ Baseline Run Duration         : {b_results['elapsed_sec']:.2f}s")
    logger.info(f"⏱️ Proposed Run Duration         : {p_results['elapsed_sec']:.2f}s ({speedup_pct:+.2f}% faster)")
    logger.info("=" * 80)

    summary = {
        "exp_id": "EXP-180",
        "verdict": verdict,
        "hypothesis": "Differentiable Fast-Weight Hebbian Plasticity with element-wise exponentiation eliminates CPU-GPU sync stalls and enables backpropagation to somatic states.",
        "architecture_delta": "Replaced FastWeightHebbianPlasticity with ProposedFastWeightHebbianPlasticity using differentiable element-wise exponentiation.",
        "metrics": {
            "baseline_final_loss": b_results["final_loss"],
            "proposed_final_loss": p_results["final_loss"],
            "loss_improvement": loss_improvement,
            "baseline_final_fe": b_results["final_fe"],
            "proposed_final_fe": p_results["final_fe"],
            "fe_reduction_pct": fe_reduction_pct,
            "baseline_elapsed_sec": b_results["elapsed_sec"],
            "proposed_elapsed_sec": p_results["elapsed_sec"],
            "speedup_pct": speedup_pct
        }
    }

    with open("experiments/exp_180_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("✅ EXP-180 execution completed successfully. Results recorded in experiments/exp_180_results.json.")


if __name__ == "__main__":
    main()
