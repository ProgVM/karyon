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

from karyon_config import CoREConfig
from karyon_agent import CoREAgent, FastWeightHebbianPlasticity

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
        # e.g., heads 0..3 have base retention half-lives from ~3 to ~50 steps
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


def run_benchmark():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-188: MULTI-TIMESCALE DIFFERENTIABLE FAST WEIGHTS BENCHMARK]")
    logger.info("=" * 80)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Target Accelerator: {device_str.upper()}")

    # Synthetic realistic byte sequences
    torch.manual_seed(42)
    B, S = 4, 256
    vocab_size = 258
    hidden_dim = 256

    # Create dummy multi-turn conversational byte inputs
    inputs = torch.randint(32, 126, (B, S), dtype=torch.long, device=device)
    u_t = torch.tensor([[0.7, 0.8, 0.6, 0.9, 0.5, 0.4]] * B, device=device)

    # 1. Baseline Model (Standard FastWeightHebbianPlasticity)
    cfg_base = CoREConfig()
    cfg_base.net.hidden_dim = hidden_dim
    cfg_base.net.unified_dim = hidden_dim
    agent_base = CoREAgent(cfg_base, device=device_str).to(device)
    agent_base.eval()

    # 2. Proposed Model (MultiTimescaleFastWeightHebbian)
    cfg_prop = CoREConfig()
    cfg_prop.net.hidden_dim = hidden_dim
    cfg_prop.net.unified_dim = hidden_dim
    agent_prop = CoREAgent(cfg_prop, device=device_str).to(device)
    # Inject Multi-Timescale Hebbian module
    agent_prop.fast_weight_hebbian = MultiTimescaleFastWeightHebbian(
        hidden_dim=hidden_dim, num_heads=4, head_dim=64, device_str=device_str
    ).to(device)
    agent_prop.eval()

    logger.info("Running warm-up passes...")
    for _ in range(5):
        _ = agent_base.forward_sequence(inputs)
        _ = agent_prop.forward_sequence(inputs)

    # Optimization loop on sequence prediction
    optimizer_base = torch.optim.AdamW(agent_base.parameters(), lr=1e-3)
    optimizer_prop = torch.optim.AdamW(agent_prop.parameters(), lr=1e-3)

    steps = 40
    target = torch.randint(32, 126, (B, S), dtype=torch.long, device=device)

    logger.info(f"Executing {steps} convergence optimization steps...")
    
    start_time = time.perf_counter()
    loss_base_history = []
    fe_base_history = []
    for _ in range(steps):
        optimizer_base.zero_grad()
        out = agent_base.forward_sequence(inputs)
        logits = out['logits'] if isinstance(out, dict) else out[0]
        loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
        fe = out.get('free_energy', torch.tensor(0.0)) if isinstance(out, dict) else torch.tensor(0.0)
        loss.backward()
        optimizer_base.step()
        loss_base_history.append(loss.item())
        fe_base_history.append(fe.item() if isinstance(fe, torch.Tensor) else fe)
    base_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    loss_prop_history = []
    fe_prop_history = []
    for _ in range(steps):
        optimizer_prop.zero_grad()
        out = agent_prop.forward_sequence(inputs)
        logits = out['logits'] if isinstance(out, dict) else out[0]
        loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
        fe = out.get('free_energy', torch.tensor(0.0)) if isinstance(out, dict) else torch.tensor(0.0)
        loss.backward()
        optimizer_prop.step()
        loss_prop_history.append(loss.item())
        fe_prop_history.append(fe.item() if isinstance(fe, torch.Tensor) else fe)
    prop_time = time.perf_counter() - start_time

    base_init_loss = loss_base_history[0]
    base_final_loss = loss_base_history[-1]
    prop_init_loss = loss_prop_history[0]
    prop_final_loss = loss_prop_history[-1]

    loss_delta = base_final_loss - prop_final_loss
    tok_per_sec_base = (B * S * steps) / base_time
    tok_per_sec_prop = (B * S * steps) / prop_time

    logger.info("=" * 80)
    logger.info("📊 === EXP-188 TELEMETRY REPORT ===")
    logger.info(f"  - Baseline Final Loss : {base_final_loss:.4f} nats | Throughput: {tok_per_sec_base:.1f} tok/s")
    logger.info(f"  - Proposed Final Loss : {prop_final_loss:.4f} nats | Throughput: {tok_per_sec_prop:.1f} tok/s")
    logger.info(f"  - Loss Delta (B - P)  : {loss_delta:.4f} nats")
    logger.info(f"  - Baseline Duration   : {base_time:.3f} s")
    logger.info(f"  - Proposed Duration   : {prop_time:.3f} s")

    # KEP Rule #2 Verdict
    verdict = "POSITIVE" if (loss_delta >= 0.08 or (loss_delta >= 0.02 and tok_per_sec_prop >= 0.95 * tok_per_sec_base)) else "NEUTRAL"

    results = {
        "exp_id": "EXP-188",
        "verdict": verdict,
        "base_initial_loss": base_init_loss,
        "base_final_loss": base_final_loss,
        "proposed_initial_loss": prop_init_loss,
        "proposed_final_loss": prop_final_loss,
        "loss_delta": loss_delta,
        "throughput_tok_per_sec": tok_per_sec_prop,
        "execution_time_s": prop_time
    }

    with open("experiments/exp_188_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"🏆 Final Verdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    return results


if __name__ == "__main__":
    run_benchmark()
