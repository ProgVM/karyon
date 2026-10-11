"""
EXP-449: Multi-Turn Sovereign Memory Relaxation (MTR-SHC).

Theoretical Foundations & Cybernetic Rationale:
In EXP-448, we achieved 2.0145 nats and 54.86% accuracy with a single-step read/write.
However, Bazilevs correctly identified the fundamental limitation: "не хватает ходов"
(Insufficient Internal Thinking Cycles).

Single-step reading (k=1) commits to the first associative retrieval immediately.
If the query key has ambiguous or overlapping associations (n-gram collisions),
a single matrix-vector multiplication yields an averaged superposition of memories.

In Hopfield Networks and Recurrent Cortical Circuits:
Retrieval is a dynamical process governed by energy minimization:
    dΨ/dt = -grad E(Ψ)
By introducing Multi-Turn Sovereign Memory Relaxation (MTR, K=3 internal cycles):
1. Cycle 0: Ψ^{(0)} = q_ngram
   Read memory: r^{(0)} = [Ψ^{(0)} S_fast, Ψ^{(0)} S_slow]
2. Cycle k (k=1..K-1):
   Ψ^{(k)} = Norm(Ψ^{(k-1)} + \tanh(W_feedback * r^{(k-1)}))
   The query refines itself by conditioning on what was retrieved!
3. Final Cycle: Readout snaps cleanly to the sharpest energy attractor basin.

This provides the system with multi-step reasoning depth over its own internal memory
WITHOUT requiring token-level auto-regressive unrolling!
"""

import time
import json
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-449-MTR")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP449Config:
    exp_id: str = "EXP-449"
    dim: int = 258
    num_heads: int = 8
    dim_k: int = 64
    dim_v: int = 64
    ngram_order: int = 4
    thinking_turns: int = 3  # K=3 internal relaxation turns
    learning_rate: float = 0.015
    stream_length: int = 4000
    device_str: str = DEVICE_STR


class MultiTurnSovereignMemory(nn.Module):
    """
    Sovereign Dual-Timescale Holographic Memory with Multi-Turn Internal Relaxation.
    """
    def __init__(self, config: EXP449Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.H = config.num_heads
        self.D_k = config.dim_k
        self.D_v = config.dim_v
        self.L = config.ngram_order
        self.K = config.thinking_turns

        # Byte Embedding
        self.byte_emb = nn.Embedding(self.dim, 256)

        # Projections
        self.W_k = nn.Linear(256, self.H * self.D_k, bias=False)
        self.W_v = nn.Linear(256, self.H * self.D_v, bias=False)

        # Lie-Algebra Phase Generators for Multi-Order N-Gram binding
        self.lag_phases = nn.Parameter(torch.randn(self.L, self.H, self.D_k, self.D_k) * 0.05)
        self.lag_weights = nn.Parameter(torch.ones(self.L) / self.L)

        # Multi-Turn Internal Feedback Matrix: Maps retrieved values [H, 2*D_v] back to Query space [H, D_k]
        self.W_feedback = nn.Linear(2 * self.D_v, self.D_k, bias=False)
        self.feedback_gate = nn.Parameter(torch.tensor(0.2))

        # Dynamic Precision Sharpening
        self.w_beta = nn.Parameter(torch.tensor(0.5))
        self.b_beta = nn.Parameter(torch.tensor(1.0))

        # Output projection combining Fast & Slow heads
        self.W_out = nn.Linear(self.H * self.D_v * 2, self.dim, bias=False)

        # Sovereign Law Synthesizer for dual timescale plasticity
        self.law_synthesizer = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Linear(32, 6)  # [alpha_fast, alpha_slow, eta_fast, eta_slow, innov_gain, mix_gate]
        )

        # Dual-Timescale Persistent Memory Matrices:
        self.S_fast = torch.zeros(self.H, self.D_k, self.D_v, device=DEVICE)
        self.S_slow = torch.zeros(self.H, self.D_k, self.D_v, device=DEVICE)

        # History buffer of past keys
        self.k_history = [torch.zeros(self.H, 1, self.D_k, device=DEVICE) for _ in range(self.L)]

        self.reset_state()

    def reset_state(self):
        self.S_fast.zero_()
        self.S_slow.zero_()
        self.k_history = [torch.zeros(self.H, 1, self.D_k, device=DEVICE) for _ in range(self.L)]

    def step(self, byte_idx: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        x_tensor = torch.tensor([byte_idx], device=DEVICE)
        target_tensor = torch.tensor([target_byte], device=DEVICE)

        emb = self.byte_emb(x_tensor)  # [1, 256]

        curr_k = F.normalize(self.W_k(emb).view(self.H, 1, self.D_k), p=2, dim=-1)
        curr_v = F.normalize(self.W_v(emb).view(self.H, 1, self.D_v), p=2, dim=-1)

        # Update sliding history of key vectors:
        new_history = [curr_k] + [kh.detach() for kh in self.k_history[:-1]]
        self.k_history = new_history

        # Construct Continuous Multi-Order N-Gram Key Vector:
        w_l = F.softmax(self.lag_weights, dim=0)
        accum_k = torch.zeros(self.H, 1, self.D_k, device=DEVICE)

        for idx_lag in range(self.L):
            Lie_skew = (self.lag_phases[idx_lag] - self.lag_phases[idx_lag].transpose(1, 2)) * 0.5
            rot_k = torch.bmm(new_history[idx_lag], Lie_skew)
            accum_k = accum_k + w_l[idx_lag] * (new_history[idx_lag] + rot_k)

        k_ngram = F.normalize(accum_k, p=2, dim=-1)  # [H, 1, D_k]

        # Detach states for clean continuous recurrence
        S_f = self.S_fast.detach()
        S_s = self.S_slow.detach()

        # ---------------------------------------------------------
        # MULTI-TURN SOVEREIGN MEMORY RELAXATION (K Turns)
        # ---------------------------------------------------------
        q_curr = k_ngram  # Initial query [H, 1, D_k]
        read_combined = None

        for turn in range(self.K):
            read_fast = torch.bmm(q_curr, S_f).squeeze(1)  # [H, D_v]
            read_slow = torch.bmm(q_curr, S_s).squeeze(1)  # [H, D_v]
            read_combined = torch.cat([read_fast, read_slow], dim=-1)  # [H, 2 * D_v]

            if turn < self.K - 1:
                # Recurrent thought update: Project retrieved contents back into query space
                feedback_delta = torch.tanh(self.W_feedback(read_combined)).unsqueeze(1)  # [H, 1, D_k]
                gate = torch.sigmoid(self.feedback_gate)
                q_curr = F.normalize(q_curr + gate * feedback_delta, p=2, dim=-1)

        # Final settled readout:
        read_flat = read_combined.view(1, -1)  # [1, H * 2 * D_v]

        # Modern Hopfield Dynamic Precision Snapping:
        read_norm = torch.norm(read_flat)
        beta = 1.0 + F.softplus(self.w_beta * read_norm + self.b_beta)
        logits = beta * self.W_out(read_flat)

        loss = F.cross_entropy(logits, target_tensor)
        pred = torch.argmax(logits, dim=-1).item()
        surprise = loss.item()

        # 3. Prediction Errors and Orthogonal Innovation:
        v_pred_fast = torch.bmm(k_ngram, S_f)
        v_pred_slow = torch.bmm(k_ngram, S_s)

        error_fast = curr_v - v_pred_fast
        error_slow = curr_v - v_pred_slow

        # Orthogonal Subspace Innovation:
        k_outer_proj = torch.bmm(k_ngram, torch.bmm(k_ngram.transpose(1, 2), curr_v))
        innov_v = curr_v - 0.5 * k_outer_proj

        # 4. Sovereign Dual-Timescale Law Synthesis:
        err_fast_norm = torch.norm(error_fast, dim=-1).squeeze(1)
        err_slow_norm = torch.norm(error_slow, dim=-1).squeeze(1)
        s_fast_norm = torch.norm(S_f, dim=(-1, -2))
        s_slow_norm = torch.norm(S_s, dim=(-1, -2))
        surp_ten = torch.full((self.H,), min(surprise / 3.0, 3.0), device=DEVICE)
        coher_ten = torch.sum(curr_v * v_pred_slow, dim=-1).squeeze(1)

        head_features = torch.stack([
            err_fast_norm,
            err_slow_norm,
            s_fast_norm / 10.0,
            s_slow_norm / 10.0,
            surp_ten,
            coher_ten
        ], dim=-1)

        law_params = self.law_synthesizer(head_features)

        alpha_fast = torch.clamp(torch.sigmoid(law_params[:, 0]), min=0.60, max=0.88).view(self.H, 1, 1)
        alpha_slow = torch.clamp(torch.sigmoid(law_params[:, 1]), min=0.92, max=0.999).view(self.H, 1, 1)

        eta_fast = (F.softplus(law_params[:, 2]) + 0.1).view(self.H, 1, 1)
        eta_slow = (F.softplus(law_params[:, 3]) + 0.05).view(self.H, 1, 1)
        innov_gain = torch.clamp(torch.sigmoid(law_params[:, 4]) * 1.5, min=0.1, max=1.5).view(self.H, 1, 1)

        # Delta & Hebbian operators for fast & slow tiers
        delta_fast = torch.bmm(k_ngram.transpose(1, 2), error_fast)
        hebb_fast = torch.bmm(k_ngram.transpose(1, 2), curr_v)

        delta_slow = torch.bmm(k_ngram.transpose(1, 2), error_slow)
        hebb_slow_innov = torch.bmm(k_ngram.transpose(1, 2), innov_v * innov_gain)

        new_S_fast = alpha_fast * S_f + eta_fast * (delta_fast + 0.5 * hebb_fast)
        new_S_slow = alpha_slow * S_s + eta_slow * (delta_slow + 0.8 * hebb_slow_innov)

        # Ensure bounded norms
        norm_f = torch.norm(new_S_fast)
        if norm_f > 30.0:
            new_S_fast = new_S_fast * (30.0 / norm_f)
        norm_s = torch.norm(new_S_slow)
        if norm_s > 30.0:
            new_S_slow = new_S_slow * (30.0 / norm_s)

        self.S_fast = new_S_fast
        self.S_slow = new_S_slow

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        meta = {
            "loss": loss.item(),
            "is_correct": pred == target_byte,
            "entropy": entropy,
            "beta": beta.item(),
            "s_fast_norm": torch.norm(new_S_fast).item(),
            "s_slow_norm": torch.norm(new_S_slow).item(),
            "mean_alpha_fast": alpha_fast.mean().item(),
            "mean_alpha_slow": alpha_slow.mean().item()
        }

        return loss, meta


def run_exp_449():
    logger.info("=" * 80)
    logger.info("=== STARTING EXP-449: MULTI-TURN SOVEREIGN MEMORY RELAXATION ===")
    logger.info("=" * 80)

    config = EXP449Config()
    model = MultiTurnSovereignMemory(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    stream_data = (
        "def karyon_sovereign_autopoiesis(stream):\n"
        "    super_operator = Liouvillian(phase_space)\n"
        "    return super_operator.synthesize()\n\n"
        "#include <torch/extension.h>\n"
        "void dynamic_kernel(float* x, float* h) {\n"
        "    // Tensor core phase alignment\n"
        "}\n\n"
        "HTTP/1.1 200 OK\r\nContent-Type: application/kcore\r\n\r\n"
        "Mind is substrate-independent continuous field dynamics.\n"
        "Ashby homeostatic ultrastability balances energy and surprise.\n"
    ).encode("utf-8")

    cycles = (config.stream_length // len(stream_data)) + 1
    raw_bytes = list(stream_data) * cycles
    raw_bytes = raw_bytes[:config.stream_length]

    period = len(stream_data)
    logger.info("Stream loaded: %d bytes | Period: %d bytes on %s", len(raw_bytes), period, config.device_str)

    model.reset_state()
    t_start = time.time()

    losses = []
    entropies = []
    correct_preds = 0
    total_loss = 0.0

    logger.info("Evaluating K=%d Thinking Turns Internal Relaxation...", config.thinking_turns)

    for t in range(len(raw_bytes) - 1):
        x_byte = raw_bytes[t]
        target_byte = raw_bytes[t + 1]

        optimizer.zero_grad()
        loss, meta = model.step(x_byte, target_byte)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if meta["is_correct"]:
            correct_preds += 1

        total_loss += meta["loss"]
        losses.append(meta["loss"])
        entropies.append(meta["entropy"])

        if (t + 1) % 500 == 0:
            avg_loss = total_loss / (t + 1)
            acc = (correct_preds / (t + 1)) * 100.0
            logger.info(
                "Step %4d/%4d | Loss: %.4f | Acc: %5.2f%% | Beta: %.2f | S_fast: %.2f | S_slow: %.2f | "
                "a_fast: %.3f | a_slow: %.3f",
                t + 1, len(raw_bytes) - 1, avg_loss, acc,
                meta["beta"], meta["s_fast_norm"], meta["s_slow_norm"],
                meta["mean_alpha_fast"], meta["mean_alpha_slow"]
            )

    t_elapsed = time.time() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_accuracy = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    throughput = (len(raw_bytes) - 1) / t_elapsed

    midpoint = len(losses) // 2
    second_half_losses = losses[midpoint:]
    low_loss_count = sum(1 for loss_val in second_half_losses if loss_val < 1.0)
    ultra_low_loss_count = sum(1 for loss_val in second_half_losses if loss_val < 0.2)
    near_zero_loss_count = sum(1 for loss_val in second_half_losses if loss_val < 0.05)

    low_loss_mass = (low_loss_count / len(second_half_losses)) * 100.0
    ultra_low_mass = (ultra_low_loss_count / len(second_half_losses)) * 100.0
    near_zero_mass = (near_zero_loss_count / len(second_half_losses)) * 100.0

    tail_slice = losses[-period:]
    tail_retention = (sum(1 for loss_val in tail_slice if loss_val < 0.2) / len(tail_slice)) * 100.0
    avg_final_entropy = sum(entropies[-500:]) / 500.0

    logger.info("=" * 80)
    logger.info("=== EXP-449 COMPLETE ===")
    logger.info("Final Loss               : %.4f nats", final_loss)
    logger.info("Final Accuracy           : %.2f%%", final_accuracy)
    logger.info("Throughput               : %.2f steps/sec", throughput)
    logger.info("2nd Half Low Loss (<1.0) : %.2f%%", low_loss_mass)
    logger.info("2nd Half Ultra Low (<0.2): %.2f%%", ultra_low_mass)
    logger.info("2nd Half Near Zero(<0.05): %.2f%%", near_zero_mass)
    logger.info("Tail Period Retention    : %.2f%%", tail_retention)
    logger.info("Mean Final Entropy       : %.3f nats", avg_final_entropy)
    logger.info("Elapsed Time             : %.2f s", t_elapsed)
    logger.info("=" * 80)

    results = {
        "exp_id": config.exp_id,
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_accuracy, 2),
        "second_half_low_loss_mass": round(low_loss_mass, 2),
        "second_half_ultra_low_loss_mass": round(ultra_low_mass, 2),
        "second_half_near_zero_loss_mass": round(near_zero_mass, 2),
        "tail_retention_rate": round(tail_retention, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_449_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_449()
