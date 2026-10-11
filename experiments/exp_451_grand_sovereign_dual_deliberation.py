"""
EXP-451: Sovereign Holographic Continuous Dual-Stage Deliberation (SHC-CDSD).

Theoretical & Physical Convergence:
In EXP-448:
- Dual-Timescale Holographic Memory (Fast transient + Deep crystal) with Orthogonal Subspace Innovation
  proved to hold massive associative capacity, achieving 70.66% tail retention and 42% near-zero loss.
In EXP-450:
- Bazilevs' insight on "не хватает ходов" was validated when gated by surprise:
  Adaptive System 1 / System 2 thinking turns dropped loss to 1.6947 and boosted near-zero mass to 51.85%!

The Ultimate Grand Synthesis (EXP-451):
What if we unite:
1. Dual-Timescale Hologram (S_fast for high-frequency transients, S_slow for permanent invariants)
2. Gram-Schmidt Orthogonal Subspace Innovation (storing pure novelty in S_slow)
3. Multi-Order Continuous Lie-Algebra N-gram Key Construction
4. Adaptive Surprise-Gated Deliberation (System 2 thinking turns triggered when initial readout entropy H > tau)
5. Sovereign Dynamic Tensor Law Synthesis (endogenous alpha, eta, lambda derivation per head)

Predicted Outcome:
Shatter the 1.49 nats barrier, pushing final loss towards ~1.20-1.30 nats,
exceeding 75% accuracy, and driving near-zero loss mass (>60%) while sustaining >150 steps/s!
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
logger = logging.getLogger("EXP-451-SHC-CDSD")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP451Config:
    exp_id: str = "EXP-451"
    dim: int = 258
    num_heads: int = 8
    dim_k: int = 64
    dim_v: int = 64
    ngram_order: int = 4
    max_thinking_turns: int = 4
    surprise_entropy_threshold: float = 1.0  # Slightly crisper threshold for System 2
    learning_rate: float = 0.015
    stream_length: int = 4000
    device_str: str = DEVICE_STR


class GrandSovereignHolographicMemory(nn.Module):
    """
    Unified Sovereign Architecture: Dual-Timescale Orthogonal Hologram + Adaptive Deliberation.
    """
    def __init__(self, config: EXP451Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.H = config.num_heads
        self.D_k = config.dim_k
        self.D_v = config.dim_v
        self.L = config.ngram_order
        self.K_max = config.max_thinking_turns
        self.tau_h = config.surprise_entropy_threshold

        # Byte Embedding
        self.byte_emb = nn.Embedding(self.dim, 256)

        # Projections
        self.W_k = nn.Linear(256, self.H * self.D_k, bias=False)
        self.W_v = nn.Linear(256, self.H * self.D_v, bias=False)

        # Lie-Algebra Phase Generators for Multi-Order N-Gram binding
        self.lag_phases = nn.Parameter(torch.randn(self.L, self.H, self.D_k, self.D_k) * 0.05)
        self.lag_weights = nn.Parameter(torch.ones(self.L) / self.L)

        # Deliberative Feedback Matrix (maps 2*D_v back into D_k query space)
        self.W_feedback = nn.Linear(2 * self.D_v, self.D_k, bias=False)
        self.feedback_gate = nn.Parameter(torch.tensor(0.25))

        # Modern Hopfield Dynamic Precision Sharpening
        self.w_beta = nn.Parameter(torch.tensor(0.5))
        self.b_beta = nn.Parameter(torch.tensor(1.0))

        # Output projection combining Fast & Slow heads
        self.W_out = nn.Linear(self.H * self.D_v * 2, self.dim, bias=False)

        # Sovereign Multi-Head Law Synthesizer
        self.law_synthesizer = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Linear(32, 6)  # [alpha_fast, alpha_slow, eta_fast, eta_slow, innov_gain, decay_slow]
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

        # -------------------------------------------------------------
        # System 1: Fast Immediate Holographic Readout
        # -------------------------------------------------------------
        q_curr = k_ngram
        read_fast = torch.bmm(q_curr, S_f).squeeze(1)  # [H, D_v]
        read_slow = torch.bmm(q_curr, S_s).squeeze(1)  # [H, D_v]
        read_combined = torch.cat([read_fast, read_slow], dim=-1)

        read_flat = read_combined.view(1, -1)
        read_norm = torch.norm(read_flat)
        beta = 1.0 + F.softplus(self.w_beta * read_norm + self.b_beta)
        logits = beta * self.W_out(read_flat)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        turns_taken = 1

        # -------------------------------------------------------------
        # System 2: Surprise-Gated Deliberation Turns
        # -------------------------------------------------------------
        if entropy > self.tau_h:
            gate = torch.sigmoid(self.feedback_gate)
            for _ in range(self.K_max - 1):
                feedback = torch.tanh(self.W_feedback(read_combined)).unsqueeze(1)  # [H, 1, D_k]
                q_curr = F.normalize(q_curr + gate * feedback, p=2, dim=-1)

                read_fast = torch.bmm(q_curr, S_f).squeeze(1)
                read_slow = torch.bmm(q_curr, S_s).squeeze(1)
                read_combined = torch.cat([read_fast, read_slow], dim=-1)

                read_flat = read_combined.view(1, -1)
                read_norm = torch.norm(read_flat)
                beta = 1.0 + F.softplus(self.w_beta * read_norm + self.b_beta)
                logits = beta * self.W_out(read_flat)

                turns_taken += 1

                with torch.no_grad():
                    p_curr = F.softmax(logits, dim=-1)
                    curr_ent = -torch.sum(p_curr * torch.log(p_curr + 1e-9)).item()
                if curr_ent < self.tau_h * 0.7:
                    break

        loss = F.cross_entropy(logits, target_tensor)
        pred = torch.argmax(logits, dim=-1).item()
        surprise = loss.item()

        # -------------------------------------------------------------
        # 3. Orthogonal Subspace Innovation & Error Dynamics
        # -------------------------------------------------------------
        v_pred_fast = torch.bmm(k_ngram, S_f)
        v_pred_slow = torch.bmm(k_ngram, S_s)

        error_fast = curr_v - v_pred_fast
        error_slow = curr_v - v_pred_slow

        # Gram-Schmidt Innovation: Orthogonalize w.r.t current state
        k_outer_proj = torch.bmm(k_ngram, torch.bmm(k_ngram.transpose(1, 2), curr_v))
        innov_v = curr_v - 0.5 * k_outer_proj

        # 4. Sovereign Endogenous Law Synthesis
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
        decay_slow = (0.01 * torch.sigmoid(law_params[:, 5])).view(self.H, 1, 1)

        # Delta & Hebbian operators
        delta_fast = torch.bmm(k_ngram.transpose(1, 2), error_fast)
        hebb_fast = torch.bmm(k_ngram.transpose(1, 2), curr_v)

        delta_slow = torch.bmm(k_ngram.transpose(1, 2), error_slow)
        hebb_slow_innov = torch.bmm(k_ngram.transpose(1, 2), innov_v * innov_gain)

        new_S_fast = alpha_fast * S_f + eta_fast * (delta_fast + 0.5 * hebb_fast)
        new_S_slow = (alpha_slow - decay_slow) * S_s + eta_slow * (delta_slow + 0.8 * hebb_slow_innov)

        norm_f = torch.norm(new_S_fast)
        if norm_f > 30.0:
            new_S_fast = new_S_fast * (30.0 / norm_f)
        norm_s = torch.norm(new_S_slow)
        if norm_s > 30.0:
            new_S_slow = new_S_slow * (30.0 / norm_s)

        self.S_fast = new_S_fast
        self.S_slow = new_S_slow

        meta = {
            "loss": loss.item(),
            "is_correct": pred == target_byte,
            "entropy": entropy,
            "turns_taken": turns_taken,
            "beta": beta.item(),
            "s_fast_norm": torch.norm(new_S_fast).item(),
            "s_slow_norm": torch.norm(new_S_slow).item(),
            "mean_alpha_fast": alpha_fast.mean().item(),
            "mean_alpha_slow": alpha_slow.mean().item()
        }

        return loss, meta


def run_exp_451():
    logger.info("=" * 80)
    logger.info("=== STARTING EXP-451: GRAND SOVEREIGN HOLOGRAPHIC DUAL-STAGE DELIBERATION ===")
    logger.info("=" * 80)

    config = EXP451Config()
    model = GrandSovereignHolographicMemory(config).to(DEVICE)
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
    turns_history = []
    correct_preds = 0
    total_loss = 0.0

    logger.info("Evaluating Grand Sovereign Architecture...")

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
        turns_history.append(meta["turns_taken"])

        if (t + 1) % 500 == 0:
            avg_loss = total_loss / (t + 1)
            acc = (correct_preds / (t + 1)) * 100.0
            avg_turns = sum(turns_history[-500:]) / 500.0
            logger.info(
                "Step %4d/%4d | Loss: %.4f | Acc: %5.2f%% | Turns: %.2f | Beta: %.2f | S_f: %.2f | S_s: %.2f | a_f: %.3f | a_s: %.3f",
                t + 1, len(raw_bytes) - 1, avg_loss, acc, avg_turns,
                meta["beta"], meta["s_fast_norm"], meta["s_slow_norm"],
                meta["mean_alpha_fast"], meta["mean_alpha_slow"]
            )

    t_elapsed = time.time() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_accuracy = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    throughput = (len(raw_bytes) - 1) / t_elapsed
    overall_avg_turns = sum(turns_history) / len(turns_history)

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
    logger.info("=== EXP-451 COMPLETE ===")
    logger.info("Final Loss               : %.4f nats", final_loss)
    logger.info("Final Accuracy           : %.2f%%", final_accuracy)
    logger.info("Average Thinking Turns   : %.2f turns/byte", overall_avg_turns)
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
        "average_thinking_turns": round(overall_avg_turns, 2),
        "second_half_low_loss_mass": round(low_loss_mass, 2),
        "second_half_ultra_low_loss_mass": round(ultra_low_mass, 2),
        "second_half_near_zero_loss_mass": round(near_zero_mass, 2),
        "tail_retention_rate": round(tail_retention, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_451_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_451()
