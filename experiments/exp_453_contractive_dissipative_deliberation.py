"""
EXP-453: Contractive Dissipative Deliberation (CDD).

Theoretical Formulation & Cybernetic Resolution:
In EXP-452, unbounded thinking without dissipative friction resulted in
ergodic drift / chaotic limit cycles on S^{D-1}, where average turns hit 31.24,
entropy exploded to 4.254 nats, and final loss degraded to 4.7582 nats.

In Classical Dynamical Systems and Banach Fixed-Point Theorem:
A discrete recurrence q_{k+1} = T(q_k) converges to a unique stable fixed point
if and only if T is a strict contraction mapping (Lipschitz constant L < 1), or
if the step dissipation satisfies Cauchy convergence:
    sum_{k=0}^infty ||Delta q_k|| < infty.

Architectural Design of EXP-453:
1. Contractive Dissipative Step Schedule:
   The feedback step amplitude scales exponentially with turn index k:
       g_k = g_0 * (gamma_damp)^k,   where gamma_damp = 0.55, g_0 = 0.35.
   This guarantees that:
       g_0 = 0.35, g_1 = 0.1925, g_2 = 0.1058, g_3 = 0.0582, g_4 = 0.0320, g_5 = 0.0176...
2. Dissipative Kinetic Stopping Criterion:
   Halting occurs naturally when the kinetic energy of the thought update
   drops below the physical dissipation noise floor:
       kinetic_impulse = ||g_k * feedback_k||_2 < epsilon_halt (0.015).
   Or when prediction certainty satisfies entropy threshold H <= tau_h.
3. System 1 Reflex Preservation:
   If initial prediction entropy H_0 <= tau_reflex (1.2 nats), zero additional
   thinking turns are executed (instant motor reflex).

Let us test whether contractive dissipative deliberation completely eradicates
the chaotic over-thinking trap while enabling sovereign variable-depth settling!
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
logger = logging.getLogger("EXP-453-CDD")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP453Config:
    exp_id: str = "EXP-453"
    dim: int = 258
    num_heads: int = 8
    dim_k: int = 64
    dim_v: int = 64
    g_0: float = 0.35                  # Initial feedback coupling strength
    gamma_damp: float = 0.55           # Contractive dissipation factor (Lipschitz damping)
    epsilon_halt: float = 0.012        # Kinetic impulse noise floor for halting
    entropy_reflex: float = 1.20       # System 1 threshold (reflex vs deliberation)
    entropy_satisfy: float = 0.35      # Epistemic satisfaction target
    max_safety_turns: int = 16         # Asymptotic safety ceiling
    learning_rate: float = 0.015
    stream_length: int = 4000
    device_str: str = DEVICE_STR


class ContractiveDissipativeMemory(nn.Module):
    """
    Sovereign Memory with Contractive Dissipative Deliberation (CDD).
    Strict Lyapunov contraction dynamics preventing chaotic over-thinking drift.
    """
    def __init__(self, config: EXP453Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.H = config.num_heads
        self.D_k = config.dim_k
        self.D_v = config.dim_v
        self.g_0 = config.g_0
        self.gamma = config.gamma_damp
        self.eps_halt = config.epsilon_halt
        self.tau_reflex = config.entropy_reflex
        self.tau_satisfy = config.entropy_satisfy
        self.k_safety = config.max_safety_turns

        # Byte Embedding
        self.byte_emb = nn.Embedding(self.dim, 256)

        # Projections
        self.W_q = nn.Linear(256, self.H * self.D_k, bias=False)
        self.W_k = nn.Linear(256, self.H * self.D_k, bias=False)
        self.W_v = nn.Linear(256, self.H * self.D_v, bias=False)

        # Dissipative Feedback Matrix (maps readout into query space)
        self.W_feedback = nn.Linear(self.D_v, self.D_k, bias=False)

        # Modern Hopfield Dynamic Precision Sharpening
        self.w_beta = nn.Parameter(torch.tensor(0.5))
        self.b_beta = nn.Parameter(torch.tensor(1.0))

        # Readout Projection
        self.W_out = nn.Linear(self.H * self.D_v, self.dim, bias=False)

        # Sovereign Multi-Head Law Synthesizer (Champion from EXP-441 / EXP-450)
        self.law_synthesizer = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 4)
        )

        # Memory State Tensor: S in R^(H x D_k x D_v)
        self.S_memory = torch.zeros(self.H, self.D_k, self.D_v, device=DEVICE)

        self.reset_state()

    def reset_state(self):
        self.S_memory.zero_()

    def step(self, byte_idx: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        x_tensor = torch.tensor([byte_idx], device=DEVICE)
        target_tensor = torch.tensor([target_byte], device=DEVICE)

        emb = self.byte_emb(x_tensor)  # [1, 256]

        q_init = F.normalize(self.W_q(emb).view(self.H, 1, self.D_k), p=2, dim=-1)
        k = F.normalize(self.W_k(emb).view(self.H, 1, self.D_k), p=2, dim=-1)
        v = F.normalize(self.W_v(emb).view(self.H, 1, self.D_v), p=2, dim=-1)

        S = self.S_memory.detach()  # [H, D_k, D_v]

        # -------------------------------------------------------------
        # 1. Turn 0: Instant Reflex (System 1)
        # -------------------------------------------------------------
        q_curr = q_init
        read_heads = torch.bmm(q_curr, S).squeeze(1)  # [H, D_v]
        read_flat = read_heads.view(1, -1)
        read_norm = torch.norm(read_flat)
        beta = 1.0 + F.softplus(self.w_beta * read_norm + self.b_beta)
        logits = beta * self.W_out(read_flat)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        turns_taken = 1

        # -------------------------------------------------------------
        # 2. Contractive Dissipative Deliberation (System 2):
        # Only triggered if surprise / uncertainty is high!
        # Step sizes diminish exponentially (g_step = g_0 * gamma^turn).
        # -------------------------------------------------------------
        if entropy > self.tau_reflex:
            for turn_idx in range(self.k_safety - 1):
                g_step = self.g_0 * (self.gamma ** turn_idx)

                # Feedback vector
                fb = torch.tanh(self.W_feedback(read_heads)).unsqueeze(1)  # [H, 1, D_k]
                impulse = g_step * fb

                # Check kinetic impulse floor (Cauchy convergence)
                impulse_norm = torch.norm(impulse).item()
                if impulse_norm < self.eps_halt:
                    # Dissipative settling complete: velocity dropped below threshold
                    break

                # Contractive update onto unit hypersphere
                q_curr = F.normalize(q_curr + impulse, p=2, dim=-1)

                # Re-read memory with settled query
                read_heads = torch.bmm(q_curr, S).squeeze(1)
                read_flat = read_heads.view(1, -1)
                read_norm = torch.norm(read_flat)
                beta = 1.0 + F.softplus(self.w_beta * read_norm + self.b_beta)
                logits = beta * self.W_out(read_flat)

                turns_taken += 1

                # Check epistemic satisfaction
                with torch.no_grad():
                    p_curr = F.softmax(logits, dim=-1)
                    curr_ent = -torch.sum(p_curr * torch.log(p_curr + 1e-9)).item()

                if curr_ent <= self.tau_satisfy:
                    break

        loss = F.cross_entropy(logits, target_tensor)
        pred = torch.argmax(logits, dim=-1).item()
        surprise = loss.item()

        # -------------------------------------------------------------
        # 3. Sovereign Endogenous Law Synthesis (EXP-441/450)
        # -------------------------------------------------------------
        v_pred = torch.bmm(k, S)
        error = v - v_pred

        err_norms = torch.norm(error, dim=-1).squeeze(1)
        s_norms = torch.norm(S, dim=(-1, -2))
        surp_ten = torch.full((self.H,), min(surprise / 3.0, 3.0), device=DEVICE)
        coher_ten = torch.sum(v * v_pred, dim=-1).squeeze(1)

        head_features = torch.stack([err_norms, s_norms / 10.0, surp_ten, coher_ten], dim=-1)
        law_params = self.law_synthesizer(head_features)

        alpha = torch.clamp(torch.sigmoid(law_params[:, 0]), min=0.70, max=0.999).view(self.H, 1, 1)
        eta_delta = (F.softplus(law_params[:, 1]) + 0.1).view(self.H, 1, 1)
        eta_hebb = F.softplus(law_params[:, 2]).view(self.H, 1, 1)
        lam_att = (0.05 * torch.sigmoid(law_params[:, 3])).view(self.H, 1, 1)

        delta_op = torch.bmm(k.transpose(1, 2), error)
        hebb_op = torch.bmm(k.transpose(1, 2), v)

        new_S = alpha * S + eta_delta * delta_op + eta_hebb * hebb_op - lam_att * S

        norm_s = torch.norm(new_S)
        if norm_s > 30.0:
            new_S = new_S * (30.0 / norm_s)

        self.S_memory = new_S

        meta = {
            "loss": loss.item(),
            "is_correct": pred == target_byte,
            "entropy": entropy,
            "turns_taken": turns_taken,
            "beta": beta.item(),
            "s_norm": torch.norm(new_S).item(),
            "mean_alpha": alpha.mean().item(),
            "mean_eta_delta": eta_delta.mean().item()
        }

        return loss, meta


def run_exp_453():
    logger.info("=" * 80)
    logger.info("=== STARTING EXP-453: CONTRACTIVE DISSIPATIVE DELIBERATION (CDD) ===")
    logger.info("=" * 80)

    config = EXP453Config()
    model = ContractiveDissipativeMemory(config).to(DEVICE)
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

    logger.info("Evaluating Contractive Dissipation (g_0=%.2f, gamma=%.2f, eps_halt=%.3f)...",
                config.g_0, config.gamma_damp, config.epsilon_halt)

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
            max_turns_window = max(turns_history[-500:])
            min_turns_window = min(turns_history[-500:])
            logger.info(
                "Step %4d/%4d | Loss: %.4f | Acc: %5.2f%% | Turns: Avg %.2f [Min %d, Max %d] | Beta: %.2f | S: %.2f | a: %.3f | eta_d: %.3f",
                t + 1, len(raw_bytes) - 1, avg_loss, acc, avg_turns, min_turns_window, max_turns_window,
                meta["beta"], meta["s_norm"], meta["mean_alpha"], meta["mean_eta_delta"]
            )

    t_elapsed = time.time() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_accuracy = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    throughput = (len(raw_bytes) - 1) / t_elapsed
    overall_avg_turns = sum(turns_history) / len(turns_history)
    max_turns_conducted = max(turns_history)
    min_turns_conducted = min(turns_history)

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
    logger.info("=== EXP-453 COMPLETE ===")
    logger.info("Final Loss               : %.4f nats", final_loss)
    logger.info("Final Accuracy           : %.2f%%", final_accuracy)
    logger.info("Average Thinking Turns   : %.2f turns/byte [Min: %d, Max: %d]",
                overall_avg_turns, min_turns_conducted, max_turns_conducted)
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
        "max_thinking_turns": int(max_turns_conducted),
        "min_thinking_turns": int(min_turns_conducted),
        "second_half_low_loss_mass": round(low_loss_mass, 2),
        "second_half_ultra_low_loss_mass": round(ultra_low_mass, 2),
        "second_half_near_zero_loss_mass": round(near_zero_mass, 2),
        "tail_retention_rate": round(tail_retention, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_453_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_453()
