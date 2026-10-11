"""
EXP-450: Surprise-Gated Adaptive Sovereign Thinking Turns (SGA-STT).

Theoretical Genesis & Cybernetic Axiom:
In EXP-449, applying uniform K=3 thinking turns to EVERY byte caused over-smoothing
on simple deterministic transitions (final loss 2.2225 vs 2.0145).
Why? Because when the transition is obvious (e.g. 'e' after 'th'), forcing multiple
feedback turns perturbs an already clean attractor!

In Active Inference and Karl Friston's Free Energy Principle:
Cognitive processing is strictly dynamic:
- Under LOW surprise (F_t < tau): The system executes an instantaneous fast reflex (K=1).
- Under HIGH surprise (F_t >= tau): The system awakens internal deliberation (System 2, K=2..4)
  allowing the query to iteratively recurse and settle until energy dissipation Delta E < epsilon.

Architecture of EXP-450:
1. Fast Feedforward Readout (Turn 0):
   q^{(0)} = q_ngram. Instant readout r^{(0)}.
   Calculate initial prediction entropy / surprise proxy:
   H_0 = -sum(p * log p).
2. Endogenous Gate for Deliberation:
   If H_0 > tau_surprise:
       Execute Adaptive Recurrent Settling:
       q^{(k+1)} = Norm(q^{(k)} + tanh(W_fb * r^{(k)}))
       until entropy drops below threshold or max_turns K_max reached!
3. Synthesized Plasticity of EXP-441:
   Combines the proven champion tensor law synthesizer of EXP-441 (1.4976 nats, 68.54% acc)
   with surprise-gated deliberative thinking turns.

Let's test this breakthrough hypothesis!
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
logger = logging.getLogger("EXP-450-SGA")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP450Config:
    exp_id: str = "EXP-450"
    dim: int = 258
    num_heads: int = 8
    dim_k: int = 64
    dim_v: int = 64
    max_thinking_turns: int = 4
    surprise_entropy_threshold: float = 1.2  # Threshold nats to trigger thinking turns
    learning_rate: float = 0.015
    stream_length: int = 4000
    device_str: str = DEVICE_STR


class SurpriseGatedThinkingMemory(nn.Module):
    """
    Sovereign Memory with Surprise-Gated Adaptive Thinking Turns (Active Inference System 1/System 2).
    """
    def __init__(self, config: EXP450Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.H = config.num_heads
        self.D_k = config.dim_k
        self.D_v = config.dim_v
        self.K_max = config.max_thinking_turns
        self.tau_h = config.surprise_entropy_threshold

        # Byte Embedding
        self.byte_emb = nn.Embedding(self.dim, 256)

        # Projections
        self.W_q = nn.Linear(256, self.H * self.D_k, bias=False)
        self.W_k = nn.Linear(256, self.H * self.D_k, bias=False)
        self.W_v = nn.Linear(256, self.H * self.D_v, bias=False)

        # Deliberative Feedback Network: maps retrieved readout [H, D_v] into query adjustment [H, D_k]
        self.W_feedback = nn.Linear(self.D_v, self.D_k, bias=False)
        self.feedback_gate = nn.Parameter(torch.tensor(0.2))

        # Precision Sharpening
        self.w_beta = nn.Parameter(torch.tensor(0.5))
        self.b_beta = nn.Parameter(torch.tensor(1.0))

        # Readout Projection
        self.W_out = nn.Linear(self.H * self.D_v, self.dim, bias=False)

        # Sovereign Multi-Head Law Synthesizer (Champion from EXP-441)
        self.law_synthesizer = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 4)
        )

        # Multi-Head Memory State Tensor: S in R^(H x D_k x D_v)
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
        # 1. Turn 0 (System 1 - Fast Reflexive Readout):
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
        # 2. Surprise-Gated Deliberation (System 2 Thinking Turns):
        # Only triggered if initial prediction is uncertain/surprising!
        # -------------------------------------------------------------
        if entropy > self.tau_h:
            gate = torch.sigmoid(self.feedback_gate)
            for _ in range(self.K_max - 1):
                feedback = torch.tanh(self.W_feedback(read_heads)).unsqueeze(1)  # [H, 1, D_k]
                q_curr = F.normalize(q_curr + gate * feedback, p=2, dim=-1)

                # Re-read memory with refined query:
                read_heads = torch.bmm(q_curr, S).squeeze(1)
                read_flat = read_heads.view(1, -1)
                read_norm = torch.norm(read_flat)
                beta = 1.0 + F.softplus(self.w_beta * read_norm + self.b_beta)
                logits = beta * self.W_out(read_flat)

                turns_taken += 1

                # Check if energy/uncertainty settled:
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    curr_entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
                if curr_entropy < self.tau_h * 0.8:
                    break

        loss = F.cross_entropy(logits, target_tensor)
        pred = torch.argmax(logits, dim=-1).item()
        surprise = loss.item()

        # -------------------------------------------------------------
        # 3. Sovereign Endogenous Law Synthesis (EXP-441 Engine):
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


def run_exp_450():
    logger.info("=" * 80)
    logger.info("=== STARTING EXP-450: SURPRISE-GATED ADAPTIVE THINKING TURNS ===")
    logger.info("=" * 80)

    config = EXP450Config()
    model = SurpriseGatedThinkingMemory(config).to(DEVICE)
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

    logger.info("Evaluating Adaptive Thinking Turns (Threshold H > %.2f, Max Turns: %d)...",
                config.surprise_entropy_threshold, config.max_thinking_turns)

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
                "Step %4d/%4d | Loss: %.4f | Acc: %5.2f%% | Turns: %.2f | Beta: %.2f | S_norm: %.2f | alpha: %.3f | eta_d: %.3f",
                t + 1, len(raw_bytes) - 1, avg_loss, acc, avg_turns,
                meta["beta"], meta["s_norm"], meta["mean_alpha"], meta["mean_eta_delta"]
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
    logger.info("=== EXP-450 COMPLETE ===")
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

    with open("experiments/exp_450_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_450()
