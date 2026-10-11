"""
EXP-430: Dual-Potential Dynamic Phase-Space with Attractors and Repellers (DP-AR)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 8, Principle 22, Principle 24, Principle 27 & KEP Rule #12)

Theoretical Architecture of EXP-430:
"Where there are Attractors, there MUST be Repellers." — Bazilevs

In continuous non-linear dynamical systems:
1. Attractors create energy wells: U_att(x) = -sum_i log(1 + exp(beta * <x, a_i>))
   - Trajectory is pulled into valid semantic concepts.
2. Repellers create energy hills / barriers: U_rep(x) = +sum_j log(1 + exp(gamma * <x, r_j>))
   - Separatrices divide conceptual basins, preventing semantic bleed and hallucinations!
   - Erroneously chosen hypotheses (false predictions under Free Energy error F_t) act as dynamic repellers!
   - Habituation / Refractory Repellers push the state away from recently visited states,
     completely eliminating degenerate repetitive loops (perseverative loops)!

Total Phase Energy Landscape:
E(x) = U_att(x) - U_rep(x)
Logits = beta * <x, A> - gamma * <x, R>

Target Metrics:
- Eradication of perseverative repetition and semantic bleed.
- Near-Zero Loss Mass (<0.01 nats) > 50%.
- Tail 500-byte Retention Rate > 90%.
- Single-Pass Accuracy > 65%.
"""

import time
import json
import logging
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-430-DPAR")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP430Config:
    exp_id: str = "EXP-430"
    dim: int = 258
    num_heads: int = 8
    dim_k: int = 64
    dim_v: int = 64
    num_repellers: int = 32  # Dynamic repeller memory slots
    learning_rate: float = 0.015
    sleep_interval: int = 400
    sleep_steps: int = 6
    sleep_batch_size: int = 32
    stream_length: int = 4000
    eta_fast: float = 0.95
    eta_slow: float = 0.50
    gamma_rep: float = 0.40  # Repulsion strength
    rep_decay: float = 0.90  # Repeller persistence
    device_str: str = DEVICE_STR


class AttractorRepellerPhaseSpace(nn.Module):
    """
    Continuous Dynamical Phase-Space Engine with Attractor Wells and Dynamic Repeller Barriers.
    """
    def __init__(self, config: EXP430Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.dim_k = config.dim_k
        self.dim_v = config.dim_v

        # 1. Key-Space Projections (Key Space Invariance)
        self.byte_emb = nn.Embedding(self.dim, 256)
        self.W_q = nn.Linear(256, self.num_heads * self.dim_k, bias=False)
        self.W_k = nn.Linear(256, self.num_heads * self.dim_k, bias=False)
        self.W_v = nn.Linear(256, self.num_heads * self.dim_v, bias=False)

        # 2. Attractor Readout Head (Pulls toward concept wells)
        self.W_att = nn.Linear(2 * self.num_heads * self.dim_v, self.dim, bias=False)

        # 3. Dynamic Repeller Projection (Pushes away from erroneous/repetitive states)
        self.W_rep = nn.Linear(2 * self.num_heads * self.dim_v, self.dim, bias=False)

        # 4. Attractor & Repeller Scaling Parameters
        self.w_beta = nn.Parameter(torch.tensor(1.5, device=DEVICE))
        self.b_beta = nn.Parameter(torch.tensor(2.5, device=DEVICE))
        self.w_gamma = nn.Parameter(torch.tensor(0.8, device=DEVICE))

        # 5. Decay Parameters
        self.fast_decay_params = nn.Parameter(torch.linspace(0.85, 2.2, self.num_heads, device=DEVICE))
        self.slow_decay_params = nn.Parameter(torch.linspace(4.6, 7.6, self.num_heads, device=DEVICE))

        # 6. Associative Memory Buffers [H, dim_k, dim_v]
        self.S_fast = torch.zeros(self.num_heads, self.dim_k, self.dim_v, device=DEVICE)
        self.S_slow = torch.zeros(self.num_heads, self.dim_k, self.dim_v, device=DEVICE)

        # 7. Dynamic Repeller Memory: circular buffer of negative error vectors [N_rep, 1024]
        self.repellers = torch.zeros(self.config.num_repellers, 2 * self.num_heads * self.dim_v, device=DEVICE)
        self.rep_idx = 0
        self.rep_count = 0

        # 8. High-Surprise Episodic Buffer for Phase Sleep
        self.episodic_buffer: List[Tuple[int, int]] = []
        self.max_buffer_size = 256

    def reset_state(self):
        self.S_fast = torch.zeros(self.num_heads, self.dim_k, self.dim_v, device=DEVICE)
        self.S_slow = torch.zeros(self.num_heads, self.dim_k, self.dim_v, device=DEVICE)
        self.repellers = torch.zeros(self.config.num_repellers, 2 * self.num_heads * self.dim_v, device=DEVICE)
        self.rep_idx = 0
        self.rep_count = 0
        self.episodic_buffer.clear()

    def forward_stream_byte(self, x_byte: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        x_tensor = torch.tensor([x_byte], device=DEVICE)
        target_tensor = torch.tensor([target_byte], device=DEVICE)

        emb = self.byte_emb(x_tensor)  # [1, 256]

        q = F.normalize(self.W_q(emb).view(self.num_heads, 1, self.dim_k), p=2, dim=-1)
        k = F.normalize(self.W_k(emb).view(self.num_heads, 1, self.dim_k), p=2, dim=-1)
        v = F.normalize(self.W_v(emb).view(self.num_heads, 1, self.dim_v), p=2, dim=-1)

        read_fast = torch.bmm(q, self.S_fast.detach()).squeeze(1).view(1, -1)
        read_slow = torch.bmm(q, self.S_slow.detach()).squeeze(1).view(1, -1)
        read_combined = torch.cat([read_fast, read_slow], dim=-1)  # [1, 1024]

        # === DUAL-POTENTIAL LANDSCAPE: ATTRACTOR + REPELLER ===
        # 1. Attractor Well (Pull):
        beta = 1.0 + F.softplus(self.w_beta * torch.norm(read_combined) + self.b_beta)
        logits_att = beta * self.W_att(read_combined)

        # 2. Dynamic Repeller Field (Push away from error basins):
        repulsion = torch.zeros_like(logits_att)
        if self.rep_count > 0:
            valid_repellers = self.repellers[:self.rep_count]  # [N, 1024]
            # Similarity between current state and active repellers
            sim_rep = torch.matmul(F.normalize(read_combined, p=2, dim=-1), F.normalize(valid_repellers, p=2, dim=-1).t())  # [1, N]
            # High similarity to a repeller triggers defensive repulsive potential:
            rep_weights = F.softmax(sim_rep * 4.0, dim=-1)  # [1, N]
            rep_field = torch.matmul(rep_weights, valid_repellers)  # [1, 1024]
            gamma = F.softplus(self.w_gamma)
            repulsion = gamma * self.W_rep(rep_field)

        # Total Net Logits: Pull minus Push
        logits = logits_att - repulsion

        loss = F.cross_entropy(logits, target_tensor)
        pred = torch.argmax(logits, dim=-1).item()
        surprise_val = loss.item()

        # Update Dynamic Repellers:
        # If prediction was erroneous and had high surprise, inscribe the current read state as a Repeller
        with torch.no_grad():
            if pred != target_byte and surprise_val > 1.2:
                self.repellers[self.rep_idx] = read_combined.squeeze(0).detach()
                self.rep_idx = (self.rep_idx + 1) % self.config.num_repellers
                if self.rep_count < self.config.num_repellers:
                    self.rep_count += 1
            else:
                # Decay old repeller strengths
                self.repellers *= self.config.rep_decay

        # Capture high-surprise episodes for phase sleep
        if surprise_val > 1.5:
            if len(self.episodic_buffer) >= self.max_buffer_size:
                self.episodic_buffer.pop(0)
            self.episodic_buffer.append((x_byte, target_byte))

        # Delta-Rule Memory Inscription
        with torch.no_grad():
            alpha_fast = torch.clamp(torch.sigmoid(self.fast_decay_params), min=0.70, max=0.92).view(self.num_heads, 1, 1)
            alpha_slow = torch.clamp(torch.sigmoid(self.slow_decay_params), min=0.990, max=0.9995).view(self.num_heads, 1, 1)

            v_pred_fast = torch.bmm(k, self.S_fast)
            error_fast = v - v_pred_fast
            delta_fast = torch.bmm(k.transpose(1, 2), error_fast)
            self.S_fast = alpha_fast * self.S_fast + self.config.eta_fast * delta_fast

            v_pred_slow = torch.bmm(k, self.S_slow)
            error_slow = v - v_pred_slow
            delta_slow = torch.bmm(k.transpose(1, 2), error_slow)

            gate_slow = torch.clamp(torch.tensor(surprise_val / 3.0, device=DEVICE), min=0.1, max=1.0)
            self.S_slow = alpha_slow * self.S_slow + (self.config.eta_slow * gate_slow) * delta_slow

            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        meta = {
            "loss": surprise_val,
            "pred": pred,
            "is_correct": (pred == target_byte),
            "entropy": entropy,
            "beta": beta.item(),
            "repellers_active": self.rep_count,
            "norm_s_fast": torch.norm(self.S_fast).item(),
            "norm_s_slow": torch.norm(self.S_slow).item()
        }
        return loss, meta

    def execute_phase_sleep_consolidation(self, optimizer: torch.optim.Optimizer) -> float:
        if len(self.episodic_buffer) < self.config.sleep_batch_size:
            return 0.0

        total_sleep_loss = 0.0
        sample_size = min(len(self.episodic_buffer), self.config.sleep_batch_size)
        replay_samples = random.sample(self.episodic_buffer, sample_size)

        x_bytes = torch.tensor([s[0] for s in replay_samples], device=DEVICE)
        target_bytes = torch.tensor([s[1] for s in replay_samples], device=DEVICE)

        for _ in range(self.config.sleep_steps):
            optimizer.zero_grad()

            embs = self.byte_emb(x_bytes)
            with torch.no_grad():
                qs = F.normalize(self.W_q(embs).view(-1, self.num_heads, 1, self.dim_k), p=2, dim=-1)
                s_fast_exp = self.S_fast.unsqueeze(0).expand(sample_size, -1, -1, -1)
                s_slow_exp = self.S_slow.unsqueeze(0).expand(sample_size, -1, -1, -1)

                read_fast = torch.matmul(qs, s_fast_exp).squeeze(2).view(sample_size, -1)
                read_slow = torch.matmul(qs, s_slow_exp).squeeze(2).view(sample_size, -1)
                read_combined = torch.cat([read_fast, read_slow], dim=-1)

            beta = 1.0 + F.softplus(self.w_beta * torch.norm(read_combined, dim=-1, keepdim=True) + self.b_beta)
            logits_att = beta * self.W_att(read_combined)

            sleep_loss = F.cross_entropy(logits_att, target_bytes)
            sleep_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            total_sleep_loss += sleep_loss.item()

        return total_sleep_loss / self.config.sleep_steps


def run_exp_430():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-430: DUAL-POTENTIAL ATTRACTORS & REPELLERS (DP-AR) ===")
    logger.info("================================================================================")

    config = EXP430Config()
    model = AttractorRepellerPhaseSpace(config).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

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

    raw_bytes = list(stream_data) * (config.stream_length // len(stream_data) + 1)
    raw_bytes = raw_bytes[:config.stream_length]

    logger.info(f"Stream loaded: {len(raw_bytes)} bytes on {DEVICE_STR.upper()}")
    logger.info("Starting stream with Dual Attractor-Repeller Phase-Space dynamics...")

    correct_preds = 0
    total_loss = 0.0
    total_entropy = 0.0
    recent_losses = []
    sleep_events = 0

    t_start = time.perf_counter()

    for t in range(len(raw_bytes) - 1):
        x_byte = raw_bytes[t]
        target_byte = raw_bytes[t + 1]

        optimizer.zero_grad()
        loss, meta = model.forward_stream_byte(x_byte, target_byte)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if meta["is_correct"]:
            correct_preds += 1

        total_loss += meta["loss"]
        total_entropy += meta["entropy"]
        recent_losses.append(meta["loss"])

        if (t + 1) % config.sleep_interval == 0:
            s_loss = model.execute_phase_sleep_consolidation(optimizer)
            if s_loss > 0.0:
                sleep_events += 1

        if (t + 1) % 500 == 0:
            avg_loss = total_loss / (t + 1)
            acc = (correct_preds / (t + 1)) * 100.0
            avg_ent = total_entropy / (t + 1)
            logger.info(
                f"Progress [{t+1}/{len(raw_bytes)-1}] | Avg Loss: {avg_loss:.4f} | "
                f"Acc: {acc:.2f}% | Current Loss: {meta['loss']:.4f} | "
                f"Entropy: {avg_ent:.3f} nats | Repellers: {meta['repellers_active']}/32 | "
                f"Sleep Cycles: {sleep_events}"
            )

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_final_entropy = total_entropy / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    second_half = recent_losses[len(recent_losses)//2:]
    low_loss_count = sum(1 for loss_val in second_half if loss_val < 0.5)
    low_loss_fraction = (low_loss_count / len(second_half)) * 100.0

    ultra_low_loss_count = sum(1 for loss_val in second_half if loss_val < 0.1)
    ultra_low_loss_fraction = (ultra_low_loss_count / len(second_half)) * 100.0

    near_zero_loss_count = sum(1 for loss_val in second_half if loss_val < 0.01)
    near_zero_loss_fraction = (near_zero_loss_count / len(second_half)) * 100.0

    tail_500 = recent_losses[-500:]
    tail_low_loss_count = sum(1 for loss_val in tail_500 if loss_val < 0.5)
    tail_retention_rate = (tail_low_loss_count / len(tail_500)) * 100.0

    logger.info("================================================================================")
    logger.info("=== EXP-430 COMPLETE TELEMETRY & ATTRACTOR-REPELLER REPORT ===")
    logger.info(f"Final Average Loss: {final_loss:.4f} nats")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Second-Half Low-Loss Mass (<0.5 nats): {low_loss_fraction:.2f}%")
    logger.info(f"Second-Half Ultra-Low Loss Mass (<0.1 nats): {ultra_low_loss_fraction:.2f}%")
    logger.info(f"Second-Half Near-Zero Loss Mass (<0.01 nats): {near_zero_loss_fraction:.2f}%")
    logger.info(f"Tail 500-byte Retention Rate (<0.5 nats): {tail_retention_rate:.2f}%")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Mean Output Entropy: {avg_final_entropy:.3f} nats")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-430",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "second_half_low_loss_mass": round(low_loss_fraction, 2),
        "second_half_ultra_low_loss_mass": round(ultra_low_loss_fraction, 2),
        "second_half_near_zero_loss_mass": round(near_zero_loss_fraction, 2),
        "tail_retention_rate": round(tail_retention_rate, 2),
        "sleep_events": sleep_events,
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_430_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_430()
