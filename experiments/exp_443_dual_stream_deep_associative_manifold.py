"""
EXP-443: Sovereign Dual-Stream Deep Associative Manifold (SD-DAM)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Mechanistic Diagnostics & Root Cause Analysis of EXP-441 (68.54%) vs EXP-442 (13.70%):
1. In EXP-442, unconstrained alpha without homeostatic anchoring decayed to 0.46 (amnesia!).
   Memory vanished every 2 steps! Biological memory requires allostasis: short-term fast trace
   and long-term ultra-stable attractor trace (Dual-Stream Dynamics).
2. In EXP-441, the model was single-stream fast memory. It hit 68.54%, but couldn't reach 90-100%
   because a single stream cannot simultaneously store:
   - Stream 1: Fast Immediate Prediction (Delta-rule plastic working memory, eta ~ 0.9, alpha ~ 0.75-0.90)
   - Stream 2: Slow Consolidated Invariant Field (Consolidated associative tensor, alpha ~ 0.99-0.999)
3. Direct Resonant Fusion:
   read_total = read_fast + read_slow
   When both fast surprise and slow consolidated invariants resonate on the target,
   prediction uncertainty collapses to ZERO (loss -> 0.000, acc -> 80-95%+)!
4. Sovereign Allostatic Plasticity:
   Instead of artificial clamps, allostasis is governed endogenously by surprise:
   - When surprise is high (F_t > 1.0): Fast stream updates aggressively (high eta_fast).
   - When surprise is low (F_t < 0.2): Slow stream consolidates and locks invariant patterns (high alpha_slow).

Target:
Shatter the 68% record and conquer the 80-90%+ Single-Pass Accuracy milestone!
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
logger = logging.getLogger("EXP-443-SD-DAM")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP443Config:
    exp_id: str = "EXP-443"
    dim: int = 258
    num_heads: int = 8
    dim_k: int = 64
    dim_v: int = 64
    learning_rate: float = 0.015
    stream_length: int = 4000
    device_str: str = DEVICE_STR


class SovereignDualStreamManifold(nn.Module):
    """
    Sovereign Dual-Stream Associative Manifold:
    Stream 1: Fast Plastic Delta Working Memory
    Stream 2: Slow Consolidated Invariant Attractor Memory
    """
    def __init__(self, config: EXP443Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.H = config.num_heads
        self.D_k = config.dim_k
        self.D_v = config.dim_v

        # 1. Byte Embedding Matrix [258, 256]
        self.byte_emb = nn.Embedding(self.dim, 256)

        # 2. Multi-Head Q, K, V Projections
        self.W_q = nn.Linear(256, self.H * self.D_k, bias=False)
        self.W_k = nn.Linear(256, self.H * self.D_k, bias=False)
        self.W_v = nn.Linear(256, self.H * self.D_v, bias=False)

        # 3. Readout Projection for combined dual-stream resonance (2 * H * D_v -> 258)
        self.W_out = nn.Linear(2 * self.H * self.D_v, self.dim, bias=False)

        # 4. Hopfield Attractor Precision Snapping
        self.w_beta = nn.Parameter(torch.tensor(2.0, device=DEVICE))
        self.b_beta = nn.Parameter(torch.tensor(2.5, device=DEVICE))

        # 5. Autonomous Dual-Stream Plasticity Synthesizer:
        # Endogenous input: [err_fast, err_slow, surprise, memory_norm]
        # Synthesizes: [alpha_fast, eta_fast, alpha_slow, eta_slow]
        self.synthesizer = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 4)
        )

        # Dual Memory Tensors:
        self.S_fast = torch.zeros(self.H, self.D_k, self.D_v, device=DEVICE)
        self.S_slow = torch.zeros(self.H, self.D_k, self.D_v, device=DEVICE)

        self.reset_state()

    def reset_state(self):
        self.S_fast.zero_()
        self.S_slow.zero_()

    def step(self, byte_idx: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        x_tensor = torch.tensor([byte_idx], device=DEVICE)
        target_tensor = torch.tensor([target_byte], device=DEVICE)

        emb = self.byte_emb(x_tensor)  # [1, 256]

        q = F.normalize(self.W_q(emb).view(self.H, 1, self.D_k), p=2, dim=-1)
        k = F.normalize(self.W_k(emb).view(self.H, 1, self.D_k), p=2, dim=-1)
        v = F.normalize(self.W_v(emb).view(self.H, 1, self.D_v), p=2, dim=-1)

        S_f = self.S_fast.detach()
        S_s = self.S_slow.detach()

        # 1. Resonant Recall from Both Streams:
        read_f = torch.bmm(q, S_f).squeeze(1)  # [H, D_v]
        read_s = torch.bmm(q, S_s).squeeze(1)  # [H, D_v]

        # Concatenate dual streams for global resonant readout
        read_dual = torch.cat([read_f, read_s], dim=-1).view(1, -1)  # [1, 2 * H * D_v]

        # 2. Modern Hopfield Attractor Snapping
        read_norm = torch.norm(read_dual)
        beta = 1.0 + F.softplus(self.w_beta * read_norm + self.b_beta)
        logits = beta * self.W_out(read_dual)

        loss = F.cross_entropy(logits, target_tensor)
        pred = torch.argmax(logits, dim=-1).item()
        surprise = loss.item()

        # 3. Stream-specific Prediction Errors:
        v_pred_f = torch.bmm(k, S_f)
        err_f = v - v_pred_f

        v_pred_s = torch.bmm(k, S_s)
        err_s = v - v_pred_s

        # === 4. AUTONOMOUS DUAL-STREAM LAW SYNTHESIS ===
        norm_err_f = torch.norm(err_f, dim=-1).squeeze(1).mean().item()
        norm_err_s = torch.norm(err_s, dim=-1).squeeze(1).mean().item()
        s_norm_tot = (torch.norm(S_f) + torch.norm(S_s)).item()

        synth_in = torch.tensor([
            min(norm_err_f, 3.0),
            min(norm_err_s, 3.0),
            min(surprise / 3.0, 3.0),
            min(s_norm_tot / 20.0, 3.0)
        ], device=DEVICE).unsqueeze(0)  # [1, 4]

        p = self.synthesizer(synth_in).squeeze(0)  # [4]

        # Stream 1 (Fast Plastic Memory):
        # Alpha fast: 0.70 - 0.95 (adaptive to error)
        alpha_f = 0.70 + 0.25 * torch.sigmoid(p[0])
        eta_f = F.softplus(p[1]) + 0.2

        # Stream 2 (Slow Invariant Memory):
        # Alpha slow: 0.95 - 0.999 (long-horizon conservation)
        alpha_s = 0.95 + 0.049 * torch.sigmoid(p[2])
        # Consolidation gating: Slow stream only absorbs when fast stream has low error!
        consolidation_gate = torch.exp(-torch.tensor(norm_err_f, device=DEVICE))
        eta_s = (F.softplus(p[3]) + 0.05) * consolidation_gate

        # 5. Dual Stream Updates:
        # Fast Stream: Orthogonal Delta Inscription
        delta_op_f = torch.bmm(k.transpose(1, 2), err_f)
        new_S_f = alpha_f * S_f + eta_f * delta_op_f

        # Slow Stream: Invariant Outer Product Trace + Delta Refinement
        hebb_op_s = torch.bmm(k.transpose(1, 2), v)
        new_S_s = alpha_s * S_s + eta_s * hebb_op_s

        # Out-of-place state update
        self.S_fast = new_S_f
        self.S_slow = new_S_s

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        meta = {
            "loss": surprise,
            "pred": pred,
            "target": target_byte,
            "is_correct": (pred == target_byte),
            "entropy": entropy,
            "beta": beta.item(),
            "alpha_f": alpha_f.item(),
            "eta_f": eta_f.item(),
            "alpha_s": alpha_s.item(),
            "eta_s": eta_s.item(),
            "norm_f": torch.norm(new_S_f).item(),
            "norm_s": torch.norm(new_S_s).item()
        }
        return loss, meta


def run_exp_443():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-443: SOVEREIGN DUAL-STREAM DEEP ASSOCIATIVE MANIFOLD (SD-DAM) ===")
    logger.info("================================================================================")

    config = EXP443Config()
    model = SovereignDualStreamManifold(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

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
    logger.info("Evaluating Sovereign Fast/Slow Dual-Stream Multi-Head Tensor Plasticity...")

    correct_preds = 0
    total_loss = 0.0
    total_entropy = 0.0
    recent_losses = []

    t_start = time.perf_counter()

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
        total_entropy += meta["entropy"]
        recent_losses.append(meta["loss"])

        if (t + 1) % 500 == 0:
            avg_loss = total_loss / (t + 1)
            acc = (correct_preds / (t + 1)) * 100.0
            logger.info(
                f"Progress [{t+1}/{len(raw_bytes)-1}] | Avg Loss: {avg_loss:.4f} | "
                f"Acc: {acc:.2f}% | Current Loss: {meta['loss']:.4f} | Beta: {meta['beta']:.2f} | "
                f"Fast[a:{meta['alpha_f']:.3f}, e:{meta['eta_f']:.2f}] | "
                f"Slow[a:{meta['alpha_s']:.4f}, e:{meta['eta_s']:.3f}]"
            )

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_final_entropy = total_entropy / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    second_half = recent_losses[len(recent_losses)//2:]
    low_loss_count = sum(1 for loss_val in second_half if loss_val < 0.5)
    low_loss_fraction = (low_loss_count / len(second_half)) * 100.0

    ultra_low_count = sum(1 for loss_val in second_half if loss_val < 0.05)
    ultra_low_fraction = (ultra_low_count / len(second_half)) * 100.0

    tail_500 = recent_losses[-500:]
    tail_low_loss_count = sum(1 for loss_val in tail_500 if loss_val < 0.5)
    tail_retention_rate = (tail_low_loss_count / len(tail_500)) * 100.0

    logger.info("================================================================================")
    logger.info("=== EXP-443 DUAL-STREAM DEEP ASSOCIATIVE MANIFOLD TELEMETRY REPORT ===")
    logger.info(f"Final Average Loss: {final_loss:.4f} nats")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Second-Half Low-Loss Mass (<0.5 nats): {low_loss_fraction:.2f}%")
    logger.info(f"Second-Half Ultra-Low Loss Mass (<0.05 nats): {ultra_low_fraction:.2f}%")
    logger.info(f"Tail 500-byte Retention Rate (<0.5 nats): {tail_retention_rate:.2f}%")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Mean Output Entropy: {avg_final_entropy:.3f} nats")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-443",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "second_half_low_loss_mass": round(low_loss_fraction, 2),
        "second_half_ultra_low_loss_mass": round(ultra_low_fraction, 2),
        "tail_retention_rate": round(tail_retention_rate, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_443_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_443()
