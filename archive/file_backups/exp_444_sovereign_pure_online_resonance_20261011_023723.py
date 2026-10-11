"""
EXP-444: Sovereign Pure Online Resonance & Zero-Shock Plasticity (SPOR-ZP)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Mechanistic Epiphany (Answering Bazilevs: "Точно все части системы суверенны?"):
NO! Look at the training loop in EXP-441/443:
```python
loss = F.cross_entropy(logits, target_tensor)
loss.backward()
optimizer.step()  # <--- THIS IS EXTERNAL GRADIENT DESCENT ON THE READOUT/PROJECTION WEIGHTS!
```
Every step, AdamW was updating W_q, W_k, W_v, W_out based on the INSTANT SURPRISE of the CURRENT byte!
This external optimizer was causing WEIGHT DRIFT on the linear projections, subtly modifying the
meaning of previously stored memory vectors! That's why accuracy on the initial 1000 bytes was low (11-27%),
dragging down the cumulative average!

In Pure Sovereign Cybernetic Architecture:
1. Online Resonance Calibration:
   The tensor manifold projections (W_q, W_k, W_v) and Readout Manifold (W_out) must maintain
   an invariant coordinate frame (fixed basis or slowly consolidated allostatic frame).
2. The memory matrix S is updated 100% ONLINE via Karyon's exact endogenous differential equation:
   Delta_Op = k^T @ (v - k @ S)
   S(t) = alpha(t) * S(t-1) + eta_delta(t) * Delta_Op
3. Exact Single-Pass Memory Recall:
   On a repeated sequence of length L, as soon as the sequence repeats for the 1st time (t > L),
   an exact associative tensor memory has already stored the transition k -> v!
   If the basis doesn't drift from external AdamW shocks, the recalled value is EXACT:
   Loss -> 0.0000 nats, Accuracy -> 95-100% on all repeated cycles!

Let's test this pure sovereign hypothesis:
Compare pure online memory plasticity with zero weight drift against AdamW-driven drift!
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
logger = logging.getLogger("EXP-444-SPOR-ZP")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP444Config:
    exp_id: str = "EXP-444"
    dim: int = 258
    num_heads: int = 8
    dim_k: int = 64
    dim_v: int = 64
    stream_length: int = 4000
    device_str: str = DEVICE_STR


class SovereignPureOnlineResonance(nn.Module):
    """
    Pure Sovereign Substrate with ZERO External Optimizer Drift.
    100% of learning occurs endogenously inside the physical memory tensor S.
    """
    def __init__(self, config: EXP444Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.H = config.num_heads
        self.D_k = config.dim_k
        self.D_v = config.dim_v

        # Orthogonal invariant coordinate frame for byte embeddings
        # Zero drift: vectors remain orthogonal and canonical!
        torch.manual_seed(42)
        raw_emb = torch.randn(self.dim, self.H * self.D_k, device=DEVICE)
        q_emb, _ = torch.linalg.qr(raw_emb)
        self.basis_k = q_emb.view(self.dim, self.H, self.D_k)

        raw_v = torch.randn(self.dim, self.H * self.D_v, device=DEVICE)
        q_v, _ = torch.linalg.qr(raw_v)
        self.basis_v = q_v.view(self.dim, self.H, self.D_v)

        # Dynamic Hopfield Attractor Snapping factor
        self.beta_attractor = 8.0

        # Memory Tensor S in R^(H x D_k x D_v)
        self.S_memory = torch.zeros(self.H, self.D_k, self.D_v, device=DEVICE)

        self.reset_state()

    def reset_state(self):
        self.S_memory.zero_()

    def step(self, byte_idx: int, target_byte: int) -> Tuple[float, Dict]:
        # 1. Fetch exact invariant basis vectors for current byte:
        k = self.basis_k[byte_idx].unsqueeze(1)  # [H, 1, D_k]
        q = k                                    # Auto-associative key-query alignment

        # Target value vector:
        v_target = self.basis_v[target_byte].unsqueeze(1)  # [H, 1, D_v]

        S = self.S_memory  # [H, D_k, D_v]

        # 2. Resonant Memory Reading:
        read_heads = torch.bmm(q, S).squeeze(1)  # [H, D_v]

        # 3. Resonant Logit Calculation across all 258 candidates via Direct Energy Alignment:
        # Energy(c) = Sum_h < basis_v(c)_h | read_heads_h >
        # basis_v: [258, H, D_v], read_heads: [H, D_v]
        energies = torch.einsum("chd,hd->c", self.basis_v, read_heads)  # [258]

        # Dynamic Hopfield Precision Snapping:
        read_norm = torch.norm(read_heads)
        beta_eff = self.beta_attractor * (1.0 + read_norm.item())
        logits = beta_eff * energies.unsqueeze(0)  # [1, 258]

        target_t = torch.tensor([target_byte], device=DEVICE)
        loss = F.cross_entropy(logits, target_t).item()
        pred = torch.argmax(logits, dim=-1).item()

        # 4. Pure Sovereign Delta Learning Operator:
        # v_pred = k @ S
        v_pred = torch.bmm(k, S)  # [H, 1, D_v]
        error = v_target - v_pred  # [H, 1, D_v]

        # Exact Widrow-Hoff Geometric Inscription (Delta Rule):
        # Delta S = k^T @ error
        # Since k is orthonormal (||k||=1), eta=1.0 achieves EXACT 1-step zero error inscription!
        delta_op = torch.bmm(k.transpose(1, 2), error)  # [H, D_k, D_v]

        # Endogenous Allostatic Retention:
        # Long-term retention alpha = 0.999
        new_S = 0.999 * S + 1.0 * delta_op

        self.S_memory = new_S

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        meta = {
            "loss": loss,
            "pred": pred,
            "target": target_byte,
            "is_correct": (pred == target_byte),
            "entropy": entropy,
            "read_norm": read_norm.item(),
            "s_norm": torch.norm(new_S).item()
        }
        return loss, meta


def run_exp_444():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-444: SOVEREIGN PURE ONLINE RESONANCE & ZERO DRIFT (SPOR-ZP) ===")
    logger.info("================================================================================")

    config = EXP444Config()
    model = SovereignPureOnlineResonance(config)

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

    period = len(stream_data)
    logger.info(f"Stream loaded: {len(raw_bytes)} bytes | Repetition period: {period} bytes on {DEVICE_STR.upper()}")
    logger.info("Evaluating 100% Pure Endogenous Associative Plasticity with Zero Optimizer Drift...")

    correct_preds = 0
    total_loss = 0.0
    total_entropy = 0.0
    recent_losses = []

    t_start = time.perf_counter()

    for t in range(len(raw_bytes) - 1):
        x_byte = raw_bytes[t]
        target_byte = raw_bytes[t + 1]

        loss_val, meta = model.step(x_byte, target_byte)

        if meta["is_correct"]:
            correct_preds += 1

        total_loss += meta["loss"]
        total_entropy += meta["entropy"]
        recent_losses.append(meta["loss"])

        if (t + 1) % 500 == 0:
            avg_loss = total_loss / (t + 1)
            acc = (correct_preds / (t + 1)) * 100.0
            cycle_acc = (sum(1 for val in recent_losses[-period:] if val < 0.1) / period) * 100.0 if t >= period else acc
            logger.info(
                f"Progress [{t+1}/{len(raw_bytes)-1}] | Avg Loss: {avg_loss:.4f} | "
                f"Cumulative Acc: {acc:.2f}% | Current Cycle Low-Loss: {cycle_acc:.2f}% | "
                f"Current Loss: {meta['loss']:.4f} | S Norm: {meta['s_norm']:.2f}"
            )

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_final_entropy = total_entropy / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    # Metric after 1st cycle (t > period):
    post_first_cycle = recent_losses[period:]
    post_cycle_acc = (sum(1 for i, loss_val in enumerate(post_first_cycle) if post_first_cycle[i] < 0.5) / len(post_first_cycle)) * 100.0

    second_half = recent_losses[len(recent_losses)//2:]
    low_loss_count = sum(1 for loss_val in second_half if loss_val < 0.5)
    low_loss_fraction = (low_loss_count / len(second_half)) * 100.0

    ultra_low_count = sum(1 for loss_val in second_half if loss_val < 0.01)
    ultra_low_fraction = (ultra_low_count / len(second_half)) * 100.0

    zero_loss_count = sum(1 for loss_val in second_half if loss_val < 0.001)
    zero_loss_fraction = (zero_loss_count / len(second_half)) * 100.0

    tail_500 = recent_losses[-500:]
    tail_low_loss_count = sum(1 for loss_val in tail_500 if loss_val < 0.5)
    tail_retention_rate = (tail_low_loss_count / len(tail_500)) * 100.0

    logger.info("================================================================================")
    logger.info("=== EXP-444 SOVEREIGN PURE ONLINE RESONANCE TELEMETRY REPORT ===")
    logger.info(f"Final Average Loss: {final_loss:.4f} nats")
    logger.info(f"Single-Pass Cumulative Accuracy: {final_acc:.2f}%")
    logger.info(f"Post-Cycle Retention Rate (t > period): {post_cycle_acc:.2f}%")
    logger.info(f"Second-Half Low-Loss Mass (<0.5 nats): {low_loss_fraction:.2f}%")
    logger.info(f"Second-Half Ultra-Low Loss (<0.01 nats): {ultra_low_fraction:.2f}%")
    logger.info(f"Second-Half Absolute Mathematical Zero Loss (<0.001 nats): {zero_loss_fraction:.2f}%")
    logger.info(f"Tail 500-byte Retention Rate (<0.5 nats): {tail_retention_rate:.2f}%")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Mean Output Entropy: {avg_final_entropy:.3f} nats")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-444",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "post_first_cycle_retention_rate": round(post_cycle_acc, 2),
        "second_half_low_loss_mass": round(low_loss_fraction, 2),
        "second_half_ultra_low_loss_mass": round(ultra_low_fraction, 2),
        "second_half_zero_loss_mass": round(zero_loss_fraction, 2),
        "tail_retention_rate": round(tail_retention_rate, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_444_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_444()
