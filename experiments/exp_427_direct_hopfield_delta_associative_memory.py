"""
EXP-427: Direct Hopfield Delta-Rule Orthogonal Associative Memory (D-DOAM)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 8, Principle 22, Principle 27 & KEP Rule #12)

Theoretical Architecture of EXP-427:
1. Exact Orthogonal Delta-Rule Update:
   S_{t+1} = alpha * S_t + eta * k_t^T (v_t - k_t S_t)
   Since k_t is L2-normalized (||k_t||_2 = 1.0), k_t (k_t^T v_t) = v_t.
   When memory already accurately contains v_t (i.e. k_t S_t = v_t), the delta term (v_t - k_t S_t) IS ZERO.
   This prevents catastrophic additive noise accumulation!

2. Direct Hopfield Attractor Readout (Zero-MLP Bottleneck):
   q = W_q(emb) [H, 1, dim_k]
   read = q @ S [H, 1, dim_v] -> flatten to [1, H * dim_v]
   logits = beta * W_out(read)
   Direct pathway from associative state to logits allows the Hopfield energy landscape
   to sharp-snap logits onto target bytes without gradient dilution!

3. Dual-Rate Memory Topology:
   - S_fast: Fast working memory (decay alpha_fast = 0.85)
   - S_slow: Long-horizon consolidated memory (decay alpha_slow = 0.999) with surprise-driven gating.

Target Metrics:
- Single-Pass Accuracy > 70%
- Second-Half Low-Loss Mass (<0.5 nats) > 85%
- Near-Zero Loss (<0.01 nats) > 40%
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
logger = logging.getLogger("EXP-427-DDOAM")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP427Config:
    exp_id: str = "EXP-427"
    dim: int = 258
    num_heads: int = 8
    dim_k: int = 64
    dim_v: int = 64
    learning_rate: float = 0.012
    stream_length: int = 4000
    eta_fast: float = 0.95
    eta_slow: float = 0.50
    device_str: str = DEVICE_STR


class DirectHopfieldDeltaAssociativeMemory(nn.Module):
    """
    Direct Hopfield Delta-Rule Orthogonal Associative Memory (D-DOAM)
    Dual-Rate (Fast + Slow) with Direct Logit Mapping & Delta Noise Shielding.
    """
    def __init__(self, config: EXP427Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.dim_k = config.dim_k
        self.dim_v = config.dim_v

        # 1. Byte Embedding [258, 256]
        self.byte_emb = nn.Embedding(self.dim, 256)

        # 2. Key, Query, Value Projections
        self.W_q = nn.Linear(256, self.num_heads * self.dim_k, bias=False)
        self.W_k = nn.Linear(256, self.num_heads * self.dim_k, bias=False)
        self.W_v = nn.Linear(256, self.num_heads * self.dim_v, bias=False)

        # 3. Direct Readout Projection: total output of heads (2 * H * dim_v = 2 * 8 * 64 = 1024) -> 258
        self.W_out = nn.Linear(2 * self.num_heads * self.dim_v, self.dim, bias=False)

        # 4. Hopfield Inverse Temperature / Beta Snapping
        self.w_beta = nn.Parameter(torch.tensor(1.5, device=DEVICE))
        self.b_beta = nn.Parameter(torch.tensor(2.5, device=DEVICE))

        # 5. Fast and Slow Decay Parameters
        self.fast_decay_params = nn.Parameter(torch.linspace(0.85, 2.2, self.num_heads, device=DEVICE))
        self.slow_decay_params = nn.Parameter(torch.linspace(4.6, 7.6, self.num_heads, device=DEVICE))

        # 6. Memory Buffers [H, dim_k, dim_v]
        self.S_fast = torch.zeros(self.num_heads, self.dim_k, self.dim_v, device=DEVICE)
        self.S_slow = torch.zeros(self.num_heads, self.dim_k, self.dim_v, device=DEVICE)

    def reset_state(self):
        self.S_fast = torch.zeros(self.num_heads, self.dim_k, self.dim_v, device=DEVICE)
        self.S_slow = torch.zeros(self.num_heads, self.dim_k, self.dim_v, device=DEVICE)

    def forward_stream_byte(self, x_byte: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        x_tensor = torch.tensor([x_byte], device=DEVICE)
        target_tensor = torch.tensor([target_byte], device=DEVICE)

        emb = self.byte_emb(x_tensor)  # [1, 256]

        # Q, K, V
        q = F.normalize(self.W_q(emb).view(self.num_heads, 1, self.dim_k), p=2, dim=-1)
        k = F.normalize(self.W_k(emb).view(self.num_heads, 1, self.dim_k), p=2, dim=-1)
        v = F.normalize(self.W_v(emb).view(self.num_heads, 1, self.dim_v), p=2, dim=-1)

        # Direct Readout from Fast and Slow Memories
        read_fast = torch.bmm(q, self.S_fast.detach()).squeeze(1).view(1, -1)  # [1, H * dim_v]
        read_slow = torch.bmm(q, self.S_slow.detach()).squeeze(1).view(1, -1)  # [1, H * dim_v]

        read_combined = torch.cat([read_fast, read_slow], dim=-1)  # [1, 1024]

        # Hopfield Attractor Energy Snapping
        beta = 1.0 + F.softplus(self.w_beta * torch.norm(read_combined) + self.b_beta)
        logits = beta * self.W_out(read_combined)  # [1, 258]

        loss = F.cross_entropy(logits, target_tensor)
        pred = torch.argmax(logits, dim=-1).item()
        surprise_val = loss.item()

        # === DELTA-RULE MEMORY INSCRIPTION ===
        with torch.no_grad():
            alpha_fast = torch.clamp(torch.sigmoid(self.fast_decay_params), min=0.70, max=0.92).view(self.num_heads, 1, 1)
            alpha_slow = torch.clamp(torch.sigmoid(self.slow_decay_params), min=0.990, max=0.9995).view(self.num_heads, 1, 1)

            # 1. Fast Memory Delta Update
            v_pred_fast = torch.bmm(k, self.S_fast)  # [H, 1, dim_v]
            error_fast = v - v_pred_fast  # [H, 1, dim_v]
            delta_fast = torch.bmm(k.transpose(1, 2), error_fast)  # [H, dim_k, dim_v]
            self.S_fast = alpha_fast * self.S_fast + self.config.eta_fast * delta_fast

            # 2. Slow Memory Delta Update (Gated by surprise)
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
            "norm_s_fast": torch.norm(self.S_fast).item(),
            "norm_s_slow": torch.norm(self.S_slow).item()
        }
        return loss, meta


def run_exp_427():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-427: DIRECT HOPFIELD DELTA ASSOCIATIVE MEMORY (D-DOAM) ===")
    logger.info("================================================================================")

    config = EXP427Config()
    model = DirectHopfieldDeltaAssociativeMemory(config).to(DEVICE)

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
    logger.info("Starting single-pass stream with Direct Hopfield Delta Readout...")

    correct_preds = 0
    total_loss = 0.0
    total_entropy = 0.0
    recent_losses = []

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

        if (t + 1) % 500 == 0:
            avg_loss = total_loss / (t + 1)
            acc = (correct_preds / (t + 1)) * 100.0
            avg_ent = total_entropy / (t + 1)
            logger.info(
                f"Progress [{t+1}/{len(raw_bytes)-1}] | Avg Loss: {avg_loss:.4f} | "
                f"Acc: {acc:.2f}% | Current Loss: {meta['loss']:.4f} | "
                f"Entropy: {avg_ent:.3f} nats | Beta: {meta['beta']:.2f} | "
                f"|S_fast|: {meta['norm_s_fast']:.1f} | |S_slow|: {meta['norm_s_slow']:.1f}"
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

    S_slow_flat = model.S_slow.view(-1, config.dim_v)
    svd_vals = torch.linalg.svdvals(S_slow_flat)
    s_slow_pr = ((torch.sum(svd_vals ** 2) ** 2) / (torch.sum(svd_vals ** 4) + 1e-9)).item()

    logger.info("================================================================================")
    logger.info("=== EXP-427 COMPLETE TELEMETRY & LONG-HORIZON RETENTION REPORT ===")
    logger.info(f"Final Average Loss: {final_loss:.4f} nats")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Second-Half Low-Loss Mass (<0.5 nats): {low_loss_fraction:.2f}%")
    logger.info(f"Second-Half Ultra-Low Loss Mass (<0.1 nats): {ultra_low_loss_fraction:.2f}%")
    logger.info(f"Second-Half Near-Zero Loss Mass (<0.01 nats): {near_zero_loss_fraction:.2f}%")
    logger.info(f"Tail 500-byte Retention Rate (<0.5 nats): {tail_retention_rate:.2f}%")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Mean Output Entropy: {avg_final_entropy:.3f} nats")
    logger.info(f"Consolidated Slow Memory Effective Rank (PR): {s_slow_pr:.2f} / 64")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-427",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "second_half_low_loss_mass": round(low_loss_fraction, 2),
        "second_half_ultra_low_loss_mass": round(ultra_low_loss_fraction, 2),
        "second_half_near_zero_loss_mass": round(near_zero_loss_fraction, 2),
        "tail_retention_rate": round(tail_retention_rate, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "slow_memory_pr": round(s_slow_pr, 2),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_427_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_427()
