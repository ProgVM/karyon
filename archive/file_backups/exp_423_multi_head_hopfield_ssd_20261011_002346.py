"""
EXP-423: Hierarchical Multi-Head Associative SSD with Modern Hopfield Snapping (HMA-SSD)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Theoretical Architecture of EXP-423:
1. Multi-Head Associative State Tensor S_t in R^{H x D_k x D_v} (H=8 heads):
   Splits memory into 8 independent parallel frequency heads.
   Each head operates with its own decay rate alpha_h in [0.50, 0.99]!

2. Modern Hopfield Attractor Snapping (Beta Scaling):
   Readout applies a soft-max Modern Hopfield Energy Snapping step:
   z_t = ModernHopfield(read_t, Beta=30.0)
   To crush residual noise and snap probability mass directly onto exact target bytes!

3. Objective: Drive Single-Pass Accuracy > 70% and Second-Half Loss < 0.5 nats.
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
logger = logging.getLogger("EXP-423-HMASSD")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP423Config:
    exp_id: str = "EXP-423"
    dim: int = 258
    num_heads: int = 8
    dim_k: int = 64
    dim_v: int = 64
    learning_rate: float = 0.012
    stream_length: int = 3000
    device_str: str = DEVICE_STR


class MultiHeadHopfieldSSD(nn.Module):
    """
    Hierarchical Multi-Head Associative State Space Field (HMA-SSD)
    with Modern Hopfield Energy Snapping.
    """
    def __init__(self, config: EXP423Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.dim_k = config.dim_k
        self.dim_v = config.dim_v

        # 1. Byte Embedding Matrix [258, 256]
        self.byte_emb = nn.Embedding(self.dim, 256)

        # 2. Multi-Head Key, Query, Value Projections [256 -> H * dim_k/v]
        self.W_q = nn.Linear(256, self.num_heads * self.dim_k, bias=False)
        self.W_k = nn.Linear(256, self.num_heads * self.dim_k, bias=False)
        self.W_v = nn.Linear(256, self.num_heads * self.dim_v, bias=False)

        # 3. Multi-Head Log-Spaced Decay Rates
        # Half-lives spanning from fast (decay=0.60) to slow (decay=0.98)
        init_decays = torch.linspace(0.40, 3.0, self.num_heads, device=DEVICE)
        self.decay_params = nn.Parameter(init_decays)

        # 4. Multi-Head Readout Projection
        self.W_out = nn.Linear(self.num_heads * self.dim_v, self.dim, bias=False)

        # 5. Hopfield Inverse Temperature / Beta
        self.w_beta = nn.Parameter(torch.tensor(1.5, device=DEVICE))
        self.b_beta = nn.Parameter(torch.tensor(2.5, device=DEVICE))

        # 6. Group Normalization per Head
        self.gn_heads = nn.GroupNorm(num_groups=self.num_heads, num_channels=self.num_heads * self.dim_k)

        # 7. Dynamic Multi-Head State Buffer [H, dim_k, dim_v]
        self.S_memory = torch.zeros(self.num_heads, self.dim_k, self.dim_v, device=DEVICE)

    def reset_state(self):
        self.S_memory = torch.zeros(self.num_heads, self.dim_k, self.dim_v, device=DEVICE)

    def forward_stream_byte(self, x_byte: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        x_tensor = torch.tensor([x_byte], device=DEVICE)
        target_tensor = torch.tensor([target_byte], device=DEVICE)

        emb = self.byte_emb(x_tensor)  # [1, 256]

        # Multi-Head Q, K, V
        q = self.W_q(emb).view(1, self.num_heads, self.dim_k)  # [1, H, dim_k]
        k = self.W_k(emb).view(1, self.num_heads, self.dim_k)  # [1, H, dim_k]
        v = self.W_v(emb).view(1, self.num_heads, self.dim_v)  # [1, H, dim_v]

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        v = F.normalize(v, p=2, dim=-1)

        # 1. Multi-Head Memory Reading: read_h = q_h @ S_h
        # S_memory: [H, dim_k, dim_v]
        # q: [1, H, dim_k] -> bmm -> [1, H, dim_v]
        read_heads = torch.matmul(q, self.S_memory.detach()).squeeze(0)  # [H, dim_v]
        read_flat = read_heads.view(1, -1)  # [1, H * dim_v]

        # 2. Modern Hopfield Attractor Snapping
        beta = 1.0 + F.softplus(self.w_beta * torch.norm(read_flat) + self.b_beta)
        logits = beta * self.W_out(read_flat)  # [1, 258]

        loss = F.cross_entropy(logits, target_tensor)

        pred = torch.argmax(logits, dim=-1).item()
        surprise_val = loss.item()

        # 3. Out-of-place Multi-Head State Inscription with Head-Specific Decays
        with torch.no_grad():
            decays = torch.clamp(torch.sigmoid(self.decay_params), min=0.50, max=0.99)  # [H]
            decays_view = decays.view(self.num_heads, 1, 1)

            # k: [1, H, dim_k] -> [H, dim_k, 1]
            # v: [1, H, dim_v] -> [H, 1, dim_v]
            k_h = k.squeeze(0).unsqueeze(-1)
            v_h = v.squeeze(0).unsqueeze(1)
            outer_h = torch.matmul(k_h, v_h)  # [H, dim_k, dim_v]

            self.S_memory = decays_view * self.S_memory + outer_h

            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        meta = {
            "loss": surprise_val,
            "pred": pred,
            "is_correct": (pred == target_byte),
            "entropy": entropy,
            "beta": beta.item(),
            "mean_decay": decays.mean().item(),
            "s_norm": torch.norm(self.S_memory).item()
        }
        return loss, meta


def run_exp_423():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-423: MULTI-HEAD HOPFIELD ASSOCIATIVE SSD (HMA-SSD) ===")
    logger.info("================================================================================")

    config = EXP423Config()
    model = MultiHeadHopfieldSSD(config).to(DEVICE)

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
    ).encode("utf-8")

    raw_bytes = list(stream_data) * (config.stream_length // len(stream_data) + 1)
    raw_bytes = raw_bytes[:config.stream_length]

    logger.info(f"Stream loaded: {len(raw_bytes)} bytes on {DEVICE_STR.upper()}")
    logger.info("Starting single-pass stream with 8 Parallel Frequency Memory Heads in R^{8x64x64}...")

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
                f"Entropy: {avg_ent:.3f} nats | Beta: {meta['beta']:.2f} | MeanDecay: {meta['mean_decay']:.3f}"
            )

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_final_entropy = total_entropy / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    # Calculate low-loss fraction (<0.5 nats) in the second half of the stream
    second_half = recent_losses[len(recent_losses)//2:]
    low_loss_count = sum(1 for loss_val in second_half if loss_val < 0.5)
    low_loss_fraction = (low_loss_count / len(second_half)) * 100.0

    # Calculate Spectral Participation Ratio across Multi-Head S_memory
    S_flat = model.S_memory.view(-1, config.dim_v)
    S_svd = torch.linalg.svdvals(S_flat)
    s_pr = ((torch.sum(S_svd ** 2) ** 2) / (torch.sum(S_svd ** 4) + 1e-9)).item()

    logger.info("================================================================================")
    logger.info("=== EXP-423 COMPLETE SPECTRAL & TELEMETRY REPORT ===")
    logger.info(f"Final Average Loss: {final_loss:.4f} nats")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Second-Half Low-Loss Mass (<0.5 nats): {low_loss_fraction:.2f}%")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Mean Output Entropy: {avg_final_entropy:.3f} nats")
    logger.info(f"Multi-Head Memory Matrix S Effective Rank (PR): {s_pr:.2f} / 64")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-423",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "second_half_low_loss_mass": round(low_loss_fraction, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "memory_participation_ratio": round(s_pr, 2),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_423_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_423()
