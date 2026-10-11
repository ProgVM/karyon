"""
EXP-421: Sovereign Holographic State-Space Memory Field (HSMF-Zero)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Theoretical Architecture of EXP-421:
1. Multi-Dimensional Fast Key-Value Associative Field S_t in C^{D_k x D_v}:
   S_t = d * S_{t-1} + k_t * v_t^T (Linear State-Space Duality + Fast Holographic Trace)
2. Dual-Drive Memory System:
   - Fast Associative Memory Matrix S_t for instant zero-shot context sequence binding.
   - Continuous Field Coupling parameters (W_q, W_k, W_v, W_out, beta) optimized online.
3. Multi-Order Continuous Context Convolution:
   State wavepacket Psi_t integrates past context via continuous decay and non-linear gauge rotation.
"""

import math
import time
import json
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-421-HSMF")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP421Config:
    exp_id: str = "EXP-421"
    dim: int = 258
    dim_k: int = 128
    dim_v: int = 128
    learning_rate: float = 0.008
    stream_length: int = 3000
    device_str: str = DEVICE_STR


class HolographicStateSpaceField(nn.Module):
    """
    Sovereign Holographic State-Space Memory Field combining
    Continuous Gauge Rotation with Key-Value Associative Memory S in C^{D_k x D_v}.
    """
    def __init__(self, config: EXP421Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.dim_k = config.dim_k
        self.dim_v = config.dim_v

        # 1. Byte Embedding Matrix [258, 128]
        self.byte_emb = nn.Embedding(self.dim, 128)

        # 2. Key, Query, Value Projections
        self.W_q = nn.Linear(128, self.dim_k, bias=False)
        self.W_k = nn.Linear(128, self.dim_k, bias=False)
        self.W_v = nn.Linear(128, self.dim_v, bias=False)

        # 3. Dynamic Memory Decay Factor
        self.decay_param = nn.Parameter(torch.tensor(1.5, device=DEVICE)) # softplus(1.5) ≈ 0.85

        # 4. Readout Resonant Projection Matrix [dim_v -> 258]
        self.W_out = nn.Linear(self.dim_v, self.dim, bias=False)

        # 5. Inverse Temperature / Sharpness Factor beta
        self.w_beta = nn.Parameter(torch.tensor(1.0, device=DEVICE))
        self.b_beta = nn.Parameter(torch.tensor(2.0, device=DEVICE))

        # 6. Persistent Associative State Memory S in R^{dim_k x dim_v}
        self.register_buffer("S_memory", torch.zeros(self.dim_k, self.dim_v, device=DEVICE))

    def reset_state(self):
        self.S_memory.zero_()

    def forward_stream_byte(self, x_byte: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        x_tensor = torch.tensor([x_byte], device=DEVICE)
        target_tensor = torch.tensor([target_byte], device=DEVICE)

        # Embed byte
        emb = self.byte_emb(x_tensor) # [1, 128]
        q = self.W_q(emb) # [1, dim_k]
        k = self.W_k(emb) # [1, dim_k]
        v = self.W_v(emb) # [1, dim_v]

        # 1. Read from Associative Memory Matrix S_t
        read = torch.matmul(q, self.S_memory) # [1, dim_v]

        # 2. Resonant Logit Output
        beta = 1.0 + F.softplus(self.w_beta * torch.norm(read) + self.b_beta)
        logits = beta * self.W_out(read) # [1, 258]

        loss = F.cross_entropy(logits, target_tensor)

        pred = torch.argmax(logits, dim=-1).item()
        surprise_val = loss.item()

        # 3. Update Fast Holographic Associative Matrix S_{t+1}
        with torch.no_grad():
            decay = torch.clamp(torch.sigmoid(self.decay_param), min=0.50, max=0.99)
            # Inscribe transition: Key(x_t) -> Value(x_t)
            new_S = decay * self.S_memory + torch.outer(k.squeeze(0), v.squeeze(0))
            self.S_memory.copy_(new_S)

            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        meta = {
            "loss": surprise_val,
            "pred": pred,
            "is_correct": (pred == target_byte),
            "entropy": entropy,
            "beta": beta.item(),
            "decay": decay.item(),
            "s_norm": torch.norm(self.S_memory).item()
        }
        return loss, meta


def run_exp_421():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-421: HOLOGRAPHIC STATE-SPACE MEMORY FIELD (HSMF-ZERO) ===")
    logger.info("================================================================================")

    config = EXP421Config()
    model = HolographicStateSpaceField(config).to(DEVICE)

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
    logger.info("Starting single-pass stream with Holographic Associative Memory S in C^{128x128}...")

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
                f"Entropy: {avg_ent:.3f} nats | Beta: {meta['beta']:.2f} | Decay: {meta['decay']:.3f}"
            )

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_final_entropy = total_entropy / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    # Calculate low-loss fraction (<0.5 nats) in the second half of the stream
    second_half = recent_losses[len(recent_losses)//2:]
    low_loss_count = sum(1 for l in second_half if l < 0.5)
    low_loss_fraction = (low_loss_count / len(second_half)) * 100.0

    # Calculate Spectral Participation Ratio of S_memory
    S_svd = torch.linalg.svdvals(model.S_memory)
    s_pr = ((torch.sum(S_svd ** 2) ** 2) / (torch.sum(S_svd ** 4) + 1e-9)).item()

    logger.info("================================================================================")
    logger.info("=== EXP-421 COMPLETE SPECTRAL & TELEMETRY REPORT ===")
    logger.info(f"Final Average Loss: {final_loss:.4f} nats")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Second-Half Low-Loss Mass (<0.5 nats): {low_loss_fraction:.2f}%")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Mean Output Entropy: {avg_final_entropy:.3f} nats")
    logger.info(f"Memory Matrix S Effective Rank (PR): {s_pr:.2f} / 128")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-421",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "second_half_low_loss_mass": round(low_loss_fraction, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "memory_participation_ratio": round(s_pr, 2),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_421_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_421()
