"""
EXP-438: Sovereign Context-Phase Holographic Synthesizer (CPHS)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Mechanistic Breakthrough Analysis:
In EXP-436, 1st-order association M * x_stim -> y_target achieved 43.51% accuracy, but hit a ceiling because
single byte keys ('a') cannot distinguish higher-order sequence context ("ba" -> "n" vs "ca" -> "r").
A 1st-order matrix overwrites previous associations when the same byte appears in different contexts!

Solution (EXP-438):
1. Context-Modulated Wavepacket Key:
   x_key(t) = x_stim(t) + LieRotate(h_context(t-1), theta)
   Now, 'a' after 'b' produces an orthogonal key vector in C^D from 'a' after 'c'!

2. Exact Geometric Projection on Context-Key:
   O_1 = (y_target - y_rec) * x_key^dagger / (||x_key||^2 + eps)
   Stores exact high-order N-gram transitions without crosstalk!

3. Continuous Recurrent Phase History:
   h_context(t) = UnitaryPhaseRotate(h_context(t-1) + x_stim(t))

4. Sovereign Operator Synthesizer:
   Autonomously synthesizes dynamic laws w_k(t) balancing Exact LMS Projection, Lie Phase Flow, and Attractor Compression.

Target:
Break through 70-90%+ single-pass accuracy and eliminate higher-order sequence crosstalk!
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
logger = logging.getLogger("EXP-438-CPHS")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP438Config:
    exp_id: str = "EXP-438"
    vocab_dim: int = 258
    heads: int = 8
    head_dim: int = 96
    num_operators: int = 4
    learning_rate: float = 0.015
    stream_length: int = 4000
    dt: float = 1.0
    device_str: str = DEVICE_STR


class ContextPhaseHolographicSynthesizer(nn.Module):
    """
    Sovereign Context-Phase Holographic Memory Substrate.
    Eradicates 1st-order crosstalk by binding immediate stimulus x_stim with continuous phase context h_context.
    """
    def __init__(self, config: EXP438Config):
        super().__init__()
        self.config = config
        self.vocab_dim = config.vocab_dim
        self.H = config.heads
        self.D = config.head_dim
        self.total_dim = self.H * self.D
        self.num_ops = config.num_operators
        self.dt = config.dt

        # Complex embeddings for input and target
        self.emb_real = nn.Embedding(self.vocab_dim, self.total_dim)
        self.emb_imag = nn.Embedding(self.vocab_dim, self.total_dim)

        # Skew-Hermitian Lie Generator for Context Rotation
        self.A_ctx = nn.Parameter(torch.randn(self.H, self.D, self.D, device=DEVICE) * 0.02)
        self.A_mem = nn.Parameter(torch.randn(self.H, self.D, self.D, device=DEVICE) * 0.01)

        # Context Coupling Factor
        self.ctx_scale = nn.Parameter(torch.tensor(0.8, device=DEVICE))

        # Autonomous Law Synthesizer Network
        self.synthesizer = nn.Sequential(
            nn.Linear(self.H * 3, 32),
            nn.SiLU(),
            nn.Linear(32, self.num_ops)
        )

        # RMS readback parameter
        self.norm_scale = nn.Parameter(torch.ones(self.total_dim, device=DEVICE))

        # Readout Projection
        self.readout_real = nn.Linear(self.total_dim, self.vocab_dim, bias=False)
        self.readout_imag = nn.Linear(self.total_dim, self.vocab_dim, bias=False)

        # Persistent Memory Tensor: M in C^(H x D x D)
        self.M_real = torch.zeros(self.H, self.D, self.D, device=DEVICE)
        self.M_imag = torch.zeros(self.H, self.D, self.D, device=DEVICE)

        # Continuous Recurrent Phase Context Vector: h_ctx in C^(H x D x 1)
        self.h_ctx_r = torch.zeros(self.H, self.D, 1, device=DEVICE)
        self.h_ctx_i = torch.zeros(self.H, self.D, 1, device=DEVICE)

        self.reset_state()

    def reset_state(self):
        self.M_real.zero_()
        self.M_imag.zero_()
        self.h_ctx_r.zero_()
        self.h_ctx_i.zero_()

    def step(self, byte_idx: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        idx_t = torch.tensor([byte_idx], device=DEVICE)
        target_t = torch.tensor([target_byte], device=DEVICE)

        # 1. Complex Multi-Head Embeddings
        xr = self.emb_real(idx_t).view(self.H, self.D, 1)  # [H, D, 1]
        xi = self.emb_imag(idx_t).view(self.H, self.D, 1)  # [H, D, 1]

        yr = self.emb_real(target_t).view(self.H, self.D, 1)  # [H, D, 1]
        yi = self.emb_imag(target_t).view(self.H, self.D, 1)  # [H, D, 1]

        M_r = self.M_real.detach()  # [H, D, D]
        M_i = self.M_imag.detach()  # [H, D, D]

        h_r = self.h_ctx_r.detach()  # [H, D, 1]
        h_i = self.h_ctx_i.detach()  # [H, D, 1]

        # 2. Rotate Continuous Phase Context h_ctx via Lie Commutator
        H_ctx_skew = (self.A_ctx - self.A_ctx.transpose(1, 2)) * 0.5
        rot_h_r = -torch.bmm(H_ctx_skew, h_i)
        rot_h_i = torch.bmm(H_ctx_skew, h_r)

        # 3. Form Context-Modulated Key Vector: x_key = x_stim + alpha * rot_h
        alpha = F.softplus(self.ctx_scale)
        key_r = xr + alpha * rot_h_r
        key_i = xi + alpha * rot_h_i

        # 4. Resonant Complex Recall from Memory Tensor: y_rec = M * x_key
        rec_r = torch.bmm(M_r, key_r) - torch.bmm(M_i, key_i)  # [H, D, 1]
        rec_i = torch.bmm(M_r, key_i) + torch.bmm(M_i, key_r)  # [H, D, 1]

        # Combine Key Stimulus + Memory Recall
        comb_r = (key_r + rec_r).view(1, self.total_dim)  # [1, total_dim]
        comb_i = (key_i + rec_i).view(1, self.total_dim)  # [1, total_dim]

        # RMS-Normalization
        mag_sq = comb_r ** 2 + comb_i ** 2
        rms = torch.sqrt(torch.mean(mag_sq, dim=-1, keepdim=True) + 1e-6)
        norm_r = (comb_r / rms) * self.norm_scale
        norm_i = (comb_i / rms) * self.norm_scale

        logits = self.readout_real(norm_r) - self.readout_imag(norm_i)

        loss = F.cross_entropy(logits, target_t)
        pred = torch.argmax(logits, dim=-1).item()
        surprise = loss.item()

        # 5. Complex Geometric Error: delta = target - recalled
        delta_r = yr - rec_r  # [H, D, 1]
        delta_i = yi - rec_i  # [H, D, 1]

        # === 6. ATOMIC OPERATOR BASIS WITH CONTEXT-KEY PROJECTION ===
        # O_1: Exact Geometric Projection on Context-Key: delta * key^dagger / (||key||^2 + eps)
        key_norm_sq = torch.sum(key_r ** 2 + key_i ** 2, dim=(1, 2), keepdim=True) + 1e-6  # [H, 1, 1]
        kr_t = key_r.transpose(1, 2)  # [H, 1, D]
        ki_t = key_i.transpose(1, 2)  # [H, 1, D]

        p1_r = torch.bmm(delta_r, kr_t) + torch.bmm(delta_i, ki_t)
        p1_i = torch.bmm(delta_i, kr_t) - torch.bmm(delta_r, ki_t)

        o1_r = p1_r / key_norm_sq  # [H, D, D]
        o1_i = p1_i / key_norm_sq  # [H, D, D]

        # O_2: Memory Lie Phase Rotation: [H_mem, M]
        H_mem_skew = (self.A_mem - self.A_mem.transpose(1, 2)) * 0.5
        o2_r = torch.bmm(H_mem_skew, M_r) - torch.bmm(M_r, H_mem_skew)
        o2_i = torch.bmm(H_mem_skew, M_i) - torch.bmm(M_i, H_mem_skew)

        # O_3: Nonlinear Attractor Compression: -0.05 * M * ||M||
        m_norms = torch.sqrt(torch.sum(M_r ** 2 + M_i ** 2, dim=(-1, -2), keepdim=True) + 1e-8)
        o3_r = -0.05 * M_r * m_norms
        o3_i = -0.05 * M_i * m_norms

        # O_4: Leaky Trace Decay: -0.02 * M
        o4_r = -0.02 * M_r
        o4_i = -0.02 * M_i

        # === 7. SOVEREIGN FORMULA SYNTHESIS ===
        err_norm = torch.sqrt(torch.sum(delta_r ** 2 + delta_i ** 2, dim=1)).view(1, self.H)  # [1, H]
        m_head_norm = m_norms.view(1, self.H)  # [1, H]
        surp_feat = torch.full((1, self.H), surprise, device=DEVICE)  # [1, H]

        synth_input = torch.cat([err_norm, m_head_norm, surp_feat], dim=-1)
        synth_logits = self.synthesizer(synth_input)
        w = F.softmax(synth_logits, dim=-1).squeeze(0)  # [4]

        # Synthesized Tensor Differential Equation:
        dM_r = w[0] * o1_r + w[1] * o2_r + w[2] * o3_r + w[3] * o4_r
        dM_i = w[0] * o1_i + w[1] * o2_i + w[2] * o3_i + w[3] * o4_i

        # State Integration
        new_M_r = M_r + self.dt * dM_r
        new_M_i = M_i + self.dt * dM_i

        # Allostatic Norm Bounding
        new_norms = torch.sqrt(torch.sum(new_M_r ** 2 + new_M_i ** 2, dim=(-1, -2), keepdim=True) + 1e-8)
        max_bound = 15.0
        scale = torch.clamp(max_bound / new_norms, max=1.0)
        self.M_real = new_M_r * scale
        self.M_imag = new_M_i * scale

        # Update Continuous Phase Context Vector h_ctx(t)
        new_h_r = rot_h_r + xr
        new_h_i = rot_h_i + xi
        h_norm = torch.sqrt(torch.sum(new_h_r ** 2 + new_h_i ** 2, dim=1, keepdim=True) + 1e-8)
        self.h_ctx_r = new_h_r / (h_norm + 1e-5)
        self.h_ctx_i = new_h_i / (h_norm + 1e-5)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        meta = {
            "loss": surprise,
            "pred": pred,
            "target": target_byte,
            "is_correct": (pred == target_byte),
            "entropy": entropy,
            "weights": w.detach().cpu().numpy().tolist()
        }
        return loss, meta


def run_exp_438():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-438: SOVEREIGN CONTEXT-PHASE HOLOGRAPHIC SYNTHESIZER (CPHS) ===")
    logger.info("================================================================================")

    config = EXP438Config()
    model = ContextPhaseHolographicSynthesizer(config).to(DEVICE)
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
    logger.info("Evaluating Context-Phase Wavepacket Key Projection and Formula Synthesis...")

    correct_preds = 0
    total_loss = 0.0
    total_entropy = 0.0
    recent_losses = []
    op_weight_accum = [0.0] * config.num_operators

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

        for i in range(config.num_operators):
            op_weight_accum[i] += meta["weights"][i]

        if (t + 1) % 500 == 0:
            avg_loss = total_loss / (t + 1)
            acc = (correct_preds / (t + 1)) * 100.0
            cur_weights = [round(w_val / (t + 1), 3) for w_val in op_weight_accum]
            logger.info(
                f"Progress [{t+1}/{len(raw_bytes)-1}] | Avg Loss: {avg_loss:.4f} | "
                f"Acc: {acc:.2f}% | Current Loss: {meta['loss']:.4f} | "
                f"Mean Op Weights [ContextDelta, LieRot, Attractor, Leaky]: {cur_weights}"
            )

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_final_entropy = total_entropy / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    mean_op_dist = [round(w_val / (len(raw_bytes) - 1), 4) for w_val in op_weight_accum]

    second_half = recent_losses[len(recent_losses)//2:]
    low_loss_count = sum(1 for loss_val in second_half if loss_val < 0.5)
    low_loss_fraction = (low_loss_count / len(second_half)) * 100.0

    tail_500 = recent_losses[-500:]
    tail_low_loss_count = sum(1 for loss_val in tail_500 if loss_val < 0.5)
    tail_retention_rate = (tail_low_loss_count / len(tail_500)) * 100.0

    logger.info("================================================================================")
    logger.info("=== EXP-438 CONTEXT-PHASE LAW SYNTHESIS TELEMETRY REPORT ===")
    logger.info(f"Final Average Loss: {final_loss:.4f} nats")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Synthesized Operator Distribution: {mean_op_dist}")
    logger.info(f"Second-Half Low-Loss Mass (<0.5 nats): {low_loss_fraction:.2f}%")
    logger.info(f"Tail 500-byte Retention Rate (<0.5 nats): {tail_retention_rate:.2f}%")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Mean Output Entropy: {avg_final_entropy:.3f} nats")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-438",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "synthesized_operator_distribution": {
            "O1_context_exact_delta": mean_op_dist[0],
            "O2_lie_rotation": mean_op_dist[1],
            "O3_attractor_compression": mean_op_dist[2],
            "O4_leaky_decay": mean_op_dist[3]
        },
        "second_half_low_loss_mass": round(low_loss_fraction, 2),
        "tail_retention_rate": round(tail_retention_rate, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_438_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_438()
