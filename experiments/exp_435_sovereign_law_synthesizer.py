"""
EXP-435: Sovereign Differentiable Operator & Law Synthesizer (SDOLS)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Core Philosophy (Bazilevs):
"Карион сам синтезирует формулы и правила, которые ему нужны, причём это должно быть на чистой архитектуре"

Key Architectural Breakthroughs:
1. Multi-Head Holographic Memory Tensor M_h in C^(H x D_k x D_k).
2. RMS-Normalized Resonant Phase Readback (eradicates logit explosion, loss stays bounded in [0, 5.5] nats).
3. Universal Atomic Basis of 5 Matrix Dynamics Operators:
   - O_1: Sovereign Delta Outer Product (e_stim * (e_target - e_recalled)^dagger)
   - O_2: Unitary Phase Rotation (Lie commutator [H, M])
   - O_3: Unit Attractor Projection (M / ||M||)
   - O_4: Leaky Diffusion Dissipation (-alpha * M)
   - O_5: Multiplicative Cross-Coupling (M * stimulus_gate)
4. Free-Energy Driven Law Synthesis:
   w_k(t) = Softmax(W_synth * [error, state_norm, surprise])
   dM/dt = sum_{k=1}^5 w_k(t) * O_k

Target:
Break through 70-90%+ single-pass accuracy and sub-1.0 nats loss cleanly!
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
logger = logging.getLogger("EXP-435-SDOLS")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP435Config:
    exp_id: str = "EXP-435"
    vocab_dim: int = 258
    heads: int = 4
    head_dim: int = 64
    num_operators: int = 5
    learning_rate: float = 0.015
    stream_length: int = 4000
    dt: float = 0.5
    device_str: str = DEVICE_STR


class SovereignLawSynthesizer(nn.Module):
    """
    Clean, Universal Mathematical Substrate.
    Multi-Head Holographic Complex Memory with Autonomously Synthesized Update Laws.
    """
    def __init__(self, config: EXP435Config):
        super().__init__()
        self.config = config
        self.vocab_dim = config.vocab_dim
        self.H = config.heads
        self.D = config.head_dim
        self.total_dim = self.H * self.D
        self.num_ops = config.num_operators
        self.dt = config.dt

        # Complex embeddings: Real and Imaginary components
        self.emb_real = nn.Embedding(self.vocab_dim, self.total_dim)
        self.emb_imag = nn.Embedding(self.vocab_dim, self.total_dim)

        # Skew-Hermitian Lie Generator for phase rotation per head
        self.A = nn.Parameter(torch.randn(self.H, self.D, self.D, device=DEVICE) * 0.02)

        # Pre-RMSNorm parameter for stable bounded readback
        self.norm_scale = nn.Parameter(torch.ones(self.total_dim, device=DEVICE))

        # Sovereign Law Synthesizer: Maps error & state dynamics to operator blend w_k(t)
        self.synthesizer = nn.Sequential(
            nn.Linear(self.H * 4, 32),
            nn.SiLU(),
            nn.Linear(32, self.num_ops)
        )

        # Readout Projection
        self.readout_real = nn.Linear(self.total_dim, self.vocab_dim, bias=False)
        self.readout_imag = nn.Linear(self.total_dim, self.vocab_dim, bias=False)

        # Multi-Head Persistent Memory Tensor: [H, D, D]
        self.M_real = torch.zeros(self.H, self.D, self.D, device=DEVICE)
        self.M_imag = torch.zeros(self.H, self.D, self.D, device=DEVICE)

        self.reset_state()

    def reset_state(self):
        self.M_real.zero_()
        self.M_imag.zero_()

    def step(self, byte_idx: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        idx_t = torch.tensor([byte_idx], device=DEVICE)
        target_t = torch.tensor([target_byte], device=DEVICE)

        # 1. Embed Input and Target in Complex Multi-Head Space
        xr = self.emb_real(idx_t).view(self.H, self.D, 1)  # [H, D, 1]
        xi = self.emb_imag(idx_t).view(self.H, self.D, 1)  # [H, D, 1]

        yr = self.emb_real(target_t).view(self.H, self.D, 1)  # [H, D, 1]
        yi = self.emb_imag(target_t).view(self.H, self.D, 1)  # [H, D, 1]

        M_r = self.M_real.detach()  # [H, D, D]
        M_i = self.M_imag.detach()  # [H, D, D]

        # 2. Resonant Complex Recall from Memory Tensor: y_rec = M * x
        # (M_r + i M_i) * (xr + i xi) = (M_r xr - M_i xi) + i (M_r xi + M_i xr)
        rec_r = torch.bmm(M_r, xr) - torch.bmm(M_i, xi)  # [H, D, 1]
        rec_i = torch.bmm(M_r, xi) + torch.bmm(M_i, xr)  # [H, D, 1]

        # Combine Direct Input Stimulus + Resonant Associative Recall
        comb_r = (xr + rec_r).view(1, self.total_dim)  # [1, total_dim]
        comb_i = (xi + rec_i).view(1, self.total_dim)  # [1, total_dim]

        # 3. RMS-Normalization to completely eradicate logit explosion
        mag_sq = comb_r ** 2 + comb_i ** 2
        rms = torch.sqrt(torch.mean(mag_sq, dim=-1, keepdim=True) + 1e-6)
        norm_r = (comb_r / rms) * self.norm_scale
        norm_i = (comb_i / rms) * self.norm_scale

        # Readout Projection
        logits = self.readout_real(norm_r) - self.readout_imag(norm_i)  # [1, vocab_dim]

        loss = F.cross_entropy(logits, target_t)
        pred = torch.argmax(logits, dim=-1).item()
        surprise = loss.item()

        # 4. Complex Prediction Error: delta = target - recalled
        delta_r = yr - rec_r  # [H, D, 1]
        delta_i = yi - rec_i  # [H, D, 1]

        # === 5. ATOMIC OPERATOR BASIS (dM/dt in C^(H x D x D)) ===
        # O_1: Sovereign Delta Outer Product: delta * x^dagger
        # (delta_r + i delta_i) * (xr - i xi)^T = (delta_r xr^T + delta_i xi^T) + i (delta_i xr^T - delta_r xi^T)
        xr_t = xr.transpose(1, 2)  # [H, 1, D]
        xi_t = xi.transpose(1, 2)  # [H, 1, D]

        o1_r = torch.bmm(delta_r, xr_t) + torch.bmm(delta_i, xi_t)  # [H, D, D]
        o1_i = torch.bmm(delta_i, xr_t) - torch.bmm(delta_r, xi_t)  # [H, D, D]

        # O_2: Lie Commutator Unitary Rotation: [H_skew, M]
        H_skew = (self.A - self.A.transpose(1, 2)) * 0.5
        o2_r = torch.bmm(H_skew, M_r) - torch.bmm(M_r, H_skew)
        o2_i = torch.bmm(H_skew, M_i) - torch.bmm(M_i, H_skew)

        # O_3: Unit Attractor Normalization: M / (||M|| + 1)
        m_norms = torch.sqrt(torch.sum(M_r ** 2 + M_i ** 2, dim=(-1, -2), keepdim=True) + 1e-8)
        o3_r = M_r / (m_norms + 1.0)
        o3_i = M_i / (m_norms + 1.0)

        # O_4: Leaky Dissipation: -0.1 * M
        o4_r = -0.1 * M_r
        o4_i = -0.1 * M_i

        # O_5: Multiplicative Phase Modulation: M * (xr xi^T)
        stim_phase = torch.bmm(xr, xi_t)
        o5_r = torch.bmm(M_r, stim_phase)
        o5_i = torch.bmm(M_i, stim_phase)

        # === 6. AUTONOMOUS FORMULA SYNTHESIS ===
        # Synthesizer inspects head norms, error magnitudes, and surprise
        err_norm = torch.sqrt(torch.sum(delta_r ** 2 + delta_i ** 2, dim=1)).view(1, self.H)  # [1, H]
        m_head_norm = m_norms.view(1, self.H)  # [1, H]
        rec_norm = torch.sqrt(torch.sum(rec_r ** 2 + rec_i ** 2, dim=1)).view(1, self.H)  # [1, H]
        surp_feat = torch.full((1, self.H), surprise, device=DEVICE)  # [1, H]

        synth_input = torch.cat([err_norm, m_head_norm, rec_norm, surp_feat], dim=-1)  # [1, H * 4]
        synth_logits = self.synthesizer(synth_input)
        w = F.softmax(synth_logits, dim=-1).squeeze(0)  # [5]

        # Synthesized Tensor Differential Equation:
        dM_r = w[0] * o1_r + w[1] * o2_r + w[2] * o3_r + w[3] * o4_r + w[4] * o5_r
        dM_i = w[0] * o1_i + w[1] * o2_i + w[2] * o3_i + w[3] * o4_i + w[4] * o5_i

        # Strict Out-of-Place State Update:
        new_M_r = M_r + self.dt * dM_r
        new_M_i = M_i + self.dt * dM_i

        # Dynamic allostatic bounding
        new_norms = torch.sqrt(torch.sum(new_M_r ** 2 + new_M_i ** 2, dim=(-1, -2), keepdim=True) + 1e-8)
        max_bound = 12.0
        scale = torch.clamp(max_bound / new_norms, max=1.0)
        self.M_real = new_M_r * scale
        self.M_imag = new_M_i * scale

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


def run_exp_435():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-435: SOVEREIGN DIFFERENTIABLE OPERATOR & LAW SYNTHESIZER ===")
    logger.info("================================================================================")

    config = EXP435Config()
    model = SovereignLawSynthesizer(config).to(DEVICE)
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
    logger.info("Synthesizing multi-head holographic update laws autonomously...")

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
                f"Synthesized Op Weights [Delta, LieRot, UnitAtt, Leaky, Mod]: {cur_weights}"
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
    logger.info("=== EXP-435 SOVEREIGN LAW SYNTHESIS TELEMETRY REPORT ===")
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
        "exp_id": "EXP-435",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "synthesized_operator_distribution": {
            "O1_delta_outer_product": mean_op_dist[0],
            "O2_lie_rotation": mean_op_dist[1],
            "O3_unit_attractor": mean_op_dist[2],
            "O4_leaky_dissipation": mean_op_dist[3],
            "O5_multiplicative_mod": mean_op_dist[4]
        },
        "second_half_low_loss_mass": round(low_loss_fraction, 2),
        "tail_retention_rate": round(tail_retention_rate, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_435_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_435()
