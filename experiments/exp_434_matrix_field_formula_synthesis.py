"""
EXP-434: Autonomous Matrix-Field Operator Synthesis (AMFOS)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Problem Diagnosis (EXP-433):
Vector-only state Psi in C^512 hit a capacity wall at ~30% single-pass accuracy due to vector interference.

Solution (EXP-434):
Upgrade the universal substrate from 1D vector Psi to 2D Complex Matrix Field M in C^(128 x 128)
combined with Channel-Wise / Block-Wise Autonomous Operator Synthesis!

Atomic Matrix Operator Basis (dM/dt):
O_1: Associative Outer Product Trace (Hebbian phase encoding: e_stim * e_target^dagger)
O_2: Unitary Phase Rotation (Commutator Lie algebra: [H_skew, M] = H M - M H)
O_3: Bilinear Matrix Scaling & Transport (M * W_trans)
O_4: Nonlinear Matrix Attractor Well (Saturating energy compression: M / (1 + ||M||^2))
O_5: Error-Driven Matrix Repeller (Outer product error pulse away from mistake projection)

Sovereign Channel-Wise Formula Synthesis:
w_k(t) = Softmax(W_synth * state_summary)  --> Channel/Block adaptive operator selection!
dM/dt = sum_{k=1}^5 w_k(t) * O_k(M, stim, target)
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
logger = logging.getLogger("EXP-434-AMFOS")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP434Config:
    exp_id: str = "EXP-434"
    vocab_dim: int = 258
    mat_dim: int = 128
    num_operators: int = 5
    learning_rate: float = 0.015
    stream_length: int = 4000
    dt: float = 0.35
    device_str: str = DEVICE_STR


class AutonomousMatrixFieldSynthesizer(nn.Module):
    """
    Karyon's 2D Holographic Complex Matrix Substrate.
    Autonomously synthesizes its own 2D matrix field update laws dM/dt on the fly.
    """
    def __init__(self, config: EXP434Config):
        super().__init__()
        self.config = config
        self.vocab_dim = config.vocab_dim
        self.mat_dim = config.mat_dim
        self.dt = config.dt
        self.num_ops = config.num_operators

        # 1. Complex Embeddings for Input Stimulus
        self.embed_real = nn.Embedding(self.vocab_dim, self.mat_dim)
        self.embed_imag = nn.Embedding(self.vocab_dim, self.mat_dim)

        # 2. Skew-Hermitian Lie Operator Generator for Matrix Phase Rotation
        self.A = nn.Parameter(torch.randn(self.mat_dim, self.mat_dim, device=DEVICE) * 0.02)

        # 3. Bilinear Matrix Transport Weights
        self.W_trans_r = nn.Linear(self.mat_dim, self.mat_dim, bias=False)
        self.W_trans_i = nn.Linear(self.mat_dim, self.mat_dim, bias=False)

        # 4. Attractor and Repeller Parameters
        self.gamma_rep = nn.Parameter(torch.tensor(0.8, device=DEVICE))

        # 5. Channel-Adaptive Formula Synthesizer
        self.synthesizer = nn.Sequential(
            nn.Linear(self.mat_dim * 2, 64),
            nn.SiLU(),
            nn.Linear(64, self.num_ops)
        )

        # 6. Born-Rule Observable Readout
        self.readout_real = nn.Linear(self.mat_dim, self.vocab_dim, bias=False)
        self.readout_imag = nn.Linear(self.mat_dim, self.vocab_dim, bias=False)

        # 7. Persistent Complex 2D Matrix State M = M_real + i * M_imag in C^(128 x 128)
        self.M_real = torch.randn(self.mat_dim, self.mat_dim, device=DEVICE) * 0.01
        self.M_imag = torch.randn(self.mat_dim, self.mat_dim, device=DEVICE) * 0.01

        # Dynamic Repeller Memory Matrix
        self.rep_M_real = torch.zeros(self.mat_dim, self.mat_dim, device=DEVICE)
        self.rep_M_imag = torch.zeros(self.mat_dim, self.mat_dim, device=DEVICE)
        self.rep_strength = 0.0

        self.reset_state()

    def reset_state(self):
        self.M_real = torch.randn(self.mat_dim, self.mat_dim, device=DEVICE) * 0.01
        self.M_imag = torch.randn(self.mat_dim, self.mat_dim, device=DEVICE) * 0.01
        self.rep_M_real = torch.zeros(self.mat_dim, self.mat_dim, device=DEVICE)
        self.rep_M_imag = torch.zeros(self.mat_dim, self.mat_dim, device=DEVICE)
        self.rep_strength = 0.0

    def step(self, byte_idx: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        idx_t = torch.tensor([byte_idx], device=DEVICE)
        target_t = torch.tensor([target_byte], device=DEVICE)

        # Complex Input Vector e_x = e_r + i * e_i
        e_r = self.embed_real(idx_t)  # [1, mat_dim]
        e_i = self.embed_imag(idx_t)  # [1, mat_dim]

        # Complex Target Vector e_y = t_r + i * t_i
        t_r = self.embed_real(target_t)  # [1, mat_dim]
        t_i = self.embed_imag(target_t)  # [1, mat_dim]

        M_r_det = self.M_real.detach()
        M_i_det = self.M_imag.detach()

        # === 1. RESONANT RECALL FROM 2D MATRIX FIELD ===
        # y_recalled = M * e_x  (Complex Matrix-Vector Product)
        e_r_col = e_r.t()  # [mat_dim, 1]
        e_i_col = e_i.t()  # [mat_dim, 1]

        y_rec_r = torch.matmul(M_r_det, e_r_col) - torch.matmul(M_i_det, e_i_col)  # [mat_dim, 1]
        y_rec_i = torch.matmul(M_r_det, e_i_col) + torch.matmul(M_i_det, e_r_col)  # [mat_dim, 1]

        y_rec_r = y_rec_r.t()  # [1, mat_dim]
        y_rec_i = y_rec_i.t()  # [1, mat_dim]

        # Combine input stimulus + matrix recall
        field_r = e_r + y_rec_r
        field_i = e_i + y_rec_i

        # === 2. BORN-RULE READOUT PROJECTION ===
        logits = self.readout_real(field_r) - self.readout_imag(field_i)  # [1, vocab_dim]

        loss = F.cross_entropy(logits, target_t)
        pred = torch.argmax(logits, dim=-1).item()
        surprise = loss.item()

        # === 3. EVALUATE ATOMIC MATRIX OPERATORS (dM/dt) ===
        # O_1: Associative Holographic Trace (Outer product: e_x * e_y^T)
        o1_r = torch.matmul(e_r_col, t_r) + torch.matmul(e_i_col, t_i)  # [mat_dim, mat_dim]
        o1_i = torch.matmul(e_i_col, t_r) - torch.matmul(e_r_col, t_i)  # [mat_dim, mat_dim]

        # O_2: Lie Algebra Commutator Phase Rotation ([H_skew, M] = H M - M H)
        H_skew = (self.A - self.A.t()) * 0.5
        o2_r = torch.matmul(H_skew, M_r_det) - torch.matmul(M_r_det, H_skew)
        o2_i = torch.matmul(H_skew, M_i_det) - torch.matmul(M_i_det, H_skew)

        # O_3: Bilinear Matrix Transport
        o3_r = self.W_trans_r(M_r_det)
        o3_i = self.W_trans_i(M_i_det)

        # O_4: Landau Saturating Matrix Attractor Well
        m_norm_sq = torch.sum(M_r_det ** 2 + M_i_det ** 2) + 1.0
        o4_r = -M_r_det / m_norm_sq
        o4_i = -M_i_det / m_norm_sq

        # O_5: Matrix Error Repeller Pulse
        diff_M_r = M_r_det - self.rep_M_real
        diff_M_i = M_i_det - self.rep_M_imag
        rep_dist_sq = torch.sum(diff_M_r ** 2 + diff_M_i ** 2) + 0.1
        rep_factor = (self.rep_strength * F.softplus(self.gamma_rep)) / (rep_dist_sq ** 2)
        o5_r = rep_factor * diff_M_r
        o5_i = rep_factor * diff_M_i

        # === 4. SOVEREIGN CHANNEL-ADAPTIVE FORMULA SYNTHESIS ===
        diag_r = torch.diag(M_r_det).unsqueeze(0)  # [1, mat_dim]
        diag_i = torch.diag(M_i_det).unsqueeze(0)  # [1, mat_dim]
        state_summary = torch.cat([diag_r, diag_i], dim=-1)  # [1, mat_dim * 2]

        synth_logits = self.synthesizer(state_summary)
        w = F.softmax(synth_logits, dim=-1).squeeze(0)  # [5]

        # Synthesized Matrix Differential Equation:
        dM_r = w[0] * o1_r + w[1] * o2_r + w[2] * o3_r + w[3] * o4_r + w[4] * o5_r
        dM_i = w[0] * o1_i + w[1] * o2_i + w[2] * o3_i + w[3] * o4_i + w[4] * o5_i

        # Continuous Integration (Strict Out-of-Place Propagation)
        new_M_r = M_r_det + dM_r * self.dt
        new_M_i = M_i_det + dM_i * self.dt

        # Norm bounding for numerical stability
        mat_norm = torch.sqrt(torch.sum(new_M_r ** 2 + new_M_i ** 2) + 1e-8)
        if mat_norm > 15.0:
            new_M_r = new_M_r * (15.0 / mat_norm)
            new_M_i = new_M_i * (15.0 / mat_norm)

        self.M_real = new_M_r
        self.M_imag = new_M_i

        # Update repeller state under surprise
        with torch.no_grad():
            if pred != target_byte and surprise > 1.0:
                self.rep_M_real = self.M_real.clone()
                self.rep_M_imag = self.M_imag.clone()
                self.rep_strength = min(self.rep_strength + 0.8, 3.0)
            else:
                self.rep_strength *= 0.92

            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        meta = {
            "loss": surprise,
            "pred": pred,
            "target": target_byte,
            "is_correct": (pred == target_byte),
            "entropy": entropy,
            "weights": w.detach().cpu().numpy().tolist(),
            "rep_strength": self.rep_strength
        }
        return loss, meta


def run_exp_434():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-434: AUTONOMOUS MATRIX-FIELD OPERATOR SYNTHESIS (AMFOS) ===")
    logger.info("================================================================================")

    config = EXP434Config()
    model = AutonomousMatrixFieldSynthesizer(config).to(DEVICE)
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
    logger.info("Synthesizing 2D Complex Matrix Field equations autonomously across stream...")

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
                f"Mean Op Weights [Assoc, LieRot, Trans, Att, Rep]: {cur_weights}"
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
    logger.info("=== EXP-434 MATRIX-FIELD FORMULA SYNTHESIS TELEMETRY REPORT ===")
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
        "exp_id": "EXP-434",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "synthesized_operator_distribution": {
            "O1_associative_trace": mean_op_dist[0],
            "O2_lie_rotation": mean_op_dist[1],
            "O3_matrix_transport": mean_op_dist[2],
            "O4_landau_attractor": mean_op_dist[3],
            "O5_matrix_repeller": mean_op_dist[4]
        },
        "second_half_low_loss_mass": round(low_loss_fraction, 2),
        "tail_retention_rate": round(tail_retention_rate, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_434_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_434()
