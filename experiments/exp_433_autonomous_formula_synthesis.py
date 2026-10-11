"""
EXP-433: Autonomous Endogenous Formula Synthesis Substrate (AEFS)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Core Philosophy (Bazilevs):
"Карион сам синтезирует формулы и правила, которые ему нужны, причём это должно быть на чистой архитектуре"
Eradicate ALL hardcoded human rules:
- No hardcoded update formulas!
- No rigid supervisory networks or disjointed modules!
- Karyon autonomously synthesizes the differential equation governing its own state dPsi/dt
  from an atomic basis of universal mathematical operators under Variational Free Energy!

Atomic Mathematical Operator Basis:
O_1: Linear Transport & Projection (Leaky continuous drift)
O_2: Unitary Phase Rotation / Lie Algebra (Skew-Hermitian phase conservation: [A - A^T] * Psi)
O_3: Bilinear Multiplicative Conjunction (Direct causal interaction: Psi * stim)
O_4: Nonlinear Attractor Well (Landau double-well energy compression)
O_5: Error-Driven Dynamic Repeller Pulse (Local repulsive potential away from surprise pole)

Endogenous Synthesis Mechanism:
w_k(t) = Softmax(W_synth * Psi(t) + b_synth)
dPsi/dt = sum_{k=1}^5 w_k(t) * O_k(Psi(t), stim(t))
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
logger = logging.getLogger("EXP-433-AEFS")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP433Config:
    exp_id: str = "EXP-433"
    dim: int = 258
    hidden_dim: int = 512
    num_operators: int = 5
    learning_rate: float = 0.012
    stream_length: int = 4000
    dt: float = 0.25
    device_str: str = DEVICE_STR


class EndogenousFormulaSynthesizer(nn.Module):
    """
    Karyon's Clean Universal Mathematical Substrate.
    Autonomously synthesizes its own dynamic update laws without hardcoded human formulas.
    """
    def __init__(self, config: EXP433Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.dt = config.dt
        self.num_ops = config.num_operators

        # 1. Sensory Coupling into Complex Field
        self.coupling_real = nn.Embedding(self.dim, self.hidden_dim)
        self.coupling_imag = nn.Embedding(self.dim, self.hidden_dim)

        # 2. Atomic Operator Basis Parameters:
        # Operator 1: Linear Transport
        self.W_trans = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # Operator 2: Unitary Phase Rotation (Skew-Hermitian matrix)
        self.A = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim, device=DEVICE) * 0.02)
        self.omega = nn.Parameter(torch.linspace(-1.0, 1.0, self.hidden_dim, device=DEVICE))

        # Operator 3: Bilinear Scaling
        self.bilinear_scale = nn.Parameter(torch.ones(self.hidden_dim, device=DEVICE))

        # Operator 4: Landau Attractor Well
        self.alpha_att = nn.Parameter(torch.tensor(1.2, device=DEVICE))
        self.beta_att = nn.Parameter(torch.tensor(1.0, device=DEVICE))

        # Operator 5: Dynamic Repeller Pole Coordinate
        self.rep_pole_real = nn.Parameter(torch.zeros(self.hidden_dim, device=DEVICE), requires_grad=False)
        self.rep_pole_imag = nn.Parameter(torch.zeros(self.hidden_dim, device=DEVICE), requires_grad=False)
        self.rep_strength = 0.0
        self.gamma_rep = nn.Parameter(torch.tensor(0.5, device=DEVICE))

        # 3. Sovereign Formula Synthesizer (State-derived Operator Mixture)
        # Directly measures Psi and synthesizes the blend w_k of mathematical laws!
        self.synthesizer = nn.Linear(self.hidden_dim * 2, self.num_ops)

        # 4. Born Rule Readout Projection
        self.obs_real = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.obs_imag = nn.Linear(self.hidden_dim, self.dim, bias=False)

        # 5. Continuous Wavepacket State Psi(t)
        self.psi_real = torch.randn(1, self.hidden_dim, device=DEVICE) * 0.05
        self.psi_imag = torch.randn(1, self.hidden_dim, device=DEVICE) * 0.05
        self._normalize_wavepacket()

    def _normalize_wavepacket(self):
        norm = torch.sqrt(torch.sum(self.psi_real ** 2 + self.psi_imag ** 2) + 1e-8)
        self.psi_real = self.psi_real / norm
        self.psi_imag = self.psi_imag / norm

    def reset_state(self):
        self.psi_real = torch.randn(1, self.hidden_dim, device=DEVICE) * 0.05
        self.psi_imag = torch.randn(1, self.hidden_dim, device=DEVICE) * 0.05
        self._normalize_wavepacket()
        self.rep_pole_real.zero_()
        self.rep_pole_imag.zero_()
        self.rep_strength = 0.0

    def step(self, byte_idx: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        idx_t = torch.tensor([byte_idx], device=DEVICE)
        target_t = torch.tensor([target_byte], device=DEVICE)

        stim_r = self.coupling_real(idx_t)
        stim_i = self.coupling_imag(idx_t)

        psi_r_det = self.psi_real.detach()
        psi_i_det = self.psi_imag.detach()

        # === EVALUATE ATOMIC OPERATORS ===
        # O_1: Linear Transport
        o1_r = self.W_trans(psi_r_det) + stim_r
        o1_i = self.W_trans(psi_i_det) + stim_i

        # O_2: Unitary Phase Rotation (Skew-Hermitian Lie Flow)
        H_skew = (self.A - self.A.t()) * 0.5
        H_diag = torch.diag(self.omega)
        H_tot = H_skew + H_diag
        o2_r = -torch.matmul(psi_i_det, H_tot) + stim_r
        o2_i = torch.matmul(psi_r_det, H_tot) + stim_i

        # O_3: Bilinear Multiplicative Interaction
        o3_r = (psi_r_det * stim_r - psi_i_det * stim_i) * self.bilinear_scale
        o3_i = (psi_r_det * stim_i + psi_i_det * stim_r) * self.bilinear_scale

        # O_4: Landau Attractor Well (Nonlinear Energy Minimum)
        psi_sq = torch.sum(psi_r_det ** 2 + psi_i_det ** 2, dim=-1, keepdim=True)
        landau_factor = -F.softplus(self.alpha_att) + F.softplus(self.beta_att) * psi_sq
        o4_r = -landau_factor * psi_r_det + stim_r
        o4_i = -landau_factor * psi_i_det + stim_i

        # O_5: Error Repeller Impulse (Repulsive gradient away from mistake pole)
        diff_r = psi_r_det - self.rep_pole_real
        diff_i = psi_i_det - self.rep_pole_imag
        dist_sq = torch.sum(diff_r ** 2 + diff_i ** 2, dim=-1, keepdim=True) + 0.1
        rep_force = (self.rep_strength * F.softplus(self.gamma_rep)) / (dist_sq ** 2)
        o5_r = rep_force * diff_r + stim_r
        o5_i = rep_force * diff_i + stim_i

        # === ENDOGENOUS FORMULA SYNTHESIS ===
        # State self-reflection: system inspects its own state configuration
        psi_concat = torch.cat([psi_r_det, psi_i_det], dim=-1)
        synth_logits = self.synthesizer(psi_concat)
        formula_weights = F.softmax(synth_logits, dim=-1)  # [1, 5]

        w = formula_weights.squeeze(0)
        # Synthesized Differential Equation:
        d_psi_r = w[0] * o1_r + w[1] * o2_r + w[2] * o3_r + w[3] * o4_r + w[4] * o5_r
        d_psi_i = w[0] * o1_i + w[1] * o2_i + w[2] * o3_i + w[3] * o4_i + w[4] * o5_i

        # Continuous Integration
        new_psi_r = psi_r_det + self.dt * d_psi_r
        new_psi_i = psi_i_det + self.dt * d_psi_i

        # Unitary probability mass conservation
        norm = torch.sqrt(torch.sum(new_psi_r ** 2 + new_psi_i ** 2, dim=-1, keepdim=True) + 1e-8)
        self.psi_real = new_psi_r / norm
        self.psi_imag = new_psi_i / norm

        # Observable Readout Projection (Measurement)
        logits = self.obs_real(self.psi_real) - self.obs_imag(self.psi_imag)

        loss = F.cross_entropy(logits, target_t)
        pred = torch.argmax(logits, dim=-1).item()
        surprise = loss.item()

        # Update repeller pole naturally under high surprise
        with torch.no_grad():
            if pred != target_byte and surprise > 1.2:
                self.rep_pole_real.copy_(self.psi_real.squeeze(0))
                self.rep_pole_imag.copy_(self.psi_imag.squeeze(0))
                self.rep_strength = min(self.rep_strength + 0.5, 2.0)
            else:
                self.rep_strength *= 0.95

            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        meta = {
            "loss": surprise,
            "pred": pred,
            "target": target_byte,
            "is_correct": (pred == target_byte),
            "entropy": entropy,
            "weights": formula_weights.detach().cpu().numpy().tolist()[0],
            "rep_strength": self.rep_strength
        }
        return loss, meta


def run_exp_433():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-433: AUTONOMOUS ENDOGENOUS FORMULA SYNTHESIS (AEFS) ===")
    logger.info("================================================================================")

    config = EXP433Config()
    model = EndogenousFormulaSynthesizer(config).to(DEVICE)
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
    logger.info("Synthesizing dynamic equations autonomously across byte stream...")

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
                f"Mean Op Weights [Trans, Rot, Bili, Att, Rep]: {cur_weights}"
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
    logger.info("=== EXP-433 AUTONOMOUS FORMULA SYNTHESIS TELEMETRY REPORT ===")
    logger.info(f"Final Average Loss: {final_loss:.4f} nats")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Synthesized Formula Weight Distribution: {mean_op_dist}")
    logger.info(f"Second-Half Low-Loss Mass (<0.5 nats): {low_loss_fraction:.2f}%")
    logger.info(f"Tail 500-byte Retention Rate (<0.5 nats): {tail_retention_rate:.2f}%")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Mean Output Entropy: {avg_final_entropy:.3f} nats")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-433",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "synthesized_operator_distribution": {
            "O1_linear_transport": mean_op_dist[0],
            "O2_unitary_rotation": mean_op_dist[1],
            "O3_bilinear_interaction": mean_op_dist[2],
            "O4_landau_attractor": mean_op_dist[3],
            "O5_error_repeller": mean_op_dist[4]
        },
        "second_half_low_loss_mass": round(low_loss_fraction, 2),
        "tail_retention_rate": round(tail_retention_rate, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_433_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_433()
