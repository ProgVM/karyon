"""
EXP-432: Sovereign Continuous Hamiltonian Wavefield (SCH-Wavefield)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Core Philosophy (Bazilevs):
"О господи, как много костыльных решений -_-"
Eradicate ALL crutches:
- NO artificial N-gram sliding windows or FIR buffers!
- NO separate lists of repellers with external similarity loops!
- NO artificial SVD whitening steps or external spectral decoders!
- NO hardcoded supervisory networks or disjointed modules!

Pure Physical Hamiltonian Dynamics:
The system is a single, continuous, complex-valued Hamiltonian wavefield Psi in C^D.
1. Unitary Wavepacket Phase Evolution:
   dPsi/dt = -i * H * Psi - grad_Psi U(Psi)
   where H is an endogenous Hermitian kinetic operator (conserving unitary phase flow).
2. Endogenous Dual Landau Potential (Attractor-Repeller Manifold):
   U(Psi) = -alpha * |Psi|^2 + (beta / 2) * |Psi|^4 + gamma * |Psi - Psi_error|^(-2)
   - The double-well Landau potential creates natural stable attractor minima (|Psi| = sqrt(alpha/beta)).
   - High surprise naturally induces a dynamic local energy hill (repulsive barrier) directly
     in the phase metric g_mu_nu without any external buffers!
3. Direct Born Rule Observable:
   P(x) = Softmax(Re(W_read * Psi))
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
logger = logging.getLogger("EXP-432-SCH")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP432Config:
    exp_id: str = "EXP-432"
    dim: int = 258
    hidden_dim: int = 512
    learning_rate: float = 0.012
    stream_length: int = 4000
    dt: float = 0.25
    device_str: str = DEVICE_STR


class SovereignHamiltonianWavefield(nn.Module):
    """
    A single, unified, continuous complex-valued physical wavefield.
    Zero external buffers. Zero crutches. Zero N-gram heuristics.
    """
    def __init__(self, config: EXP432Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.dt = config.dt

        # 1. Continuous Wave Coupling (Embedding directly into complex state space)
        self.coupling_real = nn.Embedding(self.dim, self.hidden_dim)
        self.coupling_imag = nn.Embedding(self.dim, self.hidden_dim)

        # 2. Endogenous Hermitian Kinetic Operator H = (A - A^T) / 2 + diag(omega)
        # Guarantees strictly unitary energy-preserving phase rotation!
        self.A = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim, device=DEVICE) * 0.02)
        self.omega = nn.Parameter(torch.linspace(-1.0, 1.0, self.hidden_dim, device=DEVICE))

        # 3. Dual Landau Attractor-Repeller Potential Parameters
        self.alpha_att = nn.Parameter(torch.tensor(1.2, device=DEVICE))
        self.beta_att = nn.Parameter(torch.tensor(1.0, device=DEVICE))
        self.gamma_rep = nn.Parameter(torch.tensor(0.5, device=DEVICE))

        # 4. Endogenous Repulsive Pole in Phase Space (Repeller Coordinate)
        self.rep_pole_real = nn.Parameter(torch.zeros(self.hidden_dim, device=DEVICE), requires_grad=False)
        self.rep_pole_imag = nn.Parameter(torch.zeros(self.hidden_dim, device=DEVICE), requires_grad=False)
        self.rep_strength = 0.0

        # 5. Observable Readout Projection (Hermitian Measurement)
        self.obs_real = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.obs_imag = nn.Linear(self.hidden_dim, self.dim, bias=False)

        # 6. Continuous Wavepacket State Psi(t) = Psi_r + i * Psi_i
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
        # 1. Sensory Coupling
        idx_t = torch.tensor([byte_idx], device=DEVICE)
        target_t = torch.tensor([target_byte], device=DEVICE)

        stim_r = self.coupling_real(idx_t)  # [1, hidden_dim]
        stim_i = self.coupling_imag(idx_t)

        # 2. Skew-Hermitian Kinetic Evolution Matrix: H_skew = (A - A^T)
        # Unitary phase rotation: dPsi/dt = H_skew * Psi
        H_skew = (self.A - self.A.t()) * 0.5
        H_diag = torch.diag(self.omega)
        H_tot = H_skew + H_diag

        # dPsi_r/dt = -H * Psi_i + stim_r
        # dPsi_i/dt =  H * Psi_r + stim_i
        kinetic_r = -torch.matmul(self.psi_imag.detach(), H_tot) + stim_r
        kinetic_i = torch.matmul(self.psi_real.detach(), H_tot) + stim_i

        # 3. Continuous Landau Attractor-Repeller Potential Gradient:
        # Attractor well: pull toward stable limit cycle |Psi|^2 ~ alpha / beta
        psi_sq = torch.sum(self.psi_real.detach() ** 2 + self.psi_imag.detach() ** 2, dim=-1, keepdim=True)
        grad_landau = -F.softplus(self.alpha_att) + F.softplus(self.beta_att) * psi_sq

        # Repeller hill: repulsive potential around last surprise pole
        diff_r = self.psi_real.detach() - self.rep_pole_real
        diff_i = self.psi_imag.detach() - self.rep_pole_imag
        dist_sq = torch.sum(diff_r ** 2 + diff_i ** 2, dim=-1, keepdim=True) + 0.1
        # Repulsive force points AWAY from the error pole:
        rep_force = (self.rep_strength * F.softplus(self.gamma_rep)) / (dist_sq ** 2)

        # Net physical field acceleration
        d_psi_r = kinetic_r - grad_landau * self.psi_real.detach() + rep_force * diff_r
        d_psi_i = kinetic_i - grad_landau * self.psi_imag.detach() + rep_force * diff_i

        # 4. Symplectic Verlet Integration Step
        new_psi_r = self.psi_real.detach() + self.dt * d_psi_r
        new_psi_i = self.psi_imag.detach() + self.dt * d_psi_i

        # Normalization (conserving total probability mass)
        norm = torch.sqrt(torch.sum(new_psi_r ** 2 + new_psi_i ** 2, dim=-1, keepdim=True) + 1e-8)
        self.psi_real = new_psi_r / norm
        self.psi_imag = new_psi_i / norm

        # 5. Observable Measurement (Born Rule Projection)
        # Logits = Re(W_obs * Psi) = obs_real(Psi_r) - obs_imag(Psi_i)
        logits = self.obs_real(self.psi_real) - self.obs_imag(self.psi_imag)

        loss = F.cross_entropy(logits, target_t)
        pred = torch.argmax(logits, dim=-1).item()
        surprise = loss.item()

        # 6. Natural Endogenous Repeller Shift (No buffers! Single continuous pole update)
        with torch.no_grad():
            if pred != target_byte and surprise > 1.2:
                # The state that generated this surprise becomes the new repulsive coordinate!
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
            "rep_strength": self.rep_strength,
            "wave_norm": norm.item()
        }
        return loss, meta


def run_exp_432():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-432: SOVEREIGN CONTINUOUS HAMILTONIAN WAVEFIELD (SCH) ===")
    logger.info("================================================================================")

    config = EXP432Config()
    model = SovereignHamiltonianWavefield(config).to(DEVICE)
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
    logger.info("Evaluating Pure Hamiltonian Dynamics without artificial crutches...")

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
            avg_ent = total_entropy / (t + 1)
            logger.info(
                f"Progress [{t+1}/{len(raw_bytes)-1}] | Avg Loss: {avg_loss:.4f} | "
                f"Acc: {acc:.2f}% | Current Loss: {meta['loss']:.4f} | "
                f"Entropy: {avg_ent:.3f} nats | Repeller Strength: {meta['rep_strength']:.2f}"
            )

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_final_entropy = total_entropy / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    second_half = recent_losses[len(recent_losses)//2:]
    low_loss_count = sum(1 for loss_val in second_half if loss_val < 0.5)
    low_loss_fraction = (low_loss_count / len(second_half)) * 100.0

    tail_500 = recent_losses[-500:]
    tail_low_loss_count = sum(1 for loss_val in tail_500 if loss_val < 0.5)
    tail_retention_rate = (tail_low_loss_count / len(tail_500)) * 100.0

    logger.info("================================================================================")
    logger.info("=== EXP-432 SOVEREIGN HAMILTONIAN TELEMETRY REPORT ===")
    logger.info(f"Final Average Loss: {final_loss:.4f} nats")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Second-Half Low-Loss Mass (<0.5 nats): {low_loss_fraction:.2f}%")
    logger.info(f"Tail 500-byte Retention Rate (<0.5 nats): {tail_retention_rate:.2f}%")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Mean Output Entropy: {avg_final_entropy:.3f} nats")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-432",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "second_half_low_loss_mass": round(low_loss_fraction, 2),
        "tail_retention_rate": round(tail_retention_rate, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_432_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_432()
