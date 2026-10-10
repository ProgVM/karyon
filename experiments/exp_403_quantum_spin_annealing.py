"""
EXP-403: Quantum-Thermodynamic Spin Wavefield & Gauge Annealing (QTSW-GA)
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Date: October 2026
Standard: KEP v16.0 Sovereign Master (Rubicon 400 Series, Principle 27 Sovereign Genesis)

Core Scientific Breakthroughs in EXP-403:
1. Complex Heisenberg Spin Wavefield Psi_j in C^D (D=258):
   State is governed by a dissipative complex Ginzburg-Landau / Quantum Spin Glass Hamiltonian:
   H_Q = - 1/2 sum_{j,k} J_{jk} (Psi_j^* Psi_k + h.c.) - sum_j h_j(u_t)^* Psi_j + lambda/2 sum_j |Psi_j|^4
2. Dissipative Complex SDE Integration:
   d Psi / dt = - (i/hbar * dH/dPsi^* + gamma(T_t) * dH/dPsi^*)
   combining coherent phase precession with thermodynamic energy dissipation.
3. Self-Organizing Critical Gauge Temperature T_t:
   T_t dynamically adapts to maintain order parameter near the critical threshold,
   balancing quantum wave coherence and thermodynamic susceptibility.
4. Born Rule Next-Byte Probability:
   P(j) = |Psi_j|^2 / sum_k |Psi_k|^2
"""

import math
import time
import json
import logging
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-403-QTSW")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP403Config:
    exp_id: str = "EXP-403"
    dim: int = 258
    learning_rate: float = 0.005
    target_norm_crit: float = 1.0
    stream_length: int = 2500
    integration_steps: int = 2
    dt: float = 0.20
    device_str: str = DEVICE_STR


class QuantumSpinWaveNexus(nn.Module):
    """
    Unified Quantum-Thermodynamic Spin Wavefield Core.
    """
    def __init__(self, config: EXP403Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.steps = config.integration_steps
        self.dt = config.dt

        # Complex Hermitian Exchange Matrix J = J_real + i * J_imag (Hermitian: J = J^dagger)
        # J_real symmetric, J_imag antisymmetric, zero diagonals
        j_r = torch.randn(self.dim, self.dim, device=DEVICE) / math.sqrt(self.dim)
        j_real_sym = 0.5 * (j_r + j_r.T)
        j_real_sym.fill_diagonal_(0.0)
        self.j_real = nn.Parameter(j_real_sym)

        j_i = torch.randn(self.dim, self.dim, device=DEVICE) / math.sqrt(self.dim)
        j_imag_anti = 0.5 * (j_i - j_i.T)
        j_imag_anti.fill_diagonal_(0.0)
        self.j_imag = nn.Parameter(j_imag_anti)

        # Complex External Driving Field h(u_t) = h_real + i * h_imag
        self.byte_field_real = nn.Embedding(self.dim, self.dim).to(DEVICE)
        self.byte_field_imag = nn.Embedding(self.dim, self.dim).to(DEVICE)

        # Nonlinear quartic self-interaction parameter lambda
        self.nonlin_lambda = nn.Parameter(torch.tensor(0.1, device=DEVICE))

    def forward_step(
        self, psi_real: torch.Tensor, psi_imag: torch.Tensor, byte_curr: int, curr_temp: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """
        Integrates dissipative complex Ginzburg-Landau dynamics.
        psi = psi_real + i * psi_imag [dim]
        returns: (psi_real_next, psi_imag_next, probs, next_temp, phase_coherence)
        """
        # Ensure exact Hermitian symmetry: J_real sym, J_imag antisym
        j_r = 0.5 * (self.coupling_real() + self.coupling_real().T)
        j_r = j_r - torch.diag(torch.diag(j_r))

        j_i = 0.5 * (self.coupling_imag() - self.coupling_imag().T)
        j_i = j_i - torch.diag(torch.diag(j_i))

        # External field from byte
        byte_tensor = torch.tensor(byte_curr, device=DEVICE, dtype=torch.long)
        h_r = self.byte_field_real(byte_tensor)
        h_i = self.byte_field_imag(byte_tensor)

        p_r = psi_real
        p_i = psi_imag
        eff_gamma = max(curr_temp * 0.2, 0.02)

        for _ in range(self.steps):
            # Matrix multiplication with complex J: (J_r + i J_i) * (p_r + i p_i)
            # Real part: J_r p_r - J_i p_i
            # Imag part: J_r p_i + J_i p_r
            field_r = torch.mv(j_r, p_r) - torch.mv(j_i, p_i) + h_r
            field_i = torch.mv(j_r, p_i) + torch.mv(j_i, p_r) + h_i

            # Nonlinear self-interaction: lambda * |psi|^2 * psi
            norm_sq = p_r ** 2 + p_i ** 2
            v_r = self.nonlin_lambda * norm_sq * p_r
            v_i = self.nonlin_lambda * norm_sq * p_i

            dH_r = field_r - v_r
            dH_i = field_i - v_i

            # Dissipative Ginzburg-Landau step:
            # d(psi)/dt = - i dH/d(psi*) - gamma * dH/d(psi*)
            # where -i (dH_r + i dH_i) = dH_i - i dH_r
            dp_r = (dH_i - eff_gamma * dH_r) * self.dt
            dp_i = (-dH_r - eff_gamma * dH_i) * self.dt

            p_r = p_r + dp_r
            p_i = p_i + dp_i

        # Quantum Normalization to avoid runaway
        psi_norm = torch.sqrt(torch.sum(p_r ** 2 + p_i ** 2) + 1e-8)
        p_r_norm = p_r / psi_norm
        p_i_norm = p_i / psi_norm

        # Born Rule Probability: P(j) = |psi_j|^2 = p_r^2 + p_i^2
        probs = p_r_norm ** 2 + p_i_norm ** 2
        probs = probs / (torch.sum(probs) + 1e-8)

        # Phase coherence sigma(Phi)
        phases = torch.atan2(p_i_norm, p_r_norm + 1e-8)
        phase_coherence = torch.std(phases).item()

        # SOC Temperature Update
        # If norm before normalization was high, heat up to dissipate; if low, cool down
        norm_val = psi_norm.item()
        err = norm_val - self.config.target_norm_crit
        next_temp = curr_temp + 0.05 * err
        next_temp = max(0.05, min(2.0, next_temp))

        return p_r_norm, p_i_norm, probs, next_temp, phase_coherence

    def coupling_real(self):
        return self.j_real

    def coupling_imag(self):
        return self.j_imag


def run_exp_403():
    config = EXP403Config()
    model = QuantumSpinWaveNexus(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    logger.info("================================================================================")
    logger.info("STARTING EXP-403: QUANTUM-THERMODYNAMIC SPIN WAVEFIELD & GAUGE ANNEALING (QTSW-GA)")
    logger.info(f"Complex Hilbert Space C^{config.dim} | Integration Steps: {config.integration_steps} | Device: {config.device_str}")
    logger.info("================================================================================")

    # Synthetic stress corpus
    torch.manual_seed(42)
    motifs = [
        b"QUANTUM_THERMODYNAMIC_SPIN_WAVEFIELD_GAUGE_ANNEALING_KARYON_RUBICON_400_EXP403\n",
        b"GET /api/v2/quantum_spin_wave_annealing HTTP/1.1\r\nHost: karyon.ai\r\n\r\n",
        b"def quantum_ginzburg_landau(psi, J, h, gamma):\n    return -1j*dH - gamma*dH\n",
        b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00>\x00\x01\x00\x00\x00"
    ]
    corpus = []
    while len(corpus) < config.stream_length + 100:
        for m in motifs:
            corpus.extend(list(m))
            corpus.extend(list(torch.randint(0, 256, (6,)).numpy()))
    corpus = corpus[:config.stream_length]

    # Initial wavepacket state: equal superposition with random phases
    initial_phases = torch.rand(config.dim, device=DEVICE) * 2 * math.pi
    psi_r = torch.cos(initial_phases) / math.sqrt(config.dim)
    psi_i = torch.sin(initial_phases) / math.sqrt(config.dim)
    curr_temp = 0.5

    total_loss = 0.0
    correct_count = 0
    recent_losses = []
    phase_coherence_trajectory = []
    temp_trajectory = []

    t_start = time.time()

    for step in range(len(corpus) - 1):
        byte_curr = corpus[step]
        byte_next = corpus[step + 1]

        # Quantum Spin Wave step
        psi_r_next, psi_i_next, probs, curr_temp, phase_coh = model.forward_step(
            psi_r, psi_i, byte_curr, curr_temp
        )

        # Cross-Entropy / Surprisal Loss via Born Rule
        target_tensor = torch.tensor(byte_next, device=DEVICE, dtype=torch.long)
        loss = -torch.log(probs[target_tensor] + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update wavepacket state (detach for stream)
        psi_r = psi_r_next.detach()
        psi_i = psi_i_next.detach()

        # Telemetry
        loss_val = loss.item()
        total_loss += loss_val
        recent_losses.append(loss_val)
        if len(recent_losses) > 50:
            recent_losses.pop(0)

        pred_byte = torch.argmax(probs).item()
        if pred_byte == byte_next:
            correct_count += 1

        phase_coherence_trajectory.append(phase_coh)
        temp_trajectory.append(curr_temp)

        if (step + 1) % 250 == 0:
            avg_recent_loss = sum(recent_losses) / len(recent_losses)
            acc = (correct_count / (step + 1)) * 100.0
            avg_phase_coh = sum(phase_coherence_trajectory[-250:]) / 250.0
            avg_temp = sum(temp_trajectory[-250:]) / 250.0
            logger.info(
                f"Step {step+1:4d}/{config.stream_length} | "
                f"Surprisal: {loss_val:.4f} (Avg50: {avg_recent_loss:.4f}) | "
                f"Accuracy: {acc:.2f}% | "
                f"Phase Coh: {avg_phase_coh:.4f} | Temp T: {avg_temp:.3f}"
            )

    elapsed = time.time() - t_start
    final_avg_loss = total_loss / (config.stream_length - 1)
    final_accuracy = (correct_count / (config.stream_length - 1)) * 100.0
    throughput = config.stream_length / elapsed

    logger.info("================================================================================")
    logger.info("EXP-403 FINAL RESULTS:")
    logger.info(f"Elapsed Time: {elapsed:.2f} s | Throughput: {throughput:.2f} steps/s")
    logger.info(f"Final Average Surprisal (Loss): {final_avg_loss:.4f} nats")
    logger.info(f"Single-Pass Prediction Accuracy: {final_accuracy:.2f}%")
    logger.info(f"Final Phase Coherence: {phase_coherence_trajectory[-1]:.4f} | Final Temp: {temp_trajectory[-1]:.3f}")
    logger.info("================================================================================")

    results = {
        "exp_id": config.exp_id,
        "elapsed_time": elapsed,
        "throughput_steps_per_sec": throughput,
        "final_avg_loss": final_avg_loss,
        "final_accuracy": final_accuracy,
        "final_phase_coherence": phase_coherence_trajectory[-1],
        "final_temperature": temp_trajectory[-1],
        "verdict": "🟢 POSITIVE" if final_accuracy > 12.0 or final_avg_loss < 3.5 else "⚪ NEUTRAL"
    }

    with open("experiments/exp_403_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_403()
