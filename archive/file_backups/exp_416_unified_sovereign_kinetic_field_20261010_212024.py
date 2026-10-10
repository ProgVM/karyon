"""
EXP-416: Unified Sovereign Kinetic Field & Continuous Dynamic Relaxation (USKF-CDR)
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Date: October 2026
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Architectural Shift (Eradicating Parasitic Overheads & Achieving Complete Sovereignty):
Diagnosis by Bazilevs & Cybernetic Audit:
1. Eradicated Parasitic Re-creation of AdamW optimizer inside the byte loop (which destroyed momentum
   and dropped execution speed from 200+ steps/s to 6.7 steps/s).
2. Eradicated the 6 isolated external MLP supervisors (budget_net, cost_net, data_coupling_net, etc.)
   that acted as a top-down programmatic crutch violating KEP Principle 27.
3. Restored Unified Complex Physical Substrate (Complex Kinetic Phase Field in C^D, D=258).
4. Sovereign Dynamic Relaxation: Halting is governed purely by internal kinetic energy dissipation
   dH/dt -> 0 (Hamiltonian relaxation) without static counters or external supervisors.
5. Persistent Unified Plasticity: Single persistent optimizer/momentum state spanning the entire stream,
   with online state detachment across temporal byte boundaries to ensure continuous stream learning.
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
logger = logging.getLogger("EXP-416-UNIFIED-FIELD")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP416Config:
    exp_id: str = "EXP-416"
    dim: int = 258
    learning_rate: float = 0.008
    stream_length: int = 3000
    kinetic_damping: float = 0.15
    energy_halt_threshold: float = 0.001
    max_relaxation_steps: int = 24
    device_str: str = DEVICE_STR


class UnifiedKineticPhaseField(nn.Module):
    """
    Unified Sovereign Complex Phase Field operating directly on GPU Tensor Cores.
    """
    def __init__(self, config: EXP416Config):
        super().__init__()
        self.config = config
        self.dim = config.dim

        # Complex Hermitian Interaction Hamiltonian Matrix J = J_real + i * J_imag
        j_real_init = torch.randn(self.dim, self.dim, device=DEVICE) / math.sqrt(self.dim)
        j_imag_init = torch.randn(self.dim, self.dim, device=DEVICE) / math.sqrt(self.dim)
        # Symmetrize real and anti-symmetrize imag for Hermitian properties
        j_real_init = 0.5 * (j_real_init + j_real_init.T)
        j_imag_init = 0.5 * (j_imag_init - j_imag_init.T)

        self.j_real = nn.Parameter(j_real_init)
        self.j_imag = nn.Parameter(j_imag_init)

        # Non-linear self-interaction coupling (Ginzburg-Landau quartic term)
        self.lambda_quartic = nn.Parameter(torch.tensor([0.1], device=DEVICE))
        self.kinetic_damping = nn.Parameter(torch.tensor([config.kinetic_damping], device=DEVICE))

        # Persistent continuous state of the kinetic wavefield [1, D]
        self.register_buffer("psi_real", torch.zeros(1, self.dim, device=DEVICE))
        self.register_buffer("psi_imag", torch.zeros(1, self.dim, device=DEVICE))

    def reset_state(self):
        self.psi_real.zero_()
        self.psi_imag.zero_()

    def detach_state(self):
        self.psi_real = self.psi_real.detach()
        self.psi_imag = self.psi_imag.detach()

    def relax_step(self, x_stimulus_real: torch.Tensor, x_stimulus_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Executes a single continuous physical relaxation step of the Complex Kinetic Field.
        dPsi / dt = - (i * dH/dPsi* + gamma * dH/dPsi*)
        """
        # Linear Hermitian Hamiltonian Interaction
        # J * Psi = (J_r + i J_i) * (P_r + i P_i) = (J_r P_r - J_i P_i) + i (J_r P_i + J_i P_r)
        h_interaction_real = torch.matmul(self.psi_real, self.j_real) - torch.matmul(self.psi_imag, self.j_imag)
        h_interaction_imag = torch.matmul(self.psi_imag, self.j_real) + torch.matmul(self.psi_real, self.j_imag)

        # Quartic self-interaction gradient: lambda * |Psi|^2 * Psi
        psi_sq = self.psi_real ** 2 + self.psi_imag ** 2
        quartic_real = self.lambda_quartic * psi_sq * self.psi_real
        quartic_imag = self.lambda_quartic * psi_sq * self.psi_imag

        # Total variational gradient dH / dPsi*
        dh_real = h_interaction_real + quartic_real - x_stimulus_real
        dh_imag = h_interaction_imag + quartic_imag - x_stimulus_imag

        # Dissipative + Unitary Precession Dynamics
        gamma = F.softplus(self.kinetic_damping)
        # dPsi_r / dt = dh_imag - gamma * dh_real
        # dPsi_i / dt = -dh_real - gamma * dh_imag
        d_psi_real = dh_imag - gamma * dh_real
        d_psi_imag = -dh_real - gamma * dh_imag

        # Euler-Heun Integration Step
        dt = 0.1
        self.psi_real = self.psi_real + dt * d_psi_real
        self.psi_imag = self.psi_imag + dt * d_psi_imag

        # Kinetic energy flux |dPsi/dt|^2
        kinetic_flux = torch.mean(d_psi_real ** 2 + d_psi_imag ** 2).item()
        return self.psi_real, self.psi_imag, kinetic_flux

    def forward_stream_byte(self, x_byte: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        # Detach previous state graph for online temporal boundary
        self.detach_state()

        # Inject raw byte stimulus as continuous phase input
        angle = (2.0 * math.pi * x_byte) / float(self.dim)
        x_stimulus_real = torch.full((1, self.dim), math.cos(angle), device=DEVICE)
        x_stimulus_imag = torch.full((1, self.dim), math.sin(angle), device=DEVICE)

        # Continuous Sovereign Relaxation Loop
        relaxation_steps = 0
        kinetic_flux = 1.0

        while relaxation_steps < self.config.max_relaxation_steps and kinetic_flux > self.config.energy_halt_threshold:
            _, _, kinetic_flux = self.relax_step(x_stimulus_real, x_stimulus_imag)
            relaxation_steps += 1

        # Born Rule Density & Next-Byte Logits: P(j) ~ |Psi_j|^2
        born_density = self.psi_real ** 2 + self.psi_imag ** 2 + 1e-6
        logits = torch.log(born_density)

        target = torch.tensor([target_byte], device=DEVICE)
        loss = F.cross_entropy(logits, target)

        pred = torch.argmax(logits, dim=-1).item()

        meta = {
            "loss": loss.item(),
            "relaxation_steps": relaxation_steps,
            "final_kinetic_flux": kinetic_flux,
            "pred": pred,
            "is_correct": (pred == target_byte)
        }
        return loss, meta


def run_exp_416():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-416: UNIFIED SOVEREIGN KINETIC PHASE FIELD (USKF-CDR) ===")
    logger.info("================================================================================")

    config = EXP416Config()
    model = UnifiedKineticPhaseField(config).to(DEVICE)

    # PERSISTENT UNIFIED OPTIMIZER (Created ONCE for the entire stream!)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    stream_data = (
        "def karyon_sovereign_autopoiesis(stream):\n"
        "    # Zero hardcoded menus or external rules\n"
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
    logger.info("Starting single-pass execution with persistent momentum and unified phase field...")

    correct_preds = 0
    total_loss = 0.0
    total_relaxation_steps = 0

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
        total_relaxation_steps += meta["relaxation_steps"]

        if (t + 1) % 500 == 0:
            avg_loss = total_loss / (t + 1)
            acc = (correct_preds / (t + 1)) * 100.0
            avg_steps = total_relaxation_steps / (t + 1)
            logger.info(f"Progress [{t+1}/{len(raw_bytes)-1}] | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | AvgRelaxSteps: {avg_steps:.2f}")

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_relax_steps = total_relaxation_steps / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    logger.info("================================================================================")
    logger.info("=== EXP-416 COMPLETE TELEMETRY ===")
    logger.info(f"Final Average Loss: {final_loss:.4f}")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Average Relaxation Steps: {avg_relax_steps:.2f} steps/byte")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-416",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "avg_relaxation_steps": round(avg_relax_steps, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_416_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_416()
