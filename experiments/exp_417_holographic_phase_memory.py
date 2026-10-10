"""
EXP-417: Holographic Phase Associative Memory & Sovereign Kinetic Field (HPAM-SKF)
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Date: October 2026
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Theoretical & Architectural Foundations:
1. Building on the breakthrough of EXP-416 (Unified Sovereign Kinetic Field, loss 3.8385, speed 96.1 steps/s).
2. Eradicating the chronic memory amnesia bottleneck:
   - Replaces leaky vector decay h_t with a Holographic Phase Associative Memory Matrix M in C^{D x D}.
   - Outer-product phase interference: Delta M = eta(F_t) * (Psi * Psi^dagger).
   - Surprise-gated plastic inscription: Surprise F_t modulates synaptic consolidation dynamically.
   - Associative memory readout: Memory acts as an internal resonant field inside Ginzburg-Landau relaxation:
     field = (J + alpha * M) * Psi.
3. Clean GPU Tensor-Core execution: No external MLP supervisors, no dictionary lookups,
   100% vectorized complex matrix operations with persistent momentum.
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
logger = logging.getLogger("EXP-417-HPAM")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP417Config:
    exp_id: str = "EXP-417"
    dim: int = 258
    learning_rate: float = 0.005
    stream_length: int = 3000
    kinetic_damping: float = 0.20
    dt: float = 0.05
    energy_halt_threshold: float = 0.0005
    max_relaxation_steps: int = 8
    target_norm_crit: float = 1.0
    memory_decay: float = 0.995
    memory_coupling_init: float = 0.25
    device_str: str = DEVICE_STR


class HolographicAssociativeKineticField(nn.Module):
    """
    Sovereign Complex Phase Field with Holographic Phase Associative Memory Matrix (HPAM).
    Operates natively on GPU Tensor Cores in C^{D x D}.
    """
    def __init__(self, config: EXP417Config):
        super().__init__()
        self.config = config
        self.dim = config.dim

        # 1. Base Static Interaction Hamiltonian J = J_real + i * J_imag
        j_real_init = torch.randn(self.dim, self.dim, device=DEVICE) / math.sqrt(self.dim)
        j_imag_init = torch.randn(self.dim, self.dim, device=DEVICE) / math.sqrt(self.dim)
        j_real_init = 0.5 * (j_real_init + j_real_init.T)
        j_imag_init = 0.5 * (j_imag_init - j_imag_init.T)

        self.j_real = nn.Parameter(j_real_init)
        self.j_imag = nn.Parameter(j_imag_init)

        # 2. Input Stimulus Projection
        self.w_stim_real = nn.Parameter(torch.randn(self.dim, self.dim, device=DEVICE) / math.sqrt(self.dim))
        self.w_stim_imag = nn.Parameter(torch.randn(self.dim, self.dim, device=DEVICE) / math.sqrt(self.dim))

        # 3. Dynamic Field Parameters
        self.nonlin_lambda = nn.Parameter(torch.tensor(0.05, device=DEVICE))
        self.kinetic_damping = nn.Parameter(torch.tensor(config.kinetic_damping, device=DEVICE))
        self.memory_coupling = nn.Parameter(torch.tensor(config.memory_coupling_init, device=DEVICE))

        # 4. Continuous State Buffers
        self.psi_real = torch.zeros(self.dim, device=DEVICE)
        self.psi_imag = torch.zeros(self.dim, device=DEVICE)
        self.curr_temp = 0.5

        # 5. Holographic Associative Memory Matrix M in C^{D x D} (Persistent Hebbian Phase Traces)
        self.register_buffer("m_real", torch.zeros(self.dim, self.dim, device=DEVICE))
        self.register_buffer("m_imag", torch.zeros(self.dim, self.dim, device=DEVICE))

    def reset_state(self):
        self.psi_real = torch.zeros(self.dim, device=DEVICE)
        self.psi_imag = torch.zeros(self.dim, device=DEVICE)
        self.curr_temp = 0.5
        self.m_real.zero_()
        self.m_imag.zero_()

    def forward_stream_byte(self, x_byte: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        # Detach previous state out-of-place for online single-pass stream processing
        p_r = self.psi_real.detach()
        p_i = self.psi_imag.detach()

        # Stimulus vector projection
        x_onehot = F.one_hot(torch.tensor(x_byte, device=DEVICE), num_classes=self.dim).float()
        h_r = torch.mv(self.w_stim_real, x_onehot)
        h_i = torch.mv(self.w_stim_imag, x_onehot)

        eff_gamma = torch.clamp(F.softplus(self.kinetic_damping) * self.curr_temp, min=0.02, max=2.0)
        dt = self.config.dt

        # Total Effective Hamiltonian = J + alpha * M_associative
        # Alpha is bounded positive coupling strength
        alpha_mem = F.softplus(self.memory_coupling)
        eff_j_real = self.j_real + alpha_mem * self.m_real
        eff_j_imag = self.j_imag + alpha_mem * self.m_imag

        steps_done = 0
        kinetic_flux = 1.0

        # Continuous Dynamic Relaxation Loop
        while steps_done < self.config.max_relaxation_steps and kinetic_flux > self.config.energy_halt_threshold:
            # Complex Matrix Multiplication: (Eff_J_r + i Eff_J_i) * (p_r + i p_i)
            field_r = torch.mv(eff_j_real, p_r) - torch.mv(eff_j_imag, p_i) + h_r
            field_i = torch.mv(eff_j_real, p_i) + torch.mv(eff_j_imag, p_r) + h_i

            # Quartic Ginzburg-Landau self-interaction
            norm_sq = p_r ** 2 + p_i ** 2
            v_r = self.nonlin_lambda * norm_sq * p_r
            v_i = self.nonlin_lambda * norm_sq * p_i

            dH_r = field_r - v_r
            dH_i = field_i - v_i

            # Dissipative Ginzburg-Landau SDE integration
            dp_r = (dH_i - eff_gamma * dH_r) * dt
            dp_i = (-dH_r - eff_gamma * dH_i) * dt

            p_r = p_r + dp_r
            p_i = p_i + dp_i

            kinetic_flux = torch.mean(dp_r ** 2 + dp_i ** 2).item()
            steps_done += 1

        # Wavefunction Normalization
        psi_norm = torch.sqrt(torch.sum(p_r ** 2 + p_i ** 2) + 1e-7)
        p_r_norm = p_r / psi_norm
        p_i_norm = p_i / psi_norm

        self.psi_real = p_r_norm.detach()
        self.psi_imag = p_i_norm.detach()

        # Born Rule Probability: P(j) = |Psi_j|^2
        probs = p_r_norm ** 2 + p_i_norm ** 2
        probs = probs / (torch.sum(probs) + 1e-7)
        logits = torch.log(probs + 1e-7).unsqueeze(0)

        target = torch.tensor([target_byte], device=DEVICE)
        loss = F.cross_entropy(logits, target)

        pred = torch.argmax(logits, dim=-1).item()
        surprise_val = loss.item()

        # Holographic Memory Inscription (Outer-Product Phase Interference gated by Surprise)
        # Delta M = Psi * Psi^dagger:
        # Re(Delta M) = p_r p_r^T + p_i p_i^T
        # Im(Delta M) = p_i p_r^T - p_r p_i^T
        with torch.no_grad():
            plastic_gain = min(0.1, 0.01 * surprise_val)
            delta_m_r = torch.outer(p_r_norm, p_r_norm) + torch.outer(p_i_norm, p_i_norm)
            delta_m_i = torch.outer(p_i_norm, p_r_norm) - torch.outer(p_r_norm, p_i_norm)

            # Continuous decay + plastic inscription
            self.m_real = self.config.memory_decay * self.m_real + plastic_gain * delta_m_r
            self.m_imag = self.config.memory_decay * self.m_imag + plastic_gain * delta_m_i

            # Dynamic Temperature update
            norm_val = psi_norm.item()
            err = norm_val - self.config.target_norm_crit
            self.curr_temp = max(0.05, min(2.0, self.curr_temp + 0.05 * err))

        meta = {
            "loss": loss.item(),
            "relaxation_steps": steps_done,
            "final_kinetic_flux": kinetic_flux,
            "pred": pred,
            "is_correct": (pred == target_byte),
            "mem_norm": torch.norm(self.m_real).item()
        }
        return loss, meta


def run_exp_417():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-417: HOLOGRAPHIC PHASE ASSOCIATIVE MEMORY (HPAM-SKF) ===")
    logger.info("================================================================================")

    config = EXP417Config()
    model = HolographicAssociativeKineticField(config).to(DEVICE)

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
    logger.info("Starting single-pass execution with Holographic Phase Associative Memory...")

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
            mem_norm = meta["mem_norm"]
            logger.info(
                f"Progress [{t+1}/{len(raw_bytes)-1}] | Loss: {avg_loss:.4f} | "
                f"Acc: {acc:.2f}% | AvgRelaxSteps: {avg_steps:.2f} | MemNorm: {mem_norm:.2f}"
            )

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_relax_steps = total_relaxation_steps / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    logger.info("================================================================================")
    logger.info("=== EXP-417 COMPLETE TELEMETRY ===")
    logger.info(f"Final Average Loss: {final_loss:.4f}")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Average Relaxation Steps: {avg_relax_steps:.2f} steps/byte")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-417",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "avg_relaxation_steps": round(avg_relax_steps, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_417_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_417()
