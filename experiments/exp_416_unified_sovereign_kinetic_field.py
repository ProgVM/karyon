"""
EXP-416: Unified Sovereign Kinetic Phase Field & Continuous Dynamic Relaxation (USKF-CDR)
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Date: October 2026
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Architectural Synthesis:
1. Eradicated Parasitic Re-creation of AdamW optimizer inside the byte loop.
2. Eradicated the 6 isolated external MLP supervisors (budget_net, cost_net, data_coupling_net, etc.).
3. Unified Complex Physical Substrate (Complex Quantum Spin Field in C^D, D=258) with strict
   quantum state normalization (preventing numerical explosion / NaN).
4. Out-of-place state propagation to guarantee clean PyTorch Autograd computation.
5. Dynamic Endogenous Relaxation: Solves the dissipative Ginzburg-Landau equations until
   phase flux halts (or max dynamic depth is reached).
6. Exact Born Rule Probability P(j) = |Psi_j|^2 mapped cleanly to cross-entropy loss.
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
    learning_rate: float = 0.005
    stream_length: int = 3000
    kinetic_damping: float = 0.20
    dt: float = 0.05
    energy_halt_threshold: float = 0.0005
    max_relaxation_steps: int = 8
    target_norm_crit: float = 1.0
    device_str: str = DEVICE_STR


class UnifiedKineticPhaseField(nn.Module):
    """
    Unified Sovereign Complex Phase Field operating directly on GPU Tensor Cores.
    """
    def __init__(self, config: EXP416Config):
        super().__init__()
        self.config = config
        self.dim = config.dim

        # Complex Hermitian Interaction Matrix J = J_real + i * J_imag
        j_real_init = torch.randn(self.dim, self.dim, device=DEVICE) / math.sqrt(self.dim)
        j_imag_init = torch.randn(self.dim, self.dim, device=DEVICE) / math.sqrt(self.dim)
        # Enforce Hermitian: J_real is symmetric, J_imag is anti-symmetric
        j_real_init = 0.5 * (j_real_init + j_real_init.T)
        j_imag_init = 0.5 * (j_imag_init - j_imag_init.T)

        self.j_real = nn.Parameter(j_real_init)
        self.j_imag = nn.Parameter(j_imag_init)

        # Input phase projection embedding
        self.w_stim_real = nn.Parameter(torch.randn(self.dim, self.dim, device=DEVICE) / math.sqrt(self.dim))
        self.w_stim_imag = nn.Parameter(torch.randn(self.dim, self.dim, device=DEVICE) / math.sqrt(self.dim))

        self.nonlin_lambda = nn.Parameter(torch.tensor(0.05, device=DEVICE))
        self.kinetic_damping = nn.Parameter(torch.tensor(config.kinetic_damping, device=DEVICE))

        # Persistent continuous state buffer
        self.psi_real = torch.zeros(self.dim, device=DEVICE)
        self.psi_imag = torch.zeros(self.dim, device=DEVICE)
        self.curr_temp = 0.5

    def reset_state(self):
        self.psi_real = torch.zeros(self.dim, device=DEVICE)
        self.psi_imag = torch.zeros(self.dim, device=DEVICE)
        self.curr_temp = 0.5

    def forward_stream_byte(self, x_byte: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        # Detach previous state for online single-pass stream processing (out-of-place)
        p_r = self.psi_real.detach()
        p_i = self.psi_imag.detach()

        # Input stimulus as one-hot projected through stimulus tensors
        x_onehot = F.one_hot(torch.tensor(x_byte, device=DEVICE), num_classes=self.dim).float()
        h_r = torch.mv(self.w_stim_real, x_onehot)
        h_i = torch.mv(self.w_stim_imag, x_onehot)

        eff_gamma = torch.clamp(F.softplus(self.kinetic_damping) * self.curr_temp, min=0.02, max=2.0)
        dt = self.config.dt

        steps_done = 0
        kinetic_flux = 1.0

        # Continuous Dynamic Relaxation Loop
        while steps_done < self.config.max_relaxation_steps and kinetic_flux > self.config.energy_halt_threshold:
            # Complex Matrix Multiplication: (J_r + i J_i) * (p_r + i p_i)
            field_r = torch.mv(self.j_real, p_r) - torch.mv(self.j_imag, p_i) + h_r
            field_i = torch.mv(self.j_real, p_i) + torch.mv(self.j_imag, p_r) + h_i

            # Quartic Ginzburg-Landau self-interaction
            norm_sq = p_r ** 2 + p_i ** 2
            v_r = self.nonlin_lambda * norm_sq * p_r
            v_i = self.nonlin_lambda * norm_sq * p_i

            dH_r = field_r - v_r
            dH_i = field_i - v_i

            # Dissipative Ginzburg-Landau SDE:
            # d(psi)/dt = (dH_i - gamma * dH_r) * dt
            dp_r = (dH_i - eff_gamma * dH_r) * dt
            dp_i = (-dH_r - eff_gamma * dH_i) * dt

            p_r = p_r + dp_r
            p_i = p_i + dp_i

            kinetic_flux = torch.mean(dp_r ** 2 + dp_i ** 2).item()
            steps_done += 1

        # Quantum Wavefunction Normalization (prevents runaway & NaN)
        psi_norm = torch.sqrt(torch.sum(p_r ** 2 + p_i ** 2) + 1e-7)
        p_r_norm = p_r / psi_norm
        p_i_norm = p_i / psi_norm

        # Save normalized state out-of-place for the next temporal byte
        self.psi_real = p_r_norm.detach()
        self.psi_imag = p_i_norm.detach()

        # Born Rule Probability: P(j) = |Psi_j|^2 = p_r^2 + p_i^2
        probs = p_r_norm ** 2 + p_i_norm ** 2
        probs = probs / (torch.sum(probs) + 1e-7)
        logits = torch.log(probs + 1e-7).unsqueeze(0)

        target = torch.tensor([target_byte], device=DEVICE)
        loss = F.cross_entropy(logits, target)

        pred = torch.argmax(logits, dim=-1).item()

        # Dynamic SOC temperature update
        with torch.no_grad():
            norm_val = psi_norm.item()
            err = norm_val - self.config.target_norm_crit
            self.curr_temp = max(0.05, min(2.0, self.curr_temp + 0.05 * err))

        meta = {
            "loss": loss.item(),
            "relaxation_steps": steps_done,
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

    # PERSISTENT UNIFIED OPTIMIZER (Single instance across the entire stream!)
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
