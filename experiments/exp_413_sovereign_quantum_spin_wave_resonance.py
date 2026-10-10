"""
EXP-413: Sovereign Quantum-Spin Complex Wave Resonance (QSCW-Res)
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Date: October 2026
Standard: KEP v16.0 Sovereign Master (Principle 1, Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Theoretical Foundation:
Synthesis of EXP-403 (Record loss 2.9913 with Heisenberg Spin Wave Coherence) and
EXP-408 (4.6699 Sovereign Recurrent Thinking Loops & Prediction Error Pressure):

1. Complex Hilbert Space Representation (Psi in C^D):
   Every state is represented by dual canonical conjugates (real and imaginary components):
      Psi_t = Re(Psi_t) + i * Im(Psi_t) = A_t * exp(i * Theta_t)
   This allows constructive interference of coherent symbolic patterns while destructive
   interference cancels high-frequency byte noise.
2. Heisenberg Quantum Spin Exchange Dynamics:
   Operators evolve via unitary complex transformations:
      W = W_real + i * W_imag
   Spin exchange between input x and hidden state h:
      H_spin = J * sum_k (S_x^k . S_h^k)
      Phase rotation: Psi_{t+1} = Psi_t * exp(-i * H_spin * dt)
3. Sovereign Quantum Recurrent Thinking Loops (k = 1 ... K_max):
   Phase coherence relaxes iteratively during thinking steps, guided by complex sensory error pressure:
      epsilon_complex = (x_target - pred_probs) + i * phase_discrepancy
4. Zero-Hardcode Epigenetic Locks (mu_i):
   Consolidated wave operators are locked by epigenetic methylation against chaotic phase decoherence.
"""

import math
import time
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-413-QSCW-RES")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP413Config:
    exp_id: str = "EXP-413"
    dim: int = 258
    rank: int = 32
    max_synthesized_nodes: int = 16
    max_thinking_steps: int = 5
    learning_rate: float = 0.0035
    stream_length: int = 3000
    spin_exchange_j: float = 0.15
    device_str: str = DEVICE_STR


class ComplexQuantumOperator:
    """
    Complex-valued Quantum Spin Operator acting on C^D.
    """
    def __init__(self, dim: int, rank: int, device: torch.device):
        self.dim = dim
        self.rank = rank
        self.device = device

        # Real and imaginary components of unitary multilinear kernels
        self.w1_real = (torch.randn(dim, rank, device=device) / math.sqrt(dim)).requires_grad_(True)
        self.w1_imag = (torch.randn(dim, rank, device=device) / math.sqrt(dim)).requires_grad_(True)
        self.w2_real = (torch.randn(dim, rank, device=device) / math.sqrt(dim)).requires_grad_(True)
        self.w2_imag = (torch.randn(dim, rank, device=device) / math.sqrt(dim)).requires_grad_(True)
        self.w3_real = (torch.randn(dim, dim, device=device) / math.sqrt(dim)).requires_grad_(True)
        self.w3_imag = (torch.randn(dim, dim, device=device) / math.sqrt(dim)).requires_grad_(True)

        self.alpha_epi = torch.zeros(1, device=device).requires_grad_(True)
        self.methylation_lock = 0.0
        self.vitality = 1.0
        self.age = 0

    def get_params(self) -> List[torch.Tensor]:
        return [self.w1_real, self.w1_imag, self.w2_real, self.w2_imag, self.w3_real, self.w3_imag, self.alpha_epi]

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor,
                h_real: torch.Tensor, h_imag: torch.Tensor,
                err_real: torch.Tensor, err_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complex multiplication and quantum interference:
        (a + ib)(c + id) = (ac - bd) + i(ad + bc)
        """
        # Complex input with error pressure
        xr = x_real + err_real
        xi = x_imag + err_imag

        # u1 = x @ W1
        u1_r = torch.matmul(xr, self.w1_real) - torch.matmul(xi, self.w1_imag)
        u1_i = torch.matmul(xr, self.w1_imag) + torch.matmul(xi, self.w1_real)

        # u2 = h @ W2
        u2_r = torch.matmul(h_real, self.w2_real) - torch.matmul(h_imag, self.w2_imag)
        u2_i = torch.matmul(h_real, self.w2_imag) + torch.matmul(h_imag, self.w2_real)

        # Bilinear core = u1 * u2
        core_r = u1_r * u2_r - u1_i * u2_i
        core_i = u1_r * u2_i + u1_i * u2_r

        # Drift = core @ W1.t + x @ W3
        drift_r = (torch.matmul(core_r, self.w1_real.t()) - torch.matmul(core_i, self.w1_imag.t()) +
                   torch.matmul(xr, self.w3_real) - torch.matmul(xi, self.w3_imag))
        drift_i = (torch.matmul(core_r, self.w1_imag.t()) + torch.matmul(core_i, self.w1_real.t()) +
                   torch.matmul(xr, self.w3_imag) + torch.matmul(xi, self.w3_real))

        gate = torch.tanh(self.alpha_epi)
        return gate * drift_r, gate * drift_i


class SovereignQuantumSuperOperator(nn.Module):
    """
    Super-Operator Phi governing quantum phase dynamics and spin exchange.
    """
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.dim = dim
        self.rank = rank

        # Complex coupling net (operates on concatenated real & imag channels)
        self.data_coupling_net = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.Tanh(),
            nn.Linear(dim, rank)
        ).to(DEVICE)

        self.halting_net = nn.Sequential(
            nn.Linear(dim * 4, rank),
            nn.Tanh(),
            nn.Linear(rank, 1)
        ).to(DEVICE)

        self.genesis_potential = nn.Sequential(
            nn.Linear(dim * 4 + 1, rank),
            nn.Tanh(),
            nn.Linear(rank, 5)
        ).to(DEVICE)

        self.lock_evaluator = nn.Sequential(
            nn.Linear(rank + dim * 2, rank),
            nn.Tanh(),
            nn.Linear(rank, 1)
        ).to(DEVICE)

        self.hyper_synth = nn.Sequential(
            nn.Linear(dim * 2 + rank, rank * 4),
            nn.Tanh(),
            nn.Linear(rank * 4, (dim * rank * 2 + dim * dim) * 2)
        ).to(DEVICE)

    def compute_data_coupling(self, x_r: torch.Tensor, x_i: torch.Tensor,
                              h_r: torch.Tensor, h_i: torch.Tensor, num_nodes: int) -> torch.Tensor:
        combined = torch.cat([x_r, x_i, h_r, h_i], dim=-1)
        phi_vec = self.data_coupling_net(combined)
        phase_freq = phi_vec[:, :num_nodes] if num_nodes <= self.rank else F.pad(phi_vec, (0, num_nodes - self.rank))
        return F.softmax(phase_freq, dim=-1)

    def compute_halting_probability(self, h_curr_r: torch.Tensor, h_curr_i: torch.Tensor,
                                    h_prev_r: torch.Tensor, h_prev_i: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_curr_r, h_curr_i, h_prev_r, h_prev_i], dim=-1)
        return torch.sigmoid(self.halting_net(combined))

    def compute_endogenous_potential(self, x_r: torch.Tensor, x_i: torch.Tensor,
                                     h_r: torch.Tensor, h_i: torch.Tensor,
                                     free_energy: torch.Tensor) -> Dict[str, torch.Tensor]:
        inp = torch.cat([x_r, x_i, h_r, h_i, free_energy.view(1, 1)], dim=-1)
        fluxes = self.genesis_potential(inp)

        sprout_flux = torch.sigmoid(fluxes[:, 0])
        apoptosis_flux = torch.sigmoid(fluxes[:, 1])
        time_dilation = torch.exp(fluxes[:, 2]).clamp(0.1, 3.0)
        phase_curvature = torch.tanh(fluxes[:, 3])
        memory_freeze_flux = torch.sigmoid(fluxes[:, 4])

        return {
            "sprout_flux": sprout_flux,
            "apoptosis_flux": apoptosis_flux,
            "time_dilation": time_dilation,
            "phase_curvature": phase_curvature,
            "memory_freeze_flux": memory_freeze_flux
        }

    def compute_operator_lock(self, h_r: torch.Tensor, h_i: torch.Tensor, op: ComplexQuantumOperator) -> float:
        op_sig = torch.mean(op.w1_real + op.w1_imag, dim=0, keepdim=True)
        combined = torch.cat([op_sig, h_r, h_i], dim=-1)
        return torch.sigmoid(self.lock_evaluator(combined)).item()

    def synthesize_operator_tensors(self, seed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                        torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self.hyper_synth(seed)
        s1 = self.dim * self.rank
        s2 = self.dim * self.rank
        s3 = self.dim * self.dim
        half = (s1 + s2 + s3)

        r_part = flat[:, :half]
        i_part = flat[:, half:]

        w1_r = r_part[:, :s1].view(self.dim, self.rank) / math.sqrt(self.dim)
        w2_r = r_part[:, s1:s1+s2].view(self.dim, self.rank) / math.sqrt(self.dim)
        w3_r = r_part[:, s1+s2:].view(self.dim, self.dim) / math.sqrt(self.dim)

        w1_i = i_part[:, :s1].view(self.dim, self.rank) / math.sqrt(self.dim)
        w2_i = i_part[:, s1:s1+s2].view(self.dim, self.rank) / math.sqrt(self.dim)
        w3_i = i_part[:, s1+s2:].view(self.dim, self.dim) / math.sqrt(self.dim)

        return w1_r, w1_i, w2_r, w2_i, w3_r, w3_i


class QuantumSovereignMorphicEngine(nn.Module):
    """
    Sovereign Quantum-Spin Wave Engine in Complex Hilbert Space C^D.
    """
    def __init__(self, config: EXP413Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.rank = config.rank
        self.max_nodes = config.max_synthesized_nodes
        self.max_thinking_steps = config.max_thinking_steps
        self.j_spin = config.spin_exchange_j

        self.super_op = SovereignQuantumSuperOperator(self.dim, self.rank)
        # Complex readout: output probability from probability amplitude |Psi|^2
        self.readout_real = nn.Linear(self.dim, self.dim, bias=False).to(DEVICE)
        self.readout_imag = nn.Linear(self.dim, self.dim, bias=False).to(DEVICE)

        # Primordial crystalline wave operator
        self.operators: List[ComplexQuantumOperator] = [
            ComplexQuantumOperator(self.dim, self.rank, DEVICE)
        ]
        self.operators[0].alpha_epi.data.fill_(1.2)
        self.operators[0].methylation_lock = 1.0

        # Complex state Psi = h_real + i * h_imag
        self.h_real = torch.zeros(1, self.dim, device=DEVICE)
        self.h_imag = torch.zeros(1, self.dim, device=DEVICE)

        self.err_real = torch.zeros(1, self.dim, device=DEVICE)
        self.err_imag = torch.zeros(1, self.dim, device=DEVICE)
        self.genesis_accumulator = torch.zeros(1, device=DEVICE)

    def heisenberg_spin_exchange(self, h_r: torch.Tensor, h_i: torch.Tensor,
                                 x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unitary Heisenberg quantum spin exchange evolution:
        U = exp(-i * H_spin * dt)
        """
        # Spin dot product S_x . S_h
        spin_dot = x_r * h_r + x_i * h_i
        theta = self.j_spin * spin_dot  # Phase rotation angle

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Complex rotation: h * exp(-i * theta) = (h_r*cos + h_i*sin) + i*(h_i*cos - h_r*sin)
        rot_r = h_r * cos_t + h_i * sin_t
        rot_i = h_i * cos_t - h_r * sin_t
        return rot_r, rot_i

    def forward_step(self, x_byte: int, target_byte: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Encode byte onto quantum phase wheel: Psi_x = cos(phi) + i * sin(phi)
        phi_byte = 2.0 * math.pi * (x_byte / float(self.dim))
        x_base = F.one_hot(torch.tensor([x_byte], device=DEVICE), num_classes=self.dim).float()
        x_real = x_base * math.cos(phi_byte)
        x_imag = x_base * math.sin(phi_byte)

        num_ops = len(self.operators)

        # 1. Heisenberg Spin Exchange Pre-coupling
        h_spin_r, h_spin_i = self.heisenberg_spin_exchange(self.h_real, self.h_imag, x_real, x_imag)

        # 2. Covariant Coupling
        coupling_weights = self.super_op.compute_data_coupling(x_real, x_imag, h_spin_r, h_spin_i, num_ops)

        # 3. Sovereign Quantum Recurrent Thinking Loop
        hr_rec = h_spin_r
        hi_rec = h_spin_i
        hr_prev = hr_rec
        hi_prev = hi_rec
        thinking_cycles = 0

        for k in range(self.max_thinking_steps):
            thinking_cycles += 1
            op_out_r = []
            op_out_i = []

            for i, op in enumerate(self.operators):
                w_in = coupling_weights[:, i:i+1]
                xr_in = w_in * x_real + (1.0 - w_in) * hr_rec
                xi_in = w_in * x_imag + (1.0 - w_in) * hi_rec

                yr, yi = op.forward(xr_in, xi_in, hr_rec, hi_rec, self.err_real, self.err_imag)
                op_out_r.append(yr)
                op_out_i.append(yi)

            drift_r = torch.sum(torch.stack(op_out_r, dim=0), dim=0)
            drift_i = torch.sum(torch.stack(op_out_i, dim=0), dim=0)

            # Continuous non-linear wave integration
            hr_next = torch.tanh(hr_rec + 0.5 * drift_r)
            hi_next = torch.tanh(hi_rec + 0.5 * drift_i)

            halt_prob = self.super_op.compute_halting_probability(hr_next, hi_next, hr_prev, hi_prev)
            hr_prev = hr_rec
            hi_prev = hi_rec
            hr_rec = hr_next
            hi_rec = hi_next

            if halt_prob.item() > 0.65 and k >= 1:
                break

        # 4. Quantum Readout: Born rule probability amplitude squared
        logits_r = self.readout_real(hr_rec)
        logits_i = self.readout_imag(hi_rec)
        prob_amplitude_sq = logits_r.pow(2) + logits_i.pow(2)  # |Psi|^2
        logits = torch.log(prob_amplitude_sq + 1e-6)

        loss = F.cross_entropy(logits, torch.tensor([target_byte], device=DEVICE))

        # 5. Complex Error Pressure Update
        with torch.no_grad():
            target_one_hot = F.one_hot(torch.tensor([target_byte], device=DEVICE), num_classes=self.dim).float()
            pred_probs = F.softmax(logits, dim=-1)
            err_amplitude = target_one_hot - pred_probs
            self.err_real = 0.8 * self.err_real + 0.2 * (err_amplitude * math.cos(phi_byte))
            self.err_imag = 0.8 * self.err_imag + 0.2 * (err_amplitude * math.sin(phi_byte))

        # 6. Endogenous Potential
        phenom = self.super_op.compute_endogenous_potential(x_real, x_imag, hr_rec, hi_rec, loss.detach())
        tau = phenom["time_dilation"]
        sprout_flux = phenom["sprout_flux"]
        apoptosis_flux = phenom["apoptosis_flux"]
        freeze_flux = phenom["memory_freeze_flux"]

        self.h_real = torch.tanh(hr_rec * tau).detach()
        self.h_imag = torch.tanh(hi_rec * tau).detach()

        # Update methylation locks
        locked_nodes_count = 0
        for i, op in enumerate(self.operators):
            op.age += 1
            if i > 0:
                target_lock = self.super_op.compute_operator_lock(self.h_real, self.h_imag, op) * freeze_flux.item()
                op.methylation_lock = 0.95 * op.methylation_lock + 0.05 * target_lock
            if op.methylation_lock > 0.60:
                locked_nodes_count += 1

        # Spontaneous Genesis
        self.genesis_accumulator += sprout_flux.squeeze() - 0.38
        sprouted = False
        pruned = False

        if self.genesis_accumulator.item() > 1.0 and len(self.operators) < self.max_nodes:
            seed = torch.cat([self.h_real, self.h_imag, sprout_flux.repeat(1, self.rank)], dim=-1)
            w1_r, w1_i, w2_r, w2_i, w3_r, w3_i = self.super_op.synthesize_operator_tensors(seed)

            new_op = ComplexQuantumOperator(self.dim, self.rank, DEVICE)
            new_op.w1_real.data.copy_(w1_r)
            new_op.w1_imag.data.copy_(w1_i)
            new_op.w2_real.data.copy_(w2_r)
            new_op.w2_imag.data.copy_(w2_i)
            new_op.w3_real.data.copy_(w3_r)
            new_op.w3_imag.data.copy_(w3_i)
            new_op.alpha_epi.data.fill_(0.01)

            self.operators.append(new_op)
            self.genesis_accumulator.data.fill_(0.0)
            sprouted = True

        # Endogenous Apoptosis
        for op in self.operators[1:]:
            op_impact = torch.abs(torch.tanh(op.alpha_epi)).item()
            effective_decay = apoptosis_flux.item() * 0.02 * (1.0 - op.methylation_lock)
            op.vitality = op.vitality * (1.0 - effective_decay) + op_impact * effective_decay

        if len(self.operators) > 2:
            survivors = [self.operators[0]]
            for op in self.operators[1:]:
                if op.methylation_lock > 0.50 or op.vitality > 0.08:
                    survivors.append(op)
                else:
                    pruned = True
            self.operators = survivors

        meta = {
            "loss": loss.item(),
            "tau": tau.item(),
            "thinking_cycles": thinking_cycles,
            "sprouted": sprouted,
            "pruned": pruned,
            "active_nodes": len(self.operators),
            "locked_nodes": locked_nodes_count,
            "pred": torch.argmax(logits, dim=-1).item()
        }

        return loss, logits, meta


def run_exp_413():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-413: SOVEREIGN QUANTUM-SPIN COMPLEX WAVE RESONANCE (QSCW) ===")
    logger.info("================================================================================")

    config = EXP413Config()
    engine = QuantumSovereignMorphicEngine(config)

    def get_all_params():
        params = (list(engine.super_op.parameters()) +
                  list(engine.readout_real.parameters()) +
                  list(engine.readout_imag.parameters()))
        for op in engine.operators:
            params.extend(op.get_params())
        return params

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
    logger.info("Starting single-pass autopoietic execution (N=1)...")

    sprout_events = 0
    prune_events = 0
    correct_preds = 0
    total_loss = 0.0
    total_thinking_cycles = 0

    t_start = time.perf_counter()

    for t in range(len(raw_bytes) - 1):
        x_byte = raw_bytes[t]
        target_byte = raw_bytes[t + 1]

        param_groups = [
            {'params': (list(engine.super_op.parameters()) +
                        list(engine.readout_real.parameters()) +
                        list(engine.readout_imag.parameters())),
             'lr': config.learning_rate}
        ]
        for op in engine.operators:
            effective_lr = config.learning_rate * max(0.05, 1.0 - op.methylation_lock * 0.85)
            param_groups.append({'params': op.get_params(), 'lr': effective_lr})

        optimizer = torch.optim.AdamW(param_groups)
        optimizer.zero_grad()

        loss, logits, meta = engine.forward_step(x_byte, target_byte)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(get_all_params(), max_norm=1.0)
        optimizer.step()

        if meta["sprouted"]:
            sprout_events += 1
            logger.info(f"[STEP {t}] 🌱 SPONTANEOUS GENESIS! Nodes: {meta['active_nodes']} | Cycles: {meta['thinking_cycles']}")

        if meta["pruned"]:
            prune_events += 1
            logger.info(f"[STEP {t}] 🍂 TARGETED APOPTOSIS! Nodes: {meta['active_nodes']}")

        if meta["pred"] == target_byte:
            correct_preds += 1

        total_loss += meta["loss"]
        total_thinking_cycles += meta["thinking_cycles"]

        if (t + 1) % 500 == 0:
            avg_loss = total_loss / (t + 1)
            acc = (correct_preds / (t + 1)) * 100.0
            avg_cycles = total_thinking_cycles / (t + 1)
            logger.info(f"Progress [{t+1}/{len(raw_bytes)-1}] | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Think: {avg_cycles:.2f} | Nodes: {meta['active_nodes']} (Locked: {meta['locked_nodes']})")

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_thinking_depth = total_thinking_cycles / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    logger.info("================================================================================")
    logger.info("=== EXP-413 COMPLETE TELEMETRY ===")
    logger.info(f"Final Average Loss: {final_loss:.4f}")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Average Thinking Depth: {avg_thinking_depth:.2f} cycles/byte")
    logger.info(f"Sprout Events: {sprout_events}")
    logger.info(f"Prune Events: {prune_events}")
    logger.info(f"Final Active Nodes: {len(engine.operators)}")
    logger.info(f"Final Locked Nodes: {meta['locked_nodes']}")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-413",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "avg_thinking_depth": round(avg_thinking_depth, 2),
        "sprout_events": sprout_events,
        "prune_events": prune_events,
        "final_active_nodes": len(engine.operators),
        "final_locked_nodes": meta["locked_nodes"],
        "throughput_steps_per_sec": round(throughput, 2),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_413_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_413()
