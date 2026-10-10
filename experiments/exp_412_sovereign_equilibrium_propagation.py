"""
EXP-412: Continuous Sovereign Equilibrium Propagation (CSEP)
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Date: October 2026
Standard: KEP v16.0 Sovereign Master (Principle 1, Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Theoretical Foundation:
Diagnosis from EXP-411:
First-order naive local Hebbian plasticity (dW = eta * (error (x) activity)) failed (loss 10.11)
because individual operators had no coordinate alignment with downstream prediction errors.
In biophysics and analog physical substrates (Hopfield, Scellier & Bengio 2017, Whittington-Bogacz 2017),
the exact mathematical equivalence to backpropagation without autograd or external optimizers
is achieved via EQUILIBRIUM PROPAGATION / ENERGY RELAXATION:

Energy Landscape:
   E(h) = 0.5 * ||h||^2 - \sum_i Contraction_i(x, h, W_i) + 0.5 * beta_clamp * ||Readout(h) - target||^2

Two Continuous Physical Phases:
1. Free Phase:
   The network state settles to a local energy minimum h_free where dE/dh = 0 (no target pressure).
   Prediction hat{x} = Readout(h_free) is emitted.
2. Clamped / Nudged Phase:
   A gentle nudging target force is applied at the output. State relaxes to h_clamped.
3. Pure Local Contrastive Learning Rule:
   dW_ij = (1 / beta_clamp) * [ dE_local/dW(h_clamped) - dE_local/dW(h_free) ]
   - 100% strictly local in space and time.
   - Zero autograd computation graphs retained.
   - Zero Adam / AdamW / SGD optimizers!
   - Pure physical analog relaxation!
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
logger = logging.getLogger("EXP-412-CSEP")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP412Config:
    exp_id: str = "EXP-412"
    dim: int = 258
    rank: int = 32
    max_synthesized_nodes: int = 16
    free_settle_steps: int = 4
    nudged_settle_steps: int = 3
    beta_nudge: float = 0.25
    learning_rate: float = 0.04
    stream_length: int = 3000
    device_str: str = DEVICE_STR


class EquilibriumOperator:
    """
    Operator defining a local continuous energy contribution E_op(x, h).
    """
    def __init__(self, dim: int, rank: int, device: torch.device):
        self.dim = dim
        self.rank = rank
        self.device = device

        self.w1 = torch.randn(dim, rank, device=device) / math.sqrt(dim)
        self.w2 = torch.randn(dim, rank, device=device) / math.sqrt(dim)
        self.w3 = torch.randn(dim, dim, device=device) / math.sqrt(dim)
        self.alpha_epi = torch.tensor([0.0], device=device)
        self.methylation_lock = 0.0
        self.vitality = 1.0
        self.age = 0

    def energy_drift(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes state drift dh = -dE/dh and intermediate activation traces.
        """
        u1 = torch.matmul(x, self.w1)      # [1, rank]
        u2 = torch.matmul(h, self.w2)      # [1, rank]
        core = u1 * u2                     # [1, rank]

        # Reconstruction force from multi-linear contraction
        drift = torch.matmul(core, self.w2.t()) + torch.matmul(x, self.w3)
        gated_drift = torch.tanh(self.alpha_epi) * drift
        return gated_drift, u1, u2

    def local_contrastive_adapt(self, x: torch.Tensor, h_free: torch.Tensor, u1_free: torch.Tensor, u2_free: torch.Tensor,
                                h_nudge: torch.Tensor, u1_nudge: torch.Tensor, u2_nudge: torch.Tensor,
                                beta_nudge: float, lr: float):
        """
        Equilibrium Propagation contrastive weight update:
        dW = (lr / beta) * [ Grad(nudge) - Grad(free) ]
        """
        effective_lr = (lr / beta_nudge) * max(0.01, 1.0 - self.methylation_lock)
        decay = 0.0005 * effective_lr

        # Contrastive tensor co-adaptation
        # Free vs Nudge correlation
        grad1_free = torch.matmul(x.t(), (u2_free * u2_free))
        grad1_nudge = torch.matmul(x.t(), (u2_nudge * u2_nudge))
        delta_w1 = grad1_nudge - grad1_free

        grad2_free = torch.matmul(h_free.t(), (u1_free * u1_free))
        grad2_nudge = torch.matmul(h_nudge.t(), (u1_nudge * u1_nudge))
        delta_w2 = grad2_nudge - grad2_free

        grad3_free = torch.matmul(x.t(), h_free)
        grad3_nudge = torch.matmul(x.t(), h_nudge)
        delta_w3 = grad3_nudge - grad3_free

        # In-place physical update
        self.w1.mul_(1.0 - decay).add_(delta_w1, alpha=effective_lr)
        self.w2.mul_(1.0 - decay).add_(delta_w2, alpha=effective_lr)
        self.w3.mul_(1.0 - decay).add_(delta_w3, alpha=effective_lr)

        # Epigenetic plasticity
        delta_epi = torch.sum(h_nudge.pow(2) - h_free.pow(2)).item()
        self.alpha_epi.add_(delta_epi * effective_lr).clamp_(-3.0, 3.0)


class SovereignSuperOperator(nn.Module):
    """
    Super-Operator Phi governing covariant coupling and autonomous operator genesis.
    """
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.dim = dim
        self.rank = rank

        self.w_couple = nn.Parameter(torch.randn(dim * 2, rank) / math.sqrt(dim * 2))
        self.w_genesis = nn.Parameter(torch.randn(dim * 2 + 1, 5) / math.sqrt(dim * 2 + 1))
        self.w_hyper = nn.Parameter(torch.randn(dim + rank, rank * 4) / math.sqrt(dim + rank))
        self.w_hyper_out = nn.Parameter(torch.randn(rank * 4, dim * rank * 2 + dim * dim) / math.sqrt(rank * 4))

    def compute_data_coupling(self, x: torch.Tensor, h: torch.Tensor, num_nodes: int) -> torch.Tensor:
        combined = torch.cat([x, h], dim=-1)
        phi_vec = torch.matmul(combined, self.w_couple)
        phase_freq = phi_vec[:, :num_nodes] if num_nodes <= self.rank else F.pad(phi_vec, (0, num_nodes - self.rank))
        return F.softmax(phase_freq, dim=-1)

    def compute_endogenous_potential(self, x: torch.Tensor, h: torch.Tensor, free_energy: torch.Tensor) -> Dict[str, torch.Tensor]:
        inp = torch.cat([x, h, free_energy.view(1, 1)], dim=-1)
        fluxes = torch.matmul(inp, self.w_genesis)

        sprout_flux = torch.sigmoid(fluxes[:, 0])
        apoptosis_flux = torch.sigmoid(fluxes[:, 1])
        time_dilation = torch.exp(fluxes[:, 2]).clamp(0.1, 3.0)
        plasticity_flux = torch.sigmoid(fluxes[:, 3]) * 2.0
        memory_freeze_flux = torch.sigmoid(fluxes[:, 4])

        return {
            "sprout_flux": sprout_flux,
            "apoptosis_flux": apoptosis_flux,
            "time_dilation": time_dilation,
            "plasticity_flux": plasticity_flux,
            "memory_freeze_flux": memory_freeze_flux
        }

    def synthesize_operator_tensors(self, seed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(torch.matmul(seed, self.w_hyper))
        flat = torch.matmul(hidden, self.w_hyper_out)

        s1 = self.dim * self.rank
        s2 = self.dim * self.rank
        p1 = flat[:, :s1].view(self.dim, self.rank) / math.sqrt(self.dim)
        p2 = flat[:, s1:s1+s2].view(self.dim, self.rank) / math.sqrt(self.dim)
        p3 = flat[:, s1+s2:].view(self.dim, self.dim) / math.sqrt(self.dim)
        return p1, p2, p3


class EquilibriumAutopoieticEngine(nn.Module):
    """
    Equilibrium Propagation Engine:
    Zero Optimizers. Two physical settling phases (Free and Nudged).
    """
    def __init__(self, config: EXP412Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.rank = config.rank
        self.max_nodes = config.max_synthesized_nodes
        self.free_settle_steps = config.free_settle_steps
        self.nudged_settle_steps = config.nudged_settle_steps
        self.beta_nudge = config.beta_nudge
        self.lr = config.learning_rate

        self.super_op = SovereignSuperOperator(self.dim, self.rank).to(DEVICE)
        self.w_readout = torch.randn(self.dim, self.dim, device=DEVICE) / math.sqrt(self.dim)

        # Primordial crystalline node
        self.operators: List[EquilibriumOperator] = [
            EquilibriumOperator(self.dim, self.rank, DEVICE)
        ]
        self.operators[0].alpha_epi.fill_(1.2)
        self.operators[0].methylation_lock = 1.0

        self.h = torch.zeros(1, self.dim, device=DEVICE)
        self.genesis_accumulator = torch.zeros(1, device=DEVICE)

    def relax_state(self, x: torch.Tensor, h_init: torch.Tensor, num_steps: int,
                    target_force: torch.Tensor = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Continuous physical settling to energy minimum dE/dh = 0.
        """
        h = h_init.clone()
        num_ops = len(self.operators)
        coupling = self.super_op.compute_data_coupling(x, h, num_ops)

        last_traces = []
        dt = 0.35

        for _ in range(num_steps):
            total_drift = torch.zeros_like(h)
            traces = []
            for i, op in enumerate(self.operators):
                w_in = coupling[:, i:i+1]
                x_in = w_in * x + (1.0 - w_in) * h
                op_drift, u1, u2 = op.energy_drift(x_in, h)
                total_drift += op_drift
                traces.append((u1, u2))

            # Internal recurrent restoration force
            # dh = - (h - total_drift)
            dh = -h + torch.tanh(total_drift)

            # If nudged phase, add output target pressure
            if target_force is not None:
                dh += target_force

            h = h + dt * dh
            last_traces = traces

        return h, last_traces

    def forward_and_adapt(self, x_byte: int, target_byte: int) -> Tuple[float, int, Dict]:
        with torch.no_grad():
            x = F.one_hot(torch.tensor([x_byte], device=DEVICE), num_classes=self.dim).float()
            target_one_hot = F.one_hot(torch.tensor([target_byte], device=DEVICE), num_classes=self.dim).float()

            # =========================================================================
            # PHASE 1: FREE PHASE SETTLING (Unclamped Energy Minimum)
            # =========================================================================
            h_free, traces_free = self.relax_state(x, self.h, self.free_settle_steps, target_force=None)

            # Emit Prediction
            logits_free = torch.matmul(h_free, self.w_readout)
            probs_free = F.softmax(logits_free, dim=-1)
            pred_byte = torch.argmax(logits_free, dim=-1).item()

            target_prob = probs_free[0, target_byte].item()
            loss = -math.log(max(target_prob, 1e-7))

            # Compute Output Target Force for Nudged Phase
            # F_target = beta_nudge * (target - probs) @ W_readout.T
            error_out = target_one_hot - probs_free
            nudging_force = self.beta_nudge * torch.matmul(error_out, self.w_readout.t())

            # =========================================================================
            # PHASE 2: NUDGED PHASE SETTLING (State Relaxed Under Target Force)
            # =========================================================================
            h_nudge, traces_nudge = self.relax_state(x, h_free, self.nudged_settle_steps, target_force=nudging_force)

            # =========================================================================
            # PHASE 3: CONTRASTIVE EQUILIBRIUM WEIGHT ADAPTATION (Zero Optimizers!)
            # =========================================================================
            phenom = self.super_op.compute_endogenous_potential(x, h_free, torch.tensor([loss], device=DEVICE))
            plasticity_gain = phenom["plasticity_flux"].item()
            effective_lr = self.lr * plasticity_gain

            # Readout Contrastive Update:
            # dW_readout = (lr / beta) * (h_nudge.T @ target - h_free.T @ probs_free)
            delta_readout = (torch.matmul(h_nudge.t(), target_one_hot) - torch.matmul(h_free.t(), probs_free)) / self.beta_nudge
            self.w_readout.mul_(0.9995).add_(delta_readout, alpha=effective_lr)

            # Operator Contrastive Updates
            for i, op in enumerate(self.operators):
                if i < len(traces_free) and i < len(traces_nudge):
                    u1_f, u2_f = traces_free[i]
                    u1_n, u2_n = traces_nudge[i]
                    op.local_contrastive_adapt(x, h_free, u1_f, u2_f, h_nudge, u1_n, u2_n, self.beta_nudge, effective_lr)

            # Update continuous state to next step
            tau = phenom["time_dilation"]
            self.h = torch.tanh(h_free * tau)

            # Epigenetic Memory Freezing
            freeze_flux = phenom["memory_freeze_flux"]
            locked_nodes_count = 0
            for i, op in enumerate(self.operators):
                op.age += 1
                if i > 0:
                    target_lock = freeze_flux.item() * (1.0 / (1.0 + math.exp(-op.age * 0.01)))
                    op.methylation_lock = 0.95 * op.methylation_lock + 0.05 * target_lock
                if op.methylation_lock > 0.60:
                    locked_nodes_count += 1

            # Genesis & Apoptosis
            sprout_flux = phenom["sprout_flux"]
            apoptosis_flux = phenom["apoptosis_flux"]
            self.genesis_accumulator += sprout_flux.squeeze() - 0.38
            sprouted = False
            pruned = False

            if self.genesis_accumulator.item() > 1.0 and len(self.operators) < self.max_nodes:
                seed = torch.cat([self.h, sprout_flux.repeat(1, self.rank)], dim=-1)
                p1, p2, p3 = self.super_op.synthesize_operator_tensors(seed)

                new_op = EquilibriumOperator(self.dim, self.rank, DEVICE)
                new_op.w1.copy_(p1)
                new_op.w2.copy_(p2)
                new_op.w3.copy_(p3)
                new_op.alpha_epi.fill_(0.01)

                self.operators.append(new_op)
                self.genesis_accumulator.fill_(0.0)
                sprouted = True

            for op in self.operators[1:]:
                op_impact = abs(math.tanh(op.alpha_epi.item()))
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
                "loss": loss,
                "plasticity": effective_lr,
                "sprouted": sprouted,
                "pruned": pruned,
                "active_nodes": len(self.operators),
                "locked_nodes": locked_nodes_count,
                "pred": pred_byte
            }

            return loss, pred_byte, meta


def run_exp_412():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-412: CONTINUOUS SOVEREIGN EQUILIBRIUM PROPAGATION (CSEP) ===")
    logger.info("================================================================================")

    config = EXP412Config()
    engine = EquilibriumAutopoieticEngine(config)

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
    logger.info("Starting single-pass autopoietic execution (N=1) with EQUILIBRIUM PROPAGATION...")

    sprout_events = 0
    prune_events = 0
    correct_preds = 0
    total_loss = 0.0

    t_start = time.perf_counter()

    for t in range(len(raw_bytes) - 1):
        x_byte = raw_bytes[t]
        target_byte = raw_bytes[t + 1]

        loss, pred_byte, meta = engine.forward_and_adapt(x_byte, target_byte)

        if meta["sprouted"]:
            sprout_events += 1
            logger.info(f"[STEP {t}] 🌱 SPONTANEOUS GENESIS! Nodes: {meta['active_nodes']}")

        if meta["pruned"]:
            prune_events += 1
            logger.info(f"[STEP {t}] 🍂 TARGETED APOPTOSIS! Nodes: {meta['active_nodes']}")

        if pred_byte == target_byte:
            correct_preds += 1

        total_loss += loss

        if (t + 1) % 500 == 0:
            avg_loss = total_loss / (t + 1)
            acc = (correct_preds / (t + 1)) * 100.0
            logger.info(f"Progress [{t+1}/{len(raw_bytes)-1}] | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Nodes: {meta['active_nodes']} (Locked: {meta['locked_nodes']})")

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    throughput = (len(raw_bytes) - 1) / t_elapsed

    logger.info("================================================================================")
    logger.info("=== EXP-412 COMPLETE TELEMETRY ===")
    logger.info(f"Final Average Loss: {final_loss:.4f}")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Sprout Events: {sprout_events}")
    logger.info(f"Prune Events: {prune_events}")
    logger.info(f"Final Active Nodes: {len(engine.operators)}")
    logger.info(f"Final Locked Nodes: {meta['locked_nodes']}")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-412",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "sprout_events": sprout_events,
        "prune_events": prune_events,
        "final_active_nodes": len(engine.operators),
        "final_locked_nodes": meta["locked_nodes"],
        "throughput_steps_per_sec": round(throughput, 2),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_412_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_412()
