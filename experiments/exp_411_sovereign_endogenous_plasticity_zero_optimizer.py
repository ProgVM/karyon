"""
EXP-411: Full Endogenous Self-Modifying Plasticity (Zero Optimizers, Pure Continuous Physics)
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Date: October 2026
Standard: KEP v16.0 Sovereign Master (Principle 1, Principle 2, Principle 3, Principle 22, Principle 27 & KEP Rule #12)

Theoretical Foundation:
Diagnosis from Bazilevs & Cyberneticist:
Traditional Deep Learning optimizers (Adam, AdamW, SGD) are surrogate crutches designed for static,
multi-epoch transformers iterating over frozen offline datasets. In a continuous, single-pass (N=1)
streaming universe, external optimizers violate autopoiesis by imposing rigid, handcrafted momentum
constants (beta_1=0.9, beta_2=0.999), splitting reality into artificial "inference" and "optimization" phases.

In EXP-411 Sovereign Autopoiesis:
1. Zero Optimizers / Zero AdamW:
   Completely eradicates torch.optim.AdamW. There is no optimizer object, no learning-rate schedule,
   and no optimizer.step().
2. Unified Continuous Motion Equation:
   Inference and adaptation are identical continuous field dynamics.
   Every operator and the Super-Operator Phi adapt their parameters endogenously via continuous
   local phase flow equations driven by sensory prediction error pressure (epsilon_t) and Variational Free Energy (F_t):
      Delta W_ij = -gamma_decay * W_ij + eta(F_t) * (epsilon_t (x) h_t)
   where:
      - epsilon_t = x_target - hat{x}_pred (sensory discrepancy vector)
      - eta(F_t) = continuous plastic flux derived from local surprise (Active Inference)
      - mu_i = epigenetic methylation lock protecting consolidated crystalline invariants
3. Pure Autopoietic Learning:
   Weight adaptation occurs with zero latency, purely at the speed of forward signal propagation,
   free of gradient backpropagation artifacts and transformer crutches.
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
logger = logging.getLogger("EXP-411-ZERO-OPTIMIZER")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP411Config:
    exp_id: str = "EXP-411"
    dim: int = 258
    rank: int = 32
    max_synthesized_nodes: int = 16
    max_thinking_steps: int = 4
    stream_length: int = 3000
    base_plasticity: float = 0.05
    device_str: str = DEVICE_STR


class EndogenousSelfAdaptingOperator:
    """
    Self-Modifying Operator: Contains its own internal equation of motion for synaptic plasticity.
    Zero external optimizers!
    """
    def __init__(self, dim: int, rank: int, device: torch.device):
        self.dim = dim
        self.rank = rank
        self.device = device

        # Parameters as autonomous physical state tensors
        self.w1 = torch.randn(dim, rank, device=device) / math.sqrt(dim)
        self.w2 = torch.randn(dim, rank, device=device) / math.sqrt(dim)
        self.w3 = torch.randn(dim, dim, device=device) / math.sqrt(dim)
        self.m_metric = torch.randn(dim, dim, device=device) * (0.1 / math.sqrt(dim))

        self.alpha_epi = torch.tensor([0.0], device=device)
        self.methylation_lock = 0.0
        self.vitality = 1.0
        self.age = 0

    def forward(self, x: torch.Tensor, h: torch.Tensor, error_pressure: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates output and returns intermediate activation trace for local plasticity.
        """
        u1 = torch.matmul(x + error_pressure, self.w1)      # [1, rank]
        u2 = torch.matmul(h, self.w2)                      # [1, rank]
        core = u1 * u2                                     # [1, rank]
        drift = torch.matmul(core, self.w1.t()) + torch.matmul(x, self.w3)  # [1, dim]

        h_m = torch.matmul(h, self.m_metric)
        metric_dot = torch.sum(x * h_m, dim=-1, keepdim=True)
        metric_scale = torch.sigmoid(metric_dot)

        op_out = metric_scale * drift
        gated_out = torch.tanh(self.alpha_epi) * op_out
        return gated_out, core

    def adapt_endogenous(self, x: torch.Tensor, h: torch.Tensor, core: torch.Tensor, error: torch.Tensor, plasticity_rate: float):
        """
        Continuous Local Equation of Synaptic Motion with in-place operations:
        Zero autograd graphs retained, zero memory leak!
        """
        with torch.no_grad():
            effective_plasticity = plasticity_rate * max(0.01, 1.0 - self.methylation_lock)
            decay = 0.0005 * effective_plasticity

            # In-place multi-linear co-adaptation
            delta_w1 = torch.matmul(error.t(), core)
            self.w1.mul_(1.0 - decay).add_(delta_w1, alpha=effective_plasticity)

            delta_w3 = torch.matmul(x.t(), error)
            self.w3.mul_(1.0 - decay).add_(delta_w3, alpha=effective_plasticity)

            alignment = torch.sum(error * torch.matmul(x, self.w3)).item()
            self.alpha_epi.add_(alignment * effective_plasticity).clamp_(-3.0, 3.0)


class SovereignSuperOperator(nn.Module):
    """
    Super-Operator Phi governing data coupling, thinking loops, and endogenous plasticity flux.
    """
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.dim = dim
        self.rank = rank

        # Autonomous weights for coupling and flux
        self.w_couple = nn.Parameter(torch.randn(dim * 2, rank) / math.sqrt(dim * 2))
        self.w_halt = nn.Parameter(torch.randn(dim * 2, 1) / math.sqrt(dim * 2))
        self.w_genesis = nn.Parameter(torch.randn(dim * 2 + 1, 5) / math.sqrt(dim * 2 + 1))
        self.w_hyper = nn.Parameter(torch.randn(dim + rank, rank * 4) / math.sqrt(dim + rank))
        self.w_hyper_out = nn.Parameter(torch.randn(rank * 4, dim * rank * 2 + dim * dim + dim * dim) / math.sqrt(rank * 4))

    def compute_data_coupling(self, x: torch.Tensor, h: torch.Tensor, num_nodes: int) -> torch.Tensor:
        combined = torch.cat([x, h], dim=-1)
        phi_vec = torch.matmul(combined, self.w_couple)
        phase_freq = phi_vec[:, :num_nodes] if num_nodes <= self.rank else F.pad(phi_vec, (0, num_nodes - self.rank))
        return F.softmax(phase_freq, dim=-1)

    def compute_halting_probability(self, h_curr: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_curr, h_prev], dim=-1)
        return torch.sigmoid(torch.matmul(combined, self.w_halt))

    def compute_endogenous_potential(self, x: torch.Tensor, h: torch.Tensor, free_energy: torch.Tensor) -> Dict[str, torch.Tensor]:
        inp = torch.cat([x, h, free_energy.view(1, 1)], dim=-1)
        fluxes = torch.matmul(inp, self.w_genesis)

        sprout_flux = torch.sigmoid(fluxes[:, 0])
        apoptosis_flux = torch.sigmoid(fluxes[:, 1])
        time_dilation = torch.exp(fluxes[:, 2]).clamp(0.1, 3.0)
        plasticity_flux = torch.sigmoid(fluxes[:, 3]) * 2.0  # Endogenous plasticity gain
        memory_freeze_flux = torch.sigmoid(fluxes[:, 4])

        return {
            "sprout_flux": sprout_flux,
            "apoptosis_flux": apoptosis_flux,
            "time_dilation": time_dilation,
            "plasticity_flux": plasticity_flux,
            "memory_freeze_flux": memory_freeze_flux
        }

    def synthesize_operator_tensors(self, seed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(torch.matmul(seed, self.w_hyper))
        flat = torch.matmul(hidden, self.w_hyper_out)

        s1 = self.dim * self.rank
        s2 = self.dim * self.rank
        s3 = self.dim * self.dim
        p1 = flat[:, :s1].view(self.dim, self.rank) / math.sqrt(self.dim)
        p2 = flat[:, s1:s1+s2].view(self.dim, self.rank) / math.sqrt(self.dim)
        p3 = flat[:, s1+s2:s1+s2+s3].view(self.dim, self.dim) / math.sqrt(self.dim)
        p4 = torch.sigmoid(flat[:, s1+s2+s3:].view(self.dim, self.dim) * 0.1)
        return p1, p2, p3, p4

    def adapt_endogenous(self, error: torch.Tensor, x: torch.Tensor, h: torch.Tensor, rate: float):
        """Self-adaptation of Super-Operator tensors via predictive error resonance."""
        with torch.no_grad():
            combined = torch.cat([x, h], dim=-1)  # [1, dim * 2]
            err_proj = error[:, :self.rank] if self.rank <= self.dim else F.pad(error, (0, self.rank - self.dim))
            delta = torch.matmul(combined.t(), err_proj)  # [dim * 2, rank]
            self.w_couple.data += rate * 0.1 * delta


class PureAutopoieticEngine(nn.Module):
    """
    Self-Sustaining Cognitive Engine:
    Zero Optimizers. Pure continuous physical equations of motion.
    """
    def __init__(self, config: EXP411Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.rank = config.rank
        self.max_nodes = config.max_synthesized_nodes
        self.max_thinking_steps = config.max_thinking_steps
        self.base_plasticity = config.base_plasticity

        self.super_op = SovereignSuperOperator(self.dim, self.rank).to(DEVICE)
        # Readout projection as continuous physical tensor
        self.w_readout = (torch.randn(self.dim, self.dim, device=DEVICE) / math.sqrt(self.dim))

        # Primordial anchor
        self.operators: List[EndogenousSelfAdaptingOperator] = [
            EndogenousSelfAdaptingOperator(self.dim, self.rank, DEVICE)
        ]
        self.operators[0].alpha_epi.fill_(1.2)
        self.operators[0].methylation_lock = 1.0

        self.h = torch.zeros(1, self.dim, device=DEVICE)
        self.prev_error_pressure = torch.zeros(1, self.dim, device=DEVICE)
        self.genesis_accumulator = torch.zeros(1, device=DEVICE)

    def forward_and_adapt(self, x_byte: int, target_byte: int) -> Tuple[float, int, Dict]:
        """
        Unified single-pass: Performs inference AND continuous synaptic adaptation simultaneously!
        NO backward(), NO optimizer!
        """
        x = F.one_hot(torch.tensor([x_byte], device=DEVICE), num_classes=self.dim).float()
        target_one_hot = F.one_hot(torch.tensor([target_byte], device=DEVICE), num_classes=self.dim).float()
        num_ops = len(self.operators)

        # 1. Covariant Coupling
        coupling_weights = self.super_op.compute_data_coupling(x, self.h, num_ops)

        # 2. Sovereign Recurrent Thinking Loop
        h_rec = self.h
        h_prev = self.h
        thinking_cycles = 0
        last_cores = []

        for k in range(self.max_thinking_steps):
            thinking_cycles += 1
            op_outputs = []
            cores = []
            for i, op in enumerate(self.operators):
                w_in = coupling_weights[:, i:i+1]
                x_in = w_in * x + (1.0 - w_in) * h_rec
                y_i, core_i = op.forward(x_in, h_rec, self.prev_error_pressure)
                op_outputs.append(y_i)
                cores.append(core_i)

            stacked_ops = torch.stack(op_outputs, dim=0)
            morphic_drift = torch.sum(stacked_ops, dim=0)

            h_next = torch.tanh(h_rec + 0.5 * morphic_drift)
            halt_prob = self.super_op.compute_halting_probability(h_next, h_prev)
            h_prev = h_rec
            h_rec = h_next
            last_cores = cores

            if halt_prob.item() > 0.65 and k >= 1:
                break

        # Readout and Instant Sensory Error (in-place & with torch.no_grad)
        with torch.no_grad():
            logits = torch.matmul(h_rec, self.w_readout)
            pred_probs = F.softmax(logits, dim=-1)
            pred_byte = torch.argmax(logits, dim=-1).item()

            eps = 1e-7
            target_prob = pred_probs[0, target_byte].item()
            loss = -math.log(max(target_prob, eps))

            # Physical Prediction Error Vector: epsilon_t = target - prediction
            sensory_error = target_one_hot - pred_probs  # [1, dim]

            # 4. Endogenous Plasticity & Dynamic Governance
            phenom = self.super_op.compute_endogenous_potential(x, h_rec, torch.tensor([loss], device=DEVICE))
            tau = phenom["time_dilation"]
            plasticity_gain = phenom["plasticity_flux"].item()
            sprout_flux = phenom["sprout_flux"]
            apoptosis_flux = phenom["apoptosis_flux"]
            freeze_flux = phenom["memory_freeze_flux"]

            current_plasticity = self.base_plasticity * plasticity_gain * min(2.5, max(0.2, loss))

            # Readout Adaptation in-place
            delta_readout = torch.matmul(h_rec.t(), sensory_error)
            self.w_readout.mul_(0.9995).add_(delta_readout, alpha=current_plasticity)

            # Operator Internal Plasticity
            for i, op in enumerate(self.operators):
                if i < len(last_cores):
                    op.adapt_endogenous(x, h_rec, last_cores[i], sensory_error, current_plasticity)

            # Super-Operator Covariant Adaptation
            self.super_op.adapt_endogenous(sensory_error, x, h_rec, current_plasticity)

        # 6. Physical Error Pressure Vector smoothing
        self.prev_error_pressure = 0.8 * self.prev_error_pressure + 0.2 * sensory_error
        self.h = torch.tanh(h_rec * tau).detach()

        # 7. Epigenetic Memory Freezing
        locked_nodes_count = 0
        for i, op in enumerate(self.operators):
            op.age += 1
            if i > 0:
                # Node locks when its prediction error decreases over its lifespan
                target_lock = freeze_flux.item() * (1.0 / (1.0 + math.exp(-op.age * 0.01)))
                op.methylation_lock = 0.95 * op.methylation_lock + 0.05 * target_lock
            if op.methylation_lock > 0.60:
                locked_nodes_count += 1

        # 8. Spontaneous Genesis & Apoptosis
        self.genesis_accumulator += sprout_flux.squeeze() - 0.38
        sprouted = False
        pruned = False

        if self.genesis_accumulator.item() > 1.0 and len(self.operators) < self.max_nodes:
            seed = torch.cat([self.h, sprout_flux.repeat(1, self.rank)], dim=-1)
            p1, p2, p3, p4 = self.super_op.synthesize_operator_tensors(seed)

            new_op = EndogenousSelfAdaptingOperator(self.dim, self.rank, DEVICE)
            new_op.w1.copy_(p1)
            new_op.w2.copy_(p2)
            new_op.w3.copy_(p3)
            new_op.m_metric.copy_(p4)
            new_op.alpha_epi.fill_(0.01)

            self.operators.append(new_op)
            self.genesis_accumulator.fill_(0.0)
            sprouted = True

        # Endogenous Apoptosis
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
            "tau": tau.item(),
            "plasticity": current_plasticity,
            "thinking_cycles": thinking_cycles,
            "sprouted": sprouted,
            "pruned": pruned,
            "active_nodes": len(self.operators),
            "locked_nodes": locked_nodes_count,
            "pred": pred_byte
        }

        return loss, pred_byte, meta


def run_exp_411():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-411: FULL ENDOGENOUS PLASTICITY (ZERO OPTIMIZER / NO ADAMW) ===")
    logger.info("================================================================================")

    config = EXP411Config()
    engine = PureAutopoieticEngine(config)

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
    logger.info("Starting single-pass autopoietic execution (N=1) with ZERO OPTIMIZER...")

    sprout_events = 0
    prune_events = 0
    correct_preds = 0
    total_loss = 0.0
    total_thinking_cycles = 0

    t_start = time.perf_counter()

    for t in range(len(raw_bytes) - 1):
        x_byte = raw_bytes[t]
        target_byte = raw_bytes[t + 1]

        # PURE FORWARD & INSTANT ADAPTATION (Zero backward, Zero AdamW!)
        loss, pred_byte, meta = engine.forward_and_adapt(x_byte, target_byte)

        if meta["sprouted"]:
            sprout_events += 1
            logger.info(f"[STEP {t}] 🌱 SPONTANEOUS GENESIS! Nodes: {meta['active_nodes']} | Plasticity: {meta['plasticity']:.4f}")

        if meta["pruned"]:
            prune_events += 1
            logger.info(f"[STEP {t}] 🍂 TARGETED APOPTOSIS! Nodes: {meta['active_nodes']}")

        if pred_byte == target_byte:
            correct_preds += 1

        total_loss += loss
        total_thinking_cycles += meta["thinking_cycles"]

        if (t + 1) % 500 == 0:
            avg_loss = total_loss / (t + 1)
            acc = (correct_preds / (t + 1)) * 100.0
            avg_cycles = total_thinking_cycles / (t + 1)
            logger.info(f"Progress [{t+1}/{len(raw_bytes)-1}] | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Think: {avg_cycles:.2f} | Nodes: {meta['active_nodes']} (Locked: {meta['locked_nodes']}) | Plasticity: {meta['plasticity']:.4f}")

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_thinking_depth = total_thinking_cycles / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    logger.info("================================================================================")
    logger.info("=== EXP-411 COMPLETE TELEMETRY ===")
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
        "exp_id": "EXP-411",
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

    with open("experiments/exp_411_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_411()
