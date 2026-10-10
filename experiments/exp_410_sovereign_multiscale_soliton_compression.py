"""
EXP-410: Sovereign Multiscale Soliton Compression & Spherical Normalization (SMSC-Norm)
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Date: October 2026
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 21, Principle 22, Principle 27 & KEP Rule #12)

Theoretical Foundation:
Diagnosis from EXP-408 and EXP-409:
In EXP-408, granting recurrent thinking loops (k=1..5) and error pressure dropped loss to 4.6699.
In EXP-409, expanding rank to 64 with unnormalized outer products led to gradient dispersion (loss 4.8313).
Root physical bottleneck:
1. Micro-scale byte noise: Operating strictly on single-byte delta transitions forces the network
   to conflate high-frequency phonotactic byte transitions with low-frequency semantic syntactic context.
2. Norm Drift: Without continuous manifold normalization, additive recurrence causes amplitude dilation.

In EXP-410 Sovereign Autopoiesis:
1. Continuous Spherical Normalization (RMSNorm Manifold Anchor):
   Every recurrent transition and multilinear operator output is projected onto an energy-conserving
   spherical manifold, completely eradicating gradient explosion and amplitude dispersion.
2. Endogenous Multiscale Soliton Dynamics (Fast Micro-Phase h_fast + Slow Macro-Soliton h_slow):
   Super-Operator Phi natively integrates across dual continuous timescales:
      h_fast: High-frequency byte-level state responding to immediate transitions (tau_fast ~ 1.0)
      h_slow: Low-frequency macro-soliton integrating persistent lexical/syntactic invariants (tau_slow ~ 0.05)
      h_joint: Nonlinear wave interference h_joint = RMSNorm(W_int * [h_fast, h_slow])
3. Retains Validated Features from EXP-407/408:
   - Zero-hardcode Epigenetic Memory Freezing (mu_i) protecting consolidated crystalline nodes.
   - Sovereign Recurrent Thinking Loops (k = 1 ... K_max) with endogenous halting potential gamma_halt.
   - Continuous Prediction Error Pressure epsilon_t acting as direct physical force.
   - Optimal phase rank R = 32.
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
logger = logging.getLogger("EXP-410-SMSC-NORM")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP410Config:
    exp_id: str = "EXP-410"
    dim: int = 258
    rank: int = 32               # Calibrated optimal rank from EXP-407/408
    max_synthesized_nodes: int = 16
    max_thinking_steps: int = 5
    learning_rate: float = 0.0035
    stream_length: int = 3000
    device_str: str = DEVICE_STR


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Continuous Spherical RMS Normalization mapping states to unit-energy manifold."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class EndogenousOperator:
    """
    Continuous Mathematical Operator synthesized and maintained endogenously (Rank 32).
    """
    def __init__(self, dim: int, rank: int, device: torch.device):
        self.dim = dim
        self.rank = rank
        self.device = device

        self.w1 = (torch.randn(dim, rank, device=device) / math.sqrt(dim)).requires_grad_(True)
        self.w2 = (torch.randn(dim, rank, device=device) / math.sqrt(dim)).requires_grad_(True)
        self.w3 = (torch.randn(dim, dim, device=device) / math.sqrt(dim)).requires_grad_(True)
        self.m_metric = (torch.randn(dim, dim, device=device) * (0.1 / math.sqrt(dim))).requires_grad_(True)

        self.alpha_epi = torch.zeros(1, device=device).requires_grad_(True)
        self.methylation_lock = 0.0
        self.vitality = 1.0
        self.age = 0

    def get_params(self) -> List[torch.Tensor]:
        return [self.w1, self.w2, self.w3, self.m_metric, self.alpha_epi]

    def forward(self, x: torch.Tensor, h: torch.Tensor, error_pressure: torch.Tensor) -> torch.Tensor:
        u1 = torch.matmul(x + error_pressure, self.w1)
        u2 = torch.matmul(h, self.w2)
        core = u1 * u2
        drift = torch.matmul(core, self.w1.t()) + torch.matmul(x, self.w3)

        h_m = torch.matmul(h, self.m_metric)
        metric_dot = torch.sum(x * h_m, dim=-1, keepdim=True)
        metric_scale = torch.sigmoid(metric_dot)

        op_out = metric_scale * drift
        # Energy-conserving spherical projection
        return rms_norm(torch.tanh(self.alpha_epi) * op_out)


class SovereignSuperOperator(nn.Module):
    """
    The Super-Operator Phi governing multiscale soliton interference and normalized phase dynamics.
    """
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.dim = dim
        self.rank = rank

        # 1. Covariant Data-Manifold Coupling
        self.data_coupling_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, rank)
        ).to(DEVICE)

        # 2. Endogenous Halting & Thought Settling Network
        self.halting_net = nn.Sequential(
            nn.Linear(dim * 2, rank),
            nn.Tanh(),
            nn.Linear(rank, 1)
        ).to(DEVICE)

        # 3. Multiscale Wave Interference Synthesizer:
        # Fuses fast micro-wave and slow macro-soliton into joint state
        self.soliton_fusion_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        ).to(DEVICE)

        # 4. Soliton Timescale Governance: [tau_fast, lambda_slow, slow_injection_gain]
        self.timescale_net = nn.Sequential(
            nn.Linear(dim * 2, rank),
            nn.Tanh(),
            nn.Linear(rank, 3)
        ).to(DEVICE)

        # 5. Endogenous Genesis & Freezing Potential Network
        self.genesis_potential = nn.Sequential(
            nn.Linear(dim * 2 + 1, rank),
            nn.Tanh(),
            nn.Linear(rank, 5)
        ).to(DEVICE)

        # 6. Operator Lock Evaluator
        self.lock_evaluator = nn.Sequential(
            nn.Linear(rank + dim, rank),
            nn.Tanh(),
            nn.Linear(rank, 1)
        ).to(DEVICE)

        # 7. Raw Hyper-Synthesis Projection
        self.hyper_synth = nn.Sequential(
            nn.Linear(dim + rank, rank * 4),
            nn.Tanh(),
            nn.Linear(rank * 4, dim * rank * 2 + dim * dim + dim * dim)
        ).to(DEVICE)

    def compute_data_coupling(self, x: torch.Tensor, h: torch.Tensor, num_nodes: int) -> torch.Tensor:
        combined = torch.cat([x, h], dim=-1)
        phi_vec = self.data_coupling_net(combined)
        phase_freq = phi_vec[:, :num_nodes] if num_nodes <= self.rank else F.pad(phi_vec, (0, num_nodes - self.rank))
        weights = F.softmax(phase_freq, dim=-1)
        return weights

    def compute_halting_probability(self, h_curr: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_curr, h_prev], dim=-1)
        return torch.sigmoid(self.halting_net(combined))

    def compute_multiscale_dynamics(self, h_fast: torch.Tensor, h_slow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes dynamic time-constants and non-linear soliton wave interference.
        """
        combined = torch.cat([h_fast, h_slow], dim=-1)
        scales = self.timescale_net(combined)

        tau_fast = torch.exp(scales[:, 0:1]).clamp(0.2, 2.5)
        lambda_slow = torch.sigmoid(scales[:, 1:2]) * 0.95 + 0.04  # Invariant persistence [0.04, 0.99]
        slow_gain = torch.sigmoid(scales[:, 2:3])

        # Wave interference
        fused = self.soliton_fusion_net(combined)
        h_joint = rms_norm(h_fast + slow_gain * fused)

        return h_joint, tau_fast, lambda_slow

    def compute_endogenous_potential(self, x: torch.Tensor, h: torch.Tensor, free_energy: torch.Tensor) -> Dict[str, torch.Tensor]:
        inp = torch.cat([x, h, free_energy.view(1, 1)], dim=-1)
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

    def compute_operator_lock(self, h: torch.Tensor, op: EndogenousOperator) -> float:
        op_sig = torch.mean(op.w1, dim=0, keepdim=True)
        combined = torch.cat([op_sig, h], dim=-1)
        return torch.sigmoid(self.lock_evaluator(combined)).item()

    def synthesize_operator_tensors(self, seed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self.hyper_synth(seed)
        s1 = self.dim * self.rank
        s2 = self.dim * self.rank
        s3 = self.dim * self.dim
        p1 = flat[:, :s1].view(self.dim, self.rank) / math.sqrt(self.dim)
        p2 = flat[:, s1:s1+s2].view(self.dim, self.rank) / math.sqrt(self.dim)
        p3 = flat[:, s1+s2:s1+s2+s3].view(self.dim, self.dim) / math.sqrt(self.dim)
        p4 = torch.sigmoid(flat[:, s1+s2+s3:].view(self.dim, self.dim) * 0.1)
        return p1, p2, p3, p4


class AutonomousSovereignMorphicEngine(nn.Module):
    """
    Complete Autonomous Engine featuring Multiscale Soliton Dynamics & Spherical RMSNorm.
    """
    def __init__(self, config: EXP410Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.rank = config.rank
        self.max_nodes = config.max_synthesized_nodes
        self.max_thinking_steps = config.max_thinking_steps

        self.super_op = SovereignSuperOperator(self.dim, self.rank)
        self.readout = nn.Linear(self.dim, self.dim, bias=False).to(DEVICE)

        # Primordial crystalline anchor
        self.operators: List[EndogenousOperator] = [EndogenousOperator(self.dim, self.rank, DEVICE)]
        self.operators[0].alpha_epi.data.fill_(1.2)
        self.operators[0].methylation_lock = 1.0

        # Dual continuous states: Fast micro-phase + Slow macro-soliton
        self.h_fast = torch.zeros(1, self.dim, device=DEVICE)
        self.h_slow = torch.zeros(1, self.dim, device=DEVICE)
        self.prev_error_pressure = torch.zeros(1, self.dim, device=DEVICE)
        self.genesis_accumulator = torch.zeros(1, device=DEVICE)

    def forward_step(self, x_byte: int, target_byte: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        x = F.one_hot(torch.tensor([x_byte], device=DEVICE), num_classes=self.dim).float()
        num_ops = len(self.operators)

        # 1. Multiscale Wave Fusion
        h_joint, tau_fast, lambda_slow = self.super_op.compute_multiscale_dynamics(self.h_fast, self.h_slow)

        # 2. Covariant Coupling
        coupling_weights = self.super_op.compute_data_coupling(x, h_joint, num_ops)

        # 3. Sovereign Recurrent Thinking Loop over Spherical Manifold
        h_rec = h_joint
        h_prev = h_joint
        thinking_cycles = 0

        for k in range(self.max_thinking_steps):
            thinking_cycles += 1
            op_outputs = []
            for i, op in enumerate(self.operators):
                w_in = coupling_weights[:, i:i+1]
                x_in = w_in * x + (1.0 - w_in) * h_rec
                y_i = op.forward(x_in, h_rec, self.prev_error_pressure)
                op_outputs.append(y_i)

            stacked_ops = torch.stack(op_outputs, dim=0)
            morphic_drift = torch.sum(stacked_ops, dim=0)

            # Continuous normalized evolution step
            h_next = rms_norm(torch.tanh(h_rec + 0.5 * morphic_drift))

            halt_prob = self.super_op.compute_halting_probability(h_next, h_prev)
            h_prev = h_rec
            h_rec = h_next

            if halt_prob.item() > 0.65 and k >= 1:
                break

        # Final Readout from normalized joint manifold
        logits = self.readout(h_rec)
        loss = F.cross_entropy(logits, torch.tensor([target_byte], device=DEVICE))

        # 4. Continuous Error Pressure Vector Update
        with torch.no_grad():
            target_one_hot = F.one_hot(torch.tensor([target_byte], device=DEVICE), num_classes=self.dim).float()
            pred_probs = F.softmax(logits, dim=-1)
            sensory_error = target_one_hot - pred_probs
            self.prev_error_pressure = 0.8 * self.prev_error_pressure + 0.2 * sensory_error

        # 5. Dual-Timescale State Integration
        phenom = self.super_op.compute_endogenous_potential(x, h_rec, loss.detach())
        sprout_flux = phenom["sprout_flux"]
        apoptosis_flux = phenom["apoptosis_flux"]
        freeze_flux = phenom["memory_freeze_flux"]

        # Fast micro-state updates with dynamic tau
        new_h_fast = rms_norm(torch.tanh(h_rec * tau_fast))
        # Slow macro-soliton continuously accumulates invariant context
        new_h_slow = rms_norm(lambda_slow * self.h_slow + (1.0 - lambda_slow) * new_h_fast)

        self.h_fast = new_h_fast.detach()
        self.h_slow = new_h_slow.detach()

        # Update methylation locks
        locked_nodes_count = 0
        for i, op in enumerate(self.operators):
            op.age += 1
            if i > 0:
                target_lock = self.super_op.compute_operator_lock(self.h_fast, op) * freeze_flux.item()
                op.methylation_lock = 0.95 * op.methylation_lock + 0.05 * target_lock
            if op.methylation_lock > 0.60:
                locked_nodes_count += 1

        # Spontaneous morphogenesis
        self.genesis_accumulator += sprout_flux.squeeze() - 0.38
        sprouted = False
        pruned = False

        if self.genesis_accumulator.item() > 1.0 and len(self.operators) < self.max_nodes:
            seed = torch.cat([self.h_fast, sprout_flux.repeat(1, self.rank)], dim=-1)
            p1, p2, p3, p4 = self.super_op.synthesize_operator_tensors(seed)

            new_op = EndogenousOperator(self.dim, self.rank, DEVICE)
            new_op.w1.data.copy_(p1)
            new_op.w2.data.copy_(p2)
            new_op.w3.data.copy_(p3)
            new_op.m_metric.data.copy_(p4)
            new_op.alpha_epi.data.fill_(0.01)

            self.operators.append(new_op)
            self.genesis_accumulator.data.fill_(0.0)
            sprouted = True

        # Endogenous Apoptosis with Epigenetic Protection
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
            "tau_fast": tau_fast.item(),
            "lambda_slow": lambda_slow.item(),
            "thinking_cycles": thinking_cycles,
            "sprouted": sprouted,
            "pruned": pruned,
            "active_nodes": len(self.operators),
            "locked_nodes": locked_nodes_count,
            "sprout_flux": sprout_flux.item(),
            "freeze_flux": freeze_flux.item(),
            "pred": torch.argmax(logits, dim=-1).item()
        }

        return loss, logits, meta


def run_exp_410():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-410: MULTISCALE SOLITON COMPRESSION & RMSNORM (SMSC-NORM) ===")
    logger.info("================================================================================")

    config = EXP410Config()
    engine = AutonomousSovereignMorphicEngine(config)

    def get_all_params():
        params = list(engine.super_op.parameters()) + list(engine.readout.parameters())
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
            {'params': list(engine.super_op.parameters()) + list(engine.readout.parameters()), 'lr': config.learning_rate}
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
            logger.info(f"[STEP {t}] 🌱 SPONTANEOUS GENESIS! Nodes: {meta['active_nodes']} | Think: {meta['thinking_cycles']}")

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
            logger.info(f"Progress [{t+1}/{len(raw_bytes)-1}] | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Think: {avg_cycles:.2f} | Nodes: {meta['active_nodes']} (Locked: {meta['locked_nodes']}) | Soliton: {meta['lambda_slow']:.3f}")

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_thinking_depth = total_thinking_cycles / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    logger.info("================================================================================")
    logger.info("=== EXP-410 COMPLETE TELEMETRY ===")
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
        "exp_id": "EXP-410",
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

    with open("experiments/exp_410_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_410()
