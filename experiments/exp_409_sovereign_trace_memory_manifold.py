"""
EXP-409: Sovereign Trace Memory Manifold & High-Rank Multilinear Dynamics (STMM-HRMD)
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Date: October 2026
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 21, Principle 22, Principle 27 & KEP Rule #12)

Theoretical Foundation:
Diagnosis from EXP-408:
Decoupling problem time from thinking time (k=1..5) and injecting sensory error pressure epsilon_t
lowered the Single-Pass loss to 4.6699. However, the system lacked a high-capacity outer-product
associative trace memory: long-range syntactic patterns (keywords, headers, declarations)
decayed within the single hidden vector h_t. Furthermore, low rank (R=32) restricted multilinear expressive capacity.

In EXP-409 Sovereign Autopoiesis:
1. Sovereign Trace Memory Manifold (STMM, M_t in R^{D x R_mem}):
   Super-Operator Phi is granted an endogenous continuous outer-product trace manifold:
      M_{t+1} = lambda_decay(h_t) * M_t + Phi_write(h_t) (x) Phi_key(x_t)
      h_recall = M_t * Phi_read(h_t)
   Both writing, associative retrieval, and non-linear memory decay are entirely self-governed
   by the continuous potentials of Phi.
2. High-Rank Tensor Dynamics (Rank expanded from 32 to 64):
   Doubles the multilinear phase rank of all dynamically synthesized operators, quadrupling expressive capacity.
3. Full Recurrent Thinking & Error Pressure:
   Maintains the validated SRT (recurrent thinking cycles) and PEP (continuous sensory error pressure) from EXP-408.
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
logger = logging.getLogger("EXP-409-STMM-HRMD")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP409Config:
    exp_id: str = "EXP-409"
    dim: int = 258
    rank: int = 64              # Expanded from 32 to 64 for 4x multilinear capacity
    mem_rank: int = 64          # Rank for the continuous Trace Memory Manifold
    max_synthesized_nodes: int = 16
    max_thinking_steps: int = 5
    learning_rate: float = 0.0035
    stream_length: int = 3000
    device_str: str = DEVICE_STR


class EndogenousOperator:
    """
    Continuous Mathematical Operator synthesized and maintained endogenously (Rank 64).
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
        return torch.tanh(self.alpha_epi) * op_out


class SovereignSuperOperator(nn.Module):
    """
    The Super-Operator Phi governing high-rank dynamics, recurrent thinking, and trace memory.
    """
    def __init__(self, dim: int, rank: int, mem_rank: int):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.mem_rank = mem_rank

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

        # 3. Trace Memory Portals (Write Key/Value & Read Query):
        # Maps state to associative write/read operators
        self.mem_write_val = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh()
        ).to(DEVICE)

        self.mem_write_key = nn.Sequential(
            nn.Linear(dim, mem_rank),
            nn.Tanh()
        ).to(DEVICE)

        self.mem_read_query = nn.Sequential(
            nn.Linear(dim, mem_rank),
            nn.Tanh()
        ).to(DEVICE)

        # Endogenous memory gating: [write_gain, read_gain, decay_factor]
        self.mem_governance = nn.Sequential(
            nn.Linear(dim * 2, rank),
            nn.Tanh(),
            nn.Linear(rank, 3)
        ).to(DEVICE)

        # 4. Endogenous Genesis & Freezing Potential Network
        self.genesis_potential = nn.Sequential(
            nn.Linear(dim * 2 + 1, rank),
            nn.Tanh(),
            nn.Linear(rank, 5)
        ).to(DEVICE)

        # 5. Operator Lock Evaluator
        self.lock_evaluator = nn.Sequential(
            nn.Linear(rank + dim, rank),
            nn.Tanh(),
            nn.Linear(rank, 1)
        ).to(DEVICE)

        # 6. Raw Hyper-Synthesis Projection
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

    def compute_trace_memory_interaction(self, x: torch.Tensor, h: torch.Tensor, m_trace: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Endogenous associative write and read operations on the trace matrix M_trace [1, dim, mem_rank].
        """
        combined = torch.cat([x, h], dim=-1)
        gov = self.mem_governance(combined)  # [1, 3]

        write_gain = torch.sigmoid(gov[:, 0:1])
        read_gain = torch.sigmoid(gov[:, 1:2])
        decay_factor = torch.sigmoid(gov[:, 2:3]) * 0.98  # Memory persistence

        # Read: h_read = M_trace * query
        q = self.mem_read_query(h)  # [1, mem_rank]
        # m_trace is [1, dim, mem_rank], q is [1, mem_rank] -> [1, dim, 1]
        recalled = torch.bmm(m_trace, q.unsqueeze(-1)).squeeze(-1)  # [1, dim]
        recalled = read_gain * torch.tanh(recalled)

        # Write: new trace outer product = val (x) key
        v = self.mem_write_val(h)   # [1, dim]
        k = self.mem_write_key(x)   # [1, mem_rank]
        delta_m = torch.bmm(v.unsqueeze(-1), k.unsqueeze(1))  # [1, dim, mem_rank]

        # Update trace: decay old, accumulate new
        m_next = decay_factor.view(1, 1, 1) * m_trace + write_gain.view(1, 1, 1) * delta_m

        return recalled, m_next

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
    Complete Autonomous Engine featuring Trace Memory Manifold, Recurrent Thinking & Error Pressure.
    """
    def __init__(self, config: EXP409Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.rank = config.rank
        self.mem_rank = config.mem_rank
        self.max_nodes = config.max_synthesized_nodes
        self.max_thinking_steps = config.max_thinking_steps

        self.super_op = SovereignSuperOperator(self.dim, self.rank, self.mem_rank)
        self.readout = nn.Linear(self.dim, self.dim, bias=False).to(DEVICE)

        # Primordial crystalline anchor
        self.operators: List[EndogenousOperator] = [EndogenousOperator(self.dim, self.rank, DEVICE)]
        self.operators[0].alpha_epi.data.fill_(1.2)
        self.operators[0].methylation_lock = 1.0

        self.h = torch.zeros(1, self.dim, device=DEVICE)
        # Trace Memory Manifold M_t [1, dim, mem_rank]
        self.m_trace = torch.zeros(1, self.dim, self.mem_rank, device=DEVICE)
        self.prev_error_pressure = torch.zeros(1, self.dim, device=DEVICE)
        self.genesis_accumulator = torch.zeros(1, device=DEVICE)

    def forward_step(self, x_byte: int, target_byte: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        x = F.one_hot(torch.tensor([x_byte], device=DEVICE), num_classes=self.dim).float()
        num_ops = len(self.operators)

        # 1. Trace Memory Associative Recall and Update
        recalled_memory, m_next = self.super_op.compute_trace_memory_interaction(x, self.h, self.m_trace)
        self.m_trace = m_next.detach()  # Persistent continuous trace manifold

        # 2. Dynamic Covariant Coupling
        coupling_weights = self.super_op.compute_data_coupling(x, self.h, num_ops)

        # 3. Sovereign Recurrent Thinking Loop
        h_rec = self.h + recalled_memory  # Memory primes latent settling
        h_prev = self.h
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

            h_next = torch.tanh(h_rec + 0.5 * morphic_drift)

            halt_prob = self.super_op.compute_halting_probability(h_next, h_prev)
            h_prev = h_rec
            h_rec = h_next

            if halt_prob.item() > 0.65 and k >= 1:
                break

        # Final Readout
        logits = self.readout(h_rec)
        loss = F.cross_entropy(logits, torch.tensor([target_byte], device=DEVICE))

        # 4. Continuous Error Pressure Update
        with torch.no_grad():
            target_one_hot = F.one_hot(torch.tensor([target_byte], device=DEVICE), num_classes=self.dim).float()
            pred_probs = F.softmax(logits, dim=-1)
            sensory_error = target_one_hot - pred_probs
            self.prev_error_pressure = 0.8 * self.prev_error_pressure + 0.2 * sensory_error

        # 5. Endogenous Potential
        phenom = self.super_op.compute_endogenous_potential(x, h_rec, loss.detach())
        tau = phenom["time_dilation"]
        sprout_flux = phenom["sprout_flux"]
        apoptosis_flux = phenom["apoptosis_flux"]
        freeze_flux = phenom["memory_freeze_flux"]

        self.h = torch.tanh(h_rec * tau).detach()

        # Update methylation locks
        locked_nodes_count = 0
        for i, op in enumerate(self.operators):
            op.age += 1
            if i > 0:
                target_lock = self.super_op.compute_operator_lock(self.h, op) * freeze_flux.item()
                op.methylation_lock = 0.95 * op.methylation_lock + 0.05 * target_lock
            if op.methylation_lock > 0.60:
                locked_nodes_count += 1

        # Spontaneous morphogenesis
        self.genesis_accumulator += sprout_flux.squeeze() - 0.38
        sprouted = False
        pruned = False

        if self.genesis_accumulator.item() > 1.0 and len(self.operators) < self.max_nodes:
            seed = torch.cat([self.h, sprout_flux.repeat(1, self.rank)], dim=-1)
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
            "tau": tau.item(),
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


def run_exp_409():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-409: SOVEREIGN TRACE MEMORY MANIFOLD & HIGH-RANK DYNAMICS ===")
    logger.info("================================================================================")

    config = EXP409Config()
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
            logger.info(f"Progress [{t+1}/{len(raw_bytes)-1}] | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Avg Think Depth: {avg_cycles:.2f} | Nodes: {meta['active_nodes']} (Locked: {meta['locked_nodes']})")

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_thinking_depth = total_thinking_cycles / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    logger.info("================================================================================")
    logger.info("=== EXP-409 COMPLETE TELEMETRY ===")
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
        "exp_id": "EXP-409",
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

    with open("experiments/exp_409_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_409()
