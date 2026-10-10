"""
EXP-395: Pure Endogenous Structural Genesis Substrate (PESGS)
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Date: October 2026
Standard: KEP v16.0 Sovereign Master (Strict Principle 27: Universal Continuous Substrate, Zero Hardcoded Cognitive Classes)

Architectural Implementation:
1. Complete Eradication of Hardcoded Architectural Classes:
   No static "LaminarLayers", no hardcoded "PACDecoders", no rigid "Neurotransmitter" classes.
2. Universal Continuous Operator Genesis:
   Karyon is provided strictly with a physical manifold Psi in R^D and an autopoietic bank of primitive tensor kernels:
   - Kernel 0: Leaky Continuous Integration ( dPsi/dt = - gamma * Psi + W_in u )
   - Kernel 1: Bilinear Conjunction / Gating ( Psi_out = (W_a Psi) * (W_b Psi) )
   - Kernel 2: Saturated Attractor Projection ( Psi_out = tanh( W_att Psi ) )
   - Kernel 3: Continuous Fast-Weight Hebbian Plasticity ( W_fast <- alpha * W_fast + eta * Psi * u^T )
3. Endogenous Topological Routing Matrix:
   The routing between operators, the depth of thinking recirculation k in [1..K_max],
   and the dynamic time-constants tau are governed entirely by internal Free Energy gradients:
   F_t = E_hopfield(Psi_motor) + lambda_strain * Strain(Psi).
4. Strict Single-Pass Streaming Paradigm (N=1, zero epochs):
   Temporal reality is presented once, online, with instantaneous continual learning.
"""

import os
import sys
import math
import time
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-395")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class EXP395Config:
    exp_id: str = "EXP-395"
    dim: int = 128
    num_basins: int = 258
    max_operators: int = 16
    initial_operators: int = 4
    max_latent_cycles: int = 4
    learning_rate: float = 0.003
    fast_weight_eta: float = 0.05
    beta_hopfield: float = 16.0
    device: str = str(DEVICE)


class AtomicPhysicalKernel(nn.Module):
    """
    Primitive Continuous Tensor Transformation (Substrate Atom).
    Can express:
    0: Leaky integrator
    1: Bilinear interaction
    2: Non-linear attractor
    """
    def __init__(self, dim: int, kernel_type: int):
        super().__init__()
        self.dim = dim
        self.kernel_type = kernel_type
        
        self.W1 = nn.Linear(dim, dim, bias=False)
        self.W2 = nn.Linear(dim, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))
        
        # Continuous timescale parameter (endogenous integration rate)
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        # Epigenetic zero-shock gating factor
        self.alpha_epi = nn.Parameter(torch.tensor(0.0))
        
        # State register for leaky integration
        self.register_buffer("state_memory", torch.zeros(dim))

    def reset_state(self):
        self.state_memory.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = torch.exp(-torch.clamp(self.log_tau, -3.0, 3.0))
        
        if self.kernel_type == 0:
            # Leaky integration: s_t = (1 - tau)*s_{t-1} + tau * W1(x)
            proj = self.W1(x) + self.bias
            s_next = (1.0 - tau) * self.state_memory + tau * proj
            self.state_memory = s_next.detach()
            h = s_next
        elif self.kernel_type == 1:
            # Bilinear interaction / gating: W1(x) * sigmoid(W2(x))
            h = self.W1(x) * torch.sigmoid(self.W2(x) + self.bias)
        else:
            # Non-linear saturated attractor: tanh(W1(x)) + LayerNorm
            h = torch.tanh(self.W1(x) + self.bias)
            
        # Epigenetic bypass: guarantees identity when newly sprouted
        graft = torch.tanh(self.alpha_epi)
        return x + graft * (h - x)


class EndogenousAutopoieticSubstrate(nn.Module):
    """
    KEP Principle 27 Embodiment:
    Universal substrate where Karyon endogenously coordinates its own operators,
    routing topology, fast-weights, and thinking depth.
    """
    def __init__(self, config: EXP395Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        
        # 1. Continuous Hopfield Action Basins (Hypersphere S^{D-1})
        raw_basins = torch.randn(config.num_basins, config.dim)
        self.basins = nn.Parameter(F.normalize(raw_basins, p=2, dim=-1))
        
        # 2. Dynamic Bank of Atomic Continuous Kernels
        self.kernels = nn.ModuleList([
            AtomicPhysicalKernel(self.dim, kernel_type=i % 3)
            for i in range(config.initial_operators)
        ])
        for k in self.kernels:
            k.alpha_epi.data.fill_(1.0) # Initial base kernels active
            
        # 3. Endogenous Routing Field:
        # Karyon computes its own coupling tensor between kernels
        self.W_route = nn.Linear(self.dim, config.max_operators, bias=False)
        
        # 4. Continuous Fast-Weight Hebbian Matrix (Plastic Online Working Memory)
        self.register_buffer("W_fast", torch.zeros(self.dim, self.dim))
        
        # 5. Endogenous Internal Drive States (Energy, Noradrenaline, Dopamine)
        self.register_buffer("drive_vector", torch.tensor([1.0, 0.20, 0.50])) # [Energy, NA, DA]
        
        # Telemetry
        self.sprout_events = 0
        self.godel_rejections = 0
        self.step_counter = 0

    def get_normalized_basins(self) -> torch.Tensor:
        return F.normalize(self.basins, p=2, dim=-1)

    def perceive_byte(self, byte_idx: int) -> torch.Tensor:
        basins = self.get_normalized_basins()
        return basins[byte_idx]

    def reset_states(self):
        for k in self.kernels:
            k.reset_state()
        self.W_fast.zero_()

    def forward_autopoietic_step(
        self,
        u_t: torch.Tensor
    ) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Executes endogenous cognitive cycle:
        - Injects fast-weight trace
        - Recirculates through dynamic routing of operators
        - Halts thinking endogenously when delta-Psi stabilizes
        - Collapses state into Hopfield potential well
        """
        # 1. Fast-Weight Working Memory Injection: W_fast * u_t
        fast_recalled = torch.matmul(self.W_fast, u_t)
        psi_t = F.layer_norm(u_t + 0.5 * fast_recalled, (self.dim,))
        
        num_active = len(self.kernels)
        thinking_cycles = 1
        
        # 2. Endogenous Recurrent Thinking Loop
        for k in range(self.config.max_latent_cycles):
            psi_prev = psi_t
            
            # Compute endogenous dynamic routing coefficients
            routing_logits = self.W_route(psi_t)[:num_active]
            routing_weights = F.softmax(routing_logits, dim=-1) # [num_active]
            
            # Aggregate operator transformations
            op_outputs = []
            for i, kernel in enumerate(self.kernels):
                out_i = kernel(psi_t)
                op_outputs.append(out_i)
                
            stacked_ops = torch.stack(op_outputs, dim=0) # [num_active, dim]
            # Convex combination of continuous operators
            psi_transformed = torch.sum(routing_weights.unsqueeze(-1) * stacked_ops, dim=0)
            
            # Non-linear integration into physical manifold
            psi_t = F.layer_norm(psi_t + psi_transformed, (self.dim,))
            
            # Halting condition driven by internal equilibrium
            delta_psi = torch.norm(psi_t - psi_prev).item()
            if delta_psi < 0.05 and k > 0:
                thinking_cycles = k + 1
                break
            thinking_cycles = k + 1
            
        # 3. Collapse in Hopfield Energy Landscape
        psi_norm = F.normalize(psi_t, p=2, dim=-1)
        B = self.get_normalized_basins()
        
        dopamine = self.drive_vector[2].item()
        beta_eff = self.config.beta_hopfield * (1.0 + 1.5 * max(0.0, dopamine))
        
        # Alignments (energies)
        alignments = torch.matmul(B, psi_norm) # [258]
        collapsed_action = torch.argmax(alignments).item()
        
        # 4. Instant Fast-Weight Hebbian Plasticity Update:
        # W_fast <- 0.95 * W_fast + eta * (psi_t * u_t^T)
        with torch.no_grad():
            delta_W = torch.outer(psi_t, u_t)
            self.W_fast.mul_(0.95).add_(delta_W, alpha=self.config.fast_weight_eta)
            
        diag = {
            "thinking_cycles": thinking_cycles,
            "active_kernels": num_active,
            "noradrenaline": self.drive_vector[1].item(),
            "dopamine": self.drive_vector[2].item()
        }
        return psi_t, collapsed_action, diag

    def compute_free_energy(self, psi_t: torch.Tensor, target_idx: int) -> torch.Tensor:
        psi_norm = F.normalize(psi_t, p=2, dim=-1)
        B = self.get_normalized_basins()
        target_basin = B[target_idx]
        
        dopamine = self.drive_vector[2].item()
        beta_eff = self.config.beta_hopfield * (1.0 + 1.5 * max(0.0, dopamine))
        
        alignments = torch.matmul(B, psi_norm)
        
        log_part = torch.logsumexp(beta_eff * alignments, dim=-1)
        target_pot = beta_eff * alignments[target_idx]
        fe_hopfield = (log_part - target_pot) / beta_eff
        
        commit_strain = 1.0 - torch.dot(psi_norm, target_basin)
        return fe_hopfield + commit_strain

    def update_drives(self, fe_val: float):
        with torch.no_grad():
            # Noradrenaline surges with Free Energy surprise
            curr_na = self.drive_vector[1].item()
            new_na = 0.90 * curr_na + 0.10 * math.tanh(fe_val / 2.0)
            
            # Dopamine rewards successful low-surprise prediction
            curr_da = self.drive_vector[2].item()
            new_da = 0.90 * curr_da + 0.10 * math.exp(-fe_val)
            
            # Energy consumption
            curr_e = self.drive_vector[0].item()
            new_e = max(0.1, min(1.0, curr_e + 0.02 * new_da - 0.01 * new_na))
            
            self.drive_vector.copy_(torch.tensor([new_e, new_na, new_da], device=DEVICE))

    def attempt_kernel_morphogenesis(self, recent_bytes: List[int]):
        """
        Autonomously synthesizes and appends a new elementary physical kernel
        if approved by the Gödel Sandbox trial.
        """
        if len(self.kernels) >= self.config.max_operators or len(recent_bytes) < 16:
            return
            
        next_type = len(self.kernels) % 3
        candidate = AtomicPhysicalKernel(self.dim, kernel_type=next_type).to(DEVICE)
        
        # Micro-adaptation in Gödel Sandbox
        test_kernel = AtomicPhysicalKernel(self.dim, kernel_type=next_type).to(DEVICE)
        test_kernel.load_state_dict(candidate.state_dict())
        test_kernel.alpha_epi.data.fill_(0.20)
        opt_sandbox = torch.optim.SGD(test_kernel.parameters(), lr=0.01)
        
        with torch.no_grad():
            base_fe = 0.0
            for t in range(len(recent_bytes) - 1):
                u = self.perceive_byte(recent_bytes[t])
                target = recent_bytes[t + 1]
                psi, _, _ = self.forward_autopoietic_step(u)
                fe = self.compute_free_energy(psi, target)
                base_fe += fe.item()
                
        # Adapt candidate
        mut_fe = 0.0
        for t in range(len(recent_bytes) - 1):
            u = self.perceive_byte(recent_bytes[t])
            target = recent_bytes[t + 1]
            psi, _, _ = self.forward_autopoietic_step(u)
            # Evaluate addition of candidate
            cand_out = test_kernel(psi)
            psi_cand = F.layer_norm(psi + 0.2 * cand_out, (self.dim,))
            fe = self.compute_free_energy(psi_cand, target)
            
            opt_sandbox.zero_grad()
            fe.backward()
            opt_sandbox.step()
            mut_fe += fe.item()
            
        # Gödel acceptance criterion: proven reduction of surprise
        if mut_fe < base_fe * 0.995:
            candidate.load_state_dict(test_kernel.state_dict())
            self.kernels.append(candidate)
            self.sprout_events += 1
            logger.info(f"✨ [GÖDEL ACCEPTED] Sprouted Endogenous Physical Kernel #{len(self.kernels)} (type={next_type})!")
        else:
            self.godel_rejections += 1


def run_exp395():
    logger.info("===============================================================================")
    logger.info("=== KEP EXP-395: ENDOGENOUS STRUCTURAL GENESIS SUBSTRATE (PESGS) =============")
    logger.info("=== STRICT KEP PRINCIPLE 27 EMBODIMENT | ZERO HARDCODED CLASSES | N=1 STREAM ==")
    logger.info("===============================================================================")

    config = EXP395Config()
    model = EndogenousAutopoieticSubstrate(config).to(DEVICE)
    
    # Real-world temporal text stream (11 216 raw bytes)
    text_corpus = (
        "User: Explain the physical difference between static and SDE recurrent integration.\n"
        "Karyon: Static next-token projection is algebraic. SDE recurrent integration treats internal states "
        "as a dynamic vector field traversing a non-Euclidean phase manifold.\n\n"
        "Coding Implementation:\n"
        "template <typename T> torch::Tensor parallel_ssd_scan(torch::Tensor q, torch::Tensor k) {\n"
        "    auto out = torch::matmul(q, k.transpose(-1, -2));\n"
        "    return torch::group_norm(out, 8);\n"
        "}\n\n"
        "Scientific Reasoning: Under Free Energy minimization, epistemic entropy drives curiosity. "
        "When unexpected sensory surprise enters the cognitive gateway, noradrenaline surges, "
        "triggering epigenetic sprouting of new laminar cortical sheets.\n"
    ) * 16

    raw_bytes = [ord(c) for c in text_corpus]
    total_tokens = len(raw_bytes)
    logger.info(f"Loaded Reality Stream: {total_tokens} Bytes | Strict Single-Pass N=1 (NO EPOCHS)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    model.reset_states()

    fe_history = []
    bit_err_history = []
    exact_acc_history = []
    
    start_time = time.time()
    surprise_counter = 0

    # STRICT SINGLE-PASS REALITY TIME STREAM t = 0..T-1
    for t in range(total_tokens - 1):
        byte_in = raw_bytes[t]
        byte_target = raw_bytes[t + 1]
        
        # 1. Perception
        u_t = model.perceive_byte(byte_in)
        
        # 2. Endogenous Cognitive Thinking Cycle & Attractor Snapping
        optimizer.zero_grad()
        psi_t, action_byte, diag = model.forward_autopoietic_step(u_t)
        
        # 3. Hopfield Free Energy
        fe_loss = model.compute_free_energy(psi_t, byte_target)
        fe_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        fe_val = fe_loss.item()
        fe_history.append(fe_val)
        
        # Physical metric telemetry
        xor_diff = action_byte ^ byte_target
        bit_diff = bin(xor_diff).count('1')
        bit_err_history.append(bit_diff)
        exact_acc_history.append(1 if bit_diff == 0 else 0)
        
        # 4. Endogenous Homeostasis Modulation
        model.update_drives(fe_val)
        
        # 5. Endogenous Morphogenesis Triggered by Surprise
        if fe_val > 1.8:
            surprise_counter += 1
            if surprise_counter >= 50:
                context_window = raw_bytes[max(0, t - 32):t + 1]
                model.attempt_kernel_morphogenesis(context_window)
                surprise_counter = 0
        else:
            surprise_counter = max(0, surprise_counter - 1)
            
        if (t + 1) % 500 == 0 or t == total_tokens - 2:
            recent_fe = fe_history[-500:]
            recent_bits = bit_err_history[-500:]
            recent_acc = exact_acc_history[-500:]
            
            avg_fe = sum(recent_fe) / len(recent_fe)
            avg_bits = sum(recent_bits) / len(recent_bits)
            acc = sum(recent_acc) / len(recent_acc) * 100.0
            
            logger.info(
                f"Time Step {t+1:05d}/{total_tokens} | "
                f"Free Energy: {avg_fe:.4f} | "
                f"Bit Error: {avg_bits:.2f}/8.0 ({avg_bits/8.0*100:.1f}%) | "
                f"Exact Match: {acc:.1f}% | "
                f"Kernels: {len(model.kernels)} | "
                f"NA: {model.drive_vector[1].item():.3f} | "
                f"DA: {model.drive_vector[2].item():.3f} | "
                f"Sprouts: {model.sprout_events} | "
                f"Rejections: {model.godel_rejections}"
            )
            
    elapsed = time.time() - start_time
    final_1000_bits = bit_err_history[-1000:]
    final_avg_bits = sum(final_1000_bits) / len(final_1000_bits)
    final_fe = sum(fe_history[-1000:]) / len(fe_history[-1000:])
    final_acc = sum(exact_acc_history[-1000:]) / 1000.0 * 100.0
    
    logger.info("\n=== EXP-395 FINAL SCIENTIFIC TELEMETRY ===")
    logger.info(f"Execution Duration: {elapsed:.2f} s")
    logger.info(f"Final Free Energy: {final_fe:.4f}")
    logger.info(f"Final Bit Error: {final_avg_bits:.2f} / 8.0 bits ({final_avg_bits/8.0*100:.1f}%)")
    logger.info(f"Final Exact Accuracy: {final_acc:.1f}%")
    logger.info(f"Active Endogenous Kernels: {len(model.kernels)}")
    logger.info(f"Sprout Events: {model.sprout_events} (Gödel Rejections: {model.godel_rejections})")
    
    results = {
        "exp_id": "EXP-395",
        "final_free_energy": final_fe,
        "final_bit_errors": final_avg_bits,
        "final_exact_accuracy": final_acc,
        "active_kernels": len(model.kernels),
        "sprout_events": model.sprout_events,
        "godel_rejections": model.godel_rejections,
        "elapsed_time": elapsed
    }
    
    with open("experiments/exp_395_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_exp395()
