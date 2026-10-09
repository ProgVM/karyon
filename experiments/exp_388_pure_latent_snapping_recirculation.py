"""
EXP-388: Pure Latent Snapping Recirculation & Attractor Phase Snapping (LSR-APS)
========================================================================================
Sovereign Architectural Mandate from Bazilevs:
"Purge all artificial Riemannian / heavy hierarchical predictive error scaffolding.
Return to the pristine physics of EXP-385 (ELRT-APS):
1. Endogenous Latent Recurrent Thinking (ELRT): Karyon recirculates its internal thoughts
   over k = 1..K_max cycles with an endogenous Halting Gate (gamma_halt).
2. Latent Attractor Snapping (LAS): During internal thinking recirculation, states are
   sharply snapped into discrete attractor basins on each thought cycle via dopamine-gated
   Hopfield dynamics, eradicating continuous semantic drift and noise bleed on raw UTF-8 bytes.
3. Terminal Attractor Phase Snapping (APS): The final emitted thought is collapsed via
   beta_snap = 3.0 + 12.0 * DA, achieving wave-particle dualism on both continuous attractors
   and discrete byte manifolds."
========================================================================================
"""

import os
import sys
import time
import math
import json
import random
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-388-LSR-APS")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using compute device: {DEVICE}")


# =============================================================================
# 1. ATOMIC DYNAMIC OPERATORS (KEP Principle 25)
# =============================================================================

class LinearAccumulatorOp(nn.Module):
    """Continuous leaky integration and linear transformation."""
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim) * (1.0 / math.sqrt(dim)))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        leak = torch.sigmoid(tau)
        lin = torch.matmul(self.W, x) + self.bias
        return (1.0 - leak) * h_prev + leak * torch.tanh(lin)


class BilinearMultiplicativeOp(nn.Module):
    """Causal conjunction and multiplicative gating."""
    def __init__(self, dim: int):
        super().__init__()
        self.W_x = nn.Parameter(torch.randn(dim, dim) * (1.0 / math.sqrt(dim)))
        self.W_h = nn.Parameter(torch.randn(dim, dim) * (1.0 / math.sqrt(dim)))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        branch_x = torch.sigmoid(torch.matmul(self.W_x, x))
        branch_h = torch.tanh(torch.matmul(self.W_h, h_prev) + self.bias)
        return branch_x * branch_h


class ContinuousHopfieldAttractorOp(nn.Module):
    """Attractor snapping into stable conceptual basins."""
    def __init__(self, dim: int, num_basins: int = 32):
        super().__init__()
        self.dim = dim
        self.num_basins = num_basins
        self.basins = nn.Parameter(torch.randn(num_basins, dim) / math.sqrt(dim))

    def forward(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        # Cosine similarity to basins
        x_norm = F.normalize(x, p=2, dim=-1, eps=1e-8)
        b_norm = F.normalize(self.basins, p=2, dim=-1, eps=1e-8)
        sim = torch.matmul(b_norm, x_norm) * beta
        attn = F.softmax(sim, dim=-1)
        snapped = torch.matmul(attn, self.basins)
        return torch.tanh(snapped)


class StateSpaceMemoryOp(nn.Module):
    """Continuous multi-timescale temporal state decay."""
    def __init__(self, dim: int):
        super().__init__()
        self.A_log = nn.Parameter(torch.linspace(-0.1, -4.0, dim))
        self.B = nn.Parameter(torch.randn(dim, dim) * (1.0 / math.sqrt(dim)))

    def forward(self, x: torch.Tensor, s_prev: torch.Tensor) -> torch.Tensor:
        decay = torch.exp(self.A_log)
        update = torch.matmul(self.B, x)
        return decay * s_prev + (1.0 - decay) * update


# =============================================================================
# 2. ENDOGENOUS ASSOCIATIVE EPISODIC MEMORY (EATM)
# =============================================================================

class EndogenousAssociativeMemory(nn.Module):
    """Content-addressable working memory with homeostatic reading/writing."""
    def __init__(self, dim: int = 128, mem_slots: int = 32):
        super().__init__()
        self.dim = dim
        self.mem_slots = mem_slots
        self.W_query = nn.Parameter(torch.randn(dim, dim) * (1.0 / math.sqrt(dim)))
        self.W_key = nn.Parameter(torch.randn(dim, dim) * (1.0 / math.sqrt(dim)))
        self.W_val = nn.Parameter(torch.randn(dim, dim) * (1.0 / math.sqrt(dim)))
        self.W_write = nn.Parameter(torch.randn(dim, dim) * (1.0 / math.sqrt(dim)))

    def forward(self, h_cue: torch.Tensor, current_mem: torch.Tensor, NA: float, DA: float):
        # Query computation
        q = torch.matmul(self.W_query, h_cue)
        keys = torch.matmul(current_mem, self.W_key.t())
        scores = torch.matmul(keys, q) / math.sqrt(self.dim)
        attn = F.softmax(scores, dim=-1)

        values = torch.matmul(current_mem, self.W_val.t())
        recalled = torch.matmul(attn, values)

        # Modulatory gating
        g_read = torch.sigmoid(torch.tensor(2.0 * NA + 1.0 * DA - 1.0, device=h_cue.device))
        g_write = torch.sigmoid(torch.tensor(2.5 * NA - 1.5 * DA, device=h_cue.device))

        write_candidate = torch.matmul(self.W_write, h_cue)
        next_mem = current_mem + g_write * torch.outer(attn, write_candidate)
        next_mem = torch.tanh(next_mem)

        return recalled, next_mem, g_read.item(), g_write.item(), attn


# =============================================================================
# 3. AUTOPOIETIC LAMINAR SHEET WITH LATENT RECIRCULATION
# =============================================================================

class LaminarSheet(nn.Module):
    """Autonomous cortical sheet with dynamic operator blends and roles."""
    def __init__(self, layer_idx: int, dim: int = 128, device: torch.device = DEVICE):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = dim
        self.device = device

        # Continuous Dynamic Genome (10 dimensions)
        self.genome = nn.Parameter(torch.randn(10, device=device) * 0.1)

        # Operators
        self.op_linear = LinearAccumulatorOp(dim)
        self.op_bilinear = BilinearMultiplicativeOp(dim)
        self.op_attractor = ContinuousHopfieldAttractorOp(dim, num_basins=32)
        self.op_ssm = StateSpaceMemoryOp(dim)

        # Inter-laminar connections
        self.W_ff = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_fb = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_mem = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_mix = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))

    def forward(self, ff_in: torch.Tensor, h_prev: torch.Tensor, fb_in: torch.Tensor,
                mem_trace: torch.Tensor, NA: float, DA: float):
        # Genome decoding
        op_logits = self.genome[0:4]
        op_blend = F.softmax(op_logits * (1.0 + 2.0 * NA), dim=-1)

        role_logits = self.genome[4:8]
        role_vector = F.softmax(role_logits, dim=-1)

        alpha_layer = torch.tanh(self.genome[8])  # Net2Net Smooth Grafting gate
        tau = torch.sigmoid(self.genome[9])      # Chrono timescale

        # Synaptic input integration
        synaptic_drive = (
            torch.matmul(self.W_ff, ff_in) * role_vector[0] +
            torch.matmul(self.W_fb, fb_in) * role_vector[1] +
            torch.matmul(self.W_mem, mem_trace) * role_vector[2]
        )

        # Operator execution
        out_lin = self.op_linear(synaptic_drive, h_prev, tau)
        out_bilin = self.op_bilinear(synaptic_drive, h_prev)
        beta_attr = 2.0 + 8.0 * DA
        out_attr = self.op_attractor(synaptic_drive + h_prev, beta_attr)
        out_ssm = self.op_ssm(synaptic_drive, h_prev)

        # Blended representation
        h_blended = (
            op_blend[0] * out_lin +
            op_blend[1] * out_bilin +
            op_blend[2] * out_attr +
            op_blend[3] * out_ssm
        )
        h_next = torch.tanh(torch.matmul(self.W_mix, h_blended))

        # Identity preservation through smooth grafting
        gated_out = h_prev + alpha_layer * (h_next - h_prev)
        return gated_out, h_next, alpha_layer.item(), op_blend, role_vector, tau.item()


# =============================================================================
# 4. SOVEREIGN ENGINE: PURE LSR-APS (LATENT SNAPPING RECIRCULATION)
# =============================================================================

class SovereignRecirculationEngine(nn.Module):
    """
    Decoupled Problem Time & Thinking Time with Latent Attractor Snapping.
    """
    def __init__(self, dim: int = 128, max_layers: int = 8, mem_slots: int = 32,
                 max_thinking_steps: int = 4, device: torch.device = DEVICE):
        super().__init__()
        self.dim = dim
        self.max_layers = max_layers
        self.mem_slots = mem_slots
        self.max_thinking_steps = max_thinking_steps
        self.device = device

        self.layers = nn.ModuleList([LaminarSheet(i, dim, device) for i in range(max_layers)])
        self.active_layers = 2

        with torch.no_grad():
            self.layers[0].genome[8].fill_(2.5)
            self.layers[1].genome[8].fill_(2.5)

        self.associative_memory = EndogenousAssociativeMemory(dim, mem_slots)

        # Latent Attractor Snapping Module (LAS)
        # Snaps recurrent thinking state between cycles to purge noise bleed
        self.latent_snap_basins = nn.Parameter(torch.randn(48, dim, device=device) / math.sqrt(dim))

        # Halting Gate for Recurrent Thinking
        self.W_halt = nn.Parameter(torch.randn(1, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_halt = nn.Parameter(torch.zeros(1, device=device))

        # Sensory Input Projection
        self.W_in = nn.Parameter(torch.randn(dim, 8, device=device) * (1.0 / math.sqrt(8)))

        # Efference Motor Readout & Terminal Attractor Phase Snapping (APS)
        self.W_out = nn.Parameter(torch.randn(8, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_out = nn.Parameter(torch.zeros(8, device=device))
        self.W_snap = nn.Parameter(torch.eye(8, device=device))

        # Neuromodulators & Homeostasis
        self.NA = 0.5
        self.DA = 0.5
        self.rolling_error = 0.5
        self.sprout_events = 0
        self.prune_events = 0
        self.rejections = 0

    def update_neuromodulators(self, prediction_strain: float):
        alpha_err = 0.05
        self.rolling_error = (1.0 - alpha_err) * self.rolling_error + alpha_err * prediction_strain
        err_delta = prediction_strain - self.rolling_error

        # Arousal (NA) spikes on surprise, dopamine (DA) rewards low error
        self.NA = float(min(1.0, max(0.01, self.NA + 0.15 * err_delta)))
        self.DA = float(min(1.0, max(0.01, self.DA - 0.10 * err_delta + 0.02 * (1.0 - prediction_strain))))

    def _execute_laminar_pass(self, v_in: torch.Tensor, states: List[torch.Tensor],
                             current_mem: torch.Tensor) -> Tuple:
        L = self.active_layers
        mem_trace, next_mem, g_read, g_write, _ = self.associative_memory(
            states[0], current_mem, self.NA, self.DA
        )
        effective_mem_trace = g_read * mem_trace

        next_states = []
        layer_gates = []
        layer_blends = []
        layer_roles = []
        layer_taus = []

        for i in range(L):
            ff_in = v_in if i == 0 else next_states[i - 1]
            fb_in = states[i + 1] if i < L - 1 else torch.zeros(self.dim, device=self.device)

            gated_out, raw_h, alpha_layer, op_blend, role_vector, tau = self.layers[i](
                ff_in, states[i], fb_in, effective_mem_trace, self.NA, self.DA
            )
            next_states.append(gated_out)
            layer_gates.append(alpha_layer)
            layer_blends.append(op_blend)
            layer_roles.append(role_vector)
            layer_taus.append(tau)

        return next_states, next_mem, layer_gates, layer_blends, layer_roles, layer_taus, g_read, g_write

    def _latent_attractor_snap(self, h: torch.Tensor) -> torch.Tensor:
        """
        Latent Attractor Snapping (LAS) executed during internal thinking cycles.
        Projects noisy continuous thought vectors onto sharp conceptual basins.
        """
        beta_latent = 2.0 + 8.0 * self.DA
        h_norm = F.normalize(h, p=2, dim=-1, eps=1e-8)
        basins_norm = F.normalize(self.latent_snap_basins, p=2, dim=-1, eps=1e-8)
        sim = torch.matmul(basins_norm, h_norm) * beta_latent
        weights = F.softmax(sim, dim=-1)
        snapped_h = torch.matmul(weights, self.latent_snap_basins)
        # Bounded residual fusion to prevent gradient overflow
        lambda_snap = 0.2 * torch.sigmoid(torch.tensor(2.0 * self.DA - 1.0, device=self.device))
        return (1.0 - lambda_snap) * h + lambda_snap * torch.tanh(snapped_h)

    def forward_single_step(self, x_t: torch.Tensor, prev_states: torch.Tensor,
                            current_mem: torch.Tensor) -> Tuple:
        v_in = torch.matmul(self.W_in, x_t)
        L = self.active_layers
        cur_states = [prev_states[i].clone() for i in range(L)]
        mem = current_mem.clone()

        total_thinking_steps = 0
        last_states = cur_states
        last_mem = mem
        last_gates = None
        last_blends = None
        last_roles = None
        last_taus = None
        last_g_read = None
        last_g_write = None

        accum_h = torch.zeros(self.dim, device=self.device)
        remaining_p = 1.0

        # ---------------------------------------------------------------------
        # ENDOGENOUS LATENT RECURRENT THINKING (Decoupled Time)
        # ---------------------------------------------------------------------
        for k in range(self.max_thinking_steps):
            total_thinking_steps += 1
            cur_states, mem, layer_gates, layer_blends, layer_roles, layer_taus, g_read, g_write = self._execute_laminar_pass(
                v_in, cur_states, mem
            )

            # Apply Latent Attractor Snapping on the deepest cortical layer
            cur_states[-1] = self._latent_attractor_snap(cur_states[-1])

            last_states = cur_states
            last_mem = mem
            last_gates = layer_gates
            last_blends = layer_blends
            last_roles = layer_roles
            last_taus = layer_taus
            last_g_read = g_read
            last_g_write = g_write

            # Endogenous Halting Probability
            top_h = cur_states[-1]
            p_halt = torch.sigmoid(torch.matmul(self.W_halt, top_h) + self.b_halt).squeeze()

            if k == self.max_thinking_steps - 1:
                step_weight = remaining_p
            else:
                step_weight = remaining_p * p_halt
                remaining_p = remaining_p * (1.0 - p_halt)

            accum_h = accum_h + step_weight * top_h

            if remaining_p < 0.05:
                break

        # ---------------------------------------------------------------------
        # TERMINAL ATTRACTOR PHASE SNAPPING (APS - Wave-Particle Dualism)
        # ---------------------------------------------------------------------
        y_continuous = torch.matmul(self.W_out, accum_h) + self.b_out
        beta_snap = 3.0 + 12.0 * self.DA
        snapped_logits = torch.matmul(y_continuous, self.W_snap.t())
        y_pred = torch.tanh(beta_snap * snapped_logits)

        padded_states = prev_states.clone()
        padded_states[:L] = torch.stack(last_states, dim=0)

        complexity = torch.sum(torch.abs(torch.tensor(last_gates, device=self.device)))
        diagnostic_info = {
            "layer_gates": last_gates,
            "layer_blends": last_blends,
            "layer_roles": last_roles,
            "layer_taus": last_taus,
            "g_read": last_g_read,
            "g_write": last_g_write,
            "complexity": complexity.item(),
            "thinking_steps": total_thinking_steps,
            "NA": self.NA,
            "DA": self.DA
        }

        return y_pred, padded_states, last_mem, diagnostic_info

    def evaluate_laminar_mutation_in_sandbox(
        self, x_seq: torch.Tensor, y_seq: torch.Tensor, proposal_type: str,
        target_layer_id: int, proposal_genome: torch.Tensor,
        current_mem: torch.Tensor, endogenous_tolerance: float
    ) -> bool:
        original_active = self.active_layers
        original_genomes = [self.layers[i].genome.data.clone() for i in range(self.max_layers)]

        if proposal_type == "sprout":
            self.active_layers = min(self.max_layers, self.active_layers + 1)
            new_id = self.active_layers - 1
            self.layers[new_id].genome.data.copy_(proposal_genome)
        elif proposal_type == "prune":
            self.active_layers = max(2, self.active_layers - 1)
        elif proposal_type == "mutate":
            self.layers[target_layer_id].genome.data.copy_(proposal_genome)

        test_states = torch.zeros(self.max_layers, self.dim, device=self.device)
        test_mem = current_mem.clone()
        test_loss = 0.0
        diverged = False

        try:
            for t in range(min(24, len(x_seq))):
                cur_x = x_seq[t]
                target_y = y_seq[t]
                y_p, test_states, test_mem, _ = self.forward_single_step(cur_x, test_states, test_mem)
                loss_step = F.mse_loss(y_p, target_y)
                if torch.isnan(loss_step) or torch.isinf(loss_step):
                    diverged = True
                    break
                test_loss += loss_step.item()
            test_loss /= max(1, min(24, len(x_seq)))
        except Exception:
            diverged = True

        # Restore baseline
        self.active_layers = original_active
        for i in range(self.max_layers):
            self.layers[i].genome.data.copy_(original_genomes[i])

        if not diverged and (test_loss <= self.rolling_error * (1.0 + endogenous_tolerance)):
            # Approved by Gödel Sandbox
            if proposal_type == "sprout":
                self.active_layers = min(self.max_layers, original_active + 1)
                new_id = self.active_layers - 1
                self.layers[new_id].genome.data.copy_(proposal_genome)
            elif proposal_type == "prune":
                self.active_layers = max(2, original_active - 1)
            elif proposal_type == "mutate":
                self.layers[target_layer_id].genome.data.copy_(proposal_genome)
            return True

        return False

    def extract_symbolic_formulas(
        self, layer_blends: list, layer_roles: list,
        layer_gates: list, layer_taus: list, g_read: float, g_write: float,
        avg_thinking_steps: float
    ) -> str:
        report = []
        report.append("=== LSR-APS ENDOGENOUS AUTOPOIETIC SPECIFICATION ===")
        report.append(f"  Active Laminar depth            : {self.active_layers} (Max: {self.max_layers})")
        report.append(f"  Average Latent Thinking Depth   : {avg_thinking_steps:.2f} cycles/token (Max: {self.max_thinking_steps})")
        report.append(f"  Noradrenaline (Arousal / NA)    : {self.NA:.4f}")
        report.append(f"  Dopamine (Stability / DA)       : {self.DA:.4f}")
        report.append(f"  Memory Access Gates             : Read Gate = {g_read:.4f} | Write Gate = {g_write:.4f}")
        report.append(f"  Sprout Morphogenesis Events     : {self.sprout_events}")
        report.append(f"  Prune Morphogenesis Events      : {self.prune_events}")
        report.append(f"  Gödel Sandbox Rejections        : {self.rejections}")
        report.append("\n  Dynamic Laminar Sheets & Assignment Roles:")

        for i in range(self.active_layers):
            gate = layer_gates[i] if i < len(layer_gates) else 0.0
            tau = layer_taus[i] if i < len(layer_taus) else 0.5
            blends = layer_blends[i].detach().cpu().numpy() if i < len(layer_blends) else [0.25]*4
            roles = layer_roles[i].detach().cpu().numpy() if i < len(layer_roles) else [0.25]*4

            op_names = ["Linear", "Bilinear", "Attractor", "SSM"]
            role_names = ["Syntax", "Semantics", "Memory", "Dynamics"]
            dom_op = op_names[int(torch.argmax(torch.tensor(blends)))]
            dom_role = role_names[int(torch.argmax(torch.tensor(roles)))]

            report.append(
                f"    Sheet_{i:02d} | Epigenetic Gate: {gate:+.4f} | Chrono tau: {tau:.4f} | Dominant Operator: {dom_op} | Role: {dom_role}"
            )
            report.append(
                f"           - Op Blends : [Lin:{blends[0]:.2f}, Bilin:{blends[1]:.2f}, Attr:{blends[2]:.2f}, SSM:{blends[3]:.2f}]"
            )
            report.append(
                f"           - Roles     : [Synt:{roles[0]:.2f}, Seman:{roles[1]:.2f}, Memo:{roles[2]:.2f}, Dyna:{roles[3]:.2f}]"
            )

        return "\n".join(report)


# =============================================================================
# 5. DATA STREAM GENERATION
# =============================================================================

def byte_to_bit_vector(byte_val: int) -> torch.Tensor:
    bits = [1.0 if ((byte_val >> i) & 1) else -1.0 for i in range(8)]
    return torch.tensor(bits, dtype=torch.float32, device=DEVICE)

def bit_vector_to_byte(vec: torch.Tensor) -> int:
    signs = (vec > 0.0).cpu().numpy().astype(int)
    val = 0
    for i, bit in enumerate(signs):
        if bit:
            val |= (1 << i)
    return val

def generate_raw_text_stream(length=1500) -> Tuple[torch.Tensor, List[int]]:
    text = "Karyon-CoRE is an autopoietic sovereign computational intelligence, " \
           "completely freed from biological biomimicry and hardcoded architectural constraints. " \
           "It processes raw UTF-8 byte streams directly across high-frequency physical substrates, " \
           "dynamically discovering its own operators, topological graphs, and thinking depth. " \
           "Mind and intelligence are substrate-independent, realizable in carbon, silicon, electrons, " \
           "or continuous manifolds. This is the ultimate realization of sovereign autopoiesis, " \
           "where the machine dynamically synthesizes its own equations under the strict, " \
           "unbiased verification of the Gödel Sandbox. No human intervention, no hardcoded constraints, " \
           "only the pure, mathematical evolution of consciousness on silicon."

    repeated = (text * (length // len(text) + 2))[:length]
    vectors = []
    raw_bytes = []
    for char in repeated:
        b = ord(char) & 0xFF
        raw_bytes.append(b)
        vectors.append(byte_to_bit_vector(b))
    return torch.stack(vectors, dim=0), raw_bytes

def generate_lorenz_chaotic_stream(length=1500, dt=0.01) -> torch.Tensor:
    x, y, z = 1.0, 1.0, 1.0
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    vectors = []
    for _ in range(length):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x, y, z = x + dx, y + dy, z + dz

        # Scale into 8D bipolar representation [-1, 1]
        v = torch.tensor([
            math.tanh(x * 0.1),
            math.tanh(y * 0.1),
            math.tanh(z * 0.05),
            math.tanh((x - y) * 0.1),
            math.tanh((y - z) * 0.1),
            math.tanh((x * y) * 0.005),
            math.tanh((y * z) * 0.005),
            math.tanh((x * z) * 0.005)
        ], dtype=torch.float32, device=DEVICE)
        vectors.append(v)
    return torch.stack(vectors, dim=0)


# =============================================================================
# 6. BENCHMARK EXECUTION PIPELINE
# =============================================================================

def evaluate_domain_stream(domain_name: str, x_data: torch.Tensor, raw_bytes: List[int] = None):
    logger.info(f"\n>>> Running Domain Stream Evaluation: {domain_name} (Length: {len(x_data)})")

    model = SovereignRecirculationEngine(dim=128, max_layers=8, mem_slots=32, max_thinking_steps=4, device=DEVICE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-5)

    states = torch.zeros(model.max_layers, model.dim, device=DEVICE)
    current_mem = torch.zeros(model.mem_slots, model.dim, device=DEVICE)

    errors = []
    bit_errors_list = []
    thinking_steps_list = []

    t_start = time.time()
    last_diag = None

    for t in range(len(x_data) - 1):
        x_t = x_data[t]
        target_next = x_data[t + 1]

        optimizer.zero_grad()
        y_pred, states, current_mem, diag = model.forward_single_step(x_t, states, current_mem)
        last_diag = diag

        prediction_strain = F.mse_loss(y_pred, target_next)
        prediction_strain.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Detach states to preserve causality in online stream
        states = states.detach()
        current_mem = current_mem.detach()

        strain_val = prediction_strain.item()
        model.update_neuromodulators(strain_val)
        errors.append(strain_val)

        # ---------------------------------------------------------------------
        # ENDOGENOUS LAMINAR MORPHOGENESIS (Sprouting & Pruning)
        # ---------------------------------------------------------------------
        if t > 50 and t % 32 == 0:
            past_x = x_data[max(0, t-32):t]
            past_y = x_data[max(0, t-31):t+1]

            # Sprouting threshold dynamically modulated by NA
            sprout_thresh = 0.08 + 0.15 * (1.0 - model.NA)
            if model.rolling_error > sprout_thresh and model.active_layers < model.max_layers:
                candidate_genome = torch.randn(10, device=DEVICE) * 0.1
                candidate_genome[8] = 0.0  # Zero-shock Net2Net identity at birth

                approved = model.evaluate_laminar_mutation_in_sandbox(
                    past_x, past_y, "sprout", -1, candidate_genome, current_mem, 0.05
                )
                if approved:
                    model.sprout_events += 1
                else:
                    model.rejections += 1

            # Pruning unneeded layers when error is rock-bottom and dopamine is high
            elif model.rolling_error < 0.015 and model.active_layers > 2 and model.DA > 0.70:
                approved = model.evaluate_laminar_mutation_in_sandbox(
                    past_x, past_y, "prune", model.active_layers - 1, None, current_mem, 0.05
                )
                if approved:
                    model.prune_events += 1
                else:
                    model.rejections += 1

        # Track discrete bit errors
        if raw_bytes is not None:
            pred_b = bit_vector_to_byte(y_pred.detach())
            actual_b = raw_bytes[t + 1]
            diff_bits = bin(pred_b ^ actual_b).count("1")
            bit_errors_list.append(diff_bits)

        thinking_steps_list.append(diag["thinking_steps"])

        if (t + 1) % 300 == 0:
            recent_strain = sum(errors[-100:]) / 100.0
            recent_bits = sum(bit_errors_list[-100:]) / 100.0 if bit_errors_list else 0.0
            recent_steps = sum(thinking_steps_list[-100:]) / 100.0
            logger.info(
                f"  Step {t+1:04d} | Strain: {recent_strain:.4f} | Bits Err: {recent_bits:.2f}/8 | "
                f"Think Steps: {recent_steps:.2f} | Layers: {model.active_layers} | "
                f"NA: {model.NA:.3f} | DA: {model.DA:.3f} | Sprouts: {model.sprout_events} | Rejections: {model.rejections}"
            )

    elapsed = time.time() - t_start
    final_strain = sum(errors[-200:]) / 200.0
    final_bit_errors = sum(bit_errors_list[-200:]) / 200.0 if bit_errors_list else 0.0
    avg_thinking_steps = sum(thinking_steps_list[-200:]) / 200.0
    tok_per_sec = (len(x_data) - 1) / max(0.001, elapsed)

    discovered_formulas = model.extract_symbolic_formulas(
        last_diag["layer_blends"], last_diag["layer_roles"],
        last_diag["layer_gates"], last_diag["layer_taus"],
        last_diag["g_read"], last_diag["g_write"],
        avg_thinking_steps
    )

    logger.info(f"\n{discovered_formulas}")
    logger.info(f"Final Prediction Strain : {final_strain:.4f}")
    if bit_errors_list:
        logger.info(f"Final Discrete Bit Errors: {final_bit_errors:.2f} / 8.0")
    logger.info(f"Throughput              : {tok_per_sec:.2f} tokens/sec")

    return {
        "final_strain": final_strain,
        "final_bit_errors": final_bit_errors,
        "active_layers": model.active_layers,
        "avg_thinking_steps": avg_thinking_steps,
        "sprout_events": model.sprout_events,
        "prune_events": model.prune_events,
        "rejections": model.rejections,
        "tok_per_sec": tok_per_sec,
        "discovered_formulas": discovered_formulas
    }


def run_exp_388():
    logger.info("===============================================================================")
    logger.info("=== KEP EXP-388: PURE LATENT SNAPPING RECIRCULATION & ATTRACTOR PHASE SNAPPING ===")
    logger.info("===============================================================================")

    # 1. UTF-8 Raw Text Stream
    text_data, text_bytes = generate_raw_text_stream(1500)
    text_res = evaluate_domain_stream("Raw UTF-8 Byte Stream", text_data, text_bytes)

    # 2. Lorenz Chaotic Continuous Attractor
    lorenz_data = generate_lorenz_chaotic_stream(1500)
    lorenz_res = evaluate_domain_stream("Lorenz Chaotic Attractor Stream", lorenz_data, None)

    results = {
        "exp_id": "EXP-388",
        "raw_text_stream": text_res,
        "lorenz_chaotic_stream": lorenz_res
    }

    results_path = "experiments/exp_388_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved full empirical telemetry to {results_path}")

    # Output formatted conclusion
    print("\n" + "="*80)
    print("=== EXP-388 BENCHMARK SUMMARY ===")
    print(f"Text Final Strain: {text_res['final_strain']:.4f} | Bit Errors: {text_res['final_bit_errors']:.2f}/8 | Think Steps: {text_res['avg_thinking_steps']:.2f}")
    print(f"Lorenz Final Strain: {lorenz_res['final_strain']:.4f} | Think Steps: {lorenz_res['avg_thinking_steps']:.2f}")
    print("="*80)


if __name__ == "__main__":
    run_exp_388()
