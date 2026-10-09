"""
EXP-385: Endogenous Latent Recurrent Thinking & Attractor Phase Snapping (ELRT-APS)
========================================================================================
Sovereign Architectural Mandate from Bazilevs & Gema:
"всем чем-либо должен управлять сам Карион, а не хардкодоп снаружи"
"Даём. Я поражаюсь, что ты не сделал это ранее."

To shatter the final barrier preventing prediction error from plunging toward 0.000:
1. Decoupling Problem Time from Thinking Time (KEP Principle 21 & Principle 22.3):
   - Rather than forcing a single feed-forward pass per input token (T=1), Karyon possesses
     endogenous authority over internal latent thinking cycles (k = 1..K_max).
   - Recirculates internal states through its laminar sheets until its endogenous Halting
     Gate (gamma_halt) decides that internal Free Energy / Prediction Entropy has reached
     equilibrium, or internal cognitive budget is satisfied.
2. Wave-Particle Dualism & Attractor Phase Snapping (KEP Principle 22.4):
   - Continuous realm: Smooth state dynamics, continuous Hopfield memory reading/writing,
     and multi-timescale laminar integration.
   - Discrete realm: Motor emission undergoes sharp Attractor Phase Snapping (APS).
     Output logits/bits are snapped into sharp attractor basins via dynamic precision gain:
     y_snapped = tanh(beta_snap * y_continuous)
     where beta_snap is endogenously modulated by Dopamine (stability) and Noradrenaline (certainty):
     beta_snap = 4.0 + 16.0 * DA
3. Endogenous Associative Trace Memory (EATM):
   - Content-addressable continuous Hopfield memory (K=32 slots) remains tightly bound,
     queried and updated endogenously across thinking iterations.
4. Full Sovereign Laminar Autopoiesis:
   - Sprouting, pruning, and Gödel Machine Sandbox evaluation remain 100% endogenous.
========================================================================================
"""

import os
import time
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. ATOMIC PRIMITIVE OPERATORS (Principle 25)
# =============================================================================

class LinearAccumulatorOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.bias = nn.Parameter(torch.zeros(dim, device=device))

    def forward(self, x: torch.Tensor, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        act = torch.tanh(torch.matmul(x, self.W.t()) + self.bias)
        return (1.0 - tau) * h + tau * act


class BilinearMultiplicativeOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W2 = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))

    def forward(self, x: torch.Tensor, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        g1 = torch.tanh(torch.matmul(x, self.W1.t()))
        g2 = torch.sigmoid(torch.matmul(h, self.W2.t()))
        return (1.0 - tau) * h + tau * (g1 * g2)


class SaturatedAttractorOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.beta = nn.Parameter(torch.tensor([12.0], device=device))

    def forward(self, x: torch.Tensor, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        proj = torch.matmul(h, self.W.t())
        snapped = torch.tanh(self.beta * proj)
        return (1.0 - tau) * h + tau * torch.tanh(snapped + x)


class StateSpaceMemoryOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.log_alpha = nn.Parameter(torch.linspace(-3.0, -0.5, dim, device=device))

    def forward(self, x: torch.Tensor, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.log_alpha)
        decay = torch.clamp(1.0 - alpha * tau, 0.0, 1.0)
        proj = torch.tanh(torch.matmul(x, self.W.t()))
        return decay * h + (1.0 - decay) * proj


# =============================================================================
# 2. ENDOGENOUS ASSOCIATIVE TRACE MEMORY (EATM)
# =============================================================================

class EndogenousAssociativeMemory(nn.Module):
    def __init__(self, slots: int = 32, dim: int = 128, device: torch.device = DEVICE):
        super().__init__()
        self.slots = slots
        self.dim = dim
        self.device = device

        init_mem = torch.randn(slots, dim, device=device)
        init_mem = init_mem / (torch.norm(init_mem, dim=-1, keepdim=True) + 1e-6)
        self.memory_slots = nn.Parameter(init_mem)

        self.W_q = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_v = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_gate = nn.Parameter(torch.randn(2, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_gate = nn.Parameter(torch.zeros(2, device=device))

    def forward(self, state_in: torch.Tensor, current_mem: torch.Tensor, NA: float, DA: float):
        query = torch.matmul(state_in, self.W_q.t())
        query_norm = query / (torch.norm(query, dim=-1, keepdim=True) + 1e-6)
        mem_norm = current_mem / (torch.norm(current_mem, dim=-1, keepdim=True) + 1e-6)

        beta_mem = 4.0 * (1.0 + 2.0 * NA)
        sim = torch.matmul(mem_norm, query_norm) / math.sqrt(self.dim)
        attn = torch.softmax(beta_mem * sim, dim=-1)

        retrieved_trace = torch.matmul(attn, current_mem)

        gates = torch.sigmoid(torch.matmul(self.W_gate, state_in) + self.b_gate)
        g_read = gates[0]
        g_write = gates[1] * (0.2 + 0.8 * (1.0 - DA))

        value = torch.tanh(torch.matmul(state_in, self.W_v.t()))
        write_weights = attn.unsqueeze(-1)
        updated_mem = current_mem + g_write * write_weights * (value - current_mem)

        return retrieved_trace, updated_mem, g_read, g_write, attn


# =============================================================================
# 3. DYNAMIC LAMINAR SHEET WITH MEMORY RECEPTORS
# =============================================================================

class LaminarSheet(nn.Module):
    def __init__(self, layer_id: int, dim: int = 128, device: torch.device = DEVICE):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.device = device

        self.genome = nn.Parameter(torch.randn(10, device=device) * 0.1)

        self.op_linear = LinearAccumulatorOp(dim, device)
        self.op_bilinear = BilinearMultiplicativeOp(dim, device)
        self.op_attractor = SaturatedAttractorOp(dim, device)
        self.op_ssm = StateSpaceMemoryOp(dim, device)

        self.W_ff = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_fb = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_mem = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.bias = nn.Parameter(torch.zeros(dim, device=device))

    def forward(self, x_in: torch.Tensor, h_prev: torch.Tensor, feedback_in: torch.Tensor, mem_trace: torch.Tensor, NA: float, DA: float) -> tuple:
        op_blend = torch.softmax(self.genome[0:4], dim=-1)
        alpha_layer = torch.tanh(self.genome[4])

        log_tau = self.genome[5]
        tau_base = torch.sigmoid(log_tau)
        tau = torch.clamp(tau_base * (1.0 + 1.2 * NA - 0.4 * DA), 0.01, 0.99)

        role_vector = torch.softmax(self.genome[6:10], dim=-1)

        mixed_input = (
            torch.matmul(x_in, self.W_ff.t())
            + torch.matmul(feedback_in, self.W_fb.t())
            + torch.matmul(mem_trace, self.W_mem.t())
            + self.bias
        )

        h0 = self.op_linear(mixed_input, h_prev, tau)
        h1 = self.op_bilinear(mixed_input, h_prev, tau)
        h2 = self.op_attractor(mixed_input, h_prev, tau)
        h3 = self.op_ssm(mixed_input, h_prev, tau)

        h_next = op_blend[0] * h0 + op_blend[1] * h1 + op_blend[2] * h2 + op_blend[3] * h3
        gated_output = alpha_layer * h_next

        return gated_output, h_next, alpha_layer, op_blend, role_vector, tau


# =============================================================================
# 4. SOVEREIGN ENGINE WITH ENDOGENOUS LATENT THINKING & PHASE SNAPPING
# =============================================================================

class SovereignThinkingEngine(nn.Module):
    def __init__(self, dim: int = 128, max_layers: int = 8, mem_slots: int = 32, max_thinking_steps: int = 4, device: torch.device = DEVICE):
        super().__init__()
        self.dim = dim
        self.max_layers = max_layers
        self.mem_slots = mem_slots
        self.max_thinking_steps = max_thinking_steps
        self.device = device

        self.layers = nn.ModuleList([LaminarSheet(i, dim, device) for i in range(max_layers)])
        self.active_layers = 2

        with torch.no_grad():
            self.layers[0].genome[4].fill_(2.5)
            self.layers[1].genome[4].fill_(2.5)
            self.layers[0].genome[6].fill_(2.0)
            self.layers[1].genome[7].fill_(2.0)

        # Endogenous Memory Manifold
        self.associative_memory = EndogenousAssociativeMemory(slots=mem_slots, dim=dim, device=device)

        self.W_in = nn.Parameter(torch.randn(dim, 8, device=device) * (1.0 / math.sqrt(8)))
        self.W_out = nn.Parameter(torch.randn(8, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_out = nn.Parameter(torch.zeros(8, device=device))

        # Endogenous Halting Gate & Recurrent Thinking Readout
        self.W_halt = nn.Parameter(torch.randn(1, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_halt = nn.Parameter(torch.tensor([-0.5], device=device))  # Biased to allow initial thinking

        # Endogenous Attractor Snapping Kernel (Hopfield Phase Transition)
        self.W_snap = nn.Parameter(torch.randn(8, 8, device=device) * (1.0 / math.sqrt(8)))

        # Ashby Neuromodulators
        self.NA = 0.5
        self.DA = 0.5
        self.rolling_error = 1.0

        # Endogenous Meta-Law Generator
        self.W_meta = nn.Parameter(torch.randn(4, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_meta = nn.Parameter(torch.zeros(4, device=device))

        self.sprout_events = 0
        self.prune_events = 0
        self.rejections = 0

    def update_neuromodulators(self, prediction_error: float):
        self.NA = 0.90 * self.NA + 0.10 * math.tanh(prediction_error * 2.0)
        error_delta = self.rolling_error - prediction_error
        self.DA = 0.90 * self.DA + 0.10 * torch.sigmoid(torch.tensor([error_delta * 5.0])).item()
        self.rolling_error = 0.95 * self.rolling_error + 0.05 * prediction_error

    def _execute_laminar_pass(self, v_in: torch.Tensor, states: torch.Tensor, current_mem: torch.Tensor):
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

        next_states = torch.stack(next_states, dim=0)
        layer_gates = torch.stack(layer_gates, dim=0)
        layer_blends = torch.stack(layer_blends, dim=0)
        layer_roles = torch.stack(layer_roles, dim=0)

        return next_states, next_mem, layer_gates, layer_blends, layer_roles, layer_taus, g_read, g_write

    def forward(self, x_t: torch.Tensor, prev_states: torch.Tensor, current_mem: torch.Tensor) -> tuple:
        """
        Decoupled Problem Time vs Thinking Time (Recurrent Latent Depth).
        Recirculates representations endogenously across k = 1..max_thinking_steps.
        """
        v_in = torch.matmul(x_t, self.W_in.t())
        L = self.active_layers
        cur_states = prev_states[:L]
        mem = current_mem

        # Recurrent Latent Thinking Loop
        total_thinking_steps = 0
        halting_weights = []
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

        for k in range(self.max_thinking_steps):
            total_thinking_steps += 1
            cur_states, mem, layer_gates, layer_blends, layer_roles, layer_taus, g_read, g_write = self._execute_laminar_pass(
                v_in, cur_states, mem
            )
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

            halting_weights.append(step_weight)
            accum_h = accum_h + step_weight * top_h

            # If confidence is exceptionally high, stop early
            if remaining_p < 0.05:
                break

        # ---------------------------------------------------------------------
        # ATTRACTOR PHASE SNAPPING (APS - Principle 22.4 Wave-Particle Dualism)
        # ---------------------------------------------------------------------
        y_continuous = torch.matmul(self.W_out, accum_h) + self.b_out

        # Endogenous Snapping Precision modulated by Dopamine stability
        beta_snap = 3.0 + 12.0 * self.DA
        snapped_logits = torch.matmul(y_continuous, self.W_snap.t())
        y_pred = torch.tanh(beta_snap * snapped_logits)

        padded_states = prev_states.clone()
        padded_states[:L] = last_states

        complexity = torch.sum(torch.abs(last_gates))
        meta_vector = torch.sigmoid(torch.matmul(self.W_meta, accum_h) + self.b_meta)

        return (
            y_pred, padded_states, last_mem, complexity, last_blends,
            last_roles, last_gates, meta_vector, last_taus, last_g_read, last_g_write,
            total_thinking_steps
        )

    # -------------------------------------------------------------------------
    # GÖDEL MACHINE SANDBOX WITH ENDOGENOUS TOLERANCE
    # -------------------------------------------------------------------------
    def evaluate_laminar_mutation_in_sandbox(
        self, x_seq: torch.Tensor, target_seq: torch.Tensor, proposal_type: str,
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
                tgt_x = target_seq[t]
                (
                    y_pred, next_states, next_mem_step, _, _, _, _, _, _, _, _, _
                ) = self(cur_x, test_states, test_mem)
                test_states = next_states.detach()
                test_mem = next_mem_step.detach()

                if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                    diverged = True
                    break
                test_loss += 0.5 * torch.sum((y_pred - tgt_x)**2).item()
        except Exception:
            diverged = True

        self.active_layers = original_active
        for i in range(self.max_layers):
            self.layers[i].genome.data.copy_(original_genomes[i])

        if diverged:
            return False

        baseline_states = torch.zeros(self.max_layers, self.dim, device=self.device)
        baseline_mem = current_mem.clone()
        baseline_loss = 0.0
        try:
            for t in range(min(24, len(x_seq))):
                cur_x = x_seq[t]
                tgt_x = target_seq[t]
                (
                    y_pred, next_states, next_mem_step, _, _, _, _, _, _, _, _, _
                ) = self(cur_x, baseline_states, baseline_mem)
                baseline_states = next_states.detach()
                baseline_mem = next_mem_step.detach()
                baseline_loss += 0.5 * torch.sum((y_pred - tgt_x)**2).item()
        except Exception:
            return False

        max_acceptable_loss = baseline_loss * (1.0 + 0.20 * endogenous_tolerance)
        if test_loss <= max_acceptable_loss:
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
        self, layer_blends: torch.Tensor, layer_roles: torch.Tensor,
        layer_gates: torch.Tensor, layer_taus: list, g_read: float, g_write: float,
        avg_thinking_steps: float
    ) -> str:
        report = []
        report.append("=== ELRT-APS ENDOGENOUS AUTOPOIETIC SPECIFICATION ===")
        report.append(f"  Active Laminar depth            : {self.active_layers} (Max: {self.max_layers})")
        report.append(f"  Average Latent Thinking Depth   : {avg_thinking_steps:.2f} cycles/token (Max: {self.max_thinking_steps})")
        report.append(f"  Noradrenaline (Arousal / NA)    : {self.NA:.4f}")
        report.append(f"  Dopamine (Stability / DA)       : {self.DA:.4f}")
        report.append(f"  Memory Access Gates             : Read Gate = {g_read:.4f} | Write Gate = {g_write:.4f}")
        report.append(f"  Sprout Morphogenesis Events     : {self.sprout_events}")
        report.append(f"  Prune Morphogenesis Events      : {self.prune_events}")
        report.append(f"  Gödel Sandbox Rejections        : {self.rejections}")

        role_names = ["Syntax", "Semantics", "Memory", "Dynamics"]
        op_names = ["Linear", "Bilinear", "Attractor", "SSM"]

        report.append("\n  Dynamic Laminar Sheets & Assignment Roles:")
        for i in range(self.active_layers):
            gate = layer_gates[i].item()
            blends = layer_blends[i].tolist()
            roles = layer_roles[i].tolist()
            tau_val = layer_taus[i].item() if isinstance(layer_taus[i], torch.Tensor) else layer_taus[i]

            dom_op = op_names[np.argmax(blends)]
            dom_role = role_names[np.argmax(roles)]

            report.append(f"    Sheet_{i:02d} | Epigenetic Gate: {gate:.4f} | Chrono tau: {tau_val:.4f} | Dominant Operator: {dom_op} | Role: {dom_role}")
            report.append(f"           - Op Blends : [Lin:{blends[0]:.2f}, Bilin:{blends[1]:.2f}, Attr:{blends[2]:.2f}, SSM:{blends[3]:.2f}]")
            report.append(f"           - Roles     : [Synt:{roles[0]:.2f}, Seman:{roles[1]:.2f}, Memo:{roles[2]:.2f}, Dyna:{roles[3]:.2f}]")
        return "\n".join(report)


# =============================================================================
# 5. DATA GENERATION & BENCHMARK SUITE
# =============================================================================

def byte_to_bit_vector(b: int) -> torch.Tensor:
    bits = [1.0 if ((b >> i) & 1) else -1.0 for i in range(8)]
    return torch.tensor(bits, dtype=torch.float32, device=DEVICE)


def bit_vector_to_byte(vec: torch.Tensor) -> int:
    signs = (vec > 0.0).cpu().numpy().astype(int)
    val = 0
    for i, bit in enumerate(signs):
        if bit:
            val |= (1 << i)
    return val


def generate_raw_text_stream(length=1000):
    text = "The physical universe is a non-linear continuous field of mathematical operators. " \
           "Karyon-CoRE operates directly at the raw UTF-8 byte level, bypassing artificial tokenizers. " \
           "By minimizing variational free energy, the system self-organizes its internal attractors. " \
           "Sovereign Gödelian Autopoiesis guarantees zero-shock structural morphogenesis. " \
           "The mind is substrate-independent, realizing itself in silicon, carbon, or photons. " \
           "We reject the static neuromorphic assumptions of terrestrial biology. " \
           "To solve complex multi-step reasoning, we must scale the compositional depth of our " \
           "laminar neural sheets. Information is handled as raw byte streams, spatial tensors, " \
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


def generate_lorenz_chaotic_stream(length=1000, dt=0.01):
    x, y, z = 1.0, 1.0, 1.0
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    vectors = []
    for _ in range(length):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x += dx
        y += dy
        z += dz
        v = [
            math.tanh(x * 0.05),
            math.tanh(y * 0.05),
            math.tanh(z * 0.03),
            math.tanh((x - y) * 0.05),
            math.tanh((y - z) * 0.05),
            math.tanh((x * y) * 0.001),
            math.tanh((y * z) * 0.001),
            math.tanh((x * z) * 0.001)
        ]
        vectors.append(v)
    return torch.tensor(vectors, dtype=torch.float32, device=DEVICE)


def run_evaluation(domain_name, x_data, raw_bytes=None, is_text=False):
    print(f"\nEvaluating ELRT-APS on Domain: {domain_name}...")
    length = x_data.size(0)

    model = SovereignThinkingEngine(dim=128, max_layers=8, mem_slots=32, max_thinking_steps=4, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2, weight_decay=1e-5)

    states = torch.zeros(model.max_layers, model.dim, device=DEVICE)
    current_mem = model.associative_memory.memory_slots.clone()

    errors = []
    bit_mismatches = []
    thinking_step_history = []
    t0 = time.time()

    last_blends = None
    last_roles = None
    last_gates = None
    last_taus = None
    last_g_read = 0.0
    last_g_write = 0.0

    for t in range(length - 1):
        cur_vec = x_data[t]
        tgt_vec = x_data[t + 1]

        optimizer.zero_grad()
        (
            y_pred, next_states, next_mem, complexity, layer_blends,
            layer_roles, layer_gates, meta_vector, layer_taus, g_read, g_write,
            thinking_steps
        ) = model(cur_vec, states, current_mem)

        states = next_states.detach()
        current_mem = next_mem.detach()

        last_blends = layer_blends.detach()
        last_roles = layer_roles.detach()
        last_gates = layer_gates.detach()
        last_taus = layer_taus
        last_g_read = g_read.item()
        last_g_write = g_write.item()
        thinking_step_history.append(thinking_steps)

        # Endogenous parameter synthesis via meta_vector
        sprout_thresh = 0.30 + 0.40 * meta_vector[0].item()
        prune_thresh = 0.05 + 0.15 * meta_vector[1].item()
        godel_tolerance = meta_vector[2].item()
        mutation_volatility = 0.01 + 0.20 * meta_vector[3].item()

        # Endogenous Complexity Penalty modulated by Dopamine
        beta_complexity = 0.05 * (1.0 - model.DA)

        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        free_energy = prediction_strain + beta_complexity * complexity + 0.02 * model.active_layers

        free_energy.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        model.update_neuromodulators(strain_val)
        errors.append(strain_val)

        # ---------------------------------------------------------------------
        # ENDOGENOUS LAMINAR MORPHOGENESIS (Sprouting & Pruning)
        # ---------------------------------------------------------------------
        if t > 50 and t % 32 == 0:
            past_x = x_data[max(0, t-32):t]
            past_y = x_data[max(0, t-31):t+1]

            # Sprouting proposal based on endogenous threshold
            if model.rolling_error > sprout_thresh and model.active_layers < model.max_layers:
                candidate_genome = torch.randn(10, device=DEVICE) * 0.1
                candidate_genome[4] = 0.0  # Net2Net Smooth Grafting

                if model.evaluate_laminar_mutation_in_sandbox(
                    past_x, past_y, "sprout", -1, candidate_genome, current_mem, godel_tolerance
                ):
                    model.sprout_events += 1
                    print(f"  [ELRT LAMINAR SPROUT] Added Sheet {model.active_layers} at step {t+1}. Total Layers: {model.active_layers} | Tolerance: {godel_tolerance:.4f}")
                else:
                    model.rejections += 1

            # Pruning proposal based on endogenous threshold
            elif model.rolling_error < prune_thresh and model.active_layers > 2:
                top_gate = torch.tanh(model.layers[model.active_layers - 1].genome[4]).item()
                if abs(top_gate) < 0.15:
                    if model.evaluate_laminar_mutation_in_sandbox(
                        past_x, past_y, "prune", -1, None, current_mem, godel_tolerance
                    ):
                        model.prune_events += 1
                        print(f"  [ELRT LAMINAR PRUNE] Pruned Sheet {model.active_layers} at step {t+1}. Total Layers: {model.active_layers} | Tolerance: {godel_tolerance:.4f}")
                    else:
                        model.rejections += 1

            # Layer Genome Mutation based on endogenous volatility
            else:
                target_layer_id = random.randint(0, model.active_layers - 1)
                mutation_delta = torch.randn(10, device=DEVICE) * mutation_volatility
                proposed_genome = model.layers[target_layer_id].genome.data + mutation_delta

                if model.evaluate_laminar_mutation_in_sandbox(
                    past_x, past_y, "mutate", target_layer_id, proposed_genome, current_mem, godel_tolerance
                ):
                    pass
                else:
                    model.rejections += 1

        if is_text and raw_bytes is not None:
            pred_byte = bit_vector_to_byte(y_pred)
            tgt_byte = raw_bytes[t + 1]
            bit_diff = bin(pred_byte ^ tgt_byte).count('1')
            bit_mismatches.append(bit_diff)

        if (t + 1) % 256 == 0:
            tail_err = np.mean(errors[-64:])
            tail_bits = np.mean(bit_mismatches[-64:]) if is_text else 0.0
            tail_think = np.mean(thinking_step_history[-64:])
            print(
                f"  Step {t+1:04d}/{length} | Layers: {model.active_layers:02d} | Think: {tail_think:.1f}x | NA: {model.NA:.3f} | DA: {model.DA:.3f} "
                f"| g_read: {last_g_read:.2f} | g_write: {last_g_write:.2f} | Strain: {tail_err:.4f}"
                + (f" | Bit Errors: {tail_bits:.2f}/8" if is_text else "")
            )

    elapsed = time.time() - t0
    tok_per_sec = length / elapsed if elapsed > 0 else 0.0

    q_len = max(1, len(errors) // 4)
    final_strain = float(np.mean(errors[-q_len:]))
    final_bit_errors = float(np.mean(bit_mismatches[-q_len:])) if len(bit_mismatches) > 0 else 0.0
    avg_thinking_steps = float(np.mean(thinking_step_history))

    discovered_formulas = model.extract_symbolic_formulas(
        last_blends, last_roles, last_gates, last_taus, last_g_read, last_g_write, avg_thinking_steps
    )
    print("\n-------------------------------------------------------------------------------")
    print(discovered_formulas)
    print("-------------------------------------------------------------------------------\n")

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


def run_exp_385():
    print("===============================================================================")
    print("=== KEP EXP-385: ENDOGENOUS LATENT RECURRENT THINKING & PHASE SNAPPING      ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    # Generate streams
    text_data, text_bytes = generate_raw_text_stream(1500)
    lorenz_data = generate_lorenz_chaotic_stream(1500)

    # Run evaluations
    r1 = run_evaluation("Raw UTF-8 Text Stream", text_data, text_bytes, is_text=True)
    r2 = run_evaluation("Chaotic Lorenz Attractor", lorenz_data, is_text=False)

    summary_results = {
        "exp_id": "EXP-385",
        "raw_text_stream": r1,
        "lorenz_chaotic_stream": r2
    }

    # KEP Rule #2 Verdict
    is_positive = (r1["final_bit_errors"] < 1.5) and (r2["final_strain"] < 0.10)
    verdict = "🟢 POSITIVE" if is_positive else "⚪ NEUTRAL"
    print(f"👑 FINAL VERDICT: {verdict}")

    summary_results["verdict"] = verdict

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_385_results.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    return summary_results


if __name__ == "__main__":
    run_exp_385()
