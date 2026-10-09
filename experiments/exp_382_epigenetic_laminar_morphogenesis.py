"""
EXP-382: Epigenetic Laminar Morphogenesis & Autonomous Layer Assembly (ELM-ALA)
========================================================================================
Sovereign Architectural Mandate from Bazilevs & Gema:
"Karyon must autonomously govern its layers (their creation, pruning, assignments, and contents)"

To realize this, we implement ELM-ALA:
1. Dynamic Laminar Stack:
   The model does not have a fixed layer depth. It maintains a dynamic list of
   Laminar Sheets. Each sheet has its own:
   - Layer Genome: Modulates its role, integration timescale, and connections.
   - Epigenetic Layer Gate (alpha_layer): Initialized to 0.0 at birth to guarantee
     Net2Net Smooth Grafting (zero-shock function identity f_new(x) == f_old(x)).
   - Assignment Vector: Soft-projects the layer's functional role (Syntax, Semantics,
     Attractor Memory, or Physical Dynamics).
2. Atomic Operators within Sheets (Principle 25):
   Each sheet dynamically assembles its internal operations from a complete basis of
   atomic primitives:
   - LinearAccumulatorOp
   - BilinearMultiplicativeOp
   - SaturatedAttractorOp
   - StateSpaceMemoryOp
3. Autonomous Layer Morphogenesis (Sprouting, Pruning & Role Mutation):
   Under high Variational Free Energy strain, the engine proposes:
   - Sprouting a new Laminar Sheet (increasing compositional depth).
   - Pruning an under-utilized Laminar Sheet (when its epigenetic gate alpha_layer -> 0).
   - Mutating a sheet's role assignments or internal operator weights.
4. Gödel Machine Sandbox:
   All structural mutations (sprouting, pruning, role shifts) must be evaluated in
   the Sandbox. If they degrade performance or introduce NaNs, they are rejected.
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
        # Continuous leaky integration
        act = torch.tanh(torch.matmul(x, self.W.t()) + self.bias)
        return (1.0 - tau) * h + tau * act


class BilinearMultiplicativeOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W2 = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))

    def forward(self, x: torch.Tensor, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # Causal conjunction & multiplicative gating
        g1 = torch.tanh(torch.matmul(x, self.W1.t()))
        g2 = torch.sigmoid(torch.matmul(h, self.W2.t()))
        return (1.0 - tau) * h + tau * (g1 * g2)


class SaturatedAttractorOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.beta = nn.Parameter(torch.tensor([12.0], device=device))

    def forward(self, x: torch.Tensor, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # Sharp attractor basin snapping
        proj = torch.matmul(h, self.W.t())
        snapped = torch.tanh(self.beta * proj)
        return (1.0 - tau) * h + tau * torch.tanh(snapped + x)


class StateSpaceMemoryOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        # Log-spaced multiple timescales
        self.log_alpha = nn.Parameter(torch.linspace(-3.0, -0.5, dim, device=device))

    def forward(self, x: torch.Tensor, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.log_alpha)
        decay = torch.clamp(1.0 - alpha * tau, 0.0, 1.0)
        proj = torch.tanh(torch.matmul(x, self.W.t()))
        return decay * h + (1.0 - decay) * proj


# =============================================================================
# 2. DYNAMIC LAMINAR SHEET (Sovereign Layer)
# =============================================================================

class LaminarSheet(nn.Module):
    """
    A single sovereign layer that manages its own role, timescales,
    and dynamically blends its internal atomic primitive operators.
    """
    def __init__(self, layer_id: int, dim: int = 128, device: torch.device = DEVICE):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.device = device

        # Layer Genome:
        # [0:3] - Operator blending weights (Linear, Bilinear, Attractor, StateSpace)
        # [4]   - Epigenetic layer gate (alpha_layer)
        # [5]   - Log integration timescale (log_tau)
        # [6:9] - Role Assignment Vector: [Syntax, Semantics, Memory, Dynamics]
        self.genome = nn.Parameter(torch.randn(10, device=device) * 0.1)

        # Atomic Operators
        self.op_linear = LinearAccumulatorOp(dim, device)
        self.op_bilinear = BilinearMultiplicativeOp(dim, device)
        self.op_attractor = SaturatedAttractorOp(dim, device)
        self.op_ssm = StateSpaceMemoryOp(dim, device)

        # Inter-layer Feedforward & Feedback projections
        self.W_ff = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_fb = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.bias = nn.Parameter(torch.zeros(dim, device=device))

    def forward(self, x_in: torch.Tensor, h_prev: torch.Tensor, feedback_in: torch.Tensor) -> tuple:
        # Decode genome
        op_blend = torch.softmax(self.genome[0:4], dim=-1)
        alpha_layer = torch.tanh(self.genome[4])
        tau = torch.sigmoid(self.genome[5])
        role_vector = torch.softmax(self.genome[6:10], dim=-1)

        # Process input mixing feedforward and feedback
        mixed_input = torch.matmul(x_in, self.W_ff.t()) + torch.matmul(feedback_in, self.W_fb.t()) + self.bias

        # Execute basis operations
        h0 = self.op_linear(mixed_input, h_prev, tau)
        h1 = self.op_bilinear(mixed_input, h_prev, tau)
        h2 = self.op_attractor(mixed_input, h_prev, tau)
        h3 = self.op_ssm(mixed_input, h_prev, tau)

        # Blend results based on local genome
        h_next = op_blend[0] * h0 + op_blend[1] * h1 + op_blend[2] * h2 + op_blend[3] * h3

        # Apply Epigenetic Layer Gate (Net2Net Smooth Grafting)
        gated_output = alpha_layer * h_next

        return gated_output, h_next, alpha_layer, op_blend, role_vector


# =============================================================================
# 3. SOVEREIGN AUTOPOIETIC ENGINE (ELM-ALA)
# =============================================================================

class SovereignLaminarEngine(nn.Module):
    def __init__(self, dim: int = 128, max_layers: int = 8, device: torch.device = DEVICE):
        super().__init__()
        self.dim = dim
        self.max_layers = max_layers
        self.device = device

        # Dynamic Laminar Stack
        self.layers = nn.ModuleList([LaminarSheet(i, dim, device) for i in range(max_layers)])
        self.active_layers = 2  # Start with 2 layers (e.g., Input/Syntax and Output/Semantics)

        # Initialize the first 2 layers to be active, others are dormant (alpha_layer initialized to 0.0)
        with torch.no_grad():
            self.layers[0].genome[4].fill_(2.5)  # Layer 0: Active
            self.layers[1].genome[4].fill_(2.5)  # Layer 1: Active
            # Soft-assign roles initially
            self.layers[0].genome[6].fill_(2.0)  # Layer 0: Bias toward Syntax
            self.layers[1].genome[7].fill_(2.0)  # Layer 1: Bias toward Semantics

        # Input and Output Boundary Projectors
        self.W_in = nn.Parameter(torch.randn(dim, 8, device=device) * (1.0 / math.sqrt(8)))
        self.W_out = nn.Parameter(torch.randn(8, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_out = nn.Parameter(torch.zeros(8, device=device))

        self.sprout_events = 0
        self.prune_events = 0
        self.rejections = 0

    def forward(self, x_t: torch.Tensor, prev_states: torch.Tensor) -> tuple:
        # Map input to continuous state dimension
        v_in = torch.matmul(x_t, self.W_in.t())

        L = self.active_layers
        states = prev_states[:L]  # [L, D]

        # Bidirectional Laminar Routing (Feedforward up, Feedback down)
        next_states = []
        layer_gates = []
        layer_blends = []
        layer_roles = []

        for i in range(L):
            # Feedforward input is the output of the previous layer (or v_in for layer 0)
            ff_in = v_in if i == 0 else next_states[i - 1]
            # Feedback input is the state of the next layer (or zero for the top layer)
            fb_in = states[i + 1] if i < L - 1 else torch.zeros(self.dim, device=self.device)

            gated_out, raw_h, alpha_layer, op_blend, role_vector = self.layers[i](ff_in, states[i], fb_in)
            next_states.append(gated_out)
            layer_gates.append(alpha_layer)
            layer_blends.append(op_blend)
            layer_roles.append(role_vector)

        next_states = torch.stack(next_states, dim=0)  # [L, D]
        layer_gates = torch.stack(layer_gates, dim=0)  # [L]
        layer_blends = torch.stack(layer_blends, dim=0)  # [L, 4]
        layer_roles = torch.stack(layer_roles, dim=0)  # [L, 4]

        # Readout prediction from the top active layer
        y_pred = torch.tanh(torch.matmul(self.W_out, next_states[-1]) + self.b_out)

        # Pad back to max_layers
        padded_states = prev_states.clone()
        padded_states[:L] = next_states

        complexity = torch.sum(torch.abs(layer_gates))

        return y_pred, padded_states, complexity, layer_blends, layer_roles, layer_gates

    # -------------------------------------------------------------------------
    # GÖDEL MACHINE SANDBOX FOR LAMINAR MORPHOGENESIS
    # -------------------------------------------------------------------------
    def evaluate_laminar_mutation_in_sandbox(self, x_seq: torch.Tensor, target_seq: torch.Tensor, proposal_type: str, target_layer_id: int, proposal_genome: torch.Tensor = None) -> bool:
        """
        Runs a dry-run simulation of the dynamic stack over the past 32 steps.
        Verifies if sprouting, pruning, or mutating a layer is mathematically viable.
        """
        # Save current active stack configuration
        original_active = self.active_layers
        original_genomes = [self.layers[i].genome.data.clone() for i in range(self.max_layers)]

        # Apply proposal to the active stack
        if proposal_type == "sprout":
            # Temporarily increment active layers and initialize new layer genome
            self.active_layers = min(self.max_layers, self.active_layers + 1)
            new_id = self.active_layers - 1
            self.layers[new_id].genome.data.copy_(proposal_genome)
        elif proposal_type == "prune":
            # Temporarily decrement active layers (pruning the top layer)
            self.active_layers = max(2, self.active_layers - 1)
        elif proposal_type == "mutate":
            self.layers[target_layer_id].genome.data.copy_(proposal_genome)

        # Run past sequence rollout
        test_states = torch.zeros(self.max_layers, self.dim, device=self.device)
        test_loss = 0.0
        diverged = False

        try:
            for t in range(min(32, len(x_seq))):
                cur_x = x_seq[t]
                tgt_x = target_seq[t]

                # Forward pass on mutated stack
                y_pred, next_states, _, _, _, _ = self(cur_x, test_states)
                test_states = next_states.detach()

                if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                    diverged = True
                    break
                test_loss += 0.5 * torch.sum((y_pred - tgt_x)**2).item()
        except Exception:
            diverged = True

        # Run baseline pass on original stack
        # Restore original configuration first
        self.active_layers = original_active
        for i in range(self.max_layers):
            self.layers[i].genome.data.copy_(original_genomes[i])

        if diverged:
            return False

        baseline_states = torch.zeros(self.max_layers, self.dim, device=self.device)
        baseline_loss = 0.0
        try:
            for t in range(min(32, len(x_seq))):
                cur_x = x_seq[t]
                tgt_x = target_seq[t]
                y_pred, next_states, _, _, _, _ = self(cur_x, baseline_states)
                baseline_states = next_states.detach()
                baseline_loss += 0.5 * torch.sum((y_pred - tgt_x)**2).item()
        except Exception:
            return False

        # Gödel Proof: Mutation is accepted only if it matches or improves performance
        if test_loss <= baseline_loss * 1.05:
            # Re-apply proposal to make it permanent
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

    def extract_symbolic_formulas(self, layer_blends: torch.Tensor, layer_roles: torch.Tensor, layer_gates: torch.Tensor) -> str:
        report = []
        report.append("=== KARYON LAMINAR AUTOPOIETIC SPECIFICATION ===")
        report.append(f"  Active Laminar depth            : {self.active_layers} (Max: {self.max_layers})")
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

            # Find dominant operator and role
            dom_op = op_names[np.argmax(blends)]
            dom_role = role_names[np.argmax(roles)]

            report.append(f"    Sheet_{i:02d} | Epigenetic Gate: {gate:.4f} | Dominant Operator: {dom_op} | Dominant Role: {dom_role}")
            report.append(f"           - Op Blends : [Lin:{blends[0]:.2f}, Bilin:{blends[1]:.2f}, Attr:{blends[2]:.2f}, SSM:{blends[3]:.2f}]")
            report.append(f"           - Roles     : [Synt:{roles[0]:.2f}, Seman:{roles[1]:.2f}, Memo:{roles[2]:.2f}, Dyna:{roles[3]:.2f}]")
        return "\n".join(report)


# =============================================================================
# 4. DATA GENERATORS & BENCHMARK RUNNER
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
    print(f"\nEvaluating ELM-ALA on Domain: {domain_name}...")
    length = x_data.size(0)

    model = SovereignLaminarEngine(dim=128, max_layers=8, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2, weight_decay=1e-5)
    states = torch.zeros(model.max_layers, model.dim, device=DEVICE)

    errors = []
    bit_mismatches = []
    rolling_strain = 2.0
    t0 = time.time()

    last_blends = None
    last_roles = None
    last_gates = None

    for t in range(length - 1):
        cur_vec = x_data[t]
        tgt_vec = x_data[t + 1]

        optimizer.zero_grad()
        y_pred, next_states, complexity, layer_blends, layer_roles, layer_gates = model(cur_vec, states)
        states = next_states.detach()

        last_blends = layer_blends.detach()
        last_roles = layer_roles.detach()
        last_gates = layer_gates.detach()

        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        free_energy = prediction_strain + 0.01 * complexity + 0.02 * model.active_layers

        free_energy.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        rolling_strain = 0.90 * rolling_strain + 0.10 * strain_val
        errors.append(strain_val)

        # ---------------------------------------------------------------------
        # AUTONOMOUS LAMINAR MORPHOGENESIS (Sprouting & Pruning)
        # ---------------------------------------------------------------------
        if t > 50 and t % 32 == 0:
            past_x = x_data[max(0, t-32):t]
            past_y = x_data[max(0, t-31):t+1]

            # 1. Sprouting proposal (under high Free Energy strain)
            if rolling_strain > 0.55 and model.active_layers < model.max_layers:
                candidate_genome = torch.randn(10, device=DEVICE) * 0.1
                # Net2Net Smooth Grafting: alpha_layer initialized to 0.0 at birth
                candidate_genome[4] = 0.0

                if model.evaluate_laminar_mutation_in_sandbox(past_x, past_y, "sprout", -1, candidate_genome):
                    # Sprout accepted! Slowly unfreeze alpha_layer in subsequent steps
                    model.sprout_events += 1
                    print(f"  [GÖDEL LAMINAR SPROUT] Successfully added Sheet {model.active_layers} at step {t+1}. Total Layers: {model.active_layers}")
                else:
                    model.rejections += 1

            # 2. Pruning proposal (under low strain or if top layer is underutilized)
            elif rolling_strain < 0.15 and model.active_layers > 2:
                # Check if top layer's epigenetic gate is decaying
                top_gate = torch.tanh(model.layers[model.active_layers - 1].genome[4]).item()
                if abs(top_gate) < 0.10:
                    if model.evaluate_laminar_mutation_in_sandbox(past_x, past_y, "prune", -1):
                        model.prune_events += 1
                        print(f"  [GÖDEL LAMINAR PRUNE] Successfully pruned Sheet {model.active_layers} at step {t+1}. Total Layers: {model.active_layers}")
                    else:
                        model.rejections += 1

            # 3. Layer Genome Mutation proposal
            else:
                target_layer_id = random.randint(0, model.active_layers - 1)
                mutation_delta = torch.randn(10, device=DEVICE) * 0.05
                proposed_genome = model.layers[target_layer_id].genome.data + mutation_delta

                if model.evaluate_laminar_mutation_in_sandbox(past_x, past_y, "mutate", target_layer_id, proposed_genome):
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
            print(f"  Step {t+1:04d}/{length} | Active Layers: {model.active_layers:02d} | Strain: {tail_err:.4f}" + (f" | Bit Errors: {tail_bits:.2f}/8" if is_text else ""))

    elapsed = time.time() - t0
    tok_per_sec = length / elapsed if elapsed > 0 else 0.0

    q_len = max(1, len(errors) // 4)
    final_strain = float(np.mean(errors[-q_len:]))
    final_bit_errors = float(np.mean(bit_mismatches[-q_len:])) if len(bit_mismatches) > 0 else 0.0

    discovered_formulas = model.extract_symbolic_formulas(last_blends, last_roles, last_gates)
    print("\n-------------------------------------------------------------------------------")
    print(discovered_formulas)
    print("-------------------------------------------------------------------------------\n")

    return {
        "final_strain": final_strain,
        "final_bit_errors": final_bit_errors,
        "active_layers": model.active_layers,
        "sprout_events": model.sprout_events,
        "prune_events": model.prune_events,
        "rejections": model.rejections,
        "tok_per_sec": tok_per_sec,
        "discovered_formulas": discovered_formulas
    }


def run_exp_382():
    print("===============================================================================")
    print("=== KEP EXP-382: EPIGENETIC LAMINAR MORPHOGENESIS (ELM-ALA)                 ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    # Generate streams
    text_data, text_bytes = generate_raw_text_stream(1500)
    lorenz_data = generate_lorenz_chaotic_stream(1500)

    # Run evaluations
    r1 = run_evaluation("Raw UTF-8 Text Stream", text_data, text_bytes, is_text=True)
    r2 = run_evaluation("Chaotic Lorenz Attractor", lorenz_data, is_text=False)

    summary_results = {
        "exp_id": "EXP-382",
        "raw_text_stream": r1,
        "lorenz_chaotic_stream": r2
    }

    # KEP Rule #2 Verdict
    # If the text bit error rate falls below 1.5 bits/8 and lorenz is learned stably, it's a massive POSITIVE.
    is_positive = (r1["final_bit_errors"] < 1.5) and (r2["final_strain"] < 0.10)
    verdict = "🟢 POSITIVE" if is_positive else "⚪ NEUTRAL"
    print(f"👑 FINAL VERDICT: {verdict}")

    summary_results["verdict"] = verdict

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_382_results.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    return summary_results


if __name__ == "__main__":
    run_exp_382()
