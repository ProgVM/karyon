"""
EXP-381: Sovereign Gödelian Autopoiesis with Massive Parametric Capacity (SGA-MPC)
========================================================================================
Sovereign Architectural Mandate from Bazilevs & Gema:
To destroy prediction surprise in high-entropy domains (like raw language streams),
we scale the internal parametric capacity of our Sovereign Gödelian Engine to a massive scale:
- Hidden Dimension (dim): 64 -> 256 (16x increase in matrix parameter space per node)
- Max Nodes (max_nodes): 16 -> 24
- Active Nodes (active_count): 4 -> 8 (2x active computational pathways)
- Sequence context length for evaluation: 2048

This scales the active parameter space to millions of continuous variables, giving
the system the necessary "variety" (Ashby's Law of Requisite Variety) to master
the raw UTF-8 Byte text stream, while maintaining the protective Gödel Machine Sandbox
to prevent self-lobotomy.

Evaluated on:
1. Domain 1: Raw Byte UTF-8 Text Stream (Linguistic Stream)
2. Domain 2: Chaotic Lorenz Attractor (Nonlinear Physics)
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
import copy

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. DATA GENERATORS (MASSIVE CAPACITY TEST)
# =============================================================================

def generate_domain1_text(length=2048):
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
        bits = [1.0 if ((b >> i) & 1) else -1.0 for i in range(8)]
        vectors.append(bits)
    return torch.tensor(vectors, dtype=torch.float32, device=DEVICE), raw_bytes


def generate_domain2_lorenz(length=2048, dt=0.01):
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


# =============================================================================
# 2. MASSIVE CAPACITY GÖDELIAN ENGINE
# =============================================================================

class HomoiconicNode(nn.Module):
    def __init__(self, node_id: int, dim: int = 256, device: torch.device = DEVICE):
        super().__init__()
        self.node_id = node_id
        self.dim = dim
        self.device = device

        # Genome Vector
        self.genome = nn.Parameter(torch.randn(6, device=device) * 0.1)

        # Massive Projections
        self.W_in = nn.Parameter(torch.randn(dim, 8, device=device) * (1.0 / math.sqrt(8)))
        self.W_state = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_bilinear = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.bias = nn.Parameter(torch.zeros(dim, device=device))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, coupling_input: torch.Tensor):
        g_force = torch.softmax(self.genome[:4], dim=-1)
        epi_gate = torch.tanh(self.genome[4])
        log_tau = self.genome[5]
        tau = torch.sigmoid(log_tau)

        V_in = torch.matmul(x_t, self.W_in.t())
        V_state = torch.matmul(h_prev, self.W_state.t()) + self.bias

        # Basic force fields
        f0 = V_state
        f1 = V_state * torch.sigmoid(torch.matmul(h_prev, self.W_bilinear.t()))
        f2 = torch.sin(V_state)
        f3 = torch.tanh(V_state)

        local_blend = g_force[0] * f0 + g_force[1] * f1 + g_force[2] * f2 + g_force[3] * f3

        # Integrator step with high capacity and selective coupling
        h_next = (1.0 - tau) * h_prev + tau * torch.tanh(local_blend + V_in + coupling_input)
        gated_h = epi_gate * h_next

        return gated_h, h_next, epi_gate, g_force, tau


class SovereignGoedelianEngine(nn.Module):
    def __init__(self, dim: int = 256, max_nodes: int = 24, device: torch.device = DEVICE):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes
        self.device = device

        # Massive AST Node pool
        self.nodes = nn.ModuleList([HomoiconicNode(i, dim, device) for i in range(max_nodes)])
        self.active_count = 8  # Scale active nodes

        # Initialize active nodes
        with torch.no_grad():
            for i in range(8):
                self.nodes[i].genome[4].fill_(3.0)

        # Conformal boundary readouts
        self.W_out = nn.Parameter(torch.randn(8, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_out = nn.Parameter(torch.zeros(8, device=device))

        # Dynamic Observer Attention
        self.W_obs = nn.Parameter(torch.randn(dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_obs = nn.Parameter(torch.zeros(1, device=device))

        # Selective Directed Affinity Projections
        self.W_src = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_tgt = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.tau_aff = nn.Parameter(torch.tensor([0.45], device=device))

        self.sprout_events = 0
        self.rejections = 0

    def forward(self, x_t: torch.Tensor, prev_states: torch.Tensor):
        A = self.active_count
        states = prev_states[:A]

        # 1. Compute Selective Directed Coupling
        src_proj = torch.matmul(states, self.W_src)
        tgt_proj = torch.matmul(states, self.W_tgt)
        affinity = torch.matmul(src_proj, tgt_proj.t()) / math.sqrt(self.dim)

        E_matrix = torch.sigmoid(15.0 * (affinity - self.tau_aff))
        E_matrix = E_matrix * (1.0 - torch.eye(A, device=self.device))

        coupling_inputs = torch.matmul(E_matrix, states)

        # 2. Step active nodes sequentially
        next_active_states = []
        node_gates = []
        node_forces = []
        node_taus = []

        for i in range(A):
            gated_h, raw_h, gate, force, tau = self.nodes[i](x_t, states[i], coupling_inputs[i])
            next_active_states.append(raw_h)
            node_gates.append(gate)
            node_forces.append(force)
            node_taus.append(tau)

        next_active_states = torch.stack(next_active_states, dim=0)
        node_gates = torch.stack(node_gates, dim=0)
        node_forces = torch.stack(node_forces, dim=0)
        node_taus = torch.stack(node_taus, dim=0)

        # 3. Dynamic Observer & State Collapse
        mean_state = torch.mean(next_active_states * node_gates.unsqueeze(-1), dim=0)
        omega_t = torch.sigmoid(torch.sum(mean_state * self.W_obs) + self.b_obs)

        beta_collapse = 2.5
        collapsed_states = (1.0 - omega_t) * next_active_states + omega_t * torch.tanh(beta_collapse * next_active_states)

        # 4. Directed Teleportation
        delta_s = collapsed_states - next_active_states
        teleported_delta = torch.matmul(E_matrix, delta_s)
        final_active_states = collapsed_states + teleported_delta

        # 5. Readout Prediction
        ensemble_signal = torch.sum(node_gates.unsqueeze(-1) * final_active_states, dim=0)
        y_pred = torch.tanh(torch.matmul(self.W_out, ensemble_signal) + self.b_out)

        # Pad back to max_nodes
        next_states = prev_states.clone()
        next_states[:A] = final_active_states

        complexity = torch.sum(torch.abs(node_gates))

        return y_pred, next_states, complexity, omega_t, E_matrix, node_forces, node_taus

    # -------------------------------------------------------------------------
    # GÖDEL MACHINE SANDBOX VALIDATION
    # -------------------------------------------------------------------------
    def evaluate_mutation_in_sandbox(self, x_seq: torch.Tensor, target_seq: torch.Tensor, candidate_node_id: int, candidate_genome: torch.Tensor) -> bool:
        test_node = copy.deepcopy(self.nodes[candidate_node_id])
        with torch.no_grad():
            test_node.genome.copy_(candidate_genome)

        test_states = torch.zeros(self.active_count, self.dim, device=self.device)
        test_loss = 0.0

        try:
            for t in range(min(32, len(x_seq))):
                cur_x = x_seq[t]
                tgt_x = target_seq[t]
                _, raw_h, _, _, _ = test_node(cur_x, test_states[candidate_node_id], torch.zeros(self.dim, device=self.device))

                if torch.isnan(raw_h).any() or torch.isinf(raw_h).any():
                    return False

                pred_contrib = torch.tanh(torch.matmul(self.W_out, raw_h))
                test_loss += 0.5 * torch.sum((pred_contrib - tgt_x)**2).item()

            baseline_loss = 0.0
            baseline_states = torch.zeros(self.active_count, self.dim, device=self.device)
            for t in range(min(32, len(x_seq))):
                cur_x = x_seq[t]
                tgt_x = target_seq[t]
                _, raw_h, _, _, _ = self.nodes[candidate_node_id](cur_x, baseline_states[candidate_node_id], torch.zeros(self.dim, device=self.device))
                pred_contrib = torch.tanh(torch.matmul(self.W_out, raw_h))
                baseline_loss += 0.5 * torch.sum((pred_contrib - tgt_x)**2).item()

            if test_loss <= baseline_loss * 1.05:
                return True
        except Exception:
            return False

        return False


# =============================================================================
# 3. EVALUATION PIPELINE
# =============================================================================

def evaluate_massive_domain(domain_name, x_data, raw_bytes=None, is_text=False):
    print(f"\nEvaluating Massive Domain: {domain_name}...")
    length = x_data.size(0)

    model = SovereignGoedelianEngine(dim=256, max_nodes=24, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2, weight_decay=1e-5)
    states = torch.zeros(model.max_nodes, model.dim, device=DEVICE)

    errors = []
    bit_mismatches = []
    rolling_strain = 2.0
    t0 = time.time()

    for t in range(length - 1):
        cur_vec = x_data[t]
        tgt_vec = x_data[t + 1]

        optimizer.zero_grad()
        y_pred, next_states, complexity, omega_t, E_matrix, node_forces, node_taus = model(cur_vec, states)
        states = next_states.detach()

        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        free_energy = prediction_strain + 0.02 * complexity + 0.01 * model.active_count

        free_energy.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        rolling_strain = 0.90 * rolling_strain + 0.10 * strain_val
        errors.append(strain_val)

        # Massive Gödel Sandbox Morphogenesis
        if t > 50 and t % 32 == 0:
            if rolling_strain > 0.50 and model.active_count < model.max_nodes:
                candidate_id = model.active_count
                candidate_genome = torch.randn(6, device=DEVICE) * 0.1
                candidate_genome[4] = 2.0

                past_x = x_data[max(0, t-32):t]
                past_y = x_data[max(0, t-31):t+1]

                if model.evaluate_mutation_in_sandbox(past_x, past_y, candidate_id, candidate_genome):
                    model.nodes[candidate_id].genome.data.copy_(candidate_genome)
                    model.active_count += 1
                    model.sprout_events += 1
                    print(f"  [GÖDEL SPROUT ACCEPTED] Sprouted Node {candidate_id} at step {t+1}. Active: {model.active_count}")
                else:
                    model.rejections += 1
            else:
                target_node_id = random.randint(0, model.active_count - 1)
                mutation_delta = torch.randn(6, device=DEVICE) * 0.05
                proposed_genome = model.nodes[target_node_id].genome.data + mutation_delta

                past_x = x_data[max(0, t-32):t]
                past_y = x_data[max(0, t-31):t+1]

                if model.evaluate_mutation_in_sandbox(past_x, past_y, target_node_id, proposed_genome):
                    model.nodes[target_node_id].genome.data.copy_(proposed_genome)
                else:
                    model.rejections += 1

        if is_text and raw_bytes is not None:
            # Bit parity check
            signs = (y_pred > 0.0).cpu().numpy().astype(int)
            pred_byte = 0
            for i, bit in enumerate(signs):
                if bit:
                    pred_byte |= (1 << i)
            tgt_byte = raw_bytes[t + 1]
            bit_diff = bin(pred_byte ^ tgt_byte).count('1')
            bit_mismatches.append(bit_diff)

        if (t + 1) % 256 == 0:
            tail_err = np.mean(errors[-64:])
            tail_bits = np.mean(bit_mismatches[-64:]) if is_text else 0.0
            print(f"  Step {t+1:04d}/{length} | Active Nodes: {model.active_count:02d} | Strain: {tail_err:.4f}" + (f" | Bit Errors: {tail_bits:.2f}/8" if is_text else ""))

    elapsed = time.time() - t0
    tok_per_sec = length / elapsed if elapsed > 0 else 0.0

    q_len = max(1, len(errors) // 4)
    final_strain = float(np.mean(errors[-q_len:]))
    final_bit_errors = float(np.mean(bit_mismatches[-q_len:])) if len(bit_mismatches) > 0 else 0.0

    print(f"  [{domain_name} Massive Complete] Final Strain: {final_strain:.4f} | Bit Errors: {final_bit_errors:.2f}/8 | Active Nodes: {model.active_count} | Sprout Events: {model.sprout_events} | Rejections: {model.rejections} | Throughput: {tok_per_sec:.1f} tok/s")

    return {
        "final_strain": final_strain,
        "final_bit_errors": final_bit_errors,
        "active_nodes": model.active_count,
        "sprout_events": model.sprout_events,
        "rejections": model.rejections,
        "tok_per_sec": tok_per_sec
    }


def run_exp_381():
    print("===============================================================================")
    print("=== KEP EXP-381: MASSIVE PARAMETRIC CAPACITY TEST (SGA-MPC)                 ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    # Generate long streams
    text_data, text_bytes = generate_domain1_text(2048)
    lorenz_data = generate_domain2_lorenz(2048)

    # Run massive evaluations
    r1 = evaluate_massive_domain("Domain 1: Massive Raw UTF-8 Text", text_data, text_bytes, is_text=True)
    r2 = evaluate_massive_domain("Domain 2: Massive Chaotic Lorenz Attractor", lorenz_data, is_text=False)

    summary_results = {
        "exp_id": "EXP-381",
        "domain_1_text": r1,
        "domain_2_lorenz": r2
    }

    print("\n===============================================================================")
    print("=== MASSIVE GÖDELIAN SUITE SUMMARY TELEMETRY ===")
    print("===============================================================================")
    print(f"  Domain 1 (Massive Text)   | Strain: {r1['final_strain']:.4f} | Bit Errors: {r1['final_bit_errors']:.2f}/8 | Speed: {r1['tok_per_sec']:.1f} tok/s")
    print(f"  Domain 2 (Massive Lorenz) | Strain: {r2['final_strain']:.4f} | Speed: {r2['tok_per_sec']:.1f} tok/s")
    print("-------------------------------------------------------------------------------")

    # KEP Rule #2 Verdict
    # If massive text bit errors plunge significantly below 2.22 (e.g., < 1.4 bits/8), it is a massive POSITIVE.
    is_positive = (r1["final_bit_errors"] < 1.4) and (r2["final_strain"] < 0.05)
    verdict = "🟢 POSITIVE" if is_positive else "⚪ NEUTRAL"
    print(f"👑 VERDICT: {verdict}")

    summary_results["verdict"] = verdict

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_381_results.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    return summary_results


if __name__ == "__main__":
    run_exp_381()
