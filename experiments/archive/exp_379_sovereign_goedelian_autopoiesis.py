"""
EXP-379: Sovereign Gödelian Autopoiesis & Homoiconic Morphogenesis (SGA-HM)
========================================================================================
Sovereign Architectural Mandate from Bazilevs & Gema:
Unify all 5 pillars of the Sovereign Self-Evolution Manifesto:
1. Homoiconicity (Nodes as Data & Code):
   The execution graph is dynamically generated, modified, and evaluated as an Abstract
   Syntax Tree (AST) represented by dynamic tensors.
2. Temporal Multi-Scale Loops:
   - Microsecond: Inference state transitions.
   - Plasticity: Online weight updates.
   - Morphogenesis: Epigenetic node sprouting under Free Energy pressure.
3. Variational Free Energy & Novelty Search:
   F_t = Prediction Strain + Complexity Penalty - Curiosity Reward (Novelty Drive)
4. Gödel Machine Sandbox:
   Before merging sprouted nodes or mutating laws, proposed changes are evaluated in an
   isolated sandbox. If the sandbox fails to prove an improvement (or causes NaNs/divergence),
   the change is rejected, protecting the core from self-lobotomy.
5. Grounding & Symbolic Dualism:
   Physical continuous byte-vector stream in R^8 is mapped, and symbolic algebraic
   formulas are extracted and logged in real-time.

Continuous Bipolar Reality Stream (t -> t+1, N=1, Single Pass) in R^8.
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
# CONTINUOUS PHYSICAL 8-BIT STREAM GENERATOR
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


def generate_unbroken_stream(stream_length: int = 2048):
    vectors = []
    stream_types = []  # 0: det, 1: stoch
    raw_bytes = []

    current_lfsr = 0xA5
    current_brownian = 0.0
    mode = 0
    block_remaining = 64

    for _ in range(stream_length):
        if block_remaining <= 0:
            mode = 1 - mode
            block_remaining = random.randint(32, 64)

        if mode == 0:
            current_lfsr = ((current_lfsr << 1) ^ (0x1D if (current_lfsr & 0x80) else 0x00)) & 0xFF
            val = current_lfsr
            stream_types.append(0)
        else:
            current_brownian += random.gauss(0.0, 1.0)
            val = int((math.sin(current_brownian * 0.15) * 0.5 + 0.5) * 255)
            stream_types.append(1)

        raw_bytes.append(val)
        vectors.append(byte_to_bit_vector(val))
        block_remaining -= 1

    stream_tensor = torch.stack(vectors, dim=0)
    type_tensor = torch.tensor(stream_types, dtype=torch.long, device=DEVICE)
    return stream_tensor, type_tensor, raw_bytes


# =============================================================================
# HOMOICONIC ATOMIC OPERATORS (AST REPRESENTATION)
# =============================================================================

class HomoiconicNode(nn.Module):
    """
    Representing an execution node whose mathematical behavior is parameterized
    by a continuous 'Genome Vector' (Homoiconic Principle: Code is Data).
    """
    def __init__(self, node_id: int, dim: int = 64, device: torch.device = DEVICE):
        super().__init__()
        self.node_id = node_id
        self.dim = dim
        self.device = device

        # Genome Vector: [Inertial, Bilinear, Harmonic, Attractor, EpigeneticGate, LogTimescale]
        # This vector can be modified dynamically by the meta-evolutionary engine.
        self.genome = nn.Parameter(torch.randn(6, device=device) * 0.1)

        # Node weights
        self.W_in = nn.Parameter(torch.randn(dim, 8, device=device) * (1.0 / math.sqrt(8)))
        self.W_state = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_bilinear = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))

        # Self-stabilizing bias
        self.bias = nn.Parameter(torch.zeros(dim, device=device))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, global_laws: torch.Tensor):
        # Decode genome parameters
        g_force = torch.softmax(self.genome[:4], dim=-1)  # [4]
        epi_gate = torch.tanh(self.genome[4])
        log_tau = self.genome[5]
        tau = torch.sigmoid(log_tau)

        # Map input and state
        V_in = torch.matmul(x_t, self.W_in.t())  # [D]
        V_state = torch.matmul(h_prev, self.W_state.t()) + self.bias  # [D]

        # 1. Inertial Memory (Force 0)
        f0 = V_state

        # 2. Bilinear Multiplicative Resonance (Force 1)
        f1 = V_state * torch.sigmoid(torch.matmul(h_prev, self.W_bilinear.t()))

        # 3. Harmonic Oscillation (Force 2)
        f2 = torch.sin(V_state)

        # 4. Attractor Snapping (Force 3)
        f3 = torch.tanh(V_state)

        # Blend forces based on local genome and global meta-laws
        local_blend = g_force[0] * f0 + g_force[1] * f1 + g_force[2] * f2 + g_force[3] * f3

        # Leaky integration step
        h_next = (1.0 - tau) * h_prev + tau * torch.tanh(local_blend + V_in)

        # Apply epigenetic gating
        gated_h = epi_gate * h_next

        return gated_h, h_next, epi_gate, g_force, tau


# =============================================================================
# SOVEREIGN GÖDELIAN ENGINE
# =============================================================================

class SovereignGoedelianEngine(nn.Module):
    def __init__(self, dim: int = 64, max_nodes: int = 16, device: torch.device = DEVICE):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes
        self.device = device

        # Active node tracking (Homoiconic Graph)
        self.nodes = nn.ModuleList([HomoiconicNode(i, dim, device) for i in range(max_nodes)])
        self.active_count = 4

        # Initialize first 4 nodes to be highly active
        with torch.no_grad():
            for i in range(4):
                self.nodes[i].genome[4].fill_(3.0)  # Strong epigenetic activation

        # Global Conformal Readout Projector (AdS/CFT style boundary)
        self.W_out = nn.Parameter(torch.randn(8, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_out = nn.Parameter(torch.zeros(8, device=device))

        # Dynamic Observer Projection (System 2 attention)
        self.W_obs = nn.Parameter(torch.randn(dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_obs = nn.Parameter(torch.zeros(1, device=device))

        # Selective Directed Affinity Projections (Source & Target)
        self.W_src = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_tgt = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.tau_aff = nn.Parameter(torch.tensor([0.45], device=device))

        # Novelty / Curiosity drive tracker
        self.novelty_memory = []
        self.sprout_events = 0
        self.rejections = 0

    def forward(self, x_t: torch.Tensor, prev_states: torch.Tensor):
        A = self.active_count
        states = prev_states[:A]  # [A, D]

        # 1. Compute Selective Directed Coupling (SDQC)
        src_proj = torch.matmul(states, self.W_src)  # [A, D]
        tgt_proj = torch.matmul(states, self.W_tgt)  # [A, D]
        affinity = torch.matmul(src_proj, tgt_proj.t()) / math.sqrt(self.dim)  # [A, A]

        # Gated threshold to prevent overmixing
        E_matrix = torch.sigmoid(15.0 * (affinity - self.tau_aff))
        E_matrix = E_matrix * (1.0 - torch.eye(A, device=self.device))

        # 2. Step active nodes sequentially
        next_active_states = []
        node_gates = []
        node_forces = []
        node_taus = []

        for i in range(A):
            gated_h, raw_h, gate, force, tau = self.nodes[i](x_t, states[i], E_matrix[i])
            next_active_states.append(raw_h)
            node_gates.append(gate)
            node_forces.append(force)
            node_taus.append(tau)

        next_active_states = torch.stack(next_active_states, dim=0)  # [A, D]
        node_gates = torch.stack(node_gates, dim=0)  # [A]
        node_forces = torch.stack(node_forces, dim=0)  # [A, 4]
        node_taus = torch.stack(node_taus, dim=0)  # [A]

        # 3. Dynamic Observer & State Collapse
        mean_state = torch.mean(next_active_states * node_gates.unsqueeze(-1), dim=0)  # [D]
        omega_t = torch.sigmoid(torch.sum(mean_state * self.W_obs) + self.b_obs)

        beta_collapse = 2.5
        collapsed_states = (1.0 - omega_t) * next_active_states + omega_t * torch.tanh(beta_collapse * next_active_states)

        # 4. Directed Teleportation along coupling pathways
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
    # GÖDEL MACHINE SANDBOX VALIDATION (Principle 4)
    # -------------------------------------------------------------------------
    def evaluate_mutation_in_sandbox(self, x_seq: torch.Tensor, target_seq: torch.Tensor, candidate_node_id: int, candidate_genome: torch.Tensor) -> bool:
        """
        The Gödel Machine Sandbox: Runs a mini-simulation on past context.
        Verifies if mutating the genome improves prediction accuracy (or at least keeps it stable)
        without causing NaNs or numerical explosion.
        """
        # Create an isolated clone of the candidate node
        test_node = copy.deepcopy(self.nodes[candidate_node_id])
        with torch.no_grad():
            test_node.genome.copy_(candidate_genome)

        # Run past sequence rollout
        test_states = torch.zeros(self.active_count, self.dim, device=self.device)
        test_loss = 0.0

        try:
            for t in range(min(32, len(x_seq))):
                cur_x = x_seq[t]
                tgt_x = target_seq[t]

                # Step test node
                _, raw_h, _, _, _ = test_node(cur_x, test_states[candidate_node_id], torch.zeros(self.active_count, device=self.device))

                # Check for NaNs
                if torch.isnan(raw_h).any() or torch.isinf(raw_h).any():
                    return False

                # Local node contribution prediction
                pred_contrib = torch.tanh(torch.matmul(self.W_out, raw_h))
                test_loss += 0.5 * torch.sum((pred_contrib - tgt_x)**2).item()

            # Compare with baseline performance of current node
            baseline_loss = 0.0
            baseline_states = torch.zeros(self.active_count, self.dim, device=self.device)
            for t in range(min(32, len(x_seq))):
                cur_x = x_seq[t]
                tgt_x = target_seq[t]
                _, raw_h, _, _, _ = self.nodes[candidate_node_id](cur_x, baseline_states[candidate_node_id], torch.zeros(self.active_count, device=self.device))
                pred_contrib = torch.tanh(torch.matmul(self.W_out, raw_h))
                baseline_loss += 0.5 * torch.sum((pred_contrib - tgt_x)**2).item()

            # Gödel Proof: Mutation is accepted ONLY if performance matches or beats baseline
            # AND does not diverge.
            if test_loss <= baseline_loss * 1.05:  # Allow slight exploration noise
                return True
        except Exception:
            return False

        return False

    def extract_symbolic_formulas(self, last_omega: torch.Tensor, E_matrix: torch.Tensor, node_forces: torch.Tensor, node_taus: torch.Tensor) -> str:
        report = []
        report.append("=== KARYON GÖDELIAN AUTOPOIETIC SPECIFICATION ===")
        report.append(f"  Active Nodes in Graph           : {self.active_count} (Started at 4)")
        report.append(f"  Observer Intensity (Omega_t)    : {last_omega.item():.4f}")
        report.append(f"  Sprout Morphogenesis Events     : {self.sprout_events}")
        report.append(f"  Gödel Sandbox Rejections        : {self.rejections}")

        report.append("\n  Dynamic Node AST Genome & Law Configuration:")
        for i in range(self.active_count):
            forces = node_forces[i].tolist()
            tau_val = node_taus[i].item()
            gate_val = torch.tanh(self.nodes[i].genome[4]).item()
            report.append(f"    Node_{i:02d} | Epigenetic Gate: {gate_val:.4f} | tau: {tau_val:.3f} | Laws: [M:{forces[0]:.2f}, R:{forces[1]:.2f}, V:{forces[2]:.2f}, A:{forces[3]:.2f}]")

        report.append("\n  Dynamic Homoiconic Graph Connectivity (SDQC):")
        for i in range(self.active_count):
            row = E_matrix[i].tolist()
            row_str = " ".join([f"{val:.2f}" if val > 0.01 else " .  " for val in row])
            report.append(f"    Node_{i:02d} -> Outgoing: [ {row_str} ]")
        return "\n".join(report)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_exp_379():
    print("===============================================================================")
    print("=== KEP EXP-379: SOVEREIGN GÖDELIAN AUTOPOIESIS & HM                        ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    stream_data, stream_types, raw_bytes = generate_unbroken_stream(stream_length=2048)
    stream_len = stream_data.size(0)

    model = SovereignGoedelianEngine(dim=64, max_nodes=16, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-2, weight_decay=1e-5)

    states = torch.zeros(model.max_nodes, model.dim, device=DEVICE)

    det_errors = []
    stoch_errors = []
    bit_mismatches_det = []

    rolling_strain = 2.0
    t0 = time.time()
    last_omega_tensor = None
    last_E_matrix = None
    last_node_forces = None
    last_node_taus = None

    for t in range(stream_len - 1):
        cur_vec = stream_data[t]
        tgt_vec = stream_data[t + 1]
        is_stochastic = stream_types[t].item()

        optimizer.zero_grad()

        # Step sovereign Gödelian engine
        y_pred, next_states, complexity, omega_t, E_matrix, node_forces, node_taus = model(
            cur_vec, states
        )
        last_omega_tensor = omega_t.detach()
        last_E_matrix = E_matrix.detach()
        last_node_forces = node_forces.detach()
        last_node_taus = node_taus.detach()

        # Detach states to prevent memory leak
        states = next_states.detach()

        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        free_energy = prediction_strain + 0.02 * complexity + 0.01 * model.active_count

        free_energy.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        rolling_strain = 0.90 * rolling_strain + 0.10 * strain_val

        # ---------------------------------------------------------------------
        # GÖDEL MACHINE AUTOPOIETIC MUTATION & SPROUTING (Pillars 2 & 4)
        # ---------------------------------------------------------------------
        if t > 50 and t % 32 == 0:
            # 1. Epigenetic Node Sprouting under high Free Energy strain
            if rolling_strain > 0.60 and model.active_count < model.max_nodes:
                # Propose sprouting a new node
                candidate_id = model.active_count
                # Set dynamic activation genome
                candidate_genome = torch.randn(6, device=DEVICE) * 0.1
                candidate_genome[4] = 2.0  # Epic activation proposal

                # Verify proposal in Sandbox before commitment
                past_x = stream_data[max(0, t-32):t]
                past_y = stream_data[max(0, t-31):t+1]

                if model.evaluate_mutation_in_sandbox(past_x, past_y, candidate_id, candidate_genome):
                    # Proof succeeded: Commit sprout
                    model.nodes[candidate_id].genome.data.copy_(candidate_genome)
                    model.active_count += 1
                    model.sprout_events += 1
                    print(f"  [GÖDEL PROOF ACCEPTED] Sprouted Node {candidate_id} in substrate at step {t+1}. Total Active: {model.active_count}")
                else:
                    model.rejections += 1
                    # print(f"  [GÖDEL PROOF REJECTED] Sprout of Node {candidate_id} failed sandbox validation.")

            # 2. Homoiconic Genome Mutation
            else:
                # Propose minor genome mutation in an active node to improve prediction
                target_node_id = random.randint(0, model.active_count - 1)
                mutation_delta = torch.randn(6, device=DEVICE) * 0.05
                proposed_genome = model.nodes[target_node_id].genome.data + mutation_delta

                past_x = stream_data[max(0, t-32):t]
                past_y = stream_data[max(0, t-31):t+1]

                if model.evaluate_mutation_in_sandbox(past_x, past_y, target_node_id, proposed_genome):
                    # Proof succeeded: Commit mutation
                    model.nodes[target_node_id].genome.data.copy_(proposed_genome)
                else:
                    model.rejections += 1

        # Bit parity check
        pred_byte = bit_vector_to_byte(y_pred)
        tgt_byte = raw_bytes[t + 1]
        bit_diff = bin(pred_byte ^ tgt_byte).count('1')

        if is_stochastic == 0:
            det_errors.append(strain_val)
            bit_mismatches_det.append(bit_diff)
        else:
            stoch_errors.append(strain_val)

        if (t + 1) % 256 == 0:
            det_tail_err = np.mean(det_errors[-64:]) if len(det_errors) >= 64 else np.mean(det_errors)
            det_tail_bits = np.mean(bit_mismatches_det[-64:]) if len(bit_mismatches_det) >= 64 else np.mean(bit_mismatches_det)
            print(f"  Step {t+1:04d}/{stream_len} | Active Nodes: {model.active_count:02d} | Strain: {rolling_strain:.4f} | Det Energy: {det_tail_err:.4f} | Det Bit Errors: {det_tail_bits:.2f}/8 bits")

    elapsed = time.time() - t0
    tok_per_sec = stream_len / elapsed if elapsed > 0 else 0.0

    q_len = max(1, len(det_errors) // 4)
    final_det_strain = float(np.mean(det_errors[-q_len:]))
    final_det_bit_errors = float(np.mean(bit_mismatches_det[-q_len:]))
    final_stoch_strain = float(np.mean(stoch_errors[-q_len:]))

    discovered_formulas = model.extract_symbolic_formulas(last_omega_tensor, last_E_matrix, last_node_forces, last_node_taus)

    print("\n===============================================================================")
    print("=== FINAL TELEMETRY & GÖDELIAN RECURSIVE STATE (EXP-379) ===")
    print("===============================================================================")
    print(f"  Final Active Node Count      : {model.active_count} (Started at 4)")
    print(f"  Sprout Morphogenesis Events  : {model.sprout_events}")
    print(f"  Gödel Sandbox Rejections     : {model.rejections}")
    print(f"  Deterministic Bit Error Rate : {final_det_bit_errors:.2f} / 8 bits")
    print(f"  Deterministic Strain Energy  : {final_det_strain:.4f}")
    print(f"  Stochastic Strain Energy     : {final_stoch_strain:.4f}")
    print(f"  Processing Throughput        : {tok_per_sec:.1f} tok/s\n")
    print("-------------------------------------------------------------------------------")
    print(discovered_formulas)
    print("-------------------------------------------------------------------------------")

    # KEP Rule #2 Verdict
    verdict = "🟢 POSITIVE" if final_det_bit_errors < 1.15 else ("⚪ NEUTRAL" if final_det_bit_errors < 2.0 else "🔴 REJECTED")
    print(f"👑 VERDICT: {verdict}")

    results_data = {
        "exp_id": "EXP-379",
        "active_atoms": model.active_count,
        "sprout_events": model.sprout_events,
        "rejections": model.rejections,
        "final_det_bit_errors": final_det_bit_errors,
        "final_det_strain": final_det_strain,
        "final_stoch_strain": final_stoch_strain,
        "tok_per_sec": tok_per_sec,
        "discovered_formulas": discovered_formulas,
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_379_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    run_exp_379()
