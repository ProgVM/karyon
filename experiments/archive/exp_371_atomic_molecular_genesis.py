"""
EXP-371: Atomic-Molecular Cybernetic Genesis (AMCG)
====================================================
Sovereign Architectural Mandate from Bazilevs & Gema:
"Represent the cognitive substrate as a dynamic chemical system of Mathematical Atoms and Molecules.
AtoMs are composed of subatomic particles (Protons, Neutrons, Electrons) which dynamically determine
their mathematical structures, equations, and dimensions. Atoms bond covalently to form complex
Molecules (computational graphs), which self-evolve under Free Energy pressure."

Theoretical Mapping:
1. Subatomic Particles:
   - Protons (P): Drive linear projections and coordinate transformations (W * x + b).
   - Neutrons (N): Provide memory, leaky integration, and temporal stabilization (decay gates 1-tau).
   - Electrons (E): Determine non-linear phase transitions and multiplicative gates (tanh, sigmoid, sin, cos).
2. Mathematical Atoms (Elements):
   - Defined by their particle count (P, N, E).
   - A dynamic compiler instantiates the corresponding PyTorch equations on-the-fly.
   - Example: Element 1 (Hydrogen: 1P, 1N, 1E) -> Leaky Integrator.
   - Example: Element 2 (Helium: 2P, 0N, 2E) -> Bilinear Gated Unit.
   - Example: Element 6 (Carbon: 4P, 2N, 4E) -> Multi-Timescale Recurrent Attractor.
3. Covalent Bonding & Molecular Graphs:
   - Atoms share representation fields (electrons) through a differentiable Bonding Matrix.
   - Complex Molecules (e.g. DNA chains, metabolic rings) emerge as self-organized recurrent sub-graphs.
4. Epigenetic Morphogenesis & Net2Net Smooth Grafting:
   - New atoms are sprouted when prediction strain exceeds the homeostatic threshold.
   - Newly sprouted atoms are bound into the molecular network with an Epigenetic Gate (alpha_epi = 0.0)
     to guarantee strict zero-shock function identity at birth.
5. Continuous Bipolar Reality Stream (t -> t+1, N=1, Single Pass) in R^8.
====================================================================================
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
# MATHEMATICAL ATOM: DYNAMIC COMPILER ON SUBATOMIC CONFIGURATIONS
# =============================================================================

class MathematicalAtom(nn.Module):
    def __init__(self, atom_id: int, num_protons: int, num_neutrons: int, num_electrons: int, dim: int, device: torch.device):
        super().__init__()
        self.atom_id = atom_id
        self.num_protons = max(1, num_protons)
        self.num_neutrons = max(0, num_neutrons)
        self.num_electrons = max(1, num_electrons)
        self.dim = dim
        self.device = device

        # Linear projections (Protons)
        self.projections = nn.ParameterList()
        for _ in range(self.num_protons):
            self.projections.append(nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim))))

        # Leaky decay factors (Neutrons)
        self.decays = nn.ParameterList()
        for _ in range(self.num_neutrons):
            self.decays.append(nn.Parameter(torch.full((dim,), -1.0, device=device)))

        # Nonlinear activations & gates (Electrons)
        self.gate_weights = nn.ParameterList()
        for _ in range(self.num_electrons):
            self.gate_weights.append(nn.Parameter(torch.randn(dim, dim, device=device) * (0.5 / math.sqrt(dim))))

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        # x: [dim], prev_state: [dim]
        # 1. Evaluate Protons (Linear spaces)
        proton_fields = []
        for p_mat in self.projections:
            proton_fields.append(torch.matmul(p_mat, x))

        # Combine proton fields
        combined_proton = sum(proton_fields) / len(proton_fields)

        # 2. Evaluate Electrons (Nonlinear gates and activations)
        electron_fields = []
        for i, e_mat in enumerate(self.gate_weights):
            gated = torch.matmul(e_mat, combined_proton)
            if i % 3 == 0:
                electron_fields.append(torch.tanh(gated))
            elif i % 3 == 1:
                electron_fields.append(torch.sigmoid(gated))
            else:
                electron_fields.append(torch.sin(gated))

        combined_electron = sum(electron_fields) / len(electron_fields)

        # 3. Evaluate Neutrons (Memory Leaky Integration)
        if self.num_neutrons > 0:
            decay_fields = []
            for n_dec in self.decays:
                tau = torch.sigmoid(n_dec)
                decay_fields.append((1.0 - tau) * prev_state + tau * combined_electron)
            final_out = sum(decay_fields) / len(decay_fields)
        else:
            final_out = combined_electron

        return final_out


# =============================================================================
# COGNITIVE MOLECULE: ATOMIC-MOLECULAR CYBERNETIC GENESIS ENGINE
# =============================================================================

class AtomicMolecularGenesisEngine(nn.Module):
    def __init__(self, in_dim: int = 8, dim: int = 64, max_atoms: int = 12, device: torch.device = DEVICE):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.max_atoms = max_atoms
        self.device = device

        # Input projection (Sensory Gateway)
        self.W_in = nn.Parameter(torch.randn(dim, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))
        self.b_in = nn.Parameter(torch.zeros(dim, device=device))

        self.atoms = nn.ModuleList()
        self.atom_configs = []  # List of (P, N, E)
        self.alpha_epi = nn.ParameterList()

        # Start with a simple periodic table:
        # Atom 0: Hydrogen (1P, 1N, 1E) -> Leaky Integrator
        # Atom 1: Helium (2P, 0N, 2E) -> Bilinear Gated Unit
        self.active_atoms = 0
        self._add_atom(num_protons=1, num_neutrons=1, num_electrons=1)
        self._add_atom(num_protons=2, num_neutrons=0, num_electrons=2)

        # Readout: forced molecular routing strictly from active atoms
        self.readout_weights = nn.Parameter(torch.randn(in_dim, max_atoms, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.readout_bias = nn.Parameter(torch.zeros(in_dim, device=device))

        self.sprout_events = 0

    def _add_atom(self, num_protons: int, num_neutrons: int, num_electrons: int):
        if len(self.atoms) >= self.max_atoms:
            return False

        atom = MathematicalAtom(
            atom_id=self.active_atoms,
            num_protons=num_protons,
            num_neutrons=num_neutrons,
            num_electrons=num_electrons,
            dim=self.dim,
            device=self.device
        )
        self.atoms.append(atom)
        self.atom_configs.append((num_protons, num_neutrons, num_electrons))

        # Epigenetic Gate (Net2Net Smooth Grafting)
        self.alpha_epi.append(nn.Parameter(torch.tensor(0.0, device=self.device)))

        self.active_atoms += 1
        return True

    def sprout_atom(self):
        if self.active_atoms < self.max_atoms:
            # Dynamically evolve heavier elements:
            # Carbon (4P, 2N, 4E), Oxygen (6P, 2N, 6E), Iron (8P, 4N, 8E)...
            p = 2 + (self.active_atoms * 2)
            n = 1 + (self.active_atoms // 2)
            e = 2 + (self.active_atoms * 2)
            if self._add_atom(num_protons=p, num_neutrons=n, num_electrons=e):
                self.sprout_events += 1
                return True
        return False

    def forward(self, x_t: torch.Tensor, prev_states: list):
        # x_t: [8], prev_states: list of [64]
        h_in = torch.tanh(torch.matmul(self.W_in, x_t) + self.b_in)

        atom_outputs = []
        next_states = []

        for i in range(self.active_atoms):
            atom = self.atoms[i]
            st = prev_states[i]

            raw_out = atom(h_in, st)

            # Epigenetic gating
            gate = torch.tanh(self.alpha_epi[i])
            gated_out = gate * raw_out

            atom_outputs.append(gated_out)
            next_states.append(raw_out)

        # Forced Molecular Routing Readout
        y_pred = torch.zeros(self.in_dim, device=self.device)
        for i in range(self.in_dim):
            val = 0.0
            for j in range(self.active_atoms):
                val = val + torch.sum(self.readout_weights[i, j] * atom_outputs[j])
            y_pred[i] = torch.tanh(val + self.readout_bias[i])

        # Complexity penalty
        complexity = 0.0
        for i in range(self.active_atoms):
            complexity = complexity + torch.abs(torch.tanh(self.alpha_epi[i]))

        return y_pred, next_states, complexity

    def extract_symbolic_formulas(self) -> str:
        report = []
        report.append("=== KARYON PERIODIC TABLE & MOLECULAR SPECIFICATION ===")
        for i in range(self.active_atoms):
            p, n, e = self.atom_configs[i]
            gate_val = torch.tanh(self.alpha_epi[i]).item()
            report.append(f"  [Atom {i}]: Config = ({p} Protons, {n} Neutrons, {e} Electrons) | Epigenetic Gate = {gate_val:.4f}")
        report.append(f"  Molecular Readout: y_pred = tanh( sum_{{i=0}}^{{{self.active_atoms-1}}} W_readout_i * (Gate_i * Atom_i(h_in)) )")
        return "\n".join(report)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_exp_371():
    print("===============================================================================")
    print("=== KEP EXP-371: ATOMIC-MOLECULAR CYBERNETIC GENESIS (AMCG)                 ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    stream_data, stream_types, raw_bytes = generate_unbroken_stream(stream_length=2048)
    stream_len = stream_data.size(0)

    model = AtomicMolecularGenesisEngine(in_dim=8, dim=64, max_atoms=12, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-2, weight_decay=1e-5)

    states = [torch.zeros(model.dim, device=DEVICE) for _ in range(model.active_atoms)]

    det_errors = []
    stoch_errors = []
    bit_mismatches_det = []

    rolling_strain = 2.0
    t0 = time.time()

    for t in range(stream_len - 1):
        cur_vec = stream_data[t]
        tgt_vec = stream_data[t + 1]
        is_stochastic = stream_types[t].item()

        optimizer.zero_grad()

        # Step atomic-molecular engine
        y_pred, next_states, complexity = model(cur_vec, states)

        # Detach states for next step
        states = [s.detach() for s in next_states]

        # Physical quadratic strain + Epigenetic complexity penalty
        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        free_energy = prediction_strain + 0.02 * complexity + 0.01 * model.active_atoms

        free_energy.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        rolling_strain = 0.90 * rolling_strain + 0.10 * strain_val

        # Autopoietic sprouting: evolve heavier elements if strain > 0.75
        if t > 40 and t % 32 == 0:
            if rolling_strain > 0.75:
                if model.sprout_atom():
                    states.append(torch.zeros(model.dim, device=DEVICE))
                    p, n, e = model.atom_configs[-1]
                    print(f"  [MORPHOGENESIS] Sprouted heavier Atom {model.active_atoms-1} ({p}P, {n}N, {e}E) at step {t+1}. Active: {model.active_atoms}")
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-2, weight_decay=1e-5)

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
            print(f"  Step {t+1:04d}/{stream_len} | Active Atoms: {model.active_atoms:02d} | Strain: {rolling_strain:.4f} | Det Energy: {det_tail_err:.4f} | Det Bit Errors: {det_tail_bits:.2f}/8 bits")

    elapsed = time.time() - t0
    tok_per_sec = stream_len / elapsed if elapsed > 0 else 0.0

    q_len = max(1, len(det_errors) // 4)
    final_det_strain = float(np.mean(det_errors[-q_len:]))
    final_det_bit_errors = float(np.mean(bit_mismatches_det[-q_len:]))
    final_stoch_strain = float(np.mean(stoch_errors[-q_len:]))

    discovered_formulas = model.extract_symbolic_formulas()

    print("\n===============================================================================")
    print("=== FINAL TELEMETRY & AUTOPOIETIC DISCOVERED SYSTEM (EXP-371) ===")
    print("===============================================================================")
    print(f"  Final Active Atom Count      : {model.active_atoms} (Started at 2)")
    print(f"  Sprout Morphogenesis Events  : {model.sprout_events}")
    print(f"  Deterministic Bit Error Rate : {final_det_bit_errors:.2f} / 8 bits")
    print(f"  Deterministic Strain Energy  : {final_det_strain:.4f}")
    print(f"  Stochastic Strain Energy     : {final_stoch_strain:.4f}")
    print(f"  Processing Throughput        : {tok_per_sec:.1f} tok/s\n")
    print("-------------------------------------------------------------------------------")
    print(discovered_formulas)
    print("-------------------------------------------------------------------------------")

    # KEP Rule #2 Verdict
    verdict = "🟢 POSITIVE" if final_det_bit_errors < 1.0 else ("⚪ NEUTRAL" if final_det_bit_errors < 2.0 else "🔴 REJECTED")
    print(f"👑 VERDICT: {verdict}")

    results_data = {
        "exp_id": "EXP-371",
        "active_atoms": model.active_atoms,
        "sprout_events": model.sprout_events,
        "final_det_bit_errors": final_det_bit_errors,
        "final_det_strain": final_det_strain,
        "final_stoch_strain": final_stoch_strain,
        "tok_per_sec": tok_per_sec,
        "discovered_formulas": discovered_formulas,
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_371_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    run_exp_371()
