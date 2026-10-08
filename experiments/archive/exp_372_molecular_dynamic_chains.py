"""
EXP-372: Molecular Dynamic Chains with Covalent Routing (MDC-CR)
================================================================
Sovereign Architectural Mandate from Bazilevs & Gema:
"Atoms must bond covalently into molecular chains and cascades, not stack in parallel monolithic sum-clouds."

Theoretical Mapping:
1. Lightweight Mathematical Atoms (Light Elements):
   - Restricts atomic configurations to lean, high-speed structures to prevent the CPU/GPU loop overhead of EXP-371.
   - Atom Type A (Hydrogen-like: 1 Proton, 1 Neutron, 1 Electron) -> Leaky Integrator.
   - Atom Type B (Helium-like: 2 Protons, 0 Neutrons, 2 Electrons) -> Bilinear Gated Unit.
   - Atom Type C (Lithium-like: 2 Protons, 1 Neutron, 2 Electrons) -> Recurrent Attractor.
2. Covalent Bonding & Sequential Molecular Cascades:
   - Rejects parallel summation. Atoms bond sequentially to form a Molecular Chain.
   - The input to Atom_k is the output of the preceding Atom_{k-1} in the molecular chain:
     v_0 = h_in
     v_k = Atom_k(v_{k-1}, state_{k-1})
   - This creates deep, non-linear compositional pathways (DNA-like chains) where representation
     grows exponentially with chain depth.
3. Epigenetic Net2Net Grafting:
   - Newly sprouted atoms are appended to the end of the molecular chain.
   - An Epigenetic Gate (alpha_epi = 0.0 at birth) wraps the new atom, ensuring that
     v_new = (1 - Gate) * v_{k-1} + Gate * Atom_k(v_{k-1})
     At birth, Gate = 0.0, which guarantees f_new(x) === f_old(x) with strict zero-loss jump!
4. Continuous Bipolar Reality Stream (t -> t+1, N=1, Single Pass) in R^8.
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
# LIGHTWEIGHT MATHEMATICAL ATOMS
# =============================================================================

class LightweightAtom(nn.Module):
    def __init__(self, atom_type: str, dim: int, device: torch.device):
        super().__init__()
        self.atom_type = atom_type
        self.dim = dim
        self.device = device

        if atom_type == "hydrogen":
            # 1 Proton, 1 Neutron, 1 Electron -> Leaky Integrator
            self.W = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
            self.b = nn.Parameter(torch.zeros(dim, device=device))
            self.log_tau = nn.Parameter(torch.full((dim,), -1.0, device=device))
        elif atom_type == "helium":
            # 2 Protons, 0 Neutrons, 2 Electrons -> Bilinear Gated Unit
            self.W_a = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
            self.W_b = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        else:
            # Lithium: 2 Protons, 1 Neutron, 2 Electrons -> Recurrent Attractor
            self.W_a = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
            self.W_b = nn.Parameter(torch.randn(dim, dim, device=device) * (0.5 / math.sqrt(dim)))
            self.log_tau = nn.Parameter(torch.full((dim,), -1.0, device=device))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.atom_type == "hydrogen":
            tau = torch.sigmoid(self.log_tau)
            drive = torch.tanh(torch.matmul(self.W, x) + self.b)
            return (1.0 - tau) * state + tau * drive
        elif self.atom_type == "helium":
            a = torch.tanh(torch.matmul(self.W_a, x))
            b = torch.sigmoid(torch.matmul(self.W_b, x))
            return a * b
        else:
            # Lithium
            tau = torch.sigmoid(self.log_tau)
            drive = torch.tanh(torch.matmul(self.W_a, x))
            u = drive + state
            u_norm = u / (torch.norm(u, p=2, dim=-1, keepdim=True) + 1e-5)
            snapped = torch.tanh(torch.matmul(self.W_b, u_norm))
            return (1.0 - tau) * state + tau * snapped


# =============================================================================
# COVALENT MOLECULAR CHAIN ENGINE
# =============================================================================

class CovalentMolecularChainEngine(nn.Module):
    def __init__(self, in_dim: int = 8, dim: int = 64, max_atoms: int = 12, device: torch.device = DEVICE):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.max_atoms = max_atoms
        self.device = device

        # Sensory Gateway projection
        self.W_in = nn.Parameter(torch.randn(dim, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))
        self.b_in = nn.Parameter(torch.zeros(dim, device=device))

        self.atoms = nn.ModuleList()
        self.atom_types = []
        self.alpha_epi = nn.ParameterList()

        # Start with a minimal molecular chain of 2 atoms
        self.active_atoms = 0
        self._add_atom("hydrogen")
        self._add_atom("helium")

        # Readout: forced molecular routing strictly from the terminal active atom in the chain
        self.W_out = nn.Parameter(torch.randn(in_dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_out = nn.Parameter(torch.zeros(in_dim, device=device))

        self.sprout_events = 0

    def _add_atom(self, atom_type: str):
        if len(self.atoms) >= self.max_atoms:
            return False

        atom = LightweightAtom(atom_type, self.dim, self.device)
        self.atoms.append(atom)
        self.atom_types.append(atom_type)

        # Epigenetic gate (Net2Net Smooth Grafting)
        self.alpha_epi.append(nn.Parameter(torch.tensor(0.0, device=self.device)))

        self.active_atoms += 1
        return True

    def sprout_atom(self):
        if self.active_atoms < self.max_atoms:
            types = ["hydrogen", "helium", "lithium"]
            new_type = types[self.active_atoms % len(types)]
            if self._add_atom(new_type):
                self.sprout_events += 1
                return True
        return False

    def forward(self, x_t: torch.Tensor, prev_states: list):
        # Sensory projection
        h_in = torch.tanh(torch.matmul(self.W_in, x_t) + self.b_in)

        current_val = h_in
        next_states = []
        complexity = 0.0

        for i in range(self.active_atoms):
            atom = self.atoms[i]
            st = prev_states[i]

            # Compute atomic transformation
            raw_out = atom(current_val, st)

            # Epigenetic Gate (Net2Net Smooth Grafting):
            # newly sprouted atoms (initialized with Gate=0.0) cleanly pass the preceding value current_val
            gate = torch.tanh(self.alpha_epi[i])
            gated_out = (1.0 - gate) * current_val + gate * raw_out

            current_val = gated_out
            next_states.append(raw_out)
            complexity = complexity + torch.abs(gate)

        # Readout strictly from the terminal covalent node
        y_pred = torch.tanh(torch.matmul(self.W_out, current_val) + self.b_out)

        return y_pred, next_states, complexity

    def extract_symbolic_formulas(self) -> str:
        report = []
        report.append("=== KARYON COVALENT MOLECULAR CHAIN SPECIFICATION ===")
        report.append("  h_in = tanh(W_in * x_t + b_in)")
        for i in range(self.active_atoms):
            gate_val = torch.tanh(self.alpha_epi[i]).item()
            report.append(f"  Atom_{i} ({self.atom_types[i]}): Gate = {gate_val:.4f} | Output = (1 - Gate)*Out_{i-1} + Gate*Atom_{i}(Out_{i-1})")
        report.append("  Readout: y_pred = tanh(W_out * Out_terminal + b_out)")
        return "\n".join(report)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_exp_372():
    print("===============================================================================")
    print("=== KEP EXP-372: MOLECULAR DYNAMIC CHAINS (MDC-CR)                          ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    stream_data, stream_types, raw_bytes = generate_unbroken_stream(stream_length=2048)
    stream_len = stream_data.size(0)

    model = CovalentMolecularChainEngine(in_dim=8, dim=64, max_atoms=12, device=DEVICE)
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

        # Step molecular chain
        y_pred, next_states, complexity = model(cur_vec, states)

        # Detach states
        states = [s.detach() for s in next_states]

        # Physical quadratic strain + Epigenetic complexity penalty
        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        free_energy = prediction_strain + 0.02 * complexity + 0.01 * model.active_atoms

        free_energy.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        rolling_strain = 0.90 * rolling_strain + 0.10 * strain_val

        # Autopoietic sprouting: expand chain if strain > 0.75
        if t > 40 and t % 32 == 0:
            if rolling_strain > 0.75:
                if model.sprout_atom():
                    states.append(torch.zeros(model.dim, device=DEVICE))
                    print(f"  [MORPHOGENESIS] Sprouted Atom {model.active_atoms-1} ({model.atom_types[-1]}) in molecular chain at step {t+1}. Length: {model.active_atoms}")
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
    print("=== FINAL TELEMETRY & MOLECULAR DISCOVERED SYSTEM (EXP-372) ===")
    print("===============================================================================")
    print(f"  Final Molecular Chain Length : {model.active_atoms} (Started at 2)")
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
        "exp_id": "EXP-372",
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
    with open("experiments/exp_372_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    run_exp_372()
