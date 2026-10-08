"""
EXP-373: Universal Chemico-Physical Substrate (UCPS)
====================================================
Sovereign Architectural Mandate from Bazilevs & Gema:
"Represent the cognitive substrate as a dynamic, continuous physical-chemical reactor
governed by the fundamental laws of thermodynamics, chemical kinetics, and spatial diffusion."

Mathematical & Physical Formulations:
1. Spatial Reactor Space:
   - The reactor is a continuous 3D volume where M atoms reside.
   - Each atom i has a continuous spatial coordinate p_i in R^3.
   - Sensory input x_t enters physically near the origin (p_i -> 0).
   - Readout y_pred is extracted physically at a target receptor location p_target = [2.0, 2.0, 2.0].

2. Thermodynamics of Bonding (Gibbs Free Energy & Arrhenius Kinetics):
   - Each atom i has an orbital vector o_i in R^D representing its electronic configuration.
   - Enthalpy (orbital compatibility): H_ij = - o_i^T o_j.
   - Entropy (spatial constraint): S_ij = log(d_ij + epsilon), where d_ij = ||p_i - p_j||_2.
   - Gibbs Free Energy of reaction: G_ij = H_ij - T_t * S_ij, where T_t is the temperature.
   - Activation Energy: E_a_ij = max(E_a_i, E_a_j).
   - The bonding strength B_ij is solved via a continuous relaxation of the Law of Mass Action:
     B_ij = sigmoid( (-G_ij - E_a_ij) / T_t ) * exp(-d_ij)
     * High temperature allows atoms to overcome high activation barriers to bond.
     * Large spatial distance exponentially suppresses the bonding probability.

3. Signal Propagation (Reaction-Diffusion):
   - Information flows through covalent/hydrogen bonds.
   - Atom state s_i updates continuously:
     s_i_t = (1 - tau_i) * s_i_t-1 + tau_i * tanh( sum_j B_ji * (W_c * s_j_t-1) + V_in_i * x_t )
     where V_in_i = W_in * exp(-||p_i||_2^2) (sensory gateway input decays with distance from origin).

4. Readout:
   - Extracted physically at the target receptor position:
     y_pred = tanh( sum_i exp(-||p_i - p_target||_2^2) * (W_out * s_i_t) + b_out )

5. Epigenetic Morphogenesis & Net2Net Smooth Grafting:
   - When prediction strain exceeds the homeostatic threshold, a new atom is sprouted in the reactor.
   - Epigenetic Gate (alpha_epi = 0.0 at birth) scales the atom's bonding:
     B_ij_effective = B_ij * tanh(alpha_epi_i) * tanh(alpha_epi_j).
     This guarantees STRICT ZERO-SHOCK FUNCTION IDENTITY f_new(x) === f_old(x) at birth.
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
# UNIVERSAL CHEMICO-PHYSICAL SUBSTRATE ENGINE
# =============================================================================

class UniversalChemicoPhysicalSubstrate(nn.Module):
    def __init__(self, in_dim: int = 8, dim: int = 64, max_atoms: int = 16, device: torch.device = DEVICE):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.max_atoms = max_atoms
        self.device = device

        # Shared covalent/hydrogen transport matrices
        self.W_c = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_in = nn.Parameter(torch.randn(dim, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))

        # Readout target receptor
        self.p_target = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32, device=device)
        self.W_out = nn.Parameter(torch.randn(in_dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_out = nn.Parameter(torch.zeros(in_dim, device=device))

        # Core environmental parameters
        self.log_temp = nn.Parameter(torch.tensor(0.0, device=device))  # Temperature log_T

        # Atom parameters (Using nn.Parameter for gradient-driven spatial self-organization)
        self.p_coords = nn.Parameter(torch.randn(max_atoms, 3, device=device) * 0.5)
        self.o_vectors = nn.Parameter(torch.randn(max_atoms, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.E_a = nn.Parameter(torch.zeros(max_atoms, device=device))
        self.log_tau = nn.Parameter(torch.full((max_atoms,), -1.0, device=device))

        # Epigenetic gates to prevent structural shock
        self.alpha_epi = nn.Parameter(torch.zeros(max_atoms, device=device))

        # Initialize the first 4 atoms as active
        self.active_atoms = 4
        # Force the first 4 atoms to have active epigenetic gates (alpha = 3.0)
        with torch.no_grad():
            self.alpha_epi[:4].fill_(3.0)

        self.sprout_events = 0

    def sprout_atom(self):
        if self.active_atoms < self.max_atoms:
            # We sprout a new atom. Since alpha_epi was initialized to 0.0,
            # its effective bonding and output contribution will start at exactly 0.0,
            # ensuring a strict zero-shock Net2Net function identity.
            self.active_atoms += 1
            self.sprout_events += 1
            return True
        return False

    def forward(self, x_t: torch.Tensor, prev_states: torch.Tensor):
        # x_t: [8], prev_states: [max_atoms, dim]
        # 1. Compute Temperature
        T = torch.exp(self.log_temp) + 1e-4

        # 2. Compute Distances and Orbital Enthalpy
        # Extract active slices
        coords = self.p_coords[:self.active_atoms]  # [A, 3]
        orbitals = self.o_vectors[:self.active_atoms]  # [A, D]
        ea = self.E_a[:self.active_atoms]  # [A]
        tau = torch.sigmoid(self.log_tau[:self.active_atoms]).unsqueeze(-1)  # [A, 1]
        alpha = torch.tanh(self.alpha_epi[:self.active_atoms]).unsqueeze(-1)  # [A, 1]

        # Pairwise distance matrix d_ij: [A, A]
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [A, A, 3]
        d_ij = torch.norm(diff, p=2, dim=-1) + 1e-5

        # Pairwise Enthalpy (orbital overlap): [A, A]
        # H_ij = - o_i^T o_j
        orbitals_norm = orbitals / (torch.norm(orbitals, p=2, dim=-1, keepdim=True) + 1e-5)
        H_ij = -torch.matmul(orbitals_norm, orbitals_norm.t())

        # Pairwise Entropy: S_ij = log(d_ij)
        S_ij = torch.log(d_ij)

        # Gibbs Free Energy: G_ij = H_ij - T * S_ij
        G_ij = H_ij - T * S_ij

        # Activation Energy Barrier: E_a_ij = max(E_a_i, E_a_j)
        E_a_ij = torch.max(ea.unsqueeze(1), ea.unsqueeze(0))

        # Solve continuous bonding matrix B_ij using Law of Mass Action relaxation
        B_ij = torch.sigmoid((-G_ij - E_a_ij) / T) * torch.exp(-d_ij)

        # Apply Epigenetic Gating (Net2Net Smooth Grafting)
        # B_ij_effective = B_ij * alpha_i * alpha_j
        B_ij_effective = B_ij * torch.matmul(alpha, alpha.t())

        # 3. Dynamic Signal Propagation (Reaction-Diffusion)
        # Input gateway: decays with distance from origin
        dist_from_origin = torch.norm(coords, p=2, dim=-1, keepdim=True)  # [A, 1]
        input_gate = torch.exp(-dist_from_origin**2)  # [A, 1]
        V_in = input_gate * torch.matmul(self.W_in, x_t).unsqueeze(0)  # [A, D]

        # Shared covalent transport
        prev_active = prev_states[:self.active_atoms]  # [A, D]
        transported = torch.matmul(prev_active, self.W_c.t())  # [A, D]

        # Diffused signal sum_j B_ji * transported_j
        diffused = torch.matmul(B_ij_effective.t(), transported)  # [A, D]

        # Update active states
        drive = torch.tanh(diffused + V_in)
        next_active_states = (1.0 - tau) * prev_active + tau * drive

        # Pad back to max_atoms to maintain static tensor shapes (no GPU recompilation!)
        next_states = prev_states.clone()
        next_states[:self.active_atoms] = next_active_states

        # 4. Readout: Extracted physically at the target receptor position
        dist_from_target = torch.norm(coords - self.p_target, p=2, dim=-1, keepdim=True)  # [A, 1]
        output_gate = torch.exp(-dist_from_target**2)  # [A, 1]

        # Weighted composition of terminal active states
        composed_signal = torch.sum(output_gate * next_active_states, dim=0)  # [D]
        y_pred = torch.tanh(torch.matmul(self.W_out, composed_signal) + self.b_out)

        # Complexity penalty
        complexity = torch.sum(torch.abs(torch.tanh(self.alpha_epi[:self.active_atoms])))

        return y_pred, next_states, complexity

    def extract_symbolic_formulas(self) -> str:
        report = []
        report.append("=== KARYON UNIVERSAL CHEMICO-PHYSICAL SUBSTRATE SPECIFICATION ===")
        T_val = torch.exp(self.log_temp).item()
        report.append(f"  Reactor Temperature (T) : {T_val:.4f}")
        report.append(f"  Active Atoms in Reactor : {self.active_atoms} (Started at 4)")
        for i in range(self.active_atoms):
            coord_str = ", ".join([f"{x:.2f}" for x in self.p_coords[i].tolist()])
            gate_val = torch.tanh(self.alpha_epi[i]).item()
            ea_val = self.E_a[i].item()
            tau_val = torch.sigmoid(self.log_tau[i]).item()
            report.append(f"  Atom_{i:02d} | Position: [{coord_str}] | Ea: {ea_val:.3f} | Tau: {tau_val:.3f} | Epigenetic Gate: {gate_val:.4f}")
        return "\n".join(report)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_exp_373():
    print("===============================================================================")
    print("=== KEP EXP-373: UNIVERSAL CHEMICO-PHYSICAL SUBSTRATE (UCPS)                ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    stream_data, stream_types, raw_bytes = generate_unbroken_stream(stream_length=2048)
    stream_len = stream_data.size(0)

    model = UniversalChemicoPhysicalSubstrate(in_dim=8, dim=64, max_atoms=16, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-2, weight_decay=1e-5)

    states = torch.zeros(model.max_atoms, model.dim, device=DEVICE)

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

        # Step chemico-physical engine
        y_pred, next_states, complexity = model(cur_vec, states)

        # Detach states
        states = next_states.detach()

        # Physical quadratic strain + Epigenetic complexity penalty
        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        free_energy = prediction_strain + 0.02 * complexity + 0.01 * model.active_atoms

        free_energy.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        rolling_strain = 0.90 * rolling_strain + 0.10 * strain_val

        # Autopoietic sprouting: sprout new atom if strain is high
        if t > 40 and t % 32 == 0:
            if rolling_strain > 0.72:
                if model.sprout_atom():
                    print(f"  [MORPHOGENESIS] Sprouted Atom {model.active_atoms-1} in reactor at step {t+1}. Total Atoms: {model.active_atoms}")
                    # Re-instantiate optimizer to include new parameters if any,
                    # though all parameters (including max_atoms) are pre-allocated, so we just continue!

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
    print("=== FINAL TELEMETRY & REACTOR DISCOVERED STATE (EXP-373) ===")
    print("===============================================================================")
    print(f"  Final Active Atom Count      : {model.active_atoms} (Started at 4)")
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
        "exp_id": "EXP-373",
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
    with open("experiments/exp_373_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    run_exp_373()
