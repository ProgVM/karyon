"""
EXP-375: Quantum Observer Effect & Wave-Function Collapse (QOE-WFC)
===================================================================
Sovereign Architectural Mandate from Bazilevs & Gema:
"Introduce the Quantum Observer Effect into the continuous autopoietic substrate:
Unobserved internal states evolve as continuous wave superpositions across synthesized laws.
When an observation occurs (readout/external environment request), the interaction laws and hidden states
collapse into sharp discrete attractors, eradicating law jitter and micro-phase noise."

Theoretical Formulation:
1. Dual-Phase Substrate Dynamics:
   - Unobserved Internal Evolution (High Entropic Freedom, Omega_t -> 0):
     * Interaction law logits exist in a continuous superposed mixture over the 4 basis forces.
     * Hidden states s_i evolve smoothly as fluid vector fields.
   - Observed Measurement Event (Wave Collapse, Omega_t -> 1):
     * The observation intensity Omega_t is dynamically computed:
       Omega_t = sigmoid( W_obs * Mean(s_t) + b_obs )
     * Law Temperature Scaling:
       T_eff = T_base * (1.0 - 0.95 * Omega_t) + 1e-4
       lambda_ij = Softmax( law_logits_ij / T_eff )
       As Omega_t -> 1, T_eff -> 0, causing the law distribution lambda_ij to collapse argmax-like
       into a single deterministic physical law.
     * State Attraction Collapse:
       s_i_collapsed = (1 - Omega_t) * s_i + Omega_t * tanh( beta_collapse * s_i )

2. Epigenetic Net2Net Smooth Grafting:
   - Sprouted units start with alpha_epi = 0.0, scaling their interaction and readout fields to 0.0,
     guaranteeing 100% zero-shock function identity at birth.

3. Continuous Bipolar Reality Stream (t -> t+1, N=1, Single Pass) in R^8.
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
# OBSERVER EFFECT & WAVE-FUNCTION COLLAPSE ENGINE
# =============================================================================

class ObserverEffectEngine(nn.Module):
    def __init__(self, in_dim: int = 8, dim: int = 64, max_units: int = 16, device: torch.device = DEVICE):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.max_units = max_units
        self.device = device

        # Input and output gateways
        self.W_in = nn.Parameter(torch.randn(max_units, dim, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))
        self.W_out = nn.Parameter(torch.randn(in_dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_out = nn.Parameter(torch.zeros(in_dim, device=device))

        # Dynamic Observer Projection (Computes Observation Intensity \Omega_t in [0, 1])
        self.W_obs = nn.Parameter(torch.randn(dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_obs = nn.Parameter(torch.zeros(1, device=device))

        # Dynamic Law Projector
        self.W_law = nn.Parameter(torch.randn(4, dim * 2, device=device) * (1.0 / math.sqrt(dim * 2)))
        self.b_law = nn.Parameter(torch.zeros(4, device=device))

        # Shared Force Projection matrices
        self.W_force0 = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_force1_a = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_force1_b = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_force2 = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_force3 = nn.Parameter(torch.randn(dim, dim, device=device) * (0.5 / math.sqrt(dim)))

        # Integration decay times
        self.log_tau = nn.Parameter(torch.full((max_units,), -1.0, device=device))

        # Epigenetic gates
        self.alpha_epi = nn.Parameter(torch.zeros(max_units, device=device))

        # Start with 4 active units
        self.active_units = 4
        with torch.no_grad():
            self.alpha_epi[:4].fill_(3.0)

        self.sprout_events = 0

    def sprout_unit(self):
        if self.active_units < self.max_units:
            self.active_units += 1
            self.sprout_events += 1
            return True
        return False

    def forward(self, x_t: torch.Tensor, prev_states: torch.Tensor):
        # x_t: [8], prev_states: [max_units, dim]
        # Extract active slices
        states = prev_states[:self.active_units]  # [A, D]
        tau = torch.sigmoid(self.log_tau[:self.active_units]).unsqueeze(-1)  # [A, 1]
        alpha = torch.tanh(self.alpha_epi[:self.active_units]).unsqueeze(-1)  # [A, 1]

        A = self.active_units

        # 1. Compute Dynamic Observation Intensity \Omega_t
        mean_state = torch.mean(states * alpha, dim=0)  # [D]
        omega_t = torch.sigmoid(torch.sum(mean_state * self.W_obs) + self.b_obs)  # Scalar in [0, 1]

        # 2. Compute Pairwise Law Logits & Apply Quantum Law Collapse
        states_i = states.unsqueeze(1).expand(-1, A, -1)  # [A, A, D]
        states_j = states.unsqueeze(0).expand(A, -1, -1)  # [A, A, D]
        pairs = torch.cat([states_i, states_j], dim=-1)  # [A, A, D*2]

        law_logits = torch.matmul(pairs, self.W_law.t()) + self.b_law  # [A, A, 4]

        # Effective Law Temperature shrinks as \Omega_t -> 1 (Law Collapse)
        T_base = 1.0
        T_eff = T_base * (1.0 - 0.95 * omega_t) + 1e-4
        laws = torch.softmax(law_logits / T_eff, dim=-1)  # [A, A, 4]

        # 3. Compute Basis Field Forces
        f0 = torch.matmul(states, self.W_force0.t()).unsqueeze(0).expand(A, -1, -1)  # [A, A, D]

        f1_a = torch.tanh(torch.matmul(states, self.W_force1_a.t()))
        f1_b = torch.sigmoid(torch.matmul(states, self.W_force1_b.t()))
        f1 = (f1_a.unsqueeze(0) * f1_b.unsqueeze(1))  # [A, A, D]

        f2_raw = torch.matmul(states, self.W_force2.t())
        f2 = torch.sin(f2_raw).unsqueeze(0).expand(A, -1, -1)  # [A, A, D]

        f3_raw = torch.matmul(states, self.W_force3.t())
        f3_norm = f3_raw / (torch.norm(f3_raw, p=2, dim=-1, keepdim=True) + 1e-5)
        f3 = torch.tanh(f3_norm).unsqueeze(0).expand(A, -1, -1)  # [A, A, D]

        forces = torch.stack([f0, f1, f2, f3], dim=2)  # [A, A, 4, D]

        # 4. Apply Synthesized Laws
        synthesized_fields = torch.sum(laws.unsqueeze(-1) * forces, dim=2)  # [A, A, D]

        # Net2Net Epigenetic Gating
        gated_fields = synthesized_fields * torch.matmul(alpha, alpha.t()).unsqueeze(-1)  # [A, A, D]
        total_fields = torch.sum(gated_fields, dim=1)  # [A, D]

        # Input gateway
        V_in = torch.squeeze(torch.matmul(self.W_in[:A], x_t.unsqueeze(-1)), -1)  # [A, D]

        # Update active states
        drive = torch.tanh(total_fields + V_in)
        next_active_states = (1.0 - tau) * states + tau * drive

        # 5. Apply Quantum State Collapse (State Attraction)
        # As \Omega_t -> 1, states collapse towards sharp bipolar attractors (\tanh(\beta * s))
        beta_collapse = 2.5
        collapsed_states = (1.0 - omega_t) * next_active_states + omega_t * torch.tanh(beta_collapse * next_active_states)

        # Pad back to max_units
        next_states = prev_states.clone()
        next_states[:self.active_units] = collapsed_states

        # 6. Readout
        ensemble_signal = torch.sum(alpha * collapsed_states, dim=0)  # [D]
        y_pred = torch.tanh(torch.matmul(self.W_out, ensemble_signal) + self.b_out)

        complexity = torch.sum(torch.abs(torch.tanh(self.alpha_epi[:self.active_units])))

        return y_pred, next_states, complexity, laws, omega_t

    def extract_symbolic_formulas(self, last_laws: torch.Tensor, last_omega: torch.Tensor) -> str:
        report = []
        report.append("=== KARYON QUANTUM OBSERVER EFFECT & WAVE COLLAPSE SPECIFICATION ===")
        report.append(f"  Active Units in Substrate : {self.active_units} (Started at 4)")
        report.append(f"  Observation Intensity (Omega_t) : {last_omega.item():.4f} (0=Fluid Wave, 1=Sharp Collapse)")

        mean_laws = torch.mean(last_laws, dim=(0, 1)).tolist()
        report.append("  Collapsed Global Law Distribution:")
        report.append(f"    - Force 0 (Inertial Memory)     : {mean_laws[0]*100:.1f}%")
        report.append(f"    - Force 1 (Bilinear Resonance)  : {mean_laws[1]*100:.1f}%")
        report.append(f"    - Force 2 (Harmonic Vibration)   : {mean_laws[2]*100:.1f}%")
        report.append(f"    - Force 3 (Attractor Snapping)  : {mean_laws[3]*100:.1f}%")

        report.append("\n  Individual Unit Collapsed Preferences:")
        for i in range(self.active_units):
            unit_laws = torch.mean(last_laws[:, i, :], dim=0).tolist()
            gate_val = torch.tanh(self.alpha_epi[i]).item()
            report.append(f"    Unit_{i:02d} | Epigenetic Gate: {gate_val:.4f} | Collapsed Laws: [M:{unit_laws[0]:.2f}, R:{unit_laws[1]:.2f}, V:{unit_laws[2]:.2f}, A:{unit_laws[3]:.2f}]")
        return "\n".join(report)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_exp_375():
    print("===============================================================================")
    print("=== KEP EXP-375: QUANTUM OBSERVER EFFECT & WAVE COLLAPSE (QOE-WFC)          ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    stream_data, stream_types, raw_bytes = generate_unbroken_stream(stream_length=2048)
    stream_len = stream_data.size(0)

    model = ObserverEffectEngine(in_dim=8, dim=64, max_units=16, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-2, weight_decay=1e-5)

    states = torch.zeros(model.max_units, model.dim, device=DEVICE)

    det_errors = []
    stoch_errors = []
    bit_mismatches_det = []

    rolling_strain = 2.0
    t0 = time.time()
    last_laws_tensor = None
    last_omega_tensor = None

    for t in range(stream_len - 1):
        cur_vec = stream_data[t]
        tgt_vec = stream_data[t + 1]
        is_stochastic = stream_types[t].item()

        optimizer.zero_grad()

        # Step quantum observer engine
        y_pred, next_states, complexity, laws, omega_t = model(cur_vec, states)
        last_laws_tensor = laws.detach()
        last_omega_tensor = omega_t.detach()

        # Detach states
        states = next_states.detach()

        # Physical quadratic strain + Epigenetic complexity penalty
        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        free_energy = prediction_strain + 0.02 * complexity + 0.01 * model.active_units

        free_energy.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        rolling_strain = 0.90 * rolling_strain + 0.10 * strain_val

        # Autopoietic sprouting: sprout new unit if prediction strain is high
        if t > 40 and t % 32 == 0:
            if rolling_strain > 0.70:
                if model.sprout_unit():
                    print(f"  [MORPHOGENESIS] Sprouted Unit {model.active_units-1} in substrate at step {t+1}. Total Units: {model.active_units}")

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
            print(f"  Step {t+1:04d}/{stream_len} | Active Units: {model.active_units:02d} | Strain: {rolling_strain:.4f} | Det Energy: {det_tail_err:.4f} | Omega: {omega_t.item():.3f} | Det Bit Errors: {det_tail_bits:.2f}/8 bits")

    elapsed = time.time() - t0
    tok_per_sec = stream_len / elapsed if elapsed > 0 else 0.0

    q_len = max(1, len(det_errors) // 4)
    final_det_strain = float(np.mean(det_errors[-q_len:]))
    final_det_bit_errors = float(np.mean(bit_mismatches_det[-q_len:]))
    final_stoch_strain = float(np.mean(stoch_errors[-q_len:]))

    discovered_formulas = model.extract_symbolic_formulas(last_laws_tensor, last_omega_tensor)

    print("\n===============================================================================")
    print("=== FINAL TELEMETRY & OBSERVER COLLAPSE STATE (EXP-375) ===")
    print("===============================================================================")
    print(f"  Final Active Unit Count      : {model.active_units} (Started at 4)")
    print(f"  Sprout Morphogenesis Events  : {model.sprout_events}")
    print(f"  Final Observation Intensity  : {last_omega_tensor.item():.4f}")
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
        "exp_id": "EXP-375",
        "active_atoms": model.active_units,
        "sprout_events": model.sprout_events,
        "final_det_bit_errors": final_det_bit_errors,
        "final_det_strain": final_det_strain,
        "final_stoch_strain": final_stoch_strain,
        "tok_per_sec": tok_per_sec,
        "final_omega": last_omega_tensor.item(),
        "discovered_formulas": discovered_formulas,
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_375_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    run_exp_375()
