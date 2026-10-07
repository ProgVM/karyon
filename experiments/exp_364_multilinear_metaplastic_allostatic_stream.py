"""
EXP-364: Multilinear Meta-Plastic Allostatic Continuous Stream Engine (MMP-ACSE)
================================================================================
Mandate from Bazilevs & Empirical Lessons:
1. Zero Softmax / Zero Categorical Probabilities:
   - Operates in physical 8-bit bipolar state space R^8 (b_i in {-1.0, +1.0}).
   - Output is a continuous physical phase-lock. Error is Euclidean strain E(t) = 0.5 * ||y_pred - y_true||^2.
2. Unbroken Single-Pass Continuous Temporal Stream (t -> t+1, N=1):
   - Continuous online learning. No batches, no epoch resets.
3. First-Order Overdamped Differential Dynamics (Zero inertial overshoot):
   - Eradicates second-order momentum overshoot on discrete bit flips.
4. Higher-Order Multilinear Tensor Conjunctions:
   - Causal tensor multiplications (x @ W_a) * (x @ W_b) and (s @ V_a) * (s @ V_b)
     to natively solve non-linear bitwise logic (XOR, bit-shifts, parity) in continuous R^8.
5. Ashby Allostatic Plasticity Gating (Melting vs Crystallization):
   - Plasticity melts when prediction strain E(t) rises (rapid self-reconfiguration).
   - Plasticity crystallizes (eta -> 0) when prediction strain E(t) -> 0 (permanent algorithmic lock-in).

Target:
- Deterministic Bit Error Rate -> 0.00 / 8 bits.
- Deterministic Strain Energy -> 0.00.
================================================================================
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
# CONTINUOUS 8-BIT PHYSICAL STREAM GENERATOR
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


def generate_dual_stream(stream_length: int = 2048):
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
# MULTILINEAR META-PLASTIC CONTINUOUS STREAM ENGINE
# =============================================================================

class MultilinearMetaPlasticEngine(nn.Module):
    def __init__(self, in_dim: int = 8, hidden_dim: int = 64, rank: int = 32, device: torch.device = DEVICE):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.device = device

        # Linear input projections
        self.W_in = nn.Parameter(torch.randn(hidden_dim, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))

        # Higher-Order Multilinear Conjunction Tensor factors (for native XOR / parity continuous solving)
        self.U1 = nn.Parameter(torch.randn(rank, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))
        self.U2 = nn.Parameter(torch.randn(rank, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))
        self.W_conj = nn.Parameter(torch.randn(hidden_dim, rank, device=device) * (1.0 / math.sqrt(rank)))

        # State-State Multilinear Conjunction
        self.V1 = nn.Parameter(torch.randn(rank, hidden_dim, device=device) * (1.0 / math.sqrt(hidden_dim)))
        self.V2 = nn.Parameter(torch.randn(rank, hidden_dim, device=device) * (1.0 / math.sqrt(hidden_dim)))
        self.W_state_conj = nn.Parameter(torch.randn(hidden_dim, rank, device=device) * (1.0 / math.sqrt(rank)))

        # Base recurrent transition matrix
        self.W_base = nn.Parameter(torch.randn(hidden_dim, hidden_dim, device=device) * (0.5 / math.sqrt(hidden_dim)))

        # Meta-Plastic Tensor: continuously self-modifies in stream
        self.W_plastic = nn.Parameter(torch.zeros(hidden_dim, hidden_dim, device=device))

        # Ashby Allostatic Plasticity Hyperparameters
        self.base_plasticity_rate = nn.Parameter(torch.tensor(0.08, device=device))
        self.plasticity_decay = nn.Parameter(torch.tensor(0.990, device=device))
        self.homeostatic_setpoint = nn.Parameter(torch.tensor(0.25, device=device))

        # Continuous relaxation time constant
        self.log_tau = nn.Parameter(torch.tensor(-0.8, device=device))  # tau ~ 0.45

        # Readout to 8-bit physical bipolar attractor
        self.W_out = nn.Parameter(torch.randn(in_dim, hidden_dim, device=device) * (1.0 / math.sqrt(hidden_dim)))

        self.crystallization_events = 0
        self.melting_events = 0

    def forward_step(self, x_t: torch.Tensor, s_prev: torch.Tensor, current_strain: float):
        # x_t: [8], s_prev: [H]
        x = x_t.unsqueeze(0)       # [1, 8]
        s = s_prev.unsqueeze(0)    # [1, H]

        tau = torch.clamp(torch.sigmoid(self.log_tau) * 0.8 + 0.1, 0.1, 0.9)

        # 1. Linear Input Drive
        lin_in = torch.matmul(x, self.W_in.t())  # [1, H]

        # 2. Input Multilinear Conjunction: (x @ U1^T) * (x @ U2^T)
        c_in = torch.matmul(x, self.U1.t()) * torch.matmul(x, self.U2.t())  # [1, rank]
        conj_in = torch.matmul(c_in, self.W_conj.t())  # [1, H]

        # 3. State Multilinear Conjunction: (s @ V1^T) * (s @ V2^T)
        c_state = torch.matmul(s, self.V1.t()) * torch.matmul(s, self.V2.t())  # [1, rank]
        conj_state = torch.matmul(c_state, self.W_state_conj.t())  # [1, H]

        # 4. Effective Evolving Recurrent Operator
        W_eff = self.W_base + self.W_plastic
        rec_flow = torch.matmul(s, W_eff.t())  # [1, H]

        # First-Order Overdamped Differential Flow (No inertia, zero overshoot)
        total_drift = lin_in + conj_in + conj_state + rec_flow
        s_target = torch.tanh(total_drift)
        s_next = (1.0 - tau) * s + tau * s_target

        # Physical 8-bit Phase-Lock Readout
        y_pred = torch.tanh(torch.matmul(s_next, self.W_out.t())).squeeze(0)  # [8]

        # Ashby Allostatic Melting vs Crystallization:
        # If strain > setpoint -> Melting (eta increases to fast adapt)
        # If strain <= setpoint -> Crystallization (eta -> 0, locks in the discovered formula)
        strain_diff = current_strain - self.homeostatic_setpoint.item()
        melt_factor = 1.0 / (1.0 + math.exp(-4.0 * strain_diff))  # Sigmoidal melting curve

        strain_vec = (s_next - s).squeeze(0)  # [H]
        with torch.no_grad():
            if melt_factor > 0.3:
                # System is in plastic melting regime: adapt operator
                dW = torch.outer(strain_vec, s.squeeze(0)) * 0.05
                eff_rate = self.base_plasticity_rate * melt_factor
                self.W_plastic.data = torch.clamp(
                    self.plasticity_decay * self.W_plastic.data + eff_rate * dW,
                    -2.0, 2.0
                )
                self.melting_events += 1
            else:
                # System is crystallized: maintain invariant structure
                self.crystallization_events += 1

        internal_energy = 0.5 * torch.sum(strain_vec**2)
        return y_pred, s_next.squeeze(0), internal_energy


# =============================================================================
# CONTINUOUS STREAM BENCHMARK RUNNER (SINGLE PASS N=1)
# =============================================================================

def run_exp_364_benchmark():
    print("===============================================================================")
    print("=== KEP EXP-364: MULTILINEAR META-PLASTIC ALLOLOSTATIC STREAM ENGINE        ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    stream_data, stream_types, raw_bytes = generate_dual_stream(stream_length=2048)
    stream_len = stream_data.size(0)

    model = MultilinearMetaPlasticEngine(in_dim=8, hidden_dim=64, rank=32, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-2, weight_decay=1e-5)

    s_state = torch.zeros(model.hidden_dim, device=DEVICE)

    det_errors = []
    stoch_errors = []
    bit_mismatches_det = []
    energies_det = []
    energies_stoch = []

    rolling_strain = 1.0
    t0 = time.time()

    for t in range(stream_len - 1):
        cur_vec = stream_data[t]
        tgt_vec = stream_data[t + 1]
        is_stochastic = stream_types[t].item()

        optimizer.zero_grad()

        # Step continuous engine with active allostatic strain
        y_pred, s_state, internal_energy = model.forward_step(cur_vec, s_state, rolling_strain)
        s_state = s_state.detach()

        # Pure physical quadratic strain energy
        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        total_loss = prediction_strain + 0.002 * internal_energy

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        rolling_strain = 0.90 * rolling_strain + 0.10 * strain_val

        # Exact bit-level parity check
        pred_byte = bit_vector_to_byte(y_pred)
        tgt_byte = raw_bytes[t + 1]
        bit_diff = bin(pred_byte ^ tgt_byte).count('1')

        if is_stochastic == 0:
            det_errors.append(strain_val)
            bit_mismatches_det.append(bit_diff)
            energies_det.append(internal_energy.item())
        else:
            stoch_errors.append(strain_val)
            energies_stoch.append(internal_energy.item())

        if (t + 1) % 256 == 0:
            det_tail_err = np.mean(det_errors[-64:]) if len(det_errors) >= 64 else np.mean(det_errors)
            det_tail_bits = np.mean(bit_mismatches_det[-64:]) if len(bit_mismatches_det) >= 64 else np.mean(bit_mismatches_det)
            print(f"  Step {t+1:04d}/{stream_len} | Strain: {rolling_strain:.4f} | Det Energy: {det_tail_err:.4f} | Det Bit Errors: {det_tail_bits:.2f}/8 bits")

    elapsed = time.time() - t0
    tok_per_sec = stream_len / elapsed if elapsed > 0 else 0.0

    q_len = max(1, len(det_errors) // 4)
    final_det_strain = float(np.mean(det_errors[-q_len:]))
    final_det_bit_errors = float(np.mean(bit_mismatches_det[-q_len:]))
    final_stoch_strain = float(np.mean(stoch_errors[-q_len:]))
    final_det_energy = float(np.mean(energies_det[-q_len:]))

    print("\n===============================================================================")
    print("=== FINAL TELEMETRY RESULTS (EXP-364) ===")
    print("===============================================================================")
    print(f"  Deterministic Bit Error Rate : {final_det_bit_errors:.2f} / 8 bits")
    print(f"  Deterministic Strain Energy  : {final_det_strain:.4f}")
    print(f"  Stochastic Strain Energy     : {final_stoch_strain:.4f}")
    print(f"  Conserved Internal Energy    : {final_det_energy:.4f}")
    print(f"  Melting Events (Adaptation)  : {model.melting_events}")
    print(f"  Crystallized Events (Lock-In): {model.crystallization_events}")
    print(f"  Processing Throughput        : {tok_per_sec:.1f} tok/s")

    verdict = "🟢 POSITIVE" if final_det_bit_errors < 1.0 else ("⚪ NEUTRAL" if final_det_bit_errors < 2.0 else "🔴 REJECTED")
    print(f"👑 VERDICT: {verdict}")

    results_data = {
        "exp_id": "EXP-364",
        "final_det_bit_errors": final_det_bit_errors,
        "final_det_strain": final_det_strain,
        "final_stoch_strain": final_stoch_strain,
        "final_det_energy": final_det_energy,
        "melting_events": model.melting_events,
        "crystallization_events": model.crystallization_events,
        "tok_per_sec": tok_per_sec,
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_364_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    run_exp_364_benchmark()
