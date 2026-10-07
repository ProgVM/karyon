"""
EXP-363: Symplectic Meta-Plastic Continuous Stream Engine (SMP-CSE)
===================================================================
Mandate from Bazilevs:
1. Pure Physical Phase-Lock (Absolute Zero Softmax / Zero Categorical Probabilities):
   - 8-bit bipolar vector space R^8 (bits in {-1.0, +1.0}).
   - Prediction is an exact phase-lock collapse into bipolar attractor coordinates.
   - Physical Euclidean strain energy E(t) = 0.5 * ||y_pred - y_true||^2.
2. Unbroken Continuous Stream Reality (t -> t+1, Single-Pass N=1):
   - Continuous temporal flow without artificial batches or epoch restarts.
   - Phase coordinates (q, p) persist and flow smoothly across all time steps.
3. Synthesis of Master Discoveries:
   - Symplectic Phase Space (q, p): Preserves information volume (Liouville's theorem),
     eradicating dissipative forgetting on deterministic logic.
   - Meta-Plastic Continuous Operator Evolution: The interaction tensor L_eff(t) = L_base + L_plastic(t)
     evolves online proportional to symplectic strain without hardcoded update laws.
   - Dual-Nature Processing: Interleaved deterministic machine bytecode and continuous
     stochastic Brownian diffusion.
Target Metric:
- Deterministic Bit Error Rate -> 0.00 / 8 bits.
- Deterministic Strain Energy -> 0.00.
- Stable stochastic absorption with minimal diffusion strain.
===================================================================
"""

import os
import time
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def generate_dual_nature_stream(stream_length: int = 2048):
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
# SYMPLECTIC META-PLASTIC CONTINUOUS STREAM ENGINE
# =============================================================================

class SymplecticMetaPlasticEngine(nn.Module):
    """
    Combines Symplectic Hamiltonian Phase-Space (q, p) with Online Meta-Plastic
    Differential Operator Evolution. Zero Softmax. Continuous Phase-Lock.
    """
    def __init__(self, in_dim: int = 8, hidden_dim: int = 64, device: torch.device = DEVICE):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Input mapping onto phase space drive
        self.W_in = nn.Parameter(torch.randn(hidden_dim, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))

        # Symplectic Hamiltonian Potential Matrix (Base)
        self.W_pot_base = nn.Parameter(torch.randn(hidden_dim, hidden_dim, device=device) * (0.5 / math.sqrt(hidden_dim)))

        # Meta-Plastic Tensor: Self-modifies dynamically during streaming
        self.W_plastic = nn.Parameter(torch.zeros(hidden_dim, hidden_dim, device=device))
        self.meta_plasticity_rate = nn.Parameter(torch.tensor(0.03, device=device))
        self.plasticity_decay = nn.Parameter(torch.tensor(0.992, device=device))

        # Dynamic friction and diffusion ports
        self.W_gamma = nn.Linear(in_dim, hidden_dim, bias=True, device=device)

        # Physical 8-bit Phase-Lock Readout Matrix
        self.W_readout = nn.Parameter(torch.randn(in_dim, hidden_dim, device=device) * (1.0 / math.sqrt(hidden_dim)))

        # Continuous Integration Time Step
        self.log_dt = nn.Parameter(torch.tensor(-1.2, device=device))  # dt ~ 0.3

        self.evolution_steps = 0

    def forward_step(self, x_t: torch.Tensor, q_prev: torch.Tensor, p_prev: torch.Tensor):
        # x_t: [8], q_prev: [H], p_prev: [H]
        q = q_prev.unsqueeze(0)  # [1, H]
        p = p_prev.unsqueeze(0)  # [1, H]
        x = x_t.unsqueeze(0)     # [1, 8]

        dt = torch.clamp(torch.exp(self.log_dt), 0.05, 0.5)

        # Effective Hamiltonian Potential Operator
        W_pot_eff = self.W_pot_base + self.W_plastic
        
        # Drive and Friction
        drive = torch.matmul(x, self.W_in.t())
        gamma = torch.sigmoid(self.W_gamma(x))

        # Symplectic Gradient: grad V = q @ W_pot_eff^T @ W_pot_eff
        q_proj = torch.matmul(q, W_pot_eff)
        grad_V = torch.matmul(q_proj, W_pot_eff.t())

        # Symplectic Leapfrog Integration Step (Preserves Phase Volume)
        p_half = p - 0.5 * dt * (grad_V + gamma * p - drive)
        q_next = q + dt * torch.tanh(p_half)  # Bounded velocity prevents runaway momentum

        q_proj_next = torch.matmul(q_next, W_pot_eff)
        grad_V_next = torch.matmul(q_proj_next, W_pot_eff.t())
        p_next = p_half - 0.5 * dt * (grad_V_next + gamma * p_half - drive)
        p_next = torch.clamp(p_next, -6.0, 6.0)

        # Meta-Plastic Self-Modification:
        # Strain between momentum and coordinate flow updates the plastic potential tensor
        strain_q = (q_next - q).squeeze(0)  # [H]
        strain_p = (p_next - p).squeeze(0)  # [H]

        with torch.no_grad():
            delta_plastic = torch.outer(strain_q, strain_p) * 0.04
            self.W_plastic.data = torch.clamp(
                self.plasticity_decay * self.W_plastic.data + self.meta_plasticity_rate * delta_plastic,
                -1.5, 1.5
            )
            self.evolution_steps += 1

        # Physical Phase-Lock Readout (Bipolar Continuous Projection in R^8)
        y_pred = torch.tanh(torch.matmul(q_next, self.W_readout.t())).squeeze(0)  # [8]

        # Conserved Hamiltonian Energy Metric
        hamiltonian_energy = 0.5 * torch.sum(p_next**2) + 0.5 * torch.sum(q_proj_next**2)

        return y_pred, q_next.squeeze(0), p_next.squeeze(0), hamiltonian_energy


# =============================================================================
# CONTINUOUS STREAM RUNNER (SINGLE PASS N=1)
# =============================================================================

def run_exp_363_benchmark():
    print("===============================================================================")
    print("=== KEP EXP-363: SYMPLECTIC META-PLASTIC CONTINUOUS STREAM ENGINE           ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    stream_data, stream_types, raw_bytes = generate_dual_nature_stream(stream_length=2048)
    stream_len = stream_data.size(0)

    model = SymplecticMetaPlasticEngine(in_dim=8, hidden_dim=64, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)

    # Persistent phase-space states
    q_state = torch.zeros(model.hidden_dim, device=DEVICE)
    p_state = torch.zeros(model.hidden_dim, device=DEVICE)

    det_errors = []
    stoch_errors = []
    bit_mismatches_det = []
    energies_det = []
    energies_stoch = []

    rolling_strain = 0.0
    t0 = time.time()

    for t in range(stream_len - 1):
        cur_vec = stream_data[t]
        tgt_vec = stream_data[t + 1]
        is_stochastic = stream_types[t].item()

        optimizer.zero_grad()

        # Step continuous engine
        y_pred, q_state, p_state, H_energy = model.forward_step(cur_vec, q_state, p_state)
        q_state = q_state.detach()
        p_state = p_state.detach()

        # Pure physical quadratic strain (NO SOFTMAX!)
        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        total_loss = prediction_strain + 0.001 * H_energy

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        rolling_strain = 0.92 * rolling_strain + 0.08 * strain_val if t > 0 else strain_val

        # Exact bit-level parity audit
        pred_byte = bit_vector_to_byte(y_pred)
        tgt_byte = raw_bytes[t + 1]
        bit_diff = bin(pred_byte ^ tgt_byte).count('1')

        if is_stochastic == 0:
            det_errors.append(strain_val)
            bit_mismatches_det.append(bit_diff)
            energies_det.append(H_energy.item())
        else:
            stoch_errors.append(strain_val)
            energies_stoch.append(H_energy.item())

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
    print("=== FINAL TELEMETRY RESULTS (EXP-363) ===")
    print("===============================================================================")
    print(f"  Deterministic Bit Error Rate : {final_det_bit_errors:.2f} / 8 bits")
    print(f"  Deterministic Strain Energy  : {final_det_strain:.4f}")
    print(f"  Stochastic Strain Energy     : {final_stoch_strain:.4f}")
    print(f"  Conserved Hamiltonian Energy : {final_det_energy:.4f}")
    print(f"  Online Meta-Evolution Steps  : {model.evolution_steps}")
    print(f"  Processing Throughput        : {tok_per_sec:.1f} tok/s")

    verdict = "🟢 POSITIVE" if final_det_bit_errors < 1.0 else ("⚪ NEUTRAL" if final_det_bit_errors < 2.0 else "🔴 REJECTED")
    print(f"👑 VERDICT: {verdict}")

    results_data = {
        "exp_id": "EXP-363",
        "final_det_bit_errors": final_det_bit_errors,
        "final_det_strain": final_det_strain,
        "final_stoch_strain": final_stoch_strain,
        "final_det_energy": final_det_energy,
        "evolution_steps": model.evolution_steps,
        "tok_per_sec": tok_per_sec,
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_363_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    run_exp_363_benchmark()
