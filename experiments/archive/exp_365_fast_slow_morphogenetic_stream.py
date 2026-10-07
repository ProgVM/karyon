"""
EXP-365: Continuous Sovereign Morphogenesis & Fast-Slow Plastic Memory (CSM-FSPM)
=================================================================================
Refined based on EXP-362 (1.73 bit error rate breakthrough) and EXP-364 Autopsy:

1. Zero Softmax / Zero Categorical Distributions:
   - Output operates as a physical 8-bit bipolar vector in R^8.
   - Physical strain energy loss: E(t) = 0.5 * ||y_pred - y_true||^2.
2. Unbroken Single-Pass Continuous Temporal Stream (t -> t+1, N=1):
   - Zero batches, zero epoch resets.
3. Dual Fast-Slow Plastic Continuous Manifold:
   - Base Slow Manifold W_base: Learns continuous invariant structure via AdamW.
   - Fast Epigenetic Meta-Plasticity W_fast: Real-time strain correlation update dW = eta * (strain (x) s).
   - Normalized RMS Flow: Prevents saturation of tanh bounds, keeping gradients alive throughout the stream.
4. Continuous Morphogenetic Channel Sprouting:
   - Starts at D=32 channels. When average strain exceeds homeostatic tolerance, network
     smoothly sprouts new functional channels (D -> D + 8) with zero-shock Net2Net gating.

Target Metric:
- Deterministic Bit Error Rate -> < 1.00 / 8 bits.
- Deterministic Strain Energy -> < 1.50.
=================================================================================
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


def generate_stream(stream_length: int = 2048):
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
# CONTINUOUS FAST-SLOW MORPHOGENETIC ENGINE
# =============================================================================

class FastSlowMorphogeneticEngine(nn.Module):
    def __init__(self, in_dim: int = 8, initial_dim: int = 32, max_dim: int = 128, device: torch.device = DEVICE):
        super().__init__()
        self.in_dim = in_dim
        self.current_dim = initial_dim
        self.max_dim = max_dim
        self.device = device

        # Base invariant weights
        self.W_in = nn.Parameter(torch.randn(max_dim, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))
        self.W_base = nn.Parameter(torch.randn(max_dim, max_dim, device=device) * (0.5 / math.sqrt(initial_dim)))
        self.W_out = nn.Parameter(torch.randn(in_dim, max_dim, device=device) * (1.0 / math.sqrt(initial_dim)))

        # Fast Epigenetic Meta-Plastic Tensor
        self.W_fast = nn.Parameter(torch.zeros(max_dim, max_dim, device=device))

        # Dynamic allostatic parameters
        self.plasticity_lr = nn.Parameter(torch.tensor(0.04, device=device))
        self.fast_decay = nn.Parameter(torch.tensor(0.992, device=device))

        self.sprout_events = 0
        self.evolution_steps = 0

    def sprout_channels(self, delta_d: int = 8):
        if self.current_dim + delta_d <= self.max_dim:
            old_d = self.current_dim
            new_d = old_d + delta_d
            with torch.no_grad():
                # Smooth Net2Net birth: newly expanded channels have minimal initial weight
                self.W_base.data[old_d:new_d, :] *= 0.05
                self.W_base.data[:, old_d:new_d] *= 0.05
                self.W_out.data[:, old_d:new_d] *= 0.05
            self.current_dim = new_d
            self.sprout_events += 1
            return True
        return False

    def forward_step(self, x_t: torch.Tensor, s_prev: torch.Tensor):
        d = self.current_dim
        x = x_t.unsqueeze(0)  # [1, 8]
        s = s_prev[:d].unsqueeze(0)  # [1, d]

        # Active submatrices
        W_in_sub = self.W_in[:d, :]
        W_base_sub = self.W_base[:d, :d]
        W_fast_sub = self.W_fast[:d, :d]
        W_out_sub = self.W_out[:, :d]

        # Effective composite operator
        W_eff = W_base_sub + W_fast_sub

        drive = torch.matmul(x, W_in_sub.t())
        flow = torch.matmul(s, W_eff.t())

        # RMS Normalization keeps signals in optimal dynamic range without hardcoding
        rms = torch.sqrt(torch.mean((drive + flow)**2, dim=-1, keepdim=True) + 1e-5)
        normalized_drive = (drive + flow) / rms

        # Overdamped first-order relaxation
        tau = 0.35
        s_target = torch.tanh(normalized_drive)
        s_next = (1.0 - tau) * s + tau * s_target

        # Real-time Fast Epigenetic Plasticity Update
        strain = (s_next - s).squeeze(0)  # [d]
        with torch.no_grad():
            dW = torch.outer(strain, s.squeeze(0)) * 0.04
            self.W_fast.data[:d, :d] = torch.clamp(
                self.fast_decay * W_fast_sub + self.plasticity_lr * dW,
                -1.2, 1.2
            )
            self.evolution_steps += 1

        # Physical 8-bit Phase-Lock Readout
        y_pred = torch.tanh(torch.matmul(s_next, W_out_sub.t())).squeeze(0)

        s_out = s_prev.clone()
        s_out[:d] = s_next.squeeze(0)

        internal_energy = 0.5 * torch.sum(strain**2)
        return y_pred, s_out, internal_energy


# =============================================================================
# CONTINUOUS STREAM BENCHMARK RUNNER
# =============================================================================

def run_exp_365():
    print("===============================================================================")
    print("=== KEP EXP-365: CONTINUOUS FAST-SLOW MORPHOGENETIC ENGINE (CSM-FSPM)       ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    stream_data, stream_types, raw_bytes = generate_stream(stream_length=2048)
    stream_len = stream_data.size(0)

    model = FastSlowMorphogeneticEngine(in_dim=8, initial_dim=32, max_dim=128, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)

    s_state = torch.zeros(model.max_dim, device=DEVICE)

    det_errors = []
    stoch_errors = []
    bit_mismatches_det = []
    energies_det = []
    energies_stoch = []

    rolling_strain = 2.0
    t0 = time.time()

    for t in range(stream_len - 1):
        cur_vec = stream_data[t]
        tgt_vec = stream_data[t + 1]
        is_stochastic = stream_types[t].item()

        optimizer.zero_grad()

        # Step continuous engine
        y_pred, s_state, internal_energy = model.forward_step(cur_vec, s_state)
        s_state = s_state.detach()

        # Physical quadratic strain
        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        total_loss = prediction_strain + 0.002 * internal_energy

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        rolling_strain = 0.92 * rolling_strain + 0.08 * strain_val

        # Autopoietic morphological expansion
        if t > 30 and t % 32 == 0:
            if rolling_strain > 1.2:
                if model.sprout_channels(delta_d=8):
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)

        # Exact bit-level parity audit
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
            print(f"  Step {t+1:04d}/{stream_len} | Dim: {model.current_dim:02d} | Strain: {rolling_strain:.4f} | Det Energy: {det_tail_err:.4f} | Det Bit Errors: {det_tail_bits:.2f}/8 bits")

    elapsed = time.time() - t0
    tok_per_sec = stream_len / elapsed if elapsed > 0 else 0.0

    q_len = max(1, len(det_errors) // 4)
    final_det_strain = float(np.mean(det_errors[-q_len:]))
    final_det_bit_errors = float(np.mean(bit_mismatches_det[-q_len:]))
    final_stoch_strain = float(np.mean(stoch_errors[-q_len:]))
    final_det_energy = float(np.mean(energies_det[-q_len:]))

    print("\n===============================================================================")
    print("=== FINAL TELEMETRY RESULTS (EXP-365) ===")
    print("===============================================================================")
    print(f"  Final Active State Dimension : {model.current_dim} (Started at 32)")
    print(f"  Sprout Morphogenesis Events  : {model.sprout_events}")
    print(f"  Deterministic Bit Error Rate : {final_det_bit_errors:.2f} / 8 bits")
    print(f"  Deterministic Strain Energy  : {final_det_strain:.4f}")
    print(f"  Stochastic Strain Energy     : {final_stoch_strain:.4f}")
    print(f"  Internal Dynamic Energy      : {final_det_energy:.4f}")
    print(f"  Online Meta-Evolution Steps  : {model.evolution_steps}")
    print(f"  Processing Throughput        : {tok_per_sec:.1f} tok/s")

    verdict = "🟢 POSITIVE" if final_det_bit_errors < 1.0 else ("⚪ NEUTRAL" if final_det_bit_errors < 2.0 else "🔴 REJECTED")
    print(f"👑 VERDICT: {verdict}")

    results_data = {
        "exp_id": "EXP-365",
        "final_dim": model.current_dim,
        "sprout_events": model.sprout_events,
        "final_det_bit_errors": final_det_bit_errors,
        "final_det_strain": final_det_strain,
        "final_stoch_strain": final_stoch_strain,
        "final_det_energy": final_det_energy,
        "evolution_steps": model.evolution_steps,
        "tok_per_sec": tok_per_sec,
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_365_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    run_exp_365()
