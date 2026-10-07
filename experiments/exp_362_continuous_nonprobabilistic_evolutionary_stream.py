"""
EXP-362: Continuous Non-Probabilistic Self-Evolving Stream Architectures
=======================================================================
Mandate & Foundational Principles from Bazilevs:
1. Complete Eradication of Probabilities & Softmax Roulette:
   - Bytes are physical 8-bit state vectors in R^8 (b_i in {-1.0, +1.0}).
   - Prediction is a continuous physical phase-lock. Error is physical Euclidean/Hamiltonian
     strain energy E(t) = 0.5 * ||y_pred - y_true||^2.
   - On deterministic streams: Error E(t) -> 0.0, Bit Error Rate -> 0.0%.
   - On stochastic streams: Optimal continuous physical diffusion & energy minimization.
2. True Unbroken Continuous Streaming (t -> t+1, Single Pass N=1):
   - No batches, no epoch resets. Time is an unbroken causal stream.
   - States persist and evolve continuously across reality.
3. Sovereign Evolution Without Predefined Laws:
   - The system must autonomously modify, create, and adapt its own structure, operators,
     and update dynamics during learning.

Evaluated Self-Evolving Paradigms:
- Paradigm 1: Topological Morphogenesis (Sprouting & pruning continuous state dimensions and coupling tensors under error pressure)
- Paradigm 2: Meta-Plastic Differential Evolution (Autonomously evolving the differential update operator dPsi/dt in real time)
- Paradigm 3: Holographic Continuous Interference Field (Nonlinear wavepacket interference where resonant modes self-assemble)
=======================================================================
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
# CONTINUOUS 8-BIT PHYSICAL STREAM GENERATOR (DETERMINISTIC & STOCHASTIC)
# =============================================================================

def byte_to_bit_vector(b: int) -> torch.Tensor:
    """Converts integer byte [0..255] into physical bipolar 8-bit vector in {-1.0, +1.0}^8."""
    bits = [1.0 if ((b >> i) & 1) else -1.0 for i in range(8)]
    return torch.tensor(bits, dtype=torch.float32, device=DEVICE)


def bit_vector_to_byte(vec: torch.Tensor) -> int:
    """Converts continuous physical vector to nearest discrete byte via sign phase-lock."""
    signs = (vec > 0.0).cpu().numpy().astype(int)
    val = 0
    for i, bit in enumerate(signs):
        if bit:
            val |= (1 << i)
    return val


def generate_physical_stream(stream_length: int = 2048):
    """
    Generates unbroken stream of physical 8-dimensional bit vectors:
    - Deterministic mode: LFSR byte dynamics with nonlinear feedback (Strict bitwise parity)
    - Stochastic mode: Continuous Brownian walk discretized into 8-bit physical space
    """
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

    stream_tensor = torch.stack(vectors, dim=0)  # [stream_length, 8]
    type_tensor = torch.tensor(stream_types, dtype=torch.long, device=DEVICE)
    return stream_tensor, type_tensor, raw_bytes


# =============================================================================
# PARADIGM 1: TOPOLOGICAL MORPHOGENESIS (AUTONOMOUS DIMENSIONAL SPROUTING/PRUNING)
# =============================================================================

class DynamicMorphologyEngine(nn.Module):
    """
    Starts with minimal internal state space dim=16.
    When physical prediction strain E(t) > threshold, it autonomously sprouts new
    orthogonal state dimensions and interaction tensors with zero initial shock.
    """
    def __init__(self, in_dim: int = 8, initial_hidden: int = 16, max_hidden: int = 128, device: torch.device = DEVICE):
        super().__init__()
        self.in_dim = in_dim
        self.current_hidden = initial_hidden
        self.max_hidden = max_hidden
        self.device = device

        # Pre-allocated maximum capacity tensor buffers
        self.W_in = nn.Parameter(torch.randn(max_hidden, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))
        self.W_rec = nn.Parameter(torch.randn(max_hidden, max_hidden, device=device) * (0.5 / math.sqrt(initial_hidden)))
        self.W_out = nn.Parameter(torch.randn(in_dim, max_hidden, device=device) * (1.0 / math.sqrt(initial_hidden)))

        # Dynamic metric coupling and time scale
        self.metric_diag = nn.Parameter(torch.ones(max_hidden, device=device))
        self.tau = nn.Parameter(torch.tensor(0.3, device=device))

        self.sprout_events = 0

    def sprout_dimensions(self, delta_d: int = 8):
        if self.current_hidden + delta_d <= self.max_hidden:
            # Net2Net smooth birth: new incoming/outgoing connections initialized softly
            old_h = self.current_hidden
            new_h = old_h + delta_d
            with torch.no_grad():
                self.W_rec.data[old_h:new_h, :] *= 0.01
                self.W_rec.data[:, old_h:new_h] *= 0.01
                self.W_out.data[:, old_h:new_h] *= 0.01
            self.current_hidden = new_h
            self.sprout_events += 1
            return True
        return False

    def forward_step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        # x_t: [8], h_prev: [max_hidden]
        h = self.current_hidden
        x = x_t.unsqueeze(0)  # [1, 8]
        s = h_prev[:h].unsqueeze(0)  # [1, h]

        W_in_sub = self.W_in[:h, :]  # [h, 8]
        W_rec_sub = self.W_rec[:h, :h]  # [h, h]
        W_out_sub = self.W_out[:, :h]  # [8, h]

        # Continuous multilinear field flow
        in_drive = torch.matmul(x, W_in_sub.t())  # [1, h]
        rec_drive = torch.matmul(s, W_rec_sub.t())  # [1, h]

        # Physical hyperbolic relaxation (no preset math lego menus, just continuous tensor flow)
        effective_tau = torch.clamp(torch.sigmoid(self.tau), 0.05, 0.8)
        s_target = torch.tanh(in_drive + rec_drive)
        s_next = (1.0 - effective_tau) * s + effective_tau * s_target

        # Physical 8-bit phase-lock readout (Continuous bipolar attractor)
        y_pred = torch.tanh(torch.matmul(s_next, W_out_sub.t())).squeeze(0)  # [8]

        h_out = h_prev.clone()
        h_out[:h] = s_next.squeeze(0)

        # Internal field strain energy
        internal_energy = 0.5 * torch.sum((s_next - s)**2)
        return y_pred, h_out, internal_energy


# =============================================================================
# PARADIGM 2: META-PLASTIC DIFFERENTIAL EVOLUTION (SELF-EVOLVING UPDATE OPERATOR)
# =============================================================================

class MetaPlasticEvolutionEngine(nn.Module):
    """
    The differential evolution operator itself dL/dt evolves continuously based on local
    physical strain. The system learns its own continuous update laws without human intervention.
    """
    def __init__(self, in_dim: int = 8, hidden_dim: int = 48, device: torch.device = DEVICE):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.W_in = nn.Parameter(torch.randn(hidden_dim, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))
        self.W_base = nn.Parameter(torch.randn(hidden_dim, hidden_dim, device=device) * (0.5 / math.sqrt(hidden_dim)))
        self.W_out = nn.Parameter(torch.randn(in_dim, hidden_dim, device=device) * (1.0 / math.sqrt(hidden_dim)))

        # Plastic meta-operator (continually synthesized on every step)
        self.W_plastic = nn.Parameter(torch.zeros(hidden_dim, hidden_dim, device=device))
        self.meta_rate = nn.Parameter(torch.tensor(0.02, device=device))
        self.decay = nn.Parameter(torch.tensor(0.995, device=device))

        self.sprout_events = 0

    def forward_step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x = x_t.unsqueeze(0)  # [1, 8]
        s = h_prev.unsqueeze(0)  # [1, H]

        # Effective synthesized operator
        W_eff = self.W_base + self.W_plastic

        drive = torch.matmul(x, self.W_in.t())
        flow = torch.matmul(s, W_eff.t())
        s_next = torch.tanh(0.7 * s + 0.3 * (drive + flow))

        # Real-time meta-plasticity: Operator self-modifies via correlation strain
        strain = (s_next - s).squeeze(0)  # [H]
        with torch.no_grad():
            dW = torch.outer(strain, s.squeeze(0)) * 0.05
            self.W_plastic.data = torch.clamp(self.decay * self.W_plastic.data + self.meta_rate * dW, -1.0, 1.0)
            self.sprout_events += 1

        y_pred = torch.tanh(torch.matmul(s_next, self.W_out.t())).squeeze(0)
        internal_energy = 0.5 * torch.sum(strain**2)
        return y_pred, s_next.squeeze(0), internal_energy


# =============================================================================
# PARADIGM 3: HOLOGRAPHIC CONTINUOUS INTERFERENCE FIELD (SPECTRAL RESONANCE)
# =============================================================================

class HolographicInterferenceEngine(nn.Module):
    """
    Continuous wavepacket interference. State is a multi-frequency wavefield Psi in C^D.
    Incoming data acts as phase-shift perturbation. Interference creates sharp constructive
    resonances for deterministic logic and diffuse wave scattering for stochastic inputs.
    """
    def __init__(self, in_dim: int = 8, modes: int = 32, device: torch.device = DEVICE):
        super().__init__()
        self.in_dim = in_dim
        self.modes = modes
        self.device = device

        # Wavefield amplitudes & spatial frequencies
        self.freq_real = nn.Parameter(torch.randn(modes, in_dim, device=device) * 0.5)
        self.freq_imag = nn.Parameter(torch.randn(modes, in_dim, device=device) * 0.5)
        self.coupling = nn.Parameter(torch.randn(modes, modes, device=device) * (0.5 / math.sqrt(modes)))
        self.readout = nn.Parameter(torch.randn(in_dim, modes * 2, device=device) * (1.0 / math.sqrt(modes)))

        self.sprout_events = 0

    def forward_step(self, x_t: torch.Tensor, wave_prev: torch.Tensor):
        # wave_prev: [modes, 2] (Real and Imaginary components of wavepacket)
        x = x_t.unsqueeze(0)  # [1, 8]

        # Phase perturbation from input
        phase_r = torch.matmul(x, self.freq_real.t()).squeeze(0)  # [modes]
        phase_i = torch.matmul(x, self.freq_imag.t()).squeeze(0)  # [modes]

        u_r = wave_prev[:, 0]
        u_i = wave_prev[:, 1]

        # Wave interference rotation: (u_r + i*u_i) * exp(i * phase)
        rot_r = u_r * torch.cos(phase_r) - u_i * torch.sin(phase_i)
        rot_i = u_r * torch.sin(phase_i) + u_i * torch.cos(phase_r)

        # Coupled spatial dispersion
        disp_r = torch.matmul(self.coupling, rot_r)
        disp_i = torch.matmul(self.coupling, rot_i)

        next_r = torch.tanh(0.6 * u_r + 0.4 * disp_r)
        next_i = torch.tanh(0.6 * u_i + 0.4 * disp_i)
        wave_next = torch.stack([next_r, next_i], dim=-1)  # [modes, 2]

        wave_flat = wave_next.view(1, -1)  # [1, modes*2]
        y_pred = torch.tanh(torch.matmul(wave_flat, self.readout.t())).squeeze(0)  # [8]

        internal_energy = 0.5 * torch.sum((wave_next - wave_prev)**2)
        return y_pred, wave_next, internal_energy


# =============================================================================
# CONTINUOUS SINGLE-PASS PHYSICAL BENCHMARK HARNESS
# =============================================================================

def evaluate_evolutionary_physical_stream(name: str, model: nn.Module, p_type: str, stream_data: torch.Tensor, stream_types: torch.Tensor, raw_bytes: list):
    print(f"\n===============================================================================")
    print(f"=== NON-PROBABILISTIC STREAM BENCHMARK: {name} ===")
    print(f"===============================================================================")

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-5)
    t0 = time.time()
    stream_len = stream_data.size(0)

    det_errors = []
    stoch_errors = []
    det_energies = []
    stoch_energies = []
    bit_mismatches_det = []

    # Initialize continuous state
    if p_type == "P1":
        state = torch.zeros(model.max_hidden, device=DEVICE)
    elif p_type == "P2":
        state = torch.zeros(model.hidden_dim, device=DEVICE)
    elif p_type == "P3":
        state = torch.zeros(model.modes, 2, device=DEVICE)

    rolling_strain = 0.0

    for t in range(stream_len - 1):
        cur_vec = stream_data[t]
        tgt_vec = stream_data[t + 1]
        is_stochastic = stream_types[t].item()

        optimizer.zero_grad()

        # Step continuous physical system
        y_pred, state, internal_energy = model.forward_step(cur_vec, state)
        state = state.detach()

        # Physical Quadratic Strain Energy: E(t) = 0.5 * ||y_pred - tgt_vec||^2 (NO SOFTMAX!)
        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        total_loss = prediction_strain + 0.005 * internal_energy

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        rolling_strain = 0.92 * rolling_strain + 0.08 * strain_val if t > 0 else strain_val

        # Autopoietic morphological evolution triggered on strain
        if p_type == "P1" and t > 40 and t % 32 == 0:
            if rolling_strain > 1.0:
                if model.sprout_dimensions(delta_d=8):
                    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-5)

        # Evaluate Physical Bit Errors (exact parity check)
        pred_byte = bit_vector_to_byte(y_pred)
        tgt_byte = raw_bytes[t + 1]

        # Count bit mismatch count (0 to 8 bits)
        bit_diff = bin(pred_byte ^ tgt_byte).count('1')

        if is_stochastic == 0:
            det_errors.append(strain_val)
            det_energies.append(internal_energy.item())
            bit_mismatches_det.append(bit_diff)
        else:
            stoch_errors.append(strain_val)
            stoch_energies.append(internal_energy.item())

        if (t + 1) % 256 == 0:
            det_tail_err = np.mean(det_errors[-64:]) if len(det_errors) >= 64 else np.mean(det_errors)
            det_tail_bit_err = np.mean(bit_mismatches_det[-64:]) if len(bit_mismatches_det) >= 64 else np.mean(bit_mismatches_det)
            print(f"  Step {t+1:04d}/{stream_len} | Strain: {rolling_strain:.4f} | Det Energy: {det_tail_err:.4f} | Det Bit Errors: {det_tail_bit_err:.2f}/8 bits")

    elapsed = time.time() - t0
    tok_per_sec = stream_len / elapsed if elapsed > 0 else 0.0

    # Metrics on final quarter of stream
    q_len = max(1, len(det_errors) // 4)
    final_det_strain = float(np.mean(det_errors[-q_len:]))
    final_det_bit_errors = float(np.mean(bit_mismatches_det[-q_len:]))
    final_stoch_strain = float(np.mean(stoch_errors[-q_len:]))
    final_stoch_energy = float(np.mean(stoch_energies[-q_len:]))

    sprouts = getattr(model, 'sprout_events', 0)

    print(f"\n  [FINAL TELEMETRY: {name}]")
    print(f"  --> Final Det Strain Energy : {final_det_strain:.4f} (Bit Mismatch: {final_det_bit_errors:.2f}/8 bits)")
    print(f"  --> Final Stoch Strain Energy: {final_stoch_strain:.4f} (Internal Energy: {final_stoch_energy:.4f})")
    print(f"  --> Evolution Events: {sprouts} | Throughput: {tok_per_sec:.1f} tok/s")

    return {
        "paradigm": name,
        "type": p_type,
        "final_det_strain": final_det_strain,
        "final_det_bit_errors": final_det_bit_errors,
        "final_stoch_strain": final_stoch_strain,
        "final_stoch_energy": final_stoch_energy,
        "sprout_events": sprouts,
        "tok_per_sec": tok_per_sec,
        "elapsed_sec": elapsed
    }


def run_exp_362_benchmark():
    print("===============================================================================")
    print("=== KEP EXP-362: NON-PROBABILISTIC SELF-EVOLVING STREAM (TRI-BATTLE)        ===")
    print("===============================================================================")
    print(f"Substrate Device: {DEVICE}")

    stream_data, stream_types, raw_bytes = generate_physical_stream(stream_length=2048)

    # 1. Topological Morphogenesis (Sprouting state dimensions under strain)
    m1 = DynamicMorphologyEngine(in_dim=8, initial_hidden=16, max_hidden=128, device=DEVICE)
    res_p1 = evaluate_evolutionary_physical_stream("Paradigm 1: Topological Morphogenesis (Sprouting Subspace)", m1, "P1", stream_data, stream_types, raw_bytes)

    # 2. Meta-Plastic Differential Evolution (Self-evolving differential operator)
    m2 = MetaPlasticEvolutionEngine(in_dim=8, hidden_dim=48, device=DEVICE)
    res_p2 = evaluate_evolutionary_physical_stream("Paradigm 2: Meta-Plastic Differential Evolution", m2, "P2", stream_data, stream_types, raw_bytes)

    # 3. Holographic Interference Field (Resonant mode assembly)
    m3 = HolographicInterferenceEngine(in_dim=8, modes=32, device=DEVICE)
    res_p3 = evaluate_evolutionary_physical_stream("Paradigm 3: Holographic Interference Wavefield", m3, "P3", stream_data, stream_types, raw_bytes)

    print("\n===============================================================================")
    print("=== FINAL COMPARATIVE STREAM TELEMETRY ===")
    print("===============================================================================")
    all_res = [res_p1, res_p2, res_p3]
    for r in all_res:
        print(f"{r['paradigm']}:")
        print(f"  Det Strain Energy : {r['final_det_strain']:.4f} (Avg Bit Error: {r['final_det_bit_errors']:.2f}/8 bits)")
        print(f"  Stoch Strain Energy: {r['final_stoch_strain']:.4f} (Internal Energy: {r['final_stoch_energy']:.4f})")
        print(f"  Evolution Events  : {r['sprout_events']} | Speed: {r['tok_per_sec']:.1f} tok/s\n")

    winner = min(all_res, key=lambda x: (x['final_det_strain'] + x['final_det_bit_errors']))
    print(f"👑 VICTORIOUS SOVEREIGN ARCHITECTURE: {winner['paradigm']} (Det Strain: {winner['final_det_strain']:.4f}, Bit Err: {winner['final_det_bit_errors']:.2f})")

    verdict = "🟢 POSITIVE" if winner['final_det_bit_errors'] < 2.0 else "⚪ NEUTRAL"

    results_data = {
        "exp_id": "EXP-362",
        "results": all_res,
        "winner": winner['paradigm'],
        "winner_det_strain": float(winner['final_det_strain']),
        "winner_bit_errors": float(winner['final_det_bit_errors']),
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_362_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    run_exp_362_benchmark()
