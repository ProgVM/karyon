"""
EXP-370: Vector Autopoietic Operator Genesis with Field-Theoretical Routing (VAOG-FTR)
======================================================================================
Sovereign Mathematical & Biophysical Blueprint:
1. Pure Vector State Space (D=64):
   - Rejects the fragile, slow scalar-level representation of EXP-368/369.
   - All state representations, inputs, and operator outputs are continuous vector fields
     in R^64, fully utilizing GPU Tensor Cores and preventing gradient dispersion.
2. Complete Basis of Atomic Vector Operators:
   - Operators are not simple scalar functions, but rich vector-field transformations:
     * LeakyIntegratorOp: Continuous temporal decay + non-linear projection.
     * BilinearConjunctionOp: Multiplicative gating and causal conjunction.
     * AttractorSnappingOp: Saturated nonlinear snapping (continuous Hopfield dynamics).
     * SymplecticRotationOp: Phase-space rotation preserving energy.
3. Epigenetic Morphogenesis & Net2Net Smooth Grafting (KEP Principle 15 & 16):
   - Karyon begins with a minimal set of active vector operators (K=2).
   - When prediction strain exceeds the homeostatic threshold, a new operator node is sprouted.
   - To prevent structural shock and catastrophic forgetting, the newly sprouted node is
     wrapped in an Epigenetic Gating coefficient (alpha_epi), initialized at exactly 0.0.
     This guarantees STRICT ZERO-SHOCK FUNCTION IDENTITY f_new(x) === f_old(x) at birth.
4. Clean Out-of-Place Recurrent State Passing (KEP Rule #1.1 Compliance):
   - Clean functional state updates (prev_states -> next_states) avoiding in-place autograd corruption.
5. Forced Vector Routing (Anti-Shortcut Mandate):
   - Readout MUST combine representations strictly from the active vector operators,
     preventing the shortcut learning of raw inputs.
6. Continuous Reality Stream (t -> t+1, N=1, Single Pass):
   - Zero epochs, zero batching, zero Softmax.
   - Evaluated on a continuous physical 8-bit bipolar vector stream.
======================================================================================
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
# ATOMIC VECTOR OPERATOR CLASSES
# =============================================================================

class LeakyIntegratorOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b = nn.Parameter(torch.zeros(dim, device=device))
        self.log_tau = nn.Parameter(torch.full((dim,), -1.0, device=device))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        tau = torch.sigmoid(self.log_tau)
        drive = torch.tanh(torch.matmul(self.W, x) + self.b)
        return (1.0 - tau) * state + tau * drive


class BilinearConjunctionOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.W_a = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_b = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        a = torch.tanh(torch.matmul(self.W_a, x))
        b = torch.sigmoid(torch.matmul(self.W_b, state))
        return a * b


class AttractorSnappingOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim, device=device) * (0.5 / math.sqrt(dim)))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        u = x + state
        u_norm = u / (torch.norm(u, p=2, dim=-1, keepdim=True) + 1e-5)
        snapped = torch.tanh(torch.matmul(self.W, u_norm))
        return snapped


class SymplecticRotationOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, dim, device=device) * (0.2 / math.sqrt(dim)))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        W_skew = self.A - self.A.t()
        rotated = torch.matmul(W_skew, state)
        return torch.tanh(rotated + x)


# =============================================================================
# VECTOR AUTOPOIETIC GENESIS ENGINE
# =============================================================================

class VectorAutopoieticGenesisEngine(nn.Module):
    def __init__(self, in_dim: int = 8, dim: int = 64, max_operators: int = 8, device: torch.device = DEVICE):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.max_ops = max_operators
        self.device = device

        self.W_in = nn.Parameter(torch.randn(dim, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))
        self.b_in = nn.Parameter(torch.zeros(dim, device=device))

        self.operators = nn.ModuleList()
        self.op_types = []
        self.alpha_epi = nn.ParameterList()

        self.active_ops = 0
        self._add_operator("leaky")
        self._add_operator("bilinear")

        self.readout_weights = nn.Parameter(torch.randn(in_dim, max_operators, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.readout_bias = nn.Parameter(torch.zeros(in_dim, device=device))

        self.sprout_events = 0

    def _add_operator(self, op_type: str):
        if len(self.operators) >= self.max_ops:
            return False

        if op_type == "leaky":
            op = LeakyIntegratorOp(self.dim, self.device)
        elif op_type == "bilinear":
            op = BilinearConjunctionOp(self.dim, self.device)
        elif op_type == "attractor":
            op = AttractorSnappingOp(self.dim, self.device)
        else:
            op = SymplecticRotationOp(self.dim, self.device)

        self.operators.append(op)
        self.op_types.append(op_type)
        self.alpha_epi.append(nn.Parameter(torch.tensor(0.0, device=self.device)))

        self.active_ops += 1
        return True

    def sprout_operator(self):
        if self.active_ops < self.max_ops:
            types = ["leaky", "bilinear", "attractor", "symplectic"]
            new_type = types[self.active_ops % len(types)]
            if self._add_operator(new_type):
                self.sprout_events += 1
                return True
        return False

    def forward(self, x_t: torch.Tensor, prev_states: list):
        # x_t: [8], prev_states: list of [64]
        h_in = torch.tanh(torch.matmul(self.W_in, x_t) + self.b_in)

        op_outputs = []
        next_states = []

        for i in range(self.active_ops):
            op = self.operators[i]
            st = prev_states[i]

            raw_out = op(h_in, st)

            # Epigenetic gate
            gate = torch.tanh(self.alpha_epi[i])
            gated_out = gate * raw_out

            op_outputs.append(gated_out)
            next_states.append(raw_out)

        # Forced Vector Routing Readout
        y_pred = torch.zeros(self.in_dim, device=self.device)
        for i in range(self.in_dim):
            val = 0.0
            for j in range(self.active_ops):
                val = val + torch.sum(self.readout_weights[i, j] * op_outputs[j])
            y_pred[i] = torch.tanh(val + self.readout_bias[i])

        complexity = 0.0
        for i in range(self.active_ops):
            complexity = complexity + torch.abs(torch.tanh(self.alpha_epi[i]))

        return y_pred, next_states, complexity

    def extract_symbolic_formulas(self) -> str:
        report = []
        report.append("=== KARYON VECTOR AUTOPOIETIC GENESIS SYSTEM ===")
        for i in range(self.active_ops):
            gate_val = torch.tanh(self.alpha_epi[i]).item()
            report.append(f"  [Operator Node {i}]: Type = {self.op_types[i]} | Epigenetic Gate = {gate_val:.4f}")
        report.append(f"  Readout: y_pred = tanh( sum_{{i=0}}^{{{self.active_ops-1}}} W_readout_i * (Gate_i * Op_i(h_in)) )")
        return "\n".join(report)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_exp_370():
    print("===============================================================================")
    print("=== KEP EXP-370: VECTOR AUTOPOIETIC OPERATOR GENESIS (VAOG-FTR)             ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    stream_data, stream_types, raw_bytes = generate_unbroken_stream(stream_length=2048)
    stream_len = stream_data.size(0)

    model = VectorAutopoieticGenesisEngine(in_dim=8, dim=64, max_operators=8, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-2, weight_decay=1e-5)

    states = [torch.zeros(model.dim, device=DEVICE) for _ in range(model.active_ops)]

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

        # Step vector autopoietic engine
        y_pred, next_states, complexity = model(cur_vec, states)

        # Detach states for next step
        states = [s.detach() for s in next_states]

        # Physical quadratic strain + Epigenetic complexity penalty
        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        free_energy = prediction_strain + 0.02 * complexity + 0.01 * model.active_ops

        free_energy.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        rolling_strain = 0.90 * rolling_strain + 0.10 * strain_val

        # Autopoietic sprouting: expand operator pool if strain > 0.8
        if t > 40 and t % 32 == 0:
            if rolling_strain > 0.8:
                if model.sprout_operator():
                    states.append(torch.zeros(model.dim, device=DEVICE))
                    print(f"  [MORPHOGENESIS] Sprouted new vector operator at step {t+1}. Active: {model.active_ops}")
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
            print(f"  Step {t+1:04d}/{stream_len} | Active Ops: {model.active_ops:02d} | Strain: {rolling_strain:.4f} | Det Energy: {det_tail_err:.4f} | Det Bit Errors: {det_tail_bits:.2f}/8 bits")

    elapsed = time.time() - t0
    tok_per_sec = stream_len / elapsed if elapsed > 0 else 0.0

    q_len = max(1, len(det_errors) // 4)
    final_det_strain = float(np.mean(det_errors[-q_len:]))
    final_det_bit_errors = float(np.mean(bit_mismatches_det[-q_len:]))
    final_stoch_strain = float(np.mean(stoch_errors[-q_len:]))

    discovered_formulas = model.extract_symbolic_formulas()

    print("\n===============================================================================")
    print("=== FINAL TELEMETRY & AUTOPOIETIC DISCOVERED SYSTEM (EXP-370) ===")
    print("===============================================================================")
    print(f"  Final Active Operator Count  : {model.active_ops} (Started at 2)")
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
        "exp_id": "EXP-370",
        "active_ops": model.active_ops,
        "sprout_events": model.sprout_events,
        "final_det_bit_errors": final_det_bit_errors,
        "final_det_strain": final_det_strain,
        "final_stoch_strain": final_stoch_strain,
        "tok_per_sec": tok_per_sec,
        "discovered_formulas": discovered_formulas,
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_370_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    run_exp_370()
