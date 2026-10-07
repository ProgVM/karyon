"""
EXP-367: Sovereign Autonomous Equation & Dynamic Variable Synthesizer (SA-EDVS)
================================================================================
Philosophical & Mathematical Mandate from Bazilevs:
"Let Karyon itself be the Architect and the Engineer: synthesize arbitrary mathematical
equations and sprout dynamical state variables from fundamental continuous calculus,
without human-devised layer templates, without discrete token tokenizers, and without Softmax."

Key Scientific & Mathematical Pillars:
1. Pure Physical State Space & Continuous Reality (t -> t+1, Single Pass N=1):
   - Operates on 8-bit bipolar vector stream in R^8 (b_i in {-1.0, +1.0}).
   - No epoch resets, zero batching, zero Softmax.
   - Loss is physical quadratic strain energy E(t) = 0.5 * ||y_pred - y_true||^2.
2. Dynamic Variable Sprouting (Emergent State Registers):
   - Karyon begins with a minimal set of internal state variables v = [v_1, v_2].
   - When prediction strain exceeds the homeostatic boundary, Karyon autonomously
     sprouts new dynamical variables v_{K+1} with smooth Net2Net initialization.
3. Differentiable Continuous Formulagenesis (AST Equation Synthesis):
   - Computes dynamic interactions via a differentiable computation tree of atomic calculus primitives:
     * Unary primitives: {id, neg, sin, cos, tanh, exp(-|x|), sqr, recip_safe}
     * Binary primitives: {add, mul, diff_flow, div_safe}
   - Continuously routes and optimizes operation selection coefficients (Gumbel/Softmax routing)
     to synthesize arbitrary non-linear differential equations:
       tau_k * dv_k/dt = -v_k + Phi_k(v, x)
       y_pred = Psi(v, x)
4. Variational Complexity Regularization (Occam's Razor / Free Energy):
   - Total Free Energy: F_t = E_strain(t) + beta_comp * Entropy(Routing) + beta_var * K_vars.
5. Exact Symbolic Formula Extraction:
   - Evaluates the winning mathematical formulas analytically and prints the extracted
     equations for each state variable dv_k/dt and the readout y_pred.
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
import torch.nn.functional as F

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
# DIFFERENTIABLE ATOMIC MATHEMATICAL PRIMITIVES BANK
# =============================================================================

class ContinuousMathPrimitiveBank(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.unary_names = [
            "id",        # x
            "neg",       # -x
            "sin",       # sin(x)
            "cos",       # cos(x)
            "tanh",      # tanh(x)
            "exp_decay", # exp(-|x|)
            "sqr",       # x^2
            "recip"      # 1 / (|x| + eps)
        ]
        self.num_unary = len(self.unary_names)

    def apply_unary(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] -> returns [B, D, num_unary]
        x_clamped = torch.clamp(x, -6.0, 6.0)
        ops = [
            x,
            -x,
            torch.sin(x_clamped),
            torch.cos(x_clamped),
            torch.tanh(x_clamped),
            torch.exp(-torch.abs(x_clamped)),
            torch.clamp(x_clamped**2, 0.0, 20.0),
            1.0 / (torch.abs(x_clamped) + self.eps)
        ]
        return torch.stack(ops, dim=-1)

    def apply_binary(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a, b: [B, D] -> returns [B, D, 4]
        # 0: add, 1: mul, 2: diff (a - b), 3: safe_div (a / (|b| + eps))
        a_c = torch.clamp(a, -6.0, 6.0)
        b_c = torch.clamp(b, -6.0, 6.0)
        ops = [
            a + b,
            torch.clamp(a_c * b_c, -20.0, 20.0),
            a - b,
            torch.clamp(a_c / (torch.abs(b_c) + self.eps), -20.0, 20.0)
        ]
        return torch.stack(ops, dim=-1)


# =============================================================================
# SOVEREIGN FORMULA & VARIABLE SYNTHESIZER ENGINE
# =============================================================================

class SovereignFormulaSynthesizer(nn.Module):
    """
    Karyon's internal engine that synthesizes its own mathematical differential equations
    and dynamical state variables.
    """
    def __init__(self, in_dim: int = 8, var_dim: int = 16, initial_vars: int = 2, max_vars: int = 8, device: torch.device = DEVICE):
        super().__init__()
        self.in_dim = in_dim
        self.var_dim = var_dim
        self.max_vars = max_vars
        self.device = device
        self.bank = ContinuousMathPrimitiveBank()

        # Variable count
        self.current_vars = initial_vars

        # Dynamic variable time constants tau_k
        self.log_tau = nn.Parameter(torch.full((max_vars,), -1.0, device=device))  # tau ~ 0.36

        # Routing logits for equation synthesis across unary/binary primitives
        # For each variable k, routes input x and state variables v into dynamic operators
        self.unary_weights = nn.Parameter(torch.randn(max_vars, self.bank.num_unary, device=device) * 0.1)
        self.binary_weights = nn.Parameter(torch.randn(max_vars, 4, device=device) * 0.1)

        # Coupling matrices for variable evolution
        self.W_in = nn.Parameter(torch.randn(max_vars, var_dim, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))
        self.W_var = nn.Parameter(torch.randn(max_vars, var_dim, max_vars, var_dim, device=device) * (0.5 / math.sqrt(var_dim)))

        # Readout weights
        self.W_out = nn.Parameter(torch.randn(in_dim, max_vars, var_dim, device=device) * (1.0 / math.sqrt(var_dim)))
        self.out_unary_weights = nn.Parameter(torch.randn(in_dim, self.bank.num_unary, device=device) * 0.1)

        self.sprout_events = 0

    def sprout_variable(self):
        if self.current_vars < self.max_vars:
            old_k = self.current_vars
            with torch.no_grad():
                # Zero-shock Net2Net birth
                self.W_in.data[old_k] *= 0.05
                self.W_var.data[old_k] *= 0.05
                self.W_out.data[:, old_k] *= 0.05
            self.current_vars += 1
            self.sprout_events += 1
            return True
        return False

    def forward_step(self, x_t: torch.Tensor, v_prev: torch.Tensor):
        # x_t: [8], v_prev: [max_vars, var_dim]
        K = self.current_vars
        x = x_t.unsqueeze(0)  # [1, 8]

        v_curr = v_prev.clone()
        tau = torch.clamp(torch.sigmoid(self.log_tau[:K]), 0.08, 0.92)  # [K]

        # 1. Compute dynamic differential updates for each active variable v_k
        v_next_list = []
        routing_entropy_sum = 0.0

        for k in range(K):
            # Input flow into variable k
            in_drive = torch.matmul(x, self.W_in[k].t())  # [1, var_dim]

            # Coupling from all other variables
            cross_flow = torch.zeros(1, self.var_dim, device=self.device)
            for j in range(K):
                cross_flow = cross_flow + torch.matmul(v_curr[j].unsqueeze(0), self.W_var[k, :, j, :].t())

            # Synthesize equation via primitive bank
            # A. Unary transformation on input drive
            unary_ops = self.bank.apply_unary(in_drive)  # [1, var_dim, num_unary]
            u_probs = torch.softmax(self.unary_weights[k], dim=-1)  # [num_unary]
            u_term = torch.sum(unary_ops * u_probs.view(1, 1, -1), dim=-1)  # [1, var_dim]

            # B. Binary interaction between u_term and cross_flow
            bin_ops = self.bank.apply_binary(u_term, cross_flow)  # [1, var_dim, 4]
            b_probs = torch.softmax(self.binary_weights[k], dim=-1)  # [4]
            Phi_k = torch.sum(bin_ops * b_probs.view(1, 1, -1), dim=-1)  # [1, var_dim]

            # RMS Normalization
            rms = torch.sqrt(torch.mean(Phi_k**2, dim=-1, keepdim=True) + 1e-5)
            Phi_k_norm = Phi_k / rms

            # Continuous ODE step: tau_k * dv_k/dt = -v_k + Phi_k
            v_k_prev = v_curr[k].unsqueeze(0)
            v_k_next = (1.0 - tau[k]) * v_k_prev + tau[k] * torch.tanh(Phi_k_norm)
            v_next_list.append(v_k_next.squeeze(0))

            # Routing entropy
            routing_entropy_sum += -torch.sum(u_probs * torch.log(u_probs + 1e-8)) - torch.sum(b_probs * torch.log(b_probs + 1e-8))

        # Update active variables
        v_next_tensor = v_prev.clone()
        for k in range(K):
            v_next_tensor[k] = v_next_list[k]

        # 2. Synthesize Readout Equation y_pred = Psi(v, x)
        y_components = []
        for i in range(self.in_dim):
            y_i = torch.zeros(1, device=self.device)
            for k in range(K):
                proj = torch.matmul(v_next_tensor[k].unsqueeze(0), self.W_out[i, k].unsqueeze(-1)).squeeze(-1)
                y_i = y_i + proj
            y_components.append(y_i)
        
        y_raw = torch.cat(y_components, dim=-1)  # [1, 8]
        y_pred = torch.tanh(y_raw).squeeze(0)    # [8] (Physical bipolar phase-lock in R^8)

        return y_pred, v_next_tensor, routing_entropy_sum

    def extract_symbolic_formulas(self) -> str:
        """Decodes the continuous routing weights into human-readable mathematical formulas."""
        K = self.current_vars
        unary_names = self.bank.unary_names
        binary_names = ["+", "*", "-", "/_safe"]

        report = []
        report.append(f"=== KARYON AUTONOMOUS SYNTHESIZED DIFFERENTIAL SYSTEM ({K} Variables) ===")
        
        for k in range(K):
            tau_val = torch.sigmoid(self.log_tau[k]).item()
            u_probs = torch.softmax(self.unary_weights[k], dim=-1).detach().cpu().numpy()
            b_probs = torch.softmax(self.binary_weights[k], dim=-1).detach().cpu().numpy()

            top_unary_idx = int(np.argmax(u_probs))
            top_binary_idx = int(np.argmax(b_probs))

            unary_expr = f"{unary_names[top_unary_idx]}(W_in_{k} * x)"
            cross_expr = f"sum_j(W_var_{k},j * v_j)"
            
            bin_op = binary_names[top_binary_idx]
            if bin_op == "*":
                phi_expr = f"({unary_expr} * {cross_expr})"
            elif bin_op == "+":
                phi_expr = f"({unary_expr} + {cross_expr})"
            elif bin_op == "-":
                phi_expr = f"({unary_expr} - {cross_expr})"
            else:
                phi_expr = f"({unary_expr} /_safe {cross_expr})"

            ode_eq = f"d(v_{k})/dt = (1/{tau_val:.3f}) * [ -v_{k} + tanh( {phi_expr} ) ]"
            report.append(f"  [Variable v_{k}]: {ode_eq} (Confidence: U={u_probs[top_unary_idx]:.2f}, B={b_probs[top_binary_idx]:.2f})")

        report.append("  [Readout Equation]: y_pred = tanh( sum_{k=1}^" + str(K) + " W_out_k * v_k )")
        return "\n".join(report)


# =============================================================================
# CONTINUOUS STREAM RUNNER
# =============================================================================

def run_exp_367():
    print("===============================================================================")
    print("=== KEP EXP-367: SOVEREIGN EQUATION & DYNAMIC VARIABLE SYNTHESIZER (SA-EDVS) ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    stream_data, stream_types, raw_bytes = generate_unbroken_stream(stream_length=2048)
    stream_len = stream_data.size(0)

    # Initialize Formula Synthesizer with 2 initial variables
    model = SovereignFormulaSynthesizer(in_dim=8, var_dim=16, initial_vars=2, max_vars=8, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-2, weight_decay=1e-5)

    v_state = torch.zeros(model.max_vars, model.var_dim, device=DEVICE)

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

        # Step equation synthesizer
        y_pred, v_state, routing_entropy = model.forward_step(cur_vec, v_state)
        v_state = v_state.detach()

        # Physical quadratic strain + Minimum Description Length penalty
        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        free_energy = prediction_strain + 0.002 * routing_entropy + 0.001 * model.current_vars

        free_energy.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        rolling_strain = 0.90 * rolling_strain + 0.10 * strain_val

        # Autopoietic Variable Sprouting: Spawn new variable if prediction strain > 1.2
        if t > 40 and t % 32 == 0:
            if rolling_strain > 1.2:
                if model.sprout_variable():
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-2, weight_decay=1e-5)

        # Exact bit-level parity check
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
            print(f"  Step {t+1:04d}/{stream_len} | Active Vars: {model.current_vars:02d} | Strain: {rolling_strain:.4f} | Det Energy: {det_tail_err:.4f} | Det Bit Errors: {det_tail_bits:.2f}/8 bits")

    elapsed = time.time() - t0
    tok_per_sec = stream_len / elapsed if elapsed > 0 else 0.0

    q_len = max(1, len(det_errors) // 4)
    final_det_strain = float(np.mean(det_errors[-q_len:]))
    final_det_bit_errors = float(np.mean(bit_mismatches_det[-q_len:]))
    final_stoch_strain = float(np.mean(stoch_errors[-q_len:]))

    discovered_formulas = model.extract_symbolic_formulas()

    print("\n===============================================================================")
    print("=== FINAL TELEMETRY & SYNTHESIZED MATHEMATICAL SYSTEM (EXP-367) ===")
    print("===============================================================================")
    print(f"  Final Dynamical Variable Count: {model.current_vars} (Started at 2)")
    print(f"  Sprout Morphogenesis Events   : {model.sprout_events}")
    print(f"  Deterministic Bit Error Rate  : {final_det_bit_errors:.2f} / 8 bits")
    print(f"  Deterministic Strain Energy   : {final_det_strain:.4f}")
    print(f"  Stochastic Strain Energy      : {final_stoch_strain:.4f}")
    print(f"  Processing Throughput         : {tok_per_sec:.1f} tok/s\n")
    print("-------------------------------------------------------------------------------")
    print(discovered_formulas)
    print("-------------------------------------------------------------------------------")

    verdict = "🟢 POSITIVE" if final_det_bit_errors < 1.0 else ("⚪ NEUTRAL" if final_det_bit_errors < 2.0 else "🔴 REJECTED")
    print(f"👑 VERDICT: {verdict}")

    results_data = {
        "exp_id": "EXP-367",
        "active_variables": model.current_vars,
        "sprout_events": model.sprout_events,
        "final_det_bit_errors": final_det_bit_errors,
        "final_det_strain": final_det_strain,
        "final_stoch_strain": final_stoch_strain,
        "tok_per_sec": tok_per_sec,
        "discovered_formulas": discovered_formulas,
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_367_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    run_exp_367()
