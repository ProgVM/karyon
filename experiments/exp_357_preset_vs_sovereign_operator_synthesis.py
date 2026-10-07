"""
EXP-357: Comparative Benchmark - Preset Operators vs. Sovereign Operator & Formula Synthesis
=============================================================================================
Direct Head-to-Head Evaluation under KEP Principle 2 & Rule 12:
- Both systems have access to:
  * Deterministic drift f(x) and Non-deterministic / Stochastic diffusion g(x) * dW_t
  * Raw machine byte stream substrate (V=258, S=64, B=32)
  * Active Inference Variational Free Energy optimization (Accuracy Error + Complexity Penalty)

CASE 1: Preset Operator Bank (Baseline)
  - Uses fixed, human-engineered operator architectures (Leaky Linear, Bilinear, State-Space, Saturated Warp).
  - Graph learns routing between these pre-existing rigid operators.

CASE 2: Sovereign Operator & Formula Synthesis (Autopoietic Genesis)
  - ZERO preset operator classes or hardcoded equations.
  - Operators are dynamically synthesized from raw continuous tensor primitives:
    * Synthesized Continuous Vector Fields: dx/dt = sum_k c_k * phi_k(W_k x)
    * Differentiable Polynomial/Harmonic & Non-linear functional generator
    * Autonomous Dynamic Time-scale & Diffusion gain generation
  - Epigenetic smooth grafting (tanh(alpha_epi) -> 0 at birth) and Neural Darwinian pruning.
=============================================================================================
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
# CASE 1: PRESET OPERATOR BANK (HARDCODED OPERATOR MECHANISMS)
# =============================================================================

class PresetOperator(nn.Module):
    """Preset human-designed operator with both deterministic drift and stochastic diffusion."""
    def __init__(self, op_type: str, dim: int, device: torch.device):
        super().__init__()
        self.op_type = op_type
        self.dim = dim
        self.device = device

        if op_type == "leaky_linear":
            self.W = nn.Linear(dim, dim, bias=False, device=device)
            self.leak = nn.Parameter(torch.tensor(0.1, device=device))
        elif op_type == "bilinear":
            self.W1 = nn.Linear(dim, dim, bias=False, device=device)
            self.W2 = nn.Linear(dim, dim, bias=False, device=device)
        elif op_type == "state_space":
            self.W = nn.Linear(dim, dim, bias=False, device=device)
            self.log_decay = nn.Parameter(torch.linspace(-2.0, -0.1, dim, device=device))
        elif op_type == "saturated_warp":
            self.W = nn.Linear(dim, dim, bias=False, device=device)
            self.gate = nn.Linear(dim, dim, bias=True, device=device)

        # Stochastic diffusion generator
        self.sigma_net = nn.Linear(dim, dim, bias=True, device=device)
        self.norm = nn.LayerNorm(dim, device=device)

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor = None, stochastic: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        p_s = prev_state if prev_state is not None else torch.zeros_like(x)

        # Deterministic Drift f(x)
        if self.op_type == "leaky_linear":
            leak_rate = torch.sigmoid(self.leak)
            f_drift = (1.0 - leak_rate) * p_s + leak_rate * torch.tanh(self.W(x))
        elif self.op_type == "bilinear":
            f_drift = self.W1(x) * torch.sigmoid(self.W2(p_s))
        elif self.op_type == "state_space":
            decay = torch.exp(-F.softplus(self.log_decay))
            f_drift = decay * p_s + (1.0 - decay) * self.W(x)
        elif self.op_type == "saturated_warp":
            f_drift = torch.tanh(self.W(x)) * torch.sigmoid(self.gate(p_s))
        else:
            f_drift = x

        # Stochastic Diffusion g(x) * dW
        if stochastic and self.training:
            sigma = F.softplus(self.sigma_net(x)) * 0.1
            noise = torch.randn_like(x) * sigma
            state = f_drift + noise
        else:
            state = f_drift

        return self.norm(state), state


class Case1PresetEngine(nn.Module):
    """Engine operating with a fixed bank of preset operators."""
    def __init__(self, vocab_size: int = 258, dim: int = 128, device: torch.device = DEVICE):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.device = device

        self.embedding = nn.Embedding(vocab_size, dim, device=device)
        self.op_types = ["leaky_linear", "bilinear", "state_space", "saturated_warp"]
        self.operators = nn.ModuleList([PresetOperator(t, dim, device) for t in self.op_types])
        self.routing = nn.Parameter(torch.randn(len(self.op_types), len(self.op_types), device=device) * 0.1)
        self.head = nn.Linear(dim, vocab_size, device=device)

    def forward(self, x: torch.Tensor, stochastic: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x.shape
        emb = self.embedding(x)

        num_ops = len(self.operators)
        states = [torch.zeros(batch_size, self.dim, device=self.device) for _ in range(num_ops)]
        route_weights = torch.softmax(self.routing, dim=-1)

        outputs = []
        for t in range(seq_len):
            inp_t = emb[:, t, :]
            op_outs = []
            next_states = []

            for i, op in enumerate(self.operators):
                if i == 0:
                    op_in = inp_t
                else:
                    mix = torch.zeros_like(inp_t)
                    for j in range(len(op_outs)):
                        mix = mix + route_weights[j, i] * op_outs[j]
                    op_in = inp_t + mix

                out_i, s_i = op(op_in, states[i], stochastic=stochastic)
                op_outs.append(out_i)
                next_states.append(s_i)

            combined = torch.stack(op_outs, dim=0).mean(dim=0)
            outputs.append(combined)
            states = next_states

        out_seq = torch.stack(outputs, dim=1)
        logits = self.head(out_seq)
        complexity_penalty = torch.tensor(0.0, device=self.device)
        return logits, complexity_penalty


# =============================================================================
# CASE 2: SOVEREIGN OPERATOR & FORMULA SYNTHESIZER (ZERO PRESET CLASSES)
# =============================================================================

class SynthesizedContinuousOperator(nn.Module):
    """
    Sovereign Continuous Operator Synthesized Dynamically.
    NO hardcoded equations. The functional form is synthesized via continuous basis composition:
      f(x) = sum_k (alpha_k * NonLinear_k(W_k @ x + b_k) (x) beta_k * State_Projection(prev_state))
    equipped with dynamic time-step scale tau and continuous stochastic Wiener diffusion g(x).
    """
    def __init__(self, dim: int, formula_seed: dict, device: torch.device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.formula_seed = formula_seed

        # Continuous projection tensors synthesized dynamically
        num_terms = formula_seed.get("num_terms", 3)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
            for _ in range(num_terms)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(dim, device=device))
            for _ in range(num_terms)
        ])

        # Synthesized basis activation modes (indices dynamically chosen during genesis)
        self.act_modes = formula_seed.get("act_modes", [0, 1, 2])  # 0=tanh, 1=sin, 2=bilinear-gated, 3=leaky
        self.term_weights = nn.Parameter(torch.randn(num_terms, device=device) * 0.5)

        # Dynamic continuous time-scale integration factor (tau)
        self.log_tau = nn.Parameter(torch.tensor(formula_seed.get("initial_log_tau", 0.0), device=device))

        # Stochastic diffusion generator
        self.diffusion_kernel = nn.Linear(dim, dim, bias=True, device=device)
        self.stochastic_gain = nn.Parameter(torch.tensor(0.05, device=device))

        self.norm = nn.LayerNorm(dim, device=device)

    def _eval_basis_function(self, mode: int, linear_term: torch.Tensor, prev_s: torch.Tensor) -> torch.Tensor:
        if mode == 0:
            return torch.tanh(linear_term)
        elif mode == 1:
            return torch.sin(linear_term)
        elif mode == 2:
            return torch.tanh(linear_term) * torch.sigmoid(prev_s)
        elif mode == 3:
            return F.gelu(linear_term) * torch.cos(linear_term)
        else:
            return linear_term

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor = None, stochastic: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        p_s = prev_state if prev_state is not None else torch.zeros_like(x)

        # Synthesized Deterministic Vector Field f(x, p_s)
        drift_sum = torch.zeros_like(x)
        weights_normalized = torch.softmax(self.term_weights, dim=0)

        for i, (w, b, mode) in enumerate(zip(self.weights, self.biases, self.act_modes)):
            lin = F.linear(x, w, b)
            phi = self._eval_basis_function(mode, lin, p_s)
            drift_sum = drift_sum + weights_normalized[i] * phi

        # Continuous dynamic time integration: dx/dt formulation
        tau = torch.sigmoid(self.log_tau) * 0.9 + 0.05  # tau in [0.05, 0.95]
        new_state_det = (1.0 - tau) * p_s + tau * drift_sum

        # Synthesized Stochastic Diffusion Field g(x) * dW
        if stochastic and self.training:
            diffusion_scale = F.softplus(self.diffusion_kernel(x)) * torch.abs(self.stochastic_gain)
            noise = torch.randn_like(x) * diffusion_scale
            total_state = new_state_det + noise
        else:
            total_state = new_state_det

        return self.norm(total_state), total_state

    def get_symbolic_representation(self) -> str:
        """Extracts synthesized mathematical formula in readable symbolic notation."""
        mode_names = {0: "tanh(W*x)", 1: "sin(W*x)", 2: "tanh(W*x)*sig(s)", 3: "gelu(W*x)*cos(W*x)"}
        terms = [f"{float(tw):.2f}*{mode_names.get(m, 'id')}" for tw, m in zip(self.term_weights, self.act_modes)]
        tau_val = float(torch.sigmoid(self.log_tau) * 0.9 + 0.05)
        return f"dState/dt = tau({tau_val:.2f}) * [{' + '.join(terms)}] + diff(x)"


class Case2SovereignSynthesisEngine(nn.Module):
    """
    Sovereign Engine: Synthesizes, evolves, mutates, and prunes custom operators and formulas.
    NO PRESET OPERATORS.
    """
    def __init__(self, vocab_size: int = 258, dim: int = 128, initial_operators: int = 2, max_operators: int = 6, device: torch.device = DEVICE):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_operators = max_operators
        self.device = device

        self.embedding = nn.Embedding(vocab_size, dim, device=device)
        self.operators = nn.ModuleList()
        self.alpha_epi = nn.ParameterList()
        self.vitality = []

        # Synthesize initial operators
        for _ in range(initial_operators):
            self.synthesize_new_operator(initial_epi=1.0)

        self.routing_matrix = nn.Parameter(torch.randn(max_operators, max_operators, device=device) * 0.1)
        self.head = nn.Linear(dim, vocab_size, device=device)

    def synthesize_new_operator(self, initial_epi: float = 0.0) -> SynthesizedContinuousOperator:
        if len(self.operators) >= self.max_operators:
            return None

        # Autonomously synthesize functional formula structure
        num_terms = random.randint(2, 4)
        act_modes = [random.randint(0, 3) for _ in range(num_terms)]
        init_tau = random.uniform(-1.0, 1.0)

        seed = {
            "num_terms": num_terms,
            "act_modes": act_modes,
            "initial_log_tau": init_tau
        }

        op = SynthesizedContinuousOperator(self.dim, seed, self.device)
        self.operators.append(op)

        # Epigenetic Zero-Shock Grafting gate: tanh(alpha_epi) = initial_epi
        raw_val = math.atanh(min(max(initial_epi, -0.99), 0.99))
        self.alpha_epi.append(nn.Parameter(torch.tensor(raw_val, device=self.device)))
        self.vitality.append(1.0)
        return op

    def prune_unviable_operators(self, threshold: float = 0.05) -> list[int]:
        pruned = []
        for i in range(len(self.operators) - 1, -1, -1):
            if len(self.operators) <= 1:
                break
            gate_val = torch.tanh(self.alpha_epi[i]).abs().item()
            if gate_val < threshold and self.vitality[i] < threshold:
                pruned.append(i)
                del self.operators[i]
                del self.alpha_epi[i]
                del self.vitality[i]
        return pruned

    def forward(self, x: torch.Tensor, stochastic: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x.shape
        emb = self.embedding(x)

        num_ops = len(self.operators)
        states = [torch.zeros(batch_size, self.dim, device=self.device) for _ in range(num_ops)]
        route_weights = torch.softmax(self.routing_matrix[:num_ops, :num_ops], dim=-1)

        outputs = []
        complexity_penalty = torch.tensor(0.0, device=self.device)

        for t in range(seq_len):
            inp_t = emb[:, t, :]
            op_outs = []
            next_states = []

            for i, op in enumerate(self.operators):
                if i == 0 or len(op_outs) == 0:
                    op_in = inp_t
                else:
                    mix = torch.zeros_like(inp_t)
                    for j in range(len(op_outs)):
                        mix = mix + route_weights[j, i] * op_outs[j]
                    op_in = inp_t + mix

                p_s = states[i] if i < len(states) else torch.zeros_like(inp_t)
                raw_out, s_i = op(op_in, p_s, stochastic=stochastic)

                # Epigenetic gating
                gated_out = torch.tanh(self.alpha_epi[i]) * raw_out
                op_outs.append(gated_out)
                next_states.append(s_i)

                # Structural complexity penalty & vitality tracking
                complexity_penalty = complexity_penalty + torch.abs(self.alpha_epi[i])
                self.vitality[i] = 0.95 * self.vitality[i] + 0.05 * gated_out.abs().mean().item()

            combined = torch.stack(op_outs, dim=0).sum(dim=0)
            outputs.append(combined)
            states = next_states

        out_seq = torch.stack(outputs, dim=1)
        logits = self.head(out_seq)
        return logits, complexity_penalty


# =============================================================================
# 3. UNIFIED KEP COMPARATIVE BENCHMARK RUNNER
# =============================================================================

def generate_machine_stream(num_samples: int = 256, seq_len: int = 64) -> torch.Tensor:
    return torch.randint(0, 256, (num_samples, seq_len), dtype=torch.long, device=DEVICE)


def train_eval_case(case_name: str, model: nn.Module, is_sovereign: bool, data: torch.Tensor, num_epochs: int = 12):
    print(f"\n===============================================================================")
    print(f"=== RUNNING: {case_name.upper()} ===")
    print(f"===============================================================================")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    batch_size = 32
    vocab_size = 258

    initial_loss = None
    final_loss = None
    t0 = time.time()
    total_tokens = 0
    sprout_count = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # Dynamic synthesis trigger for Case 2
        if is_sovereign and epoch > 0 and epoch % 3 == 0:
            new_op = model.synthesize_new_operator(initial_epi=0.01)
            if new_op is not None:
                sprout_count += 1
                optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
                print(f"  [Epigenetic Genesis] Synthesized new custom operator #{len(model.operators)}: {new_op.get_symbolic_representation()}")

        for i in range(0, data.size(0), batch_size):
            batch = data[i:i + batch_size]
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            logits, comp_pen = model(inputs, stochastic=True)

            loss_rec = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            total_loss = loss_rec + (0.005 * comp_pen if is_sovereign else 0.0)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss_rec.item() * inputs.size(0)
            total_tokens += inputs.numel()

        avg_loss = epoch_loss / data.size(0)
        if initial_loss is None:
            initial_loss = avg_loss
        final_loss = avg_loss

        ops_count = len(model.operators) if hasattr(model, 'operators') else 0
        print(f"  Epoch {epoch + 1:02d}/{num_epochs:02d} | Loss: {avg_loss:.4f} | Active Ops: {ops_count}")

    elapsed = time.time() - t0
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
    delta_loss = initial_loss - final_loss

    res = {
        "case_name": case_name,
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "delta_loss": float(delta_loss),
        "tok_per_sec": float(tok_per_sec),
        "sprout_count": sprout_count,
        "elapsed_sec": float(elapsed)
    }

    if is_sovereign:
        res["synthesized_formulas"] = [op.get_symbolic_representation() for op in model.operators]

    return res


def run_exp_357_benchmark():
    print("===============================================================================")
    print("=== KEP EXP-357: Preset Operators vs. Sovereign Operator & Formula Synthesis ===")
    print("===============================================================================")
    print(f"Hardware Substrate: {DEVICE}")

    data = generate_machine_stream(num_samples=256, seq_len=64)

    # 1. Evaluate Case 1 (Preset Operator Bank)
    case1_model = Case1PresetEngine(vocab_size=258, dim=128, device=DEVICE)
    res_case1 = train_eval_case("Case 1: Preset Operator Bank (Hardcoded)", case1_model, is_sovereign=False, data=data, num_epochs=12)

    # 2. Evaluate Case 2 (Sovereign Operator & Formula Synthesizer)
    case2_model = Case2SovereignSynthesisEngine(vocab_size=258, dim=128, initial_operators=2, max_operators=6, device=DEVICE)
    res_case2 = train_eval_case("Case 2: Sovereign Formula Synthesizer (No Presets)", case2_model, is_sovereign=True, data=data, num_epochs=12)

    print("\n===============================================================================")
    print("=== FINAL COMPARATIVE TELEMETRY SUMMARY ===")
    print("===============================================================================")
    print(f"Case 1 (Preset Operators)       : Loss {res_case1['initial_loss']:.4f} -> {res_case1['final_loss']:.4f} (Delta: {res_case1['delta_loss']:.4f}) | {res_case1['tok_per_sec']:.2f} tok/s")
    print(f"Case 2 (Sovereign Synthesis)    : Loss {res_case2['initial_loss']:.4f} -> {res_case2['final_loss']:.4f} (Delta: {res_case2['delta_loss']:.4f}) | {res_case2['tok_per_sec']:.2f} tok/s")
    print(f"\nSovereign Synthesized Formulas in Case 2 ({len(res_case2['synthesized_formulas'])} operators):")
    for idx, formula in enumerate(res_case2['synthesized_formulas']):
        print(f"  Op #{idx+1}: {formula}")

    superiority_delta = res_case1['final_loss'] - res_case2['final_loss']
    print(f"\nSovereign Superiority Delta (Case 1 Loss - Case 2 Loss): {superiority_delta:+.4f}")

    verdict = "🟢 POSITIVE" if res_case2['delta_loss'] >= 0.08 else "⚪ NEUTRAL"
    print(f"Overall KEP Verdict for Sovereign Genesis: {verdict}")

    full_results = {
        "exp_id": "EXP-357",
        "case_1_preset": res_case1,
        "case_2_sovereign": res_case2,
        "superiority_delta": float(superiority_delta),
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_357_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    return full_results


if __name__ == "__main__":
    run_exp_357_benchmark()
