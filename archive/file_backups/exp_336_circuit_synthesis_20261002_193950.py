"""
=====================================================================================
EXP-336: ENDOGENOUS CIRCUIT SYNTHESIS FROM ELEMENTARY MATHEMATICAL ATOMS
=====================================================================================
Synthesizes dynamic biophysical circuits (e.g. Bistable Double-Well Latches,
Nonlinear Oscillators, and Evidence Accumulators) strictly from 4 atomic mathematical primitives:
  1. Continuous Integrator (LinearAccumulatorOp / leaky state integration)
  2. Bilinear Multiplier (BilinearMultiplicativeOp / multiplicative conjunction)
  3. Linear Projector (LinearOp / weight transformation)
  4. Stochastic Langevin Noise (StochasticLangevinOp)

Validates emergence of bistable carry latching without any hand-crafted physical classes.
=====================================================================================
"""
import os
import sys
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))

import karyon_core
from karyon_agent import CoREAgent


# =====================================================================================
# SYNTHESIZED MICRO-CIRCUIT MOTIF (EMERGENT FROM ELEMENTARY ATOMS)
# =====================================================================================
class AtomicCircuitMotif(nn.Module):
    """
    Micro-Graph assembled from elementary mathematical atoms:
      - Atom 1: Linear Projector (W_proj * x + b)
      - Atom 2: Bilinear Multiplier (u * v)
      - Atom 3: Continuous Leaky Integrator (dx/dt = f(x, u))
      - Atom 4: Langevin Stochastic Perturbation (xi ~ N(0, sigma^2))
    
    Under Free Energy minimization, this circuit endogenously discovers
    the nonlinear feedback loop required for bistable potential wells (e.g. cubic damping: x * x * x).
    """
    def __init__(self, dim: int, num_internal_atoms: int = 4, device_str: str = "cpu"):
        super().__init__()
        self.dim = dim
        self.device = torch.device(device_str)
        self.num_internal_atoms = num_internal_atoms

        # Elementary Atom 1: Linear Input/Feedback Projectors
        self.atom_proj_in = nn.Linear(dim, dim)
        self.atom_proj_recurrent = nn.Linear(dim, dim)
        
        # Elementary Atom 2: Bilinear Multiplicative Atoms (Inter-node conjunctions)
        self.atom_mult_1 = nn.Linear(dim, dim, bias=False)
        self.atom_mult_2 = nn.Linear(dim, dim, bias=False)
        self.atom_mult_gate = nn.Linear(dim, dim, bias=False)

        # Elementary Atom 3: Continuous Leaky Integrator Parameters (Dynamic time constant dt)
        self.w_dt = nn.Linear(dim, dim)
        self.w_decay = nn.Linear(dim, dim)

        # Elementary Atom 4: Stochastic Langevin Noise Gain
        self.w_noise = nn.Linear(dim, dim)

        # Internal state buffer for continuous integration
        self.register_buffer("circuit_state", torch.zeros(1, dim))

    def reset_state(self, batch_size: int = 1):
        self.circuit_state = torch.zeros(batch_size, self.dim, device=self.circuit_state.device)

    def set_state(self, initial_state: torch.Tensor):
        self.circuit_state = initial_state.clone()

    def forward(self, x_in: torch.Tensor, steps: int = 3) -> torch.Tensor:
        """
        Executes continuous internal recurrence over elementary atoms.
        """
        B = x_in.size(0)
        if self.circuit_state.size(0) != B:
            self.reset_state(B)

        x_curr = self.circuit_state

        for _ in range(steps):
            # Atom 1: Linear projections of sensory input and recurrent state
            u_in = self.atom_proj_in(x_in)
            u_rec = self.atom_proj_recurrent(x_curr)

            # Atom 2: Multiplicative conjunctions (Enables cubic/polynomial potential synthesis: x * x * x)
            # Motif: m1 = W1(x), m2 = W2(x) -> non-linear product
            m1 = self.atom_mult_1(x_curr)
            m2 = self.atom_mult_2(x_curr)
            mult_interaction = m1 * m2 # [B, dim]

            # Gated conjunction
            gate = torch.sigmoid(self.atom_mult_gate(x_curr))
            conjunction_flux = gate * mult_interaction

            # Atom 3: Continuous Integration Flux
            # dx/dt = Linear_Feedback + Conjunction_Flux + External_Input - Decay * x
            decay = torch.nn.functional.softplus(self.w_decay(x_curr)) + 1e-4
            dt = torch.sigmoid(self.w_dt(x_curr)) * 0.4 + 0.05 # Dynamic adaptive sub-step

            # Atom 4: Langevin Stochastic Perturbation
            sigma = torch.nn.functional.softplus(self.w_noise(x_curr)) * 0.01
            noise = torch.randn_like(x_curr) * sigma

            dx_dt = u_in + u_rec + conjunction_flux - (decay * x_curr) + noise
            
            # Predictor-Corrector Euler integration
            x_curr = torch.clamp(x_curr + dt * dx_dt, -3.0, 3.0)

        self.circuit_state = x_curr.detach()
        return x_curr


# =====================================================================================
# TRACK 1: AUTONOMOUS CIRCUIT-LEVEL ARITHMETIC WITH BISTABLE LATCHING
# =====================================================================================
def run_track_1_circuit_latch(device_str: str) -> Dict[str, any]:
    print("\n" + "=" * 85)
    print("TRACK 1: ENDOGENOUS CIRCUIT SYNTHESIS FOR BISTABLE CARRY LATCHING")
    print("=" * 85)

    device = torch.device(device_str)
    dim = 16

    # Model composed of the synthesized atomic circuit motif
    class CircuitAdditionEngine(nn.Module):
        def __init__(self, dim, device_str):
            super().__init__()
            self.dim = dim
            self.digit_embed = nn.Embedding(10, dim)
            self.circuit_latch = AtomicCircuitMotif(dim=dim, device_str=device_str)
            self.w_add = nn.Linear(dim * 2 + dim, dim)
            self.w_sum = nn.Linear(dim, 10)
            self.w_circuit_in = nn.Linear(dim, dim)

        def forward(self, d1, d2, carry_state_in):
            e1 = self.digit_embed(d1)
            e2 = self.digit_embed(d2)
            
            # Combine sensory digits and current bistable latch state
            h_in = torch.cat([e1, e2, carry_state_in], dim=-1)
            h = torch.relu(self.w_add(h_in))
            sum_logits = self.w_sum(h)

            # Drive circuit motif with sum representation
            circuit_in = self.w_circuit_in(h)
            next_carry_state = self.circuit_latch(circuit_in, steps=3)
            return sum_logits, next_carry_state

    model = CircuitAdditionEngine(dim, device_str).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("  • Training Endogenous Circuit Motif on Stream Addition...")
    for step in range(1, 501):
        bs = 64
        d1 = torch.randint(0, 10, (bs,), device=device)
        d2 = torch.randint(0, 10, (bs,), device=device)

        # Binary carry truth
        c_in_binary = torch.randint(0, 2, (bs, 1), device=device).float()
        # Bistable representation (+1.0 for carry, -1.0 for no-carry)
        c_in_state = (c_in_binary * 2.0) - 1.0
        c_in_tensor = c_in_state.repeat(1, dim)

        raw_sum = d1 + d2 + c_in_binary.squeeze(-1).long()
        target_digit = raw_sum % 10
        target_carry_binary = (raw_sum >= 10).float().unsqueeze(-1)
        target_carry_tensor = ((target_carry_binary * 2.0) - 1.0).repeat(1, dim)

        model.circuit_latch.set_state(c_in_tensor)
        sum_logits, next_carry_state = model(d1, d2, c_in_tensor)

        loss_sum = nn.functional.cross_entropy(sum_logits, target_digit)
        loss_latch = nn.functional.mse_loss(next_carry_state, target_carry_tensor)
        total_loss = loss_sum + 2.0 * loss_latch

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Sequential addition evaluation across 4-digit numbers
    print("  • Evaluating 4-Digit Stream Addition via Emergent Circuit Latch...")
    num_tests = 250
    num_digits = 4
    correct_full_additions = 0
    correct_carry_digits = 0
    total_carry_instances = 0

    latch_trajectories = []

    for _ in range(num_tests):
        n1 = random.randint(1000, 9999)
        n2 = random.randint(1000, 9999)
        true_result = n1 + n2

        s1 = str(n1)[::-1]
        s2 = str(n2)[::-1]

        # Initial state: 0 carry (-1.0 basin)
        carry_tensor = torch.full((1, dim), -1.0, device=device)
        model.circuit_latch.set_state(carry_tensor)

        pred_digits = []
        for pos in range(num_digits):
            d1_val = torch.tensor([int(s1[pos])], device=device)
            d2_val = torch.tensor([int(s2[pos])], device=device)

            with torch.no_grad():
                sum_logits, carry_tensor = model(d1_val, d2_val, carry_tensor)
                pred_d = torch.argmax(sum_logits, dim=-1).item()
                pred_digits.append(pred_d)

                mean_val = carry_tensor.mean().item()
                is_carry = (mean_val > 0.0)
                latch_trajectories.append(mean_val)

                # Verification
                carry_in_num = 1 if len(latch_trajectories) > 1 and latch_trajectories[-2] > 0.0 else 0
                expected_sum = int(s1[pos]) + int(s2[pos]) + carry_in_num
                expected_d = expected_sum % 10
                expected_carry = (expected_sum >= 10)

                if expected_carry:
                    total_carry_instances += 1
                    if is_carry and pred_d == expected_d:
                        correct_carry_digits += 1

        if carry_tensor.mean().item() > 0.0:
            pred_digits.append(1)

        pred_num = int("".join(str(d) for d in pred_digits[::-1]))
        if pred_num == true_result:
            correct_full_additions += 1

    carry_acc = (correct_carry_digits / max(1, total_carry_instances)) * 100.0
    full_acc = (correct_full_additions / num_tests) * 100.0

    print(f"  ✓ Full 4-Digit Number Addition Accuracy : {full_acc:.2f}%")
    print(f"  ✓ Emergent Circuit Latching Accuracy    : {carry_acc:.2f}% (EXP-334: 83.50%, EXP-335: 36.23%)")

    return {
        "full_acc": full_acc,
        "carry_acc": carry_acc,
        "latch_trajectories": latch_trajectories[:120]
    }


# =====================================================================================
# TRACK 2: PHASE-SPACE RECIRCULATION & EVIDENCE INTEGRATION CIRCUIT
# =====================================================================================
def run_track_2_evidence_circuit(device_str: str) -> Dict[str, any]:
    print("\n" + "=" * 85)
    print("TRACK 2: EVIDENCE INTEGRATION & DECISION CIRCUIT FROM ATOMIC PRIMITIVES")
    print("=" * 85)

    device = torch.device(device_str)
    dim = 16

    # Assemble an Evidence Integration Circuit from atoms
    class EvidenceIntegrationCircuit(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.atom_drift_proj = nn.Linear(dim, dim)
            self.atom_mult = nn.Linear(dim, dim, bias=False)
            self.w_dt = nn.Linear(dim, 1)
            self.w_readout = nn.Linear(dim, 2)

        def forward(self, x_in, max_cycles=6):
            B = x_in.size(0)
            # Continuous accumulator state
            state = torch.zeros(B, self.dim, device=x_in.device)
            steps_taken = torch.zeros(B, 1, device=x_in.device)

            for _ in range(max_cycles):
                drift = self.atom_drift_proj(x_in)
                nonlin = drift * torch.tanh(self.atom_mult(state))
                dt = torch.sigmoid(self.w_dt(state)) * 0.5 + 0.1
                
                # Dynamic confidence-based halting threshold
                confidence = torch.norm(state, dim=-1, keepdim=True)
                active = (confidence < 2.5).float()
                
                state = state + active * dt * (drift + nonlin)
                steps_taken = steps_taken + active

            logits = self.w_readout(state)
            return logits, steps_taken, state

    circuit = EvidenceIntegrationCircuit(dim).to(device)
    optimizer = optim.Adam(circuit.parameters(), lr=0.01)

    print("  • Training Atomic Evidence Circuit...")
    for step in range(1, 401):
        xb = (torch.rand(64, dim, device=device) - 0.5) * 3.0
        z = xb.mean(dim=-1, keepdim=True)
        target = (z > 0.0).long().squeeze(-1)

        logits, steps, _ = circuit(xb)
        loss = nn.functional.cross_entropy(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    total_eval = 200
    x_test = (torch.rand(total_eval, dim, device=device) - 0.5) * 3.0
    z_test = x_test.mean(dim=-1, keepdim=True)
    targets = (z_test > 0.0).long().squeeze(-1)

    with torch.no_grad():
        logits_test, steps_test, states_test = circuit(x_test)
        pred = torch.argmax(logits_test, dim=-1)
        acc = (pred == targets).float().mean().item() * 100.0

    print(f"  ✓ Evidence Circuit Branch Selection Accuracy : {acc:.2f}% (EXP-334: 98.50%)")
    print(f"  ✓ Mean Recirculation Cycles                  : {steps_test.mean().item():.2f}")

    return {
        "branch_acc": acc,
        "z_vals": z_test.squeeze(-1).cpu().numpy(),
        "steps_taken": steps_test.squeeze(-1).cpu().numpy(),
        "pred_probs": torch.softmax(logits_test, dim=-1)[:, 1].cpu().numpy()
    }


# =====================================================================================
# MAIN EXPERIMENT RUNNER & VERDICT SYNTHESIS
# =====================================================================================
def run_exp_336():
    print("=" * 85)
    print("EXP-336: ENDOGENOUS CIRCUIT SYNTHESIS FROM ELEMENTARY ATOMS")
    print("=" * 85)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Substrate Compute Device: {device_str.upper()}")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t1_res = run_track_1_circuit_latch(device_str)
    t2_res = run_track_2_evidence_circuit(device_str)

    # Diagnostic Visualization
    os.makedirs("experiments", exist_ok=True)
    plot_path = "experiments/exp_336_circuit_synthesis.png"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#121212")

    for ax in axes:
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444444")

    # Panel 1: Track 1 Emergent Bistable Latch Trajectory
    axes[0].plot(t1_res["latch_trajectories"], color="#00ffcc", linewidth=1.8, label="Circuit State <x>")
    axes[0].axhline(y=1.0, color="#ffbb00", linestyle=":", label="Carry Basin (+1.0)")
    axes[0].axhline(y=-1.0, color="#ff0055", linestyle=":", label="Zero Basin (-1.0)")
    axes[0].set_title(f"Track 1: Emergent Bistable Latch\nCarry Acc: {t1_res['carry_acc']:.2f}% | Full Add: {t1_res['full_acc']:.2f}%")
    axes[0].set_xlabel("Sequential Addition Step")
    axes[0].set_ylabel("Synthesized Latch Potential State")
    axes[0].legend(facecolor="#2a2a2a", labelcolor="white")
    axes[0].grid(True, color="#333333", linestyle=":")

    # Panel 2: Track 2 Evidence Integration Decision Boundary
    z_sorted = np.argsort(t2_res["z_vals"])
    axes[1].scatter(t2_res["z_vals"][z_sorted], t2_res["pred_probs"][z_sorted], color="#ff00bb", alpha=0.7, label="P(Branch 1 | z)")
    axes[1].axvline(x=0.0, color="#00ffcc", linestyle="--", label="Equilibrium Boundary (z=0)")
    axes[1].set_title(f"Track 2: Atomic Evidence Circuit\nBranch Acc: {t2_res['branch_acc']:.2f}%")
    axes[1].set_xlabel("Internal State Metric z")
    axes[1].set_ylabel("Readout Decision Probability")
    axes[1].legend(facecolor="#2a2a2a", labelcolor="white")
    axes[1].grid(True, color="#333333", linestyle=":")

    # Panel 3: Track 2 Dynamic Recirculation Steps K(z)
    axes[2].scatter(t2_res["z_vals"], t2_res["steps_taken"], color="#9900ff", alpha=0.6, label="Thinking Steps K")
    axes[2].set_title("Endogenous Chrono-Actuation K(z)\n(Dynamic Latent Thinking Cycles)")
    axes[2].set_xlabel("Internal State Metric z")
    axes[2].set_ylabel("Cycles Taken K")
    axes[2].legend(facecolor="#2a2a2a", labelcolor="white")
    axes[2].grid(True, color="#333333", linestyle=":")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  ✓ Multi-Panel Diagnostic Plot Saved: {plot_path}")

    # KEP Rule #2 Verdict Synthesis
    print("\n" + "=" * 85)
    print("EXP-336: FINAL SYNTHESIS & VERDICT")
    print("=" * 85)

    is_track1_pass = t1_res["carry_acc"] >= 95.0
    is_track2_pass = t2_res["branch_acc"] >= 99.0

    all_passed = is_track1_pass and is_track2_pass
    verdict = "🟢 POSITIVE" if all_passed else "🔴 REJECTED"

    print(f"VERDICT: {verdict}")
    print(f"  • Track 1 (Emergent Circuit Carry Latch Acc) : {t1_res['carry_acc']:.2f}% (EXP-334: 83.50%, EXP-335: 36.23%) -> {'PASS' if is_track1_pass else 'FAIL'}")
    print(f"  • Track 1 (Full 4-Digit Number Addition Acc) : {t1_res['full_acc']:.2f}% -> {'PASS' if t1_res['full_acc'] >= 90.0 else 'FAIL'}")
    print(f"  • Track 2 (Atomic Evidence Circuit Acc)      : {t2_res['branch_acc']:.2f}% (EXP-334 Baseline: 98.50%) -> {'PASS' if is_track2_pass else 'FAIL'}")
    print("=" * 85)


if __name__ == "__main__":
    run_exp_336()
