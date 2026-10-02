"""
=====================================================================================
EXP-335: CONTINUOUS DRIFT-DIFFUSION & LANDAU DOUBLE-WELL LATCH BENCHMARK
=====================================================================================
Continuous physical dynamics implementing self-regulating, non-constant thresholds:
  1. TRACK 1: Drift-Diffusion Evidence Accumulation with Adaptive Phase Recirculation
  2. TRACK 2: Landau Double-Well Potential Bistable Latch for Multi-digit Stream Addition
=====================================================================================
"""
import os
import sys
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))

import karyon_core
from karyon_agent import CoREAgent


# =====================================================================================
# TRACK 1: DRIFT-DIFFUSION ACCUMULATION & PHASE RECIRCULATION
# =====================================================================================
def run_track_1_drift_diffusion(device_str: str) -> Dict[str, float]:
    print("\n" + "=" * 85)
    print("TRACK 1: DRIFT-DIFFUSION EVIDENCE ACCUMULATION & PHASE RECIRCULATION")
    print("=" * 85)

    device = torch.device(device_str)
    dim = 16

    agent = CoREAgent(vocab_size=258, embed_dim=dim, device=device_str)
    agent.to(device)

    # Sprouts organelles
    roll_idx = agent.add_node("organelle_roll", "NonLinearTransform", is_core=True, initial_alpha=1.0)
    flip_idx = agent.add_node("organelle_flip", "NonLinearTransform", is_core=True, initial_alpha=1.0)

    param_map = agent.graph.named_parameters_map()

    def apply_roll(x):
        return torch.roll(x, shifts=2, dims=-1) * 1.05 + 0.05

    def apply_flip(x):
        return torch.flip(x, dims=[-1]) * -0.9 + torch.sin(x * 1.5) * 0.1

    # Pre-train roll
    opt_roll = optim.Adam([p for n, p in param_map.items() if f"node_{roll_idx}_" in n], lr=0.01)
    for _ in range(200):
        xb = (torch.rand(64, dim, device=device) - 0.5) * 3.0
        yb = apply_roll(xb)
        w_up = param_map[f"node_{roll_idx}_organelle_roll_w_up"]
        b_up = param_map[f"node_{roll_idx}_organelle_roll_b_up"]
        w_down = param_map[f"node_{roll_idx}_organelle_roll_w_down"]
        b_down = param_map[f"node_{roll_idx}_organelle_roll_b_down"]
        h = torch.matmul(xb, w_up.t()) + b_up
        pred = torch.matmul(nn.functional.gelu(h), w_down.t()) + b_down
        loss = nn.functional.mse_loss(pred, yb)
        opt_roll.zero_grad()
        loss.backward()
        opt_roll.step()
    agent.lock_node(roll_idx, 1.0)

    # Pre-train flip
    opt_flip = optim.Adam([p for n, p in param_map.items() if f"node_{flip_idx}_" in n], lr=0.01)
    for _ in range(200):
        xb = (torch.rand(64, dim, device=device) - 0.5) * 3.0
        yb = apply_flip(xb)
        w_up = param_map[f"node_{flip_idx}_organelle_flip_w_up"]
        b_up = param_map[f"node_{flip_idx}_organelle_flip_b_up"]
        w_down = param_map[f"node_{flip_idx}_organelle_flip_w_down"]
        b_down = param_map[f"node_{flip_idx}_organelle_flip_b_down"]
        h = torch.matmul(xb, w_up.t()) + b_up
        pred = torch.matmul(nn.functional.gelu(h), w_down.t()) + b_down
        loss = nn.functional.mse_loss(pred, yb)
        opt_flip.zero_grad()
        loss.backward()
        opt_flip.step()
    agent.lock_node(flip_idx, 1.0)

    # Drift-Diffusion Evidence Accumulator Model
    class DriftDiffusionModel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.w_drift = nn.Linear(dim, 1) # Maps state to drift rate mu
            self.w_noise = nn.Linear(dim, 1) # Dynamic noise scale based on state uncertainty
            self.w_decision = nn.Linear(1, 2)

        def forward(self, x_state, max_steps=8):
            B = x_state.size(0)
            # mu = drift rate
            mu = self.w_drift(x_state) # [B, 1]
            # Dynamic uncertainty-driven noise
            noise_scale = torch.softplus(self.w_noise(x_state)) + 1e-5 # [B, 1]

            theta = torch.zeros((B, 1), device=x_state.device)
            steps_taken = torch.zeros((B, 1), device=x_state.device)
            
            # Evidence accumulation loop (recirculation cycles)
            for k in range(max_steps):
                # Only update nodes that haven't hit threshold yet (continuous self-regulation)
                # Dynamic threshold derived from internal somatic state energy/confidence
                # No static constant threshold! Threshold is a function of mean absolute drift
                dynamic_threshold = torch.clamp(torch.abs(mu) * 1.5 + 0.1, 0.1, 2.0)
                
                # Active noise perturbation
                xi = torch.randn_like(theta) * noise_scale
                active_mask = (torch.abs(theta) < dynamic_threshold).float()
                
                theta = theta + active_mask * (mu + xi)
                steps_taken = steps_taken + active_mask

            logits = self.w_decision(theta)
            return logits, steps_taken, theta

    ddm = DriftDiffusionModel(dim).to(device)
    opt_ddm = optim.Adam(ddm.parameters(), lr=0.01)

    print("  • Training Drift-Diffusion Evidence Accumulator...")
    for step in range(1, 401):
        xb = (torch.rand(64, dim, device=device) - 0.5) * 3.0
        z = xb.mean(dim=-1, keepdim=True)
        target_branch = (z > 0.0).long().squeeze(-1)

        logits, steps, theta = ddm(xb)
        loss = nn.functional.cross_entropy(logits, target_branch)

        opt_ddm.zero_grad()
        loss.backward()
        opt_ddm.step()

    # Evaluation
    total_eval = 200
    x_test = (torch.rand(total_eval, dim, device=device) - 0.5) * 3.0
    z_test = x_test.mean(dim=-1, keepdim=True)
    true_branches = (z_test > 0.0).long().squeeze(-1)

    y_true = torch.zeros_like(x_test)
    for i in range(total_eval):
        if true_branches[i] == 1:
            y_true[i] = apply_roll(x_test[i:i+1])
        else:
            y_true[i] = apply_flip(x_test[i:i+1])

    with torch.no_grad():
        logits_test, steps_test, theta_test = ddm(x_test)
        probs = nn.functional.softmax(logits_test, dim=-1)
        pred_branches = torch.argmax(probs, dim=-1)

        correct_branches = (pred_branches == true_branches).float().sum().item()
        branch_acc = (correct_branches / total_eval) * 100.0

        y_exec = torch.zeros_like(x_test)
        w_up_r = param_map[f"node_{roll_idx}_organelle_roll_w_up"]
        b_up_r = param_map[f"node_{roll_idx}_organelle_roll_b_up"]
        w_down_r = param_map[f"node_{roll_idx}_organelle_roll_w_down"]
        b_down_r = param_map[f"node_{roll_idx}_organelle_roll_b_down"]

        w_up_f = param_map[f"node_{flip_idx}_organelle_flip_w_up"]
        b_up_f = param_map[f"node_{flip_idx}_organelle_flip_b_up"]
        w_down_f = param_map[f"node_{flip_idx}_organelle_flip_w_down"]
        b_down_f = param_map[f"node_{flip_idx}_organelle_flip_b_down"]

        for i in range(total_eval):
            xi = x_test[i:i+1]
            if pred_branches[i] == 1:
                h = torch.matmul(xi, w_up_r.t()) + b_up_r
                y_exec[i] = torch.matmul(nn.functional.gelu(h), w_down_r.t()) + b_down_r
            else:
                h = torch.matmul(xi, w_up_f.t()) + b_up_f
                y_exec[i] = torch.matmul(nn.functional.gelu(h), w_down_f.t()) + b_down_f

        track1_mse = nn.functional.mse_loss(y_exec, y_true).item()
        cos_sim = nn.functional.cosine_similarity(y_exec, y_true, dim=-1).mean().item()

    print(f"  ✓ Branch Selection Accuracy       : {branch_acc:.2f}% (EXP-334 Baseline: 98.50%)")
    print(f"  ✓ Execution MSE Loss             : {track1_mse:.6f}")
    print(f"  ✓ Mean Recirculation Steps K(z)  : {steps_test.mean().item():.2f}")

    return {
        "branch_acc": branch_acc,
        "loss": track1_mse,
        "cos_sim": cos_sim,
        "z_vals": z_test.squeeze(-1).cpu().numpy(),
        "probs_roll": probs[:, 1].cpu().numpy(),
        "true_branches": true_branches.cpu().numpy(),
        "steps_taken": steps_test.squeeze(-1).cpu().numpy()
    }


# =====================================================================================
# TRACK 2: LANDAU DOUBLE-WELL POTENTIAL LATCH
# =====================================================================================
def run_track_2_landau_latch(device_str: str) -> Dict[str, float]:
    print("\n" + "=" * 85)
    print("TRACK 2: LANDAU DOUBLE-WELL POTENTIAL BISTABLE LATCH")
    print("=" * 85)

    device = torch.device(device_str)
    dim = 16

    # Instantiate the native C++20 LandauDoubleWellOp
    landau_op = karyon_core.LandauDoubleWellOp(dim, device_str)

    class LandauAdditionCell(nn.Module):
        def __init__(self, dim, landau_op):
            super().__init__()
            self.dim = dim
            self.landau_op = landau_op
            self.digit_embed = nn.Embedding(10, dim)
            self.w_add = nn.Linear(dim * 2 + dim, dim) # Maps digits and carry state to somatic feedback
            self.w_sum = nn.Linear(dim, 10)
            self.w_carry_out = nn.Linear(dim, dim)     # Drives landau input current

        def forward(self, d1, d2, carry_state_tensor):
            # d1, d2: [B] integers
            # carry_state_tensor: [B, dim] continuous physical double-well state
            e1 = self.digit_embed(d1)
            e2 = self.digit_embed(d2)
            h_in = torch.cat([e1, e2, carry_state_tensor], dim=-1)

            h = torch.relu(self.w_add(h_in))
            sum_logits = self.w_sum(h)
            
            # Drive the double-well potential with input current
            i_in = self.w_carry_out(h)
            next_carry_state = self.landau_op(i_in) # Native continuous-time SDE integration
            return sum_logits, next_carry_state

    model = LandauAdditionCell(dim, landau_op).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("  • Training Landau Double-Well Addition Engine...")
    for step in range(1, 601):
        bs = 64
        d1 = torch.randint(0, 10, (bs,), device=device)
        d2 = torch.randint(0, 10, (bs,), device=device)
        
        # Random historical carry state
        c_in_binary = torch.randint(0, 2, (bs, 1), device=device).float()
        # Convert binary to physical attractor state (Carry=1 -> +1.0, Carry=0 -> -1.0)
        c_in_state = (c_in_binary * 2.0) - 1.0
        c_in_state_tensor = c_in_state.repeat(1, dim)

        raw_sum = d1 + d2 + c_in_binary.squeeze(-1).long()
        target_digit = raw_sum % 10
        target_carry_binary = (raw_sum >= 10).float().unsqueeze(-1)
        target_carry_state_tensor = ((target_carry_binary * 2.0) - 1.0).repeat(1, dim)

        model.landau_op.reset_state()
        # Seed the internal state of the double-well operator
        model.landau_op.forward(c_in_state_tensor)

        sum_logits, next_carry_state = model(d1, d2, c_in_state_tensor)

        loss_sum = nn.functional.cross_entropy(sum_logits, target_digit)
        loss_carry = nn.functional.mse_loss(next_carry_state, target_carry_state_tensor)
        total_loss = loss_sum + 3.0 * loss_carry

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Sequential addition test with carry propagation (e.g. 4-digit numbers)
    print("  • Evaluating Multi-Digit Landau Stream Addition...")
    num_tests = 200
    num_digits = 4
    correct_full_additions = 0
    correct_carry_digits = 0
    total_carry_instances = 0

    carry_states_log = []

    for _ in range(num_tests):
        n1 = random.randint(1000, 9999)
        n2 = random.randint(1000, 9999)
        true_result = n1 + n2

        s1 = str(n1)[::-1]
        s2 = str(n2)[::-1]

        model.landau_op.reset_state()
        # Initial carry is 0 (all -1.0 states)
        carry_state_tensor = torch.full((1, dim), -1.0, device=device)
        model.landau_op.forward(carry_state_tensor)

        pred_digits = []

        for pos in range(num_digits):
            d1_val = torch.tensor([int(s1[pos])], device=device)
            d2_val = torch.tensor([int(s2[pos])], device=device)

            with torch.no_grad():
                sum_logits, carry_state_tensor = model(d1_val, d2_val, carry_state_tensor)
                pred_d = torch.argmax(sum_logits, dim=-1).item()
                pred_digits.append(pred_d)

                # Read carry state from the mean of the double-well tensor
                mean_carry_val = carry_state_tensor.mean().item()
                is_carry = (mean_carry_val > 0.0)
                carry_states_log.append(mean_carry_val)

                # Verification
                carry_in_num = 1 if len(carry_states_log) > 1 and carry_states_log[-2] > 0.0 else 0
                expected_sum = int(s1[pos]) + int(s2[pos]) + carry_in_num
                expected_d = expected_sum % 10
                expected_carry = (expected_sum >= 10)

                if expected_carry:
                    total_carry_instances += 1
                    if is_carry and pred_d == expected_d:
                        correct_carry_digits += 1

        if carry_state_tensor.mean().item() > 0.0:
            pred_digits.append(1)

        pred_num = int("".join(str(d) for d in pred_digits[::-1]))
        if pred_num == true_result:
            correct_full_additions += 1

    carry_digit_acc = (correct_carry_digits / max(1, total_carry_instances)) * 100.0
    full_add_acc = (correct_full_additions / num_tests) * 100.0

    print(f"  ✓ Full 4-Digit Number Addition Accuracy : {full_add_acc:.2f}%")
    print(f"  ✓ Landau Carry-Bit Latching Accuracy    : {carry_digit_acc:.2f}% (EXP-334 Baseline: 83.50%)")

    return {
        "carry_digit_acc": carry_digit_acc,
        "full_add_acc": full_add_acc,
        "carry_states": carry_states_log[:100]
    }


# =====================================================================================
# MAIN RUNNER & MULTI-PANEL DIAGNOSTIC PLOTTER
# =====================================================================================
def run_exp_335():
    print("=" * 85)
    print("EXP-335: CONTINUOUS DRIFT-DIFFUSION & LANDAU LATCH BENCHMARK")
    print("=" * 85)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Substrate Compute Device: {device_str.upper()}")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t1_res = run_track_1_drift_diffusion(device_str)
    t2_res = run_track_2_landau_latch(device_str)

    # Plotting
    os.makedirs("experiments", exist_ok=True)
    plot_path = "experiments/exp_335_continuous_bifurcation.png"

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

    # Panel 1: Track 1 Drift-Diffusion Decision Boundary
    z_sorted_idx = np.argsort(t1_res["z_vals"])
    axes[0].scatter(t1_res["z_vals"][z_sorted_idx], t1_res["probs_roll"][z_sorted_idx], color="#00ffcc", alpha=0.7, label="P(roll | z)")
    axes[0].axvline(x=0.0, color="#ff0055", linestyle="--", label="Somatic Equilibrium (z=0)")
    axes[0].set_title(f"Track 1: Drift-Diffusion Branching\nAcc: {t1_res['branch_acc']:.2f}%")
    axes[0].set_xlabel("Internal State Metric z")
    axes[0].set_ylabel("Routing Probability P(roll)")
    axes[0].legend(facecolor="#2a2a2a", labelcolor="white")
    axes[0].grid(True, color="#333333", linestyle=":")

    # Panel 2: Track 1 Recirculation Steps K(z)
    axes[1].scatter(t1_res["z_vals"], t1_res["steps_taken"], color="#9900ff", alpha=0.6, label="Recirculation Steps K")
    axes[1].set_title("Chrono-Actuated Recirculation K(z)\n(Dynamic Thinking Time)")
    axes[1].set_xlabel("Internal State Metric z")
    axes[1].set_ylabel("Steps Taken K")
    axes[1].legend(facecolor="#2a2a2a", labelcolor="white")
    axes[1].grid(True, color="#333333", linestyle=":")

    # Panel 3: Track 2 Landau Double-Well Potential Latch Trajectory
    axes[2].plot(t2_res["carry_states"], color="#ffbb00", linewidth=1.8, label="Mean Well Position <x>")
    axes[2].axhline(y=1.0, color="#00ffcc", linestyle=":", label="Carry Basin (+1.0)")
    axes[2].axhline(y=-1.0, color="#ff0055", linestyle=":", label="Zero Basin (-1.0)")
    axes[2].set_title(f"Track 2: Landau Double-Well Latch\nCarry Acc: {t2_res['carry_digit_acc']:.2f}%")
    axes[2].set_xlabel("Stream Addition Step")
    axes[2].set_ylabel("Attractor State x")
    axes[2].legend(facecolor="#2a2a2a", labelcolor="white")
    axes[2].grid(True, color="#333333", linestyle=":")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  ✓ Multi-Panel Diagnostic Plot Saved: {plot_path}")

    # =========================================================================
    # KEP RULE #2 VERDICT SYNTHESIS
    # =========================================================================
    print("\n" + "=" * 85)
    print("EXP-335: FINAL SYNTHESIS & VERDICT")
    print("=" * 85)

    is_track1_pass = t1_res["branch_acc"] >= 99.0
    is_track2_pass = t2_res["carry_digit_acc"] >= 95.0

    all_passed = is_track1_pass and is_track2_pass
    verdict = "🟢 POSITIVE" if all_passed else "🔴 REJECTED"

    print(f"VERDICT: {verdict}")
    print(f"  • Track 1 (Drift-Diffusion Branching Acc) : {t1_res['branch_acc']:.2f}% (EXP-334 Baseline: 98.50%) -> {'PASS' if is_track1_pass else 'FAIL'}")
    print(f"  • Track 2 (Landau Double-Well Carry Acc)  : {t2_res['carry_digit_acc']:.2f}% (EXP-334 Baseline: 83.50%) -> {'PASS' if is_track2_pass else 'FAIL'}")
    print("=" * 85)


if __name__ == "__main__":
    run_exp_335()
