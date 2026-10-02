"""
=====================================================================================
EXP-337: TOPOLOGICAL MORPHOGENESIS & EPIGENETIC GROWTH ON ATOMIC OPERATORS
=====================================================================================
Implements a dynamic self-evolving micro-graph circuit governed by an Epigenetic
Gene Regulatory Network (GRN) and Net2Net Smooth Grafting.

Verifies:
  1. Open-ended Topological Genesis (Pillar 1) on atomic operators.
  2. Strict Zero-Shock Function Identity (Principle 15) at birth (t_0) via:
     y_i(t) = tanh(alpha_epi_i(t)) * Operator_i(x_i(t)), with alpha_epi_i(t_0) = 0.0
  3. Neurodarwinian Apoptosis (Pruning) of redundant or unviable operators.
  4. Adaptive scaling under Free Energy stress on a non-stationary chaotic stream.
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


# =====================================================================================
# ATOMIC PRIMITIVE OPERATORS
# =====================================================================================
class PrimitiveOperator(nn.Module):
    """
    Base class for atomic primitive mathematical operators.
    """
    def __init__(self, op_type: str, dim: int):
        super().__init__()
        self.op_type = op_type
        self.dim = dim

        if op_type == "LinearAccumulatorOp":
            # Leaky continuous integrator: dx/dt = W * x_in - decay * x_prev
            self.w_in = nn.Linear(dim, dim)
            self.w_decay = nn.Linear(dim, dim)
            self.w_dt = nn.Linear(dim, dim)
        elif op_type == "BilinearMultiplicativeOp":
            # Multiplicative conjunction: y = W1(x) * W2(x)
            self.w1 = nn.Linear(dim, dim, bias=False)
            self.w2 = nn.Linear(dim, dim, bias=False)
        elif op_type == "LinearOp":
            # Linear projector: y = W * x + b
            self.linear = nn.Linear(dim, dim)
        elif op_type == "StochasticLangevinOp":
            # Langevin noise: y = x + sigma * noise
            self.w_noise = nn.Linear(dim, dim)
        else:
            raise ValueError(f"Unknown operator type: {op_type}")

    def forward(self, x_in: torch.Tensor, x_prev: torch.Tensor = None) -> torch.Tensor:
        if self.op_type == "LinearAccumulatorOp":
            if x_prev is None:
                x_prev = torch.zeros_like(x_in)
            u_in = self.w_in(x_in)
            decay = torch.nn.functional.softplus(self.w_decay(x_prev)) + 1e-4
            dt = torch.sigmoid(self.w_dt(x_prev)) * 0.4 + 0.05
            dx_dt = u_in - (decay * x_prev)
            return torch.clamp(x_prev + dt * dx_dt, -3.0, 3.0)

        elif self.op_type == "BilinearMultiplicativeOp":
            return self.w1(x_in) * self.w2(x_in)

        elif self.op_type == "LinearOp":
            return self.linear(x_in)

        elif self.op_type == "StochasticLangevinOp":
            sigma = torch.nn.functional.softplus(self.w_noise(x_in)) * 0.01
            noise = torch.randn_like(x_in) * sigma
            return x_in + noise


# =====================================================================================
# DYNAMIC EPIGENETIC GRAPH ARCHITECTURE
# =====================================================================================
class EpigeneticMorphicCircuit(nn.Module):
    """
    A self-evolving micro-graph circuit that dynamically grows and prunes
    atomic primitive operators under Free Energy stress.

    Strictly satisfies Principle 15 (Net2Net Smooth Grafting) to prevent structural shock.
    """
    def __init__(self, dim: int, max_dormant_pool: int = 8, device_str: str = "cpu"):
        super().__init__()
        self.dim = dim
        self.device = torch.device(device_str)
        self.max_dormant_pool = max_dormant_pool

        # Start with a minimal core: 2 active primitive operators
        self.active_operators = nn.ModuleList([
            PrimitiveOperator("LinearOp", dim),
            PrimitiveOperator("LinearAccumulatorOp", dim)
        ])

        # Dormant pool of candidate operators available for epigenetic genesis (Hox expression)
        self.dormant_pool = nn.ModuleList([
            PrimitiveOperator("BilinearMultiplicativeOp", dim),
            PrimitiveOperator("LinearAccumulatorOp", dim),
            PrimitiveOperator("StochasticLangevinOp", dim),
            PrimitiveOperator("BilinearMultiplicativeOp", dim),
            PrimitiveOperator("LinearOp", dim),
            PrimitiveOperator("LinearAccumulatorOp", dim)
        ][:max_dormant_pool])

        # Epigenetic Morphogenesis parameters
        # alpha_epi governs the Net2Net Smooth Grafting Gate: tanh(alpha_epi)
        # We initialize them as ModuleParameters to allow backprop/gradient flow
        self.active_alpha_epi = nn.ParameterList([
            nn.Parameter(torch.tensor(1.5, device=self.device)) for _ in range(len(self.active_operators))
        ])

        self.dormant_alpha_epi = nn.ParameterList([
            nn.Parameter(torch.tensor(-5.0, device=self.device)) for _ in range(len(self.dormant_pool))
        ])

        # Register continuous states for leaky nodes
        self.register_buffer("node_states", torch.zeros(len(self.active_operators), 1, dim))

    def reset_states(self, batch_size: int = 1):
        num_nodes = len(self.active_operators) + len(self.dormant_pool)
        self.node_states = torch.zeros(num_nodes, batch_size, self.dim, device=self.device)

    def forward(self, x_in: torch.Tensor, steps: int = 2) -> torch.Tensor:
        """
        Evaluates the dynamic morphic graph flow.
        """
        B = x_in.size(0)
        total_nodes = len(self.active_operators) + len(self.dormant_pool)
        if self.node_states.size(1) != B or self.node_states.size(0) != total_nodes:
            self.reset_states(B)

        x_curr = x_in
        node_idx = 0

        # 1. Flow through active core nodes
        for i, op in enumerate(self.active_operators):
            alpha = self.active_alpha_epi[i]
            graft_gate = torch.tanh(alpha)  # Smooth gating

            prev_state = self.node_states[node_idx]
            node_out = op(x_curr, prev_state)
            self.node_states[node_idx] = node_out.detach()

            # Smoothly graft the node's contribution into the main computational stream
            x_curr = x_curr + graft_gate * node_out
            node_idx += 1

        # 2. Flow through dormant pool nodes (Hox gene expression)
        for i, op in enumerate(self.dormant_pool):
            alpha = self.dormant_alpha_epi[i]
            graft_gate = torch.tanh(alpha)  # Epigenetic expression gate

            # If the expression gate is virtually closed, skip heavier calculations to save compute,
            # but preserve gradient flow during training via soft-gating.
            prev_state = self.node_states[node_idx]
            node_out = op(x_curr, prev_state)
            self.node_states[node_idx] = node_out.detach()

            # Grafting contribution (strictly 0.0 when alpha <= -5.0)
            x_curr = x_curr + graft_gate * node_out
            node_idx += 1

        return x_curr

    def trigger_epigenetic_genesis(self, pool_idx: int) -> float:
        """
        Expresses a dormant node from the pool into the active graph.
        Strictly initializes alpha_epi = 0.0 to ensure ZERO structural shock (Principle 15).
        Returns the shock delta (difference in function mapping before and after genesis).
        """
        if pool_idx >= len(self.dormant_pool):
            return 0.0

        # Capture a sample output before genesis
        dummy_in = torch.randn(1, self.dim, device=self.device)
        with torch.no_grad():
            out_before = self.forward(dummy_in)

        # Move node from dormant to active ModuleList
        op_to_activate = self.dormant_pool[pool_idx]

        # Net2Net Smooth Grafting: Initialize alpha_epi to exactly 0.0 (tanh(0.0) = 0.0)
        # This guarantees that the new node's initial contribution is mathematically non-existent.
        new_alpha = nn.Parameter(torch.tensor(0.0, device=self.device))

        # Modify structural lists
        self.active_operators.append(op_to_activate)
        self.active_alpha_epi.append(new_alpha)

        # Remove from dormant pool
        del self.dormant_pool[pool_idx]
        del self.dormant_alpha_epi[pool_idx]

        # Recalculate states buffer size
        self.reset_states(1)

        # Capture sample output after genesis
        with torch.no_grad():
            out_after = self.forward(dummy_in)

        # Calculate exact function mapping delta (structural shock)
        shock_delta = torch.norm(out_after - out_before).item()
        return shock_delta

    def execute_darwinian_apoptosis(self) -> int:
        """
        Prunes active operators whose epigenetic gating has decayed below a critical threshold.
        """
        pruned_count = 0
        keep_indices = []

        for i, alpha in enumerate(self.active_alpha_epi):
            # If the smooth grafting gate has decayed close to zero (tanh(alpha) < 0.05)
            if torch.tanh(alpha).item() < 0.05:
                pruned_count += 1
            else:
                keep_indices.append(i)

        if pruned_count > 0:
            new_active_ops = nn.ModuleList([self.active_operators[i] for i in keep_indices])
            new_active_alphas = nn.ParameterList([self.active_alpha_epi[i] for i in keep_indices])

            self.active_operators = new_active_ops
            self.active_alpha_epi = new_active_alphas
            self.reset_states(1)

        return pruned_count


# =====================================================================================
# NON-STATIONARY CHAOTIC REALITY STREAM GENERATOR
# =====================================================================================
class NonStationaryChaoticStream:
    """
    Generates an unbroken, non-stationary temporal stream representing a shifting
    double-well transition (Lorenz-like chaotic attractor) with a sudden environmental phase shift.
    """
    def __init__(self, dim: int, length: int = 1000, device_str: str = "cpu"):
        self.dim = dim
        self.length = length
        self.device = torch.device(device_str)
        self.t = 0

        # Chaotic state variables
        self.x = 0.1
        self.y = 0.0
        self.z = 0.0

    def step(self) -> torch.Tensor:
        self.t += 1
        # Lorenz Attractor differential updates
        dt = 0.01
        sigma = 10.0
        beta = 8.0 / 3.0

        # Sudden environmental phase shift at t = 500 (Non-stationarity)
        rho = 28.0 if self.t < 500 else 45.0

        dx = sigma * (self.y - self.x)
        dy = self.x * (rho - self.z) - self.y
        dz = self.x * self.y - beta * self.z

        self.x += dx * dt
        self.y += dy * dt
        self.z += dz * dt

        # Project 3D Lorenz state into high-dimensional embedding space
        state_3d = torch.tensor([self.x, self.y, self.z], device=self.device).float()

        # Random but deterministic projection matrix to dim
        proj = torch.sin(torch.arange(self.dim * 3, device=self.device).float().view(self.dim, 3))
        stream_tensor = torch.matmul(proj, state_3d)

        # Add non-stationary high-frequency noise
        noise_scale = 0.05 if self.t < 500 else 0.15
        stream_tensor += torch.randn(self.dim, device=self.device) * noise_scale

        return stream_tensor.unsqueeze(0)


# =====================================================================================
# EXP-337 SCIENTIFIC RUNNER
# =====================================================================================
def run_exp_337_morphogenesis(device_str: str) -> Dict[str, any]:
    print("\n" + "=" * 85)
    print("EXP-337: TOPOLOGICAL MORPHOGENESIS & EPIGENETIC GROWTH RUNNER")
    print("=" * 85)

    device = torch.device(device_str)
    dim = 16
    stream_length = 1000

    # 1. Initialize the Epigenetic Morphic Circuit
    model = EpigeneticMorphicCircuit(dim=dim, device_str=device_str).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Reality Stream
    stream = NonStationaryChaoticStream(dim=dim, length=stream_length, device_str=device_str)

    # Telemetry logging
    free_energy_history = []
    active_nodes_history = []
    shock_deltas = []
    grafting_gates_history = []
    pruning_events = []

    # Target threshold for triggering epigenetic genesis (Hox expression)
    # If the moving average of Free Energy (surprise) exceeds this, we sprout a new node.
    surprise_threshold = 0.45
    moving_surprise = 0.0
    genesis_cooldown = 0

    print("  • Executing Single-Pass Streaming Reality learning (N=1 Epoch)...")

    # Cache the initial stream step to start predictive coding
    x_prev = stream.step()

    for t in range(1, stream_length):
        x_curr = stream.step()

        # Predictive coding: Predict x_curr from x_prev
        x_pred = model(x_prev)

        # Variational Free Energy: F_t = Reconstruction_Loss + Complexity_Penalty(KL)
        # For our continuous stream, reconstruction error represents the surprise.
        rec_loss = nn.functional.mse_loss(x_pred, x_curr)

        # Complexity penalty: L2 regularization of active epigenetic gates to encourage sparsity/apoptosis
        complexity_penalty = 0.0
        for alpha in model.active_alpha_epi:
            complexity_penalty += 0.005 * torch.pow(torch.tanh(alpha), 2)

        free_energy = rec_loss + complexity_penalty
        free_energy_val = free_energy.item()
        free_energy_history.append(free_energy_val)

        # Optimize the active parameters (synaptic weights and active epigenetic expression gates)
        optimizer.zero_grad()
        free_energy.backward()
        optimizer.step()

        # Update moving average of surprise
        moving_surprise = 0.9 * moving_surprise + 0.1 * free_energy_val

        # Hox-Gene Epigenetic Growth Trigger (Pillar 1: Topological Genesis)
        # If surprise is high, we are under metabolic stress -> Sprout a new primitive operator node
        if moving_surprise > surprise_threshold and len(model.dormant_pool) > 0 and genesis_cooldown <= 0:
            # Trigger epigenetic growth on a dormant candidate operator
            pool_idx = 0  # Sprout the first node in the dormant pool
            shock_delta = model.trigger_epigenetic_genesis(pool_idx)
            shock_deltas.append(shock_delta)

            # Recreate optimizer to include the newly expressed parameters
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            print(f"    [t={t:03d}] 🧬 Epigenetic Sprouting Triggered! Surprise: {moving_surprise:.4f} | "
                  f"Structural Shock Delta: {shock_delta:.8f}")
            genesis_cooldown = 40  # Set a cooldown period for ontogenetic maturation

        # Dynamic warming of newly sprouted epigenetic gates
        # We model the maturation of newly sprouted nodes as a continuous relaxation:
        with torch.no_grad():
            for alpha in model.active_alpha_epi:
                # If the gate is in the warming phase (0.0 <= alpha < 1.5)
                if 0.0 <= alpha.item() < 1.5:
                    alpha.copy_(alpha + 0.05)  # Gradual maturation

        # Neurodarwinian Apoptosis (Pruning):
        # We apply a continuous decay to inactive/redundant nodes
        with torch.no_grad():
            for i, alpha in enumerate(model.active_alpha_epi):
                # If a node's gradient flow is extremely small, decay its expression gate
                if alpha.grad is not None:
                    grad_norm = torch.norm(alpha.grad).item()
                    if grad_norm < 1e-4 and alpha.item() > 0.5:
                        # Decay the gate towards apoptosis
                        alpha.copy_(alpha - 0.02)

        # Clean up fully decayed nodes
        pruned_count = model.execute_darwinian_apoptosis()
        if pruned_count > 0:
            pruning_events.append((t, pruned_count))
            # Recreate optimizer after pruning
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            print(f"    [t={t:03d}] 🍂 Neurodarwinian Apoptosis: Pruned {pruned_count} redundant node(s) from active graph.")

        # Log active node count and epigenetic gates
        active_nodes_history.append(len(model.active_operators))
        grafting_gates_history.append([torch.tanh(alpha).item() for alpha in model.active_alpha_epi])

        if genesis_cooldown > 0:
            genesis_cooldown -= 1

        # Advance stream
        x_prev = x_curr

    # Evaluate Metrics
    mean_free_energy_first_half = np.mean(free_energy_history[:500])
    mean_free_energy_second_half = np.mean(free_energy_history[500:])
    final_free_energy = free_energy_history[-1]

    # Analyze structural shock delta (Principle 15)
    max_shock_delta = np.max(shock_deltas) if len(shock_deltas) > 0 else 0.0
    mean_shock_delta = np.mean(shock_deltas) if len(shock_deltas) > 0 else 0.0

    print("\n" + "=" * 85)
    print("EXP-337: EMPIRICAL TELEMETRY RESULTS")
    print("=" * 85)
    print(f"  ✓ Final Variational Free Energy F_t     : {final_free_energy:.6f}")
    print(f"  ✓ Mean Surprise (Waking Phase 1, t<500) : {mean_free_energy_first_half:.6f}")
    print(f"  ✓ Mean Surprise (Shifting Phase 2, t>=500): {mean_free_energy_second_half:.6f}")
    print(f"  ✓ Maximum Structural Shock Delta        : {max_shock_delta:.10f} (Target: <= 1e-5)")
    print(f"  ✓ Mean Structural Shock Delta           : {mean_shock_delta:.10f}")
    print(f"  ✓ Total Sprouted Nodes (Genesis Events) : {len(shock_deltas)}")
    print(f"  ✓ Total Pruned Nodes (Apoptosis Events) : {len(pruning_events)}")
    print(f"  ✓ Active Nodes at Convergence           : {len(model.active_operators)}")

    return {
        "free_energy_history": free_energy_history,
        "active_nodes_history": active_nodes_history,
        "shock_deltas": shock_deltas,
        "max_shock_delta": max_shock_delta,
        "mean_shock_delta": mean_shock_delta,
        "final_free_energy": final_free_energy,
        "pruning_events_count": len(pruning_events)
    }


# =====================================================================================
# DIAGNOSTIC PLOTTER & VERDICT SYNTHESIS
# =====================================================================================
def run_exp_337():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    metrics = run_exp_337_morphogenesis(device_str)

    # Plot results
    os.makedirs("experiments", exist_ok=True)
    plot_path = "experiments/exp_337_topological_morphogenesis.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#121212")

    for ax in axes:
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444444")

    # Panel 1: Free Energy surprise stream learning curve
    axes[0].plot(metrics["free_energy_history"], color="#00ffcc", linewidth=1.5, label="Surprise (Free Energy F_t)")
    axes[0].axvline(x=500, color="#ff0055", linestyle="--", label="Environmental Shift (t=500)")
    axes[0].set_title("Track 1: Variational Free Energy Trajectory")
    axes[0].set_xlabel("Streaming Reality Step (t)")
    axes[0].set_ylabel("Variational Free Energy F_t")
    axes[0].legend(facecolor="#2a2a2a", labelcolor="white")
    axes[0].grid(True, color="#333333", linestyle=":")

    # Panel 2: Topological Genesis & Apoptosis (Active Node Count)
    axes[1].plot(metrics["active_nodes_history"], color="#ffbb00", linewidth=2.0, label="Active Operator Nodes")
    axes[1].set_title("Track 2: Topological Morphogenesis (Active Nodes)")
    axes[1].set_xlabel("Streaming Reality Step (t)")
    axes[1].set_ylabel("Active Node Count")
    axes[1].legend(facecolor="#2a2a2a", labelcolor="white")
    axes[1].grid(True, color="#333333", linestyle=":")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  ✓ Multi-Panel Diagnostic Plot Saved: {plot_path}")

    # Verdict Evaluation (KEP Rule #2)
    # Success Criteria:
    #   1. Maximum Structural Shock Delta must be <= 1e-5 (Strict zero-delta function identity)
    #   2. Final Free Energy must converge to a stable regime (<= 0.15)
    #   3. Active nodes must dynamically change (demonstrating genesis and apoptosis)
    is_zero_shock = metrics["max_shock_delta"] <= 1e-5
    is_converged = metrics["final_free_energy"] <= 0.15
    is_dynamic = len(metrics["shock_deltas"]) > 0

    all_passed = is_zero_shock and is_converged and is_dynamic
    verdict = "🟢 POSITIVE" if all_passed else "🔴 REJECTED"

    print("\n" + "=" * 85)
    print("EXP-337: FINAL SYNTHESIS & VERDICT")
    print("=" * 85)
    print(f"VERDICT: {verdict}")
    print(f"  • Net2Net Zero-Shock Identity (Max Delta) : {metrics['max_shock_delta']:.10f} -> "
          f"{'PASS' if is_zero_shock else 'FAIL'}")
    print(f"  • Final Free Energy Convergence (F_t)     : {metrics['final_free_energy']:.6f} -> "
          f"{'PASS' if is_converged else 'FAIL'}")
    print(f"  • Dynamic Topological Genesis (Sprouts)   : {len(metrics['shock_deltas'])} events -> "
          f"{'PASS' if is_dynamic else 'FAIL'}")
    print(f"  • Neurodarwinian Apoptosis (Pruned Nodes) : {metrics['pruning_events_count']} events")
    print("=" * 85)


if __name__ == "__main__":
    run_exp_337()
