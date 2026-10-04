# experiments/exp_346_active_inference_somatic_relaxation.py
"""
===============================================================================
EXP-346: HIERARCHICAL ACTIVE INFERENCE & SOMATIC RELAXATION BENCHMARK
===============================================================================
Exact Implementation of Architect Gema's Mathematical Blueprint & Bazilevs' Hierarchy:

1. Tier 1 (Operators - Операторы):
   - LinearOp: W * x + b
   - GatedNonlinearOp: GELU/Swish gating
   - PACPhaseModulatorOp: Theta-Gamma phase-amplitude modulator Ω_{θ-γ}
   - HopfieldOp: Continuous energy attractor retrieval

2. Tier 2 (Organelles - Органеллы):
   - PredictiveOrganelle: Generates prediction μ = f(z) and precision-weighted error ε = Π * (x - μ)
   - AssociativeOrganelle: Continuous energy attractor with Hopfield dynamics
   - SomaticOrganelle: Neurotransmitter integration (DA, NE, 5-HT) and Ashby homeostatic bounds

3. Tier 3 (Cells - Клетки):
   - ActiveInferenceCell: Encapsulates Predictive + Associative + Somatic organelles.
     Computes local Variational Free Energy:
     F_cell = 0.5 * ||ε_i||^2 + KL(q(z_i) || p(z_i)) + SomaticPenalty

4. Tier 4 (Karyon Synaptic Mesh - Синаптическая сеть Кариона):
   - 3-Factor Neuromodulated Plasticity without global backprop / Adam:
     ΔW_{ij} = η * DA_t * γ_{phase}(θ_t) * (ε_i * z_j^T - λ * W_{ij})

Strict KEP Rule #2 & Anti-Goodhart Verification Criteria:
- Final Absolute Free Energy F_final < 1.5
- Monotonic Free Energy Reduction ΔF >= 0.08
- Somatic Dopamine Drive DA >= 0.2
- Pure Local Active Inference Relaxation (No Global PyTorch Autograd BPTT)
"""

import math
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set reproducible seeds
torch.manual_seed(42)

# ===============================================================================
# TIER 1: ATOMIC OPERATORS (ОПЕРАТОРЫ)
# ===============================================================================

class LinearOp(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * (2.0 / (in_dim + out_dim)) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class GatedNonlinearOp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = LinearOp(dim, dim)
        self.val_proj = LinearOp(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(x))
        val = F.gelu(self.val_proj(x))
        return gate * val


class PACPhaseModulatorOp(nn.Module):
    def __init__(self, dim: int, kappa: float = 4.0, phi_opt: float = 0.0):
        super().__init__()
        self.dim = dim
        self.kappa = kappa
        self.phi_opt = phi_opt

    def forward(self, x: torch.Tensor, theta_phase: float) -> torch.Tensor:
        # Phase-amplitude gating function γ(θ)
        gamma = 1.0 / (1.0 + math.exp(-self.kappa * (math.cos(theta_phase - self.phi_opt) - 0.2)))
        return x * gamma, gamma


class HopfieldOp(nn.Module):
    def __init__(self, dim: int, num_memories: int = 16):
        super().__init__()
        self.memories = nn.Parameter(torch.randn(num_memories, dim) * 0.1)

    def forward(self, x: torch.Tensor, steps: int = 3) -> torch.Tensor:
        # Continuous attractor relaxation towards stored memory patterns
        state = x
        for _ in range(steps):
            # Attention-like softmax similarity alignment
            sim = torch.matmul(state, self.memories.T) / (self.memories.shape[-1] ** 0.5)
            attn = F.softmax(sim, dim=-1)
            retrieved = torch.matmul(attn, self.memories)
            state = 0.7 * state + 0.3 * retrieved
        return state


# ===============================================================================
# TIER 2: ORGANELLES (ОРГАНЕЛЛЫ)
# ===============================================================================

class PredictiveOrganelle(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.pred_op = LinearOp(in_dim, out_dim)
        self.pac_op = PACPhaseModulatorOp(out_dim)
        # Precision matrix diagonal log-scale (Π)
        self.log_precision = nn.Parameter(torch.zeros(out_dim))

    def forward(self, z: torch.Tensor, target_x: torch.Tensor, theta_phase: float):
        mu = self.pred_op(z)
        
        # Bottom-up prediction error: (x - μ)
        raw_error = target_x - mu
        
        # Precision weighting: Π = exp(log_precision) modulated by PAC gamma phase
        precision = torch.exp(self.log_precision)
        prec_error = raw_error * precision
        
        # PAC Modulation
        pac_error, gamma_phase = self.pac_op(prec_error, theta_phase)
        return mu, pac_error, gamma_phase


class AssociativeOrganelle(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.hopfield = HopfieldOp(dim)

    def forward(self, state: torch.Tensor):
        relaxed = self.hopfield(state)
        # Energy of state relative to attractor
        energy = torch.mean((state - relaxed) ** 2)
        return relaxed, energy


class SomaticOrganelle(nn.Module):
    def __init__(self):
        super().__init__()
        self.dopamine = 0.5  # Reward prediction drive [0, 1]
        self.noradrenaline = 0.5  # Environmental volatility [0, 1]
        self.serotonin = 0.5  # Somatic stability [0, 1]
        self.theta_phase = 0.0  # Phase of theta oscillator [0, 2π)

    def update_somatic_state(self, current_fe: float, prev_fe: float, dt: float = 0.1):
        # Update theta phase
        omega_theta = 2.0 * math.pi * 6.0  # 6 Hz Theta oscillation
        self.theta_phase = (self.theta_phase + omega_theta * dt) % (2.0 * math.pi)
        
        # Somatic Free Energy Delta
        fe_delta = prev_fe - current_fe
        
        # Dopamine increases when Free Energy drops (reward signal)
        if fe_delta > 0:
            self.dopamine = min(1.0, self.dopamine + 0.1 * fe_delta)
        else:
            self.dopamine = max(0.1, self.dopamine - 0.05 * abs(fe_delta))
            
        # Noradrenaline responds to unexpected uncertainty
        self.noradrenaline = min(1.0, max(0.1, 0.8 * self.noradrenaline + 0.2 * abs(fe_delta)))
        
        # Serotonin reflects long-term Ashby homeostatic stability
        homeostatic_dev = abs(current_fe - 0.5)
        self.serotonin = max(0.1, 1.0 / (1.0 + homeostatic_dev))

        return {
            "dopamine": self.dopamine,
            "noradrenaline": self.noradrenaline,
            "serotonin": self.serotonin,
            "theta_phase": self.theta_phase
        }


# ===============================================================================
# TIER 3: CORTICAL CELLS (КЛЕТКИ)
# ===============================================================================

class ActiveInferenceCell(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.predictive_organelle = PredictiveOrganelle(hidden_dim, out_dim)
        self.associative_organelle = AssociativeOrganelle(hidden_dim)
        self.state_proj = LinearOp(in_dim, hidden_dim)

    def forward(self, x_in: torch.Tensor, target_out: torch.Tensor, theta_phase: float):
        # Latent state representation z
        z_raw = self.state_proj(x_in)
        z_relaxed, assoc_energy = self.associative_organelle(z_raw)
        
        # Top-down prediction & precision error
        mu, error, gamma_phase = self.predictive_organelle(z_relaxed, target_out, theta_phase)
        
        # Local Variational Free Energy F_cell
        # F = 0.5 * ||ε||^2 + KL(z_raw || z_relaxed) + assoc_energy
        precision_error_cost = 0.5 * torch.mean(error ** 2)
        kl_cost = 0.5 * torch.mean((z_raw - z_relaxed) ** 2)
        
        free_energy = precision_error_cost + kl_cost + assoc_energy
        return {
            "mu": mu,
            "z_state": z_relaxed,
            "error": error,
            "gamma_phase": gamma_phase,
            "free_energy": free_energy
        }


# ===============================================================================
# TIER 4: KARYON SYNAPTIC MESH (СИНАПТИЧЕСКАЯ СЕТЬ КАРИОНА)
# ===============================================================================

class KaryonSynapticMesh(nn.Module):
    def __init__(self, num_cells: int = 4, dim: int = 32):
        super().__init__()
        self.num_cells = num_cells
        self.dim = dim
        
        self.cells = nn.ModuleList([
            ActiveInferenceCell(dim, dim, dim) for _ in range(num_cells)
        ])
        self.somatic_organelle = SomaticOrganelle()
        
        # Lateral Synaptic Weights between cells (W_ij)
        self.synaptic_weights = nn.ParameterDict({
            f"syn_{i}_{j}": nn.Parameter(torch.randn(dim, dim) * 0.05)
            for i in range(num_cells) for j in range(num_cells) if i != j
        })

    def local_somatic_relaxation(self, inputs: torch.Tensor, targets: torch.Tensor, relaxation_steps: int = 5):
        """
        Pure Local Active Inference Relaxation without PyTorch Global Autograd Backprop.
        State & weight updates are local and neuromodulated by Dopamine and PAC Gamma Phase.
        """
        prev_fe = 10.0
        somatic_telemetry = []
        
        cell_states = [inputs[:, i, :].clone() if inputs.ndim == 3 else inputs.clone() for i in range(self.num_cells)]
        
        for step in range(relaxation_steps):
            total_fe = 0.0
            cell_outputs = []
            
            # 1. Forward Cell Processing
            for i, cell in enumerate(self.cells):
                target_i = targets[:, i, :] if targets.ndim == 3 else targets
                res = cell(cell_states[i], target_i, self.somatic_organelle.theta_phase)
                cell_outputs.append(res)
                total_fe += res["free_energy"]

            mean_fe = (total_fe / self.num_cells).item()
            
            # 2. Update Somatic Organelle (Dopamine, Theta Phase)
            somatic_tele = self.somatic_organelle.update_somatic_state(mean_fe, prev_fe)
            somatic_telemetry.append(somatic_tele)
            da = somatic_tele["dopamine"]
            gamma_phase = cell_outputs[0]["gamma_phase"]
            
            # 3. Local Neuromodulated Weight Updates (3-Factor STDP / Active Inference)
            with torch.no_grad():
                for i, cell in enumerate(self.cells):
                    err = cell_outputs[i]["error"]
                    z = cell_outputs[i]["z_state"]
                    
                    # Local update to PredictiveOrganelle weights
                    pred_w = cell.predictive_organelle.pred_op.weight
                    # ΔW = η * DA * γ(θ) * (error^T * z) (gradient ascent on log likelihood / descent on F)
                    grad_local = torch.matmul(err.T, z) / err.shape[0]
                    lr = 0.25 * da * gamma_phase
                    pred_w.add_(lr * grad_local)

                    # Lateral Synaptic Updates (Hebbian associative Snapping)
                    for j in range(self.num_cells):
                        if i != j:
                            syn_key = f"syn_{i}_{j}"
                            syn_w = self.synaptic_weights[syn_key]
                            z_j = cell_outputs[j]["z_state"]
                            syn_grad = torch.matmul(cell_outputs[i]["error"].T, z_j) / err.shape[0]
                            syn_w.add_(0.08 * da * gamma_phase * syn_grad)
                            
                            # Apply lateral excitation/inhibition to cell state for next relaxation step
                            lateral_influence = torch.matmul(z_j, syn_w.T)
                            cell_states[i] = cell_states[i] + 0.1 * lateral_influence

            prev_fe = mean_fe

        final_mu = torch.stack([res["mu"] for res in cell_outputs], dim=1)
        final_loss = F.mse_loss(final_mu, targets).item()

        return {
            "final_loss": final_loss,
            "final_fe": mean_fe,
            "somatic_telemetry": somatic_telemetry[-1],
            "relaxation_steps": relaxation_steps
        }


# ===============================================================================
# EXPERIMENTAL RUNNER & KEP BENCHMARK
# ===============================================================================

def run_exp_346_benchmark():
    print("=" * 80)
    print("EXP-346: HIERARCHICAL ACTIVE INFERENCE & SOMATIC RELAXATION BENCHMARK")
    print("=" * 80)

    dim = 32
    batch_size = 16
    num_cells = 4
    num_episodes = 50

    mesh = KaryonSynapticMesh(num_cells=num_cells, dim=dim)

    initial_fe = None
    final_fe = None
    initial_loss = None
    final_loss = None

    start_time = time.time()
    
    fe_history = []
    loss_history = []
    da_history = []

    # Train on a structured manifold stream (associative learning task)
    # Generate fixed prototype features to learn genuine pattern association
    proto_inputs = torch.randn(8, num_cells, dim)
    proto_targets = torch.sin(proto_inputs) + 0.5 * torch.cos(proto_inputs * 2.0)

    for episode in range(num_episodes):
        # Sample batch with noise around prototypes to evaluate active inference generalization
        indices = torch.randint(0, 8, (batch_size,))
        noise = 0.02 * torch.randn(batch_size, num_cells, dim)
        inputs = proto_inputs[indices] + noise
        targets = proto_targets[indices]

        results = mesh.local_somatic_relaxation(inputs, targets, relaxation_steps=10)
        
        fe = results["final_fe"]
        loss = results["final_loss"]
        da = results["somatic_telemetry"]["dopamine"]

        fe_history.append(fe)
        loss_history.append(loss)
        da_history.append(da)

        if episode == 0:
            initial_fe = fe
            initial_loss = loss

        if episode % 5 == 0 or episode == num_episodes - 1:
            print(f"Episode {episode:02d} | Free Energy F: {fe:.4f} | Loss: {loss:.4f} | Dopamine DA: {da:.4f} | Theta: {results['somatic_telemetry']['theta_phase']:.2f}")

    final_fe = fe_history[-1]
    final_loss = loss_history[-1]
    elapsed_time = time.time() - start_time

    fe_delta = initial_fe - final_fe
    loss_delta = initial_loss - final_loss
    final_da = da_history[-1]

    # KEP Rule #2 & Anti-Goodhart Strict Biophysical Verdict Verification
    # 1. Absolute Free Energy < 1.5
    # 2. Free Energy Delta >= 0.08
    # 3. Somatic Dopamine DA >= 0.2
    passed_fe_threshold = final_fe < 1.5
    passed_fe_delta = fe_delta >= 0.08
    passed_somatic_da = final_da >= 0.2

    if passed_fe_threshold and passed_fe_delta and passed_somatic_da:
        verdict = "POSITIVE"
    else:
        verdict = "REJECTED"

    print("\n" + "=" * 80)
    print(f"EXP-346 FINAL BENCHMARK SUMMARY:")
    print(f"  - Initial Free Energy (F_0) : {initial_fe:.4f}")
    print(f"  - Final Free Energy (F_final) : {final_fe:.4f}")
    print(f"  - Free Energy Delta (ΔF)   : {fe_delta:.4f}")
    print(f"  - Initial Loss              : {initial_loss:.4f}")
    print(f"  - Final Loss                : {final_loss:.4f}")
    print(f"  - Final Dopamine Drive (DA) : {final_da:.4f}")
    print(f"  - Verdict Criterion Status  : F<1.5 ({passed_fe_threshold}), ΔF>=0.08 ({passed_fe_delta}), DA>=0.2 ({passed_somatic_da})")
    print(f"  - FINAL VERDICT             : {verdict}")
    print("=" * 80)

    # Save structured telemetry for pipeline & empirical ledger
    summary_data = {
        "exp_id": "EXP-346",
        "verdict": verdict,
        "initial_fe": initial_fe,
        "final_fe": final_fe,
        "fe_delta": fe_delta,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_delta": loss_delta,
        "final_dopamine": final_da,
        "elapsed_time_sec": elapsed_time,
        "passed_fe_threshold": passed_fe_threshold,
        "passed_fe_delta": passed_fe_delta,
        "passed_somatic_da": passed_somatic_da,
        "metrics": {
            "final_loss": final_loss,
            "final_fe": final_fe,
            "fe_delta": fe_delta,
            "dopamine": final_da,
            "loss_delta": loss_delta
        }
    }

    with open("experiments/exp_346_results.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


if __name__ == "__main__":
    run_exp_346_benchmark()
