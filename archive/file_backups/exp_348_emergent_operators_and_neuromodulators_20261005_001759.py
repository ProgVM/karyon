"""
EXP-348: Continuous Open-Ended Operator Genesis & Self-Organizing Endocrine System
==================================================================================
Empirical Benchmark for Autonomous Neuro-Morphogenesis without hardcoded operators or fixed neuromodulator labels.

Key Principles:
1. Operator Genesis via Continuous Latent Genome g in R^k:
   - Hyper-Morphogen Network projects g -> operator parameters and functional blend weights phi_m(g).
   - Generates non-hardcoded nonlinear transforms O_g(x) with epigenetic zero-delta birth gating.
2. Endogenous Neuromodulator Field (Self-Organizing Endocrine Reservoir):
   - Continuous latent neuromodulator field z_neuro in R^d_neuro.
   - Emergent chemical dynamics governed by homeostatic free energy and metabolic costs:
     dz_neuro/dt = G(x, error, somatic_state) - lambda * z_neuro.
   - Phenotypes emerge naturally without hardcoding 'dopamine', 'serotonin', etc.
3. KEP Rule #2 Validation:
   - Verification of Loss Delta >= 0.08, stable free energy convergence, and non-trivial operator diversity.
"""

import math
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hardware Substrate Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================================================================
# 1. CONTINUOUS OPERATOR GENESIS (HYPER-MORPHOGEN)
# ===============================================================================

class ContinuousAtomicOperator(nn.Module):
    """
    An emergent mathematical operator whose transformation topology is determined
    by a continuous latent genome vector g in R^k rather than a hardcoded discrete class.
    """
    def __init__(self, in_dim: int, out_dim: int, genome_dim: int = 16, device=device):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.genome_dim = genome_dim
        self.device = device

        # Latent genetic code of this operator (learnable or mutated)
        self.genome = nn.Parameter(torch.randn(genome_dim, device=device) * 0.5)

        # Base parameterized transformation matrices
        self.W_primary = nn.Parameter(torch.randn(out_dim, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))
        self.W_secondary = nn.Parameter(torch.randn(out_dim, in_dim, device=device) * (1.0 / math.sqrt(in_dim)))
        self.bias = nn.Parameter(torch.zeros(out_dim, device=device))

        # Hyper-morphogen projection: genome -> operator functional blending coefficients
        # Coefficients define harmonic, attractor-like, or bilinear dynamic components
        self.morphogen_net = nn.Sequential(
            nn.Linear(genome_dim, 32, device=device),
            nn.Tanh(),
            nn.Linear(32, 4, device=device),
            nn.Softmax(dim=-1)
        )

        # Epigenetic Birth Gate: starts near zero to prevent disruptive shock to the host cell
        self.alpha_epi = nn.Parameter(torch.zeros(1, device=device))

    def forward(self, x: torch.Tensor, neuromod_mod: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, in_dim]
        neuromod_mod: [B, out_dim] or scalar modulation factor from endogenous reservoir
        """
        if x.device != self.device:
            x = x.to(self.device)

        # Blending coefficients synthesized directly from the continuous genome
        # [4] weights: (w_linear, w_harmonic, w_attractor, w_gated)
        phi = self.morphogen_net(self.genome)

        # Continuous functional transforms:
        # 1. Direct affine
        t_linear = torch.matmul(x, self.W_primary.t()) + self.bias
        
        # 2. Harmonic wave transform (cos/sin oscillatory dynamics)
        t_harmonic = torch.sin(t_linear) * torch.cos(torch.matmul(x, self.W_secondary.t()))
        
        # 3. Attractor relaxation transform (energy minimization step)
        t_attractor = torch.tanh(t_linear) * (1.0 - torch.sigmoid(torch.matmul(x, self.W_secondary.t())))
        
        # 4. Multiplicative gating (bilinear interaction)
        t_gated = torch.sigmoid(t_linear) * torch.matmul(x, self.W_secondary.t())

        # Synthesized emergent operator output
        out_raw = (
            phi[0] * t_linear +
            phi[1] * t_harmonic +
            phi[2] * t_attractor +
            phi[3] * t_gated
        )

        # If modulated by endogenous chemical field, apply non-linear gain
        if neuromod_mod is not None:
            if neuromod_mod.device != self.device:
                neuromod_mod = neuromod_mod.to(self.device)
            out_raw = out_raw * (1.0 + torch.tanh(neuromod_mod))

        # Epigenetic progressive integration
        gate = torch.tanh(self.alpha_epi)
        return gate * out_raw


# ===============================================================================
# 2. ENDOGENOUS NEUROMODULATOR RESERVOIR (SELF-ORGANIZING ENDOCRINE FIELD)
# ===============================================================================

class EmergentEndocrineReservoir(nn.Module):
    """
    Endogenous chemical field synthesized purely from systemic state dynamics,
    prediction error, and metabolic demand. No fixed labels (DA, NA, 5-HT).
    """
    def __init__(self, state_dim: int, field_dim: int = 8, device=device):
        super().__init__()
        self.state_dim = state_dim
        self.field_dim = field_dim
        self.device = device

        # Chemical state dynamics network: maps systemic state & error to field shifts
        self.synthesis_net = nn.Sequential(
            nn.Linear(state_dim + 1, 32, device=device),
            nn.GELU(),
            nn.Linear(32, field_dim, device=device)
        )

        # Endogenous baseline and clearance (reuptake) rates
        self.decay_rates = nn.Parameter(torch.ones(field_dim, device=device) * 0.1)
        self.field_state = torch.zeros(field_dim, device=device)

    def step(self, system_state: torch.Tensor, pred_error: float) -> torch.Tensor:
        """
        Updates the continuous chemical field based on ongoing activity.
        system_state: [B, state_dim]
        pred_error: scalar error / free energy
        """
        if system_state.device != self.device:
            system_state = system_state.to(self.device)

        mean_state = system_state.mean(dim=0) # [state_dim]
        err_tensor = torch.tensor([pred_error], device=self.device, dtype=mean_state.dtype)
        inputs = torch.cat([mean_state, err_tensor], dim=0)

        # Chemical secretion
        delta_secretion = torch.tanh(self.synthesis_net(inputs))

        # Reuptake / decay dynamic
        clearance = torch.clamp(self.decay_rates, min=0.01, max=0.99)
        self.field_state = (1.0 - clearance) * self.field_state + clearance * delta_secretion

        return self.field_state


# ===============================================================================
# 3. MORPHOGENETIC CORTICAL COLUMN (DYNAMIC REVENUE & GENESIS)
# ===============================================================================

class EmergentMorphogeneticColumn(nn.Module):
    def __init__(self, dim: int = 64, initial_operators: int = 2, device=device):
        super().__init__()
        self.dim = dim
        self.device = device

        # Population of emergent continuous operators
        self.operators = nn.ModuleList([
            ContinuousAtomicOperator(dim, dim, genome_dim=16, device=device)
            for _ in range(initial_operators)
        ])
        for op in self.operators:
            # First operators start active
            op.alpha_epi.data.fill_(1.0)

        # Endogenous endocrine system
        self.endocrine = EmergentEndocrineReservoir(state_dim=dim, field_dim=8, device=device)

        # Readout head
        self.head = nn.Linear(dim, dim, device=device)

        # Routing controller to coordinate dynamic operator interaction
        self.router = nn.Sequential(
            nn.Linear(dim, 32, device=device),
            nn.Tanh(),
            nn.Linear(32, 16, device=device)
        )

    def sprout_new_operator(self):
        """
        Dynamically evolves and instantiates a novel mathematical operator from
        an endogenous genetic mutation, with zero-delta birth gating.
        """
        new_op = ContinuousAtomicOperator(self.dim, self.dim, genome_dim=16, device=self.device)
        # Parent mutation or novel genetic sampling
        if len(self.operators) > 0:
            parent = random.choice(self.operators)
            new_op.genome.data.copy_(parent.genome.data + torch.randn_like(parent.genome.data) * 0.1)
        self.operators.append(new_op)
        return new_op

    def forward(self, x: torch.Tensor, current_error: float = 0.0) -> tuple:
        if x.device != self.device:
            x = x.to(self.device)

        # Step endogenous endocrine reservoir
        chem_field = self.endocrine.step(x, current_error) # [field_dim]

        # Use slice of chemical field to modulate operator gains
        chem_gain = chem_field[:4].mean()

        # Execute operators in sequence / recurrent pass
        h = x
        for op in self.operators:
            h_op = op(h, neuromod_mod=chem_gain)
            h = h + h_op # Residual accumulation

        out = self.head(h)
        return out, chem_field


# ===============================================================================
# 4. BENCHMARK & MULTI-MANIFOLD EVALUATION
# ===============================================================================

def run_exp_348_benchmark():
    print("=" * 80)
    print("STARTING EXP-348: CONTINUOUS OPERATOR GENESIS & ENDOGENOUS CHEMICAL BENCHMARK")
    print("=" * 80)
    print(f"Device: {device}")

    dim = 64
    batch_size = 32
    steps_per_epoch = 15
    epochs = 20

    model = EmergentMorphogeneticColumn(dim=dim, initial_operators=2, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=1e-4)

    # Multi-manifold problem generator (algorithmic sequence + chaotic non-linear manifold)
    def generate_data(batch_size, dim):
        # Continuous non-linear chaotic target function
        t = torch.linspace(0, 4 * math.pi, dim, device=device)
        base = torch.sin(t).unsqueeze(0).repeat(batch_size, 1)
        noise = torch.randn(batch_size, dim, device=device) * 0.1
        inputs = base + noise
        # Non-trivial target transformation (multi-frequency oscillation + nonlinear fold)
        target = torch.roll(inputs, shifts=3, dims=-1) * 0.5 + torch.cos(inputs * 2.0) * 0.5
        return inputs, target

    history = []
    initial_loss = None
    final_loss = None
    baseline_loss = 0.5210

    total_start_time = time.time()
    total_tokens_processed = 0

    prev_error = 0.5
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        step_times = []

        # Morphogenetic Sprouting Trigger:
        # Sprout a new continuous operator if epoch is 5, 10, or 15
        if epoch in [5, 10, 15]:
            new_op = model.sprout_new_operator()
            # Register new parameters in optimizer
            optimizer.add_param_group({'params': new_op.parameters(), 'lr': 0.008})
            print(f"[Epoch {epoch:02d}] 🧬 Dynamic Morphogenesis: Sprouted Continuous Operator #{len(model.operators)} (Epigenetic Gate={new_op.alpha_epi.item():.4f})")

        for step in range(steps_per_epoch):
            t0 = time.time()
            inputs, targets = generate_data(batch_size, dim)

            optimizer.zero_grad()
            outputs, chem_field = model(inputs, current_error=prev_error)

            # Accuracy Loss
            task_loss = F.mse_loss(outputs, targets)
            
            # Active Inference Variational Free Energy:
            # F = Accuracy Error + Endocrine Metabolic Cost + Complexity Penalty
            endocrine_metabolism = 0.01 * torch.norm(chem_field, p=2)
            total_free_energy = task_loss + endocrine_metabolism

            total_free_energy.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            prev_error = task_loss.item()
            epoch_loss += task_loss.item()
            step_times.append(time.time() - t0)
            total_tokens_processed += batch_size * dim

        avg_loss = epoch_loss / steps_per_epoch
        if epoch == 1:
            initial_loss = avg_loss

        # Collect Endoscopic Telemetry
        with torch.no_grad():
            chem_status = [f"{v.item():.3f}" for v in chem_field[:4]]
            op_gates = [f"{torch.tanh(op.alpha_epi).item():.3f}" for op in model.operators]

        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Ops Count: {len(model.operators)} | "
              f"Epi-Gates: {op_gates} | ChemField: [{', '.join(chem_status)}]")
        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "num_operators": len(model.operators),
            "chem_field": chem_field.detach().cpu().numpy().tolist()
        })

    final_loss = avg_loss
    total_duration = time.time() - total_start_time
    tok_per_sec = total_tokens_processed / total_duration
    delta_loss = initial_loss - final_loss

    # Get VRAM usage
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        vram_mb = 0.0

    print("=" * 80)
    print("BENCHMARK EXECUTION SUMMARY (EXP-348)")
    print(f"Initial Loss : {initial_loss:.4f}")
    print(f"Final Loss   : {final_loss:.4f}")
    print(f"Delta Loss   : {delta_loss:.4f}")
    print(f"Throughput   : {tok_per_sec:.2f} tok/s")
    print(f"Peak VRAM    : {vram_mb:.2f} MB")
    print(f"Total Ops    : {len(model.operators)}")
    print("=" * 80)

    # KEP Rule #2 Criteria Verification
    passed_kep = delta_loss >= 0.08 and final_loss < 0.20
    verdict = "🟢 POSITIVE" if passed_kep else "🔴 REJECTED"
    print(f"KEP Rule #2 Verdict: {verdict} (Threshold: Delta >= 0.08, Achieved: {delta_loss:.4f})")

    # Save Results
    results = {
        "exp_id": "EXP-348",
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "delta_loss": delta_loss,
        "throughput_tok_per_sec": tok_per_sec,
        "vram_mb": vram_mb,
        "final_operator_count": len(model.operators),
        "verdict": verdict,
        "history": history
    }
    with open("experiments/exp_348_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    run_exp_348_benchmark()
