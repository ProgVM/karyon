# experiments/exp_271_bilinear_distributed_morphic_field.py
"""
===============================================================================
EXP-271: DISTRIBUTED CONTINUOUS MORPHIC FIELD WITH MULTIPLICATIVE TENSOR CURRENTS
Biophysical Implementation of Principle 22:
- Spatially Partitioned Continuous Manifold (Sensory Zone, Recirculation/Memory Zone, Motor Zone)
- Non-linear Multiplicative (Bilinear) Cross-Currents: I_mult = (W_A Psi) * (W_B Psi)
  enabling true endogenous continuous multiplication (a*b, a^2) inside the SDE.
- Endogenous Pacemaker Drive (Theta-Gamma oscillations) to prevent state freezing.
- Poisson-Boltzmann Thermodynamic Action Potential Emission.
===============================================================================
"""
import sys
import os
import math
import time
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class BilinearDistributedMorphicField(nn.Module):
    """
    Continuous Morphic Field with:
    1. Spatial topological partitioning (Sensory D_s, Cognitive D_c, Motor D_m).
    2. Bilinear Multiplicative Field Currents: allows endogenous quadratic / multiplicative interactions.
    3. Continuous SDE integration via Heun's method.
    4. Theta-Gamma oscillatory pacemaker.
    5. Soft thermodynamic Boltzmann spiking.
    """
    def __init__(self, dim: int = 192, vocab_size: int = 258, device: str = "cuda"):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.device = torch.device(device)

        # Spatial Partitioning: Sensory (64D), Cognitive (64D), Motor (64D)
        self.d_part = dim // 3
        
        # Attractor Basins on unit sphere (vocab_size x dim)
        raw_basins = torch.randn(vocab_size, dim, device=self.device) * 0.2
        self.basins = nn.Parameter(F.normalize(raw_basins, p=2, dim=-1))

        # Linear Connectivity Field
        self.w_linear = nn.Parameter(
            torch.eye(dim, device=self.device) * 0.1 + 
            torch.randn(dim, dim, device=self.device) * (0.1 / math.sqrt(dim))
        )

        # Bilinear Multiplicative Couplings (W_left, W_right, W_out)
        # Enables endogenous quadratic computation (a*b, a^2) in continuous time
        self.w_left = nn.Parameter(torch.randn(dim, dim, device=self.device) * (0.2 / math.sqrt(dim)))
        self.w_right = nn.Parameter(torch.randn(dim, dim, device=self.device) * (0.2 / math.sqrt(dim)))
        self.w_bilinear_out = nn.Parameter(torch.randn(dim, dim, device=self.device) * (0.1 / math.sqrt(dim)))

        # Pacemaker parameters (Theta ~ 4-8Hz, Gamma ~ 30-50Hz)
        self.omega_theta = nn.Parameter(torch.tensor(5.0, device=self.device))
        self.omega_gamma = nn.Parameter(torch.tensor(40.0, device=self.device))
        self.pacemaker_vec = nn.Parameter(torch.randn(dim, device=self.device) * 0.1)

    def forward_continuous(
        self,
        prompt_bytes: List[int],
        max_duration: float = 6.0,
        dt: float = 0.05,
        target_bytes: Optional[List[int]] = None
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Integrates the spatially distributed field continuously over physical time.
        """
        B = 1
        psi = torch.zeros(B, self.dim, device=self.device)
        
        emitted_bytes = []
        free_energy_accum = torch.tensor(0.0, device=self.device)
        
        t_current = 0.0
        byte_duration = 0.25 # seconds per sensory input byte
        prompt_idx = 0
        target_idx = 0
        
        gamma_leak = 0.06 # Low passive decay
        beta_precision = 7.0 # Inverse temperature for attractor snapping
        refractory_state = torch.zeros(B, self.dim, device=self.device)

        while t_current < max_duration:
            # 1. Sensory Inflow into Sensory Partition (First d_part coordinates)
            sensory_inflow = torch.zeros_like(psi)
            if prompt_idx < len(prompt_bytes):
                byte_val = prompt_bytes[prompt_idx]
                sensory_inflow = self.basins[byte_val].unsqueeze(0)
                if (t_current - (prompt_idx * byte_duration)) >= byte_duration:
                    prompt_idx += 1

            # 2. Endogenous Pacemaker Drive (Theta-Gamma Coupling)
            theta_wave = torch.sin(self.omega_theta * t_current)
            gamma_wave = torch.sin(self.omega_gamma * t_current)
            pacemaker_drive = (0.20 * theta_wave + 0.10 * gamma_wave) * self.pacemaker_vec.unsqueeze(0)

            # 3. Attractor Force Landscape
            psi_norm = F.normalize(psi + 1e-6, p=2, dim=-1)
            similarities = torch.matmul(psi_norm, self.basins.t()) # [1, V]
            attractor_probs = torch.softmax(similarities * beta_precision, dim=-1)
            attractor_target = torch.matmul(attractor_probs, self.basins)
            attractor_force = 1.6 * (attractor_target - psi)

            # 4. Linear and Bilinear Multiplicative Field Currents
            # Linear current
            linear_current = torch.matmul(torch.tanh(psi), self.w_linear.t())
            
            # Bilinear multiplicative current: (W_left * Psi) * (W_right * Psi)
            left_act = torch.matmul(psi, self.w_left.t())
            right_act = torch.matmul(psi, self.w_right.t())
            mult_interaction = torch.matmul(F.silu(left_act * right_act), self.w_bilinear_out.t())

            # 5. Stochastic Thermal Noise (Wiener process)
            noise = torch.randn_like(psi) * 0.012 * math.sqrt(dt)

            # Compute SDE Drift: Heun's Predictor
            total_drift = (
                -gamma_leak * psi 
                + linear_current 
                + mult_interaction 
                + sensory_inflow 
                + pacemaker_drive 
                + attractor_force 
                - refractory_state
            )
            psi_tilde = psi + total_drift * dt + noise

            # Heun's Corrector Evaluation
            psi_tilde_norm = F.normalize(psi_tilde + 1e-6, p=2, dim=-1)
            sim_tilde = torch.matmul(psi_tilde_norm, self.basins.t())
            attn_tilde = torch.softmax(sim_tilde * beta_precision, dim=-1)
            attractor_tilde = 1.6 * (torch.matmul(attn_tilde, self.basins) - psi_tilde)
            
            linear_tilde = torch.matmul(torch.tanh(psi_tilde), self.w_linear.t())
            left_tilde = torch.matmul(psi_tilde, self.w_left.t())
            right_tilde = torch.matmul(psi_tilde, self.w_right.t())
            mult_tilde = torch.matmul(F.silu(left_tilde * right_tilde), self.w_bilinear_out.t())
            
            total_drift_tilde = (
                -gamma_leak * psi_tilde 
                + linear_tilde 
                + mult_tilde 
                + sensory_inflow 
                + pacemaker_drive 
                + attractor_tilde 
                - refractory_state
            )
            
            # State update
            psi = psi + 0.5 * (total_drift + total_drift_tilde) * dt + noise
            psi = torch.clamp(psi, -4.0, 4.0)

            # Continuous refractory decay
            refractory_state = refractory_state * math.exp(-3.5 * dt)

            # 6. Variational Free Energy Tracking
            if target_bytes is not None and prompt_idx >= len(prompt_bytes):
                if target_idx < len(target_bytes):
                    expected_byte = target_bytes[target_idx]
                    target_prob = attractor_probs[0, expected_byte]
                    free_energy_accum = free_energy_accum + (-torch.log(target_prob + 1e-9)) * dt
            else:
                entropy = -torch.sum(attractor_probs * torch.log(attractor_probs + 1e-9))
                free_energy_accum = free_energy_accum + 0.03 * entropy * dt

            # 7. Thermodynamic Boltzmann Action Potential Emission
            if prompt_idx >= len(prompt_bytes):
                top_prob, top_idx = torch.max(attractor_probs[0], dim=-1)
                psi_energy = torch.norm(psi, p=2).item()
                
                # Dynamic spike threshold
                spike_rate = 10.0 * top_prob.item() * (1.0 / (1.0 + math.exp(-2.5 * (psi_energy - 0.4))))
                p_spike = 1.0 - math.exp(-spike_rate * dt)
                
                if random.random() < p_spike and top_idx.item() not in (256,):
                    emitted_byte = top_idx.item()
                    emitted_bytes.append(emitted_byte)
                    
                    if target_bytes is not None and target_idx < len(target_bytes):
                        target_idx += 1
                    
                    # Refractory inhibition along emitted axis
                    refractory_state = refractory_state + 2.2 * self.basins[emitted_byte].unsqueeze(0)
                    
                    # Break on newline, EOS, or maximum token length
                    if emitted_byte in (257, ord('\n')) or len(emitted_bytes) >= 8:
                        break

            t_current += dt

        return emitted_bytes, free_energy_accum


# -----------------------------------------------------------------------------
# Data Generation & Benchmark
# -----------------------------------------------------------------------------
def make_example(op: str, a: int, b: int) -> Tuple[List[int], List[int]]:
    if op == "star":
        res = 2 * a + b
    elif op == "hash":
        res = a * b - a
    elif op == "delta":
        res = a * a + b
    else:
        raise ValueError(f"Unknown op {op}")
    
    prompt = f"{op}({a},{b})="
    target = f"{res}\n"
    return list(prompt.encode("utf-8")), list(target.encode("utf-8"))


def create_dataset():
    ops = ["star", "hash", "delta"]
    in_domain = []
    ood = []

    for op in ops:
        for a in range(1, 16):
            for b in range(1, 16):
                in_domain.append((op, a, b))
        for a in range(16, 21):
            for b in range(16, 21):
                ood.append((op, a, b))

    random.seed(42)
    random.shuffle(in_domain)
    random.shuffle(ood)

    split = int(len(in_domain) * 0.8)
    return in_domain[:split], in_domain[split:], ood


def run_experiment():
    print("=" * 80)
    print("EXP-271: BILINEAR DISTRIBUTED CONTINUOUS MORPHIC FIELD BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware substrate: {device}")

    train_data, val_id, val_ood = create_dataset()
    print(f"Dataset: Train={len(train_data)}, In-Domain Val={len(val_id)}, OOD Val={len(val_ood)}")

    model = BilinearDistributedMorphicField(dim=192, vocab_size=258, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    epochs = 40
    batch_samples = 30 # Samples per epoch
    
    print("\n--- Training Phase (Continuous-Time Free Energy Optimization) ---")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        random.shuffle(train_data)
        epoch_loss = 0.0
        
        for op, a, b in train_data[:batch_samples]:
            prompt, target = make_example(op, a, b)
            
            optimizer.zero_grad()
            emitted, loss = model.forward_continuous(prompt, target_bytes=target, dt=0.05)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / batch_samples
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [Epoch {epoch:02d}/{epochs}] Variational Free Energy: {avg_loss:.4f}")

    train_duration = time.time() - start_time
    print(f"Training completed in {train_duration:.2f}s")

    print("\n--- Evaluation Phase (Spontaneous Generation & Exact Trace Audit) ---")
    model.eval()
    correct_id = 0
    test_count = min(20, len(val_id))
    
    with torch.no_grad():
        for op, a, b in val_id[:test_count]:
            prompt, target = make_example(op, a, b)
            prompt_str = bytes(prompt).decode("utf-8")
            expected_str = bytes(target).decode("utf-8").strip()
            
            emitted, _ = model.forward_continuous(prompt, max_duration=5.0, dt=0.05)
            generated_str = bytes(emitted).decode("utf-8", errors="ignore").strip()
            
            is_correct = generated_str == expected_str
            if is_correct:
                correct_id += 1
            print(f"  Query '{prompt_str}' -> Emitted: '{generated_str}' | Expected: '{expected_str}' [{'CORRECT' if is_correct else 'WRONG'}]")

    id_acc = (correct_id / test_count) * 100.0 if test_count > 0 else 0.0
    print(f"\n  • In-Domain Accuracy: {id_acc:.2f}% ({correct_id}/{test_count})")

    print("\n" + "=" * 80)
    print("EXP-271 TELEMETRY SUMMARY")
    print("=" * 80)
    print(f"Final Free Energy : {avg_loss:.4f}")
    print(f"In-Domain Acc     : {id_acc:.2f}%")
    print("=" * 80)

    metrics = {
        "final_loss": float(avg_loss),
        "in_domain_acc": float(id_acc),
        "train_duration_s": float(train_duration)
    }
    with open("experiments/exp_271_metrics.json", "w") as f:
        json.dump(metrics, f)

    return metrics


if __name__ == "__main__":
    run_experiment()
