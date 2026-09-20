# experiments/exp_272_bistable_latch_morphic_field.py
"""
===============================================================================
EXP-272: BISTABLE WORKING MEMORY LATCH CONTINUOUS MORPHIC FIELD
Biophysical Implementation of Principle 22:
- Dual-Zone Manifold:
    1. Volatile Sensory-Motor Wave Zone (D_dyn = 128) with continuous Heun SDE leak & pacemaker.
    2. Non-Volatile Bistable Attractor Latch Zone (D_mem = 64) with ZERO passive leak (gamma=0)
       and cubic bistability: dM/dt = M - M^3 + W_gate * Psi_dyn.
       This acts as continuous analog hysteresis (Schmitt trigger / line attractor),
       permanently holding the tokens 'a', 'b', and 'op' across the entire thinking phase!
- Bilinear Multiplicative Currents between Dyn and Mem spaces:
    I_compute = (W_left * M) * (W_right * M) projected back into Dyn.
- Soft Boltzmann-Poisson Action Emission on Motor Coordinates.
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


class BistableLatchMorphicField(nn.Module):
    def __init__(self, d_dyn: int = 128, d_mem: int = 64, vocab_size: int = 258, device: str = "cuda"):
        super().__init__()
        self.d_dyn = d_dyn
        self.d_mem = d_mem
        self.total_dim = d_dyn + d_mem
        self.vocab_size = vocab_size
        self.device = torch.device(device)

        # Attractor Basins across the dynamic manifold (vocab_size x d_dyn)
        raw_basins = torch.randn(vocab_size, d_dyn, device=self.device) * 0.2
        self.basins = nn.Parameter(F.normalize(raw_basins, p=2, dim=-1))

        # Dynamic Recurrent Routing in Dyn space
        self.w_dyn = nn.Parameter(
            torch.eye(d_dyn, device=self.device) * 0.1 + 
            torch.randn(d_dyn, d_dyn, device=self.device) * (0.1 / math.sqrt(d_dyn))
        )

        # Sensory to Memory Gating (Projects sensory inputs into persistent bistable memory)
        self.w_write_gate = nn.Parameter(torch.randn(d_mem, d_dyn, device=self.device) * (0.2 / math.sqrt(d_dyn)))

        # Bilinear Computation from Persistent Memory into Dyn: (W_l * M) * (W_r * M)
        self.w_left = nn.Parameter(torch.randn(d_dyn, d_mem, device=self.device) * (0.25 / math.sqrt(d_mem)))
        self.w_right = nn.Parameter(torch.randn(d_dyn, d_mem, device=self.device) * (0.25 / math.sqrt(d_mem)))
        self.w_bilinear_out = nn.Parameter(torch.randn(d_dyn, d_dyn, device=self.device) * (0.15 / math.sqrt(d_dyn)))

        # Pacemaker parameters (Theta-Gamma oscillations in Dyn)
        self.omega_theta = nn.Parameter(torch.tensor(5.0, device=self.device))
        self.omega_gamma = nn.Parameter(torch.tensor(40.0, device=self.device))
        self.pacemaker_vec = nn.Parameter(torch.randn(d_dyn, device=self.device) * 0.1)

    def forward_continuous(
        self,
        prompt_bytes: List[int],
        max_duration: float = 6.0,
        dt: float = 0.05,
        target_bytes: Optional[List[int]] = None
    ) -> Tuple[List[int], torch.Tensor]:
        B = 1
        psi_dyn = torch.zeros(B, self.d_dyn, device=self.device)
        psi_mem = torch.zeros(B, self.d_mem, device=self.device)
        
        emitted_bytes = []
        free_energy_accum = torch.tensor(0.0, device=self.device)
        
        t_current = 0.0
        byte_duration = 0.25
        prompt_idx = 0
        target_idx = 0
        
        gamma_dyn = 0.08
        beta_precision = 7.0
        refractory_state = torch.zeros(B, self.d_dyn, device=self.device)

        while t_current < max_duration:
            # 1. Sensory Inflow
            sensory_inflow = torch.zeros_like(psi_dyn)
            is_sensory_active = False
            if prompt_idx < len(prompt_bytes):
                byte_val = prompt_bytes[prompt_idx]
                sensory_inflow = self.basins[byte_val].unsqueeze(0)
                is_sensory_active = True
                if (t_current - (prompt_idx * byte_duration)) >= byte_duration:
                    prompt_idx += 1

            # 2. Pacemaker Drive (Theta & Gamma waves in Dyn)
            theta_wave = torch.sin(self.omega_theta * t_current)
            gamma_wave = torch.sin(self.omega_gamma * t_current)
            pacemaker_drive = (0.20 * theta_wave + 0.10 * gamma_wave) * self.pacemaker_vec.unsqueeze(0)

            # 3. Attractor Basin Force Landscape on Dyn
            psi_norm = F.normalize(psi_dyn + 1e-6, p=2, dim=-1)
            similarities = torch.matmul(psi_norm, self.basins.t())
            attractor_probs = torch.softmax(similarities * beta_precision, dim=-1)
            attractor_target = torch.matmul(attractor_probs, self.basins)
            attractor_force = 1.6 * (attractor_target - psi_dyn)

            # 4. Bilinear Computation from Persistent Memory:
            # Multiplicative tensor cross-current from memory latches
            m_left = torch.matmul(psi_mem, self.w_left.t())
            m_right = torch.matmul(psi_mem, self.w_right.t())
            compute_current = torch.matmul(F.silu(m_left * m_right), self.w_bilinear_out.t())

            # Dynamic routing in Dyn
            dyn_routing = torch.matmul(torch.tanh(psi_dyn), self.w_dyn.t())

            # 5. Continuous Integration (Heun SDE for Dyn, Bistable Latch for Mem)
            noise_dyn = torch.randn_like(psi_dyn) * 0.012 * math.sqrt(dt)
            
            # Drift for Dyn
            f_dyn = (
                -gamma_dyn * psi_dyn
                + dyn_routing
                + compute_current
                + sensory_inflow
                + pacemaker_drive
                + attractor_force
                - refractory_state
            )
            
            # Bistable Latch dynamics for Mem:
            # dM/dt = M * (1 - M^2) + W_write * (Sensory + Dyn)
            # When input goes away, M settles into +1 or -1 stable fixed points (hysteresis!)
            write_drive = torch.matmul(sensory_inflow + 0.5 * torch.tanh(psi_dyn), self.w_write_gate.t()) if is_sensory_active else torch.zeros_like(psi_mem)
            f_mem = 0.5 * (psi_mem - torch.pow(psi_mem, 3)) + write_drive
            
            # Predictor step
            psi_dyn_tilde = psi_dyn + f_dyn * dt + noise_dyn
            psi_mem_tilde = psi_mem + f_mem * dt

            # Corrector evaluation
            psi_dyn_tilde_norm = F.normalize(psi_dyn_tilde + 1e-6, p=2, dim=-1)
            sim_tilde = torch.matmul(psi_dyn_tilde_norm, self.basins.t())
            attn_tilde = torch.softmax(sim_tilde * beta_precision, dim=-1)
            attractor_force_tilde = 1.6 * (torch.matmul(attn_tilde, self.basins) - psi_dyn_tilde)
            
            dyn_routing_tilde = torch.matmul(torch.tanh(psi_dyn_tilde), self.w_dyn.t())
            m_left_tilde = torch.matmul(psi_mem_tilde, self.w_left.t())
            m_right_tilde = torch.matmul(psi_mem_tilde, self.w_right.t())
            compute_tilde = torch.matmul(F.silu(m_left_tilde * m_right_tilde), self.w_bilinear_out.t())
            
            f_dyn_tilde = (
                -gamma_dyn * psi_dyn_tilde
                + dyn_routing_tilde
                + compute_tilde
                + sensory_inflow
                + pacemaker_drive
                + attractor_force_tilde
                - refractory_state
            )
            
            write_drive_tilde = torch.matmul(sensory_inflow + 0.5 * torch.tanh(psi_dyn_tilde), self.w_write_gate.t()) if is_sensory_active else torch.zeros_like(psi_mem)
            f_mem_tilde = 0.5 * (psi_mem_tilde - torch.pow(psi_mem_tilde, 3)) + write_drive_tilde

            # State update
            psi_dyn = psi_dyn + 0.5 * (f_dyn + f_dyn_tilde) * dt + noise_dyn
            psi_mem = psi_mem + 0.5 * (f_mem + f_mem_tilde) * dt
            
            psi_dyn = torch.clamp(psi_dyn, -4.0, 4.0)
            psi_mem = torch.clamp(psi_mem, -2.5, 2.5)

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

            # 7. Thermodynamic Action Emission
            if prompt_idx >= len(prompt_bytes):
                top_prob, top_idx = torch.max(attractor_probs[0], dim=-1)
                psi_energy = torch.norm(psi_dyn, p=2).item()
                
                spike_rate = 10.0 * top_prob.item() * (1.0 / (1.0 + math.exp(-2.5 * (psi_energy - 0.4))))
                p_spike = 1.0 - math.exp(-spike_rate * dt)
                
                if random.random() < p_spike and top_idx.item() not in (256,):
                    emitted_byte = top_idx.item()
                    emitted_bytes.append(emitted_byte)
                    
                    if target_bytes is not None and target_idx < len(target_bytes):
                        target_idx += 1
                    
                    refractory_state = refractory_state + 2.2 * self.basins[emitted_byte].unsqueeze(0)
                    
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
    print("EXP-272: BISTABLE WORKING MEMORY LATCH CONTINUOUS MORPHIC FIELD")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware substrate: {device}")

    train_data, val_id, val_ood = create_dataset()
    print(f"Dataset: Train={len(train_data)}, In-Domain Val={len(val_id)}, OOD Val={len(val_ood)}")

    model = BistableLatchMorphicField(d_dyn=128, d_mem=64, vocab_size=258, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    epochs = 40
    batch_samples = 30
    
    print("\n--- Training Phase (Continuous SDE Free Energy Optimization) ---")
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
    print("EXP-272 TELEMETRY SUMMARY")
    print("=" * 80)
    print(f"Final Free Energy : {avg_loss:.4f}")
    print(f"In-Domain Acc     : {id_acc:.2f}%")
    print("=" * 80)

    metrics = {
        "final_loss": float(avg_loss),
        "in_domain_acc": float(id_acc),
        "train_duration_s": float(train_duration)
    }
    with open("experiments/exp_272_metrics.json", "w") as f:
        json.dump(metrics, f)

    return metrics


if __name__ == "__main__":
    run_experiment()
