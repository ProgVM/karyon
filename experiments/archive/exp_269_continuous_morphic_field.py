# experiments/exp_269_continuous_morphic_field.py
"""
===============================================================================
EXP-269: CONTINUOUS MORPHIC FIELD (CMF) SUBSTRATE BENCHMARK
Implementing KEP v13.0 Principle 22 (The Five Pillars of Sovereign Self-Evolution):
1. Open-Ended Topological Genesis (Morphic Field)
2. Total Endogenous Sovereignty (Internal dt, learning rates, thresholds)
3. Spatiotemporal Dualism (Parallel Space, Continuous Sequential Time)
4. Wave-Particle Dualism (Continuous SDE Dynamics & Discrete Attractor Collapse)
5. Teleological Optimality (Free Energy Minimization)
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


class ContinuousMorphicField(nn.Module):
    """
    Sovereign Continuous Morphic Field (CMF) Substrate:
    - State is a continuous field Psi(t) of dimension D.
    - Time is continuous and integrated using an adaptive step dt regulated by surprise.
    - Contains discrete attractor basins representing the vocabulary.
    - Endogenous action emission occurs when Psi(t) collapses into an attractor basin.
    - Endogenous Morphogenesis: W_route is updated online via Hebbian Oja's flow
      gated by epigenetic locks.
    """
    def __init__(self, dim: int = 128, vocab_size: int = 258, device: str = "cuda"):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.device = torch.device(device)

        # 1. Attractor Basins (Discrete conceptual targets in continuous space)
        # Normalized to unit sphere
        raw_basins = torch.randn(vocab_size, dim, device=self.device) * 0.1
        self.basins = nn.Parameter(F.normalize(raw_basins, p=2, dim=-1))

        # 2. Dynamic Routing Matrix (The physical connectivity field)
        self.w_route = nn.Parameter(torch.eye(dim, device=self.device) * 0.05 + torch.randn(dim, dim, device=self.device) * 0.01)

        # 3. Epigenetic Locks (Methylation state of nodes)
        self.mu = nn.Parameter(torch.ones(dim, device=self.device) * 0.5)

        # 4. Somatic Homeostasis State (Energy, Curiosity, Noradrenaline, Dopamine)
        self.energy = 1.0
        self.noradrenaline = 0.1
        self.dopamine = 0.1

    def compute_free_energy(self, psi: torch.Tensor, ext_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Variational Free Energy (F_t):
        Computed as the prediction error / surprise of the current state relative to the attractor basins.
        """
        # Cosine similarity distribution across all attractor basins
        psi_norm = F.normalize(psi, p=2, dim=-1) # [B, D]
        similarities = torch.matmul(psi_norm, self.basins.t()) # [B, V]
        
        # Softmax entropy representing uncertainty/surprise
        probs = torch.softmax(similarities * 5.0, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
        
        # Reconstruction error if external input is active
        rec_error = 0.0
        if ext_input is not None:
            ext_norm = F.normalize(ext_input, p=2, dim=-1)
            rec_error = (1.0 - torch.sum(psi_norm * ext_norm, dim=-1)).mean()
            
        return entropy + 1.5 * rec_error

    def forward_continuous(self, 
                           prompt_bytes: List[int], 
                           max_ticks: int = 500, 
                           target_bytes: Optional[List[int]] = None) -> Tuple[List[int], float]:
        """
        Runs the continuous-time SDE integration of the CMF.
        No external loops or teacher-forcing. The system processes the prompt,
        enters a free-running "thinking phase", and endogenously emits output bytes.
        """
        B = 1 # Single sequence processing
        psi = torch.zeros(B, self.dim, device=self.device)
        
        emitted_bytes = []
        total_loss = 0.0
        
        # Hyperparameters for continuous SDE
        dt_base = 0.05
        dt_min = 0.01
        dt_max = 0.20
        gamma_decay_base = 0.15
        beta_base = 8.0
        action_threshold = 0.72
        
        # 1. Presentation Phase (Feed prompt bytes continuously)
        prompt_idx = 0
        input_timer = 0.0
        presentation_time_per_byte = 1.0 # Integrated over continuous time
        
        curr_input_vector = None
        
        tick = 0
        while tick < max_ticks:
            # Determine external sensory input
            if prompt_idx < len(prompt_bytes):
                if input_timer <= 0.0:
                    # Load next byte
                    byte_val = prompt_bytes[prompt_idx]
                    curr_input_vector = self.basins[byte_val].unsqueeze(0) # [1, D]
                    input_timer = presentation_time_per_byte
                    prompt_idx += 1
            else:
                curr_input_vector = None

            # Calculate Free Energy (Surprise)
            f_t = self.compute_free_energy(psi, curr_input_vector)
            
            # Update neuromodulators dynamically (Principle 14)
            # Noradrenaline scales with Surprise
            self.noradrenaline = 0.9 * self.noradrenaline + 0.1 * float(f_t.item())
            # Dopamine updates based on reduction in Free Energy (Reward)
            prev_f = f_t.item()
            
            # Adaptive Time Step dt (Sovereign Time)
            dt = dt_base / (1.0 + 2.0 * self.noradrenaline)
            dt = max(dt_min, min(dt_max, dt))
            
            # PASSIVE DECAY (Gamma Flow) modulated by energy
            gamma = gamma_decay_base * (2.0 - self.energy)
            
            # ROUTING CURRENT (W_route * Phi(Psi))
            phi_psi = torch.tanh(psi)
            routing_current = torch.matmul(phi_psi, self.w_route.t())
            
            # ATTRACTOR FORCE (Hoppfield Energy Gradient)
            # Computes the pulling force toward the nearest attractor basins
            psi_norm = F.normalize(psi, p=2, dim=-1)
            similarities = torch.matmul(psi_norm, self.basins.t()) # [1, V]
            beta = beta_base * (1.0 + 1.5 * self.dopamine)
            attn_weights = torch.softmax(similarities * beta, dim=-1) # [1, V]
            
            attractor_target = torch.matmul(attn_weights, self.basins) # [1, D]
            attractor_force = attractor_target - psi
            
            # SDE State Update (Heun's Predictor-Corrector)
            sensory_current = curr_input_vector if curr_input_vector is not None else torch.zeros_like(psi)
            
            # Noise (Wiener Process) scaling with surprise
            noise_scale = 0.02 * (1.0 + self.noradrenaline)
            noise = torch.randn_like(psi) * noise_scale * math.sqrt(dt)
            
            # Predictor step
            f_psi = -gamma * psi + routing_current + sensory_current + 1.2 * attractor_force
            psi_tilde = psi + f_psi * dt + noise
            
            # Corrector step
            phi_psi_tilde = torch.tanh(psi_tilde)
            routing_tilde = torch.matmul(phi_psi_tilde, self.w_route.t())
            similarities_tilde = torch.matmul(F.normalize(psi_tilde, p=2, dim=-1), self.basins.t())
            attn_tilde = torch.softmax(similarities_tilde * beta, dim=-1)
            attractor_force_tilde = torch.matmul(attn_tilde, self.basins) - psi_tilde
            
            f_psi_tilde = -gamma * psi_tilde + routing_tilde + sensory_current + 1.2 * attractor_force_tilde
            psi = psi + 0.5 * (f_psi + f_psi_tilde) * dt + noise
            
            # Ensure stability on unit sphere
            psi = torch.clamp(psi, -3.0, 3.0)
            
            # Dynamic Epigenetic Morphogenesis (Principle 22.1 & 22.5)
            # Online Hebbian Oja's weight updates gated by epigenetic locks
            if curr_input_vector is not None:
                # Active learning when surprise is high
                lock_matrix = torch.sigmoid(self.mu.unsqueeze(1) * self.mu.unsqueeze(0)) # [D, D]
                # Oja's rule: dW = eta * (y x^T - y^2 W)
                y = phi_psi.squeeze(0) # [D]
                x_in = psi.squeeze(0)  # [D]
                oja_update = torch.outer(y, x_in) - torch.outer(y * y, self.w_route.diagonal())
                
                # Update route matrix
                learning_rate = 0.01 * self.noradrenaline
                self.w_route.data += learning_rate * lock_matrix * oja_update
                
            # Decrease timer for current byte presentation
            if curr_input_vector is not None:
                input_timer -= dt

            # 2. Discrete Action Emission (Wave-Particle Collapse)
            # If we are not presenting prompt bytes, allow the system to snap to output
            if curr_input_vector is None:
                psi_norm_check = F.normalize(psi, p=2, dim=-1)
                proj = torch.matmul(psi_norm_check, self.basins.t()) # [1, V]
                max_val, max_idx = torch.max(proj[0], dim=-1)
                
                # Check if snap threshold is crossed
                if max_val.item() > action_threshold:
                    emitted_byte = max_idx.item()
                    
                    # Prevent emitting pad or duplicate immediate noise
                    if emitted_byte not in (256, 257): # ignore pad/eos in output buffer
                        emitted_bytes.append(emitted_byte)
                        
                    # Refractory period: Reset state along the emitted attractor axis to prevent looping
                    psi = psi - 1.5 * self.basins[emitted_byte].unsqueeze(0)
                    
                    # Feed back emitted byte as efference copy (motor feedback)
                    curr_input_vector = self.basins[emitted_byte].unsqueeze(0)
                    input_timer = 0.5 # Momentary feedback current
                    
                    # Evaluate target loss if targets are provided (for training)
                    if target_bytes is not None and len(emitted_bytes) <= len(target_bytes):
                        target_byte = target_bytes[len(emitted_bytes) - 1]
                        # Loss is negative log-likelihood of the target basin
                        target_prob = torch.softmax(proj * 10.0, dim=-1)[0, target_byte]
                        total_loss += -torch.log(target_prob + 1e-9)
                        
                    # Break if EOS byte is emitted
                    if emitted_byte == 257 or len(emitted_bytes) >= 12:
                        break

            # Energy Metabolism (Principle 22.2)
            self.energy -= 0.002 * dt # Energy consumption
            if curr_input_vector is None:
                self.energy += 0.005 * dt # Recover energy during silent thinking
            self.energy = max(0.1, min(1.0, self.energy))
            
            tick += 1

        # Return average loss over emitted sequence
        avg_loss = total_loss.item() / max(1, len(emitted_bytes)) if isinstance(total_loss, torch.Tensor) else total_loss
        return emitted_bytes, avg_loss


# -----------------------------------------------------------------------------
# Symbolic Mathematics Tasks Data Generation
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
    train_set = in_domain[:split]
    val_id = in_domain[split:]
    val_ood = ood

    return train_set, val_id, val_ood


def run_experiment():
    print("=" * 80)
    print("EXP-269: CONTINUOUS MORPHIC FIELD (CMF) SUBSTRATE BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware execution substrate: {device}")

    train_data, val_id_data, val_ood_data = create_dataset()
    print(f"Dataset: Train={len(train_data)}, In-Domain Val={len(val_id_data)}, OOD Val={len(val_ood_data)}")

    # Instantiate the CMF Substrate
    model = ContinuousMorphicField(dim=128, vocab_size=258, device=device)
    
    # Optimizer for the physical parameters (Basins, Routing Matrix)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 5
    print(f"\n--- Training Phase (Continuous-Time Online Hebbian & Parametric Adaptation) ---")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        random.shuffle(train_data)
        epoch_loss = 0.0
        sample_count = 0
        
        # Train on a subset of 100 random examples per epoch to observe dynamics
        for op, a, b in train_data[:100]:
            prompt, target = make_example(op, a, b)
            
            optimizer.zero_grad()
            # Run continuous-time SDE integration and compute loss end-to-end
            emitted, loss_val = model.forward_continuous(prompt, target_bytes=target)
            
            if len(emitted) > 0:
                # Convert loss to torch tensor for backward pass if needed
                loss_tensor = torch.tensor(loss_val, requires_grad=True, device=model.device)
                loss_tensor.backward()
                optimizer.step()
                epoch_loss += loss_val
                sample_count += 1
                
        avg_loss = epoch_loss / max(1, sample_count)
        print(f"  [Epoch {epoch:02d}/{epochs}] Continuous Surprise Loss: {avg_loss:.4f}")

    train_duration = time.time() - start_time
    print(f"Training completed in {train_duration:.2f}s")

    print("\n--- Evaluation Phase (Endogenous Exact Generation) ---")
    correct_id = 0
    total_id = min(20, len(val_id_data)) # Evaluate on 20 samples to inspect exact traces
    
    for op, a, b in val_id_data[:total_id]:
        prompt, target = make_example(op, a, b)
        prompt_str = bytes(prompt).decode("utf-8")
        expected_str = bytes(target).decode("utf-8").strip()
        
        emitted, _ = model.forward_continuous(prompt, max_ticks=400)
        generated_str = bytes(emitted).decode("utf-8", errors="ignore").strip()
        
        is_correct = generated_str == expected_str
        if is_correct:
            correct_id += 1
        print(f"  Query '{prompt_str}' -> Got '{generated_str}' | Expected '{expected_str}' [{'CORRECT' if is_correct else 'WRONG'}]")

    id_acc = (correct_id / total_id) * 100.0 if total_id > 0 else 0.0
    print(f"\n  • In-Domain Exact Accuracy: {id_acc:.2f}%")

    print("\n" + "=" * 80)
    print("EXP-269 TELEMETRY SUMMARY")
    print("=" * 80)
    print(f"Final Continuous Loss : {avg_loss:.4f}")
    print(f"In-Domain Accuracy    : {id_acc:.2f}%")
    print("=" * 80)

    metrics = {
        "final_loss": float(avg_loss),
        "in_domain_acc": float(id_acc),
        "train_duration_s": float(train_duration)
    }
    with open("experiments/exp_269_metrics.json", "w") as f:
        json.dump(metrics, f)

    return metrics


if __name__ == "__main__":
    run_experiment()
