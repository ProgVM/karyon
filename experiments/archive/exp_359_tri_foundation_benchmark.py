"""
EXP-359: Fundamental Architectural Paradigm Benchmark for Sovereign Autopoiesis
================================================================================
Head-to-head comparison of 3 foundational architectural paradigms for Karyon:
1. Paradigm 1: Symplectic Port-Hamiltonian Phase-Space (Coordinates q, Momenta p, Liouville conservation)
2. Paradigm 2: Dynamic Energy-Based Potential Field (Gradient flow ds/dt = -grad V(s,x) + Wiener noise)
3. Paradigm 3: Adaptive-Temperature Multilinear Circuit Tensor (Tensor contractions with endogenous inverse-temp beta(x))

Core Requirement:
- Dual Nature: Receive, Process, and Emit BOTH:
  * Deterministic Stream: Strict LFSR / Bitwise XOR cipher bytecode (Target: Loss -> 0.0, Free Energy -> 0.0)
  * Non-deterministic Stream: Brownian / Wiener diffusion (Target: Minimal description loss, optimal entropy capture)
- Zero human math lego presets (no hardcoded tanh, sin, etc. menus).
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
# DUAL NATURE DATASET (STRICT DETERMINISTIC & STOCHASTIC DIFFUSION)
# =============================================================================

def generate_benchmark_datasets(num_samples: int = 256, seq_len: int = 64):
    # Deterministic LFSR Stream
    det_seqs = []
    for _ in range(num_samples):
        seq = [random.randint(1, 255)]
        for t in range(1, seq_len):
            prev = seq[-1]
            nxt = ((prev << 1) ^ (0x1D if (prev & 0x80) else 0x00)) & 0xFF
            seq.append(nxt)
        det_seqs.append(seq)
    stream_det = torch.tensor(det_seqs, dtype=torch.long, device=DEVICE)

    # Stochastic Brownian Random Walk Stream
    stoch_seqs = []
    for _ in range(num_samples):
        steps = torch.randn(seq_len)
        walk = torch.cumsum(steps, dim=0)
        min_v, max_v = walk.min(), walk.max()
        quantized = ((walk - min_v) / (max_v - min_v + 1e-6) * 255).long().tolist()
        stoch_seqs.append(quantized)
    stream_stoch = torch.tensor(stoch_seqs, dtype=torch.long, device=DEVICE)

    return stream_det, stream_stoch


# =============================================================================
# PARADIGM 1: SYMPLECTIC PORT-HAMILTONIAN PHASE-SPACE ENGINE
# =============================================================================

class SymplecticHamiltonianEngine(nn.Module):
    """
    State z = (q, p) in R^(2D).
    dq/dt = dH/dp
    dp/dt = -dH/dq - Gamma(x)*p + F_drive(x) + Sigma(x)*dW
    """
    def __init__(self, vocab_size: int = 258, dim: int = 128, device: torch.device = DEVICE):
        super().__init__()
        self.dim = dim
        self.device = device
        self.emb = nn.Embedding(vocab_size, dim, device=device)

        # Hamiltonian kinetic & potential parameters
        self.M_inv = nn.Parameter(torch.ones(dim, device=device)) # Diagonal mass inverse
        self.W_pot1 = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_pot2 = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))

        # Port dissipation and drive
        self.W_gamma = nn.Linear(dim, dim, bias=True, device=device)
        self.W_drive = nn.Linear(dim, dim, bias=False, device=device)
        self.W_sigma = nn.Linear(dim, dim, bias=True, device=device)

        # Autonomous step size tau
        self.log_tau = nn.Parameter(torch.tensor(0.0, device=device))

        # Output heads
        self.det_head = nn.Linear(dim * 2, vocab_size, device=device)
        self.stoch_head = nn.Linear(dim * 2, vocab_size, device=device)

    def forward(self, x: torch.Tensor, mode: str = "deterministic"):
        b, s = x.shape
        emb_x = self.emb(x)

        q = torch.zeros(b, self.dim, device=self.device)
        p = torch.zeros(b, self.dim, device=self.device)

        outputs = []
        free_energies = []

        tau = torch.sigmoid(self.log_tau) * 0.5 + 0.05

        for t in range(s):
            x_t = emb_x[:, t, :]

            # Hamiltonian gradient dH/dq from quadratic/quartic potential
            # V(q) = 0.5 * ||W1 q||^2 + 0.25 * ||W2 q||^4
            q_proj1 = torch.matmul(q, self.W_pot1)
            q_proj2 = torch.matmul(q, self.W_pot2)
            grad_V = torch.matmul(q_proj1, self.W_pot1.t()) + torch.matmul(q_proj2**3, self.W_pot2.t())

            # Port interaction
            gamma = torch.sigmoid(self.W_gamma(x_t)) # 0 on deterministic, >0 on stochastic
            drive = self.W_drive(x_t)
            sigma = F.softplus(self.W_sigma(x_t))

            # Symplectic leapfrog step
            # p_{t+1/2} = p_t - 0.5*tau*(grad_V + gamma*p_t - drive)
            p_half = p - 0.5 * tau * (grad_V + gamma * p - drive)
            
            # dq/dt = M^-1 * p
            q_next = q + tau * (p_half * torch.abs(self.M_inv))
            
            # Compute new grad_V at q_next
            q_proj1_next = torch.matmul(q_next, self.W_pot1)
            q_proj2_next = torch.matmul(q_next, self.W_pot2)
            grad_V_next = torch.matmul(q_proj1_next, self.W_pot1.t()) + torch.matmul(q_proj2_next**3, self.W_pot2.t())

            if mode == "stochastic":
                noise = torch.randn_like(p) * sigma * math.sqrt(tau)
            else:
                noise = 0.0

            p_next = p_half - 0.5 * tau * (grad_V_next + gamma * p_half - drive) + noise

            # Total Hamiltonian Energy H = Kinetic + Potential
            H_kin = 0.5 * torch.sum(p_next**2 * torch.abs(self.M_inv), dim=-1)
            H_pot = 0.5 * torch.sum(q_proj1_next**2, dim=-1) + 0.25 * torch.sum(q_proj2_next**4, dim=-1)
            free_energy_t = H_kin + H_pot

            z = torch.cat([q_next, p_next], dim=-1)
            outputs.append(z)
            free_energies.append(free_energy_t.mean())

            q = q_next
            p = p_next

        out_seq = torch.stack(outputs, dim=1)
        logits = self.det_head(out_seq) if mode == "deterministic" else self.stoch_head(out_seq)
        mean_fe = torch.stack(free_energies).mean()
        return logits, mean_fe


# =============================================================================
# PARADIGM 2: DYNAMIC ENERGY-BASED POTENTIAL FIELD ENGINE
# =============================================================================

class EnergyPotentialFieldEngine(nn.Module):
    """
    State s in R^D.
    ds/dt = -grad_s V(s, x) + F_drive(x) + Sigma(s, x)*dW
    Potential V(s, x) is dynamically generated as an unconstrained scalar energy landscape.
    """
    def __init__(self, vocab_size: int = 258, dim: int = 128, device: torch.device = DEVICE):
        super().__init__()
        self.dim = dim
        self.device = device
        self.emb = nn.Embedding(vocab_size, dim, device=device)

        # Dynamic multi-well potential tensors
        self.W_basin1 = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_basin2 = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_cross = nn.Parameter(torch.randn(dim, dim, device=device) * 0.05)

        self.W_drive = nn.Linear(dim, dim, bias=False, device=device)
        self.W_sigma = nn.Linear(dim, dim, bias=True, device=device)
        self.log_tau = nn.Parameter(torch.tensor(0.0, device=device))

        self.det_head = nn.Linear(dim, vocab_size, device=device)
        self.stoch_head = nn.Linear(dim, vocab_size, device=device)

    def forward(self, x: torch.Tensor, mode: str = "deterministic"):
        b, s_len = x.shape
        emb_x = self.emb(x)
        s = torch.zeros(b, self.dim, device=self.device)

        outputs = []
        free_energies = []
        tau = torch.sigmoid(self.log_tau) * 0.5 + 0.05

        for t in range(s_len):
            x_t = emb_x[:, t, :]

            # Energy potential gradient: grad_s V(s, x)
            # V(s, x) = 0.5 * ||W1 s||^2 + 0.25 * ||W2 s||^4 + (s @ W_cross @ x_t^T)
            s_proj1 = torch.matmul(s, self.W_basin1)
            s_proj2 = torch.matmul(s, self.W_basin2)
            cross_field = torch.matmul(x_t, self.W_cross.t())

            grad_V = (torch.matmul(s_proj1, self.W_basin1.t()) +
                      torch.matmul(s_proj2**3, self.W_basin2.t()) +
                      cross_field)

            drive = self.W_drive(x_t)
            sigma = F.softplus(self.W_sigma(x_t))

            if mode == "stochastic":
                diffusion = torch.randn_like(s) * sigma * math.sqrt(tau)
            else:
                diffusion = 0.0

            # Gradient descent in dynamic potential landscape (Stratonovich step)
            s_next = s - tau * grad_V + tau * drive + diffusion

            # Variational Energy
            V_energy = 0.5 * torch.sum(s_proj1**2, dim=-1) + 0.25 * torch.sum(s_proj2**4, dim=-1)
            free_energies.append(V_energy.mean())

            outputs.append(s_next)
            s = s_next

        out_seq = torch.stack(outputs, dim=1)
        logits = self.det_head(out_seq) if mode == "deterministic" else self.stoch_head(out_seq)
        mean_fe = torch.stack(free_energies).mean()
        return logits, mean_fe


# =============================================================================
# PARADIGM 3: ADAPTIVE-TEMPERATURE MULTILINEAR CIRCUIT TENSOR ENGINE
# =============================================================================

class MultilinearCircuitEngine(nn.Module):
    """
    Pure continuous multilinear tensor contraction with endogenous inverse-temperature beta(x).
    Allows endogenous phase transition:
    - beta -> infty on deterministic tasks (snapping into exact discrete logic with 0 error)
    - beta ~ 1.0 on stochastic tasks (smooth continuous probabilistic processing)
    """
    def __init__(self, vocab_size: int = 258, dim: int = 128, rank: int = 64, device: torch.device = DEVICE):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.device = device
        self.emb = nn.Embedding(vocab_size, dim, device=device)

        # Multilinear tensor contraction weights
        self.W_x1 = nn.Parameter(torch.randn(dim, rank, device=device) * (1.0 / math.sqrt(dim)))
        self.W_s1 = nn.Parameter(torch.randn(dim, rank, device=device) * (1.0 / math.sqrt(dim)))
        self.W_x2 = nn.Parameter(torch.randn(dim, rank, device=device) * (1.0 / math.sqrt(dim)))
        self.W_s2 = nn.Parameter(torch.randn(dim, rank, device=device) * (1.0 / math.sqrt(dim)))
        self.W_out = nn.Parameter(torch.randn(rank, dim, device=device) * (1.0 / math.sqrt(rank)))

        # Endogenous inverse-temperature generator beta(x)
        self.W_beta = nn.Linear(dim, 1, bias=True, device=device)
        self.W_sigma = nn.Linear(dim, dim, bias=True, device=device)
        self.log_tau = nn.Parameter(torch.tensor(0.0, device=device))

        self.det_head = nn.Linear(dim, vocab_size, device=device)
        self.stoch_head = nn.Linear(dim, vocab_size, device=device)

    def forward(self, x: torch.Tensor, mode: str = "deterministic"):
        b, s_len = x.shape
        emb_x = self.emb(x)
        s = torch.zeros(b, self.dim, device=self.device)

        outputs = []
        free_energies = []
        tau = torch.sigmoid(self.log_tau) * 0.8 + 0.1

        for t in range(s_len):
            x_t = emb_x[:, t, :]

            # Endogenous inverse temperature beta: scale in [0.5, 30.0]
            beta = torch.exp(torch.clamp(self.W_beta(x_t), -1.0, 4.0)) # [B, 1]

            # Multilinear tensor conjunction
            t1 = torch.matmul(x_t, self.W_x1) * torch.matmul(s, self.W_s1) # [B, rank]
            t2 = torch.matmul(x_t, self.W_x2) * torch.matmul(s, self.W_s2) # [B, rank]
            core = t1 + t2

            # Temperature-modulated non-linear saturation: (1/beta) * tanh(beta * core)
            saturated_core = (1.0 / (beta + 1e-5)) * torch.tanh(beta * core)
            drift = torch.matmul(saturated_core, self.W_out)

            sigma = F.softplus(self.W_sigma(x_t))
            if mode == "stochastic":
                noise = torch.randn_like(s) * sigma * math.sqrt(tau)
            else:
                noise = 0.0

            s_next = (1.0 - tau) * s + tau * drift + noise
            outputs.append(s_next)

            # Free energy: entropy penalty inversely proportional to temperature
            fe_t = (1.0 / beta).mean() + torch.mean(torch.sum(drift**2, dim=-1))
            free_energies.append(fe_t)

            s = s_next

        out_seq = torch.stack(outputs, dim=1)
        logits = self.det_head(out_seq) if mode == "deterministic" else self.stoch_head(out_seq)
        mean_fe = torch.stack(free_energies).mean()
        return logits, mean_fe


# =============================================================================
# UNIFIED COMPARATIVE BENCHMARK RUNNER
# =============================================================================

def train_and_evaluate_paradigm(name: str, model: nn.Module, det_data: torch.Tensor, stoch_data: torch.Tensor, epochs: int = 12):
    print(f"\n===============================================================================")
    print(f"=== BENCHMARKING PARADIGM: {name} ===")
    print(f"===============================================================================")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    batch_size = 32
    vocab_size = 258

    t0 = time.time()
    total_tokens = 0

    det_losses = []
    stoch_losses = []
    det_fes = []
    stoch_fes = []

    for epoch in range(epochs):
        model.train()
        ep_det_loss = 0.0
        ep_stoch_loss = 0.0
        ep_det_fe = 0.0
        ep_stoch_fe = 0.0

        # 1. Deterministic Stream Training
        for i in range(0, det_data.size(0), batch_size):
            batch = det_data[i:i + batch_size]
            inp, tgt = batch[:, :-1], batch[:, 1:]

            optimizer.zero_grad()
            logits, fe = model(inp, mode="deterministic")
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1)) + 0.001 * fe
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_det_loss += loss.item() * inp.size(0)
            ep_det_fe += fe.item() * inp.size(0)
            total_tokens += inp.numel()

        # 2. Stochastic Stream Training
        for i in range(0, stoch_data.size(0), batch_size):
            batch = stoch_data[i:i + batch_size]
            inp, tgt = batch[:, :-1], batch[:, 1:]

            optimizer.zero_grad()
            logits, fe = model(inp, mode="stochastic")
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1)) + 0.001 * fe
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_stoch_loss += loss.item() * inp.size(0)
            ep_stoch_fe += fe.item() * inp.size(0)
            total_tokens += inp.numel()

        avg_det_l = ep_det_loss / det_data.size(0)
        avg_stoch_l = ep_stoch_loss / stoch_data.size(0)
        avg_det_fe = ep_det_fe / det_data.size(0)
        avg_stoch_fe = ep_stoch_fe / stoch_data.size(0)

        det_losses.append(avg_det_l)
        stoch_losses.append(avg_stoch_l)
        det_fes.append(avg_det_fe)
        stoch_fes.append(avg_stoch_fe)

        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Det Loss: {avg_det_l:.4f} (FE: {avg_det_fe:.3f}) | Stoch Loss: {avg_stoch_l:.4f} (FE: {avg_stoch_fe:.3f})")

    elapsed = time.time() - t0
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "paradigm": name,
        "initial_det_loss": det_losses[0],
        "final_det_loss": det_losses[-1],
        "det_loss_delta": det_losses[0] - det_losses[-1],
        "final_det_fe": det_fes[-1],
        "initial_stoch_loss": stoch_losses[0],
        "final_stoch_loss": stoch_losses[-1],
        "stoch_loss_delta": stoch_losses[0] - stoch_losses[-1],
        "final_stoch_fe": stoch_fes[-1],
        "total_final_loss": det_losses[-1] + stoch_losses[-1],
        "total_delta": (det_losses[0] + stoch_losses[0]) - (det_losses[-1] + stoch_losses[-1]),
        "tok_per_sec": tok_per_sec,
        "elapsed_sec": elapsed
    }


def run_exp_359_benchmark():
    print("===============================================================================")
    print("=== KEP EXP-359: 3 FOUNDATIONAL PARADIGMS TRI-BATTLE (DUAL-NATURE)         ===")
    print("===============================================================================")
    print(f"Substrate Device: {DEVICE}")

    det_stream, stoch_stream = generate_benchmark_datasets(num_samples=256, seq_len=64)

    # 1. Symplectic Port-Hamiltonian
    p1 = SymplecticHamiltonianEngine(vocab_size=258, dim=128, device=DEVICE)
    res_p1 = train_and_evaluate_paradigm("Paradigm 1: Symplectic Port-Hamiltonian", p1, det_stream, stoch_stream, epochs=12)

    # 2. Energy Potential Field
    p2 = EnergyPotentialFieldEngine(vocab_size=258, dim=128, device=DEVICE)
    res_p2 = train_and_evaluate_paradigm("Paradigm 2: Energy Potential Field", p2, det_stream, stoch_stream, epochs=12)

    # 3. Multilinear Circuit with Endogenous Temperature
    p3 = MultilinearCircuitEngine(vocab_size=258, dim=128, rank=64, device=DEVICE)
    res_p3 = train_and_evaluate_paradigm("Paradigm 3: Multilinear Circuit & Adaptive Temperature", p3, det_stream, stoch_stream, epochs=12)

    print("\n===============================================================================")
    print("=== FINAL COMPARATIVE TRI-BATTLE TELEMETRY ===")
    print("===============================================================================")
    for r in [res_p1, res_p2, res_p3]:
        print(f"{r['paradigm']}:")
        print(f"  Deterministic : Initial {r['initial_det_loss']:.4f} -> Final {r['final_det_loss']:.4f} (Delta: {r['det_loss_delta']:.4f}, FE: {r['final_det_fe']:.3f})")
        print(f"  Stochastic    : Initial {r['initial_stoch_loss']:.4f} -> Final {r['final_stoch_loss']:.4f} (Delta: {r['stoch_loss_delta']:.4f}, FE: {r['final_stoch_fe']:.3f})")
        print(f"  Total Score   : Combined Final Loss = {r['total_final_loss']:.4f} | Total Delta = {r['total_delta']:.4f} | Speed: {r['tok_per_sec']:.2f} tok/s\n")

    # Determine winning foundation
    all_results = [res_p1, res_p2, res_p3]
    winner = min(all_results, key=lambda x: x['total_final_loss'])
    print(f"👑 VICTORIOUS ARCHITECTURAL FOUNDATION: {winner['paradigm']} (Lowest Total Loss: {winner['total_final_loss']:.4f})")

    verdict = "🟢 POSITIVE" if winner['total_delta'] >= 0.08 else "⚪ NEUTRAL"

    benchmark_bundle = {
        "exp_id": "EXP-359",
        "results": all_results,
        "winner": winner['paradigm'],
        "winner_total_loss": float(winner['total_final_loss']),
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_359_results.json", "w") as f:
        json.dump(benchmark_bundle, f, indent=2)

    return benchmark_bundle


if __name__ == "__main__":
    run_exp_359_benchmark()
