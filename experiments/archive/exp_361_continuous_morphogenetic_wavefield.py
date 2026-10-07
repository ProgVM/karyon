"""
EXP-361: Continuous Sovereign Morphogenesis & Symplectic Wavefield Evolution
=============================================================================
Refined based on EXP-360 Diagnostic Insights & Bazilevs' Mandate:
1. True Evolution in Continuous Stream:
   - Evaluates 3 distinct evolutionary mechanisms with strict numerical stability:
     * Evolution Paradigm 1: Symplectic Morphogenetic Topology (Nodes sprout into coupled Hamiltonian (q, p) pairs with energy conservation).
     * Evolution Paradigm 2: Autopoietic Operator Synthesis (Continuous field operator synthesizes new differential equations without preset functions).
     * Evolution Paradigm 3: Dynamic State-Space Dimensionality Blooming (Symplectic subspace expands D -> D + dD with bounded orthogonal projection).
2. Continuous Single-Pass Temporal Stream (N=1):
   - Unbroken streaming reality t -> t+1.
   - Dual nature: Interleaved strict deterministic LFSR / algorithmic machine bytecode and stochastic Brownian noise.
3. Phase-Lock Resonance (Energy-minimizing decision snapping instead of softmax roulette).
=============================================================================
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
# CONTINUOUS DUAL-NATURE STREAM GENERATOR
# =============================================================================

def generate_continuous_stream(stream_length: int = 2048):
    stream_bytes = []
    stream_types = []  # 0: det, 1: stoch

    current_lfsr = 0x5A
    current_brownian = 0.0

    mode = 0
    block_remaining = 64

    for _ in range(stream_length):
        if block_remaining <= 0:
            mode = 1 - mode
            block_remaining = random.randint(32, 64)

        if mode == 0:
            current_lfsr = ((current_lfsr << 1) ^ (0x1D if (current_lfsr & 0x80) else 0x00)) & 0xFF
            val = current_lfsr
            stream_types.append(0)
        else:
            current_brownian += random.gauss(0.0, 1.0)
            val = int((math.sin(current_brownian * 0.1) * 0.5 + 0.5) * 255)
            stream_types.append(1)

        stream_bytes.append(val)
        block_remaining -= 1

    stream_tensor = torch.tensor(stream_bytes, dtype=torch.long, device=DEVICE)
    type_tensor = torch.tensor(stream_types, dtype=torch.long, device=DEVICE)
    return stream_tensor, type_tensor


# =============================================================================
# EVOLUTION PARADIGM 1: SYMPLECTIC MORPHOGENETIC TOPOLOGY (HAMILTONIAN SPROUTING)
# =============================================================================

class SymplecticSproutedNode(nn.Module):
    """Coupled (q, p) phase-space node with autonomous continuous time tau."""
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.W_pot = nn.Parameter(torch.randn(dim, dim, device=device) * (0.5 / math.sqrt(dim)))
        self.W_in = nn.Parameter(torch.randn(dim, dim, device=device) * (0.5 / math.sqrt(dim)))
        self.W_gamma = nn.Linear(dim, dim, bias=True, device=device)
        self.log_tau = nn.Parameter(torch.tensor(0.0, device=device))

    def step(self, x: torch.Tensor, q_prev: torch.Tensor, p_prev: torch.Tensor):
        tau = torch.sigmoid(self.log_tau) * 0.4 + 0.05
        drive = torch.matmul(x, self.W_in)

        # Hamiltonian potential gradient: grad V = W_pot^T @ W_pot @ q
        q_proj = torch.matmul(q_prev, self.W_pot)
        grad_V = torch.matmul(q_proj, self.W_pot.t())

        gamma = torch.sigmoid(self.W_gamma(x))

        # Symplectic Leapfrog Integration with Energy Bound
        p_half = p_prev - 0.5 * tau * (grad_V + gamma * p_prev - drive)
        q_next = q_prev + tau * torch.tanh(p_half) # Bounded velocity avoids explosion
        
        q_proj_next = torch.matmul(q_next, self.W_pot)
        grad_V_next = torch.matmul(q_proj_next, self.W_pot.t())
        p_next = p_half - 0.5 * tau * (grad_V_next + gamma * p_half - drive)
        p_next = torch.clamp(p_next, -5.0, 5.0)

        energy = 0.5 * torch.sum(p_next**2, dim=-1) + 0.5 * torch.sum(q_proj_next**2, dim=-1)
        return q_next, p_next, energy.mean()


class Paradigm1SymplecticMorphogenesis(nn.Module):
    def __init__(self, vocab_size: int = 258, dim: int = 64, max_nodes: int = 8, device: torch.device = DEVICE):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_nodes = max_nodes
        self.device = device

        self.emb = nn.Embedding(vocab_size, dim, device=device)
        self.nodes = nn.ModuleList([SymplecticSproutedNode(dim, device)])
        self.alpha_epi = nn.ParameterList([nn.Parameter(torch.tensor(2.0, device=device))])
        self.vitality = [1.0]

        self.readout_basis = nn.Parameter(torch.randn(vocab_size, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.sprout_count = 0
        self.prune_count = 0

    def sprout_node(self):
        if len(self.nodes) >= self.max_nodes:
            return False
        node = SymplecticSproutedNode(self.dim, self.device)
        self.nodes.append(node)
        self.alpha_epi.append(nn.Parameter(torch.tensor(0.01, device=self.device)))
        self.vitality.append(1.0)
        self.sprout_count += 1
        return True

    def prune_stale_nodes(self):
        if len(self.nodes) <= 1:
            return False
        min_idx = int(np.argmin(self.vitality))
        if self.vitality[min_idx] < 0.05 and min_idx > 0:
            del self.nodes[min_idx]
            del self.alpha_epi[min_idx]
            del self.vitality[min_idx]
            self.prune_count += 1
            return True
        return False

    def forward_step(self, byte_in: torch.Tensor, q_states: list, p_states: list):
        x = self.emb(byte_in)
        next_q = []
        next_p = []
        combined_rep = torch.zeros_like(x)
        total_energy = torch.tensor(0.0, device=self.device)

        for i, node in enumerate(self.nodes):
            q_i, p_i, e_i = node.step(x, q_states[i], p_states[i])
            gate = torch.tanh(self.alpha_epi[i])
            combined_rep = combined_rep + gate * q_i
            total_energy = total_energy + e_i
            next_q.append(q_i)
            next_p.append(p_i)
            self.vitality[i] = 0.98 * self.vitality[i] + 0.02 * float(q_i.abs().mean())

        rep_norm = F.normalize(combined_rep, dim=-1)
        basis_norm = F.normalize(self.readout_basis, dim=-1)
        resonance = torch.matmul(rep_norm, basis_norm.t()) * 16.0

        return resonance, next_q, next_p, total_energy


# =============================================================================
# EVOLUTION PARADIGM 2: AUTOPOIETIC CONTINUOUS OPERATOR SYNTHESIS
# =============================================================================

class AutopoieticContinuousSynthesizerNode(nn.Module):
    """Synthesizes dynamic multilinear differential vector fields from scratch."""
    def __init__(self, dim: int, rank: int, device: torch.device):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.device = device
        self.W1 = nn.Parameter(torch.randn(dim, rank, device=device) * (0.5 / math.sqrt(dim)))
        self.W2 = nn.Parameter(torch.randn(dim, rank, device=device) * (0.5 / math.sqrt(dim)))
        self.W_out = nn.Parameter(torch.randn(rank, dim, device=device) * (0.5 / math.sqrt(rank)))
        self.M_curv = nn.Parameter(torch.randn(dim, dim, device=device) * 0.02)
        self.log_tau = nn.Parameter(torch.tensor(0.0, device=device))

    def step(self, x: torch.Tensor, s_prev: torch.Tensor):
        tau = torch.sigmoid(self.log_tau) * 0.5 + 0.05
        # Continuous multilinear conjunction: (x @ W1) * (s @ W2)
        u1 = torch.matmul(x, self.W1)
        u2 = torch.matmul(s_prev, self.W2)
        core = torch.tanh(u1 * u2) # Bounded conjunction
        drift = torch.matmul(core, self.W_out)

        metric = torch.sigmoid(torch.sum(x * torch.matmul(s_prev, self.M_curv), dim=-1, keepdim=True))
        s_next = (1.0 - tau) * s_prev + tau * metric * drift

        energy = 0.5 * torch.sum((s_next - s_prev)**2, dim=-1)
        return s_next, energy.mean()


class Paradigm2AutopoieticSynthesizer(nn.Module):
    def __init__(self, vocab_size: int = 258, dim: int = 64, max_ops: int = 8, device: torch.device = DEVICE):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_ops = max_ops
        self.device = device

        self.emb = nn.Embedding(vocab_size, dim, device=device)
        self.ops = nn.ModuleList([AutopoieticContinuousSynthesizerNode(dim, rank=32, device=device)])
        self.alpha_epi = nn.ParameterList([nn.Parameter(torch.tensor(2.0, device=device))])
        self.vitality = [1.0]

        self.readout_basis = nn.Parameter(torch.randn(vocab_size, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.sprout_count = 0
        self.prune_count = 0

    def sprout_operator(self):
        if len(self.ops) >= self.max_ops:
            return False
        rank = random.choice([16, 32, 48])
        op = AutopoieticContinuousSynthesizerNode(self.dim, rank=rank, device=self.device)
        self.ops.append(op)
        self.alpha_epi.append(nn.Parameter(torch.tensor(0.01, device=self.device)))
        self.vitality.append(1.0)
        self.sprout_count += 1
        return True

    def prune_stale_operators(self):
        if len(self.ops) <= 1:
            return False
        min_idx = int(np.argmin(self.vitality))
        if self.vitality[min_idx] < 0.05 and min_idx > 0:
            del self.ops[min_idx]
            del self.alpha_epi[min_idx]
            del self.vitality[min_idx]
            self.prune_count += 1
            return True
        return False

    def forward_step(self, byte_in: torch.Tensor, states: list):
        x = self.emb(byte_in)
        next_states = []
        combined_rep = torch.zeros_like(x)
        total_energy = torch.tensor(0.0, device=self.device)

        for i, op in enumerate(self.ops):
            s_i, e_i = op.step(x, states[i])
            gate = torch.tanh(self.alpha_epi[i])
            combined_rep = combined_rep + gate * s_i
            total_energy = total_energy + e_i
            next_states.append(s_i)
            self.vitality[i] = 0.98 * self.vitality[i] + 0.02 * float(s_i.abs().mean())

        rep_norm = F.normalize(combined_rep, dim=-1)
        basis_norm = F.normalize(self.readout_basis, dim=-1)
        resonance = torch.matmul(rep_norm, basis_norm.t()) * 16.0

        return resonance, next_states, total_energy


# =============================================================================
# EVOLUTION PARADIGM 3: BOUNDED DIMENSIONALITY BLOOMING
# =============================================================================

class Paradigm3DimensionalBlooming(nn.Module):
    def __init__(self, vocab_size: int = 258, initial_dim: int = 32, max_dim: int = 128, device: torch.device = DEVICE):
        super().__init__()
        self.vocab_size = vocab_size
        self.current_dim = initial_dim
        self.max_dim = max_dim
        self.device = device

        self.emb_weight = nn.Parameter(torch.randn(vocab_size, max_dim, device=device) * 0.1)
        self.W_hamiltonian = nn.Parameter(torch.randn(max_dim, max_dim, device=device) * (0.5 / math.sqrt(max_dim)))
        self.readout_matrix = nn.Parameter(torch.randn(vocab_size, max_dim, device=device) * 0.1)
        self.sprout_count = 0

    def bloom_dimension(self, delta_d: int = 16):
        if self.current_dim + delta_d <= self.max_dim:
            self.current_dim += delta_d
            self.sprout_count += 1
            return True
        return False

    def forward_step(self, byte_in: torch.Tensor, q_prev: torch.Tensor, p_prev: torch.Tensor):
        d = self.current_dim
        x_full = self.emb_weight[byte_in]
        x = x_full[:, :d]

        q = q_prev[:, :d]
        p = p_prev[:, :d]

        H_sub = self.W_hamiltonian[:d, :d]
        grad_V = torch.matmul(q, H_sub) + x

        tau = 0.2
        p_half = p - 0.5 * tau * grad_V
        q_next = q + tau * torch.tanh(p_half)
        grad_V_next = torch.matmul(q_next, H_sub) + x
        p_next = torch.clamp(p_half - 0.5 * tau * grad_V_next, -5.0, 5.0)

        H_energy = 0.5 * torch.sum(p_next**2, dim=-1) + 0.5 * torch.sum(q_next**2, dim=-1)

        q_norm = F.normalize(q_next, dim=-1)
        basis_norm = F.normalize(self.readout_matrix[:, :d], dim=-1)
        resonance = torch.matmul(q_norm, basis_norm.t()) * 16.0

        q_out = torch.zeros_like(q_prev)
        p_out = torch.zeros_like(p_prev)
        q_out[:, :d] = q_next
        p_out[:, :d] = p_next

        return resonance, q_out, p_out, H_energy.mean()


# =============================================================================
# CONTINUOUS STREAM BENCHMARK RUNNER
# =============================================================================

def benchmark_stream_engine(name: str, model: nn.Module, p_type: str, stream_data: torch.Tensor, stream_types: torch.Tensor):
    print(f"\n===============================================================================")
    print(f"=== EVALUATING EVOLUTIONARY STREAM ENGINE: {name} ===")
    print(f"===============================================================================")

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-5)
    t0 = time.time()
    stream_len = stream_data.size(0)

    det_errors = []
    stoch_errors = []
    det_energies = []
    stoch_energies = []

    if p_type == "P1":
        q_states = [torch.zeros(1, model.dim, device=DEVICE)]
        p_states = [torch.zeros(1, model.dim, device=DEVICE)]
    elif p_type == "P2":
        s_states = [torch.zeros(1, model.dim, device=DEVICE)]
    elif p_type == "P3":
        q_state = torch.zeros(1, model.max_dim, device=DEVICE)
        p_state = torch.zeros(1, model.max_dim, device=DEVICE)

    rolling_loss = 0.0

    for t in range(stream_len - 1):
        cur_byte = stream_data[t:t+1]
        next_byte = stream_data[t+1:t+2]
        is_stochastic = stream_types[t].item()

        optimizer.zero_grad()

        if p_type == "P1":
            resonance, q_states, p_states, energy = model.forward_step(cur_byte, q_states, p_states)
            q_states = [q.detach() for q in q_states]
            p_states = [p.detach() for p in p_states]
        elif p_type == "P2":
            resonance, s_states, energy = model.forward_step(cur_byte, s_states)
            s_states = [s.detach() for s in s_states]
        elif p_type == "P3":
            resonance, q_state, p_state, energy = model.forward_step(cur_byte, q_state, p_state)
            q_state = q_state.detach()
            p_state = p_state.detach()

        loss = F.cross_entropy(resonance, next_byte) + 0.001 * energy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        rolling_loss = 0.9 * rolling_loss + 0.1 * loss.item() if t > 0 else loss.item()

        # Evolution during continuous stream
        if t > 50 and t % 64 == 0:
            if rolling_loss > 1.2:
                if p_type == "P1":
                    if model.sprout_node():
                        q_states.append(torch.zeros(1, model.dim, device=DEVICE))
                        p_states.append(torch.zeros(1, model.dim, device=DEVICE))
                        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-5)
                elif p_type == "P2":
                    if model.sprout_operator():
                        s_states.append(torch.zeros(1, model.dim, device=DEVICE))
                        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-5)
                elif p_type == "P3":
                    if model.bloom_dimension(16):
                        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-5)

            if p_type in ["P1", "P2"]:
                if p_type == "P1":
                    model.prune_stale_nodes()
                else:
                    model.prune_stale_operators()

        pred_byte = torch.argmax(resonance, dim=-1)
        is_error = 1.0 if pred_byte.item() != next_byte.item() else 0.0

        if is_stochastic == 0:
            det_errors.append(is_error)
            det_energies.append(energy.item())
        else:
            stoch_errors.append(is_error)
            stoch_energies.append(energy.item())

        if (t + 1) % 256 == 0:
            det_tail_err = np.mean(det_errors[-64:]) if len(det_errors) >= 64 else np.mean(det_errors)
            print(f"  Step {t+1:04d}/{stream_len} | Loss: {rolling_loss:.4f} | Det Error Rate: {det_tail_err*100:.1f}% | Energy: {energy.item():.4f}")

    elapsed = time.time() - t0
    tok_per_sec = stream_len / elapsed if elapsed > 0 else 0.0

    q_len = max(1, len(det_errors) // 4)
    final_det_error_rate = float(np.mean(det_errors[-q_len:]))
    final_det_energy = float(np.mean(det_energies[-q_len:]))
    final_stoch_error_rate = float(np.mean(stoch_errors[-q_len:]))
    final_stoch_energy = float(np.mean(stoch_energies[-q_len:]))

    sprout_events = getattr(model, 'sprout_count', 0)
    prune_events = getattr(model, 'prune_count', 0)

    print(f"  --> Final Det Error: {final_det_error_rate*100:.2f}% | Final Det Energy: {final_det_energy:.4f}")
    print(f"  --> Sprouts: {sprout_events} | Prunes: {prune_events} | Speed: {tok_per_sec:.1f} tok/s")

    return {
        "paradigm": name,
        "type": p_type,
        "final_det_error_rate": final_det_error_rate,
        "final_det_energy": final_det_energy,
        "final_stoch_error_rate": final_stoch_error_rate,
        "final_stoch_energy": final_stoch_energy,
        "final_rolling_loss": float(rolling_loss),
        "sprout_events": sprout_events,
        "prune_events": prune_events,
        "tok_per_sec": tok_per_sec,
        "elapsed_sec": elapsed
    }


def run_exp_361():
    print("===============================================================================")
    print("=== KEP EXP-361: CONTINUOUS MORPHOGENETIC WAVEFIELD BENCHMARK               ===")
    print("===============================================================================")
    print(f"Substrate Device: {DEVICE}")

    stream_data, stream_types = generate_continuous_stream(stream_length=2048)

    # 1. Symplectic Morphogenetic Topology (Hamiltonian q, p nodes)
    p1 = Paradigm1SymplecticMorphogenesis(vocab_size=258, dim=64, max_nodes=8, device=DEVICE)
    res_p1 = benchmark_stream_engine("Evolution Paradigm 1: Symplectic Morphogenetic Topology", p1, "P1", stream_data, stream_types)

    # 2. Autopoietic Continuous Synthesizer
    p2 = Paradigm2AutopoieticSynthesizer(vocab_size=258, dim=64, max_ops=8, device=DEVICE)
    res_p2 = benchmark_stream_engine("Evolution Paradigm 2: Autopoietic Operator Synthesizer", p2, "P2", stream_data, stream_types)

    # 3. Dynamic Dimensionality Blooming
    p3 = Paradigm3DimensionalBlooming(vocab_size=258, initial_dim=32, max_dim=128, device=DEVICE)
    res_p3 = benchmark_stream_engine("Evolution Paradigm 3: Dynamic Dimensionality Blooming", p3, "P3", stream_data, stream_types)

    print("\n===============================================================================")
    print("=== FINAL COMPARATIVE STREAMING TELEMETRY ===")
    print("===============================================================================")
    all_res = [res_p1, res_p2, res_p3]
    for r in all_res:
        print(f"{r['paradigm']}:")
        print(f"  Deterministic Error Rate : {r['final_det_error_rate']*100:.2f}% (Energy: {r['final_det_energy']:.4f})")
        print(f"  Stochastic Error Rate    : {r['final_stoch_error_rate']*100:.2f}% (Energy: {r['final_stoch_energy']:.4f})")
        print(f"  Morphogenetic Evolution  : Sprouts={r['sprout_events']}, Prunes={r['prune_events']} | Speed={r['tok_per_sec']:.1f} tok/s\n")

    winner = min(all_res, key=lambda x: (x['final_det_error_rate'], x['final_det_energy']))
    print(f"👑 VICTORIOUS EVOLUTIONARY ARCHITECTURE: {winner['paradigm']}")

    verdict = "🟢 POSITIVE" if winner['final_det_error_rate'] < 0.20 else "⚪ NEUTRAL"

    results_data = {
        "exp_id": "EXP-361",
        "results": all_res,
        "winner": winner['paradigm'],
        "winner_det_error_rate": float(winner['final_det_error_rate']),
        "winner_det_energy": float(winner['final_det_energy']),
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_361_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    run_exp_361()
