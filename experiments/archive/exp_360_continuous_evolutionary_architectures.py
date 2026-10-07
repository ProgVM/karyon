"""
EXP-360: Evolutionary Continuous Stream Architectures (Tri-Paradigm Benchmark)
==============================================================================
Mandate from Bazilevs:
1. True Evolution (Morphogenesis in Stream):
   - The architecture is NOT static. It autonomously creates, modifies, and evolves its own structures,
     operators, dimensions, and time scales during learning, without preset Lego blocks or fixed rules.
2. Continuous Stream Operation:
   - Data is an unbroken temporal stream (t -> t+1). No artificial batch shuffles or multi-epoch resets.
   - States persist and evolve continuously through the temporal reality.
3. Dual Nature (Deterministic & Non-deterministic):
   - Receives, processes, and produces BOTH:
     * Deterministic stream: algorithmic bytecode / parity dynamics (Loss -> 0, Energy -> 0, exact phase lock).
     * Non-deterministic stream: continuous Brownian diffusion / stochastic noise (entropy capture).
4. No Artificial Probabilities / Softmax Roulettes:
   - Discrete deterministic decisions are produced via physical phase-lock & attractor energy collapse.
   - Continuous non-deterministic emissions are produced via continuous field displacement.

Evaluated Evolutionary Paradigms:
- Paradigm A: Topological Morphogenetic Graph (sprouts & prunes autonomous continuous tensor nodes in stream)
- Paradigm B: Dynamic Dimensionality Blooming (state space dimension expands D -> D + dD, expanding metric rank)
- Paradigm C: Self-Modifying Continuous Operator Field (field equations L[Psi] self-evolve continuously via strain)
==============================================================================
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
    """
    Generates an interleaved continuous temporal stream:
    - Block 1: Deterministic algorithmic machine bytecode (XOR feedback shift register)
    - Block 2: Stochastic Brownian noise diffusion quantized into continuous byte states
    Interleaved periodically to test seamless real-time adaptation and zero-loss phase locking.
    """
    stream_bytes = []
    stream_types = []  # 0 for deterministic, 1 for stochastic

    current_lfsr = 0x5A
    current_brownian = 0.0

    mode = 0  # 0: det, 1: stoch
    block_remaining = 64

    for _ in range(stream_length):
        if block_remaining <= 0:
            mode = 1 - mode
            block_remaining = random.randint(32, 64)

        if mode == 0:
            # Deterministic algorithmic transition: bitwise rotation & nonlinear feedback
            current_lfsr = ((current_lfsr << 1) ^ (0x1D if (current_lfsr & 0x80) else 0x00)) & 0xFF
            val = current_lfsr
            stream_types.append(0)
        else:
            # Stochastic continuous random walk
            current_brownian += random.gauss(0.0, 1.0)
            # Fold into byte coordinate [0, 255]
            val = int((math.sin(current_brownian * 0.1) * 0.5 + 0.5) * 255)
            stream_types.append(1)

        stream_bytes.append(val)
        block_remaining -= 1

    stream_tensor = torch.tensor(stream_bytes, dtype=torch.long, device=DEVICE)
    type_tensor = torch.tensor(stream_types, dtype=torch.long, device=DEVICE)
    return stream_tensor, type_tensor


# =============================================================================
# PARADIGM A: TOPOLOGICAL MORPHOGENETIC GRAPH (DYNAMIC NODE SPROUTING & PRUNING)
# =============================================================================

class ContinuousTensorNode(nn.Module):
    """Dynamically sprouted continuous state node with autonomous time scale tau."""
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.W_in = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_rec = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_metric = nn.Parameter(torch.randn(dim, dim, device=device) * 0.05)
        self.log_tau = nn.Parameter(torch.tensor(0.0, device=device))

    def step(self, x: torch.Tensor, s_prev: torch.Tensor):
        # Continuous differential step without hardcoded human functions
        tau = torch.sigmoid(self.log_tau) * 0.8 + 0.1
        # Curvature metric modulation
        g = torch.sigmoid(torch.sum(x * torch.matmul(s_prev, self.W_metric), dim=-1, keepdim=True))
        drift = torch.matmul(x, self.W_in) + torch.matmul(s_prev, self.W_rec)
        # Stratonovich differential relaxation
        s_next = (1.0 - tau) * s_prev + tau * g * drift
        # Energy: strain of displacement
        energy = 0.5 * torch.sum((s_next - s_prev)**2, dim=-1)
        return s_next, energy


class ParadigmATopologicalGraph(nn.Module):
    """
    Paradigm A: Starts with 1 node. As strain / prediction error in the continuous
    stream exceeds homeostatic threshold, it autonomously sprouts new nodes and synaptic links.
    Inactive nodes with low vitality undergo apoptosis (pruning).
    """
    def __init__(self, vocab_size: int = 258, dim: int = 64, max_nodes: int = 8, device: torch.device = DEVICE):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_nodes = max_nodes
        self.device = device

        self.emb = nn.Embedding(vocab_size, dim, device=device)
        self.nodes = nn.ModuleList([ContinuousTensorNode(dim, device)])
        self.alpha_epi = nn.ParameterList([nn.Parameter(torch.tensor(2.0, device=device))])
        self.vitality = [1.0]

        # Phase-lock readout (Physical energy minimum projector)
        self.readout_basis = nn.Parameter(torch.randn(vocab_size, dim, device=device) * (1.0 / math.sqrt(dim)))

        self.sprout_count = 0
        self.prune_count = 0

    def sprout_node(self):
        if len(self.nodes) >= self.max_nodes:
            return False
        new_node = ContinuousTensorNode(self.dim, self.device)
        self.nodes.append(new_node)
        # Net2Net smooth birth: alpha_epi initialized near 0 to guarantee zero functional shock at birth
        self.alpha_epi.append(nn.Parameter(torch.tensor(0.01, device=self.device)))
        self.vitality.append(1.0)
        self.sprout_count += 1
        return True

    def prune_stale_nodes(self):
        if len(self.nodes) <= 1:
            return False
        # Find node with minimal vitality
        min_idx = int(np.argmin(self.vitality))
        if self.vitality[min_idx] < 0.05 and min_idx > 0:
            del self.nodes[min_idx]
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

        for i, node in enumerate(self.nodes):
            s_i, e_i = node.step(x, states[i])
            gate = torch.tanh(self.alpha_epi[i])
            combined_rep = combined_rep + gate * s_i
            total_energy = total_energy + e_i.mean()
            next_states.append(s_i)
            # Update vitality
            self.vitality[i] = 0.98 * self.vitality[i] + 0.02 * float(s_i.abs().mean())

        # Physical Phase-Lock Attractor Projection:
        # Instead of arbitrary softmax probabilities, finding distance to basis attractors
        # d_k = ||combined_rep - basis_k||^2. The lowest energy state snaps to 0.
        rep_norm = F.normalize(combined_rep, dim=-1)
        basis_norm = F.normalize(self.readout_basis, dim=-1)
        # Cosine affinity field (sharp physical phase resonance)
        resonance = torch.matmul(rep_norm, basis_norm.t()) * 16.0  # Sharp energy snapping

        return resonance, next_states, total_energy


# =============================================================================
# PARADIGM B: DYNAMIC DIMENSIONALITY BLOOMING (MANIFOLD MORPHOGENESIS)
# =============================================================================

class ParadigmBDimensionalBlooming(nn.Module):
    """
    Paradigm B: The architecture evolves by expanding the dimensionality of its continuous
    manifold itself (R^D -> R^{D + dD}) as surprise accumulates in the continuous stream.
    The Hamiltonian metric tensor grows in rank, creating new orthogonal degrees of freedom.
    """
    def __init__(self, vocab_size: int = 258, initial_dim: int = 32, max_dim: int = 128, device: torch.device = DEVICE):
        super().__init__()
        self.vocab_size = vocab_size
        self.current_dim = initial_dim
        self.max_dim = max_dim
        self.device = device

        self.emb_weight = nn.Parameter(torch.randn(vocab_size, max_dim, device=device) * 0.1)

        # Hamiltonian phase space coordinates: q, p in R^(max_dim)
        # Only active up to self.current_dim
        self.W_hamiltonian = nn.Parameter(torch.randn(max_dim, max_dim, device=device) * (1.0 / math.sqrt(max_dim)))
        self.M_diag = nn.Parameter(torch.ones(max_dim, device=device))
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

        # Symplectic Hamiltonian flow on active subspace
        H_sub = self.W_hamiltonian[:d, :d]
        grad_V = torch.matmul(q, H_sub) + x

        # Symplectic leapfrog update
        tau = 0.2
        p_half = p - 0.5 * tau * grad_V
        q_next = q + tau * (p_half * torch.abs(self.M_diag[:d]))
        grad_V_next = torch.matmul(q_next, H_sub) + x
        p_next = p_half - 0.5 * tau * grad_V_next

        # Phase space total energy
        H_energy = 0.5 * torch.sum(p_next**2, dim=-1) + 0.5 * torch.sum(q_next**2, dim=-1)

        # Phase-lock resonance
        q_norm = F.normalize(q_next, dim=-1)
        basis_norm = F.normalize(self.readout_matrix[:, :d], dim=-1)
        resonance = torch.matmul(q_norm, basis_norm.t()) * 16.0

        q_out = torch.zeros_like(q_prev)
        p_out = torch.zeros_like(p_prev)
        q_out[:, :d] = q_next
        p_out[:, :d] = p_next

        return resonance, q_out, p_out, H_energy.mean()


# =============================================================================
# PARADIGM C: SELF-MODIFYING CONTINUOUS OPERATOR FIELD (STRAIN-DRIVEN KERNEL)
# =============================================================================

class ParadigmCSelfModifyingField(nn.Module):
    """
    Paradigm C: Continuous physical field Psi(x, t) where the differential evolution
    operator itself continuously evolves: dL/dt = F(Psi, strain).
    The field equation adapts its own kernel weights dynamically on every incoming byte.
    """
    def __init__(self, vocab_size: int = 258, dim: int = 64, device: torch.device = DEVICE):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.device = device

        self.emb = nn.Embedding(vocab_size, dim, device=device)

        # Base evolution kernel
        self.L_base = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        # Plastic meta-kernel (evolves operator during stream)
        self.L_plastic = nn.Parameter(torch.zeros(dim, dim, device=device))
        self.plasticity_rate = nn.Parameter(torch.tensor(0.01, device=device))

        self.readout_basis = nn.Parameter(torch.randn(vocab_size, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.evolution_events = 0

    def forward_step(self, byte_in: torch.Tensor, psi_prev: torch.Tensor):
        x = self.emb(byte_in)

        # Effective operator L_eff = L_base + L_plastic
        L_eff = self.L_base + self.L_plastic

        # Differential field relaxation: dPsi/dt = L_eff @ Psi + x
        d_psi = torch.matmul(psi_prev, L_eff) + x
        tau = 0.25
        psi_next = (1.0 - tau) * psi_prev + tau * d_psi

        # Endogenous operator self-modification:
        # Plastic operator strain update: delta L proportional to outer product of field strain
        strain = (psi_next - psi_prev).detach()
        with torch.no_grad():
            delta_L = torch.matmul(strain.t(), x).clamp(-0.05, 0.05)
            self.L_plastic.data += self.plasticity_rate * delta_L
            self.evolution_events += 1

        field_energy = 0.5 * torch.sum(strain**2, dim=-1)

        # Phase-lock resonance
        psi_norm = F.normalize(psi_next, dim=-1)
        basis_norm = F.normalize(self.readout_basis, dim=-1)
        resonance = torch.matmul(psi_norm, basis_norm.t()) * 16.0

        return resonance, psi_next, field_energy.mean()


# =============================================================================
# CONTINUOUS STREAM BENCHMARK RUNNER (N=1 SINGLE PASS)
# =============================================================================

def benchmark_evolutionary_stream(name: str, model: nn.Module, paradigm_type: str, stream_data: torch.Tensor, stream_types: torch.Tensor):
    print(f"\n===============================================================================")
    print(f"=== STREAMING BENCHMARK: {name} ===")
    print(f"===============================================================================")

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-5)
    t0 = time.time()

    # Initial continuous state initialization
    batch_size = 1
    stream_len = stream_data.size(0)

    # Performance tracking in continuous stream
    det_errors = []
    stoch_errors = []
    det_energies = []
    stoch_energies = []

    # Paradigm specific continuous persistent state
    if paradigm_type == "A":
        states = [torch.zeros(batch_size, model.dim, device=DEVICE)]
    elif paradigm_type == "B":
        q_state = torch.zeros(batch_size, model.max_dim, device=DEVICE)
        p_state = torch.zeros(batch_size, model.max_dim, device=DEVICE)
    elif paradigm_type == "C":
        psi_state = torch.zeros(batch_size, model.dim, device=DEVICE)

    rolling_loss = 0.0

    # Stream processing (t -> t+1 continuously, single pass N=1)
    for t in range(stream_len - 1):
        cur_byte = stream_data[t:t+1]
        next_byte = stream_data[t+1:t+2]
        is_stochastic = stream_types[t].item()

        optimizer.zero_grad()

        # Execute single continuous step
        if paradigm_type == "A":
            resonance, states, energy = model.forward_step(cur_byte, states)
            # Detach states to prevent infinite BPTT across unbroken stream
            states = [s.detach() for s in states]
        elif paradigm_type == "B":
            resonance, q_state, p_state, energy = model.forward_step(cur_byte, q_state, p_state)
            q_state = q_state.detach()
            p_state = p_state.detach()
        elif paradigm_type == "C":
            resonance, psi_state, energy = model.forward_step(cur_byte, psi_state)
            psi_state = psi_state.detach()

        # Compute instant stream loss
        loss = F.cross_entropy(resonance, next_byte) + 0.002 * energy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Autopoietic morphological evolution triggers during stream
        rolling_loss = 0.9 * rolling_loss + 0.1 * loss.item() if t > 0 else loss.item()

        if paradigm_type == "A" and t > 50 and t % 64 == 0:
            if rolling_loss > 1.2:
                sprouted = model.sprout_node()
                if sprouted:
                    states.append(torch.zeros(batch_size, model.dim, device=DEVICE))
                    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-5)
            model.prune_stale_nodes()

        elif paradigm_type == "B" and t > 50 and t % 64 == 0:
            if rolling_loss > 1.2:
                model.bloom_dimension(delta_d=16)
                optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-5)

        # Evaluate exact phase-lock accuracy (Did resonance match target exactly?)
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

    # Final quarter metrics (tail stability)
    q_len = max(1, len(det_errors) // 4)
    final_det_error_rate = float(np.mean(det_errors[-q_len:]))
    final_det_energy = float(np.mean(det_energies[-q_len:]))
    final_stoch_error_rate = float(np.mean(stoch_errors[-q_len:]))
    final_stoch_energy = float(np.mean(stoch_energies[-q_len:]))

    sprout_events = getattr(model, 'sprout_count', 0)
    prune_events = getattr(model, 'prune_count', 0)

    print(f"  --> Final Det Error Rate: {final_det_error_rate*100:.2f}% | Final Det Energy: {final_det_energy:.4f}")
    print(f"  --> Sprout Events: {sprout_events} | Prune Events: {prune_events} | Speed: {tok_per_sec:.1f} tok/s")

    return {
        "paradigm": name,
        "type": paradigm_type,
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


def run_exp_360():
    print("===============================================================================")
    print("=== KEP EXP-360: EVOLUTIONARY CONTINUOUS STREAM ARCHITECTURES (TRI-BATTLE)  ===")
    print("===============================================================================")
    print(f"Substrate Device: {DEVICE}")

    stream_data, stream_types = generate_continuous_stream(stream_length=2048)

    # Paradigm A: Topological Morphogenetic Graph (Sprouting/Pruning nodes)
    model_a = ParadigmATopologicalGraph(vocab_size=258, dim=64, max_nodes=8, device=DEVICE)
    res_a = benchmark_evolutionary_stream("Paradigm A: Topological Morphogenetic Graph", model_a, "A", stream_data, stream_types)

    # Paradigm B: Dynamic Dimensionality Blooming (State space expansion D -> D + dD)
    model_b = ParadigmBDimensionalBlooming(vocab_size=258, initial_dim=32, max_dim=128, device=DEVICE)
    res_b = benchmark_evolutionary_stream("Paradigm B: Dynamic Dimensionality Blooming", model_b, "B", stream_data, stream_types)

    # Paradigm C: Self-Modifying Operator Field (Plastic meta-kernel L_plastic)
    model_c = ParadigmCSelfModifyingField(vocab_size=258, dim=64, device=DEVICE)
    res_c = benchmark_evolutionary_stream("Paradigm C: Self-Modifying Continuous Operator Field", model_c, "C", stream_data, stream_types)

    print("\n===============================================================================")
    print("=== FINAL COMPARATIVE STREAMING TELEMETRY ===")
    print("===============================================================================")
    all_res = [res_a, res_b, res_c]
    for r in all_res:
        print(f"{r['paradigm']}:")
        print(f"  Deterministic Error Rate : {r['final_det_error_rate']*100:.2f}% (Energy: {r['final_det_energy']:.4f})")
        print(f"  Stochastic Error Rate    : {r['final_stoch_error_rate']*100:.2f}% (Energy: {r['final_stoch_energy']:.4f})")
        print(f"  Morphogenetic Evolution  : Sprouts={r['sprout_events']}, Prunes={r['prune_events']} | Speed={r['tok_per_sec']:.1f} tok/s\n")

    # Winner: lowest deterministic error rate + lowest energy
    winner = min(all_res, key=lambda x: (x['final_det_error_rate'], x['final_det_energy']))
    print(f"👑 VICTORIOUS EVOLUTIONARY ARCHITECTURE: {winner['paradigm']}")

    verdict = "🟢 POSITIVE" if winner['final_det_error_rate'] < 0.20 else "⚪ NEUTRAL"

    results_data = {
        "exp_id": "EXP-360",
        "results": all_res,
        "winner": winner['paradigm'],
        "winner_det_error_rate": float(winner['final_det_error_rate']),
        "winner_det_energy": float(winner['final_det_energy']),
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_360_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    run_exp_360()
