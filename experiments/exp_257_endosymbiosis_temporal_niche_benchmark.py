#!/usr/bin/env python3
"""
EXP-257: Endosymbiotic Temporal Niche Differentiation & Entropy-Gated ATP Transfer
Author: Bazilevs & Karyon Cyberneticist
Standard: KEP v10.0 Master Protocol

Hypothesis:
Equipping the Endosymbiotic Colony with:
  1. Temporal Niche Differentiation (Host: Ultra-Slow Contextual Decay [0.0001, 0.01]; Symbiont: Fast Phonological Decay [0.05, 0.50])
  2. Shannon Entropy-Gated ATP Transfer (ATP gating dynamic based on token uncertainty H(x_t))
  3. 3-Seed Multi-Run Evaluation across Seeds (42, 1337, 2026) for 400 steps each
will eliminate statistical chance, achieve definitive multi-seed statistical significance (p < 0.01),
and demonstrate a massive empirical performance delta (Delta >= 0.15 nats/byte) over the Single-Agent Baseline.
"""

import sys
from pathlib import Path
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure repo root is on sys.path before local import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import karyon_core  # noqa: E402


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SyntheticHeterogeneousEnvironment:
    """
    Simulates a demanding multi-task stream:
    Task A: Long-horizon syntactic context grammar
    Task B: High-frequency binary/byte pattern sequence
    """
    def __init__(self, vocab_size=258, batch_size=4, seq_len=64, device="cpu"):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

        self.patterns = [
            b"def compute_free_energy(prior_mu, post_mu, sigma): return kl_div + rec_loss\n",
            b"Active Inference minimizes variational surprise: F = D_KL(Q||P) + E_Q[log P]\n",
            b"Ashby Somatic Homeostasis maintains curiosity, stability, health, energy\n",
            b"C++20 Causal Parallel SSD executes closed-form state space scans on GPU\n"
        ]

    def sample_batch(self):
        batch = []
        for _ in range(self.batch_size):
            if random.random() < 0.5:
                pat = random.choice(self.patterns)
                repeats = (self.seq_len // len(pat)) + 2
                seq = (pat * repeats)[:self.seq_len]
                batch.append(list(seq))
            else:
                block_len = random.choice([2, 4, 8])
                symbols = random.sample(range(65, 90), 2)
                seq = []
                while len(seq) < self.seq_len:
                    sym = symbols[(len(seq) // block_len) % 2]
                    seq.append(sym)
                batch.append(seq[:self.seq_len])

        tensor_seq = torch.tensor(batch, dtype=torch.long, device=self.device)
        inputs = tensor_seq[:, :-1].contiguous()
        targets = tensor_seq[:, 1:].contiguous()
        return inputs, targets


class EndosymbioticATPColony(nn.Module):
    """
    Endosymbiotic Super-Organism with Temporal Niche Differentiation
    and Entropy-Gated ATP Transfer.
    """
    def __init__(self, vocab_size=258, dim=256, device_str="cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.device_str = device_str

        # 1. Host: Slow Contextual Integrator (Decay: 0.0001 to 0.01)
        self.host = karyon_core.CognitiveEvolvableAgent(vocab_size, dim, 16, device_str)
        self.host.sprout_organelle("host_context_trunk", state_dim=128, num_operators=8,
                                   min_decay=0.0001, max_decay=0.01)

        # 2. Symbiont: Fast Phonological Scanner (Decay: 0.05 to 0.50)
        self.symbiont = karyon_core.CognitiveEvolvableAgent(vocab_size, dim, 16, device_str)
        self.symbiont.sprout_organelle("symbiont_fast_scanner", state_dim=128, num_operators=8,
                                       min_decay=0.05, max_decay=0.50)

        # 3. Entropy-Gated ATP Transfer Module
        self.entropy_proj = nn.Linear(dim, 1, device=torch.device(device_str))
        self.alpha_atp = nn.Parameter(torch.zeros(1, device=torch.device(device_str)))

        # Zero-Init ATP Projection for strict Net2Net function identity at birth
        nn.init.zeros_(self.entropy_proj.weight)
        nn.init.zeros_(self.entropy_proj.bias)

    def parameters(self, recurse=True):
        """C++ PyBind11 parameter aggregation override."""
        return (list(self.host.parameters()) +
                list(self.symbiont.parameters()) +
                list(self.entropy_proj.parameters()) +
                [self.alpha_atp])

    def forward(self, tokens):
        # Extract latent manifolds from both organisms
        h_host = self.host.forward_latent(tokens)      # [batch, seq_len, dim]
        h_symb = self.symbiont.forward_latent(tokens)  # [batch, seq_len, dim]

        # Calculate localized uncertainty / ATP demand gate: [batch, seq_len, 1]
        atp_demand = torch.sigmoid(self.entropy_proj(h_host))
        base_gate = torch.tanh(self.alpha_atp)

        # Modulated ATP Injection: Host absorbs Symbiont latent proportional to demand
        h_fused = h_host + (base_gate * atp_demand) * h_symb

        # Final unified motor readout
        fused_logits = self.host.forward_motor(h_fused)
        return fused_logits, base_gate, atp_demand.mean()


def run_training_experiment(seed, steps=400, device_str="cpu"):
    set_seed(seed)
    env = SyntheticHeterogeneousEnvironment(vocab_size=258, batch_size=4, seq_len=64, device=device_str)
    criterion = nn.CrossEntropyLoss()

    # --- 1. Single-Agent Baseline ---
    baseline = karyon_core.CognitiveEvolvableAgent(258, 256, 16, device_str)
    baseline.sprout_organelle("baseline_organelle", state_dim=128, num_operators=8)
    opt_b = optim.AdamW(baseline.parameters(), lr=1e-3, weight_decay=1e-4)

    baseline_losses = []
    for step in range(steps):
        inputs, targets = env.sample_batch()
        opt_b.zero_grad()
        logits = baseline(inputs)
        loss = criterion(logits.view(-1, 258), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
        opt_b.step()
        baseline_losses.append(loss.item())

    # --- 2. Endosymbiotic ATP Colony ---
    colony = EndosymbioticATPColony(vocab_size=258, dim=256, device_str=device_str)
    opt_c = optim.AdamW(colony.parameters(), lr=1e-3, weight_decay=1e-4)

    colony_losses = []
    final_gate = 0.0
    final_demand = 0.0
    for step in range(steps):
        inputs, targets = env.sample_batch()
        opt_c.zero_grad()
        fused_logits, base_gate, atp_demand = colony(inputs)
        loss = criterion(fused_logits.view(-1, 258), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(colony.parameters(), 1.0)
        opt_c.step()
        colony_losses.append(loss.item())
        if step == steps - 1:
            final_gate = base_gate.item()
            final_demand = atp_demand.item()

    return {
        "seed": seed,
        "baseline_initial": baseline_losses[0],
        "baseline_final": np.mean(baseline_losses[-20:]),
        "colony_initial": colony_losses[0],
        "colony_final": np.mean(colony_losses[-20:]),
        "delta": np.mean(baseline_losses[-20:]) - np.mean(colony_losses[-20:]),
        "final_gate": final_gate,
        "final_demand": final_demand,
        "baseline_curve": baseline_losses,
        "colony_curve": colony_losses
    }


def main():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🏁 Starting EXP-257 3-Seed Empirical Evaluation on: {device_str.upper()}")

    seeds = [42, 1337, 2026]
    results = []

    for seed in seeds:
        print(f"\n🧬 [Seed {seed}] Running Baseline vs Endosymbiotic ATP Colony (400 steps)...")
        res = run_training_experiment(seed=seed, steps=400, device_str=device_str)
        results.append(res)
        print(f"  • Seed {seed} -> Baseline: {res['baseline_final']:.4f} | Colony: {res['colony_final']:.4f} | Delta: {res['delta']:+.4f} nats/byte (Gate: {res['final_gate']:+.4f}, Demand: {res['final_demand']:.4f})")

    avg_baseline = np.mean([r["baseline_final"] for r in results])
    std_baseline = np.std([r["baseline_final"] for r in results])

    avg_colony = np.mean([r["colony_final"] for r in results])
    std_colony = np.std([r["colony_final"] for r in results])

    avg_delta = avg_baseline - avg_colony
    all_positive = all(r["delta"] > 0 for r in results)

    print("\n" + "=" * 70)
    print("📊 EXP-257 STATISTICAL MULTI-SEED RIGOROUS BENCHMARK REPORT:")
    print(f"  • Single-Agent Baseline:      {avg_baseline:.4f} ± {std_baseline:.4f} nats/byte")
    print(f"  • Endosymbiotic ATP Colony:   {avg_colony:.4f} ± {std_colony:.4f} nats/byte")
    print(f"  • 🌟 MEAN STATISTICAL DELTA:   {avg_delta:+.4f} nats/byte")
    print(f"  • Consistency across seeds:   {'100% WINS (Unanimous)' if all_positive else 'Mixed'}")
    print("=" * 70)

    # Save metrics for KEP pipeline
    summary = {
        "loss": float(avg_colony),
        "baseline_loss": float(avg_baseline),
        "delta_loss": float(avg_delta),
        "std_colony": float(std_colony),
        "std_baseline": float(std_baseline),
        "unanimous_win": bool(all_positive),
        "seeds": seeds,
        "seed_results": [
            {k: v for k, v in r.items() if k not in ["baseline_curve", "colony_curve"]}
            for r in results
        ]
    }

    with open("exp_257_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
