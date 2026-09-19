"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #258
Topic: Full Combinatorial Biophysical Ablation Study (A, B, C, AB, BC, AC, ABC)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist (2026)
Standard: KEP v10.0 Master Protocol | Rules #1, #2, #3, #6, #7, #8, #10, #11
===============================================================================

Theoretical Formulations:
- Vector A (Entropy-Adaptive Elastic Dynamics): Local Shannon uncertainty H(x_t) dynamically modulates temporal decay and memory write-gate.
- Vector B (Laminar Predictive Coding & Residual Error Routing): Deep manifold processes only unexplained prediction error residuals epsilon = x - x_hat.
- Vector C (Nonparametric Episodic Attractor Retrieval): Homeostatically gated Hopfield memory recall injecting historical context on demand.

Combinations evaluated:
1. Baseline (Clean Substrate, 0)
2. A (Entropy-Adaptive alone)
3. B (Laminar Error Residuals alone)
4. C (Nonparametric Episodic Memory alone)
5. AB (Entropy-Adaptive + Laminar Error)
6. BC (Laminar Error + Nonparametric Memory)
7. AC (Entropy-Adaptive + Nonparametric Memory)
8. ABC (Full Tri-Vector Cybernetic Symphony)
"""

import sys
from pathlib import Path
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import karyon_core  # noqa: E402


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CombinatorialBiophysicalAgent(nn.Module):
    def __init__(self, vocab_size=258, dim=256, use_a=False, use_b=False, use_c=False, device="cuda"):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.use_a = use_a
        self.use_b = use_b
        self.use_c = use_c
        self.device = device

        self.embedding = nn.Embedding(vocab_size, dim)
        self.ln_in = nn.LayerNorm(dim)

        # Stage 1: Fast Morphological Node
        self.node_fast = karyon_core.OmniMorphicNode(dim, 128, 8, device, 0.05, 0.5)

        # Vector B components (Laminar Prediction & Error Extraction)
        if self.use_b:
            self.laminar_pred_head = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )
            self.ln_residual = nn.LayerNorm(dim)

        # Stage 2: Contextual/Slow Node
        self.node_slow = karyon_core.OmniMorphicNode(dim, 128, 8, device, 0.0005, 0.02)

        # Vector C components (Episodic Hopfield Attractor Memory)
        if self.use_c:
            self.memory_keys = nn.Parameter(torch.randn(32, dim, device=device) * (1.0 / np.sqrt(dim)))
            self.memory_values = nn.Parameter(torch.randn(32, dim, device=device) * 0.02)
            self.mem_gate = nn.Sequential(
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )

        # Vector A components (Entropy-Adaptive Modulation parameter)
        if self.use_a:
            self.entropy_scale = nn.Parameter(torch.tensor(0.5, device=device))

        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.embedding.weight
        self.to(device)

    def forward(self, x):
        # x: [batch, seq_len]
        batch, seq_len = x.shape
        h0 = self.ln_in(self.embedding(x))

        # Vector A: Entropy Modulation on Input State
        if self.use_a:
            # Estimate local byte surprise / transitions
            diff = torch.cat([torch.zeros(batch, 1, self.dim, device=self.device), torch.abs(h0[:, 1:] - h0[:, :-1])], dim=1)
            local_entropy = torch.sigmoid(diff.mean(dim=-1, keepdim=True) * self.entropy_scale)
            h0 = h0 * (1.0 + 0.5 * local_entropy)

        # Stage 1: Fast dynamics
        h1 = self.ssd_fast(h0)

        # Vector B: Predictive Residual Routing
        if self.use_b:
            h1_pred = self.laminar_pred_head(h1)
            # Compute sensory prediction error residual
            residual_error = self.ln_residual(h0 - h1_pred)
            # Route prediction error into Stage 2 instead of raw h1
            h_stage2_in = h1 + residual_error
        else:
            h_stage2_in = h1

        # Stage 2: Slow contextual dynamics
        h2 = self.ssd_slow(h_stage2_in)

        # Vector C: Nonparametric Episodic Attractor Injection
        if self.use_c:
            # Continuous Hopfield cosine attention over memory basins
            norm_h2 = h2 / (torch.norm(h2, dim=-1, keepdim=True) + 1e-6)
            norm_keys = self.memory_keys / (torch.norm(self.memory_keys, dim=-1, keepdim=True) + 1e-6)
            sim = torch.einsum("btd,md->btm", norm_h2, norm_keys) * 8.0
            attn = torch.softmax(sim, dim=-1)
            retrieved = torch.einsum("btm,md->btd", attn, self.memory_values)
            gate = self.mem_gate(h2)
            h2 = h2 + gate * retrieved

        out = self.ln_out(h2 + h0)
        logits = self.head(out)
        return logits


def generate_structured_multiscale_data(num_samples=128, seq_len=128, device="cuda"):
    # Generates structured byte sequences with nested multi-scale grammar and repeating phrases
    patterns = [
        "active_inference_variational_free_energy_minimization",
        "dynamic_neural_darwinism_epigenetic_morphogenesis_growth",
        "continuous_state_space_duality_parallel_scan_acceleration",
        "ashby_homeostatic_somatic_ultrastability_attractor_basin"
    ]
    encoded_patterns = [[ord(c) for c in p] for p in patterns]

    data = []
    for _ in range(num_samples):
        seq = []
        while len(seq) < seq_len:
            pat = random.choice(encoded_patterns)
            seq.extend(pat)
            seq.append(ord(" "))
        data.append(seq[:seq_len])

    tensor_data = torch.tensor(data, dtype=torch.long, device=device)
    return tensor_data


def train_and_evaluate(variant_name, use_a, use_b, use_c, device="cuda", steps=120, lr=2e-3):
    model = CombinatorialBiophysicalAgent(vocab_size=258, dim=256, use_a=use_a, use_b=use_b, use_c=use_c, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train loop
    losses = []
    for _ in range(steps):
        batch_tokens = generate_structured_multiscale_data(num_samples=16, seq_len=128, device=device)
        inputs = batch_tokens[:, :-1]
        targets = batch_tokens[:, 1:]

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, 258), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    # Evaluation on separate held-out test data
    val_tokens = generate_structured_multiscale_data(num_samples=64, seq_len=128, device=device)
    with torch.no_grad():
        val_inputs = val_tokens[:, :-1]
        val_targets = val_tokens[:, 1:]
        val_logits = model(val_inputs)
        val_loss = criterion(val_logits.reshape(-1, 258), val_targets.reshape(-1)).item()

    return val_loss, np.mean(losses[-10:])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== EXP-258: FULL COMBINATORIAL BIOPHYSICAL ABLATION (Device: {device}) ===")

    variants = [
        ("Baseline (0)", False, False, False),
        ("Variant A (Entropy-Adaptive)", True, False, False),
        ("Variant B (Laminar Prediction Error)", False, True, False),
        ("Variant C (Episodic Memory)", False, False, True),
        ("Variant AB (Entropy + Laminar)", True, True, False),
        ("Variant BC (Laminar + Episodic)", False, True, True),
        ("Variant AC (Entropy + Episodic)", True, False, True),
        ("Variant ABC (Full Symphony)", True, True, True),
    ]

    seeds = [42, 1337]
    results = {}

    for name, a, b, c in variants:
        val_scores = []
        for s in seeds:
            set_seed(s)
            val_loss, _ = train_and_evaluate(name, a, b, c, device=device, steps=100)
            val_scores.append(val_loss)
        mean_val = float(np.mean(val_scores))
        std_val = float(np.std(val_scores))
        results[name] = {"mean": mean_val, "std": std_val, "scores": val_scores}
        print(f"-> {name:38s} | Val Loss: {mean_val:.4f} ± {std_val:.4f}")

    baseline_loss = results["Baseline (0)"]["mean"]
    print("\n" + "=" * 65)
    print(f"{'Variant':38s} | {'Val Loss':10s} | {'Delta vs Base':12s}")
    print("=" * 65)

    summary_metrics = {}
    for name in results:
        v_loss = results[name]["mean"]
        delta = baseline_loss - v_loss
        summary_metrics[name] = {"val_loss": v_loss, "delta": delta}
        print(f"{name:38s} | {v_loss:10.4f} | {delta:+12.4f} nats")
    print("=" * 65)

    # Save metrics JSON for telemetry
    with open("exp_258_metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)

    best_variant = min(results.keys(), key=lambda k: results[k]["mean"])
    print(f"\n🏆 WINNING CONFIGURATION: {best_variant} (Val Loss: {results[best_variant]['mean']:.4f})")


if __name__ == "__main__":
    main()
