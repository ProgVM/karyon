# experiments/exp_355_epigenetic_grn_soliton_resonators.py
"""
===============================================================================
EXP-355: Epigenetic Gene Regulatory Network for Dynamic Tensor Manifold Synthesis
         & Quantum-Analogue Soliton Resonators (GRN-QTMS)
===============================================================================
Autopoietic Non-Linear Field Architecture:
- Differential Riemannian Laplace-Beltrami Operator on Learned Metric g_uv
- Soliton Field Propagation via Bounded Gross-Pitaevskii Equation
- Dynamic Epigenetic Methylation (mu_k) and Gene Expression Gating
- Epsilon-stabilized Numerics & LayerNorm Field Stabilization
===============================================================================
"""

import time
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class SolitonResonator(nn.Module):
    """
    Quantum-Analogue Soliton Resonator node governed by continuous field equations.
    """
    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        # Metric generator components (Riemannian Metric g_uv)
        self.metric_proj = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.metric_proj.weight, gain=0.1)

        # Field interaction weights
        self.w_real = nn.Linear(dim, dim)
        self.w_imag = nn.Linear(dim, dim)

        self.norm_real = nn.LayerNorm(dim)
        self.norm_imag = nn.LayerNorm(dim)

        self.gene_expression = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.self_interaction = nn.Parameter(torch.tensor(0.1))  # lambda coupling

    def forward(
        self,
        psi_real: torch.Tensor,
        psi_imag: torch.Tensor,
        ext_force: torch.Tensor,
        methylation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute Riemannian metric deformation
        g_deform = self.metric_proj(ext_force)
        g_factor = torch.sigmoid(g_deform) + 0.1

        # Laplace-Beltrami continuous spatial operator approximation
        laplace_real = self.w_real(psi_real * g_factor)
        laplace_imag = self.w_imag(psi_imag * g_factor)

        # Non-linear self-interaction term: lambda * |psi|^2 * psi
        psi_sq = torch.clamp(psi_real ** 2 + psi_imag ** 2, max=10.0)
        nonlin_factor = torch.clamp(self.self_interaction, -1.0, 1.0) * psi_sq * (1.0 - methylation.unsqueeze(-1))

        # Gross-Pitaevskii d_psi / dt field update
        d_psi_real = -laplace_imag + nonlin_factor * psi_imag + ext_force
        d_psi_imag = laplace_real - nonlin_factor * psi_real

        # Step integration with LayerNorm stabilization
        psi_real_next = self.norm_real(torch.tanh(psi_real + 0.05 * d_psi_real))
        psi_imag_next = self.norm_imag(torch.tanh(psi_imag + 0.05 * d_psi_imag))

        # Energy density / wave amplitude
        amplitude = torch.sqrt(psi_real_next ** 2 + psi_imag_next ** 2 + 1e-8)

        return psi_real_next, psi_imag_next, amplitude


class EpigeneticGRNEngine(nn.Module):
    """
    Epigenetic Gene Regulatory Network controlling resonator expression and methylation locks.
    """
    def __init__(self, num_resonators: int, dim: int):
        super().__init__()
        self.num_resonators = num_resonators
        self.grn_layer = nn.Linear(dim, num_resonators)
        self.methylation_layer = nn.Linear(dim, num_resonators)

    def forward(self, state_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expression = torch.softmax(self.grn_layer(state_repr), dim=-1)
        methylation = torch.sigmoid(self.methylation_layer(state_repr))
        return expression, methylation


class GRNQTMSModel(nn.Module):
    """
    Master Model integrating Epigenetic GRN and Soliton Resonators.
    """
    def __init__(self, vocab_size: int = 258, dim: int = 128, num_resonators: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_resonators = num_resonators

        self.embedding = nn.Embedding(vocab_size, dim)
        self.resonators = nn.ModuleList([SolitonResonator(dim) for _ in range(num_resonators)])
        self.grn = EpigeneticGRNEngine(num_resonators, dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x.shape
        emb = self.embedding(x)  # [B, S, D]

        curr_psi_real = torch.zeros(batch_size, self.dim, device=x.device)
        curr_psi_imag = torch.zeros(batch_size, self.dim, device=x.device)

        field_outputs = []
        kl_penalties = []

        for t in range(seq_len):
            ext_force = emb[:, t, :]  # [B, D]
            expr, meth = self.grn(ext_force)  # [B, K], [B, K]

            step_real = torch.zeros_like(ext_force)
            step_imag = torch.zeros_like(ext_force)

            for k in range(self.num_resonators):
                res = self.resonators[k]
                pr, pi, _ = res(curr_psi_real, curr_psi_imag, ext_force, meth[:, k])
                weight = expr[:, k:k + 1]

                step_real = step_real + weight * pr
                step_imag = step_imag + weight * pi

            curr_psi_real = step_real
            curr_psi_imag = step_imag

            amp_field = torch.sqrt(curr_psi_real ** 2 + curr_psi_imag ** 2 + 1e-8)
            field_outputs.append(amp_field)

            kl_pen = (expr * torch.log(expr + 1e-8)).sum(dim=-1).mean()
            kl_penalties.append(kl_pen)

        out_field = torch.stack(field_outputs, dim=1)  # [B, S, D]
        logits = self.head(out_field)  # [B, S, V]
        total_kl_reg = torch.stack(kl_penalties).mean()

        return logits, total_kl_reg, out_field


def generate_synthetic_bytecode_stream(num_samples: int = 128, seq_len: int = 64) -> torch.Tensor:
    """Generates synthetic machine opcode / raw byte stream."""
    return torch.randint(0, 256, (num_samples, seq_len), dtype=torch.long)


def run_exp_355_benchmark():
    print("=== Commencing KEP EXP-355: GRN-QTMS Soliton Resonator Benchmark ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on hardware device: {device}")

    vocab_size = 258
    dim = 128
    seq_len = 64
    batch_size = 32
    num_epochs = 10

    model = GRNQTMSModel(vocab_size=vocab_size, dim=dim, num_resonators=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    data = generate_synthetic_bytecode_stream(num_samples=256, seq_len=seq_len).to(device)

    initial_loss = None
    final_loss = None
    t0 = time.time()
    total_tokens = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_kl = 0.0

        for i in range(0, data.size(0), batch_size):
            batch = data[i:i + batch_size]
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            logits, kl_reg, _ = model(inputs)

            loss_rec = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            total_loss = loss_rec + 0.001 * kl_reg

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss_rec.item() * inputs.size(0)
            epoch_kl += kl_reg.item() * inputs.size(0)
            total_tokens += inputs.numel()

        avg_loss = epoch_loss / data.size(0)
        avg_kl = epoch_kl / data.size(0)

        if initial_loss is None:
            initial_loss = avg_loss
        final_loss = avg_loss

        print(f"Epoch {epoch + 1:02d}/{num_epochs:02d} | CrossEntropy Loss: {avg_loss:.4f} | Epigenetic KL: {avg_kl:.4f}")

    elapsed = time.time() - t0
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
    delta_loss = initial_loss - final_loss

    print("\n--- Benchmark Summary Metrics ---")
    print(f"Initial Loss  : {initial_loss:.4f}")
    print(f"Final Loss    : {final_loss:.4f}")
    print(f"Loss Delta    : {delta_loss:.4f}")
    print(f"Throughput    : {tok_per_sec:.2f} tok/s")
    print(f"Elapsed Time  : {elapsed:.2f} s")

    verdict = "🟢 POSITIVE" if delta_loss >= 0.08 else "⚪ NEUTRAL / INCONCLUSIVE"
    print(f"Final Verdict : {verdict}")

    results = {
        "exp_id": "EXP-355",
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "delta_loss": float(delta_loss),
        "tok_per_sec": float(tok_per_sec),
        "verdict": verdict
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_355_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_355_benchmark()
