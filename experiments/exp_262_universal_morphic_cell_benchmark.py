import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class UniversalMorphicCell(nn.Module):
    """
    Universal Morphic Cell (UMC v1.0 Prototype)
    Fundamental cybernetic unit combining:
    1. Input/Output Tunnel & Cross-Cell Signal Router (Universal Bus: R_uv)
    2. Atomic Micro-Operator Core (Gamma Flow: Fast Leaky Integrator + Bilinear Gate)
    3. Differentiable Formula & Equation Synthesizer (Vector C/D Math Synthesis)
    4. Laminar Prediction & Error Tunnel (Predictive Coding: e_t = x_t - pred(x_t))
    5. Continuous Hopfield Phase Attractor (Topological Concept Basin)
    6. Epigenetic Smooth Grafting Gate (tanh(alpha_epi) * y_infant)
    """
    def __init__(self, dim=256, num_basins=16, num_formulas=4):
        super().__init__()
        self.dim = dim
        self.num_basins = num_basins
        self.num_formulas = num_formulas

        # 1. Atomic Micro-Operator Core (Gamma Flow)
        self.log_alpha = nn.Parameter(torch.randn(dim) * 0.1 - 2.0)
        self.w_atom = nn.Linear(dim, dim, bias=False)
        self.gate_atom = nn.Linear(dim, dim, bias=False)

        # 2. Differentiable Formula & Equation Synthesizer
        # Generates combinations of: Linear, SiLU, Bilinear Gating, Identity
        self.formula_weights = nn.Parameter(torch.randn(num_formulas, dim, dim) * (0.2 / (dim ** 0.5)))
        self.formula_gate = nn.Parameter(torch.randn(num_formulas, 1, 1, dim) * 0.01)
        self.formula_selector = nn.Parameter(torch.randn(num_formulas) * 0.1)

        # 3. Laminar Prediction & Error Tunnel
        self.pred_w1 = nn.Linear(dim, dim // 2, bias=False)
        self.pred_w2 = nn.Linear(dim // 2, dim, bias=False)
        self.ln_error = nn.LayerNorm(dim)

        # 4. Continuous Hopfield Phase Attractor
        self.hopfield_basins = nn.Parameter(torch.randn(num_basins, dim))
        self.hopfield_beta = 8.0

        # 5. Output Normalization & Epigenetic Zero-Shock Grafting
        self.norm = nn.LayerNorm(dim)
        self.alpha_epi = nn.Parameter(torch.zeros(1))

        # Safe Orthogonal Initializations
        nn.init.orthogonal_(self.w_atom.weight, gain=0.2)
        nn.init.orthogonal_(self.gate_atom.weight, gain=0.2)
        nn.init.orthogonal_(self.pred_w1.weight, gain=0.2)
        nn.init.orthogonal_(self.pred_w2.weight, gain=0.2)
        with torch.no_grad():
            self.hopfield_basins.copy_(F.normalize(self.hopfield_basins, p=2, dim=-1))

    def forward(self, x, tunnel_input=None):
        # x: [B, S, D]
        B, S, D = x.shape

        # Merge hyper-spatial tunnel input if connected
        if tunnel_input is not None:
            x_in = x + tunnel_input
        else:
            x_in = x

        # Step 1: Atomic Micro-Operator (Gamma Flow)
        atom_proj = F.silu(self.w_atom(x_in))
        alpha = torch.sigmoid(self.log_alpha).view(1, 1, -1)
        integrated = atom_proj * (1.0 - alpha)
        atom_gated = integrated * torch.sigmoid(self.gate_atom(x_in))

        # Step 2: Differentiable Formula Synthesizer (Tensor BMM)
        bus = atom_gated.unsqueeze(0).expand(self.num_formulas, -1, -1, -1)  # [F, B, S, D]
        bus_flat = bus.reshape(self.num_formulas, B * S, D)
        proj_flat = torch.bmm(bus_flat, self.formula_weights)
        proj_out = proj_flat.reshape(self.num_formulas, B, S, D)
        gate = torch.sigmoid(bus * self.formula_gate)
        formula_ops = F.silu(proj_out * gate)

        form_probs = F.softmax(self.formula_selector, dim=0).view(self.num_formulas, 1, 1, 1)
        synthesized_formula = (formula_ops * form_probs).sum(dim=0)  # [B, S, D]

        # Step 3: Laminar Prediction & Error Tunnel (Predictive Coding)
        pred = self.pred_w2(F.silu(self.pred_w1(synthesized_formula)))
        error_residual = self.ln_error(x_in - pred)
        h_laminar = synthesized_formula + error_residual

        # Step 4: Continuous Hopfield Phase Attractor
        basins_norm = F.normalize(self.hopfield_basins, p=2, dim=-1)  # [N, D]
        x_norm = F.normalize(h_laminar, p=2, dim=-1)                   # [B, S, D]
        sims = torch.matmul(x_norm, basins_norm.t()) * self.hopfield_beta  # [B, S, N]
        attn = F.softmax(sims, dim=-1)                                 # [B, S, N]
        attractor_out = torch.matmul(attn, basins_norm)                # [B, S, D]

        # Step 5: Fusion, Normalization & Epigenetic Zero-Shock Grafting
        cell_body = self.norm(h_laminar + attractor_out)
        out = x + torch.tanh(self.alpha_epi) * cell_body

        return out, attractor_out  # returns cell state and outgoing tunnel signal


class UniversalMorphicNetwork(nn.Module):
    """
    Universal Morphic Network (UMN):
    Assembles Universal Morphic Cells into dynamic structures, spaces, and hyper-tunnels.
    """
    def __init__(self, vocab_size=258, dim=256, num_cells=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_cells = num_cells

        self.emb = nn.Embedding(vocab_size, dim)
        self.cells = nn.ModuleList([UniversalMorphicCell(dim=dim) for _ in range(num_cells)])

        # Differentiable Inter-Cell Tunneling & Routing Matrix
        # Allows cells to form arbitrary DAGs, recurrent loops, or parallel spaces
        self.tunnel_matrix = nn.Parameter(torch.randn(num_cells, num_cells) * 0.05)

        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.norm = nn.LayerNorm(dim)

        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, tokens):
        # tokens: [B, S]
        x = self.emb(tokens)  # [B, S, D]

        tunnel_weights = F.softmax(self.tunnel_matrix, dim=-1)  # [C, C]
        cell_states = [x for _ in range(self.num_cells)]
        cell_tunnels = [None for _ in range(self.num_cells)]

        # Multi-stage Morphogenetic Transport
        for i in range(self.num_cells):
            # Compute incoming tunnel signal from all other cells
            if any(t is not None for t in cell_tunnels):
                stacked_tunnels = torch.stack([t if t is not None else torch.zeros_like(x) for t in cell_tunnels], dim=0)  # [C, B, S, D]
                # einsum routing
                in_tunnel = torch.einsum('c, cbsd -> bsd', tunnel_weights[i], stacked_tunnels)
            else:
                in_tunnel = None

            cell_out, tunnel_sig = self.cells[i](cell_states[i], tunnel_input=in_tunnel)
            cell_states[i] = cell_out
            cell_tunnels[i] = tunnel_sig

        # Global spatial pooling across the active morphic space
        final_state = self.norm(torch.stack(cell_states, dim=0).mean(dim=0))
        logits = self.head(final_state)
        return logits


class AlpacaByteStream:
    def __init__(self, json_path, max_items=2000, device='cpu'):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)[:max_items]

        raw_bytes = bytearray()
        for item in data:
            instr = item.get("instruction", "")
            inp = item.get("input", "")
            out = item.get("output", "")
            if inp:
                text = f"Instruction: {instr}\nInput: {inp}\nResponse: {out}\n\n"
            else:
                text = f"Instruction: {instr}\nResponse: {out}\n\n"
            raw_bytes.extend(text.encode('utf-8'))

        self.tensor = torch.tensor(list(raw_bytes), dtype=torch.long, device=device)
        self.total_bytes = len(self.tensor)
        print(f"  • Packed Alpaca Byte Stream: {self.total_bytes:,} bytes on {str(device).upper()}")

    def get_batch(self, batch_size=16, seq_len=128):
        max_idx = self.total_bytes - seq_len - 1
        starts = torch.randint(0, max_idx, (batch_size,), device=self.tensor.device)
        batch = torch.stack([self.tensor[s:s + seq_len] for s in starts])
        return batch


def run_experiment():
    print("=" * 80)
    print("EXP-262: UNIVERSAL MORPHIC CELL (UMC) PROTOTYPE & BENCHMARK")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target Device: {device}")

    dataset = AlpacaByteStream('data/alpaca_data.json', max_items=2000, device=device)

    vocab_size = 258
    dim = 256
    num_steps = 100
    batch_size = 16
    seq_len = 128

    # Comparing:
    # 1. Baseline Standard Model
    # 2. Universal Morphic Cell Network (1 Cell - Minimal Unit)
    # 3. Universal Morphic Network (3 Cells with Inter-Cell Tunneling & Routing)

    models_to_test = [
        ("Baseline (Standard Linear Recurrence)", None),
        ("Universal Morphic Cell (1 Cell Standalone)", 1),
        ("Universal Morphic Space (3 Inter-Connected Cells & Tunnels)", 3)
    ]

    results = {}

    for label, cell_count in models_to_test:
        print(f"\n--- Testing Architecture: {label} ---")
        torch.manual_seed(42)

        if cell_count is None:
            # Simple Baseline
            from experiments.sub_organelle_modules import OmniSubOrganellarEvolutionCore
            model = OmniSubOrganellarEvolutionCore(vocab_size=vocab_size, dim=dim, active_vectors=[]).to(device)
        else:
            model = UniversalMorphicNetwork(vocab_size=vocab_size, dim=dim, num_cells=cell_count).to(device)
            # Unfreeze epigenetic gates to activate cells
            for name, param in model.named_parameters():
                if 'alpha_epi' in name:
                    param.data.fill_(1.0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        total_loss = 0.0
        total_tokens = 0

        model.train()
        for step in range(num_steps):
            batch = dataset.get_batch(batch_size=batch_size, seq_len=seq_len)
            targets = batch[:, 1:].contiguous()
            inputs = batch[:, :-1].contiguous()

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += inputs.numel()

        elapsed = time.time() - start_time
        avg_loss = total_loss / num_steps
        tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0

        print(f"Model: {label} | Final Loss: {avg_loss:.4f} | Tok/s: {tok_per_sec:.2f}")
        results[label] = {
            "final_loss": avg_loss,
            "tok_per_sec": tok_per_sec,
            "elapsed_s": elapsed
        }

    baseline_loss = results["Baseline (Standard Linear Recurrence)"]["final_loss"]
    best_variant = min(results.keys(), key=lambda k: results[k]["final_loss"])
    best_loss = results[best_variant]["final_loss"]
    delta_loss = baseline_loss - best_loss

    print("\n" + "=" * 80)
    print("EXP-262 SUMMARY & TELEMETRY TABLE (REAL-WORLD ALPACA)")
    print("=" * 80)
    for k, v in results.items():
        loss_diff = baseline_loss - v['final_loss']
        print(f"{k:<65} | Loss: {v['final_loss']:.4f} | Delta: {loss_diff:+.4f} | Speed: {v['tok_per_sec']:.1f} tok/s")

    print(f"\nBaseline Loss: {baseline_loss:.4f}")
    print(f"Best Architecture: {best_variant} | Loss: {best_loss:.4f} | Loss Delta: {delta_loss:+.4f}")

    verdict = "POSITIVE" if delta_loss >= 0.08 else "REJECTED"
    print(f"KEP Rule #2 Verdict: {verdict}")

    report = {
        "exp_id": "EXP-262",
        "baseline_loss": baseline_loss,
        "best_variant": best_variant,
        "best_loss": best_loss,
        "delta_loss": delta_loss,
        "verdict": verdict,
        "variant_results": results
    }

    with open("experiments/exp_262_results.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    run_experiment()
