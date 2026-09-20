import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class MorphicCircuitCell(nn.Module):
    """
    Refined Morphic Circuit Cell (MCC v2.0):
    Directly builds on the empirically proven Vector A (Micro-Operators)
    and Vector D (Circuit Builder), augmented with dynamic inter-cell hyper-tunnels.
    """
    def __init__(self, dim=256, num_units=4):
        super().__init__()
        self.dim = dim
        self.num_units = num_units

        # 1. Fast Micro-Operator Core (Gamma Flow)
        self.log_alpha = nn.Parameter(torch.randn(dim) * 0.1 - 2.0)
        self.w_atom = nn.Linear(dim, dim, bias=False)
        self.gate_atom = nn.Linear(dim, dim, bias=False)

        # 2. Dynamic Circuit Builder & Signal Transporter (Vector D)
        self.routing_matrix = nn.Parameter(torch.randn(num_units, num_units) * 0.05)
        self.w_units = nn.Parameter(torch.randn(num_units, dim, dim) * (0.2 / (dim ** 0.5)))
        self.formula_gate = nn.Parameter(torch.randn(num_units, 1, 1, dim) * 0.01)
        self.unit_alphas = nn.Parameter(torch.ones(num_units, 1, 1, 1))

        # 3. Inter-Cell Hyper-Tunnel Projection
        self.tunnel_proj = nn.Linear(dim, dim, bias=False)

        self.norm = nn.LayerNorm(dim)
        self.alpha_epi = nn.Parameter(torch.ones(1))

        nn.init.orthogonal_(self.w_atom.weight, gain=0.2)
        nn.init.orthogonal_(self.gate_atom.weight, gain=0.2)
        nn.init.orthogonal_(self.tunnel_proj.weight, gain=0.2)

    def forward(self, x, tunnel_in=None):
        B, S, D = x.shape
        U = self.num_units

        # Merge incoming hyper-tunnel if present
        if tunnel_in is not None:
            x_in = x + tunnel_in
        else:
            x_in = x

        # 1. High-speed local Micro-Operator
        x_proj = F.silu(self.w_atom(x_in))
        alpha = torch.sigmoid(self.log_alpha).view(1, 1, -1)
        integrated = x_proj * (1.0 - alpha)
        gated = integrated * torch.sigmoid(self.gate_atom(x_in))
        x_local = x_in + gated

        # 2. Dynamic Formula & Circuit Routing (Vector D)
        route_weights = F.softmax(self.routing_matrix, dim=-1)  # [U, U]
        bus = x_local.unsqueeze(0).expand(U, -1, -1, -1)        # [U, B, S, D]
        routed = torch.einsum('uv, vbsd -> ubsd', route_weights, bus)

        routed_flat = routed.reshape(U, B * S, D)
        lin_flat = torch.bmm(routed_flat, self.w_units)
        lin_out = lin_flat.reshape(U, B, S, D)

        gate = torch.sigmoid(routed * self.formula_gate)
        formula_out = F.silu(lin_out * gate)
        normed = F.layer_norm(formula_out, (D,))

        circuit_out = (routed + torch.tanh(self.unit_alphas) * normed).mean(dim=0)

        # 3. Output & Tunnel Generation
        cell_out = x + torch.tanh(self.alpha_epi) * self.norm(circuit_out)
        tunnel_out = self.tunnel_proj(cell_out)

        return cell_out, tunnel_out


class MorphicCircuitSpace(nn.Module):
    """
    Connected Cognitive Space of Morphic Circuit Cells:
    Allows cells to exchange hyper-tunnel signals and self-assemble arbitrary topologies.
    """
    def __init__(self, vocab_size=258, dim=256, num_cells=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_cells = num_cells

        self.emb = nn.Embedding(vocab_size, dim)
        self.cells = nn.ModuleList([MorphicCircuitCell(dim=dim, num_units=4) for _ in range(num_cells)])
        self.cross_cell_routing = nn.Parameter(torch.randn(num_cells, num_cells) * 0.05)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, tokens):
        x = self.emb(tokens)  # [B, S, D]
        C = self.num_cells

        cross_weights = F.softmax(self.cross_cell_routing, dim=-1)  # [C, C]
        cell_states = [x for _ in range(C)]
        cell_tunnels = [None for _ in range(C)]

        for i in range(C):
            if any(t is not None for t in cell_tunnels):
                stacked = torch.stack([t if t is not None else torch.zeros_like(x) for t in cell_tunnels], dim=0)
                tunnel_sig = torch.einsum('c, cbsd -> bsd', cross_weights[i], stacked)
            else:
                tunnel_sig = None

            c_out, t_out = self.cells[i](cell_states[i], tunnel_in=tunnel_sig)
            cell_states[i] = c_out
            cell_tunnels[i] = t_out

        final_state = self.norm(torch.stack(cell_states, dim=0).mean(dim=0))
        return self.head(final_state)


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
    print("EXP-262.1: MORPHIC CIRCUIT CELL & SPACE BENCHMARK")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target Device: {device}")

    dataset = AlpacaByteStream('data/alpaca_data.json', max_items=2000, device=device)

    vocab_size = 258
    dim = 256
    num_steps = 100
    batch_size = 16
    seq_len = 128

    models_to_test = [
        ("Baseline (No Morphic Cells)", "baseline"),
        ("Morphic Circuit Cell (1 Cell Standalone)", "cell_1"),
        ("Morphic Circuit Space (2 Inter-Connected Cells & Tunnels)", "space_2"),
        ("Morphic Circuit Space (3 Inter-Connected Cells & Tunnels)", "space_3")
    ]

    results = {}

    for label, mode in models_to_test:
        print(f"\n--- Testing Architecture: {label} ---")
        torch.manual_seed(42)

        if mode == "baseline":
            from experiments.sub_organelle_modules import OmniSubOrganellarEvolutionCore
            model = OmniSubOrganellarEvolutionCore(vocab_size=vocab_size, dim=dim, active_vectors=[]).to(device)
        elif mode == "cell_1":
            model = MorphicCircuitSpace(vocab_size=vocab_size, dim=dim, num_cells=1).to(device)
        elif mode == "space_2":
            model = MorphicCircuitSpace(vocab_size=vocab_size, dim=dim, num_cells=2).to(device)
        elif mode == "space_3":
            model = MorphicCircuitSpace(vocab_size=vocab_size, dim=dim, num_cells=3).to(device)

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

    baseline_loss = results["Baseline (No Morphic Cells)"]["final_loss"]
    best_variant = min(results.keys(), key=lambda k: results[k]["final_loss"])
    best_loss = results[best_variant]["final_loss"]
    delta_loss = baseline_loss - best_loss

    print("\n" + "=" * 80)
    print("EXP-262.1 SUMMARY & TELEMETRY TABLE (REAL-WORLD ALPACA)")
    print("=" * 80)
    for k, v in results.items():
        loss_diff = baseline_loss - v['final_loss']
        print(f"{k:<65} | Loss: {v['final_loss']:.4f} | Delta: {loss_diff:+.4f} | Speed: {v['tok_per_sec']:.1f} tok/s")

    print(f"\nBaseline Loss: {baseline_loss:.4f}")
    print(f"Best Architecture: {best_variant} | Loss: {best_loss:.4f} | Loss Delta: {delta_loss:+.4f}")

    verdict = "POSITIVE" if delta_loss >= 0.08 else "REJECTED"
    print(f"KEP Rule #2 Verdict: {verdict}")

    report = {
        "exp_id": "EXP-262.1",
        "baseline_loss": baseline_loss,
        "best_variant": best_variant,
        "best_loss": best_loss,
        "delta_loss": delta_loss,
        "verdict": verdict,
        "variant_results": results
    }

    with open("experiments/exp_262_1_results.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    run_experiment()
