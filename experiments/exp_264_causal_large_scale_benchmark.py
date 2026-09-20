import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import karyon_core

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.exp_262_1_morphic_circuit_space_benchmark import AlpacaByteStream  # noqa: E402


class MorphicCircuitCell_Causal(nn.Module):
    """
    Morphic Circuit Cell with Causal SSD Integration:
    Combines C++20 Causal SSD (temporal context) with local fast-slow Operators.
    """
    def __init__(self, dim=256, num_units=4, device='cuda'):
        super().__init__()
        self.dim = dim
        self.num_units = num_units

        # C++20 Causal SSD for context
        self.ssd = karyon_core.CausalParallelSSD(dim, str(device))

        # Fast Micro-Operator Core (Gamma Flow)
        self.log_alpha = nn.Parameter(torch.randn(dim) * 0.1 - 2.0)
        self.w_atom = nn.Linear(dim, dim, bias=False)
        self.gate_atom = nn.Linear(dim, dim, bias=False)

        # Dynamic Circuit Builder & Signal Transporter (Vector D)
        self.routing_matrix = nn.Parameter(torch.randn(num_units, num_units) * 0.05)
        self.w_units = nn.Parameter(torch.randn(num_units, dim, dim) * (0.2 / (dim ** 0.5)))
        self.formula_gate = nn.Parameter(torch.randn(num_units, 1, 1, dim) * 0.01)
        self.unit_alphas = nn.Parameter(torch.ones(num_units, 1, 1, 1))

        self.tunnel_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.alpha_epi = nn.Parameter(torch.ones(1))

        nn.init.orthogonal_(self.w_atom.weight, gain=0.2)
        nn.init.orthogonal_(self.gate_atom.weight, gain=0.2)
        nn.init.orthogonal_(self.tunnel_proj.weight, gain=0.2)

    def forward(self, x, tunnel_in=None):
        B, S, D = x.shape
        U = self.num_units

        # 1. Temporal context extraction via causal parallel C++ SSD
        x_ssd = self.ssd.forward(x)

        if tunnel_in is not None:
            x_in = x_ssd + tunnel_in
        else:
            x_in = x_ssd

        # 2. Local Micro-Operator
        x_proj = F.silu(self.w_atom(x_in))
        alpha = torch.sigmoid(self.log_alpha).view(1, 1, -1)
        integrated = x_proj * (1.0 - alpha)
        gated = integrated * torch.sigmoid(self.gate_atom(x_in))
        x_local = x_in + gated

        # 3. Dynamic Circuit Builder (Vector D)
        route_weights = F.softmax(self.routing_matrix, dim=-1)
        bus = x_local.unsqueeze(0).expand(U, -1, -1, -1)
        routed = torch.einsum('uv, vbsd -> ubsd', route_weights, bus)

        routed_flat = routed.reshape(U, B * S, D)
        lin_flat = torch.bmm(routed_flat, self.w_units)
        lin_out = lin_flat.reshape(U, B, S, D)

        gate = torch.sigmoid(routed * self.formula_gate)
        formula_out = F.silu(lin_out * gate)
        normed = F.layer_norm(formula_out, (D,))

        circuit_out = (routed + torch.tanh(self.unit_alphas) * normed).mean(dim=0)

        # 4. Epigenetic zero-shock output
        cell_out = x + torch.tanh(self.alpha_epi) * self.norm(circuit_out)
        tunnel_out = self.tunnel_proj(cell_out)

        return cell_out, tunnel_out


class MorphicCircuitSpace_Causal(nn.Module):
    """
    Connected Cognitive Space of Morphic Circuit Cells with Causal SSD.
    """
    def __init__(self, vocab_size=258, dim=256, num_cells=2, device='cuda'):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_cells = num_cells

        self.emb = nn.Embedding(vocab_size, dim)
        self.cells = nn.ModuleList([MorphicCircuitCell_Causal(dim=dim, num_units=4, device=device) for _ in range(num_cells)])
        self.cross_cell_routing = nn.Parameter(torch.randn(num_cells, num_cells) * 0.05)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, tokens):
        x = self.emb(tokens)
        C = self.num_cells

        cross_weights = F.softmax(self.cross_cell_routing, dim=-1)
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


class Baseline_CausalSSD(nn.Module):
    """
    Pure C++20 Causal SSD Baseline.
    """
    def __init__(self, vocab_size=258, dim=256, device='cuda'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.ssd = karyon_core.CausalParallelSSD(dim, str(device))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, tokens):
        x = self.emb(tokens)
        x = self.ssd.forward(x)
        x = self.norm(x)
        return self.head(x)


def run_causal_large_scale():
    print("=" * 80)
    print("EXP-264: CAUSAL LARGE-SCALE DEEP TRAINING BENCHMARK (600 STEPS)")
    print("Enforcing physical arrow of time via C++20 CausalParallelSSD")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target Device: {device}")

    dataset = AlpacaByteStream('data/alpaca_data.json', max_items=5000, device=device)

    vocab_size = 258
    dim = 256
    num_steps = 600
    batch_size = 16
    seq_len = 128
    log_interval = 100

    models_to_test = [
        ("Baseline (Pure C++20 Causal SSD)", "baseline_causal"),
        ("Morphic Circuit Cell (1 Causal Cell)", "cell_causal_1"),
        ("Morphic Circuit Space (2 Causal Cells & Tunnels)", "space_causal_2")
    ]

    summary_results = {}

    for label, mode in models_to_test:
        print("\n" + "-" * 70)
        print(f"--- Launching Long-Horizon Training: {label} ({num_steps} Steps) ---")
        print("-" * 70)

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        if mode == "baseline_causal":
            model = Baseline_CausalSSD(vocab_size=vocab_size, dim=dim, device=device).to(device)
        elif mode == "cell_causal_1":
            model = MorphicCircuitSpace_Causal(vocab_size=vocab_size, dim=dim, num_cells=1, device=device).to(device)
        elif mode == "space_causal_2":
            model = MorphicCircuitSpace_Causal(vocab_size=vocab_size, dim=dim, num_cells=2, device=device).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)
        criterion = nn.CrossEntropyLoss()

        step_losses = []
        start_time = time.time()
        total_tokens = 0

        model.train()
        for step in range(1, num_steps + 1):
            batch = dataset.get_batch(batch_size=batch_size, seq_len=seq_len)
            targets = batch[:, 1:].contiguous()
            inputs = batch[:, :-1].contiguous()

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step_losses.append(loss.item())
            total_tokens += inputs.numel()

            if step % log_interval == 0 or step == num_steps:
                recent_loss = sum(step_losses[-log_interval:]) / min(len(step_losses), log_interval)
                elapsed_so_far = time.time() - start_time
                tok_s = total_tokens / elapsed_so_far if elapsed_so_far > 0 else 0
                ppl = math.exp(min(recent_loss, 20.0))
                print(f"  Step [{step:03d}/{num_steps}] | Loss: {recent_loss:.4f} | PPL: {ppl:.2f} | Speed: {tok_s:,.0f} tok/s")

        total_elapsed = time.time() - start_time
        final_loss = sum(step_losses[-50:]) / 50.0
        final_ppl = math.exp(min(final_loss, 20.0))
        avg_speed = total_tokens / total_elapsed if total_elapsed > 0 else 0

        # Auditing text generation sample (diagnostic sampling)
        model.eval()
        with torch.no_grad():
            prompt = torch.tensor([list("Instruction: What is energy?\nResponse:".encode('utf-8'))], dtype=torch.long, device=device)
            gen_tokens = prompt.clone()
            for _ in range(40):
                logits_out = model(gen_tokens)
                next_tok = torch.argmax(logits_out[:, -1, :], dim=-1, keepdim=True)
                gen_tokens = torch.cat([gen_tokens, next_tok], dim=-1)

            decoded_sample = bytes([int(t) for t in gen_tokens[0].cpu() if t < 256]).decode('utf-8', errors='replace')
            print(f"\n  [Sample Generation ({label})]:")
            print(f"  \"{decoded_sample[:100]}...\"\n")

        summary_results[label] = {
            "final_loss": final_loss,
            "final_ppl": final_ppl,
            "params": total_params,
            "avg_tok_per_sec": avg_speed,
            "elapsed_s": total_elapsed,
            "sample_output": decoded_sample[:120]
        }

    # Comparative evaluation
    baseline_loss = summary_results["Baseline (Pure C++20 Causal SSD)"]["final_loss"]
    baseline_ppl = summary_results["Baseline (Pure C++20 Causal SSD)"]["final_ppl"]

    print("\n" + "=" * 85)
    print("EXP-264 FINAL COMPARATIVE REPORT: CAUSAL LONG-HORIZON DEEP TRAINING")
    print("=" * 85)
    print(f"{'Architecture':<45} | {'Loss':<7} | {'PPL':<7} | {'Delta':<8} | {'Speed (tok/s)':<13}")
    print("-" * 85)

    for label, metrics in summary_results.items():
        delta = baseline_loss - metrics["final_loss"]
        print(f"{label:<45} | {metrics['final_loss']:.4f} | {metrics['final_ppl']:.2f} | {delta:+.4f}  | {metrics['avg_tok_per_sec']:,.0f}")

    best_variant = min(summary_results.keys(), key=lambda k: summary_results[k]["final_loss"])
    best_loss = summary_results[best_variant]["final_loss"]
    delta_loss = baseline_loss - best_loss
    verdict = "POSITIVE" if delta_loss >= 0.08 else ("NEUTRAL / INCONCLUSIVE" if delta_loss >= 0.0 else "REJECTED")

    print("=" * 85)
    print(f"Baseline Loss: {baseline_loss:.4f} (PPL: {baseline_ppl:.2f})")
    print(f"Best Architecture: {best_variant} | Loss: {best_loss:.4f} | Delta: {delta_loss:+.4f}")
    print(f"KEP Rule #2 Empirical Verdict: {verdict}")
    print("=" * 85)

    report = {
        "exp_id": "EXP-264",
        "num_steps": num_steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "baseline_loss": baseline_loss,
        "best_variant": best_variant,
        "best_loss": best_loss,
        "delta_loss": delta_loss,
        "verdict": verdict,
        "summary_results": summary_results
    }

    with open("experiments/exp_264_results.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    run_causal_large_scale()
