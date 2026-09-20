import os
import sys
import time
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.sub_organelle_modules import OmniSubOrganellarEvolutionCore  # noqa: E402


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
    print("EXP-261: FAST-SLOW DUAL-FREQUENCY COGNITIVE ENGINE BENCHMARK")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target Device: {device}")

    dataset = AlpacaByteStream('data/alpaca_data.json', max_items=2000, device=device)

    vocab_size = 258
    dim = 256
    num_steps = 100
    batch_size = 16
    seq_len = 128

    variants = [
        ("Baseline (No Sub-Organelles)", []),
        ("Vector A (High-Speed Micro-Operators)", ['A']),
        ("Vector D (Batched Formula & Circuit Builder)", ['D']),
        ("Fast-Slow Dual-Frequency Engine (FS)", ['FS'])
    ]

    results = {}

    for label, active_vecs in variants:
        print(f"\n--- Testing Variant: {label} ---")
        torch.manual_seed(42)
        model = OmniSubOrganellarEvolutionCore(vocab_size=vocab_size, dim=dim, active_vectors=active_vecs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        for name, param in model.named_parameters():
            if 'alpha_epi' in name or 'unit_alphas' in name:
                param.data.fill_(1.0)

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

        print(f"Variant: {label} | Final Loss: {avg_loss:.4f} | Tok/s: {tok_per_sec:.2f}")
        results[label] = {
            "final_loss": avg_loss,
            "tok_per_sec": tok_per_sec,
            "elapsed_s": elapsed
        }

    baseline_loss = results["Baseline (No Sub-Organelles)"]["final_loss"]
    best_variant = min(results.keys(), key=lambda k: results[k]["final_loss"])
    best_loss = results[best_variant]["final_loss"]
    delta_loss = baseline_loss - best_loss

    print("\n" + "=" * 80)
    print("EXP-261 SUMMARY & TELEMETRY TABLE (REAL-WORLD ALPACA)")
    print("=" * 80)
    for k, v in results.items():
        loss_diff = baseline_loss - v['final_loss']
        print(f"{k:<65} | Loss: {v['final_loss']:.4f} | Delta: {loss_diff:+.4f} | Speed: {v['tok_per_sec']:.1f} tok/s")

    print(f"\nBaseline Loss: {baseline_loss:.4f}")
    print(f"Best Architecture: {best_variant} | Loss: {best_loss:.4f} | Loss Delta: {delta_loss:+.4f}")

    verdict = "POSITIVE" if delta_loss >= 0.08 else "REJECTED"
    print(f"KEP Rule #2 Verdict: {verdict}")

    report = {
        "exp_id": "EXP-261",
        "baseline_loss": baseline_loss,
        "best_variant": best_variant,
        "best_loss": best_loss,
        "delta_loss": delta_loss,
        "verdict": verdict,
        "variant_results": results
    }

    with open("experiments/exp_261_results.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    run_experiment()
