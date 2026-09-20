import time
import json
import torch
import torch.nn as nn

from experiments.sub_organelle_modules import OmniSubOrganellarEvolutionCore


def run_experiment():
    print("=" * 80)
    print("EXP-259: LOW-LEVEL SUB-ORGANELLES & UNIVERSAL SIGNAL TRANSPORTERS BENCHMARK")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target Device: {device}")

    torch.manual_seed(42)
    B, S = 16, 128
    vocab_size = 258
    dim = 256
    num_batches = 50

    train_data = [torch.randint(0, 258, (B, S), device=device) for _ in range(num_batches)]

    variants = [
        ("Baseline (No Sub-Organelles)", []),
        ("Vector A (Micro-Operator Nodes)", ['A']),
        ("Vector B (Micro-Channel Fiber Sprouting)", ['B']),
        ("Vector C (Functional Gene Assembly)", ['C']),
        ("Vector D (Autonomous Circuit & Formula Builder / Signal Transport)", ['D']),
        ("Vector ABCD Synergy (Full Sub-Organellar Architecture)", ['A', 'B', 'C', 'D'])
    ]

    results = {}

    for label, active_vecs in variants:
        print(f"\n--- Testing Variant: {label} ---")
        model = OmniSubOrganellarEvolutionCore(vocab_size=vocab_size, dim=dim, active_vectors=active_vecs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        for name, param in model.named_parameters():
            if 'alpha_epi' in name or 'unit_alphas' in name:
                param.data.fill_(0.5)

        start_time = time.time()
        total_loss = 0.0
        total_tokens = 0

        model.train()
        for i, batch in enumerate(train_data):
            optimizer.zero_grad()
            targets = batch[:, 1:].contiguous()
            inputs = batch[:, :-1].contiguous()

            logits = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += inputs.numel()

        elapsed = time.time() - start_time
        avg_loss = total_loss / num_batches
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
    print("EXP-259 SUMMARY & TELEMETRY TABLE")
    print("=" * 80)
    for k, v in results.items():
        loss_diff = baseline_loss - v['final_loss']
        print(f"{k:<65} | Loss: {v['final_loss']:.4f} | Delta: {loss_diff:+.4f} | Speed: {v['tok_per_sec']:.1f} tok/s")

    print(f"\nBaseline Loss: {baseline_loss:.4f}")
    print(f"Best Architecture: {best_variant} | Loss: {best_loss:.4f} | Loss Delta: {delta_loss:+.4f}")

    verdict = "POSITIVE" if delta_loss >= 0.08 else "REJECTED"
    print(f"KEP Rule #2 Verdict: {verdict}")

    report = {
        "exp_id": "EXP-259",
        "baseline_loss": baseline_loss,
        "best_variant": best_variant,
        "best_loss": best_loss,
        "delta_loss": delta_loss,
        "verdict": verdict,
        "variant_results": results
    }

    with open("experiments/exp_259_results.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    run_experiment()
