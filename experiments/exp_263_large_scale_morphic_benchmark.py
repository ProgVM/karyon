import os
import sys
import time
import json
import math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.exp_262_1_morphic_circuit_space_benchmark import (  # noqa: E402
    MorphicCircuitSpace,
    AlpacaByteStream
)
from experiments.sub_organelle_modules import (  # noqa: E402
    OmniSubOrganellarEvolutionCore
)


def run_large_scale_experiment():
    print("=" * 80)
    print("EXP-263: LARGE-SCALE DEEP TRAINING BENCHMARK (600 STEPS, ALPACA CORPUS)")
    print("Testing long-horizon convergence, loss trajectory, and stability")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target Accelerator: {device}")

    # Load 5,000 items (~1.8 MB UTF-8 byte stream)
    dataset = AlpacaByteStream('data/alpaca_data.json', max_items=5000, device=device)

    vocab_size = 258
    dim = 256
    num_steps = 600
    batch_size = 16
    seq_len = 128
    log_interval = 100

    models_to_test = [
        ("Baseline (No Morphic Cells)", "baseline"),
        ("Morphic Circuit Cell (1 Cell Standalone)", "cell_1"),
        ("Morphic Circuit Space (2 Inter-Connected Cells & Tunnels)", "space_2")
    ]

    all_trajectories = {}
    summary_results = {}

    for label, mode in models_to_test:
        print("\n" + "-" * 70)
        print(f"--- Launching Long-Horizon Training: {label} ({num_steps} Steps) ---")
        print("-" * 70)

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        if mode == "baseline":
            model = OmniSubOrganellarEvolutionCore(vocab_size=vocab_size, dim=dim, active_vectors=[]).to(device)
        elif mode == "cell_1":
            model = MorphicCircuitSpace(vocab_size=vocab_size, dim=dim, num_cells=1).to(device)
        elif mode == "space_2":
            model = MorphicCircuitSpace(vocab_size=vocab_size, dim=dim, num_cells=2).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)
        # Cosine annealing scheduler
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
        final_loss = sum(step_losses[-50:]) / 50.0  # Average over last 50 steps
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

        all_trajectories[label] = step_losses
        summary_results[label] = {
            "final_loss": final_loss,
            "final_ppl": final_ppl,
            "params": total_params,
            "avg_tok_per_sec": avg_speed,
            "elapsed_s": total_elapsed,
            "sample_output": decoded_sample[:120]
        }

    # Comparative evaluation
    baseline_loss = summary_results["Baseline (No Morphic Cells)"]["final_loss"]
    baseline_ppl = summary_results["Baseline (No Morphic Cells)"]["final_ppl"]

    print("\n" + "=" * 85)
    print("EXP-263 FINAL COMPARATIVE REPORT: LONG-HORIZON DEEP TRAINING (600 STEPS)")
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
        "exp_id": "EXP-263",
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

    with open("experiments/exp_263_results.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    run_large_scale_experiment()
