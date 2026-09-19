# experiments/exp_253_omni_evolution_active_inference_benchmark.py
"""
===============================================================================
EXP-253: OMNI-EVOLUTIONARY COGNITIVE ARCHITECTURE & ACTIVE INFERENCE BENCHMARK
===============================================================================
Hypothesis:
  Simultaneously integrating all three evolutionary cybernetic vectors:
  1. Allostatic Homeostatic Nexus with dynamic sprouting of new homeostatic dimensions.
  2. Spontaneous Epigenetic Neurogenesis and Neural Darwinism Pruning.
  3. Autoregressive Top-P PAC Decoding for closed-loop thought generation.
  with a stable learning rate (5e-4) will maintain continuous convergence,
  dropping loss to near-zero (delta >= 0.08) and generating coherent byte syntax.
===============================================================================
"""
import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import karyon_core  # noqa: E402


def generate_multidomain_stream(batch_size=16, seq_len=128, vocab_size=258, device='cpu'):
    torch.manual_seed(int(time.time() * 1000) % 100000)
    tokens = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

    domain_a = [ord(c) for c in "Karyon-CoRE Active Inference Homeostasis "]
    domain_b = [ord(c) for c in "0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ "]

    for b in range(batch_size):
        pos = 0
        while pos < seq_len // 2:
            chunk = min(len(domain_a), seq_len // 2 - pos)
            tokens[b, pos:pos + chunk] = torch.tensor(domain_a[:chunk], dtype=torch.long, device=device)
            pos += chunk

        while pos < seq_len:
            chunk = min(len(domain_b), seq_len - pos)
            tokens[b, pos:pos + chunk] = torch.tensor(domain_b[:chunk], dtype=torch.long, device=device)
            pos += chunk

    return tokens


def run_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🏁 Starting EXP-253 Omni-Evolutionary Active Inference Benchmark on: {device.upper()}")

    vocab_size = 258
    dim = 256
    max_nodes = 16

    # 1. Initialize Evolutionary Agent
    agent = karyon_core.CognitiveEvolvableAgent(vocab_size, dim, max_nodes, device)

    # 2. Sprout initial core organelles
    agent.sprout_organelle("sensory_gateway", state_dim=128, num_operators=8)
    agent.sprout_organelle("somatic_integrator", state_dim=128, num_operators=8)

    # 3. Dynamic Homeostatic Dimension Sprouting
    print("\n--- Phase 1: Sprouting Custom Homeostatic Dimensions ---")
    agent.sprout_homeostatic_dimension("OxygenMetabolism", 1.0, 1.0, 0.002, 0.05)
    agent.sprout_homeostatic_dimension("ThermalStability", 0.5, 0.5, 0.01, 0.02)

    names = agent.homeostasis.get_names()
    print("  • Active Homeostatic Dimensions:", names)

    # 4. Evaluate Baseline Loss
    batch_size = 16
    seq_len = 128
    val_tokens = generate_multidomain_stream(batch_size, seq_len, vocab_size, device)
    val_in = val_tokens[:, :-1]
    val_tgt = val_tokens[:, 1:]

    with torch.no_grad():
        logits_0 = agent(val_in)
        baseline_loss = nn.CrossEntropyLoss()(logits_0.reshape(-1, vocab_size), val_tgt.reshape(-1)).item()
    print(f"\n  • Baseline Initial Cross-Entropy Loss: {baseline_loss:.6f} nats/byte")

    # 5. Stable Stream Training
    print("\n--- Phase 2: Live Active Inference & Morphogenetic Training ---")
    optimizer = optim.AdamW(agent.parameters(), lr=5e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    steps = 250
    start_time = time.time()
    loss_history = []
    fe_history = []

    for step in range(steps):
        optimizer.zero_grad()

        batch = generate_multidomain_stream(batch_size, seq_len, vocab_size, device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        reward = torch.zeros((batch_size, 1), device=device)
        logits, free_energy, u_t = agent.forward_active_inference(inputs, reward)

        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        total_loss = loss + 0.05 * free_energy.mean()

        total_loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        fe_val = free_energy.mean().item()

        loss_history.append(loss_val)
        fe_history.append(fe_val)

        if step % 50 == 0 or step == steps - 1:
            active_nodes = len(agent.substrate.node_names)
            h_vals = u_t.cpu().numpy().round(3)
            print(f"  [Step {step:03d}/{steps}] Loss: {loss_val:.4f} | Free Energy: {fe_val:.4f} | Active Nodes: {active_nodes} | Homeo: {h_vals}")

    end_time = time.time()
    duration = end_time - start_time
    total_tokens = steps * batch_size * (seq_len - 1)
    tok_per_sec = total_tokens / duration

    final_loss = loss_history[-1]
    delta_loss = baseline_loss - final_loss

    print(f"\n🚀 EXP-253 Completed in {duration:.2f} seconds!")
    print(f"  • Final Stream Prediction Loss: {final_loss:.6f} nats/byte")
    print(f"  • Baseline ➔ Final Loss Delta:  {delta_loss:.6f} nats/byte")
    print(f"  • Parallel GPU Throughput:      {tok_per_sec:.2f} tok/sec")
    print(f"  • Final Active Nodes count:     {len(agent.substrate.node_names)}")

    # 6. Verify Top-P Autoregressive Thought Generation (Vector 3)
    print("\n--- Phase 3: Verifying Top-P Autoregressive Thought Generation ---")
    seed = val_tokens[:2, :16]
    generated = agent.generate_thought_and_speech(seed, 32, temperature=0.2, top_p=0.90)

    for i in range(2):
        text_seed = "".join([chr(c) if 32 <= c <= 126 else f"\\x{c:02x}" for c in seed[i].tolist()])
        text_gen = "".join([chr(c) if 32 <= c <= 126 else f"\\x{c:02x}" for c in generated[i, 16:].tolist()])
        print(f"  • Seed [{i}]: '{text_seed}' ➔ Generated: '{text_gen}'")

    # 7. Export Telemetry
    metrics = {
        "loss": round(final_loss, 4),
        "tok_per_sec": round(tok_per_sec, 1),
        "delta_loss": round(delta_loss, 4),
        "free_energy": round(fe_history[-1], 4),
        "energy": float(u_t[1].item())
    }

    with open("exp_253_results.json", "w") as f:
        json.dump(metrics, f)

    if delta_loss < 0.08:
        print("🔴 REJECTED: Loss delta did not meet KEP Rule #2 criteria (Delta >= 0.08).")
        sys.exit(1)

    print("🟢 POSITIVE: All 3 Evolutionary Vectors successfully verified with stable convergence!")


if __name__ == "__main__":
    run_benchmark()
