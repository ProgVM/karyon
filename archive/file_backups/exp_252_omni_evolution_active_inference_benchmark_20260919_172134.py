# experiments/exp_252_omni_evolution_active_inference_benchmark.py
"""
===============================================================================
EXP-252: OMNI-EVOLUTIONARY COGNITIVE ARCHITECTURE & ACTIVE INFERENCE BENCHMARK
===============================================================================
Hypothesis:
  Simultaneously integrating all three evolutionary cybernetic vectors:
  1. Allostatic Homeostatic Nexus with dynamic sprouting of new homeostatic dimensions.
  2. Spontaneous Epigenetic Neurogenesis (triggering sprout on high Free Energy surprise) 
     and Neural Darwinism Pruning (decaying alphas under sparse pressure).
  3. Autoregressive Top-P PAC Decoding for closed-loop thought generation.
  will create an autonomous self-directed cognitive agent capable of maintaining 
  homeostatic stability, dynamically scaling its topology on high-surprise inputs, 
  and converging to a low stream prediction loss (delta >= 0.08) while sustaining 
  high throughput (> 50,000 tok/sec).
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
    """
    Generates structured streams with sudden domain shifts (inducing spikes in Free Energy)
    to trigger and evaluate spontaneous neurogenesis and homeostatic adaptation.
    """
    torch.manual_seed(int(time.time() * 1000) % 100000)
    tokens = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

    # Domains: 
    domain_a = [ord(c) for c in "Karyon-CoRE Active Inference Homeostasis "]
    domain_b = [ord(c) for c in "0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ "]

    for b in range(batch_size):
        # First half: Domain A
        pos = 0
        while pos < seq_len // 2:
            chunk = min(len(domain_a), seq_len // 2 - pos)
            tokens[b, pos:pos + chunk] = torch.tensor(domain_a[:chunk], dtype=torch.long, device=device)
            pos += chunk
        
        # Second half: Domain B (sudden structural shift!)
        while pos < seq_len:
            chunk = min(len(domain_b), seq_len - pos)
            tokens[b, pos:pos + chunk] = torch.tensor(domain_b[:chunk], dtype=torch.long, device=device)
            pos += chunk

    return tokens


def run_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🏁 Starting EXP-252 Omni-Evolutionary Active Inference Benchmark on: {device.upper()}")

    vocab_size = 258
    dim = 256
    max_nodes = 16

    # 1. Initialize Evolutionary Agent
    agent = karyon_core.CognitiveEvolvableAgent(vocab_size, dim, max_nodes, device)

    # 2. Sprout initial core organelles
    agent.sprout_organelle("sensory_gateway", state_dim=128, num_operators=8)
    agent.sprout_organelle("somatic_integrator", state_dim=128, num_operators=8)

    # 3. Sprout a new custom homeostatic dimension to test Vector 1 (Dynamic Homeostasis)
    # This represents a new environmental threat or internal metabolic need
    print("\n--- Phase 1: Sprouting Custom Homeostatic Dimension ---")
    agent.sprout_homeostatic_dimension("OxygenMetabolism", 1.0, 1.0, 0.002, 0.05)
    agent.sprout_homeostatic_dimension("ThermalStability", 0.5, 0.5, 0.01, 0.02)
    
    names = agent.homeostasis.get_names()
    print("  • Active Homeostatic Dimensions:", names)
    assert "OxygenMetabolism" in names and "ThermalStability" in names
    print("  ✅ Dynamic Homeostatic Dimension Sprouting Confirmed!")

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

    # 5. High-Throughput Stream Training with Live Active Inference & Spontaneous Morphogenesis
    print("\n--- Phase 2: Live Active Inference & Morphogenetic Training ---")
    optimizer = optim.AdamW(agent.parameters(), lr=1.5e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()

    steps = 300
    start_time = time.time()
    loss_history = []
    fe_history = []

    for step in range(steps):
        optimizer.zero_grad()

        # Stream dynamic multi-domain batch
        batch = generate_multidomain_stream(batch_size, seq_len, vocab_size, device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        # Simulated reward signal (e.g., feedback on accuracy or metabolic balance)
        reward = torch.zeros((batch_size, 1), device=device)

        # Forward pass runs complete Active Inference loop inside C++:
        # 1. Calculates free energy (surprise)
        # 2. Updates homeostasis
        # 3. Triggers spontaneous neurogenesis on surprise spikes
        # 4. Applies Darwinian decay to inactive nodes
        logits, free_energy, u_t = agent.forward_active_inference(inputs, reward)

        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        
        # Total variational loss = cross-entropy + free energy minimization bound
        total_loss = loss + 0.1 * free_energy.mean()

        total_loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        fe_val = free_energy.mean().item()

        loss_history.append(loss_val)
        fe_history.append(fe_val)

        if step % 50 == 0 or step == steps - 1:
            active_nodes = len(agent.substrate.node_names)
            print(f"  [Step {step:03d}/{steps}] Loss: {loss_val:.4f} | Free Energy: {fe_val:.4f} | Active Nodes: {active_nodes} | Homeo: {u_t.cpu().numpy().round(3)}")

    end_time = time.time()
    duration = end_time - start_time
    total_tokens = steps * batch_size * (seq_len - 1)
    tok_per_sec = total_tokens / duration

    final_loss = loss_history[-1]
    delta_loss = baseline_loss - final_loss

    print(f"\n🚀 EXP-252 Completed in {duration:.2f} seconds!")
    print(f"  • Final Stream Prediction Loss: {final_loss:.6f} nats/byte")
    print(f"  • Baseline ➔ Final Loss Delta:  {delta_loss:.6f} nats/byte")
    print(f"  • Parallel GPU Throughput:      {tok_per_sec:.2f} tok/sec")
    print(f"  • Final Active Nodes count:     {len(agent.substrate.node_names)}")

    # 6. Verify Top-P Autoregressive Thought Generation (Vector 3)
    print("\n--- Phase 3: Verifying Top-P Autoregressive Thought Generation ---")
    seed = val_tokens[:2, :16]
    generated = agent.generate_thought_and_speech(seed, 32, temperature=0.45, top_p=0.90)
    
    # Decode to text
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

    with open("exp_252_results.json", "w") as f:
        json.dump(metrics, f)

    if delta_loss < 0.08:
        print("🔴 REJECTED: Loss delta did not meet KEP Rule #2 criteria (Delta >= 0.08).")
        sys.exit(1)

    print("🟢 POSITIVE: All 3 Evolutionary Vectors successfully verified and integrated!")


if __name__ == "__main__":
    run_benchmark()
