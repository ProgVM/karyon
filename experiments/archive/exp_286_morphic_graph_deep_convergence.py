"""
[EXP-286] Dynamic Morphogenetic Graph Expressivity & Deep Convergence Benchmark:
Evaluates the architectural impact of LayerNorm Output Equalization and Multi-Operator
Morphogenesis (StateSpaceMemory + ContinuousHopfield) on deep loss convergence and text coherence.
"""

import sys
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath('.'))

import karyon_core as kcore
from karyon_agent import CoREAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CORPUS = [
    "Cognitive architectures model intentionality and dynamic predictive representations.",
    "Variational free energy minimization drives continuous allostatic homeostasis.",
    "Active inference couples somatic states with sensory-motor representations in brain.",
    "Continuous Hopfield attractors snap neural trajectories into discrete conceptual basins."
]


def run_benchmark():
    print("=" * 80)
    print("🚀 EXP-286: DYNAMIC MORPHIC GRAPH DEEP CONVERGENCE BENCHMARK")
    print("=" * 80)

    torch.manual_seed(42)
    vocab_size = 258
    k_dim = 256

    agent = CoREAgent(vocab_size=vocab_size, embed_dim=k_dim, use_graph=True, device=str(device)).to(device)
    
    # Add high-expressivity operators
    agent.add_node("morphic_hopfield", "ContinuousHopfield", is_core=False, initial_alpha=0.5)
    agent.add_node("morphic_ssd", "StateSpaceMemory", is_core=False, initial_alpha=0.5)

    manifest = json.loads(agent.get_topology_manifest())
    print(f"  • Active Topology: {manifest['k_nodes']} nodes, total trainable parameters: {len(list(agent.parameters()))}")

    optimizer = torch.optim.AdamW(agent.parameters(), lr=0.003, weight_decay=1e-4)

    start_time = time.time()
    total_tokens = 0
    losses = []

    print("\n  ▶ Training on Cognitive Corpus across 40 steps...")
    for step in range(40):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        text = CORPUS[step % len(CORPUS)]
        tokens = torch.tensor(list(text.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)

        logits = agent(tokens, thinking_steps=4)
        shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
        shift_labels = tokens[:, 1:].reshape(-1)

        loss = F.cross_entropy(shift_logits, shift_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        t_elapsed = time.perf_counter() - t0
        tokens_count = tokens.size(1)
        total_tokens += tokens_count
        tok_s = tokens_count / max(1e-6, t_elapsed)
        losses.append(loss.item())

        if (step + 1) % 10 == 0 or step == 0:
            print(f"    [Step {step+1:02d}/40] Loss: {loss.item():.4f} | Tok/s: {tok_s:7.1f}")

    final_loss = losses[-1]
    initial_loss = losses[0]
    delta_loss = initial_loss - final_loss
    total_elapsed = time.time() - start_time
    avg_tok_s = total_tokens / max(1e-6, total_elapsed)

    print("\n" + "=" * 80)
    print("📊 EXP-286 FINAL SUMMARY METRICS")
    print("=" * 80)
    print(f"  • Initial Loss       : {initial_loss:.4f}")
    print(f"  • Final Loss         : {final_loss:.4f} (Baseline: 2.8219)")
    print(f"  • Loss Improvement   : {delta_loss:.4f}")
    print(f"  • Average Throughput : {avg_tok_s:.1f} tok/s")

    # Speech generation audit
    print("\n  ▶ Diagnostic Speech Sampling:")
    agent.eval()
    with torch.no_grad():
        prompt = "Cognitive"
        input_tokens = torch.tensor(list(prompt.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
        gen_tokens = list(input_tokens[0].cpu().numpy())
        curr = input_tokens
        for _ in range(30):
            out = agent(curr, thinking_steps=2)
            next_token = out[:, -1, :].argmax(dim=-1).item()
            gen_tokens.append(next_token)
            curr = torch.tensor([gen_tokens], dtype=torch.long, device=device)

        gen_bytes = bytes([b for b in gen_tokens if 0 <= b < 256])
        try:
            sample_text = gen_bytes.decode('utf-8', errors='replace')
        except Exception:
            sample_text = str(gen_bytes)
        print(f"    • Prompt: '{prompt}' -> Generated: '{sample_text}'")

    results = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "delta_loss": delta_loss,
        "avg_tok_s": avg_tok_s,
        "generated_sample": sample_text
    }
    with open("exp_286_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_benchmark()
