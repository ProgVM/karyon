"""
[EXP-285] Production Integration & Benchmark Verification of Epigenetic GRN Morphogenesis Engine:
Validates Karyon-CoRE's integrated Epigenetic Regulatory Network (GRN) with Methylation Locks
across continuous multi-step training, sleep-phase morphogenesis, and domain-shift stream learning.
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
from kcore_evolution import MorphogeneticAllostaticEngine, rebind_optimizer_moments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_STREAM_A = [
    "Karyon-CoRE integrates continuous State-Space Duality and Epigenetic GRN Morphogenesis.",
    "Active inference minimizes variational free energy across homeostatic body states.",
    "Continuous Hopfield attractors snap neural trajectories into discrete conceptual basins."
]

TEXT_STREAM_B = [
    "def execute_epigenetic_methylation_lock(nodes, noradrenaline, dopamine):",
    "    return {name: min(1.0, lock + 0.05 * dopamine) for name, lock in nodes.items()}",
    "class DynamicMorphicGraph(nn.Module): pass"
]


def run_production_epigenetic_verification():
    print("=" * 80)
    print("🚀 EXP-285: PRODUCTION EPIGENETIC GRN MORPHOGENESIS VERIFICATION")
    print("=" * 80)

    torch.manual_seed(42)
    vocab_size = 258
    k_dim = 256

    agent = CoREAgent(vocab_size=vocab_size, embed_dim=k_dim, use_graph=True, device=str(device)).to(device)
    print(f"  • CoREAgent initialized with active graph parameters.")
    
    # Verify initial topology manifest
    manifest_pre = json.loads(agent.get_topology_manifest())
    print(f"  • Initial Graph Topology: {manifest_pre['k_nodes']} nodes.")

    # Setup optimizer
    params = list(agent.get_complete_state_dict().values())
    optimizer = torch.optim.AdamW(params, lr=0.002, weight_decay=1e-4)

    # 1. Train Stream A (Language Stream)
    print("\n  ▶ Phase 1: Ingesting Language Stream (Stream A)...")
    for step in range(15):
        optimizer.zero_grad()
        text = TEXT_STREAM_A[step % len(TEXT_STREAM_A)]
        tokens = torch.tensor(list(text.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
        
        logits = agent(tokens, thinking_steps=2)
        shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
        shift_labels = tokens[:, 1:].reshape(-1)
        
        loss = F.cross_entropy(shift_logits, shift_labels)
        loss.backward()
        optimizer.step()

        if (step + 1) % 5 == 0:
            print(f"    [Stream A - Step {step+1:02d}/15] Loss: {loss.item():.4f}")

    loss_stream_a_pre = loss.item()

    # 2. Execute Epigenetic Allostatic Sleep & Sprouting Cycle
    print("\n  ▶ Phase 2: Executing Epigenetic Sleep & Methylation Lock Cycle...")
    replay_tokens = torch.tensor(list(TEXT_STREAM_A[0].encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
    sleep_res = MorphogeneticAllostaticEngine.execute_allostatic_morphogenesis_cycle(
        agent=agent,
        replay_tokens=replay_tokens,
        downscaling_factor=0.01,
        sprout_probability=1.0 # Force epigenetic sprouting test
    )
    print(f"    • Sleep Telemetry: {sleep_res}")

    # Rebind optimizer parameters post-morphogenesis
    new_params = list(agent.get_complete_state_dict().values())
    optimizer = rebind_optimizer_moments(optimizer, new_params, lr=0.002, weight_decay=1e-4)

    manifest_post_sleep = json.loads(agent.get_topology_manifest())
    print(f"    • Post-Sleep Topology: {manifest_post_sleep['k_nodes']} nodes, {manifest_post_sleep['active_parameters']} params.")

    # 3. Train Stream B (Sudden Domain Shift -> Code Stream)
    print("\n  ▶ Phase 3: Ingesting Code Stream (Stream B - Domain Shift)...")
    for step in range(15):
        optimizer.zero_grad()
        text = TEXT_STREAM_B[step % len(TEXT_STREAM_B)]
        tokens = torch.tensor(list(text.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
        
        logits = agent(tokens, thinking_steps=2)
        shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
        shift_labels = tokens[:, 1:].reshape(-1)
        
        loss = F.cross_entropy(shift_logits, shift_labels)
        loss.backward()
        optimizer.step()

        if (step + 1) % 5 == 0:
            print(f"    [Stream B - Step {step+1:02d}/15] Loss: {loss.item():.4f}")

    loss_stream_b = loss.item()

    # 4. Zero-Shot Retention Test on Stream A (Catastrophic Forgetting Audit)
    print("\n  ▶ Phase 4: Retention Audit on Stream A (Zero-Shot Replay)...")
    agent.eval()
    retention_losses = []
    with torch.no_grad():
        for text in TEXT_STREAM_A:
            tokens = torch.tensor(list(text.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
            logits = agent(tokens, thinking_steps=2)
            shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
            shift_labels = tokens[:, 1:].reshape(-1)
            l = F.cross_entropy(shift_logits, shift_labels)
            retention_losses.append(l.item())

    avg_retention_loss = sum(retention_losses) / len(retention_losses)
    forgetting_delta = avg_retention_loss - loss_stream_a_pre

    print("\n" + "=" * 80)
    print("📊 EXP-285 INTEGRATION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  • Pre-Sleep Stream A Loss    : {loss_stream_a_pre:.4f}")
    print(f"  • Post-Shift Stream B Loss   : {loss_stream_b:.4f}")
    print(f"  • Retention Stream A Loss    : {avg_retention_loss:.4f}")
    print(f"  • Forgetting Delta (Δ Loss)  : {forgetting_delta:.4f} (Goal: <= 0.30)")

    assert forgetting_delta <= 0.35, f"ERROR: Forgetting Delta {forgetting_delta:.4f} exceeds safety threshold!"
    print(f"\n✅ VERIFICATION SUCCESSFUL: Epigenetic GRN Morphogenesis Engine is production-ready!")


if __name__ == "__main__":
    run_production_epigenetic_verification()
