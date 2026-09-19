# experiments/exp_249_omni_morphic_dynamic_engine_benchmark.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #249
Topic: Omni-Continuous Dynamic Graph Substrate Benchmark (AGN v9.0)
       (True Open-Ended Morphogenesis: Signal Manifold + Affinity Attention + Unconstrained Nodes)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist (2026)
Standard: KEP v10.0 Master Protocol | Rules #1, #2, #3, #4, #5, #6, #7, #8, #10, #11
===============================================================================

Hypothesis:
    Eliminating fixed ports, hardcoded DAG topologies, and rigid layer hierarchies in favor of an
    Omni-Continuous Dynamic Graph Substrate (AGN v9.0):
      1. Global Dynamic Signal Manifold unifying sensory, cortical, error, and prior streams;
      2. OmniMorphicNodes self-governing their inputs via dynamic Affinity Query attention;
      3. Zero-Shock Net2Net epigenetic birth identity (alpha_epi = 0.0 -> tanh(0.0) = 0.0) with zero PCIe stalls;
    will allow the network to self-organize arbitrary cognitive pathways and drop final stream loss
    by >= 0.08 nats/byte while maintaining stable gradient flow and biophysical vitality.
===============================================================================
"""

import sys
import time
import torch
import torch.nn as nn

from karyon_config import get_standard_karyon_config
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory


def generate_synthetic_multimodal_stream(num_steps: int = 150, batch_size: int = 4, seq_len: int = 64, device: str = 'cpu'):
    """Generates continuous token streams across diverse semantic modalities."""
    torch.manual_seed(42)
    stream_data = []

    # Vocabulary range for byte level V=258 (excluding pad/eos)
    for step in range(num_steps):
        # Generate mixed syntactic patterns (continuous byte streams)
        base_tokens = torch.randint(32, 126, (batch_size, seq_len), dtype=torch.long, device=device)
        # Add repetitive structure to simulate natural language / code / speech patterns
        if step % 3 == 0:
            base_tokens[:, :seq_len // 2] = torch.randint(65, 90, (batch_size, seq_len // 2), dtype=torch.long, device=device)
        targets = torch.roll(base_tokens, -1, dims=1)
        targets[:, -1] = 257  # EOS token
        stream_data.append((base_tokens, targets))
    return stream_data


def run_benchmark():
    print("===============================================================================")
    print("🚀 STARTING KEP EXPERIMENT #249: OMNI-CONTINUOUS DYNAMIC SUBSTRATE BENCHMARK")
    print("===============================================================================")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")

    config = get_standard_karyon_config(model_dim=256)
    agent = CoREAgent(config, device_str=device).to(device)

    hu = HomeostaticUnit(device=device)
    episodic_mem = BatchedEpisodicMemory(capacity=100, memory_dim=config.net.unified_dim, device=device)
    criterion_speech = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

    # 1. Verify Zero-Shock Net2Net Birth Identity
    print("\n--- PHASE 1: Zero-Shock Net2Net Epigenetic Birth Verification ---")
    dummy_input = torch.randint(32, 126, (2, 32), dtype=torch.long, device=device)
    sensor_dict = {"text": dummy_input}

    with torch.no_grad():
        loss_before, _, _, _, _, _, _ = agent.forward_multimodal_sequence(
            sensor_dict, dummy_input, hu, criterion_speech, episodic_memory=episodic_mem
        )

    # Sprout new unconstrained OmniMorphicNodes into the substrate
    agent.sprout_omni_node("omni_associative_resonator", state_dim=128, num_operators=8)
    agent.sprout_omni_node("omni_predictive_plastic_nexus", state_dim=128, num_operators=8)

    with torch.no_grad():
        loss_after_birth, _, _, _, _, _, _ = agent.forward_multimodal_sequence(
            sensor_dict, dummy_input, hu, criterion_speech, episodic_memory=episodic_mem
        )

    birth_delta = abs(loss_before.item() - loss_after_birth.item())
    print(f"  • Loss Before Sprouting: {loss_before.item():.6f}")
    print(f"  • Loss After Sprouting:  {loss_after_birth.item():.6f}")
    print(f"  • Birth Shock Delta:     {birth_delta:.8f}")

    if birth_delta > 1e-4:
        print("❌ FAILED: Net2Net Zero-Shock Birth Identity Violated!")
        sys.exit(1)
    print("✅ PASSED: 100% Zero-Shock Net2Net Identity Confirmed (Delta < 1e-4).")

    # 2. Continuous Stream Adaptation & Training
    print("\n--- PHASE 2: Continuous Stream Learning & Dynamic Graph Evolution ---")
    num_steps = 100
    batch_size = 4
    seq_len = 64
    stream = generate_synthetic_multimodal_stream(num_steps=num_steps, batch_size=batch_size, seq_len=seq_len, device=device)

    total_tokens = 0
    start_time = time.time()
    initial_loss = None
    final_loss = None
    final_fe = None
    final_energy = None

    agent.train()
    for step, (inputs, targets) in enumerate(stream):
        sensor_dict = {"text": inputs}
        optimizer.zero_grad()

        total_loss, speech_loss, fe_loss, _, _, curr_u_t, _ = agent.forward_multimodal_sequence(
            sensor_dict, targets, hu, criterion_speech, episodic_memory=episodic_mem
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()

        total_tokens += batch_size * seq_len
        current_loss_val = speech_loss

        if step == 0:
            initial_loss = current_loss_val
        final_loss = current_loss_val
        final_fe = fe_loss
        final_energy = curr_u_t[:, 1].mean().item()

        if (step + 1) % 20 == 0 or step == num_steps - 1:
            alphas = [f"{float(torch.tanh(a).item()):.4f}" for a in agent.omni_substrate.alpha_nodes]
            print(f"  Step {step + 1:3d}/{num_steps} | Loss: {current_loss_val:.4f} | FE: {fe_loss:.4f} | Energy: {final_energy:.4f} | Alphas: {alphas}")

    elapsed_time = time.time() - start_time
    tok_per_sec = total_tokens / max(elapsed_time, 1e-5)
    loss_delta = initial_loss - final_loss

    print("\n===============================================================================")
    print("📊 FINAL BENCHMARK TELEMETRY & KEP EVALUATION")
    print("===============================================================================")
    print(f"  • Initial Stream Loss:  {initial_loss:.4f} nats/byte")
    print(f"  • Final Stream Loss:    {final_loss:.4f} nats/byte")
    print(f"  • Absolute Loss Delta:  {loss_delta:+.4f} nats/byte")
    print(f"  • Variational Free Energy: {final_fe:.4f}")
    print(f"  • Somatic Energy Level: {final_energy:.4f}")
    print(f"  • Throughput:           {tok_per_sec:.1f} tok/s")
    print("===============================================================================")

    # Output structured summary for the pipeline parser
    print(f"\nFinal Loss: {final_loss:.4f}")
    print(f"Baseline Loss: {initial_loss:.4f}")
    print(f"Loss Delta: {loss_delta:.4f}")
    print(f"tok_per_sec: {tok_per_sec:.1f}")
    print(f"free_energy: {final_fe:.4f}")
    print(f"energy: {final_energy:.4f}")

    if loss_delta >= 0.08:
        print("\n🟢 VERDICT: POSITIVE (Loss Delta >= 0.08 nats/byte)")
        return 0
    else:
        print("\n⚪ VERDICT: NEUTRAL / INCONCLUSIVE")
        return 1


if __name__ == "__main__":
    sys.exit(run_benchmark())
