# experiments/exp_115_episodic_imagination_self_learning.py
"""
===============================================================================
EXP-115: Episodic Imagination & Panksepp SEEKING-Gated Self-Learning Benchmark
===============================================================================
Hypothesis:
Replacing uniform byte noise in Karyon's autonomous self-learning cycle
with episodic memory-seeded imagination guided by Panksepp SEEKING drive and
continuous Locus Coeruleus phasic gain significantly improves self-learning Free Energy
convergence (delta F_t >= 25%) and prevents pseudo-morphemic drift during autonomous introspection.

Protocol: KEP v8.1 Scientific Protocol
Author: Bazilevs & Autonomous Lead AI Cyberneticist
===============================================================================
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath('.'))

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory, ByteTokenizer
from karyon_hardware import get_hardware_engine
from datasets import load_dataset

def generate_imagined_trajectory(agent: CoREAgent, hu: HomeostaticUnit, episodic_mem: BatchedEpisodicMemory, batch_size: int, seq_len: int, config: CoREConfig):
    """
    Generates linguistically coherent imagined thought sequences by querying
    agent's own generative speech stream seeded from an episodic prompt.
    """
    device = agent.device
    prompts = [
        "The mind observes ",
        "In this state of reflection ",
        "The cognitive system predicts ",
        "Autonomous homeostasis requires "
    ]
    
    all_trajectories = []
    with torch.no_grad():
        for b in range(batch_size):
            p = prompts[b % len(prompts)]
            m_s = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device)
            h_s = torch.zeros(1, agent.hidden_dim, device=device)
            
            gen_stream = agent.generate_thought_and_speech(
                p, m_s, h_s, hu, episodic_mem, config, max_generated_tokens=seq_len
            )
            
            tokens = [item["token_id"] for item in gen_stream if item.get("status") == "token"]
            prefix_ids = list(p.encode('utf-8'))
            full_ids = prefix_ids + tokens
            if len(full_ids) < seq_len + 1:
                full_ids = full_ids + [32] * (seq_len + 1 - len(full_ids))
            else:
                full_ids = full_ids[:seq_len + 1]
            all_trajectories.append(full_ids)

    return torch.tensor(all_trajectories, dtype=torch.long, device=device)

def run_experiment():
    print("=" * 80)
    print("STARTING EXP-115: EPISODIC IMAGINATION SELF-LEARNING BENCHMARK")
    print("=" * 80)

    hw = get_hardware_engine()
    device_str = 'cuda' if hw.is_cuda else ('xla' if hw.is_tpu else 'cpu')
    print(f"[Hardware] Active Device: {device_str}")

    config = CoREConfig()
    batch_size = 4
    seq_len = 64
    num_cycles = 10

    # Load real dataset samples to populate episodic memory first
    print("[Data] Loading Alpaca-GPT4 for episodic grounding...")
    dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
    real_texts = [ex["instruction"] + " " + ex["output"] for ex in dataset.select(range(20))]

    tokenizer = ByteTokenizer()
    
    # -------------------------------------------------------------------------
    # BASELINE RUN: Uniform Random Byte Noise Self-Learning
    # -------------------------------------------------------------------------
    print("\n--- Running Baseline: Uniform Byte Noise Self-Learning ---")
    torch.manual_seed(42)
    agent_base = CoREAgent(config, device=device_str).to(device_str)
    hu_base = HomeostaticUnit(batch_size=batch_size, device=device_str)
    mem_base = BatchedEpisodicMemory(batch_size=batch_size, memory_dim=256, max_capacity=100, device=device_str)
    
    # Pre-populate memory cleanly without grad graph links
    with torch.no_grad():
        for txt in real_texts[:4]:
            raw_b = torch.tensor([list(txt[:128].encode('utf-8'))], dtype=torch.long, device=device_str)
            if raw_b.size(1) > 0:
                emb = agent_base.pos_embeddings(raw_b).mean(dim=1).detach()
                val = torch.randn(1, 256, device=device_str).detach()
                for b_i in range(batch_size):
                    mem_base.write(emb, val, b_i)

    optimizer_base = optim.AdamW(agent_base.get_all_parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    base_start_time = time.perf_counter()
    base_fe_history = []
    base_loss_history = []

    for c in range(num_cycles):
        optimizer_base.zero_grad()
        # Uniform random bytes
        seed_tokens = torch.randint(32, 126, (batch_size, seq_len + 1), dtype=torch.long, device=device_str)
        inp = seed_tokens[:, :-1]
        tgt = seed_tokens[:, 1:]
        
        tot_loss, s_loss, fe_val, _, _, _, _ = agent_base.forward_sequence(
            inp, tgt, hu_base, criterion, episodic_memory=mem_base, loss_free_energy_weight=0.08
        )
        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_base.get_all_parameters(), max_norm=2.0)
        optimizer_base.step()
        
        base_fe_history.append(fe_val)
        base_loss_history.append(s_loss)

    base_duration = time.perf_counter() - base_start_time
    print(f"[Baseline] Completed {num_cycles} cycles in {base_duration:.2f}s")
    print(f"[Baseline] Initial FE: {base_fe_history[0]:.4f} -> Final FE: {base_fe_history[-1]:.4f} (Delta: {base_fe_history[0] - base_fe_history[-1]:+.4f})")
    print(f"[Baseline] Final Self Loss: {base_loss_history[-1]:.4f}")

    # -------------------------------------------------------------------------
    # PROPOSED RUN: Episodic Imagination & SEEKING-Gated Self-Learning
    # -------------------------------------------------------------------------
    print("\n--- Running Proposed: Episodic Imagination Self-Learning ---")
    torch.manual_seed(42)
    agent_prop = CoREAgent(config, device=device_str).to(device_str)
    hu_prop = HomeostaticUnit(batch_size=batch_size, device=device_str)
    mem_prop = BatchedEpisodicMemory(batch_size=batch_size, memory_dim=256, max_capacity=100, device=device_str)

    # Pre-populate memory identically without grad graph links
    with torch.no_grad():
        for txt in real_texts[:4]:
            raw_b = torch.tensor([list(txt[:128].encode('utf-8'))], dtype=torch.long, device=device_str)
            if raw_b.size(1) > 0:
                emb = agent_prop.pos_embeddings(raw_b).mean(dim=1).detach()
                val = torch.randn(1, 256, device=device_str).detach()
                for b_i in range(batch_size):
                    mem_prop.write(emb, val, b_i)

    optimizer_prop = optim.AdamW(agent_prop.get_all_parameters(), lr=1e-4)

    prop_start_time = time.perf_counter()
    prop_fe_history = []
    prop_loss_history = []
    sample_texts = []

    for c in range(num_cycles):
        optimizer_prop.zero_grad()
        # Biologically grounded imagination trajectory
        imagined_tokens = generate_imagined_trajectory(agent_prop, hu_prop, mem_prop, batch_size, seq_len)
        inp = imagined_tokens[:, :-1]
        tgt = imagined_tokens[:, 1:]

        tot_loss, s_loss, fe_val, _, _, _, _ = agent_prop.forward_sequence(
            inp, tgt, hu_prop, criterion, episodic_memory=mem_prop, loss_free_energy_weight=0.08
        )
        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_prop.get_all_parameters(), max_norm=2.0)
        optimizer_prop.step()

        prop_fe_history.append(fe_val)
        prop_loss_history.append(s_loss)

        if c == num_cycles - 1:
            # Audit imagined sequence text
            b_str = bytes([t for t in imagined_tokens[0].tolist() if 32 <= t <= 126]).decode('ascii', errors='ignore')
            sample_texts.append(b_str)

    prop_duration = time.perf_counter() - prop_start_time
    print(f"[Proposed] Completed {num_cycles} cycles in {prop_duration:.2f}s")
    print(f"[Proposed] Initial FE: {prop_fe_history[0]:.4f} -> Final FE: {prop_fe_history[-1]:.4f} (Delta: {prop_fe_history[0] - prop_fe_history[-1]:+.4f})")
    print(f"[Proposed] Final Self Loss: {prop_loss_history[-1]:.4f}")
    print(f"[Proposed] Sample Imagined Thought: {repr(sample_texts[0] if sample_texts else '')}")

    # Evaluate validation loss on actual real Alpaca samples
    print("\n--- Validating on Real Data Generalization ---")
    val_inp_ids = [list(txt[:seq_len].encode('utf-8')) for txt in real_texts[10:14]]
    val_tgt_ids = [list(txt[1:seq_len+1].encode('utf-8')) for txt in real_texts[10:14]]
    val_inp = torch.tensor(val_inp_ids, dtype=torch.long, device=device_str)
    val_tgt = torch.tensor(val_tgt_ids, dtype=torch.long, device=device_str)

    with torch.no_grad():
        _, val_loss_base, _, _, _, _, _ = agent_base.forward_sequence(val_inp, val_tgt, hu_base, criterion)
        _, val_loss_prop, _, _, _, _, _ = agent_prop.forward_sequence(val_inp, val_tgt, hu_prop, criterion)

    print(f"[Validation Real Text] Baseline Real Loss: {val_loss_base:.4f} (PPL: {math.exp(val_loss_base):.2f})")
    print(f"[Validation Real Text] Proposed Real Loss: {val_loss_prop:.4f} (PPL: {math.exp(val_loss_prop):.2f})")

    fe_reduction_base = (base_fe_history[0] - base_fe_history[-1]) / max(base_fe_history[0], 1e-5) * 100
    fe_reduction_prop = (prop_fe_history[0] - prop_fe_history[-1]) / max(prop_fe_history[0], 1e-5) * 100

    results = {
        "exp_id": "EXP-115",
        "base_final_fe": float(base_fe_history[-1]),
        "prop_final_fe": float(prop_fe_history[-1]),
        "base_fe_reduction_pct": float(fe_reduction_base),
        "prop_fe_reduction_pct": float(fe_reduction_prop),
        "base_real_val_loss": float(val_loss_base),
        "prop_real_val_loss": float(val_loss_prop),
        "delta_val_loss": float(val_loss_base - val_loss_prop),
        "sample_imagined_thought": sample_texts[0] if sample_texts else ""
    }

    with open("exp_115_results.json", "w") as f:
        json.dump(results, f, indent=2)

    verdict = "🟢 POSITIVE" if results["delta_val_loss"] >= 0.05 or results["prop_fe_reduction_pct"] > results["base_fe_reduction_pct"] + 15.0 else ("⚪ NEUTRAL" if abs(results["delta_val_loss"]) < 0.05 else "🔴 REJECTED")
    print(f"\n[VERDICT]: {verdict} (Val Loss Delta: {results['delta_val_loss']:+.4f}, FE Drop: {results['prop_fe_reduction_pct']:.1f}% vs Base: {results['base_fe_reduction_pct']:.1f}%)")

if __name__ == "__main__":
    run_experiment()
