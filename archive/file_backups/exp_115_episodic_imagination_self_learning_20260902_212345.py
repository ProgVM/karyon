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

def generate_imagined_trajectory(agent: CoREAgent, hu: HomeostaticUnit, episodic_mem: BatchedEpisodicMemory, batch_size: int, seq_len: int):
    """
    Generates an internally imagined thought sequence seeded from episodic memory
    and steered by Panksepp SEEKING drive and LC phasic neural gain.
    """
    device = agent.device
    with torch.no_grad():
        affective_state = agent.affective_core.compute_affective_state(hu.state)
        seeking = affective_state["panksepp"]["SEEKING"]
        na_t = hu.state[:, 4:5]
        lc_gain = agent.lc_gain(na_t) # (B, 1) in (0, 1)

        # 1. Sample cognitive query vector from high-level latent attractor space
        u_t = hu.state
        q_imagination = agent.episodic_sensory_proj(agent.pos_embeddings.byte_embed.weight[32:96].mean(dim=0, keepdim=True).expand(batch_size, -1))
        
        # 2. Retrieve episodic anchor if memory has active traces
        active_slots = getattr(episodic_mem, 'max_active_cpu', 0)
        if active_slots > 0:
            ret_trace, _ = episodic_mem.read(q_imagination, temperature=0.08, threshold=0.30, sigmoid_beta=10.0)
            # Modulate seed by LC phasic gain
            seed_latent = ret_trace * (0.5 + 0.5 * lc_gain)
            # Project seed latent to byte token space
            seed_logits = seed_latent @ agent.pos_embeddings.byte_embed.weight.t()
            seed_token_ids = torch.argmax(seed_logits[:, 32:126], dim=-1) + 32 # ensure valid ASCII range
        else:
            # Fallback to structured semantic seed
            seed_token_ids = torch.randint(65, 90, (batch_size,), device=device)

        # 3. Autoregressively roll out short imagination sequence using Top-p nucleus sampling
        tokens = [seed_token_ids.unsqueeze(1)] # (B, 1)
        m_s1 = torch.zeros(batch_size, agent.num_heads, agent.head_k, agent.head_v, device=device)
        m_s2 = torch.zeros(batch_size, agent.num_heads, agent.head_k, agent.head_v, device=device)
        
        # Rapid short rollout of length seq_len
        curr_token = seed_token_ids.unsqueeze(1)
        for t in range(seq_len):
            emb = agent.pos_embeddings(curr_token, start_pos=t, apply_rf=False)
            h_in = agent.in_proj(emb)
            h_s1, m_s1, _ = agent.stage1(h_in, m_s1, u_t, torch.Tensor(), 1.0)
            sal_gate = agent.boundary_detector(h_s1, curr_token)
            e1, _, _ = agent.pw_lper(h_s1, torch.zeros_like(h_s1), u_t)
            h_s2, m_s2, _ = agent.stage2(e1, m_s2, u_t, sal_gate, 1.0)
            
            topdown = agent.topdown_prior_proj(h_s2)
            h_comb = h_s1 + h_s2 + (0.10 + 0.15 * lc_gain.unsqueeze(1)) * topdown
            h_flat = h_comb.view(batch_size, -1)
            
            h_rel, _ = agent.attractor_head.relax_to_minima(h_flat, u_t)
            logits = agent.volitional_head.compute_volitional_logits(h_rel, u_t, agent.pos_embeddings.byte_embed.weight)
            
            # Continuous temperature via SEEKING drive & LC Gain
            temp = 0.35 + 0.30 * (1.0 - seeking)
            probs = torch.softmax(logits[:, 32:127] / temp, dim=-1)
            next_sub = torch.multinomial(probs, num_samples=1)
            next_tok = next_sub + 32
            tokens.append(next_tok)
            curr_token = next_tok
            
        imagined_seq = torch.cat(tokens, dim=1) # (B, seq_len + 1)
        return imagined_seq

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
