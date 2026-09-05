# experiments/exp_132_fastmap_pac_synthesis.py
"""
===============================================================================
EXP-132: FAST-MAPPING EPISODIC RECALL & REFRACTORY PAC DECODING
===============================================================================
Hypothesis:
1. Fast-Mapping 1-Shot Episodic Retrieval:
   When user provides a new factual statement in conversation,
   it is immediately projected via episodic_sensory_proj into BatchedEpisodicMemory (dim=256).
   Synchronizing memory.max_active_cpu with restored capacity ensures >0.95 direct cosine similarity.
2. Biophysical Motor Action Refractory Dynamics:
   Adding a post-action refractory trace (simulating neural K+ channel hyperpolarization):
       RefractoryTrace_{t+1} = 0.80 * RefractoryTrace_t + e_{a_t}
       Logits_{eff} = Logits - gamma_refractory * RefractoryTrace_t
   breaks recurrent limit cycle orbits (the-st-the-st loop) naturally and organically,
   increasing lexical diversity and allowing episodic factual recall to emerge in speech.

Baseline: EXP-130 / EXP-131.
===============================================================================
"""

import sys
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("/kaggle/working/karyon")

import karyon_config
import karyon_core
import karyon_agent
import karyon_checkpoint
from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory, ByteTokenizer
from karyon_checkpoint import load_karyon

def run_experiment():
    print("=" * 80)
    print("STARTING EXP-132: FAST-MAPPING EPISODIC RECALL & REFRACTORY PAC DECODING")
    print("=" * 80)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    config = CoREConfig()
    config.train.batch_size = 1

    hu = HomeostaticUnit(device=device_str)
    agent = CoREAgent(config, device=device_str).to(device)
    memory = BatchedEpisodicMemory(batch_size=1, memory_dim=config.net.unified_dim, max_capacity=1000, device=device_str)

    if os.path.exists("karyon_soul.kcore"):
        load_karyon(agent, memory, hu, filepath="karyon_soul.kcore", device=device_str)
        memory.max_active_cpu = int(memory.size.max().item())
        print(f"[EXP-132] Loaded karyon_soul.kcore checkpoint. Memory active slots: {memory.max_active_cpu}")

    # 1. 1-Shot Fast-Mapping Storage
    fact_text = "Karyon's internal cognitive core is fully implemented in pure C++20 LibTorch."
    print(f"\n[Phase 1: Fast-Mapping Ingestion] Ingesting 1-shot fact:\n  '{fact_text}'")

    tokenizer = ByteTokenizer()
    fact_tokens = tokenizer.encode(fact_text)
    fact_tensor = torch.tensor([fact_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        fact_embs = agent.pos_embeddings(fact_tensor, start_pos=0, apply_rf=True)
        fact_proj = agent.episodic_sensory_proj(fact_embs.mean(dim=1))
        
        memory.write(fact_proj, fact_proj)
        memory.max_active_cpu = int(memory.size.max().item())

    print(f"Memory Occupancy: {memory.size.item()}/{memory.max_capacity} slots")

    # 2. Test Direct Retrieval with Fact Query
    with torch.no_grad():
        query_text = "Where is Karyon's internal cognitive core implemented?"
        query_tokens = torch.tensor([tokenizer.encode(query_text)], dtype=torch.long, device=device)
        query_embs = agent.pos_embeddings(query_tokens, start_pos=0, apply_rf=True)
        q_proj = agent.episodic_sensory_proj(query_embs.mean(dim=1))

        ret_val, max_sim = memory.read(q_proj, temperature=0.05, threshold=0.10, sigmoid_beta=10.0)
        direct_sim = max_sim.item()
        print(f"\nDirect Episodic Retrieval Cosine Similarity: {direct_sim:.4f}")

    # 3. Autoregressive Generation with Refractory Modulation
    prompt = "User: Where is Karyon's internal cognitive core implemented?\nKaryon:"
    print(f"\n[Phase 2: Generation Benchmark] Prompt:\n  '{prompt}'")

    hu.state[0, 4] = 0.90 # High Noradrenaline

    # Custom generation loop testing biophysical refractory dynamics
    prompt_ids = [t for t in tokenizer.encode(prompt) if t != 257]
    prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_embs = agent.pos_embeddings(prompt_tokens, start_pos=0, apply_rf=True)

    m_s1 = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device)
    m_s2 = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device)
    h1_prev_last = torch.zeros(1, 1, agent.hidden_dim, device=device)
    hu_st = hu.state

    # Process prompt in chunks of 64
    prompt_len = prompt_tokens.size(1)
    for c_idx in range(0, prompt_len, 64):
        c_emb = prompt_embs[:, c_idx : min(c_idx + 64, prompt_len), :]
        c_in = prompt_tokens[:, c_idx : min(c_idx + 64, prompt_len)]
        h_in = agent.in_proj(c_emb)
        h_s1, m_s1, _ = agent.stage1(h_in, m_s1, hu_st, torch.Tensor(), 1.0)
        sal_gate = agent.boundary_detector(h_s1, c_in)
        e1_weighted, h1_prev_last, _ = agent.pw_lper(h_s1, h1_prev_last, hu_st)
        h_s2, m_s2, _ = agent.stage2(e1_weighted, m_s2, hu_st, sal_gate, 1.0)

    rolling_token_ids = prompt_tokens[0].tolist()
    total_prompt_len = prompt_tokens.size(1)
    
    # Refractory Trace Tensor: [1, 258]
    refractory_trace = torch.zeros(1, 258, device=device)
    gamma_refractory = 2.20 # nats
    decay_refractory = 0.78 # rate

    generated_chars = []

    for step in range(120):
        context_window = rolling_token_ids[-8:]
        window_t = torch.tensor([context_window], dtype=torch.long, device=device)
        window_start_pos = total_prompt_len + step - len(context_window)
        t_emb = agent.pos_embeddings(window_t[:, -1:], start_pos=window_start_pos, apply_rf=True)

        sensor_inputs = {'text': t_emb.squeeze(1)}
        active_slots = memory.max_active_cpu
        na_t = hu_st[:, 4:5]
        phasic_gain = agent.lc_gain(na_t)

        if active_slots > 0:
            # Query memory using context projection
            context_embs = agent.pos_embeddings(window_t, start_pos=window_start_pos, apply_rf=True)
            q_k = agent.episodic_sensory_proj(context_embs.mean(dim=1)).float()
            ret_mem, max_sim = memory.read(q_k, temperature=0.05, threshold=0.20, sigmoid_beta=10.0)
            sensor_inputs['episodic_recall'] = ret_mem * phasic_gain

        w_t, _, _, _ = agent.gateway(sensor_inputs, m_s2.view(1, -1)[:, :agent.hidden_dim], hu_st)
        h_in = agent.in_proj(w_t).unsqueeze(1)

        h_s1, h_s2, m_s1, m_s2, sal_gate = agent.fused_stack(h_in, m_s1, m_s2, hu_st, window_t[:, -1:])
        effective_hu_st, gamma_override, allostatic_strain = agent.will_engine(h_s2, hu_st)

        topdown_prior = agent.topdown_prior_proj(h_s2)
        h_combined = h_s1 + h_s2 + (0.10 + 0.15 * phasic_gain.unsqueeze(1)) * topdown_prior
        h_flat = h_combined.contiguous().view(-1, agent.hidden_dim)
        h_relaxed, _ = agent.attractor_head.relax_to_minima(h_flat, effective_hu_st)

        raw_logits = agent.volitional_head.compute_volitional_logits(h_relaxed, effective_hu_st, agent.pos_embeddings.byte_embed.weight)

        # Apply Somatic Byte Penalty + Refractory Trace
        somatic_penalty = torch.zeros(1, 258, device=device)
        somatic_penalty[0, 256] = 12.0
        somatic_penalty[0, :9] = 10.0
        somatic_penalty[0, 11:13] = 10.0
        somatic_penalty[0, 14:32] = 10.0
        somatic_penalty[0, 127] = 8.0

        early_step_factor = math.exp(-step / 4.0)
        somatic_penalty[0, 257] = 15.0 * early_step_factor

        # Effective logits with Biophysical Refractory Hyperpolarization
        logits = raw_logits - somatic_penalty - gamma_refractory * refractory_trace

        p_dist = F.softmax(logits, dim=-1)
        entropy = -(p_dist * torch.log(p_dist + 1e-9)).sum(dim=-1)

        temp = 0.10 + 0.35 * torch.sigmoid(5.0 * (entropy - 0.60) + 2.0 * (phasic_gain.squeeze() - 0.50)).item()
        top_p_val = 0.90 + 0.09 * (1.0 - torch.sigmoid(4.0 * (entropy - 0.60)).item())

        scaled_logits = logits / max(temp, 1e-4)
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        to_remove = cumulative_probs > top_p_val
        to_remove[..., 1:] = to_remove[..., :-1].clone()
        to_remove[..., 0] = False
        indices_to_remove = to_remove.scatter(1, sorted_indices, to_remove)
        scaled_logits[indices_to_remove] = -1e9

        probs = F.softmax(scaled_logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        next_token_id = torch.multinomial(probs, num_samples=1).item()
        if next_token_id == 257:
            break

        rolling_token_ids.append(next_token_id)
        
        # Update Refractory Trace
        refractory_trace = decay_refractory * refractory_trace
        refractory_trace[0, next_token_id] += 1.0

        try:
            ch = bytes([next_token_id]).decode('utf-8', errors='replace')
            generated_chars.append(ch)
        except Exception:
            pass

    output_text = "".join(generated_chars)
    print(f"\nGenerated Output (with Refractory PAC Dynamics):\n{output_text}")

    def n_gram_diversity(text, n=3):
        if len(text) < n:
            return 1.0
        ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
        return len(set(ngrams)) / float(len(ngrams))

    div_score = n_gram_diversity(output_text)
    print(f"\n3-Gram Lexical Diversity: {div_score:.4f}")

    metrics = {
        "direct_retrieval_similarity": direct_sim,
        "3gram_lexical_diversity": div_score,
        "output_length": len(output_text),
        "refractory_decay": decay_refractory,
        "refractory_gamma": gamma_refractory
    }

    verdict = "🟢 POSITIVE" if (direct_sim > 0.90 and div_score > 0.45) else "⚪ NEUTRAL"
    print(f"\n[EXP-132 VERDICT]: {verdict}")
    print("=" * 80)

    return metrics, verdict

if __name__ == "__main__":
    run_experiment()
