# experiments/exp_132_fastmap_pac_synthesis.py
"""
===============================================================================
EXP-132: FAST-MAPPING EPISODIC RECALL & ENTROPY-ADAPTIVE PAC DECODING
===============================================================================
Hypothesis:
1. Fast-Mapping 1-Shot Episodic Retrieval:
   When user provides a new factual statement in conversation,
   it is immediately projected via episodic_sensory_proj into BatchedEpisodicMemory (dim=256).
   Fixing C++ memory active slot tracking ensures instant 100% cosine similarity retrieval.
2. Context-Aware Episodic Projections in Autoregressive Generation:
   Querying episodic memory using rolling context embeddings (sliding window)
   under noradrenergic phasic arousal (NA_t > 0.15) modulates sensory gating towards the target fact.
3. Entropy-Adaptive PAC Decoding:
   Morphemic boundaries (H > 0.70) explore diverse tokens, while intra-word steps (H <= 0.50)
   enforce MAP precision, eliminating sub-syllable loops.

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
    print("STARTING EXP-132: FAST-MAPPING EPISODIC RECALL & ENTROPY-ADAPTIVE PAC DECODING")
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
        # Ensure memory max_active_cpu is synchronized with restored size
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
        fact_proj = agent.episodic_sensory_proj(fact_embs.mean(dim=1)) # [1, unified_dim]
        
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

    # 3. Autoregressive Generation with Fast-Mapping Memory Injection
    prompt = "User: Where is Karyon's internal cognitive core implemented?\nKaryon:"
    print(f"\n[Phase 2: Generation Benchmark] Prompt:\n  '{prompt}'")

    # High Noradrenaline (Arousal NA = 0.90) to maximize memory gateway gain
    hu.state[0, 4] = 0.90

    gen_stream = agent.generate_thought_and_speech(
        prompt,
        m_state=torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device),
        h_state=torch.zeros(1, agent.hidden_dim, device=device),
        hu=hu,
        episodic_memory=memory,
        config=config,
        max_generated_tokens=100,
        temperature=0.45,
        top_p=0.90
    )

    chars = []
    for ev in gen_stream:
        if ev["status"] == "token":
            chars.append(ev["text"])

    output_text = "".join(chars)
    print(f"\nGenerated Output:\n{output_text}")

    # 4. Compute 3-Gram Lexical Diversity
    def n_gram_diversity(text, n=3):
        if len(text) < n:
            return 1.0
        ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
        return len(set(ngrams)) / float(len(ngrams))

    div_score = n_gram_diversity(output_text)
    print(f"\n3-Gram Lexical Diversity: {div_score:.4f}")

    metrics = {
        "memory_active_slots": memory.size.item(),
        "direct_retrieval_similarity": direct_sim,
        "3gram_lexical_diversity": div_score,
        "noradrenaline_arousal": hu.state[0, 4].item()
    }

    verdict = "🟢 POSITIVE" if (direct_sim > 0.85 and div_score > 0.40) else "⚪ NEUTRAL"
    print(f"\n[EXP-132 VERDICT]: {verdict}")
    print("=" * 80)

    return metrics, verdict

if __name__ == "__main__":
    run_experiment()
