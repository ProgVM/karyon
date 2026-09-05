# experiments/exp_132_fastmap_pac_synthesis.py
"""
===============================================================================
EXP-132: FAST-MAPPING EPISODIC RECALL & ENTROPY-ADAPTIVE PAC DECODING
===============================================================================
Hypothesis:
1. Fast-Mapping 1-Shot Episodic Retrieval:
   When user provides a new factual statement in conversation ("Fact: Karyon's core is located in C++20"),
   it is immediately encoded into BatchedEpisodicMemory with high cosine orthogonality.
   During subsequent generation, high noradrenergic arousal (NA_t > 0.15) triggers a fast read,
   biasing the hidden state towards the recalled concept without gradient parameter updates.
2. Entropy-Adaptive PAC Morphemic Decoding:
   Sub-morphemic bytes inside a word (Entropy H <= 0.50 nats) undergo MAP precision decoding (T=0.05),
   while word boundaries (Entropy H > 0.70 nats) unlock Top-p exploration (T=0.55).
   This eliminates sub-syllable repetition loops ('thesthesthe') and preserves 1-shot factual recall.

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
        print("[EXP-132] Loaded existing karyon_soul.kcore checkpoint.")

    # 1. Test 1-Shot Fast-Mapping Episodic Store
    fact_text = "Fact: Karyon's internal cognitive core is fully implemented in pure C++20 LibTorch."
    print(f"\n[Phase 1: Fast-Mapping Ingestion] Storing 1-shot fact into Episodic Memory:\n  '{fact_text}'")

    tokenizer = ByteTokenizer()
    fact_tokens = tokenizer.encode(fact_text)
    fact_tensor = torch.tensor([fact_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        fact_embs = agent.pos_embeddings(fact_tensor, start_pos=0, apply_rf=True)
        fact_unified = agent.in_proj(fact_embs).mean(dim=1) # [1, unified_dim]
        # Store key and value into memory
        memory.write(fact_unified, fact_unified)

    print(f"Memory Occupancy: {memory.size.item()}/{memory.max_capacity}")

    # 2. Benchmark Query BEFORE Fast-Mapping Activation vs AFTER
    prompt = "User: Where is Karyon's internal cognitive core implemented?\nKaryon:"
    print(f"\n[Phase 2: Generation Benchmark] Prompt:\n  '{prompt}'")

    # Set High Noradrenaline (Arousal NA = 0.85) to trigger memory retrieval
    hu.state[0, 4] = 0.85

    print("\n--- GENERATION WITH FAST-MAPPING & ENTROPY-ADAPTIVE PAC ---")
    gen_stream = agent.generate_thought_and_speech(
        prompt,
        m_state=torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device),
        h_state=torch.zeros(1, agent.hidden_dim, device=device),
        hu=hu,
        episodic_memory=memory,
        config=config,
        max_generated_tokens=120,
        temperature=0.45,
        top_p=0.90
    )

    chars = []
    high_h_count = 0
    map_step_count = 0

    for ev in gen_stream:
        if ev["status"] == "token":
            chars.append(ev["text"])

    output_text = "".join(chars)
    print(f"Generated Output:\n{output_text}")

    # Calculate 3-Gram Diversity and Fact Retrieval Accuracy
    def n_gram_diversity(text, n=3):
        if len(text) < n:
            return 1.0
        ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
        return len(set(ngrams)) / float(len(ngrams))

    div_score = n_gram_diversity(output_text)
    fact_recalled = ("C++20" in output_text or "LibTorch" in output_text or "C++" in output_text)

    print(f"\n3-Gram Lexical Diversity: {div_score:.4f}")
    print(f"1-Shot Fact Recalled in Text: {fact_recalled}")

    metrics = {
        "memory_items": memory.size.item(),
        "3gram_diversity": div_score,
        "fact_recalled": fact_recalled,
        "noradrenaline_arousal": hu.state[0, 4].item()
    }

    verdict = "🟢 POSITIVE" if (div_score > 0.40 and fact_recalled) else ("⚪ NEUTRAL" if div_score > 0.35 else "🔴 REJECTED")
    print(f"\n[EXP-132 VERDICT]: {verdict}")
    print("=" * 80)

    return metrics, verdict

if __name__ == "__main__":
    run_experiment()
