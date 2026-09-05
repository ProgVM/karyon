"""
EXP-134 Scientific Benchmark: Biophysical Laminar Alignment & Theta Phase-Reset PAC Decoding
Evaluates speech generation coherence and repetition dynamics before and after cortical alignment.
"""

import sys
import os
import torch
import torch.nn.functional as F
from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory
from karyon_checkpoint import load_karyon

def run_benchmark():
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"--- EXP-134 Diagnostic Benchmark on {device_str} ---")

    hu = HomeostaticUnit(batch_size=1, device=device_str)
    memory = BatchedEpisodicMemory(batch_size=1, memory_dim=256, max_capacity=1000, device=device_str)

    config = CoREConfig()
    config.train.batch_size = 1
    agent = CoREAgent(config, device=device_str)
    load_karyon(agent, memory, hu, filepath='karyon_soul.kcore', device=device_str)

    test_prompts = [
        "User: Hello Karyon!\nKaryon:",
        "User: What is energy?\nKaryon:",
        "User: Explain gravity simply.\nKaryon:"
    ]

    print("\n--- Diagnostic Speech Generation (Current Engine) ---")
    for p in test_prompts:
        print(f"\nPrompt: {p}")
        gen = agent.generate_thought_and_speech(
            p,
            m_state=torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device_str),
            h_state=torch.zeros(1, agent.hidden_dim, device=device_str),
            hu=hu,
            episodic_memory=memory,
            config=config,
            max_generated_tokens=40,
            temperature=0.35,
            top_p=0.90
        )
        out_text = ""
        for ev in gen:
            if ev["status"] == "token":
                out_text += ev["text"]
        print(f"Output: {out_text.strip()}")

if __name__ == "__main__":
    run_benchmark()
