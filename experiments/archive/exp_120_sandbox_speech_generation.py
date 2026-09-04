# experiments/exp_120_sandbox_speech_generation.py
"""
===============================================================================
EXP-120: System 2 Parallel Mental Sandbox Speech Generation Benchmark
===============================================================================
Hypothesis:
Integrating the System 2 Parallel Mental Sandbox (evaluate_counterfactual_rollout) 
directly into the autoregressive speech generation loop (generate_thought_and_speech)
at high-entropy boundary points (H > 0.70) will:
1. Eliminate pseudo-morphemic drift and semantic hallucination.
2. Maintain smooth Top-p nucleus sampling with Active Inference EFE modulation.
3. Preserve sub-millisecond generation speed per token.

Protocol: KEP v9.0 Scientific Protocol
Author: Bazilevs & Autonomous Lead AI Cyberneticist
===============================================================================
"""

import os
import sys
import time
import math
import json
import torch

sys.path.insert(0, os.path.abspath('.'))

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory
from karyon_hardware import get_hardware_engine


def run_experiment():
    print("=" * 80)
    print("STARTING EXP-120: PARALLEL MENTAL SANDBOX SPEECH GENERATION BENCHMARK")
    print("=" * 80)

    hw = get_hardware_engine()
    device_str = 'cuda' if hw.is_cuda else ('xla' if hw.is_tpu else 'cpu')
    print(f"[Hardware] Active Device: {device_str}")

    config = CoREConfig()
    agent = CoREAgent(config, device=device_str).to(device_str)
    hu = HomeostaticUnit(batch_size=1, device=device_str)
    episodic_mem = BatchedEpisodicMemory(batch_size=1, memory_dim=256, max_capacity=100, device=device_str)

    prompts = [
        "What is the theory of Active Inference in neuroscience?",
        "Explain how the human brain integrates sensory information.",
        "Hello! Who are you and what is your goal?"
    ]

    print("\n--- Generating Speech with Active Inference System 2 Sandbox ---")
    
    generation_telemetry = []

    for prompt in prompts:
        print(f"\n[Prompt]: {prompt}")
        t0 = time.perf_counter()
        generated_text = ""
        token_count = 0
        
        gen_stream = agent.generate_thought_and_speech(
            prompt=prompt,
            m_state=torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device_str),
            h_state=torch.zeros(1, 1, agent.hidden_dim, device=device_str),
            hu=hu,
            episodic_memory=episodic_mem,
            config=config,
            max_generated_tokens=80
        )

        for chunk in gen_stream:
            if chunk.get("status") in ["token", "speech_patch"]:
                patch_str = chunk.get("text", "")
                generated_text += patch_str
                token_count += len(patch_str.encode('utf-8'))

        dur = time.perf_counter() - t0
        tok_s = token_count / max(dur, 1e-5)
        print(f"[Generated Response]: {generated_text.strip()}")
        print(f"[Telemetry]: Tokens/Bytes: {token_count} | Duration: {dur:.2f}s | Speed: {tok_s:.1f} tok/s")
        
        generation_telemetry.append({
            "prompt": prompt,
            "response": generated_text.strip(),
            "bytes": token_count,
            "duration_sec": dur,
            "tok_per_sec": tok_s
        })

    avg_speed = sum(p["tok_per_sec"] for p in generation_telemetry) / len(generation_telemetry)
    print(f"\n[Summary] Average Speech Generation Speed with System 2 Sandbox: {avg_speed:.1f} tok/s")

    with open("exp_120_results.json", "w") as f:
        json.dump(generation_telemetry, f, indent=2)

if __name__ == "__main__":
    run_experiment()
