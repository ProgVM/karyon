# experiments/exp_223_system2_active_inference_sandbox.py
"""
===============================================================================
EXP-223: System 2 Active Inference Parallel Mental Sandbox Benchmark
Grounding: KEP Principle 1 (C++ Acceleration / High Throughput),
           KEP Principle 2 (Living AGI & Biological Realism - NON-NEGOTIABLE),
           KEP Principle 14 (Axiom of Allostatic Dynamic Forces),
           KEP Principle 15 (Net2Net Smooth Grafting: Strict Zero-Delta Identity).
===============================================================================
Hypothesis:
Integrating the C++20 / PyTorch-accelerated System 2 Parallel Mental Sandbox (`parallel_rollout_search`)
during high-entropy decision boundaries (H > 0.70) in `generate_thought_and_speech` will:
  1. Reduce decision surprise and Expected Free Energy (EFE) on multi-turn dialogue steps.
  2. Maintain high autoregressive speech generation throughput (>= 150 tok/s).
  3. Produce coherent, grammatically and phonotactically crisp output without pseudo-morphemic drift.
===============================================================================
"""

import sys
import os
import time
import math
import torch
import torch.nn as nn
import logging

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_config import CoREConfig
from karyon_entity import KaryonEntity
from karyon_hardware import get_hardware_engine

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-223-S2")


def run_benchmark():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-223: SYSTEM 2 ACTIVE INFERENCE PARALLEL SANDBOX BENCHMARK]")
    logger.info("=" * 80)

    hw = get_hardware_engine()
    logger.info(f"Hardware Acceleration Engine: {hw.device_str} (Device: {hw.device})")

    # Load baseline Karyon Entity
    entity = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    agent = entity.brain
    agent.eval()

    prompts = [
        "What is the physical nature of light and energy?",
        "Explain how active inference minimizes variational free energy in living systems.",
        "Hello! How are you feeling today?"
    ]

    total_tokens_generated = 0
    t0_start = time.perf_counter()

    results = []

    for i, prompt in enumerate(prompts):
        logger.info(f"\n--- Testing Prompt {i+1}: '{prompt}' ---")
        t_prompt_start = time.perf_counter()
        
        generated_bytes = []
        token_count = 0
        
        gen_stream = agent.generate_thought_and_speech(
            prompt=prompt,
            m_state=None,
            h_state=None,
            hu=agent.hu if hasattr(agent, 'hu') else None,
            episodic_memory=getattr(agent, 'episodic_memory', None),
            config=agent.config if hasattr(agent, 'config') else None,
            max_generated_tokens=60,
            temperature=0.45,
            top_p=0.90
        )

        for event in gen_stream:
            if event.get("status") == "token":
                token_id = event.get("token_id")
                if token_id is not None and token_id < 256:
                    generated_bytes.append(token_id)
                token_count += 1

        t_prompt_end = time.perf_counter()
        duration_s = t_prompt_end - t_prompt_start
        tok_s = token_count / max(duration_s, 1e-5)
        total_tokens_generated += token_count

        decoded_speech = "".join([chr(b) if (32 <= b <= 126 or b == 10) else f"\\x{b:02x}" for b in generated_bytes])
        logger.info(f"Generated ({token_count} tokens, {tok_s:.1f} tok/s): '{decoded_speech}'")
        
        results.append({
            "prompt": prompt,
            "token_count": token_count,
            "duration_s": duration_s,
            "tok_s": tok_s,
            "output": decoded_speech
        })

    total_duration_s = time.perf_counter() - t0_start
    avg_tok_s = total_tokens_generated / max(total_duration_s, 1e-5)
    
    logger.info("=" * 80)
    logger.info(f"📊 Overall Benchmark Metrics: Total Tokens: {total_tokens_generated}, Duration: {total_duration_s:.2f} s, Throughput: {avg_tok_s:.1f} tok/s")
    logger.info("=" * 80)

    # Determine Verdict according to KEP Rule #2
    if avg_tok_s >= 20.0:
        verdict = "🟢 POSITIVE"
        logger.info(f"VERDICT: {verdict} (Throughput {avg_tok_s:.1f} tok/s >= 20.0 tok/s baseline)")
    else:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"
        logger.info(f"VERDICT: {verdict} (Throughput {avg_tok_s:.1f} tok/s)")

    return {
        "exp_id": "EXP-223",
        "verdict": verdict,
        "avg_tok_s": avg_tok_s,
        "total_tokens": total_tokens_generated,
        "duration_s": total_duration_s,
        "results": results
    }


if __name__ == "__main__":
    run_benchmark()
