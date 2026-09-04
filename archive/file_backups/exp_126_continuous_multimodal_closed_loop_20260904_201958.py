# experiments/exp_126_continuous_multimodal_closed_loop.py
"""
===============================================================================
EXP-126: Continuous Multimodal Closed-Loop Autonomous Agent Environment
===============================================================================
Hypothesis & Operational Protocol (KEP Principle 2 & Principle 12):
1. Principle 12 (Universal Modality-Agnostic Substrate):
   An autonomous cognitive agent operating in a continuous closed-loop environment
   can process interleaved multimodal streams (Text UTF-8 bytes, 256D spatial vision,
   and 3D motor efference) on a unified representation space (D=256).

2. Closed-Loop Allostasis & Volitional Governance:
   Driven by internal homeostatic state dynamics (Energy depletion over time, Curiosity
   arousal from environmental novelty), the agent autonomously uses its native C++20
   `VolitionalActionEvaluator` (EFE) to alternate between:
   - Expressing Motor/Text Output (Action 0)
   - Conducting Mental Sandbox Counterfactual Planning (Action 1)
   - Initiating Biophysical Sleep Consolidation & SHY Synaptic Pruning (Action 2)

Target Telemetry Metrics:
- Multimodal Stream Processing Latency (ms per turn across Text, Vision, Motor)
- Homeostatic Ultrastability (Energy, Curiosity, Noradrenaline trajectories)
- Volitional Policy Distribution across 20 Continuous Closed-Loop Environment Steps
- Sleep Consolidation Recovery Rate

Protocol: KEP v9.0 Scientific Protocol
Author: Bazilevs & Autonomous Lead AI Cyberneticist
===============================================================================
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('.'))

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import (
    HomeostaticUnit,
    BatchedEpisodicMemory,
    VolitionalActionEvaluator,
    LocalNeuromodulatedPlasticity
)
from karyon_hardware import get_hardware_engine


def run_closed_loop_environment():
    print("=" * 80)
    print("STARTING EXP-126: CONTINUOUS MULTIMODAL CLOSED-LOOP AUTONOMOUS AGENT SESSION")
    print("=" * 80)

    hw = get_hardware_engine()
    device_str = 'cuda' if hw.is_cuda else ('xla' if hw.is_tpu else 'cpu')
    print(f"[Hardware Environment] Active Compute Platform: {device_str}")

    config = CoREConfig()
    batch_size = 1
    agent = CoREAgent(config, device=device_str).to(device_str)
    agent.eval()

    hu = HomeostaticUnit(batch_size=batch_size, device=device_str)
    episodic_mem = BatchedEpisodicMemory(batch_size=batch_size, memory_dim=agent.unified_dim, max_capacity=500, device=device_str)

    # Initial homeostatic states
    hu.state[0, 0] = 0.85 # High Curiosity
    hu.state[0, 1] = 0.90 # High Initial Energy
    hu.state[0, 2] = 0.80 # High Stability

    # Environmental Multimodal Generators
    def generate_environment_step(step_idx):
        # Simulated continuous environmental percepts
        text_prompt = f"Env step {step_idx}: Target object detected at spatial coordinates."
        text_bytes = torch.tensor([[ord(c) for c in text_prompt]], dtype=torch.long, device=device_str)
        
        vision_vector = torch.randn(1, 256, device=device_str) + (0.1 * step_idx) # Evolving visual scene
        motor_proprioception = torch.tensor([[0.5 * math.sin(step_idx * 0.2), 0.5 * math.cos(step_idx * 0.2), 0.1 * step_idx]], device=device_str)
        
        return text_bytes, vision_vector, motor_proprioception

    import math
    num_env_steps = 20
    action_counts = {"EXPRESS_OUTPUT": 0, "THINK_DEEPER_SANDBOX": 0, "INITIATE_SLEEP_CONSOLIDATION": 0}
    turn_telemetry = []

    print("\n--- Running 20-Step Continuous Closed-Loop Autonomous Simulation ---")

    for step in range(1, num_env_steps + 1):
        t0 = time.perf_counter()
        
        # 1. Ingest Environmental Multimodal Percepts
        text_bytes, vision_vec, motor_vec = generate_environment_step(step)

        # Process all 3 modalities on unified substrate
        h_text, fe_text, _, ms_text = agent.process_universal_stream('text', text_bytes, hu, episodic_mem)
        h_vis, fe_vis, _, ms_vis = agent.process_universal_stream('vision', vision_vec, hu, episodic_mem)
        h_mot, fe_mot, _, ms_mot = agent.process_universal_stream('motor', motor_vec, hu, episodic_mem)

        # Unified Current Mind Vector
        h_unified = (h_text + h_vis + h_mot) / 3.0
        avg_fe = (fe_text + fe_vis + fe_mot) / 3.0

        # 2. Natural Somatic Energy Depletion & Curiosity Modulation
        curiosity = hu.state[0, 0].item()
        energy = max(0.05, hu.state[0, 1].item() - 0.05) # Metabolic cost per perception
        hu.state[0, 1] = energy # Update energy in body

        # 3. Volitional Decision via C++20 EFE Action Evaluator
        action_idx = agent.efe_action_evaluator.select_volitional_action(h_unified, curiosity, energy)
        action_names = ["EXPRESS_OUTPUT", "THINK_DEEPER_SANDBOX", "INITIATE_SLEEP_CONSOLIDATION"]
        chosen_action = action_names[action_idx]
        action_counts[chosen_action] += 1

        action_detail = ""

        # 4. Execute Volitional Choice
        if chosen_action == "EXPRESS_OUTPUT":
            # Adapt local fast weights and output motor command
            agent.local_plasticity.adapt_local_fast_weights(h_unified, torch.randn_like(h_unified), 0.5, 0.5)
            action_detail = "Generated active efference response & updated local fast weights"
            
        elif chosen_action == "THINK_DEEPER_SANDBOX":
            # Execute System 2 Mental Rollout in Latent Predictor
            with torch.no_grad():
                z_prior = torch.randn(1, agent.latent_dim, device=device_str)
                h_future = agent.world_model.predict_next_state(h_unified, z_prior)
            action_detail = f"Executed 1-step System 2 counterfactual rollout (FE: {avg_fe:.4f})"
            hu.state[0, 0] = max(0.1, curiosity - 0.15) # Satisfied curiosity
            
        elif chosen_action == "INITIATE_SLEEP_CONSOLIDATION":
            # Execute Sleep 2.0
            sleep_rep = agent.execute_sleep_consolidation_2(hu, episodic_mem, num_replay_cycles=3)
            action_detail = f"Sleep 2.0 complete: Energy restored to {hu.state[0, 1].item():.2f} in {sleep_rep['duration_ms']:.2f}ms"

        step_latency_ms = (time.perf_counter() - t0) * 1000.0

        print(f"[Step {step:02d}] Energy={energy:.2f} | Curiosity={curiosity:.2f} | Action: {chosen_action:<28} | {action_detail} | Latency: {step_latency_ms:.2f}ms")

        turn_telemetry.append({
            "step": step,
            "energy": energy,
            "curiosity": curiosity,
            "action": chosen_action,
            "latency_ms": step_latency_ms,
            "avg_fe": avg_fe
        })

    # Summary
    print("\n" + "=" * 80)
    print("EXP-126 CONTINUOUS CLOSED-LOOP SIMULATION COMPLETE")
    print("=" * 80)
    print(f"• Total Environment Steps Executed : {num_env_steps}")
    print(f"• Action Distribution Breakdown   : {action_counts}")
    print(f"• Average Step Processing Latency  : {sum(t['latency_ms'] for t in turn_telemetry)/num_env_steps:.2f} ms")
    print(f"• Final Allostatic Energy          : {hu.state[0, 1].item():.2f}")
    print("=" * 80)

    results = {
        "num_env_steps": num_env_steps,
        "action_counts": action_counts,
        "avg_step_latency_ms": sum(t['latency_ms'] for t in turn_telemetry)/num_env_steps,
        "final_energy": hu.state[0, 1].item(),
        "telemetry": turn_telemetry
    }

    with open("exp_126_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_closed_loop_environment()
