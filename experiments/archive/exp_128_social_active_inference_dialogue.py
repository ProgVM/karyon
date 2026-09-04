# experiments/exp_128_social_active_inference_dialogue.py
"""
===============================================================================
EXP-128: Social Active Inference Multi-Turn Dialogue & Affective Core Test
===============================================================================
Hypothesis & Protocol (Vector B):
Evaluating the Closed-Loop Social Active Inference engine:
1. Ingests human utterances, updates internal Free Energy & Human Surprise ($F_{\text{human}}$).
2. Perceptive Rest Energy Recovery (Magistretti 2015): Listening to user input restores energy.
3. Computes Russell Circumplex (Valence, Arousal, Dominance) and Panksepp Affective Drives (SEEKING, FEAR, RAGE, PANIC).
4. Demonstrates Spontaneous Thought Generation when the human is silent (Empty Input).
5. Awake SWR Micro-Replay & Episodic Consolidation during inter-turn pause.

Author: Bazilevs & Autonomous Lead AI Cyberneticist
===============================================================================
"""

import os
import sys
import time
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath('.'))

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory
from karyon_checkpoint import load_karyon
from karyon_hardware import get_hardware_engine


def run_social_dialogue_session():
    print("=" * 80)
    print("STARTING EXP-128: SOCIAL ACTIVE INFERENCE & AFFECTIVE DIALOGUE BENCHMARK")
    print("=" * 80)

    hw = get_hardware_engine()
    device_str = 'cuda' if hw.is_cuda else ('xla' if hw.is_tpu else 'cpu')
    device = torch.device(device_str)

    config = CoREConfig()
    config.train.batch_size = 1
    kcore_path = "karyon_soul.kcore"

    agent = CoREAgent(config=config, device=device_str).to(device)
    agent.eval()

    hu = HomeostaticUnit(batch_size=1, device=device_str)
    episodic_mem = BatchedEpisodicMemory(batch_size=1, memory_dim=config.net.unified_dim, max_capacity=1000, device=device_str)

    h_fast, h_slow, epoch, story_idx = load_karyon(agent, episodic_mem, hu, filepath=kcore_path, device=device_str)
    print(f"[Identity] Loaded Karyon Soul (.kcore) | Device: {device_str.upper()}")

    # Simulated Human Dialogue Turns
    turns = [
        "Hello Karyon! Who are you?",
        "What is the source of light in our solar system?",
        "", # Empty string -> tests spontaneous thought cycle
        "Thank you for sharing your thoughts with me."
    ]

    dialogue_history = ""
    session_telemetry = []

    for turn_idx, user_input in enumerate(turns, 1):
        t0 = time.perf_counter()
        is_spontaneous = not bool(user_input.strip())

        if is_spontaneous:
            print(f"\n[Turn {turn_idx}] User is silent. Karyon initiates Spontaneous Thought Cycle...")
            full_prompt = (dialogue_history + " Karyon (Spontaneous Thought):").strip() if dialogue_history else "Karyon (Spontaneous Thought):"
        else:
            print(f"\n[Turn {turn_idx}] User: {user_input}")
            # Perceptive rest boost
            rest_boost = 0.0040 * float(len(user_input))
            hu.state[0, 1] = torch.clamp(hu.state[0, 1] + rest_boost, 0.0, 1.0)
            
            turn_str = f"User: {user_input.strip()}\nKaryon:"
            full_prompt = (dialogue_history + " " + turn_str).strip() if dialogue_history else turn_str

        # Generate thought and speech
        gen_stream = agent.generate_thought_and_speech(
            prompt=full_prompt,
            m_state=torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device),
            h_state=h_fast,
            hu=hu,
            episodic_memory=episodic_mem,
            config=config,
            max_generated_tokens=40,
            temperature=0.45,
            top_p=0.90
        )

        response_chars = []
        for event in gen_stream:
            if event["status"] == "token":
                response_chars.append(event["text"])
            elif event["status"] in ["speech_end", "exhausted"]:
                h_fast = event.get("h_state", h_fast)

        response_text = "".join(response_chars).strip()
        label = "Karyon (Spontaneous Thought)" if is_spontaneous else "Karyon"
        print(f"{label}: {response_text}")

        # Compute Affective State
        affective_state = agent.affective_core.compute_affective_state(hu.state, free_energy=0.08)
        curiosity, energy, stability, health, na, da = hu.state[0].tolist()

        print(f"  Affective State: Valence={affective_state['valence']:+.2f} | Arousal={affective_state['arousal']:.2f} | SEEKING={affective_state['panksepp']['SEEKING']:.2f}")
        print(f"  Somatic State  : Energy={energy:.2f} | Curiosity={curiosity:.2f} | Stability={stability:.2f}")

        # Update dialogue history
        if is_spontaneous:
            dialogue_history = (dialogue_history + f" Karyon (Spontaneous Thought): {response_text}").strip()
        else:
            dialogue_history = (dialogue_history + f" User: {user_input.strip()}\nKaryon: {response_text}").strip()

        # Inter-turn awake SWR micro-replay
        agent.execute_wake_swr_micro_replay(episodic_mem, num_samples=2)

        turn_duration_ms = (time.perf_counter() - t0) * 1000.0
        session_telemetry.append({
            "turn": turn_idx,
            "user_input": user_input,
            "is_spontaneous": is_spontaneous,
            "response": response_text,
            "energy": energy,
            "curiosity": curiosity,
            "valence": affective_state["valence"],
            "arousal": affective_state["arousal"],
            "turn_duration_ms": turn_duration_ms
        })

    print("\n" + "=" * 80)
    print("EXP-128 SOCIAL ACTIVE INFERENCE BENCHMARK COMPLETE")
    print("=" * 80)

    with open("exp_128_results.json", "w") as f:
        json.dump(session_telemetry, f, indent=2)

if __name__ == "__main__":
    run_social_dialogue_session()
