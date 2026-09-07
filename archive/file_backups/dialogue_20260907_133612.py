# dialogue.py
"""
Closed-Loop Social Active Inference Interactive Session.
Delegates orchestration directly to KaryonEntity as the self-contained biological cognitive substrate.
"""

import sys
import time
import json
import os
import torch
import logging
from karyon_entity import KaryonEntity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SocialDialogue")

device_str = "cuda" if torch.cuda.is_available() else "cpu"
kcore_path = "karyon_soul.kcore"

# 1. Load Unified Karyon Entity
entity = KaryonEntity.load(filepath=kcore_path, device=device_str)

logger.info(f"Loaded Karyon Soul (.kcore) | Device: {device_str.upper()} | Genome DNA -> text_dim: {entity.brain.text_dim}, hidden_dim: {entity.brain.hidden_dim}, unified_dim: {entity.brain.unified_dim}")
logger.info("Welcome to Closed-Loop Social Active Inference Session with Karyon-CoRE v31.0!")
logger.info("Type 'exit' to save state and close.")
logger.info("Type 'sleep' to trigger deep allostatic sleep & morphogenesis.")
logger.info("Press [Enter] with empty input to let Karyon think spontaneously (Inner Monologue / Spontaneous Turn).")

def render_affective_dashboard(hu_state, affective_state):
    curiosity, energy, stability, health, na, da = hu_state[0].tolist()
    valence = affective_state["valence"]
    arousal = affective_state["arousal"]
    dominance = affective_state["dominance"]
    panksepp = affective_state["panksepp"]
    
    print("\n" + "="*80)
    print(" === [KARYON SOMATIC & AFFECTIVE CORE DASHBOARD] ===")
    print("="*80)
    print(f"  Somatic State : Energy: {energy:.3f} | Health: {health:.3f} | Curiosity: {curiosity:.3f} | Stability: {stability:.3f}")
    print(f"  Neurokinetics : Noradrenaline (Arousal): {na:.3f} | Dopamine (Reward): {da:.3f}")
    print(f"  Russell Space : Valence: {valence:+.3f} | Arousal: {arousal:.3f} | Dominance: {dominance:+.3f}")
    print(f"  Panksepp Drives: SEEKING: {panksepp['SEEKING']:.3f} | FEAR: {panksepp['FEAR']:.3f} | RAGE: {panksepp['RAGE']:.3f} | PANIC: {panksepp['PANIC']:.3f}")
    
    # Simple ASCII Valence-Arousal grid
    grid_size = 5
    v_idx = int((valence + 1.0) / 2.0 * (grid_size - 1))
    a_idx = int(arousal * (grid_size - 1))
    v_idx = max(0, min(grid_size - 1, v_idx))
    a_idx = max(0, min(grid_size - 1, a_idx))
    
    print("  Russell Grid  :  [High Arousal]")
    for r in range(grid_size - 1, -1, -1):
        row_str = "                  "
        for c in range(grid_size):
            if r == a_idx and c == v_idx:
                row_str += "☼ "
            elif r == grid_size // 2 and c == grid_size // 2:
                row_str += "+ "
            else:
                row_str += "· "
        if r == grid_size // 2:
            row_str += " [Unpleasant] ───┼─── [Pleasant]"
        print(row_str)
    print("                  [Low Arousal]")
    print("="*80 + "\n")

while True:
    try:
        user_input = input("You (Human Reaction): ")
    except (KeyboardInterrupt, EOFError):
        break

    if user_input.lower() == 'exit':
        entity.save(filepath=kcore_path)
        logger.info(f"Session closed. State persisted into '{kcore_path}'.")
        break

    if user_input.lower().strip() in ['sleep', '/sleep', 'sleep!']:
        logger.info("🌙 [User requested sleep] Karyon is entering Deep Evolutionary Sleep & Synaptic Morphogenesis...")
        t_sleep_start = time.perf_counter()
        pruned_weights = entity.sleep(num_replay_cycles=4, downscaling_factor=0.02)
        sleep_duration_sec = time.perf_counter() - t_sleep_start
        logger.info(f"☀️ [Awakened] Evolutionary Sleep Complete ({sleep_duration_sec:.2f}s). Restored Energy={entity.hu.state[0, 1].item():.2f} | Pruned Synapses={pruned_weights}")
        entity.save(filepath=kcore_path)
        print(f"Karyon: *awakes from deep evolutionary sleep, synapses pruned ({pruned_weights}), energy fully restored to {entity.hu.state[0, 1].item():.2f}* I am renewed.")
        continue

    is_spontaneous = not bool(user_input.strip())
    if is_spontaneous:
        logger.info("⚡ [Human remains silent. Karyon initiates spontaneous thought & self-learning cycle...] ")
        self_learning_results = entity.self_learn(num_sequences=3, seq_len=64)
        logger.info(f"  Self-Learning Complete | Initial FE: {self_learning_results['initial_free_energy']:.4f} | Final FE: {self_learning_results['final_free_energy']:.4f}")

    # Stream interaction through KaryonEntity
    for event in entity.interact(user_input, max_tokens=120, temperature=0.45, top_p=0.90):
        status = event["status"]
        if status == "speech_start":
            if is_spontaneous:
                print("Karyon (Spontaneous Thought): ", end="", flush=True)
            else:
                print("Karyon: ", end="", flush=True)
        elif status == "token":
            print(event["text"], end="", flush=True)
        elif status == "exhausted":
            print(event["text"], end="", flush=True)
        elif status == "speech_end":
            print()
            with torch.no_grad():
                affective_state = entity.brain.affective_core.compute_affective_state(entity.hu.state, free_energy=0.05)
                render_affective_dashboard(entity.hu.state, affective_state)
