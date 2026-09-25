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

logger.info(f"Loaded Karyon Soul (.kcore) | Device: {device_str.upper()} | Genome DNA -> embed_dim: {entity.brain.embed_dim}, hidden_dim: {entity.brain.hidden_dim}, unified_dim: {entity.brain.unified_dim}")
logger.info("Welcome to Closed-Loop Social Active Inference Session with Karyon-CoRE!")
logger.info("Type 'exit' to save state and close.")
logger.info("Type 'sleep' to trigger deep allostatic sleep & morphogenesis.")
logger.info("Press [Enter] with empty input to let Karyon think spontaneously (Inner Monologue / Spontaneous Turn).")

def render_affective_dashboard(hu_state):
    curiosity, energy, stability, health, na, da = hu_state[0].tolist()
    
    print("\n" + "="*80)
    print(" === [KARYON SOMATIC & AFFECTIVE CORE DASHBOARD] ===")
    print("="*80)
    print(f"  Somatic State : Energy: {energy:.3f} | Health: {health:.3f} | Curiosity: {curiosity:.3f} | Stability: {stability:.3f}")
    print(f"  Neurokinetics : Noradrenaline (Arousal): {na:.3f} | Dopamine (Reward): {da:.3f}")
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
        entity.brain.execute_deep_allostatic_sleep(steps=10)
        sleep_duration_sec = time.perf_counter() - t_sleep_start
        logger.info(f"☀️ [Awakened] Evolutionary Sleep Complete ({sleep_duration_sec:.2f}s).")
        entity.save(filepath=kcore_path)
        print("Karyon: *awakes from deep evolutionary sleep, energy restored* I am renewed.")
        continue

    is_spontaneous = not bool(user_input.strip())
    prompt_to_pass = "..." if is_spontaneous else user_input

    # Step through entity
    karyon_response = entity.step(prompt_to_pass, thinking_steps=4, max_new_tokens=48)
    
    if is_spontaneous:
        print(f"Karyon (Spontaneous Thought): {karyon_response}")
    else:
        print(f"Karyon: {karyon_response}")

    # Render dashboard
    if entity.hu is not None:
        hu_state = entity.hu.get_states().unsqueeze(0)
    else:
        hu_state = torch.tensor([[0.85, 0.90, 0.80, 0.95, 0.25, 0.40]])
    render_affective_dashboard(hu_state)
