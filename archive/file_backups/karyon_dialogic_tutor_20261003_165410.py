# karyon_dialogic_tutor.py
"""
===============================================================================
KARYON CLOSED-LOOP DIALOGIC TUTOR & PRAGMATIC GROUNDING ENGINE (v1.0)
===============================================================================
Implements KEP Principle 26 (The Triad of Grounded Meaning & Pragmatic Agency).
The Agent acts as an active environment/tutor, routing Karyon's verbal outputs,
performing pragmatic/spelling analysis, injecting somatic neurotransmitter
rewards/stress, and executing predictive coding corrections to align trajectories.
"""

import os
import time
import torch
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from karyon_entity import KaryonEntity
from karyon_config import CoREConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DialogicTutor")

device_str = "cuda" if torch.cuda.is_available() else "cpu"
soul_input_path = "karyon_soul_v8.kcore"
soul_output_path = "karyon_soul_v9.kcore"

# 1. Define structured educational trials
TRIALS = [
    {
        "prompt": "What is this round red fruit that grows on trees?",
        "target": "apple",
        "domain": "naming"
    },
    {
        "prompt": "Spell the word apple letter by letter.",
        "target": "apple",
        "domain": "spelling"
    },
    {
        "prompt": "Where does the Earth get its light and energy from?",
        "target": "sun",
        "domain": "physics"
    },
    {
        "prompt": "What is the name of this cognitive architecture?",
        "target": "karyon",
        "domain": "identity"
    },
    {
        "prompt": "Is fire hot or cold?",
        "target": "hot",
        "domain": "physics"
    },
    {
        "prompt": "What is the opposite of hot?",
        "target": "cold",
        "domain": "physics"
    },
    {
        "prompt": "What is the sum of two and two?",
        "target": "four",
        "domain": "math"
    },
    {
        "prompt": "What is the color of the clear daytime sky?",
        "target": "blue",
        "domain": "naming"
    }
]


def detect_pseudo_morphemic_drift(text: str) -> bool:
    """
    Heuristic to detect pseudo-morphemic drift and babbling:
    - Repeated syllables (e.g., 'ananan', 'ingining', 'sinsig', 'poupou')
    - Excessive length or non-ascii strings without spaces
    """
    text_lower = text.lower()
    if len(text_lower) > 30 and " " not in text_lower:
        return True
    
    # Check for repetitive patterns
    for i in range(len(text_lower) - 5):
        sub = text_lower[i:i+3]
        if text_lower.count(sub) >= 4:
            return True
            
    # Check for babbling syllables
    for babble in ["ananan", "ingining", "sinsig", "poupou", "mememe", "tatata"]:
        if babble in text_lower:
            return True
            
    return False


def run_dialogic_tutor_session():
    logger.info("Initializing Karyon Dialogic Tutor...")
    
    # Load Karyon Entity
    if not os.path.exists(soul_input_path) and os.path.exists("karyon_soul.kcore"):
        logger.info("karyon_soul_v8.kcore not found, falling back to karyon_soul.kcore")
        entity = KaryonEntity.load(filepath="karyon_soul.kcore", device=device_str)
    else:
        entity = KaryonEntity.load(filepath=soul_input_path, device=device_str)
        
    logger.info(f"Loaded Karyon Soul onto device: {device_str}")
    
    history_metrics = []
    
    for idx, trial in enumerate(TRIALS):
        prompt = trial["prompt"]
        target = trial["target"]
        domain = trial["domain"]
        
        logger.info(f"\n=== TRIAL {idx+1}/{len(TRIALS)} | Domain: {domain.upper()} ===")
        logger.info(f"Tutor Prompt: '{prompt}'")
        
        # Get Karyon response
        t0 = time.perf_counter()
        reply = entity.step(prompt, thinking_steps=4, max_new_tokens=48)
        latency = (time.perf_counter() - t0) * 1000.0
        
        logger.info(f"Karyon Reply: '{reply}' [Latency: {latency:.1f}ms]")
        
        # Pragmatic evaluation
        has_drift = detect_pseudo_morphemic_drift(reply)
        is_correct = target.lower() in reply.lower()
        
        # Determine valence
        if is_correct and not has_drift:
            valence = 1.0
            verdict_str = "CORRECT (Coherent)"
        elif is_correct and has_drift:
            valence = 0.0
            verdict_str = "PARTIAL (Correct target but drifted/babbling)"
        else:
            valence = -1.0
            verdict_str = "INCORRECT / DRIFTED"
            
        logger.info(f"Tutor Pragmatic Analysis: Verdict={verdict_str} | Drift={has_drift} | Correct={is_correct} | Valence={valence}")
        
        # Get current somatic states
        if entity.hu is not None:
            with torch.no_grad():
                old_states = entity.hu.get_states().clone().cpu().tolist()
        else:
            old_states = [0.8, 1.0, 0.9, 1.0, 0.1, 0.1]
            
        curiosity, energy, stability, health, na, da = old_states
        
        # Inject neurotransmitter adjustments based on Tutor evaluation
        if valence == 1.0:
            da = min(1.0, da + 0.25)
            na = max(0.01, na - 0.15)
            stability = min(1.0, stability + 0.10)
            health = min(1.0, health + 0.05)
            curiosity = max(0.1, curiosity - 0.05)  # Satisfaction reduces immediate curiosity
        elif valence == -1.0:
            da = max(0.0, da - 0.20)
            na = min(1.0, na + 0.30)  # High stress/arousal on error
            stability = max(0.0, stability - 0.20)
            health = max(0.1, health - 0.05)
            curiosity = min(1.0, curiosity + 0.15)  # Frustration triggers search/curiosity
        else:
            da = max(0.0, da - 0.05)
            na = min(1.0, na + 0.10)
            stability = max(0.0, stability - 0.05)
            
        # Update homeostatic nexus states
        if entity.hu is not None:
            with torch.no_grad():
                states_tensor = entity.hu.get_states()
                states_tensor[0] = curiosity
                states_tensor[1] = energy
                states_tensor[2] = stability
                states_tensor[3] = health
                states_tensor[4] = na
                states_tensor[5] = da
                new_states = states_tensor.cpu().tolist()
        else:
            new_states = [curiosity, energy, stability, health, na, da]
            
        logger.info(f"Somatic State Shift:")
        logger.info(f"  Curiosity:     {old_states[0]:.3f} -> {new_states[0]:.3f}")
        logger.info(f"  Energy:        {old_states[1]:.3f} -> {new_states[1]:.3f}")
        logger.info(f"  Stability:     {old_states[2]:.3f} -> {new_states[2]:.3f}")
        logger.info(f"  Health:        {old_states[3]:.3f} -> {new_states[3]:.3f}")
        logger.info(f"  Noradrenaline: {old_states[4]:.3f} -> {new_states[4]:.3f} (arousal)")
        logger.info(f"  Dopamine:      {old_states[5]:.3f} -> {new_states[5]:.3f} (reward)")
        
        # Record Somatic Episode in Hopfield Memory
        somatic_msg = entity.record_somatic_feedback(valence)
        logger.info(f"Memory System: {somatic_msg}")
        
        # Execute Predictive Coding Step if incorrect or drifted
        pred_loss = 0.0
        if valence < 0.5:
            logger.info(f"Tutor corrective feedback: 'No, the correct answer is {target}. Let me teach you.'")
            pred_loss = entity.teach_predictive_step(target)
            logger.info(f"Predictive Coding Correction Step: Loss = {pred_loss:.4f}")
        else:
            logger.info("Tutor feedback: 'Excellent! Keep it up.'")
            
        # Record trial metrics
        history_metrics.append({
            "trial": idx + 1,
            "prompt": prompt,
            "reply": reply,
            "target": target,
            "correct": is_correct,
            "drift": has_drift,
            "valence": valence,
            "curiosity": new_states[0],
            "energy": new_states[1],
            "stability": new_states[2],
            "health": new_states[3],
            "na": new_states[4],
            "da": new_states[5],
            "pred_loss": pred_loss
        })
        
        # Periodic sleep cycle every 3 trials
        if (idx + 1) % 3 == 0:
            logger.info("\n=== METABOLIC TRANSITION: Entering Sleep State for Consolidation ===")
            sleep_stats = entity.execute_sleep_cycle()
            logger.info(f"Sleep Cycle Completed in {sleep_stats['duration_ms']:.1f}ms. Pruned {sleep_stats['pruned_nodes']} inactive nodes. Energy restored to 1.0.")
            
    # Save the updated soul container
    entity.save(soul_output_path)
    logger.info(f"\nSaved updated Karyon Soul to '{soul_output_path}'")
    
    # 2. Plot results
    plot_dialogic_tutor_curves(history_metrics)
    
    return history_metrics


def plot_dialogic_tutor_curves(metrics: List[Dict[str, Any]]):
    trials = [m["trial"] for m in metrics]
    da = [m["da"] for m in metrics]
    na = [m["na"] for m in metrics]
    stability = [m["stability"] for m in metrics]
    pred_loss = [m["pred_loss"] for m in metrics]
    
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Somatic states
    ax1.plot(trials, da, label="Dopamine (DA - Reward)", color="#00ffcc", marker="o", linewidth=2)
    ax1.plot(trials, na, label="Noradrenaline (NA - Stress)", color="#ff3366", marker="s", linewidth=2)
    ax1.plot(trials, stability, label="Somatic Stability", color="#33ccff", marker="^", linewidth=2)
    ax1.set_ylabel("Somatic Scale Value")
    ax1.set_title("Karyon Biophysical Somatic State Trajectory under Dialogic Tutoring")
    ax1.legend(loc="upper left")
    ax1.grid(True, color="#333333", linestyle="--")
    
    # Predictive loss
    ax2.plot(trials, pred_loss, label="Predictive Coding Loss", color="#ffcc00", marker="x", linewidth=2, linestyle="--")
    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.set_title("Predictive Coding Error Convergence")
    ax2.legend(loc="upper left")
    ax2.grid(True, color="#333333", linestyle="--")
    
    plt.tight_layout()
    plot_path = "experiments/exp_342_dialogic_tutor.png"
    os.makedirs("experiments", exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved diagnostic curves to '{plot_path}'")


if __name__ == "__main__":
    run_dialogic_tutor_session()
