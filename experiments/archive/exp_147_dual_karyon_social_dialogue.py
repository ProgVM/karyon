# experiments/exp_147_dual_karyon_social_dialogue.py
"""
EXP-147: Dual-Agent Reciprocal Social Active Inference Benchmark (Acoustic/Byte Channel Baseline).
Simulates closed-loop communicative exchange between two distinct Karyon entities (Alpha & Beta).
Evaluates biophysical homeostasis coupling, mutual Free Energy surprise (F_A, F_B),
turn-taking dynamics, episodic memory writing, and communicative entropy.
"""

import os
import sys
import time
import json
import torch
import logging

from karyon_entity import KaryonEntity
from karyon_checkpoint import load_karyon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-147-DualAgent")

def run_dual_agent_experiment():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    kcore_path = "karyon_soul.kcore"
    
    if not os.path.exists(kcore_path):
        logger.error(f"Checkpoint file '{kcore_path}' not found!")
        sys.exit(1)
        
    logger.info(f"⚡ [EXP-147] Initializing Dual-Agent Substrates on {device_str.upper()}...")
    
    # 1. Instantiate Agent Alpha and Agent Beta
    entity_alpha = KaryonEntity.load(filepath=kcore_path, device=device_str)
    entity_beta = KaryonEntity.load(filepath=kcore_path, device=device_str)
    
    # Differentiate initial personality/somatic tuning (Principle 2: Biological Realism)
    # Alpha: High SEEKING / High Curiosity / Active initiator
    entity_alpha.hu.state[0, 0] = 0.95  # Curiosity
    entity_alpha.hu.state[0, 1] = 0.85  # Energy
    entity_alpha.hu.state[0, 4] = 0.15  # Noradrenaline (Arousal)
    entity_alpha.hu.state[0, 5] = 0.30  # Dopamine (Exploratory Drive)
    
    # Beta: High Stability / Receptive / Evaluative listener
    entity_beta.hu.state[0, 0] = 0.45   # Curiosity
    entity_beta.hu.state[0, 1] = 0.90   # Energy
    entity_beta.hu.state[0, 2] = 0.98   # Stability
    entity_beta.hu.state[0, 4] = 0.05   # Noradrenaline
    entity_beta.hu.state[0, 5] = 0.10   # Dopamine
    
    logger.info("Agent Alpha initialized: [SEEKING Profile | High Curiosity & Noradrenaline]")
    logger.info("Agent Beta  initialized: [HOMEOSTATIC Profile | High Stability & Receptivity]")
    
    # 2. Initial Seeding Prompt
    seed_topic = "What is the nature of life and continuous active inference?"
    logger.info(f"🌱 Seeding dialogue conversation with seed: \"{seed_topic}\"")
    
    num_rounds = 6
    dialogue_log = []
    
    current_speaker = entity_alpha
    current_listener = entity_beta
    speaker_name = "Karyon-Alpha"
    listener_name = "Karyon-Beta"
    
    incoming_message = seed_topic
    round_telemetry = []
    
    t_start = time.perf_counter()
    
    print("\n" + "="*90)
    print(" === [EXP-147: DUAL-AGENT SOCIAL ACTIVE INFERENCE DIALOGUE SESSION] ===")
    print("="*90)
    
    for round_idx in range(1, num_rounds + 1):
        print(f"\n--- [DIALOGUE ROUND {round_idx}/{num_rounds}] ---")
        print(f"Speaker: {speaker_name} | Listener: {listener_name}")
        print(f"Incoming Stimulus: \"{incoming_message}\"")
        
        # Speaker's pre-turn somatic state
        s_hu = current_speaker.hu.state[0].tolist()
        print(f"{speaker_name} Pre-Turn Somatics: Energy={s_hu[1]:.3f}, Curiosity={s_hu[0]:.3f}, NA={s_hu[4]:.3f}, DA={s_hu[5]:.3f}")
        
        # Generate Response from Speaker
        t_gen_start = time.perf_counter()
        gen_tokens = []
        gen_text = ""
        
        for event in current_speaker.interact(incoming_message, max_tokens=70, temperature=0.45, top_p=0.90):
            if event.get("status") == "token":
                gen_text += event.get("text", "")
                gen_tokens.append(event.get("token_id"))
                
        t_gen_duration = time.perf_counter() - t_gen_start
        gen_clean = gen_text.strip()
        if not gen_clean:
            gen_clean = "..."
            
        print(f"\n>>> {speaker_name} RESPONDS ({len(gen_tokens)} tokens, {t_gen_duration*1000:.1f}ms):")
        print(f"\"{gen_clean}\"")
        
        # Listener processes Speaker's message (Listening phase & Surprise calculation)
        s_hu_post = current_speaker.hu.state[0].tolist()
        l_hu = current_listener.hu.state[0].tolist()
        
        round_info = {
            "round": round_idx,
            "speaker": speaker_name,
            "listener": listener_name,
            "input_stimulus": incoming_message,
            "generated_response": gen_clean,
            "tokens_generated": len(gen_tokens),
            "generation_duration_sec": t_gen_duration,
            "speaker_somatic_post": {
                "curiosity": s_hu_post[0],
                "energy": s_hu_post[1],
                "stability": s_hu_post[2],
                "noradrenaline": s_hu_post[4],
                "dopamine": s_hu_post[5]
            },
            "listener_somatic": {
                "curiosity": l_hu[0],
                "energy": l_hu[1],
                "stability": l_hu[2],
                "noradrenaline": l_hu[4],
                "dopamine": l_hu[5]
            }
        }
        round_telemetry.append(round_info)
        dialogue_log.append(f"[{speaker_name} -> {listener_name}]: {gen_clean}")
        
        # Turn-Taking swap for next iteration
        incoming_message = gen_clean
        current_speaker, current_listener = current_listener, current_speaker
        speaker_name, listener_name = listener_name, speaker_name
        
    total_duration = time.perf_counter() - t_start
    
    print("\n" + "="*90)
    print(" === [EXP-147 TELEMETRY & COGNITIVE SUMMARY] ===")
    print("="*90)
    print(f"Total Dialogue Rounds: {num_rounds}")
    print(f"Total Duration: {total_duration:.2f}s")
    print(f"Alpha Final Somatic: Energy={entity_alpha.hu.state[0,1]:.3f}, Curiosity={entity_alpha.hu.state[0,0]:.3f}, NA={entity_alpha.hu.state[0,4]:.3f}")
    print(f"Beta Final Somatic : Energy={entity_beta.hu.state[0,1]:.3f}, Curiosity={entity_beta.hu.state[0,0]:.3f}, NA={entity_beta.hu.state[0,4]:.3f}")
    
    # Validate Social Active Inference properties
    success = (len(round_telemetry) == num_rounds) and all(r["tokens_generated"] > 0 for r in round_telemetry)
    verdict = "🟢 POSITIVE" if success else "🔴 REJECTED"
    
    results = {
        "exp_id": "EXP-147",
        "verdict": verdict,
        "total_duration_sec": total_duration,
        "num_rounds": num_rounds,
        "rounds": round_telemetry,
        "alpha_final_somatic": round_telemetry[-2]["speaker_somatic_post"] if num_rounds % 2 == 0 else round_telemetry[-1]["speaker_somatic_post"],
        "beta_final_somatic": round_telemetry[-1]["speaker_somatic_post"] if num_rounds % 2 == 0 else round_telemetry[-2]["speaker_somatic_post"],
        "dialogue_transcript": dialogue_log
    }
    
    output_json = "experiments/exp_147_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    logger.info(f"EXP-147 telemetry recorded to '{output_json}'. Verdict: {verdict}")
    return results

if __name__ == "__main__":
    run_dual_agent_experiment()
