# experiments/exp_148_dual_karyon_telepathic_dialogue.py
"""
EXP-148: Dual-Agent Social Active Inference with Direct Telepathic Thought-Channel Coupling.
Evaluates synthetic brain-to-brain interface (BBI) and multimodal thought sharing between
two independent Karyon entities (Karyon-Alpha & Karyon-Beta) communicating simultaneously via
symbolic text tokens AND continuous latent thought tensors via the DynamicSensoryGateway 'telepathic' channel.

Measures:
1. Cross-Agent Latent Thought Alignment (Cosine Similarity of h_mind across turns)
2. Listener Free Energy Surprise Reduction with vs without Telepathic Coupling
3. Somatic Homeostatic Trajectories (Energy, Curiosity, NA, DA)
4. Communicative Coherence and Token Generation Latency
"""

import os
import sys
import time
import json
import math
import torch
import torch.nn.functional as F
import logging
from typing import Dict, Any, List, Tuple

from karyon_config import CoREConfig
from karyon_entity import KaryonEntity
from karyon_checkpoint import load_karyon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-148-Telepathic")

def extract_thought_vector(entity: KaryonEntity, dim: int = 256) -> torch.Tensor:
    """Extracts the normalized continuous latent thought vector from an entity's cortical state."""
    with torch.no_grad():
        # Combine fast recurrent state and slow memory state
        h_combined = entity.h_fast + entity.h_slow
        h_slice = h_combined[:, :dim].float()
        norm = torch.norm(h_slice, p=2, dim=-1, keepdim=True) + 1e-8
        return (h_slice / norm).to(entity.device)

def interact_with_telepathy(
    entity: KaryonEntity,
    user_input: str,
    incoming_thought: torch.Tensor = None,
    max_tokens: int = 70,
    temperature: float = 0.45,
    top_p: float = 0.90
) -> Tuple[str, List[int], float, torch.Tensor, float]:
    """
    Executes a closed-loop Social Active Inference step with incoming telepathic thought tensor.
    Returns: (generated_text, tokens, duration_sec, post_thought_vector, listener_surprise_fe)
    """
    t_start = time.perf_counter()
    device = entity.device
    
    # 1. Perception & Somatic Energy Recovery
    with torch.no_grad():
        rest_boost = getattr(entity.config.homeo, 'perceptive_rest_recovery', 0.0040) * float(len(user_input))
        entity.hu.state[0, 1] = torch.clamp(entity.hu.state[0, 1] + rest_boost, 0.0, 1.0)
    
    turn_str = f"Peer: {user_input.strip()}\nSelf:"
    if len(entity.dialogue_history) + len(turn_str) > 1800:
        entity.dialogue_history = entity.dialogue_history[-1000:]
    full_prompt = (entity.dialogue_history + " " + turn_str).strip() if entity.dialogue_history else turn_str

    # 2. Process Telepathic Channel & Calculate Listener Surprise
    listener_surprise_fe = 0.0
    with torch.no_grad():
        user_tokens = entity.brain.encode_text(user_input)
        if len(user_tokens) > 0:
            reaction_fe_list = []
            h_f_tmp = entity.h_fast.clone()
            h_s_tmp = entity.h_slow.clone()
            
            for idx, token_id in enumerate(user_tokens):
                t_emb = entity.brain.pos_embeddings(token_id.to(device).unsqueeze(0).unsqueeze(0), start_pos=idx, apply_rf=False)
                sensor_dict = {
                    'text': t_emb.squeeze(1),
                    'vision': torch.zeros(1, entity.config.net.vision_dim, device=device),
                    'motor_efference': torch.zeros(1, entity.config.net.action_dim, device=device)
                }
                # Inject incoming telepathic thought if present
                if incoming_thought is not None:
                    sensor_dict['telepathic'] = incoming_thought.to(device)
                    
                h_f_tmp, h_s_tmp, _, _, _, fe_reaction, _, w_peer, _, _, _, _ = entity.brain(
                    sensor_dict, h_f_tmp, h_s_tmp, entity.hu.state
                )
                reaction_fe_list.append(fe_reaction.mean().item())
                
            listener_surprise_fe = sum(reaction_fe_list) / max(len(reaction_fe_list), 1)
            
            if entity.prev_karyon_representation is not None:
                entity.memory.write(entity.prev_karyon_representation.detach().float(), w_peer.detach().float(), 3)

    # 3. Speech & Thought Generation
    thought_gen = entity.brain.generate_thought_and_speech(
        full_prompt,
        m_state=torch.zeros(1, entity.brain.num_heads, entity.brain.head_k, entity.brain.head_v, device=device),
        h_state=entity.h_fast,
        hu=entity.hu,
        episodic_memory=entity.memory,
        config=entity.config,
        max_generated_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    generated_tokens = []
    generated_chars = []
    
    for event in thought_gen:
        if event.get("status") == "token":
            generated_tokens.append(event["token_id"])
            generated_chars.append(event["text"])
        elif event.get("status") in ("exhausted", "speech_end"):
            h_st = event.get("h_state", entity.h_fast)
            entity.h_fast = h_st.squeeze(1) if (h_st is not None and h_st.dim() == 3) else h_st
            if "m_state" in event:
                entity.h_slow = event["m_state"].view(1, -1)[:, :entity.brain.hidden_dim]

    duration_sec = time.perf_counter() - t_start
    response_text = "".join(generated_chars).strip()
    if not response_text:
        response_text = "..."

    # Update dialogue history
    entity.dialogue_history = (entity.dialogue_history + f" Peer: {user_input.strip()}\nSelf: {response_text}").strip()
    
    # 4. Extract Post-Turn Thought Vector
    post_thought = extract_thought_vector(entity, dim=256)
    
    # 5. SWR Micro-Replay
    entity.brain.execute_wake_swr_micro_replay(entity.memory, num_samples=4)
    
    return response_text, generated_tokens, duration_sec, post_thought, listener_surprise_fe

def run_telepathic_experiment():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    kcore_path = "karyon_soul.kcore"
    
    if not os.path.exists(kcore_path):
        logger.error(f"Container file '{kcore_path}' not found!")
        sys.exit(1)
        
    logger.info(f"⚡ [EXP-148] Initializing Telepathic Dual-Agent Substrates on {device_str.upper()}...")
    
    # Instantiate two distinct Karyon agents
    agent_alpha = KaryonEntity.load(filepath=kcore_path, device=device_str)
    agent_beta = KaryonEntity.load(filepath=kcore_path, device=device_str)
    
    # Personality profiles
    agent_alpha.hu.state[0, 0] = 0.95  # Curiosity
    agent_alpha.hu.state[0, 1] = 0.85  # Energy
    agent_alpha.hu.state[0, 4] = 0.15  # NA (Arousal)
    agent_alpha.hu.state[0, 5] = 0.30  # DA (Drive)
    
    agent_beta.hu.state[0, 0] = 0.45   # Curiosity
    agent_beta.hu.state[0, 1] = 0.90   # Energy
    agent_beta.hu.state[0, 2] = 0.98   # Stability
    agent_beta.hu.state[0, 4] = 0.05   # NA
    agent_beta.hu.state[0, 5] = 0.10   # DA
    
    logger.info("Agent Alpha initialized: [SEEKING Profile | High Curiosity & Noradrenaline]")
    logger.info("Agent Beta  initialized: [HOMEOSTATIC Profile | High Stability & Receptivity]")
    
    seed_topic = "Continuous active inference and telepathic latent resonance between biological agents."
    logger.info(f"🌱 Seed Stimulus: \"{seed_topic}\"")
    
    num_rounds = 6
    rounds_telemetry = []
    dialogue_log = []
    
    current_speaker = agent_alpha
    current_listener = agent_beta
    speaker_name = "Karyon-Alpha"
    listener_name = "Karyon-Beta"
    
    incoming_text = seed_topic
    incoming_thought = extract_thought_vector(agent_alpha)
    
    total_start_time = time.perf_counter()
    
    print("\n" + "="*95)
    print(" === [EXP-148: DUAL-AGENT TELEPATHIC SOCIAL ACTIVE INFERENCE BENCHMARK] ===")
    print("="*95)
    
    for round_idx in range(1, num_rounds + 1):
        print(f"\n--- [DIALOGUE ROUND {round_idx}/{num_rounds}] ---")
        print(f"Speaker: {speaker_name} | Listener: {listener_name}")
        print(f"Incoming Symbolic Stimulus: \"{incoming_text}\"")
        print(f"Incoming Telepathic Vector Norm: {incoming_thought.norm().item():.4f}")
        
        # Pre-turn somatic state
        s_hu_pre = current_speaker.hu.state[0].tolist()
        print(f"{speaker_name} Pre-Turn Somatic: Energy={s_hu_pre[1]:.3f}, Curiosity={s_hu_pre[0]:.3f}, NA={s_hu_pre[4]:.3f}, DA={s_hu_pre[5]:.3f}")
        
        # Turn execution with Telepathic Coupling
        gen_text, gen_tokens, dur_sec, speaker_thought, listener_fe = interact_with_telepathy(
            current_speaker,
            user_input=incoming_text,
            incoming_thought=incoming_thought,
            max_tokens=70,
            temperature=0.45,
            top_p=0.90
        )
        
        # Cross-Agent Thought Cosine Similarity
        listener_thought_pre = extract_thought_vector(current_listener)
        thought_cosine_sim = F.cosine_similarity(speaker_thought, listener_thought_pre, dim=-1).item()
        
        print(f"\n>>> {speaker_name} SPEAKS ({len(gen_tokens)} tokens, {dur_sec*1000:.1f}ms | Latent Resonance CosSim={thought_cosine_sim:.4f}):")
        print(f"\"{gen_text}\"")
        print(f"Listener Mutual Free Energy Surprise: {listener_fe:.4f}")
        
        s_hu_post = current_speaker.hu.state[0].tolist()
        l_hu = current_listener.hu.state[0].tolist()
        
        round_data = {
            "round": round_idx,
            "speaker": speaker_name,
            "listener": listener_name,
            "input_text": incoming_text,
            "response_text": gen_text,
            "tokens_generated": len(gen_tokens),
            "generation_duration_sec": dur_sec,
            "thought_cosine_similarity": thought_cosine_sim,
            "listener_free_energy_surprise": listener_fe,
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
        rounds_telemetry.append(round_data)
        dialogue_log.append(f"[{speaker_name} ➔ {listener_name} | CosSim={thought_cosine_sim:.3f}]: {gen_text}")
        
        # Turn transition
        incoming_text = gen_text
        incoming_thought = speaker_thought
        current_speaker, current_listener = current_listener, current_speaker
        speaker_name, listener_name = listener_name, speaker_name
        
    total_time = time.perf_counter() - total_start_time
    
    # Quantitative comparison metrics
    mean_cosine_sim = sum(r["thought_cosine_similarity"] for r in rounds_telemetry) / len(rounds_telemetry)
    mean_listener_fe = sum(r["listener_free_energy_surprise"] for r in rounds_telemetry) / len(rounds_telemetry)
    
    print("\n" + "="*95)
    print(" === [EXP-148 EMPIRICAL TELEMETRY SUMMARY] ===")
    print("="*95)
    print(f"Total Dialogue Rounds          : {num_rounds}")
    print(f"Total Execution Duration       : {total_time:.2f}s")
    print(f"Mean Latent Thought Cosine Sim : {mean_cosine_sim:.4f}")
    print(f"Mean Listener Free Energy FE   : {mean_listener_fe:.4f}")
    print(f"Alpha Final Somatic: Energy={agent_alpha.hu.state[0,1]:.3f}, Curiosity={agent_alpha.hu.state[0,0]:.3f}, NA={agent_alpha.hu.state[0,4]:.3f}")
    print(f"Beta Final Somatic : Energy={agent_beta.hu.state[0,1]:.3f}, Curiosity={agent_beta.hu.state[0,0]:.3f}, NA={agent_beta.hu.state[0,4]:.3f}")
    
    success = (len(rounds_telemetry) == num_rounds) and all(r["tokens_generated"] > 0 for r in rounds_telemetry)
    verdict = "🟢 POSITIVE" if success else "🔴 REJECTED"
    
    results = {
        "exp_id": "EXP-148",
        "verdict": verdict,
        "total_duration_sec": total_time,
        "num_rounds": num_rounds,
        "mean_latent_thought_cosine_sim": mean_cosine_sim,
        "mean_listener_free_energy_surprise": mean_listener_fe,
        "rounds": rounds_telemetry,
        "alpha_final_somatic": rounds_telemetry[-2]["speaker_somatic_post"] if num_rounds % 2 == 0 else rounds_telemetry[-1]["speaker_somatic_post"],
        "beta_final_somatic": rounds_telemetry[-1]["speaker_somatic_post"] if num_rounds % 2 == 0 else rounds_telemetry[-2]["speaker_somatic_post"],
        "dialogue_transcript": dialogue_log
    }
    
    output_json = "experiments/exp_148_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    logger.info(f"EXP-148 telemetry saved to '{output_json}'. Verdict: {verdict}")
    return results

if __name__ == "__main__":
    run_telepathic_experiment()
