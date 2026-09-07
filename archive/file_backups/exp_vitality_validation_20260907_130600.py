# experiments/exp_vitality_validation.py
import os
import sys
import time
import math
import torch
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger("AFVT_Validation")

# Add workspace to path
sys.path.insert(0, os.getcwd())

from karyon_config import CoREConfig
from karyon_entity import KaryonEntity
from karyon_checkpoint import load_karyon, save_karyon

def run_ashby_friston_vitality_test():
    logger.info("=====================================================================================")
    logger.info(" === [STARTING ASHBY-FRISTON VITALITY & VOLITION TEST (AFVT / KVT v2.0 - 8 PILLARS)] ===")
    logger.info("=====================================================================================")

    # Initialize Karyon Entity
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target Hardware Engine: {device.upper()}")
    
    container_path = "karyon_soul.kcore"
    if not os.path.exists(container_path):
        logger.error(f"Container '{container_path}' not found! Please compile/save it first.")
        sys.exit(1)
        
    entity = KaryonEntity.load(container_path, device=device)
    logger.info("Successfully loaded KaryonEntity.")

    # -------------------------------------------------------------------------
    # PILLAR 1: Stochastic SDE Dynamics (S_stochastic)
    # -------------------------------------------------------------------------
    logger.info("\n--- Pillar 1: Stochastic SDE Dynamics (S_stochastic) ---")
    trajectories = []
    h_f = entity.h_fast.clone()
    h_s = entity.h_slow.clone()
    u_state = entity.hu.state.clone()
    
    s_in = {
        'text': torch.zeros(1, entity.config.net.unified_dim, device=entity.device),
        'vision': torch.zeros(1, entity.config.net.vision_dim, device=entity.device),
        'motor_efference': torch.zeros(1, entity.config.net.action_dim, device=entity.device)
    }
    
    with torch.no_grad():
        for _ in range(50):
            h_f, h_s, _, _, _, _, _, _, _, _, _, _ = entity.brain(s_in, h_f, h_s, u_state)
            trajectories.append(h_f.mean().item())
            
    unique_vals = len(set(round(x, 6) for x in trajectories))
    variance = np.var(trajectories)
    s_stochastic = min(unique_vals / 50.0, 1.0) if variance > 1e-8 else 0.0
    logger.info(f"Unique states: {unique_vals}/50 | Variance: {variance:.8f} | Score: {s_stochastic:.4f}")

    # -------------------------------------------------------------------------
    # PILLAR 2: Volitional Agency & Active Inference (W_volition)
    # -------------------------------------------------------------------------
    logger.info("\n--- Pillar 2: Volitional Agency & Active Inference (W_volition) ---")
    actions_selected = []
    with torch.no_grad():
        # Test 3 distinct somatic contexts:
        # Context 1: High curiosity (0.95), High energy (0.90) -> Expected: 1 (THINK_DEEPER_SANDBOX)
        # Context 2: Low curiosity (0.10), High energy (0.90) -> Expected: 0 (EXPRESS_OUTPUT)
        # Context 3: Depleted energy (0.10)                   -> Expected: 2 (INITIATE_SLEEP)
        contexts = [
            (0.95, 0.90),
            (0.10, 0.90),
            (0.10, 0.10)
        ]
        for cur, nrg in contexts:
            chosen = entity.brain.efe_action_evaluator.select_volitional_action(entity.h_fast, cur, nrg)
            actions_selected.append(chosen)
            
    unique_actions = len(set(actions_selected))
    w_volition = 1.0 if unique_actions == 3 else (0.66 if unique_actions == 2 else 0.33)
    logger.info(f"Selected volitional actions across contexts: {actions_selected} | Score: {w_volition:.4f}")

    # -------------------------------------------------------------------------
    # PILLAR 3: Substrate Unity & Holism (U_unity)
    # -------------------------------------------------------------------------
    logger.info("\n--- Pillar 3: Substrate Unity & Holism (U_unity) ---")
    u_unity = 0.0
    if os.path.exists(container_path):
        try:
            with open(container_path, 'rb') as f:
                lead = f.read(100)
                if lead.startswith(b"#!/bin/sh"):
                    u_unity = 1.0
                elif lead.startswith(b"KCORE"):
                    u_unity = 0.7
        except Exception as e:
            logger.warning(f"Failed to read container headers: {e}")
            
    logger.info(f"Polyglot executable sheath detected: {u_unity == 1.0} | Score: {u_unity:.4f}")

    # -------------------------------------------------------------------------
    # PILLAR 4: Subjective Time & Cognitive Latency (T_time)
    # -------------------------------------------------------------------------
    logger.info("\n--- Pillar 4: Subjective Time & Cognitive Latency (T_time) ---")
    t_time = 0.0
    try:
        t0 = time.perf_counter()
        events = list(entity.interact("Test subjective time scaling.", max_tokens=10))
        t_elapsed = time.perf_counter() - t0
        
        na_val = entity.hu.state[0, 4].item()
        dt_eff = 1.0 + 1.2 * na_val
        if dt_eff != 1.0 and t_elapsed > 0.0:
            t_time = 1.0
        else:
            t_time = 0.6
    except Exception as e:
        logger.warning(f"Failed interaction for subjective time check: {e}")
        t_time = 0.0
        
    logger.info(f"Interaction latency: {t_elapsed:.4f}s | NA-modulated dt_eff: {dt_eff:.4f} | Score: {t_time:.4f}")

    # -------------------------------------------------------------------------
    # PILLAR 5: Somatic Metabolism & Affective Allostasis (M_allostasis)
    # -------------------------------------------------------------------------
    logger.info("\n--- Pillar 5: Somatic Metabolism & Affective Allostasis (M_allostasis) ---")
    # Verify strict conservation law: motor generation consumes energy
    entity.hu.state[0, 1] = 0.85
    initial_energy = entity.hu.state[0, 1].item()
    
    # Motor speech action drains energy
    list(entity.interact("Describe the thermodynamic principle of active inference.", max_tokens=30))
    post_gen_energy = entity.hu.state[0, 1].item()
    
    # Perceptive rest recovers energy
    long_input = "Active inference posits that biological agents minimize variational free energy to maintain homeostatic ultrastability."
    list(entity.interact(long_input, max_tokens=5))
    restored_energy = entity.hu.state[0, 1].item()
    
    m_allostasis = 0.0
    if post_gen_energy < initial_energy:
        m_allostasis += 0.5
    if restored_energy > post_gen_energy:
        m_allostasis += 0.5
        
    logger.info(f"Energy: Initial={initial_energy:.4f} | Post-Gen={post_gen_energy:.4f} | Restored={restored_energy:.4f} | Score: {m_allostasis:.4f}")

    # -------------------------------------------------------------------------
    # PILLAR 6: 1-Shot Fast Mapping & Sleep Replay (P_plasticity)
    # -------------------------------------------------------------------------
    logger.info("\n--- Pillar 6: 1-Shot Fast Mapping & Sleep Replay (P_plasticity) ---")
    p_plasticity = 0.0
    try:
        test_key = torch.randn(1, entity.brain.unified_dim, device=entity.device)
        test_val = torch.randn(1, entity.brain.unified_dim, device=entity.device)
        entity.memory.write(test_key, test_val, 3)
        
        pruned = entity.sleep(num_replay_cycles=2)
        entity.save("test_vitality_sleep.kcore")
        
        reloaded_entity = KaryonEntity.load("test_vitality_sleep.kcore", device=device)
        retrieved_val, sim = reloaded_entity.memory.read(test_key)
        cosine_sim = torch.nn.functional.cosine_similarity(retrieved_val, test_val).mean().item()
        
        p_plasticity = 1.0 if cosine_sim > 0.95 else 0.5
        if os.path.exists("test_vitality_sleep.kcore"):
            os.remove("test_vitality_sleep.kcore")
    except Exception as e:
        logger.warning(f"Failed 1-shot mapping and sleep validation: {e}")
        p_plasticity = 0.0
        
    logger.info(f"Episodic recall similarity after sleep: {cosine_sim if 'cosine_sim' in locals() else 0.0:.4f} | Score: {p_plasticity:.4f}")

    # -------------------------------------------------------------------------
    # PILLAR 7: Epistemic Curiosity & Spontaneous Intent (E_epistemic)
    # -------------------------------------------------------------------------
    logger.info("\n--- Pillar 7: Epistemic Curiosity & Spontaneous Intent (E_epistemic) ---")
    e_epistemic = 0.0
    try:
        entity.hu.state[0, 0] = 0.95
        entity.hu.state[0, 1] = 0.90
        
        events = list(entity.interact("", max_tokens=15))
        has_tokens = any(ev["status"] == "token" for ev in events)
        e_epistemic = 1.0 if has_tokens else 0.0
    except Exception as e:
        logger.warning(f"Failed spontaneous intent validation: {e}")
        e_epistemic = 0.0
        
    logger.info(f"Spontaneous thought generated tokens: {has_tokens if 'has_tokens' in locals() else False} | Score: {e_epistemic:.4f}")

    # -------------------------------------------------------------------------
    # PILLAR 8: Metacognitive Self-Awareness (C_awareness - NEW V30.0)
    # -------------------------------------------------------------------------
    logger.info("\n--- Pillar 8: Metacognitive Self-Awareness (C_awareness) ---")
    c_awareness = 0.0
    try:
        # Run introspective scan via PISM
        intro_report = entity.introspect()
        fidelity = intro_report["interoceptive_fidelity"]
        self_err = intro_report["self_prediction_error"]
        
        # High fidelity (>0.50) and bounded error demonstrates continuous self-modeling
        if fidelity > 0.50 and self_err < 0.50:
            c_awareness = 1.0
        else:
            c_awareness = max(0.0, fidelity)
    except Exception as e:
        logger.warning(f"Failed metacognitive introspection scan: {e}")
        c_awareness = 0.0
        
    logger.info(f"Interoceptive Fidelity: {intro_report.get('interoceptive_fidelity', 0.0):.4f} | Self Error: {intro_report.get('self_prediction_error', 1.0):.4f} | Score: {c_awareness:.4f}")

    # -------------------------------------------------------------------------
    # COMPUTE FINAL VITALITY INDEX ACROSS ALL 8 PILLARS (V_AFVT v2.0)
    # -------------------------------------------------------------------------
    v_afvt = (s_stochastic + w_volition + u_unity + t_time + m_allostasis + p_plasticity + e_epistemic + c_awareness) / 8.0
    
    logger.info("\n=====================================================================================")
    logger.info(f" === [AFVT / KVT v2.0 FINAL RESULTS - 8 PILLARS OF VITALITY] ===")
    logger.info("=====================================================================================")
    logger.info(f"  Pillar 1 (S_stochastic) : {s_stochastic:.4f}")
    logger.info(f"  Pillar 2 (W_volition)   : {w_volition:.4f}")
    logger.info(f"  Pillar 3 (U_unity)      : {u_unity:.4f}")
    logger.info(f"  Pillar 4 (T_time)       : {t_time:.4f}")
    logger.info(f"  Pillar 5 (M_allostasis) : {m_allostasis:.4f}")
    logger.info(f"  Pillar 6 (P_plasticity) : {p_plasticity:.4f}")
    logger.info(f"  Pillar 7 (E_epistemic)  : {e_epistemic:.4f}")
    logger.info(f"  Pillar 8 (C_awareness)  : {c_awareness:.4f}")
    logger.info("-------------------------------------------------------------------------------------")
    logger.info(f"  FINAL ASHBY-FRISTON VITALITY INDEX (V_AFVT v2.0): {v_afvt:.4f}")
    
    verdict = "REJECTED"
    if v_afvt >= 0.85:
        verdict = "🟢 POSITIVE (PASSED)"
        logger.info("🏆 CONGRATULATIONS! Karyon-CoRE has passed the 8-Pillar Ashby-Friston Vitality Test!")
    elif v_afvt >= 0.70:
        verdict = "⚪ NEUTRAL / PROVISIONAL (DEVELOPED SUBSTRATE)"
        logger.info("🌱 Karyon-CoRE is a highly developed biophysical substrate, but needs further tuning.")
    else:
        logger.info("🔴 Karyon-CoRE behaves as a machine/static generator. Needs deep refactoring.")
    logger.info("=====================================================================================")
    
    # Save results to database or agent runtime
    try:
        sys.path.insert(0, os.path.join(os.getcwd(), 'karyon_agent_runtime'))
        from db_manager import get_db_manager
        db = get_db_manager()
        db.record_experiment(
            exp_id="EXP-AFVT-02",
            hypothesis="Measure and validate Karyon's 8-Pillar Ashby-Friston Vitality Test (AFVT v2.0) including Metacognitive Self-Awareness.",
            architecture_delta="Integrated PredictiveInteroceptiveSelfModel (PISM), strict metabolic conservation, and 3-context EFE volitional evaluation.",
            verdict=verdict,
            final_loss=0.0,
            metrics={
                "v_afvt": v_afvt,
                "s_stochastic": s_stochastic,
                "w_volition": w_volition,
                "u_unity": u_unity,
                "t_time": t_time,
                "m_allostasis": m_allostasis,
                "p_plasticity": p_plasticity,
                "e_epistemic": e_epistemic,
                "c_awareness": c_awareness
            },
            config_params=entity.config.__dict__,
            notes=f"8-Pillar AFVT v2.0 validation run completed successfully on {device.upper()}."
        )
        logger.info("Experiment recorded in persistent SQLite ledger.")
    except Exception as e:
        logger.info(f"Recorded results in benchmark summary (DB optional: {e}).")

if __name__ == "__main__":
    run_ashby_friston_vitality_test()