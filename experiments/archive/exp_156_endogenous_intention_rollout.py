# experiments/exp_156_endogenous_intention_rollout.py
"""
EXP-156: Endogenous Volitional Intention Engine in Latent Predictor Sandbox
         vs. Reactive Unplanned Thought Baseline

Hypothesis:
Equipping Karyon-CoRE's System 2 Mental Sandbox (`LatentPredictor`) with an
Endogenous Volitional Intention Engine—which executes K=4 parallel counterfactual
rollout steps searching over candidate latent trajectories to minimize Expected Free Energy
(EFE = D_KL(Q(z) || P(z)) + Ambiguity + Risk)—will generate self-initiated goal-directed
internal intentions (u_intent) during spontaneous thought.
This will lead to higher semantic intentionality, reduced variational surprise on multi-turn
spontaneous transitions, and proactive cognitive initiative without external prompts.

Architecture Delta:
1. `EndogenousIntentionEngine`:
   - Generates K=16 candidate volitional delta vectors in latent space
   - Unrolls K=4 parallel System 2 forward simulation steps in LatentPredictor
   - Computes Expected Free Energy (EFE_k) per candidate trajectory
   - Applies Softmax Boltzmann selection weighted by Somatic Precision (Dopamine/Noradrenaline)
   - Emits the selected optimal intent vector u_intent to modulate thalamic routing.
2. Comparative audit between Unplanned Reactive Monologue vs. Endogenous Intention Rollout Monologue
   across Expected Free Energy (EFE), Latent Cosine Directionality, and Somatic Vitality.
"""

import sys
import os
import time
import math
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_entity import KaryonEntity
from karyon_agent import CoREAgent

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-156")


class EndogenousIntentionEngine(nn.Module):
    """
    Endogenous Volitional Intention Rollout Engine in System 2 Mental Sandbox.
    """
    def __init__(self, agent_brain: CoREAgent, num_candidates: int = 16, rollout_steps: int = 4):
        super().__init__()
        self.brain = agent_brain
        self.world_model = agent_brain.world_model
        self.num_candidates = num_candidates
        self.rollout_steps = rollout_steps

    def generate_endogenous_intention(self, h_curr: torch.Tensor, u_t: torch.Tensor) -> dict:
        """
        h_curr: [1, H]
        u_t: [1, 6]
        """
        t0 = time.perf_counter()
        
        # Parallel candidate rollout search via world model (LatentPredictor)
        w_dummy = torch.zeros(1, self.brain.unified_dim, device=self.brain.device)
        best_thought_h, min_efe, duration_ms = self.world_model.parallel_rollout_search(
            h_curr, w_dummy, steps=self.rollout_steps
        )

        # Calculate Intent Delta vector in hidden space
        intent_vector = best_thought_h - h_curr
        intent_magnitude = float(intent_vector.norm(p=2, dim=-1).item())

        t_lat = (time.perf_counter() - t0) * 1000.0

        return {
            "best_thought_h": best_thought_h,
            "intent_vector": intent_vector,
            "intent_magnitude": intent_magnitude,
            "expected_free_energy": float(min_efe.mean().item()),
            "rollout_latency_ms": t_lat
        }


def main():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-156: ENDOGENOUS INTENTION ROLLOUT BENCHMARK]")
    logger.info("=" * 80)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"⚡ Hardware Backend: {device_str.upper()}")

    # Load active entity
    kcore_path = "karyon_soul.kcore"
    entity = KaryonEntity.load(filepath=kcore_path, device=device_str)

    intention_engine = EndogenousIntentionEngine(entity.brain, num_candidates=16, rollout_steps=4)

    logger.info("\n>>> 1. BENCHMARKING UNPLANNED REACTIVE SPONTANEOUS THOUGHT <<<")
    t0 = time.perf_counter()
    res_reactive = entity.self_learn(num_sequences=2, seq_len=32)
    t_reactive_ms = (time.perf_counter() - t0) * 1000.0

    fe_reactive_init = res_reactive["initial_free_energy"]
    fe_reactive_final = res_reactive["final_free_energy"]

    logger.info(f"Reactive Monologue Latency   : {t_reactive_ms:.1f} ms")
    logger.info(f"Initial Free Energy (Surprise): {fe_reactive_init:.4f}")
    logger.info(f"Final Free Energy (Surprise)  : {fe_reactive_final:.4f}")

    logger.info("\n>>> 2. BENCHMARKING ENDOGENOUS INTENTION ROLLOUT MONOLOGUE <<<")
    h_curr = entity.h_fast.clone()
    u_t = entity.hu.state.clone()

    intent_res = intention_engine.generate_endogenous_intention(h_curr, u_t)
    best_thought = intent_res["best_thought_h"]
    expected_fe = intent_res["expected_free_energy"]
    intent_mag = intent_res["intent_magnitude"]

    # Execute intent-modulated spontaneous turn
    entity.h_fast.copy_(0.70 * entity.h_fast + 0.30 * best_thought)
    
    t0 = time.perf_counter()
    res_intentional = entity.self_learn(num_sequences=2, seq_len=32)
    t_intentional_ms = (time.perf_counter() - t0) * 1000.0

    fe_intent_init = res_intentional["initial_free_energy"]
    fe_intent_final = res_intentional["final_free_energy"]

    logger.info(f"Intentional Rollout Latency  : {intent_res['rollout_latency_ms']:.1f} ms")
    logger.info(f"Expected Free Energy (EFE)   : {expected_fe:.4f}")
    logger.info(f"Intent Vector Magnitude      : {intent_mag:.4f}")
    logger.info(f"Post-Intent Initial FE       : {fe_intent_init:.4f}")
    logger.info(f"Post-Intent Final FE         : {fe_intent_final:.4f}")

    fe_reduction_delta = (fe_reactive_final - fe_intent_final)

    verdict = "POSITIVE" if (fe_intent_final <= fe_reactive_final and intent_mag > 0.05) else "NEUTRAL"

    logger.info("=" * 80)
    logger.info("📊 === EXP-156 EMPIRICAL TELEMETRY SUMMARY ===")
    logger.info(f"🏆 Verdict                       : 🟢 {verdict}")
    logger.info(f"🔮 Expected Free Energy (EFE)     : {expected_fe:.4f}")
    logger.info(f"⚡ Intent Vector Magnitude (||u||): {intent_mag:.4f}")
    logger.info(f"📉 Free Energy Reduction Gain    : {fe_reduction_delta:+.4f}")
    logger.info("=" * 80)

    summary = {
        "exp_id": "EXP-156",
        "verdict": verdict,
        "metrics": {
            "reactive_final_fe": fe_reactive_final,
            "intentional_final_fe": fe_intent_final,
            "fe_reduction_delta": fe_reduction_delta,
            "expected_free_energy": expected_fe,
            "intent_magnitude": intent_mag,
            "rollout_latency_ms": intent_res["rollout_latency_ms"]
        }
    }
    with open("experiments/exp_156_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("✅ EXP-156 execution complete. Results saved to experiments/exp_156_results.json.")


if __name__ == "__main__":
    main()
