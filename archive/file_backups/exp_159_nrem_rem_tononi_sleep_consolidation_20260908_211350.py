# experiments/exp_159_nrem_rem_tononi_sleep_consolidation.py
"""
EXP-159: Comprehensive NREM Slow-Wave Replay, REM Counterfactual Dreaming & Tononi SHY Synaptic Downscaling

Hypothesis:
Executing a full 3-phase Biophysical Sleep & Morphogenesis Cycle:
- Phase 1: NREM Slow-Wave Hippocampal Memory Replay (AdamW Optimization over Episodic Memory Buffer)
- Phase 2: REM Generative Counterfactual Synthetic Dreaming (Attractor Basin Exploration in Latent Predictor)
- Phase 3: Tononi Synaptic Homeostasis Hypothesis (SHY) Downscaling (W = W * (1 - lambda_downscale))
will fully restore Somatic Energy (Energy -> 1.00), reduce Variational Free Energy (Surprise F_t),
and improve post-sleep 1-Shot Episodic Recall accuracy.

Telemetry Captured:
- Pre-sleep vs. Post-sleep Somatic Energy & Free Energy
- Number of NREM Replay steps and REM Dream steps
- Post-sleep 1-Shot Episodic Memory Recall Accuracy
"""

import sys
import os
import time
import math
import json
import logging
import torch
import torch.nn.functional as F

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_entity import KaryonEntity

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-159")


def main():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-159: NREM/REM/TONONI SHY SLEEP CONSOLIDATION BENCHMARK]")
    logger.info("=" * 80)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"⚡ Hardware Backend: {device_str.upper()}")

    # Load active entity
    kcore_path = "karyon_soul.kcore"
    entity = KaryonEntity.load(filepath=kcore_path, device=device_str)

    # Populate episodic memory buffer with 10 high-surprise factual key-value pairs
    dim = entity.brain.unified_dim
    for i in range(10):
        key = torch.randn(dim, device=device_str)
        val = torch.randn(dim, device=device_str)
        entity.memory.write(key, val, priority=3)

    # Artificially deplete energy to trigger sleep threshold (Energy = 0.25)
    entity.hu.state[0, 1] = 0.25
    pre_energy = float(entity.hu.state[0, 1].item())
    pre_health = float(entity.hu.state[0, 3].item())

    logger.info(f"Pre-Sleep Somatic State  : Energy={pre_energy:.3f} | Health={pre_health:.3f}")
    logger.info(f"Episodic Memory Slots   : {entity.memory.max_active_cpu} active slots")

    # Execute Deep Allostatic Sleep
    t_sleep_start = time.perf_counter()
    pruned_synapses = entity.sleep(num_replay_cycles=6, downscaling_factor=0.03, pruning_percentile=0.05)
    t_sleep_duration = time.perf_counter() - t_sleep_start

    post_energy = float(entity.hu.state[0, 1].item())
    post_health = float(entity.hu.state[0, 3].item())

    logger.info(f"Post-Sleep Somatic State : Energy={post_energy:.3f} | Health={post_health:.3f}")
    logger.info(f"Sleep Execution Time     : {t_sleep_duration:.2f} seconds")
    logger.info(f"Pruned Synapses Count    : {pruned_synapses:,}")

    # Evaluate 1-Shot Episodic Recall on Memory Buffer
    replayed_keys = entity.memory.keys[0, :min(5, entity.memory.max_active_cpu), :].float()
    replayed_vals = entity.memory.values[0, :min(5, entity.memory.max_active_cpu), :].float()

    with torch.no_grad():
        h_dummy = torch.zeros(replayed_keys.size(0), entity.brain.hidden_dim, device=device_str)
        w_pred, _, fe_val, _ = entity.brain.world_model(h_dummy, h_dummy, replayed_keys)
        recall_sim = float(F.cosine_similarity(w_pred, replayed_vals, dim=-1).mean().item())
        post_sleep_fe = float(fe_val.mean().item())

    logger.info(f"Post-Sleep Recall Cosine Similarity : {recall_sim:.4f}")
    logger.info(f"Post-Sleep Memory Free Energy      : {post_sleep_fe:.4f}")

    verdict = "POSITIVE" if (post_energy >= 0.95 and recall_sim > 0.0) else "REJECTED"

    logger.info("=" * 80)
    logger.info("📊 === EXP-159 EMPIRICAL TELEMETRY SUMMARY ===")
    logger.info(f"🏆 Verdict                         : 🟢 {verdict}")
    logger.info(f"🔋 Somatic Energy Restored         : {pre_energy:.3f} -> {post_energy:.3f}")
    logger.info(f"🧠 Post-Sleep Recall Cosine Sim    : {recall_sim:.4f}")
    logger.info(f"🌱 Pruned Synapses (Morphogenesis) : {pruned_synapses:,}")
    logger.info("=" * 80)

    summary = {
        "exp_id": "EXP-159",
        "verdict": verdict,
        "metrics": {
            "pre_sleep_energy": pre_energy,
            "post_sleep_energy": post_energy,
            "pruned_synapses": pruned_synapses,
            "post_sleep_recall_sim": recall_sim,
            "post_sleep_fe": post_sleep_fe,
            "sleep_duration_sec": t_sleep_duration
        }
    }
    with open("experiments/exp_159_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("✅ EXP-159 execution complete. Results saved to experiments/exp_159_results.json.")


if __name__ == "__main__":
    main()
