# experiments/exp_158_noradrenergic_plasticity_gating.py
"""
EXP-158: Noradrenergic Synaptic Plasticity Gating in Local Fast Weights
         vs. Ungated Continuous Hebbian Update Baseline

Hypothesis:
Adding a sharp Noradrenergic Synaptic Gating Threshold (GABAergic/LC Gating):
  G_NA = sigmoid(12.0 * (NA_t - 0.15))
to `LocalNeuromodulatedPlasticityImpl`:
  dW_fast = G_NA * (lr * neuromodulation) * (post_err * pre_act^T)
will protect long-term fast weights from catastrophic erosion during low-arousal/mastered stream steps,
preserving memory retention accuracy on 1-shot episodic facts (retention accuracy > 95%) while
maintaining full plastic adaptation during high-surprise novelty events.

Architecture Delta:
1. `GatedLocalNeuromodulatedPlasticity`:
   - Computes Noradrenergic Gate: G_NA = torch.sigmoid(12.0 * (na_t - 0.15))
   - Modulated update: W_fast = 0.92 * W_fast + G_NA * lr * dW
2. Empirical benchmark comparing 1-Shot Episodic Memory Recall Accuracy before and after 50 routine
   un-surprising stream steps.
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

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-158")


class GatedLocalNeuromodulatedPlasticity(nn.Module):
    """
    Local Fast Weights with Sharp Noradrenergic Synaptic Gating (LC/NA Filter).
    """
    def __init__(self, in_features: int = 768, out_features: int = 768, lr: float = 0.08):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr

        self.W_base = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        self.register_buffer("W_fast", torch.zeros(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_eff = self.W_base + self.W_fast
        return F.linear(x, W_eff)

    def adapt_local_fast_weights(self, pre_act: torch.Tensor, post_err: torch.Tensor, na_t: float, da_t: float, use_gating: bool = True):
        with torch.no_grad():
            if use_gating:
                # Sharp Noradrenergic Gate: Active ONLY when Noradrenaline NA > 0.15 (High Surprise/Arousal)
                na_gate = float(torch.sigmoid(torch.tensor(12.0 * (na_t - 0.15))).item())
            else:
                # Baseline: Ungated continuous plastic leak
                na_gate = 1.0

            neuromodulation = 0.20 + 0.80 * na_t + 0.50 * da_t
            
            # dW = post_err * pre_act^T
            if post_err.dim() == 2 and pre_act.dim() == 2:
                dW = torch.bmm(post_err.unsqueeze(-1), pre_act.unsqueeze(1)).mean(0)
            else:
                dW = torch.matmul(post_err.t(), pre_act) / max(pre_act.size(0), 1)

            # Fast weight update
            self.W_fast.mul_(0.92) # Passive Hebbian decay
            self.W_fast.add_(dW * (self.lr * neuromodulation * na_gate))


def main():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-158: NORADRENERGIC PLASTICITY GATING BENCHMARK]")
    logger.info("=" * 80)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"⚡ Hardware Backend: {device_str.upper()}")

    # Load active entity
    kcore_path = "karyon_soul.kcore"
    entity = KaryonEntity.load(filepath=kcore_path, device=device_str)

    dim = entity.brain.hidden_dim
    fast_weights_ungated = GatedLocalNeuromodulatedPlasticity(in_features=dim, out_features=dim, lr=0.08).to(device_str)
    fast_weights_gated = GatedLocalNeuromodulatedPlasticity(in_features=dim, out_features=dim, lr=0.08).to(device_str)

    # 1. Encode 1-Shot High-Surprise Episodic Pattern (NA = 0.85)
    pre_key = torch.randn(1, dim, device=device_str)
    post_val = torch.randn(1, dim, device=device_str)

    fast_weights_ungated.adapt_local_fast_weights(pre_key, post_val, na_t=0.85, da_t=0.50, use_gating=False)
    fast_weights_gated.adapt_local_fast_weights(pre_key, post_val, na_t=0.85, da_t=0.50, use_gating=True)

    # Initial recall alignment
    out_ungated_0 = fast_weights_ungated(pre_key)
    out_gated_0 = fast_weights_gated(pre_key)

    sim_ungated_0 = float(F.cosine_similarity(out_ungated_0, post_val, dim=-1).item())
    sim_gated_0 = float(F.cosine_similarity(out_gated_0, post_val, dim=-1).item())

    logger.info(">>> Initial 1-Shot Episodic Fast-Weight Encoding Similarity <<<")
    logger.info(f"Ungated Initial Similarity : {sim_ungated_0:.4f}")
    logger.info(f"Gated Initial Similarity   : {sim_gated_0:.4f}")

    # 2. Simulate 30 Routine Routine Stream Steps with Low Surprise (NA = 0.05, DA = 0.10)
    logger.info("\n>>> Simulating 30 Routine Stream Steps with Low Surprise (NA = 0.05) <<<")
    for _ in range(30):
        routine_pre = torch.randn(1, dim, device=device_str)
        routine_err = torch.randn(1, dim, device=device_str) * 0.10
        
        fast_weights_ungated.adapt_local_fast_weights(routine_pre, routine_err, na_t=0.05, da_t=0.10, use_gating=False)
        fast_weights_gated.adapt_local_fast_weights(routine_pre, routine_err, na_t=0.05, da_t=0.10, use_gating=True)

    # 3. Post-Routine Episodic Recall Audit
    out_ungated_30 = fast_weights_ungated(pre_key)
    out_gated_30 = fast_weights_gated(pre_key)

    sim_ungated_30 = float(F.cosine_similarity(out_ungated_30, post_val, dim=-1).item())
    sim_gated_30 = float(F.cosine_similarity(out_gated_30, post_val, dim=-1).item())

    retention_loss_ungated = (sim_ungated_0 - sim_ungated_30) / max(sim_ungated_0, 1e-5) * 100.0
    retention_loss_gated = (sim_gated_0 - sim_gated_30) / max(sim_gated_0, 1e-5) * 100.0

    logger.info(">>> Post-Routine 1-Shot Episodic Fast-Weight Recall Audit <<<")
    logger.info(f"Ungated Post-Routine Similarity : {sim_ungated_30:.4f} (Memory Erosion: {retention_loss_ungated:.2f}%)")
    logger.info(f"Gated Post-Routine Similarity   : {sim_gated_30:.4f} (Memory Erosion: {retention_loss_gated:.2f}%)")

    memory_protection_gain = (sim_gated_30 - sim_ungated_30) / max(sim_ungated_30, 1e-5) * 100.0

    verdict = "POSITIVE" if (sim_gated_30 > sim_ungated_30 and retention_loss_gated < retention_loss_ungated) else "REJECTED"

    logger.info("=" * 80)
    logger.info("📊 === EXP-158 EMPIRICAL TELEMETRY SUMMARY ===")
    logger.info(f"🏆 Verdict                         : 🟢 {verdict}")
    logger.info(f"🛡️ Memory Retention Protection Gain : {memory_protection_gain:+.2f}% Higher Fact Retention")
    logger.info(f"🧠 Gated Episodic Retention        : {sim_gated_30:.4f} vs Ungated {sim_ungated_30:.4f}")
    logger.info("=" * 80)

    summary = {
        "exp_id": "EXP-158",
        "verdict": verdict,
        "metrics": {
            "ungated_initial_sim": sim_ungated_0,
            "gated_initial_sim": sim_gated_0,
            "ungated_post_routine_sim": sim_ungated_30,
            "gated_post_routine_sim": sim_gated_30,
            "memory_protection_gain_pct": memory_protection_gain,
            "retention_loss_ungated_pct": retention_loss_ungated,
            "retention_loss_gated_pct": retention_loss_gated
        }
    }
    with open("experiments/exp_158_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("✅ EXP-158 execution complete. Results saved to experiments/exp_158_results.json.")


if __name__ == "__main__":
    main()
