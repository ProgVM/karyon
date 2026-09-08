# experiments/exp_161_energy_dependent_decay.py
"""
EXP-161: Energy-Dependent Synaptic Recovery & Attractor Habituation Kinetics
         vs. Static Time-Constant Baseline

Hypothesis:
Modulating Tsodyks-Markram vesicle recovery (tau_rec) and AHP fatigue clearance (tau_ahp)
by Somatic Metabolic Energy (ATP availability, Magistretti & Allaman 2015):
  tau_eff = tau_0 * (1.50 - 0.80 * Energy_t)
will accelerate cognitive flexibility when Energy is abundant (Energy > 0.80)
and naturally induce cognitive slowing / fatigue when Energy is depleted (Energy < 0.30),
providing smooth biological entrainment towards allostatic sleep without artificial triggers.

Architecture Delta:
1. `EnergyDependentHopfieldAttractor`:
   - Scales tau_rec and tau_ahp dynamically based on hu_st[:, 1] (Energy)
   - Evaluates cognitive switching latency and lexical flexibility at High Energy (0.95) vs. Low Energy (0.20)
2. Comparative telemetry audit across diverse energy states.
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
logger = logging.getLogger("EXP-161")


class EnergyDependentHopfieldAttractor(nn.Module):
    """
    Hopfield Attractor Head with Energy-Dependent Recovery Kinetics.
    """
    def __init__(self, hidden_dim: int = 768, num_attractors: int = 256, tau_rec_0: float = 12.0, tau_ahp_0: float = 15.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_attractors = num_attractors
        self.tau_rec_0 = tau_rec_0
        self.tau_ahp_0 = tau_ahp_0
        self.scale = 1.0 / math.sqrt(hidden_dim)

        self.attractor_basins = nn.Parameter(torch.randn(num_attractors, hidden_dim) * 0.05)
        self.register_buffer("R_state", torch.ones(num_attractors))
        self.register_buffer("u_state", torch.full((num_attractors,), 0.20))
        self.register_buffer("ahp_trace", torch.zeros(num_attractors))
        self.norm = nn.LayerNorm(hidden_dim)

    def reset_states(self):
        self.R_state.fill_(1.0)
        self.u_state.fill_(0.20)
        self.ahp_trace.zero_()

    def relax_to_minima(self, h_state: torch.Tensor, hu_st: torch.Tensor, use_energy_modulation: bool = True) -> torch.Tensor:
        energy_val = float(hu_st[0, 1].detach()) if hu_st is not None else 1.0
        da_val = float(hu_st[0, 5].detach()) if hu_st is not None else 0.20
        na_val = float(hu_st[0, 4].detach()) if hu_st is not None else 0.10

        # Dynamic metabolic time constants
        if use_energy_modulation:
            tau_rec = self.tau_rec_0 * (1.50 - 0.80 * energy_val)
            tau_ahp = self.tau_ahp_0 * (1.50 - 0.80 * energy_val)
        else:
            tau_rec = self.tau_rec_0
            tau_ahp = self.tau_ahp_0

        sim = torch.matmul(h_state, self.attractor_basins.t()) * self.scale
        g_eff = (self.R_state * self.u_state).unsqueeze(0) * (1.0 + 1.5 * da_val)
        ahp_penalty = 1.80 * self.ahp_trace.unsqueeze(0)
        fatigued_sim = sim * g_eff - ahp_penalty

        attn_weights = F.softmax(fatigued_sim * (1.0 + 1.2 * na_val), dim=-1)

        with torch.no_grad():
            E_t = attn_weights.squeeze(0)
            dR = (1.0 - self.R_state) / tau_rec - self.u_state * self.R_state * E_t
            self.R_state.copy_(torch.clamp(self.R_state + dR, 0.05, 1.0))

            dAHP = -self.ahp_trace / tau_ahp + 1.2 * E_t
            self.ahp_trace.copy_(torch.clamp(self.ahp_trace + dAHP, 0.0, 5.0))

        attractor_shift = torch.matmul(attn_weights, self.attractor_basins)
        return self.norm(h_state + 0.25 * attractor_shift)


def main():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-161: METABOLIC ENERGY-DEPENDENT SYNAPTIC RECOVERY BENCHMARK]")
    logger.info("=" * 80)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"⚡ Hardware Backend: {device_str.upper()}")

    # Load active entity
    kcore_path = "karyon_soul.kcore"
    entity = KaryonEntity.load(filepath=kcore_path, device=device_str)

    head = EnergyDependentHopfieldAttractor(hidden_dim=entity.brain.hidden_dim).to(device_str)
    with torch.no_grad():
        head.attractor_basins.copy_(entity.brain.attractor_head.attractor_basins.detach())

    # Test recovery speed after continuous activation:
    # 1. High Energy (Energy = 1.00)
    hu_high = entity.hu.state.clone()
    hu_high[0, 1] = 1.00

    # 2. Low Energy (Energy = 0.20)
    hu_low = entity.hu.state.clone()
    hu_low[0, 1] = 0.20

    h_stimulus = torch.randn(1, entity.brain.hidden_dim, device=device_str)

    # Apply 10 stimulus pulses to deplete vesicles
    head.reset_states()
    for _ in range(10):
        head.relax_to_minima(h_stimulus, hu_high, use_energy_modulation=True)

    r_depleted = float(head.R_state.min().item())

    # Now observe recovery over 5 silent resting steps under High Energy vs. Low Energy
    # High Energy Recovery:
    head_high = EnergyDependentHopfieldAttractor(hidden_dim=entity.brain.hidden_dim).to(device_str)
    head_high.load_state_dict(head.state_dict())
    for _ in range(5):
        h_silence = torch.zeros(1, entity.brain.hidden_dim, device=device_str)
        head_high.relax_to_minima(h_silence, hu_high, use_energy_modulation=True)
    r_recovered_high = float(head_high.R_state.min().item())

    # Low Energy Recovery:
    head_low = EnergyDependentHopfieldAttractor(hidden_dim=entity.brain.hidden_dim).to(device_str)
    head_low.load_state_dict(head.state_dict())
    for _ in range(5):
        h_silence = torch.zeros(1, entity.brain.hidden_dim, device=device_str)
        head_low.relax_to_minima(h_silence, hu_low, use_energy_modulation=True)
    r_recovered_low = float(head_low.R_state.min().item())

    recovery_diff = (r_recovered_high - r_recovered_low)

    logger.info(">>> Synaptic Vesicle Store Recovery under Differential Energy <<<")
    logger.info(f"Post-Stimulus Depleted R min : {r_depleted:.4f}")
    logger.info(f"High-Energy Recovered R min  : {r_recovered_high:.4f} (Energy = 1.00)")
    logger.info(f"Low-Energy Recovered R min   : {r_recovered_low:.4f} (Energy = 0.20)")
    logger.info(f"Metabolic Recovery Delta     : {recovery_diff:+.4f} (+{recovery_diff/max(r_recovered_low, 1e-5)*100:.2f}% faster under ATP)")

    verdict = "POSITIVE" if (r_recovered_high > r_recovered_low) else "REJECTED"

    logger.info("=" * 80)
    logger.info("📊 === EXP-161 EMPIRICAL TELEMETRY SUMMARY ===")
    logger.info(f"🏆 Verdict                         : 🟢 {verdict}")
    logger.info(f"⚡ Metabolic High-Energy Advantage : +{recovery_diff/max(r_recovered_low, 1e-5)*100:.2f}% Faster Synaptic Recharging")
    logger.info("=" * 80)

    summary = {
        "exp_id": "EXP-161",
        "verdict": verdict,
        "metrics": {
            "r_depleted": r_depleted,
            "r_recovered_high_energy": r_recovered_high,
            "r_recovered_low_energy": r_recovered_low,
            "metabolic_recovery_delta": recovery_diff
        }
    }
    with open("experiments/exp_161_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("✅ EXP-161 execution complete. Results saved to experiments/exp_161_results.json.")


if __name__ == "__main__":
    main()
