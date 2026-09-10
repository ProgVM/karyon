# experiments/exp_192_allostatic_synaptic_scaling.py
"""
===============================================================================
EXP-192: Allostatic Synaptic Scaling and Dynamic Decay
Grounding: KEP Principle 2 (Biological Realism), Principle 14 (Allostatic Forces: No Static Constants).
===============================================================================
Hypothesis:
Replacing static weight decay and fixed learning rate scaling with an allostatically
modulated synaptic scaling mechanism—where synaptic decay rates and update gains are
dynamically coupled to Somatic Energy (u_t[1]) and Variational Free Energy (F_t)—will
prevent runaway excitation, stabilize gradient flows, and accelerate loss convergence.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_entity import KaryonEntity
from karyon_hardware import get_hardware_engine

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-192")


class AllostaticSynapticScaler(nn.Module):
    """
    Allostatically-modulated synaptic scaling and dynamic decay.
    """
    def __init__(self, hidden_dim: int, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.hidden_dim = hidden_dim
        
        # Dynamic scaling parameter estimator
        self.scaling_net = nn.Sequential(
            nn.Linear(6 + 1, 32),
            nn.SiLU(),
            nn.Linear(32, 2), # Outputs: [decay_rate, gain_multiplier]
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, weights: torch.Tensor, u_t: torch.Tensor, fe_loss: torch.Tensor) -> tuple:
        # Format homeostatic and free energy inputs
        fe_normalized = torch.clamp(fe_loss.detach() / 10.0, 0.0, 1.0).view(-1, 1)
        if u_t.dim() == 1:
            u_t = u_t.unsqueeze(0)
        
        scaler_input = torch.cat([u_t, fe_normalized], dim=-1)
        outputs = self.scaling_net(scaler_input)
        
        # Map outputs to biophysically realistic ranges
        # Decay rate: Tononi SHY downscaling (0.01 to 0.15)
        decay_rate = 0.01 + 0.14 * outputs[:, 0:1]
        # Gain multiplier: dynamic synaptic scaling (0.5 to 1.5)
        gain_mult = 0.5 + 1.0 * outputs[:, 1:2]
        
        # Apply allostatic synaptic scaling
        scaled_weights = weights * (1.0 - decay_rate) * gain_mult
        return scaled_weights, decay_rate, gain_mult


def evaluate_model(brain, entity, text_samples, num_steps=25, lr=1e-3):
    hw = get_hardware_engine()
    criterion = nn.CrossEntropyLoss(ignore_index=256)
    optimizer = torch.optim.AdamW(brain.parameters(), lr=lr, weight_decay=1e-4)

    step_losses = []
    step_fe_losses = []

    start_time = time.perf_counter()

    for step in range(num_steps):
        text = text_samples[step % len(text_samples)]
        prompt_ids = brain.tokenizer.encode(text)
        seq_t = torch.tensor([prompt_ids[:-1]], dtype=torch.long, device=hw.device)
        target_t = torch.tensor([prompt_ids[1:]], dtype=torch.long, device=hw.device)

        optimizer.zero_grad()
        tot_loss, speech_loss, fe_loss, _, _, _, _ = brain.forward_sequence(
            seq_t, target_t, entity.hu, criterion, chunk_size=seq_t.size(1), use_checkpointing=False
        )
        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()

        step_losses.append(float(speech_loss))
        step_fe_losses.append(float(fe_loss))

    elapsed = time.perf_counter() - start_time
    total_tokens = sum(len(brain.tokenizer.encode(t)) - 1 for t in text_samples[:num_steps])
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "final_loss": step_losses[-1],
        "initial_loss": step_losses[0],
        "mean_loss": sum(step_losses) / len(step_losses),
        "final_fe": step_fe_losses[-1],
        "initial_fe": step_fe_losses[0],
        "tok_per_sec": tok_per_sec,
        "elapsed": elapsed
    }


def run_benchmark():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-192: ALLOSTATIC SYNAPTIC SCALING BENCHMARK]")
    logger.info("=" * 80)

    hw = get_hardware_engine()
    logger.info(f"Target Accelerator: {hw.device_str.upper()}")

    text_samples = [
        "The quick brown fox jumps over the lazy dog near the riverbank with high agility.",
        "Active Inference formulates brain dynamics as continuous minimization of variational free energy.",
        "Homeostasis and allostasis regulate physiological variables through predictive bodily setpoints.",
        "Neural state space duality enables zero-loop associative parallel scans across deep cortical layers.",
        "Continuous Hopfield attractors snap neural trajectories into discrete conceptual semantic basins.",
        "Cortical laminar hierarchy routes top-down predictions and bottom-up precision-weighted error residuals."
    ] * 5

    # 1. Baseline Evaluation (EXP-191 Bidirectional Routing)
    entity_base = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_base = entity_base.brain
    logger.info("Running Baseline Evaluation...")
    b_results = evaluate_model(brain_base, entity_base, text_samples, num_steps=25)

    # 2. Proposed Evaluation (EXP-192 Allostatic Synaptic Scaling)
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain
    brain_prop.allostatic_synaptic_scaler = AllostaticSynapticScaler(
        hidden_dim=brain_prop.hidden_dim, device_str=hw.device_str
    ).to(hw.device)
    
    logger.info("Running Proposed Allostatic Synaptic Scaling Evaluation...")
    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    loss_delta = b_results['final_loss'] - p_results['final_loss']
    fe_delta = b_results['final_fe'] - p_results['final_fe']

    logger.info("=" * 80)
    logger.info("📊 === EXP-192 TELEMETRY REPORT ===")
    logger.info(f"  - Baseline Final Loss : {b_results['final_loss']:.4f} nats | Throughput: {b_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Proposed Final Loss : {p_results['final_loss']:.4f} nats | Throughput: {p_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Loss Delta (B - P)  : {loss_delta:.4f} nats")
    logger.info(f"  - Free Energy Delta   : {fe_delta:.6f}")
    logger.info(f"  - Baseline Duration   : {b_results['elapsed']:.3f} s")
    logger.info(f"  - Proposed Duration   : {p_results['elapsed']:.3f} s")

    verdict = "POSITIVE" if (loss_delta >= 0.08 or (loss_delta >= 0.02 and p_results['tok_per_sec'] >= 0.90 * b_results['tok_per_sec'])) else "NEUTRAL"

    results = {
        "exp_id": "EXP-192",
        "verdict": verdict,
        "base_initial_loss": b_results['initial_loss'],
        "base_final_loss": b_results['final_loss'],
        "proposed_initial_loss": p_results['initial_loss'],
        "proposed_final_loss": p_results['final_loss'],
        "loss_delta": loss_delta,
        "fe_delta": fe_delta,
        "throughput_tok_per_sec": p_results['tok_per_sec'],
        "execution_time_s": p_results['elapsed']
    }

    with open("experiments/exp_192_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"🏆 Final Verdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    return results


if __name__ == "__main__":
    run_benchmark()
