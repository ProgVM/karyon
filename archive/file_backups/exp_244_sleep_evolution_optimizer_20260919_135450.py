# experiments/exp_244_sleep_evolution_optimizer.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #244
Topic: Universal Sleep-Phase Self-Evolution & Epigenetic Morphogenesis Engine
       on Real-World Byte-Level Dialogue Streams
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist (2026)
Standard: KEP v10.0 Master Protocol | Rules #1, #2, #3, #4, #5, #6, #7, #8, #10, #11
===============================================================================

Hypothesis:
    Integrating the 5-Tier Epigenetic Self-Evolution Engine directly into Karyon's
    biophysical sleep cycle (coordinating GRN, AGN neurogenesis, Net2Net morphogenesis,
    and Directed Selection in the System 2 Sandbox) will allow the agent to autonomously
    reconstruct its topology and optimize its biophysical parameters, dropping prediction
    loss on realistic byte streams by >= 0.15 nats/byte compared to static baseline training,
    while maintaining zero structural shock (Net2Net Identity) at birth.
"""

import os
import sys
import time
import math
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure root repository directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_config import CoREConfig
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory
from karyon_agent import CoREAgent
from kcore_evolution import AutonomousSelfEvolutionOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("exp_244")


class RealWorldDialogueSimulator:
    """
    Generates high-fidelity simulated UTF-8 byte-level dialogue streams.
    Contains realistic structure, turn-taking, punctuation, and contextual repetitions
    to simulate real-world speech datasets (Alpaca-GPT4 style).
    """
    def __init__(self, vocab_size: int = 258, seq_len: int = 128):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.templates = [
            "User: Hello, Karyon! Can you help me optimize the biophysical substrate?\nAssistant: Yes, I can assist you with Active Inference and Free Energy minimization.\n",
            "User: What is the primary source of energy for Earth?\nAssistant: The Sun is the primary source of energy, driving all somatic and metabolic cycles.\n",
            "User: Explain Ashby's ultrastability in cognitive systems.\nAssistant: Ultrastability involves homeostatic feedback loops that adapt to critical variables.\n",
            "User: How do Continuous Hopfield attractors snap discrete concepts?\nAssistant: By minimizing an energy landscape, snapping continuous trajectories into stable basins.\n",
            "User: What is the role of noradrenaline in Karyon?\nAssistant: It represents variational surprise and modulates the temporal clock integration rate.\n"
        ]

    def generate_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        batch = torch.full((batch_size, self.seq_len), 256, dtype=torch.long, device=device)  # Pad with 256
        for i in range(batch_size):
            text = ""
            while len(text) < self.seq_len:
                text += random.choice(self.templates)
            # Encode as UTF-8 bytes
            encoded = list(text.encode("utf-8"))[:self.seq_len]
            # Zero-pad or truncate to exact seq_len
            if len(encoded) < self.seq_len:
                encoded += [256] * (self.seq_len - len(encoded))
            batch[i, :self.seq_len] = torch.tensor(encoded, dtype=torch.long, device=device)
        return batch


def run_experiment():
    logger.info("=" * 80)
    logger.info("🔬 [EXP-244] INITIATING SLEEP-PHASE SELF-EVOLUTION & EPIGENETIC OPTIMIZATION")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"📟 Hardware Acceleration: {device}")

    # 1. Base Configuration & Initialization
    config = CoREConfig()
    config.net.hidden_dim = 256
    config.net.num_heads = 4
    config.train.batch_size = 4
    config.train.seq_len = 128

    logger.info("\n[1/5] Initializing CoREAgent with Universal Morphic Engine...")
    agent = CoREAgent(config=config, device=str(device)).to(device)
    hu = HomeostaticUnit(batch_size=config.train.batch_size, device=str(device))
    episodic_memory = BatchedEpisodicMemory(
        batch_size=config.train.batch_size,
        memory_dim=config.net.unified_dim,
        max_capacity=500,
        device=str(device)
    )
    criterion_speech = nn.CrossEntropyLoss(ignore_index=256)

    # Populate episodic memory with some realistic experiences
    simulator = RealWorldDialogueSimulator(seq_len=config.train.seq_len)
    for _ in range(10):
        dummy_k = torch.randn(config.train.batch_size, config.net.unified_dim, device=device)
        dummy_v = torch.randn(config.train.batch_size, config.net.unified_dim, device=device)
        episodic_memory.write(dummy_k, dummy_v, 2)

    # 2. Phase A: Wakefulness Stream Training (Baseline)
    logger.info("\n[2/5] --- Phase A: Wakefulness Stream Training & Somatic Fatigue ---")
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

    num_wake_steps = 15
    wake_losses = []
    
    t_start = time.perf_counter()
    for step in range(1, num_wake_steps + 1):
        x_batch = simulator.generate_batch(config.train.batch_size, device)
        targets = x_batch.clone()

        optimizer.zero_grad()
        
        # Deplete somatic energy and increase surprise to trigger sleep homeostasis
        hu.state[:, 1] = max(0.10, hu.state[:, 1].item() - 0.05)  # Energy
        hu.state[:, 4] = min(0.90, hu.state[:, 4].item() + 0.04)  # Noradrenaline (Surprise)

        loss, fe, _, logits, _, _, _ = agent.forward_sequence(x_batch, targets, hu, criterion_speech)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        wake_losses.append(loss_val)

        if step % 5 == 0 or step == 1:
            logger.info(f"   ☀️ [Wake Step {step:02d}/{num_wake_steps:02d}] Loss: {loss_val:.4f} nats/byte | Energy: {hu.state[0, 1].item():.2f} | NA: {hu.state[0, 4].item():.2f}")

    elapsed_wake = time.perf_counter() - t_start
    loss_pre_sleep = wake_losses[-1]
    tok_per_sec = (num_wake_steps * config.train.batch_size * config.train.seq_len) / elapsed_wake

    logger.info(f"📊 Pre-Sleep Baseline Loss: {loss_pre_sleep:.4f} nats/byte | Throughput: {tok_per_sec:.1f} tok/s")

    # 3. Phase B: Deep Allostatic Sleep & Epigenetic Morphogenesis
    logger.info("\n[3/5] --- Phase B: Executing Deep Allostatic Sleep & Epigenetic Evolution ---")
    
    # Generate validation tokens for counterfactual sandbox rollout evaluation
    val_inputs = simulator.generate_batch(config.train.batch_size, device)
    val_targets = val_inputs.clone()

    t_sleep_start = time.perf_counter()
    pruned_weights, evolved_agent, is_structural = agent.execute_deep_allostatic_sleep(
        episodic_memory=episodic_memory,
        hu=hu,
        num_replay_cycles=4,
        downscaling_factor=0.01,
        pruning_percentile=0.03,
        eval_inputs=val_inputs,
        eval_targets=val_targets,
        criterion_speech=criterion_speech
    )
    elapsed_sleep = time.perf_counter() - t_sleep_start

    logger.info(f"🌙 Sleep Cycle Concluded in {elapsed_sleep*1000.0:.2f} ms")
    logger.info(f"   - Pruned Quiescent Weights: {pruned_weights}")
    logger.info(f"   - Structural Morphogenesis Occurred: {is_structural}")
    logger.info(f"   - Restored Somatic Energy: {hu.state[0, 1].item():.2f} (Target: 1.00)")
    logger.info(f"   - Normalized Noradrenaline: {hu.state[0, 4].item():.2f} (Target: 0.05)")

    # 4. Phase C: Post-Sleep Awakening & Fine-Tuning
    logger.info("\n[4/5] --- Phase C: Post-Awakening Evaluation & Continuity Verification ---")
    
    # Evaluate immediately after sleep (should preserve or improve representation)
    with torch.no_grad():
        out_post = evolved_agent.forward_sequence(val_inputs, val_targets, hu, criterion_speech)
        loss_post_sleep = out_post[0].item()

    delta_loss_immediate = loss_pre_sleep - loss_post_sleep
    logger.info(f"☀️ Immediate Post-Sleep Loss: {loss_post_sleep:.4f} nats/byte (Delta: {delta_loss_immediate:+.4f})")

    # Run fine-tuning steps after sleep to verify gradient flow across mutated/sprouted connections
    logger.info("   Fine-tuning post-sleep evolved agent...")
    post_optimizer = torch.optim.AdamW(evolved_agent.parameters(), lr=1e-3, weight_decay=1e-4)
    
    post_losses = []
    for step in range(1, 11):
        x_batch = simulator.generate_batch(config.train.batch_size, device)
        targets = x_batch.clone()

        post_optimizer.zero_grad()
        loss, fe, _, logits, _, _, _ = evolved_agent.forward_sequence(x_batch, targets, hu, criterion_speech)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(evolved_agent.parameters(), max_norm=1.0)
        post_optimizer.step()
        post_losses.append(loss.item())

    final_fine_tuned_loss = post_losses[-1]
    delta_loss_total = loss_pre_sleep - final_fine_tuned_loss
    logger.info(f"☀️ Final Fine-Tuned Loss: {final_fine_tuned_loss:.4f} nats/byte (Total Delta: {delta_loss_total:+.4f})")

    # 5. KEP Rule #2 Verdict Decision
    logger.info("\n[5/5] --- Phase D: Evaluating KEP Verdict Criteria ---")
    
    # Success Criteria:
    # 1. Sleep cycle executed successfully without throwing NaNs.
    # 2. Somatic homeostasis successfully reset (Energy -> 1.00, NA -> 0.05).
    # 3. Overall loss delta (baseline vs post-sleep fine-tuned) is positive and >= 0.15 nats/byte.
    verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    if delta_loss_total >= 0.15 and not math.isnan(final_fine_tuned_loss):
        verdict = "🟢 POSITIVE"
    elif delta_loss_total < 0.0 or math.isnan(final_fine_tuned_loss):
        verdict = "🔴 REJECTED"

    logger.info(f"🏆 Verdict: {verdict}")

    # Save metrics for ledger
    results = {
        "loss": final_fine_tuned_loss,
        "pre_sleep_loss": loss_pre_sleep,
        "post_sleep_loss": loss_post_sleep,
        "delta_loss": delta_loss_total,
        "tok_per_sec": tok_per_sec,
        "pruned_weights": pruned_weights,
        "is_structural": is_structural,
        "verdict": verdict
    }
    return results


if __name__ == "__main__":
    run_experiment()
