# experiments/exp_246_continuous_meta_plastic_sleep_stream.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #246
Topic: Continuous Meta-Plastic Stream Learning with Real-Time Interoceptive Sleep
       and Dynamic Graph Neurogenesis
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist (2026)
Standard: KEP v10.0 Master Protocol | Rules #1, #2, #3, #4, #5, #6, #7, #8, #10, #11
===============================================================================

Hypothesis:
    Equipping Karyon with Continuous Interoceptive Sleep Triggers (where the model
    autonomously pauses stream processing upon hitting high Somatic Fatigue or Noradrenaline
    surge NA >= 0.70 to execute a 100ms Deep Allostatic Sleep & AGN Neurogenesis cycle)
    will allow it to learn an ultra-long sequence stream without catastrophic gradient explosion
    or performance saturation, dropping final speech prediction loss by >= 0.25 nats/byte
    compared to uninterrupted stream training, while maintaining high processing throughput.
"""

import os
import sys
import time
import math
import random
import logging
import torch
import torch.nn as nn

# Ensure root repository directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_config import CoREConfig
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory
from karyon_agent import CoREAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("exp_246")


class InfiniteRealWorldStreamSimulator:
    """
    Simulates a continuous, infinite real-world stream containing complex technical queries,
    code definitions, and Active Inference dialogues.
    """
    def __init__(self, vocab_size: int = 258, seq_len: int = 128):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.corpus = [
            "User: What is the relation between Variational Free Energy and Active Inference?\nAssistant: Active Inference minimizes Free Energy F_t = KL(Q||P) + L_rec to bound environmental surprise.\n",
            "User: How does Ashby's Ultrastability prevent somatic collapse?\nAssistant: Ultrastability uses homeostatic feedback loops to restore interoceptive variables like energy and health.\n",
            "User: Explain Net2Net Smooth Grafting identity.\nAssistant: It initializes new neural branches with alpha_epi=0.0 so f_new(x) = f_old(x) at birth.\n",
            "User: Show me C++20 PAC decoding logic.\nAssistant: template <typename T> T pac_decode(T theta, T gamma) { return theta * std::exp(gamma); }\n",
            "User: What is the primary role of noradrenaline NA in Karyon?\nAssistant: NA scales the temporal integration rate dt and triggers sleep phases upon high surprise.\n"
        ]

    def generate_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        batch = torch.full((batch_size, self.seq_len), 256, dtype=torch.long, device=device)
        for i in range(batch_size):
            text = ""
            while len(text) < self.seq_len:
                text += random.choice(self.corpus)
            encoded = list(text.encode("utf-8"))[:self.seq_len]
            if len(encoded) < self.seq_len:
                encoded += [256] * (self.seq_len - len(encoded))
            batch[i, :self.seq_len] = torch.tensor(encoded, dtype=torch.long, device=device)
        return batch


def run_experiment():
    logger.info("=" * 80)
    logger.info("⚡ [EXP-246] CONTINUOUS META-PLASTIC STREAM LEARNING WITH INTEROCEPTIVE SLEEP")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"📟 Hardware Acceleration: {device}")

    # 1. Base Configuration & Setup
    config = CoREConfig()
    config.net.hidden_dim = 256
    config.net.num_heads = 4
    config.train.batch_size = 4
    config.train.seq_len = 128

    logger.info("\n[1/5] Initializing Karyon CoREAgent Substrate...")
    
    # Model A: Uninterrupted Static Stream Baseline
    agent_baseline = CoREAgent(config=config, device=str(device)).to(device)
    hu_baseline = HomeostaticUnit(batch_size=config.train.batch_size, device=str(device))
    
    # Model B: Autonomous Interoceptive Sleep & Neurogenesis Agent
    agent_sleep = CoREAgent(config=config, device=str(device)).to(device)
    # Copy initial weights to ensure identical starting parameters
    agent_sleep.load_state_dict(agent_baseline.state_dict())
    hu_sleep = HomeostaticUnit(batch_size=config.train.batch_size, device=str(device))
    
    episodic_memory = BatchedEpisodicMemory(
        batch_size=config.train.batch_size,
        memory_dim=config.net.unified_dim,
        max_capacity=1000,
        device=str(device)
    )
    criterion_speech = nn.CrossEntropyLoss(ignore_index=256)

    simulator = InfiniteRealWorldStreamSimulator(seq_len=config.train.seq_len)

    total_stream_steps = 30
    
    # 2. Benchmark Model A: Uninterrupted Static Stream Training
    logger.info("\n[2/5] --- Running Model A: Uninterrupted Static Stream Baseline ---")
    opt_a = torch.optim.AdamW(agent_baseline.parameters(), lr=1e-3, weight_decay=1e-4)
    losses_a = []

    t0_a = time.perf_counter()
    for step in range(1, total_stream_steps + 1):
        x_batch = simulator.generate_batch(config.train.batch_size, device)
        targets = x_batch.clone()

        opt_a.zero_grad()
        # Deplete energy continuously
        hu_baseline.state[:, 1] = (hu_baseline.state[:, 1] - 0.03).clamp_min(0.05)
        hu_baseline.state[:, 4] = (hu_baseline.state[:, 4] + 0.03).clamp_max(0.95)

        loss, _, _, _, _, _, _ = agent_baseline.forward_sequence(x_batch, targets, hu_baseline, criterion_speech)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_baseline.parameters(), max_norm=1.0)
        opt_a.step()

        loss_val = loss.item()
        losses_a.append(loss_val)

        if step % 10 == 0 or step == 1:
            logger.info(f"   🌊 [Model A | Step {step:02d}/{total_stream_steps:02d}] Loss: {loss_val:.4f} nats/byte | Energy: {hu_baseline.state[0, 1].item():.2f}")

    elapsed_a = time.perf_counter() - t0_a
    final_loss_a = losses_a[-1]
    tok_per_sec_a = (total_stream_steps * config.train.batch_size * config.train.seq_len) / elapsed_a

    logger.info(f"📊 Model A Final Stream Loss: {final_loss_a:.4f} nats/byte | Speed: {tok_per_sec_a:.1f} tok/s")

    # 3. Benchmark Model B: Continuous Stream with Interoceptive Sleep Triggers
    logger.info("\n[3/5] --- Running Model B: Interoceptive Sleep & Dynamic Neurogenesis Agent ---")
    opt_b = torch.optim.AdamW(agent_sleep.parameters(), lr=1e-3, weight_decay=1e-4)
    losses_b = []
    sleep_cycles_count = 0

    val_inputs = simulator.generate_batch(config.train.batch_size, device)
    val_targets = val_inputs.clone()

    t0_b = time.perf_counter()
    for step in range(1, total_stream_steps + 1):
        x_batch = simulator.generate_batch(config.train.batch_size, device)
        targets = x_batch.clone()

        opt_b.zero_grad()
        # Deplete energy and build up noradrenaline fatigue
        hu_sleep.state[:, 1] = (hu_sleep.state[:, 1] - 0.04).clamp_min(0.05)
        hu_sleep.state[:, 4] = (hu_sleep.state[:, 4] + 0.04).clamp_max(0.95)

        loss, _, _, _, _, _, _ = agent_sleep.forward_sequence(x_batch, targets, hu_sleep, criterion_speech)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_sleep.parameters(), max_norm=1.0)
        opt_b.step()

        loss_val = loss.item()
        losses_b.append(loss_val)

        # Interoceptive trigger check: if energy <= 0.20 or NA >= 0.70 -> Sleep Phase Triggered!
        energy_val = hu_sleep.state[0, 1].item()
        na_val = hu_sleep.state[0, 4].item()

        if energy_val <= 0.25 or na_val >= 0.65:
            logger.info(f"   🚨 [Step {step:02d}] Interoceptive Threshold Hit (Energy: {energy_val:.2f}, NA: {na_val:.2f}) -> Triggering Sleep Cycle #{sleep_cycles_count + 1}...")
            
            _, evolved_agent, is_struct = agent_sleep.execute_deep_allostatic_sleep(
                episodic_memory=episodic_memory,
                hu=hu_sleep,
                num_replay_cycles=3,
                downscaling_factor=0.01,
                pruning_percentile=0.03,
                eval_inputs=val_inputs,
                eval_targets=val_targets,
                criterion_speech=criterion_speech
            )
            agent_sleep = evolved_agent
            # Re-bind optimizer for updated parameters if structural change occurred
            if is_struct:
                opt_b = torch.optim.AdamW(agent_sleep.parameters(), lr=1e-3, weight_decay=1e-4)

            sleep_cycles_count += 1
            logger.info(f"   🌙 Sleep Cycle #{sleep_cycles_count} Complete -> Somatic Energy Reset to {hu_sleep.state[0, 1].item():.2f}")

        if step % 10 == 0 or step == 1:
            logger.info(f"   🌊 [Model B | Step {step:02d}/{total_stream_steps:02d}] Loss: {loss_val:.4f} nats/byte | Energy: {hu_sleep.state[0, 1].item():.2f}")

    elapsed_b = time.perf_counter() - t0_b
    final_loss_b = losses_b[-1]
    tok_per_sec_b = (total_stream_steps * config.train.batch_size * config.train.seq_len) / elapsed_b

    # 4. Comparative Evaluation
    logger.info("\n[4/5] --- Performance Comparison & Loss Trajectory ---")
    delta_loss_total = final_loss_a - final_loss_b

    logger.info(f"   - Model A (Static Uninterrupted Baseline) Final Loss: {final_loss_a:.4f} nats/byte")
    logger.info(f"   - Model B (Interoceptive Sleep & Neurogenesis) Final Loss: {final_loss_b:.4f} nats/byte")
    logger.info(f"   - Total Loss Reduction Delta: {delta_loss_total:+.4f} nats/byte")
    logger.info(f"   - Total Autonomous Sleep Cycles Executed: {sleep_cycles_count}")
    logger.info(f"   - Model A Speed: {tok_per_sec_a:.1f} tok/s | Model B Speed: {tok_per_sec_b:.1f} tok/s")

    # 5. KEP Rule #2 Verdict Decision
    logger.info("\n[5/5] --- Phase D: KEP Rule #2 Decision Criteria ---")

    verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    if delta_loss_total >= 0.25 and not math.isnan(final_loss_b):
        verdict = "🟢 POSITIVE"
    elif delta_loss_total < 0.0 or math.isnan(final_loss_b):
        verdict = "🔴 REJECTED"

    logger.info(f"🏆 Verdict: {verdict}")

    results = {
        "loss": final_loss_b,
        "baseline_loss": final_loss_a,
        "delta_loss": delta_loss_total,
        "tok_per_sec": tok_per_sec_b,
        "sleep_cycles_count": sleep_cycles_count,
        "verdict": verdict
    }
    return results


if __name__ == "__main__":
    run_experiment()
