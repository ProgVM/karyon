# experiments/exp_245_multitask_epigenetic_self_assembly.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #245
Topic: Universal Cross-Task Epigenetic Self-Assembly: Multi-Domain Stream
       Learning with Dynamic Morphic Sub-Topology Specialization
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist (2026)
Standard: KEP v10.0 Master Protocol | Rules #1, #2, #3, #4, #5, #6, #7, #8, #10, #11
===============================================================================

Hypothesis:
    Evaluating Karyon's 5-Tier Epigenetic Self-Evolution Engine across a heterogeneous
    multi-task byte stream (combining natural dialogue, executable Python/Assembly code,
    and structured JSON/mathematical logic) will demonstrate that the model autonomously
    sprouts, routes, and adapts specialized sub-topology operator bricks (via GRN
    methylation and Net2Net Smooth Grafting) to handle distinct tasks concurrently,
    lowering joint prediction loss by >= 0.20 nats/byte compared to a static single-topology
    baseline without task interference or catastrophic forgetting.
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
logger = logging.getLogger("exp_245")


class HeterogeneousMultiTaskStreamSimulator:
    """
    Simulates a continuous multi-task byte stream containing 3 distinct task domains:
    1. Task 'dialogue': Natural language Active Inference dialogue
    2. Task 'code': Executable Python/C++ code syntax
    3. Task 'math_json': Structured JSON / Mathematical logic
    """
    def __init__(self, vocab_size: int = 258, seq_len: int = 128):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.dialogue_templates = [
            "User: Explain Ashby's homeostatic ultrastability.\nAssistant: Ultrastability maintains critical variables within physiological bounds.\n",
            "User: What is Variational Free Energy?\nAssistant: Free Energy $F_t$ bounds surprise $D_{KL}(Q||P) + L_{rec}$.\n"
        ]

        self.code_templates = [
            "def sde_heun_step(h_t, dt, sigma):\n    f_pred = h_t + f(h_t) * dt\n    return h_t + 0.5 * (f(h_t) + f(f_pred)) * dt + sigma * dW\n",
            "template <typename T> inline T pac_decode(T theta, T gamma) {\n    return theta * std::exp(gamma);\n}\n"
        ]

        self.math_json_templates = [
            "{\"domain\": \"active_inference\", \"state\": {\"energy\": 1.0, \"stability\": 0.98}, \"free_energy\": 0.0012}\n",
            "{\"matrix\": \"ssd_parallel\", \"rank\": 16, \"throughput_tok_sec\": 204500, \"status\": \"OPTIMAL\"}\n"
        ]

    def generate_task_batch(self, task_type: str, batch_size: int, device: torch.device) -> torch.Tensor:
        if task_type == "dialogue":
            templates = self.dialogue_templates
        elif task_type == "code":
            templates = self.code_templates
        else:
            templates = self.math_json_templates

        batch = torch.full((batch_size, self.seq_len), 256, dtype=torch.long, device=device)
        for i in range(batch_size):
            text = ""
            while len(text) < self.seq_len:
                text += random.choice(templates)
            encoded = list(text.encode("utf-8"))[:self.seq_len]
            if len(encoded) < self.seq_len:
                encoded += [256] * (self.seq_len - len(encoded))
            batch[i, :self.seq_len] = torch.tensor(encoded, dtype=torch.long, device=device)
        return batch

    def generate_interleaved_multi_task_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Generates a batch where each sample in the batch comes from a different domain
        batch = torch.full((batch_size, self.seq_len), 256, dtype=torch.long, device=device)
        tasks = ["dialogue", "code", "math_json"]
        for i in range(batch_size):
            t_type = tasks[i % len(tasks)]
            sub_batch = self.generate_task_batch(t_type, 1, device)
            batch[i] = sub_batch[0]
        return batch


def run_experiment():
    logger.info("=" * 80)
    logger.info("🚀 [EXP-245] UNIVERSAL CROSS-TASK EPIGENETIC SELF-ASSEMBLY & MULTI-DOMAIN SPECIALIZATION")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"📟 Compute Hardware: {device}")

    # 1. Base Configuration & Agent Setup
    config = CoREConfig()
    config.net.hidden_dim = 256
    config.net.num_heads = 4
    config.train.batch_size = 6  # 2 samples per task domain
    config.train.seq_len = 128

    logger.info("\n[1/5] Initializing CoREAgent with Universal Epigenetic Substrate...")
    agent = CoREAgent(config=config, device=str(device)).to(device)
    hu = HomeostaticUnit(batch_size=config.train.batch_size, device=str(device))
    episodic_memory = BatchedEpisodicMemory(
        batch_size=config.train.batch_size,
        memory_dim=config.net.unified_dim,
        max_capacity=1000,
        device=str(device)
    )
    criterion_speech = nn.CrossEntropyLoss(ignore_index=256)

    simulator = HeterogeneousMultiTaskStreamSimulator(seq_len=config.train.seq_len)

    # 2. Phase A: Static Single-Topology Baseline Multi-Task Training
    logger.info("\n[2/5] --- Phase A: Training Static Baseline Model on Interleaved Tasks ---")
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

    baseline_steps = 15
    baseline_losses = []

    t_start = time.perf_counter()
    for step in range(1, baseline_steps + 1):
        x_batch = simulator.generate_interleaved_multi_task_batch(config.train.batch_size, device)
        targets = x_batch.clone()

        optimizer.zero_grad()

        # Somatic fatigue simulation
        hu.state[:, 1] = (hu.state[:, 1] - 0.04).clamp_min(0.10)
        hu.state[:, 4] = (hu.state[:, 4] + 0.05).clamp_max(0.90)

        loss, fe, _, logits, _, _, _ = agent.forward_sequence(x_batch, targets, hu, criterion_speech)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        baseline_losses.append(loss_val)

        if step % 5 == 0 or step == 1:
            logger.info(f"   ☀️ [Baseline Step {step:02d}/{baseline_steps:02d}] Interleaved Loss: {loss_val:.4f} nats/byte")

    elapsed_baseline = time.perf_counter() - t_start
    loss_baseline_final = baseline_losses[-1]
    tok_per_sec = (baseline_steps * config.train.batch_size * config.train.seq_len) / elapsed_baseline

    logger.info(f"📊 Baseline Static Loss: {loss_baseline_final:.4f} nats/byte | Throughput: {tok_per_sec:.1f} tok/s")

    # Measure per-domain baseline losses
    domain_baseline_losses = {}
    with torch.no_grad():
        for d_name in ["dialogue", "code", "math_json"]:
            d_batch = simulator.generate_task_batch(d_name, config.train.batch_size, device)
            d_loss = agent.forward_sequence(d_batch, d_batch, hu, criterion_speech)[0].item()
            domain_baseline_losses[d_name] = d_loss
            logger.info(f"   - Baseline '{d_name}' Loss: {d_loss:.4f} nats/byte")

    # 3. Phase B: Deep Allostatic Sleep & Cross-Task Self-Assembly
    logger.info("\n[3/5] --- Phase B: Sleep-Phase Epigenetic Morphogenesis & Sub-Topology Sprouting ---")

    val_inputs = simulator.generate_interleaved_multi_task_batch(config.train.batch_size, device)
    val_targets = val_inputs.clone()

    t_sleep_start = time.perf_counter()
    pruned_weights, evolved_agent, is_structural = agent.execute_deep_allostatic_sleep(
        episodic_memory=episodic_memory,
        hu=hu,
        num_replay_cycles=5,
        downscaling_factor=0.01,
        pruning_percentile=0.03,
        eval_inputs=val_inputs,
        eval_targets=val_targets,
        criterion_speech=criterion_speech
    )
    elapsed_sleep = time.perf_counter() - t_sleep_start

    logger.info(f"🌙 Sleep Cycle Concluded in {elapsed_sleep*1000.0:.2f} ms")
    logger.info(f"   - Pruned Weights: {pruned_weights}")
    logger.info(f"   - Structural Morphogenesis: {is_structural}")
    logger.info(f"   - Restored Energy: {hu.state[0, 1].item():.2f}")

    # 4. Phase C: Post-Sleep Multi-Task Stream Fine-Tuning & Evaluation
    logger.info("\n[4/5] --- Phase C: Evaluating Epigenetically Evolved Agent on Multi-Task Stream ---")

    post_optimizer = torch.optim.AdamW(evolved_agent.parameters(), lr=1e-3, weight_decay=1e-4)

    post_steps = 10
    post_losses = []
    for step in range(1, post_steps + 1):
        x_batch = simulator.generate_interleaved_multi_task_batch(config.train.batch_size, device)
        targets = x_batch.clone()

        post_optimizer.zero_grad()
        loss, fe, _, logits, _, _, _ = evolved_agent.forward_sequence(x_batch, targets, hu, criterion_speech)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(evolved_agent.parameters(), max_norm=1.0)
        post_optimizer.step()
        post_losses.append(loss.item())

        if step % 5 == 0 or step == 1:
            logger.info(f"   ☀️ [Self-Assembled Step {step:02d}/{post_steps:02d}] Interleaved Loss: {loss.item():.4f} nats/byte")

    final_self_assembled_loss = post_losses[-1]
    delta_loss_total = loss_baseline_final - final_self_assembled_loss

    # Measure per-domain self-assembled losses
    domain_self_assembled_losses = {}
    with torch.no_grad():
        for d_name in ["dialogue", "code", "math_json"]:
            d_batch = simulator.generate_task_batch(d_name, config.train.batch_size, device)
            d_loss = evolved_agent.forward_sequence(d_batch, d_batch, hu, criterion_speech)[0].item()
            domain_self_assembled_losses[d_name] = d_loss
            d_delta = domain_baseline_losses[d_name] - d_loss
            logger.info(f"   - Evolved '{d_name}' Loss: {d_loss:.4f} nats/byte (Delta: {d_delta:+.4f})")

    logger.info("📊 Summary Multi-Task Performance:")
    logger.info(f"   - Baseline Multi-Task Loss: {loss_baseline_final:.4f} nats/byte")
    logger.info(f"   - Self-Assembled Multi-Task Loss: {final_self_assembled_loss:.4f} nats/byte")
    logger.info(f"   - Overall Multi-Task Delta: {delta_loss_total:+.4f} nats/byte")

    # 5. KEP Rule #2 Verdict Decision
    logger.info("\n[5/5] --- Phase D: KEP Rule #2 Multi-Criteria Verdict ---")

    verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    if delta_loss_total >= 0.20 and not math.isnan(final_self_assembled_loss):
        verdict = "🟢 POSITIVE"
    elif delta_loss_total < 0.0 or math.isnan(final_self_assembled_loss):
        verdict = "🔴 REJECTED"

    logger.info(f"🏆 Verdict: {verdict}")

    results = {
        "loss": final_self_assembled_loss,
        "baseline_loss": loss_baseline_final,
        "delta_loss": delta_loss_total,
        "tok_per_sec": tok_per_sec,
        "domain_baseline": domain_baseline_losses,
        "domain_evolved": domain_self_assembled_losses,
        "is_structural": is_structural,
        "verdict": verdict
    }
    return results


if __name__ == "__main__":
    run_experiment()
