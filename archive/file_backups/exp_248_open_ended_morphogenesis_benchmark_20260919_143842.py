# experiments/exp_248_open_ended_morphogenesis_benchmark.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #248
Topic: Open-Ended Cognitive Morphogenesis Benchmark (AGN v8.0)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist (2026)
Standard: KEP v10.0 Master Protocol | Rules #1, #2, #3, #4, #5, #6, #7, #8, #10, #11
===============================================================================

Hypothesis:
    Giving Karyon absolute open-ended freedom of self-directed morphogenesis via
    Universal Cognitive Organelles (emergent, self-parameterizing neural structures)
    will allow the network to dynamically synthesize arbitrary computational mechanisms
    (such as unrecognized memory spaces or sandbox simulation manifolds) directly from
    a continuous mathematical substrate.
    This open-ended architecture will maintain 100% zero-shock Net2Net birth identity
    while demonstrating superior multi-domain stream prediction, dropping speech loss by >= 0.08 nats/byte.
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
logger = logging.getLogger("exp_248")


class UniversalMorphicCorpus:
    """
    Highly complex multi-domain stream combining code, biophysics, and active inference.
    """
    def __init__(self, seq_len: int = 128):
        self.seq_len = seq_len
        self.corpus = [
            "Query: Describe the properties of an Emergent Holographic Memory Organelle.\nResponse: It encodes memories as complex phase shifts inside a self-parameterized state space.\n",
            "Query: How does Karyon synthesize an unrecognized sandbox simulator?\nResponse: By generating dynamic projection matrices and elementary mathematical primitives from its meta-controller.\n",
            "Query: Define the mathematical boundaries of Open-Ended Morphogenesis.\nResponse: Any organelle is wrapped in a smooth epigenetic gate tanh(alpha_epi) initialized at zero.\n",
            "Query: How do newly sprouted organelles affect pre-trained representations?\nResponse: They contribute exactly zero at birth, ensuring zero-shock Net2Net identity preservation.\n"
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
    logger.info("⚡ [EXP-248] OPEN-ENDED COGNITIVE MORPHOGENESIS BENCHMARK (AGN v8.0)")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"📟 Hardware Acceleration: {device}")

    # 1. Base Configuration & Setup
    config = CoREConfig()
    config.net.hidden_dim = 256
    config.net.num_heads = 4
    config.train.batch_size = 4
    config.train.seq_len = 128

    logger.info("\n[1/5] Initializing Karyon CoREAgent...")
    agent = CoREAgent(config=config, device=str(device)).to(device)
    hu = HomeostaticUnit(batch_size=config.train.batch_size, device=str(device))
    criterion_speech = nn.CrossEntropyLoss(ignore_index=256)
    corpus = UniversalMorphicCorpus(seq_len=config.train.seq_len)

    # 2. Verify Zero-Shock Net2Net Birth Identity on Cognitive Organelle Sprouting
    logger.info("\n[2/5] --- Verifying Zero-Shock Birth Identity on Cognitive Organelle Sprouting ---")
    val_x = corpus.generate_batch(config.train.batch_size, device)
    val_y = val_x.clone()

    with torch.no_grad():
        out_before, _, _, _, _, _, _ = agent.forward_sequence(val_x, val_y, hu, criterion_speech)

    # Sprout an Emergent Holographic Memory Organelle
    sprouted_mem = agent.sprout_cognitive_organelle("emergent_holographic_memory", state_dim=128, num_operators=8)
    # Sprout a Custom Sandbox Simulator Organelle
    sprouted_sandbox = agent.sprout_cognitive_organelle("custom_sandbox_simulator", state_dim=256, num_operators=8)

    assert sprouted_mem and sprouted_sandbox, "Failed to sprout cognitive organelles!"

    with torch.no_grad():
        out_after, _, _, _, _, _, _ = agent.forward_sequence(val_x, val_y, hu, criterion_speech)

    birth_delta = abs(out_before.item() - out_after.item())
    logger.info(f"   🧬 Net2Net Zero-Shock Birth Delta: {birth_delta:.8f}")
    if birth_delta > 1e-5:
        logger.warning(f"⚠️ Warning: Non-zero birth delta ({birth_delta:.8f}) exceeds strict threshold!")
    else:
        logger.info("   ✅ Zero-Shock Function Identity Perfectly Preserved (f_new(x) == f_old(x)) at birth!")

    # 3. Stream Learning & Morphogenesis Training
    logger.info("\n[3/5] --- Executing Stream Learning with Morphic Organelle Integration ---")
    
    # Enable gradient flow into the newly sprouted organelles by setting their alphas to small positive values
    # representing early epigenetic activation
    with torch.no_grad():
        agent.organelle_alphas["emergent_holographic_memory"].fill_(0.1)
        agent.organelle_alphas["custom_sandbox_simulator"].fill_(0.1)

    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

    stream_steps = 30
    losses = []
    t0 = time.perf_counter()

    for step in range(1, stream_steps + 1):
        x_batch = corpus.generate_batch(config.train.batch_size, device)
        targets = x_batch.clone()

        optimizer.zero_grad()
        loss, fe, _, _, _, _, _ = agent.forward_sequence(x_batch, targets, hu, criterion_speech)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 5 == 0 or step == 1:
            logger.info(f"   🌊 [Step {step:02d}/{stream_steps:02d}] Loss: {loss_val:.4f} nats/byte | Free Energy: {fe:.4f} | Organelles Active: {len(agent.cognitive_organelles)}")

    elapsed = time.perf_counter() - t0
    final_loss = losses[-1]
    baseline_loss = losses[0]
    delta_loss = baseline_loss - final_loss
    tok_per_sec = (stream_steps * config.train.batch_size * config.train.seq_len) / elapsed

    # 4. Morphic Organelle Telemetry Audit
    logger.info("\n[4/5] --- Morphogenesis Telemetry & Diagnostics ---")
    logger.info(f"   - Registered Organelles: {list(agent.cognitive_organelles.keys())}")
    for name in agent.cognitive_organelles.keys():
        alpha_val = agent.organelle_alphas[name].item()
        gate_val = math.tanh(alpha_val)
        logger.info(f"     * Organelle '{name}': alpha_epi = {alpha_val:.4f} (Gating factor = {gate_val:.4f})")

    logger.info(f"   - Final Stream Loss: {final_loss:.4f} nats/byte (Initial: {baseline_loss:.4f})")
    logger.info(f"   - Total Loss Reduction: {delta_loss:+.4f} nats/byte")
    logger.info(f"   - Throughput Speed: {tok_per_sec:.1f} tok/s")

    # 5. KEP Rule #2 Verdict Decision
    logger.info("\n[5/5] --- KEP Rule #2 Decision Criteria ---")
    verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    if delta_loss >= 0.08 and not math.isnan(final_loss):
        verdict = "🟢 POSITIVE"
    elif delta_loss < 0.0 or math.isnan(final_loss):
        verdict = "🔴 REJECTED"

    logger.info(f"🏆 Verdict: {verdict}")

    results = {
        "loss": final_loss,
        "baseline_loss": baseline_loss,
        "delta_loss": delta_loss,
        "tok_per_sec": tok_per_sec,
        "organelles_count": len(agent.cognitive_organelles),
        "verdict": verdict
    }
    return results


if __name__ == "__main__":
    run_experiment()
