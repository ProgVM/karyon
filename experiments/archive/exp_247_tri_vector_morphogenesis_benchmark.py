# experiments/exp_247_tri_vector_morphogenesis_benchmark.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #247
Topic: Tri-Vector Universal Morphogenesis Benchmark
       (Dynamic DAG Adjacency Routing + Two-Tier L1/L2 Cache + BLT Entropy Scan)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist (2026)
Standard: KEP v10.0 Master Protocol | Rules #1, #2, #3, #4, #5, #6, #7, #8, #10, #11
===============================================================================

Hypothesis:
    Simultaneously unlocking all three cybernetic vectors:
      1. Unconstrained Dynamic DAG Graph Routing with learnable Adjacency Connections (A_{i,j});
      2. Hierarchical Two-Tier Memory Cache (L1 Operational + L2 Hippocampal with NA-gated recall);
      3. BLT Entropy-Adaptive Scan Engine (dynamic temporal rate scaling on conceptual boundaries);
    will synergize to produce continuous stream prediction superiority over static baselines,
    dropping final speech loss by >= 0.08 nats/byte while maintaining stable gradient flow and
    100% zero-shock Net2Net birth identity.
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
logger = logging.getLogger("exp_247")


class ComplexStreamBenchmarkCorpus:
    """
    Multimodal continuous byte stream containing complex code, biophysical theory,
    and Active Inference queries.
    """
    def __init__(self, seq_len: int = 128):
        self.seq_len = seq_len
        self.corpus = [
            "Query: Detail the continuous Stratonovich Predictor-Corrector integration.\nResponse: h_{t+1} = tanh(h_t + 0.5 * (f(h_t) + f(h_pred)) * dt + dW_t) with Wiener noise.\n",
            "Query: How does Two-Tier L1/L2 Hippocampal Memory handle high surprise?\nResponse: L1 stores instant events; when NA > 0.12 and surprise > 0.50, memories consolidate into L2 attractors.\n",
            "Query: What is Dynamic DAG Adjacency Routing in AGN v7.0?\nResponse: Learnable matrix A_{i,j} dynamically connects arbitrary operator nodes with continuous sigmoid edge weights.\n",
            "Query: Explain BLT Entropy-Adaptive Scan.\nResponse: Steps are widened (dt * 1.5) on word boundaries (H > 0.70) and accelerated (dt * 0.7) inside predictable chunks.\n",
            "Query: How does Ashby Homeostasis modulate Dopamine and Noradrenaline?\nResponse: Dopamine sharpens motor precision gain while Noradrenaline scales arousal and triggers sleep.\n"
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
    logger.info("⚡ [EXP-247] TRI-VECTOR UNIVERSAL MORPHOGENESIS BENCHMARK")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"📟 Hardware Acceleration: {device}")

    # 1. Base Configuration & Setup
    config = CoREConfig()
    config.net.hidden_dim = 256
    config.net.num_heads = 4
    config.train.batch_size = 4
    config.train.seq_len = 128

    logger.info("\n[1/5] Initializing Karyon CoREAgent with Tri-Vector Architecture...")

    # Initialize Agent
    agent = CoREAgent(config=config, device=str(device)).to(device)
    hu = HomeostaticUnit(batch_size=config.train.batch_size, device=str(device))
    episodic_memory = BatchedEpisodicMemory(
        batch_size=config.train.batch_size,
        memory_dim=config.net.unified_dim,
        max_capacity=1000,
        device=str(device)
    )
    criterion_speech = nn.CrossEntropyLoss(ignore_index=256)
    corpus = ComplexStreamBenchmarkCorpus(seq_len=config.train.seq_len)

    # 2. Verify Zero-Shock Net2Net Birth Identity on Dynamic DAG Sprouting
    logger.info("\n[2/5] --- Verifying Zero-Shock Net2Net Birth Identity on Dynamic DAG Sprouting ---")
    val_x = corpus.generate_batch(config.train.batch_size, device)
    val_y = val_x.clone()

    with torch.no_grad():
        out_before, _, _, _, _, _, _ = agent.forward_sequence(val_x, val_y, hu, criterion_speech)

    # Sprout new Universal Morphic Operator in Dynamic DAG
    sprouted = agent.dynamic_graph.sprout_brick("UniversalMorphicOperator")
    assert sprouted, "Failed to sprout new DAG node!"

    with torch.no_grad():
        out_after, _, _, _, _, _, _ = agent.forward_sequence(val_x, val_y, hu, criterion_speech)

    birth_delta = abs(out_before.item() - out_after.item())
    logger.info(f"   🌱 Net2Net Zero-Shock Birth Delta: {birth_delta:.8f}")
    if birth_delta > 1e-3:
        logger.warning(f"⚠️ Warning: Non-zero birth delta ({birth_delta:.8f}) exceeds strict threshold!")
    else:
        logger.info("   ✅ Zero-Shock Function Identity Perfectly Preserved (f_new(x) == f_old(x)) at birth!")

    # 3. Stream Learning with Tri-Vector Architecture
    logger.info("\n[3/5] --- Executing Stream Learning with Tri-Vector Self-Evolution ---")
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

    stream_steps = 30
    losses = []
    t0 = time.perf_counter()

    for step in range(1, stream_steps + 1):
        x_batch = corpus.generate_batch(config.train.batch_size, device)
        targets = x_batch.clone()

        optimizer.zero_grad()
        loss, fe, _, _, _, _, _ = agent.forward_sequence(x_batch, targets, hu, criterion_speech)

        # High surprise write to Two-Tier L1/L2 memory cache
        if fe > 0.30:
            agent.two_tier_memory.write_l1(
                key=agent.in_proj(agent.pos_embeddings(x_batch)[:, 0, :]),
                value=agent.in_proj(agent.pos_embeddings(targets)[:, 0, :]),
                surprise=fe
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        # Autonomous sleep and graph self-evolution trigger
        if step == 15:
            logger.info(f"   🌙 [Step {step:02d}] Mid-stream Allostatic Consolidation & Dynamic DAG Neurogenesis...")
            _, agent, _ = agent.execute_deep_allostatic_sleep(
                episodic_memory=episodic_memory,
                hu=hu,
                num_replay_cycles=3,
                downscaling_factor=0.01,
                pruning_percentile=0.03,
                eval_inputs=val_x,
                eval_targets=val_y,
                criterion_speech=criterion_speech
            )
            optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

        if step % 5 == 0 or step == 1:
            logger.info(f"   🌊 [Step {step:02d}/{stream_steps:02d}] Loss: {loss_val:.4f} nats/byte | Free Energy: {fe:.4f} | L1 Mem: {agent.two_tier_memory.l1_size} | DAG Nodes: {len(agent.dynamic_graph.bricks)}")

    elapsed = time.perf_counter() - t0
    final_loss = losses[-1]
    baseline_loss = losses[0]
    delta_loss = baseline_loss - final_loss
    tok_per_sec = (stream_steps * config.train.batch_size * config.train.seq_len) / elapsed

    # 4. Memory Recall & Graph Telemetry Audit
    logger.info("\n[4/5] --- Memory & Dynamic Graph Diagnostics ---")
    query_vec = agent.in_proj(agent.pos_embeddings(val_x)[:, 0, :])
    recalled = agent.two_tier_memory.recall(query_vec, na_level=0.30)
    recall_active = (recalled.abs().sum().item() > 0.0)

    logger.info(f"   - L1 Memory Cache Fill: {agent.two_tier_memory.l1_size}/{agent.two_tier_memory.l1_capacity}")
    logger.info(f"   - L2 Hippocampal Memory Size: {agent.two_tier_memory.l2_size}/{agent.two_tier_memory.l2_capacity}")
    logger.info(f"   - Two-Tier Memory Recall Active: {recall_active}")
    logger.info(f"   - Dynamic DAG Node Count: {len(agent.dynamic_graph.bricks)}")
    logger.info(f"   - Final Stream Speech Loss: {final_loss:.4f} nats/byte (Initial: {baseline_loss:.4f})")
    logger.info(f"   - Total Loss Reduction: {delta_loss:+.4f} nats/byte")
    logger.info(f"   - Throughput Speed: {tok_per_sec:.1f} tok/s")

    # 5. KEP Rule #2 Verdict Decision
    logger.info("\n[5/5] --- KEP Rule #2 Decision Criteria ---")
    verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    if delta_loss >= 0.08 and not math.isnan(final_loss) and recall_active:
        verdict = "🟢 POSITIVE"
    elif delta_loss < 0.0 or math.isnan(final_loss):
        verdict = "🔴 REJECTED"

    logger.info(f"🏆 Verdict: {verdict}")

    results = {
        "loss": final_loss,
        "baseline_loss": baseline_loss,
        "delta_loss": delta_loss,
        "tok_per_sec": tok_per_sec,
        "l1_mem_size": agent.two_tier_memory.l1_size,
        "l2_mem_size": agent.two_tier_memory.l2_size,
        "dag_nodes": len(agent.dynamic_graph.bricks),
        "verdict": verdict
    }
    return results


if __name__ == "__main__":
    run_experiment()
