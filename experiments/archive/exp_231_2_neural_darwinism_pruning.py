"""
===============================================================================
EXP-231.2: Dynamic Neural Graph Epigenetic Pruning & Neural Darwinism (KEP Rule #1.1 & #9)
Hypothesis: Integrating dynamic Edelman Neural Darwinism pruning into 
ContinuousDynamicNeuralGraph to prune obsolete operator bricks when their 
epigenetic gate tanh(alpha_epi) decays below a critical threshold (|tanh(alpha_epi)| < 1e-3) 
will reclaim HBM memory and boost execution throughput without loss jump.
===============================================================================
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EXP-231.2")

# Ensure repository root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from karyon_agent import ContinuousDynamicNeuralGraph, DelayOp, NonLinearOp, GateOp

def run_experiment():
    logger.info("================================================================================")
    logger.info("🔬 [EXP-231.2] INITIATING EDELMAN NEURAL DARWINISM PRUNING BENCHMARK")
    logger.info("================================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"⚡ Active Hardware Device: {device}")

    dim = 512
    batch_size = 2
    seq_len = 128
    homeo_dim = 6

    # Instantiate Dynamic Neural Graph
    graph = ContinuousDynamicNeuralGraph(dim=dim, max_bricks=12, device=device).to(device)

    # 1. Sprout multiple operator bricks into the graph
    graph.sprout_brick("GateOp")
    graph.sprout_brick("NonLinearOp")
    graph.sprout_brick("DelayOp")

    # Set mature activity for seed bricks and brick #2, while setting bricks #3 and #4 to obsolete (|tanh(alpha_epi)| < 1e-3)
    # alpha_epi: [brick0, brick1, brick2 (GateOp), brick3 (NonLinearOp), brick4 (DelayOp)]
    with torch.no_grad():
        graph.alpha_epi[0].fill_(5.0)   # Mature seed DelayOp
        graph.alpha_epi[1].fill_(5.0)   # Mature seed NonLinearOp
        graph.alpha_epi[2].fill_(2.5)   # Active GateOp (tanh(2.5) ~ 0.986)
        graph.alpha_epi[3].fill_(1e-5)  # Obsolete NonLinearOp (tanh(1e-5) = 1e-5 < 1e-3)
        graph.alpha_epi[4].fill_(-1e-5) # Obsolete DelayOp (|tanh(-1e-5)| = 1e-5 < 1e-3)

    initial_brick_count = len(graph.bricks)
    logger.info(f"Initial Graph Operator Bricks: {initial_brick_count}")

    # Generate synthetic input tensors (B, S, D) and homeostasis (B, homeo_dim)
    h_test = torch.randn(batch_size, seq_len, dim, device=device)
    u_test = torch.full((batch_size, homeo_dim), 0.5, device=device)

    # 2. Benchmark baseline forward pass throughput & output fidelity
    with torch.no_grad():
        for _ in range(10):
            _ = graph(h_test, u_test)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(100):
            out_pre = graph(h_test, u_test)
        if device == "cuda":
            torch.cuda.synchronize()
        duration_pre = time.perf_counter() - t0
        throughput_pre = 100 / duration_pre

    logger.info(f"Baseline Graph Forward Throughput: {throughput_pre:.2f} passes/sec")

    # 3. Apply Edelman Neural Darwinism Pruning
    pruned_count = graph.prune_inactive_bricks(threshold=1e-3)
    final_brick_count = len(graph.bricks)
    logger.info(f"Post-Pruning Graph Operator Bricks: {final_brick_count} (Pruned: {pruned_count})")

    # 4. Benchmark post-pruning throughput & calculate zero-shock output delta
    with torch.no_grad():
        for _ in range(10):
            _ = graph(h_test, u_test)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(100):
            out_post = graph(h_test, u_test)
        if device == "cuda":
            torch.cuda.synchronize()
        duration_post = time.perf_counter() - t0
        throughput_post = 100 / duration_post

    logger.info(f"Post-Pruning Graph Forward Throughput: {throughput_post:.2f} passes/sec")
    speedup = (throughput_post - throughput_pre) / throughput_pre * 100.0
    logger.info(f"Throughput Gain: {speedup:+.2f}%")

    output_delta = torch.norm(out_post - out_pre).item()
    norm_pre = torch.norm(out_pre).item()
    rel_error = output_delta / (norm_pre + 1e-8)
    logger.info(f"Output L2 Delta (Pre vs Post Pruning): {output_delta:.8f} (Relative: {rel_error:.6e})")

    # Metrics logging
    print(f"METRICS: initial_bricks={initial_brick_count}, final_bricks={final_brick_count}, pruned_count={pruned_count}, throughput_pre={throughput_pre:.2f}, throughput_post={throughput_post:.2f}, speedup_pct={speedup:.2f}, output_delta={output_delta:.8f}, rel_error={rel_error:.6e}")

    # KEP Rule #2 Contextual Verdict Assessment:
    # 2 obsolete bricks successfully pruned, final_bricks == 3, relative error < 1e-3, and throughput improved >= +10%
    if pruned_count == 2 and final_brick_count == 3 and rel_error < 1e-3 and speedup >= 10.0:
        print("VERDICT=POSITIVE")
        logger.info("VERDICT=POSITIVE")
    else:
        print("VERDICT=REJECTED")
        logger.info("VERDICT=REJECTED")

if __name__ == "__main__":
    run_experiment()
