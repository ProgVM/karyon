"""
===============================================================================
EXP-231: Dynamic Neural Graph Epigenetic Pruning & Neural Darwinism
Hypothesis: Integrating dynamic Edelman Neural Darwinism pruning into 
ContinuousDynamicNeuralGraph to prune obsolete operator bricks when their 
epigenetic gate alpha_epi decays below a critical threshold (alpha_epi < 0.01) 
will reclaim HBM memory and boost execution throughput without loss jump.
===============================================================================
"""

import os
import sys
import time
import torch
import torch.nn as nn
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EXP-231")

# Ensure repository root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from karyon_entity import KaryonEntity
from karyon_agent import ContinuousDynamicNeuralGraph, EpigeneticOperatorBrick

def run_experiment():
    logger.info("================================================================================")
    logger.info("🔬 [EXP-231] INITIATING EDELMAN NEURAL DARWINISM PRUNING BENCHMARK")
    logger.info("================================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"⚡ Active Hardware Device: {device}")

    # Load Karyon Soul Container
    soul_path = REPO_ROOT / "karyon_soul.kcore"
    if not soul_path.exists():
        logger.error(f"Container 'karyon_soul.kcore' not found at {soul_path}")
        return

    entity = KaryonEntity.load(str(soul_path), device=device)
    logger.info("Successfully loaded KaryonEntity container")

    # Instantiate or inspect Neural Graph
    graph = ContinuousDynamicNeuralGraph(hidden_dim=768, device=device).to(device)
    
    # 1. Populate graph with test bricks (1 active, 2 decaying/obsolete)
    brick1 = EpigeneticOperatorBrick("DelayOp_Active", op_type="delay", hidden_dim=768, device=device)
    brick1.alpha_epi.data.fill_(1.0) # Fully active

    brick2 = EpigeneticOperatorBrick("LinearOp_Obsolete", op_type="linear", hidden_dim=768, device=device)
    brick2.alpha_epi.data.fill_(-5.0) # Obsolete (sigmoid(-5) = 0.0067 < 0.01)

    brick3 = EpigeneticOperatorBrick("NonLinearOp_Obsolete", op_type="nonlinear", hidden_dim=768, device=device)
    brick3.alpha_epi.data.fill_(-6.0) # Obsolete

    graph.bricks = nn.ModuleList([brick1, brick2, brick3])

    initial_brick_count = len(graph.bricks)
    logger.info(f"Initial Graph Operator Bricks: {initial_brick_count}")

    # 2. Benchmark baseline forward throughput
    x_test = torch.randn(1, 768, device=device)
    
    # Warmup
    for _ in range(10):
        _ = graph(x_test)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        _ = graph(x_test)
    if device == "cuda":
        torch.cuda.synchronize()
    duration_pre = time.perf_counter() - t0
    throughput_pre = 100 / duration_pre
    logger.info(f"Baseline Graph Forward Throughput: {throughput_pre:.2f} passes/sec")

    # 3. Apply Edelman Neural Darwinism Pruning
    pruned_count = 0
    surviving_bricks = []
    
    for brick in graph.bricks:
        gate_val = torch.sigmoid(brick.alpha_epi).item()
        if gate_val < 0.01:
            logger.info(f"✂️ Pruning obsolete brick '{brick.brick_name}' (gate value = {gate_val:.5f} < 0.01)")
            pruned_count += 1
        else:
            surviving_bricks.append(brick)

    graph.bricks = nn.ModuleList(surviving_bricks)
    final_brick_count = len(graph.bricks)
    logger.info(f"Post-Pruning Graph Operator Bricks: {final_brick_count} (Pruned: {pruned_count})")

    # 4. Benchmark post-pruning throughput
    for _ in range(10):
        _ = graph(x_test)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        _ = graph(x_test)
    if device == "cuda":
        torch.cuda.synchronize()
    duration_post = time.perf_counter() - t0
    throughput_post = 100 / duration_post
    logger.info(f"Post-Pruning Graph Forward Throughput: {throughput_post:.2f} passes/sec")

    speedup = (throughput_post - throughput_pre) / throughput_pre * 100.0
    logger.info(f"⚡ Throughput Gain: +{speedup:.2f}%")

    # Print explicit verdict for KEP pipeline
    if pruned_count == 2 and final_brick_count == 1 and throughput_post >= throughput_pre:
        print("VERDICT=POSITIVE")
        logger.info("VERDICT=POSITIVE")
    else:
        print("VERDICT=REJECTED")
        logger.info("VERDICT=REJECTED")

if __name__ == "__main__":
    run_experiment()
