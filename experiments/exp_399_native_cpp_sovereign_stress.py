"""
EXP-399: Native C++20 Sovereign Morphic Graph & Fast Associative Memory Stress Suite
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Standard: KEP v16.0 Sovereign Master (Strict Principle 1 C++20, Principle 27 Autopoiesis, N=1 Stream)

Stress Test Scenarios:
1. High-load streaming perception across raw UTF-8 machine bytes and noisy text (2500+ uninterrupted steps).
2. Continuous real-time neurogenesis under stress:
   - Dynamic node sprouting via C++20 DynamicMorphicGraph::add_node
   - Epigenetic weight growth and mutation in-flight
   - Autonomous Neurodarwinian pruning of inactive paths
3. Synaptic saturation & interference stress on C++20 MultiHeadFastAssociativeMemory (6 heads).
4. Deterministic Orthogonal Metric Stress with C++20 StrictOrthogonalNexus (Margin M=0.40).
"""

import math
import time
import json
import logging
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-399-STRESS")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

# Load compiled C++20 LibTorch extension
logger.info(f"Loading native C++20 LibTorch extension on {DEVICE_STR}...")
karyon_core = load(
    name="karyon_core_ext",
    sources=["karyon_core.cpp"],
    extra_cflags=["-O3", "-std=c++20"],
    verbose=False
)
logger.info("C++20 LibTorch extension loaded successfully!")


@dataclass
class EXP399Config:
    exp_id: str = "EXP-399"
    dim: int = 258
    num_heads: int = 6
    max_nodes: int = 32
    initial_nodes: int = 4
    learning_rate: float = 0.005
    fast_weight_eta: float = 0.25
    margin: float = 0.40
    sprout_patience: int = 25
    sprout_surprise_threshold: float = 0.85
    stream_length: int = 2500
    device_str: str = DEVICE_STR


class SovereignMorphicAgent(nn.Module):
    """
    Sovereign Agent powered by C++20 DynamicMorphicGraph,
    StrictOrthogonalNexus, and MultiHeadFastAssociativeMemory.
    """
    def __init__(self, config: EXP399Config):
        super().__init__()
        self.config = config
        self.dim = config.dim

        # 1. Native C++20 Strict Orthogonal Nexus (D=258)
        self.nexus = karyon_core.StrictOrthogonalNexus(self.dim, config.device_str)

        # 2. Native C++20 Dynamic Morphic Computational Graph
        self.graph = karyon_core.DynamicMorphicGraph(self.dim, config.device_str, config.max_nodes)
        self.graph.add_node("root_accum", "LinearAccumulator", True, 1.0)
        self.graph.add_node("fast_mem", "MultiHeadFastAssociativeMemory", True, 0.5)
        self.graph.add_node("attractor", "SaturatedAttractor", False, 0.2)
        self.graph.add_node("bilinear", "BilinearMultiplicative", False, 0.1)

        # 3. Direct Fast Associative Memory handle for fast-weight writes
        self.fast_mem = karyon_core.MultiHeadFastAssociativeMemory(
            self.dim, config.num_heads, config.fast_weight_eta, config.device_str
        )

        # 4. Latent Thinking Recirculation Gating
        self.w_gate = nn.Linear(self.dim * 2, self.dim).to(DEVICE)
        self.layer_norm = nn.LayerNorm(self.dim).to(DEVICE)

    def perceive_byte(self, byte_val: int) -> torch.Tensor:
        return self.nexus.perceive(byte_val)

    def forward_step(self, x_in: torch.Tensor, prev_state: torch.Tensor, thinking_steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        # Morphic graph expects batched 2D tensor [1, dim]
        psi = prev_state.view(1, self.dim).clone()
        for _ in range(thinking_steps):
            graph_out = self.graph.forward(psi, torch.Tensor(), torch.Tensor(), 1)
            mem_out = self.fast_mem.read_memory(psi)
            combined = torch.cat([graph_out, mem_out], dim=-1)
            gate = torch.sigmoid(self.w_gate(combined))
            psi = self.layer_norm(psi + gate * (graph_out + mem_out))
        psi_flat = psi.squeeze(0)
        return psi_flat, self.nexus.compute_margin_free_energy(psi_flat, 0, self.config.margin)[0]


def run_sovereign_stress_suite():
    config = EXP399Config()
    agent = SovereignMorphicAgent(config)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    logger.info("================================================================================")
    logger.info("STARTING EXP-399 NATIVE C++20 SOVEREIGN STRESS SUITE")
    logger.info(f"Target Sequence Steps: {config.stream_length} | Device: {config.device_str}")
    logger.info("================================================================================")

    # Synthetic noisy machine byte stream with structural repeating patterns
    torch.manual_seed(42)
    corpus = []
    motifs = [
        b"GET /api/v1/telemetry HTTP/1.1\r\nHost: karyon.local\r\n\r\n",
        b"def autopoietic_loop(psi, free_energy):\n    return psi - 0.01 * grad(free_energy)\n",
        b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00>\x00\x01\x00\x00\x00",
        b"SOVEREIGN_CYBERNETIC_LAW_PRINCIPLE_27_ZERO_HARDCODE_METRIC_CONTINUOUS_GENESIS\n"
    ]
    while len(corpus) < config.stream_length + 100:
        for m in motifs:
            corpus.extend(list(m))
            # Inject structured noise
            corpus.extend(list(torch.randint(0, 256, (8,)).numpy()))

    corpus = corpus[:config.stream_length]
    logger.info(f"Generated stress corpus: {len(corpus)} raw bytes.")

    h_t = torch.zeros(config.dim, device=DEVICE)
    consecutive_high_surprise = 0
    sprout_events = 0
    prune_events = 0
    total_energy = 0.0
    correct_predictions = 0

    step_times = []
    recent_energies = []

    t_start = time.time()

    for step in range(len(corpus) - 1):
        t0 = time.time()
        byte_curr = corpus[step]
        byte_next = corpus[step + 1]

        # 1. Perception on Orthonormal Nexus
        x_curr = agent.perceive_byte(byte_curr)

        # 2. Recurrent Forward Thinking with C++20 Morphic Graph
        psi_pred, _ = agent.forward_step(x_curr, h_t, thinking_steps=2)

        # 3. Margin Free Energy via C++20 StrictOrthogonalNexus
        energy, pred_idx, target_score, max_comp = agent.nexus.compute_margin_free_energy(
            psi_pred, byte_next, config.margin
        )

        loss = energy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        # 4. In-flight Fast Associative Memory Write (Under NoGrad)
        x_target = agent.perceive_byte(byte_next)
        agent.fast_mem.write_memory(x_curr, x_target)

        # Update state
        h_t = psi_pred.detach()

        # Telemetry
        is_correct = (pred_idx == byte_next)
        if is_correct:
            correct_predictions += 1
        energy_val = energy.item()
        total_energy += energy_val
        recent_energies.append(energy_val)
        if len(recent_energies) > 50:
            recent_energies.pop(0)

        # 5. Neurogenesis Stress: Autonomous Dynamic Sprouting under persistent surprise
        if energy_val > config.sprout_surprise_threshold:
            consecutive_high_surprise += 1
        else:
            consecutive_high_surprise = max(0, consecutive_high_surprise - 1)

        if consecutive_high_surprise >= config.sprout_patience:
            consecutive_high_surprise = 0
            if agent.graph.k_nodes < config.max_nodes:
                sprout_type = "MultiHeadFastAssociativeMemory" if sprout_events % 2 == 0 else "SaturatedAttractor"
                node_name = f"sprout_{sprout_events}_{sprout_type}"
                agent.graph.add_node(node_name, sprout_type, False, 0.05)
                sprout_events += 1
                logger.info(f"[Step {step}] 🧬 Autonomous Sprout Event #{sprout_events}: Added '{node_name}' (Total nodes: {agent.graph.k_nodes})")

        # 6. Periodic Neurodarwinian Pruning Stress
        if step > 0 and step % 500 == 0:
            initial_k = agent.graph.k_nodes
            # Simulated pruning on low alpha nodes
            agent.graph.prune_inactive_nodes(0.01)
            pruned = initial_k - agent.graph.k_nodes
            if pruned > 0:
                prune_events += pruned
                logger.info(f"[Step {step}] ✂️ Pruning Event: Removed {pruned} inactive nodes (Remaining: {agent.graph.k_nodes})")

        step_times.append(time.time() - t0)

        if (step + 1) % 250 == 0:
            avg_recent_energy = sum(recent_energies) / len(recent_energies)
            accuracy = (correct_predictions / (step + 1)) * 100.0
            avg_step_ms = (sum(step_times[-250:]) / 250.0) * 1000.0
            logger.info(
                f"Step {step+1:4d}/{config.stream_length} | "
                f"Energy: {energy_val:.4f} (Avg50: {avg_recent_energy:.4f}) | "
                f"Accuracy: {accuracy:.2f}% | "
                f"Nodes: {agent.graph.k_nodes} | "
                f"Sprouts: {sprout_events} | "
                f"Step Latency: {avg_step_ms:.2f} ms"
            )

    elapsed_total = time.time() - t_start
    final_avg_energy = total_energy / (config.stream_length - 1)
    final_accuracy = (correct_predictions / (config.stream_length - 1)) * 100.0
    throughput = config.stream_length / elapsed_total

    logger.info("================================================================================")
    logger.info("EXP-399 NATIVE C++20 SOVEREIGN STRESS SUITE: FINAL RESULTS")
    logger.info(f"Total Elapsed Time: {elapsed_total:.2f} s | Throughput: {throughput:.2f} steps/s")
    logger.info(f"Final Average Free Energy: {final_avg_energy:.4f}")
    logger.info(f"Single-Pass Prediction Accuracy: {final_accuracy:.2f}%")
    logger.info(f"Total Sprout Events: {sprout_events} | Total Prune Events: {prune_events}")
    logger.info(f"Final Graph Node Count: {agent.graph.k_nodes}")
    logger.info("================================================================================")

    # Verification of zero NaNs and memory stability
    assert not math.isnan(final_avg_energy), "Free Energy produced NaN!"
    assert not math.isinf(final_avg_energy), "Free Energy produced Inf!"

    results = {
        "exp_id": config.exp_id,
        "elapsed_total": elapsed_total,
        "throughput_steps_per_sec": throughput,
        "final_avg_energy": final_avg_energy,
        "final_accuracy": final_accuracy,
        "sprout_events": sprout_events,
        "prune_events": prune_events,
        "final_nodes": agent.graph.k_nodes,
        "verdict": "POSITIVE" if final_avg_energy < 1.0 else "NEUTRAL"
    }

    with open("experiments/exp_399_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_sovereign_stress_suite()
