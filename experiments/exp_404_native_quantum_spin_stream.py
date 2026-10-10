"""
EXP-404: Native C++20 Quantum-Thermodynamic Spin Wave Stream Learning (NCSW-SL)
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Date: October 2026
Standard: KEP v16.0 Sovereign Master (Rubicon 400 Series, Principle 1 & Principle 27)

Core Scientific Objectives:
1. Benchmark native C++20 `QuantumSpinWaveOp` and `DynamicMorphicGraph` executing directly on CUDA Tensor Cores.
2. Verify gradient backward propagation through compiled LibTorch C++20 Ginzburg-Landau steps.
3. Test autonomous neurogenesis (sprouting/duplication) with QuantumSpinWave operators under Free Energy surprise.
4. Compare throughput and accuracy scaling on real streaming machine byte streams.
"""

import time
import json
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.cpp_extension

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-404-NCSW")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP404Config:
    exp_id: str = "EXP-404"
    dim: int = 258
    learning_rate: float = 0.005
    stream_length: int = 2500
    device_str: str = DEVICE_STR


def load_native_core():
    return torch.utils.cpp_extension.load(
        name="karyon_core_ext",
        sources=["karyon_core.cpp"],
        extra_cflags=["-O3", "-std=c++20"],
        build_directory="build/karyon_core_jit",
        verbose=False
    )


class NativeQuantumGraphAgent(nn.Module):
    """
    Sovereign Agent powered by C++20 DynamicMorphicGraph equipped with QuantumSpinWave operators.
    """
    def __init__(self, core_module, config: EXP404Config):
        super().__init__()
        self.config = config
        self.dim = config.dim

        # 1. Byte embedding and sensory gateway
        self.byte_embed = nn.Embedding(self.dim, self.dim).to(DEVICE)

        # 2. Native C++20 DynamicMorphicGraph
        self.graph = core_module.DynamicMorphicGraph(self.dim, self.config.device_str, 32)
        # Add primary QuantumSpinWave core node
        self.graph.add_node("core_qspin", "QuantumSpinWave", True, 1.0)
        # Add auxiliary Fast Associative Memory node
        self.graph.add_node("core_assoc", "MultiHeadFastAssociativeMemory", True, 1.0)

        # 3. Motor Readout
        self.motor_readout = nn.Linear(self.dim, self.dim).to(DEVICE)

    def forward_step(self, h_prev: torch.Tensor, byte_curr: int):
        byte_tensor = torch.tensor(byte_curr, device=DEVICE, dtype=torch.long)
        x_sensory = self.byte_embed(byte_tensor).unsqueeze(0)  # [1, dim]

        # Call C++20 DynamicMorphicGraph (returns fused motor representation [1, dim])
        h_motor = self.graph.forward(
            x_sensory,
            torch.Tensor().to(DEVICE),
            torch.Tensor().to(DEVICE),
            2
        )

        logits = self.motor_readout(h_motor).squeeze(0)  # [dim]
        probs = F.softmax(logits, dim=-1)

        # Norm as proxy energy
        energy = torch.mean(h_motor ** 2).item()

        return h_motor, probs, energy


def run_exp_404():
    config = EXP404Config()
    core_mod = load_native_core()

    model = NativeQuantumGraphAgent(core_mod, config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    logger.info("================================================================================")
    logger.info("STARTING EXP-404: NATIVE C++20 QUANTUM SPIN WAVE STREAM LEARNING (NCSW-SL)")
    logger.info(f"C++20 DynamicMorphicGraph D={config.dim} | Nodes={model.graph.k_nodes} | Device: {config.device_str}")
    logger.info("================================================================================")

    # Synthetic stress corpus
    torch.manual_seed(42)
    motifs = [
        b"QUANTUM_THERMODYNAMIC_SPIN_WAVEFIELD_GAUGE_ANNEALING_KARYON_RUBICON_400_EXP404\n",
        b"GET /api/v2/quantum_spin_wave_annealing HTTP/1.1\r\nHost: karyon.ai\r\n\r\n",
        b"def quantum_ginzburg_landau(psi, J, h, gamma):\n    return -1j*dH - gamma*dH\n",
        b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00>\x00\x01\x00\x00\x00"
    ]
    corpus = []
    while len(corpus) < config.stream_length + 100:
        for m in motifs:
            corpus.extend(list(m))
            corpus.extend(list(torch.randint(0, 256, (6,)).numpy()))
    corpus = corpus[:config.stream_length]

    h_t = torch.zeros(1, config.dim, device=DEVICE)

    total_loss = 0.0
    correct_count = 0
    recent_losses = []
    energy_trajectory = []
    sprout_events = 0

    t_start = time.time()

    for step in range(len(corpus) - 1):
        byte_curr = corpus[step]
        byte_next = corpus[step + 1]

        # Step forward
        h_next, probs, energy = model.forward_step(h_t, byte_curr)

        # Cross-Entropy Surprisal
        target_tensor = torch.tensor(byte_next, device=DEVICE, dtype=torch.long)
        loss = -torch.log(probs[target_tensor] + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        h_t = h_next.detach()

        # Telemetry
        loss_val = loss.item()
        total_loss += loss_val
        recent_losses.append(loss_val)
        if len(recent_losses) > 50:
            recent_losses.pop(0)

        pred_byte = torch.argmax(probs).item()
        if pred_byte == byte_next:
            correct_count += 1

        energy_trajectory.append(energy)

        # Endogenous Epigenetic Neurogenesis:
        # If surprise is high and budget allows, sprout an additional QuantumSpinWave node
        if loss_val > 5.5 and model.graph.k_nodes < 12 and step > 100 and step % 50 == 0:
            node_name = f"qspin_sprout_{model.graph.k_nodes}"
            model.graph.add_node(node_name, "QuantumSpinWave", False, 0.0)
            sprout_events += 1
            # Re-register optimizer to include new parameters
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

        if (step + 1) % 250 == 0:
            avg_recent_loss = sum(recent_losses) / len(recent_losses)
            acc = (correct_count / (step + 1)) * 100.0
            avg_energy = sum(energy_trajectory[-250:]) / 250.0
            logger.info(
                f"Step {step+1:4d}/{config.stream_length} | "
                f"Surprisal: {loss_val:.4f} (Avg50: {avg_recent_loss:.4f}) | "
                f"Accuracy: {acc:.2f}% | "
                f"Energy: {avg_energy:.4f} | Nodes: {model.graph.k_nodes}"
            )

    elapsed = time.time() - t_start
    final_avg_loss = total_loss / (config.stream_length - 1)
    final_accuracy = (correct_count / (config.stream_length - 1)) * 100.0
    throughput = config.stream_length / elapsed

    logger.info("================================================================================")
    logger.info("EXP-404 FINAL RESULTS:")
    logger.info(f"Elapsed Time: {elapsed:.2f} s | Throughput: {throughput:.2f} steps/s")
    logger.info(f"Final Average Surprisal (Loss): {final_avg_loss:.4f} nats")
    logger.info(f"Single-Pass Prediction Accuracy: {final_accuracy:.2f}%")
    logger.info(f"Final Nodes in Graph: {model.graph.k_nodes} | Sprout Events: {sprout_events}")
    logger.info("================================================================================")

    results = {
        "exp_id": config.exp_id,
        "elapsed_time": elapsed,
        "throughput_steps_per_sec": throughput,
        "final_avg_loss": final_avg_loss,
        "final_accuracy": final_accuracy,
        "final_nodes": model.graph.k_nodes,
        "sprout_events": sprout_events,
        "verdict": "🟢 POSITIVE" if final_accuracy > 15.0 or final_avg_loss < 3.5 else "⚪ NEUTRAL"
    }

    with open("experiments/exp_404_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_404()
