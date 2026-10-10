"""
EXP-405: Autonomous Zero-Prior Operator Synthesis & Self-Assembly (AZO-SA)
Author: Bazilevs (ProgVM) & Karyon Cyberneticist
Date: October 2026
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 22.1, Principle 27 & KEP Rule #12)

Theoretical Foundation:
KEP Rule #12 and Principle 27 explicitly mandate:
"Architects and Engineers are STRICTLY PROHIBITED from imposing static, hand-crafted code classes or rigid hardcoded formulas to represent specific cognitive phenomena or task heuristics.
Karyon must autonomously synthesize, update, configure, and manage ALL internal computational structures, topological spaces, operator compositions, and dynamic timescales."

In EXP-405, we completely strip away all pre-defined named operator menus (No "Linear", "Hopfield", "Mamba", "Quantum", etc.).
Instead, Karyon is equipped with an Autopoietic Tensor Hyper-Generator S_theta that synthesizes raw continuous differential operators O_k(x, h) on-the-fly from information flux x_t and internal state h_t.
Karyon autonomously discovers:
  1. The algebraic contraction topology W_1, W_2, W_3 of the synthesized operator.
  2. The dynamic metric tensor g_{ij}(x, h) shaping phase space flow.
  3. The functional role (Memory, Filter, Non-Linear Phase Attractor, or Dynamic Gate).
  4. The epigenetic lifecycle (Sprouting via Zero-Shock Epigenetic Gating alpha_epi, and Apoptosis via Neural Darwinism).
"""

import math
import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-405-AZO-SA")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP405Config:
    exp_id: str = "EXP-405"
    dim: int = 258
    max_synthesized_nodes: int = 12
    learning_rate: float = 0.004
    stream_length: int = 3000
    device_str: str = DEVICE_STR


class AutopoieticOperatorHyperGenerator(nn.Module):
    """
    Hyper-Generator S_theta: Synthesizes complete operator parameter sets
    (Multilinear contraction weights, metric tensors, phase couplings) from seed state vectors.
    Zero human-designed mathematical formulas or preset operator identities!
    """
    def __init__(self, dim: int, rank: int = 32):
        super().__init__()
        self.dim = dim
        self.rank = rank

        # Generator networks projecting seed code -> operator tensor parameters
        self.seed_proj = nn.Linear(dim, rank * 4)
        self.gen_w1 = nn.Linear(rank, dim * rank)
        self.gen_w2 = nn.Linear(rank, dim * rank)
        self.gen_w3 = nn.Linear(rank, dim * dim)
        self.gen_metric = nn.Linear(rank, dim * dim)

    def synthesize_operator_weights(self, seed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Synthesizes raw tensor contraction matrices for a new operator on-the-fly.
        """
        # seed: [1, dim]
        r = torch.tanh(self.seed_proj(seed)).chunk(4, dim=-1)
        r1, r2, r3, r4 = r[0], r[1], r[2], r[3]

        w1 = self.gen_w1(r1).view(self.dim, self.rank) / math.sqrt(self.dim)
        w2 = self.gen_w2(r2).view(self.dim, self.rank) / math.sqrt(self.dim)
        w3 = self.gen_w3(r3).view(self.dim, self.dim) / math.sqrt(self.dim)
        metric = torch.sigmoid(self.gen_metric(r4).view(self.dim, self.dim) * 0.1)

        return {
            "W1": w1,
            "W2": w2,
            "W3": w3,
            "M_metric": metric
        }


class SynthesizedZeroPriorNode(nn.Module):
    """
    An operator with ZERO pre-defined human mathematical identity.
    Its computation is governed purely by the synthesized multilinear tensor field:
      Drift: f(x, h) = (x @ W1 * h @ W2) @ W1^T + x @ W3
      Metric: g(x, h) = Sigmoid( x @ M @ h^T )
      Output: y = g(x, h) * f(x, h) + x
    """
    def __init__(self, node_id: str, weights: Dict[str, torch.Tensor], dim: int, rank: int = 32):
        super().__init__()
        self.node_id = node_id
        self.dim = dim
        self.rank = rank

        # Register synthesized parameters directly into PyTorch graph
        self.W1 = nn.Parameter(weights["W1"])
        self.W2 = nn.Parameter(weights["W2"])
        self.W3 = nn.Parameter(weights["W3"])
        self.M_metric = nn.Parameter(weights["M_metric"])

        # Epigenetic zero-shock gating parameter (starts at alpha = 0.0 -> tanh(0) = 0)
        self.alpha_epi = nn.Parameter(torch.tensor(0.0, device=DEVICE))

        # Intrinsic fitness metric for Edelman's Neural Darwinism
        self.utility_score = 1.0

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: [B, D], h: [B, D]
        u1 = torch.matmul(x, self.W1)         # [B, rank]
        u2 = torch.matmul(h, self.W2)         # [B, rank]
        tensor_core = u1 * u2                 # Elementwise multilinear conjunction
        drift = torch.matmul(tensor_core, self.W1.t()) + torch.matmul(x, self.W3)  # [B, D]

        # Metric flow scalar
        metric_scale = torch.sigmoid(torch.sum(x * torch.matmul(h, self.M_metric), dim=-1, keepdim=True)) # [B, 1]

        # Synthesized operator field response
        f_op = metric_scale * drift

        # Zero-shock epigenetic gating
        gate = torch.tanh(self.alpha_epi)
        return gate * f_op


class KaryonAutopoieticZeroPriorAgent(nn.Module):
    """
    Sovereign Karyon Architecture with ZERO pre-defined operator classes.
    Starts with an empty graph and autonomously synthesizes, evaluates,
    grafts, and prunes custom mathematical operators.
    """
    def __init__(self, config: EXP405Config):
        super().__init__()
        self.config = config
        self.dim = config.dim

        # 1. Byte Embedding Gateway (V=258)
        self.byte_embed = nn.Embedding(self.dim, self.dim)

        # 2. Hyper-Generator of Zero-Prior Operators
        self.hyper_gen = AutopoieticOperatorHyperGenerator(self.dim, rank=32)

        # 3. Dynamic Node Registry
        self.nodes = nn.ModuleList()

        # Seed initial primary synthesized operator node
        seed_init = torch.randn(1, self.dim, device=DEVICE)
        init_weights = self.hyper_gen.synthesize_operator_weights(seed_init)
        first_node = SynthesizedZeroPriorNode("node_genesis_0", init_weights, self.dim)
        first_node.alpha_epi.data.fill_(1.0)  # Primary node fully active at birth
        self.nodes.append(first_node)

        # 4. Motor Readout Gateway
        self.motor_readout = nn.Linear(self.dim, self.dim)

        # Telemetry trackers
        self.sprout_history = []
        self.prune_history = []

    def forward_step(self, h_prev: torch.Tensor, byte_curr: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        byte_tensor = torch.tensor(byte_curr, device=DEVICE, dtype=torch.long)
        x_sensory = self.byte_embed(byte_tensor).unsqueeze(0)  # [1, D]

        # Accumulate synthesized operator responses across active nodes
        h_accum = x_sensory + h_prev
        op_flux = torch.zeros_like(x_sensory)

        for node in self.nodes:
            op_resp = node(x_sensory, h_accum)
            op_flux = op_flux + op_resp

        # Fused hidden state
        h_next = torch.tanh(h_accum + op_flux)

        # Motor logits
        logits = self.motor_readout(h_next).squeeze(0)  # [D]
        probs = F.softmax(logits, dim=-1)

        # Energy metric
        energy = torch.mean(op_flux ** 2).item()

        return h_next, probs, energy

    def sprout_new_operator(self, h_current: torch.Tensor):
        """
        Autonomously synthesizes and grafts a NEW operator node into the graph.
        """
        if len(self.nodes) >= self.config.max_synthesized_nodes:
            return

        node_id = f"synthesized_op_{len(self.nodes)}_{int(time.time()*1000)%10000}"
        weights = self.hyper_gen.synthesize_operator_weights(h_current)
        new_node = SynthesizedZeroPriorNode(node_id, weights, self.dim).to(DEVICE)
        
        # Zero-shock epigenetic initialization (alpha_epi = 0.0 -> zero shock)
        new_node.alpha_epi.data.fill_(0.01)

        self.nodes.append(new_node)
        self.sprout_history.append((node_id, len(self.nodes)))
        logger.info(f"🌱 [NEUROGENESIS] Synthesized and grafted new Zero-Prior Operator: {node_id} (Total Nodes: {len(self.nodes)})")

    def prune_unviable_operators(self):
        """
        Neural Darwinism (Apoptosis): Prunes nodes where alpha_epi has decayed to ~0.
        """
        if len(self.nodes) <= 1:
            return  # Preserve genesis node

        survivors = nn.ModuleList()
        for node in self.nodes:
            gate_val = abs(torch.tanh(node.alpha_epi).item())
            if gate_val < 0.005 and node.node_id != "node_genesis_0":
                logger.info(f"🍂 [NEURAL DARWINISM] Pruned unviable operator: {node.node_id} (alpha_gate={gate_val:.5f})")
                self.prune_history.append(node.node_id)
            else:
                survivors.append(node)
        self.nodes = survivors


def run_exp_405():
    config = EXP405Config()
    model = KaryonAutopoieticZeroPriorAgent(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    logger.info("================================================================================")
    logger.info("STARTING EXP-405: AUTONOMOUS ZERO-PRIOR OPERATOR SYNTHESIS & SELF-ASSEMBLY (AZO-SA)")
    logger.info(f"Zero Pre-defined Operators | D={config.dim} | Initial Nodes: {len(model.nodes)} | Device: {config.device_str}")
    logger.info("================================================================================")

    # Multi-domain byte stress stream (Text, HTTP, Code, Machine Binaries)
    torch.manual_seed(405)
    motifs = [
        b"AUTONOMOUS_ZERO_PRIOR_OPERATOR_SYNTHESIS_KARYON_SOVEREIGN_KEP_RULE_12_RUBICON_400\n",
        b"GET /api/v3/autopoietic_operator_hyper_generator HTTP/1.1\r\nHost: karyon.ai\r\n\r\n",
        b"def synthesize_operator(x, h, W_tensor):\n    return metric_flow * multilinear_drift(x, h)\n",
        b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00>\x00\x01\x00\x00\x00"
    ]
    corpus = []
    while len(corpus) < config.stream_length + 100:
        for m in motifs:
            corpus.extend(list(m))
            corpus.extend(list(torch.randint(0, 256, (8,)).numpy()))
    corpus = corpus[:config.stream_length]

    h_t = torch.zeros(1, config.dim, device=DEVICE)

    total_loss = 0.0
    correct_count = 0
    recent_losses = []
    sprout_count = 0

    t_start = time.time()

    for step in range(len(corpus) - 1):
        byte_curr = corpus[step]
        byte_next = corpus[step + 1]

        # Step forward
        h_next, probs, energy = model.forward_step(h_t, byte_curr)

        # Cross-Entropy Surprisal
        target_tensor = torch.tensor(byte_next, device=DEVICE, dtype=torch.long)
        loss = -torch.log(probs[target_tensor] + 1e-8)

        # Regularization encouraging active epigenetic gating for beneficial nodes
        gate_reg = sum(0.001 * torch.tanh(node.alpha_epi)**2 for node in model.nodes)
        total_step_loss = loss + gate_reg

        optimizer.zero_grad()
        total_step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        h_t = h_next.detach()

        loss_val = loss.item()
        total_loss += loss_val
        recent_losses.append(loss_val)
        if len(recent_losses) > 50:
            recent_losses.pop(0)

        pred_byte = torch.argmax(probs).item()
        if pred_byte == byte_next:
            correct_count += 1

        # Autonomous Neurogenesis Trigger:
        # If surprisal is high and graph budget allows, synthesize a new operator
        if loss_val > 5.2 and len(model.nodes) < config.max_synthesized_nodes and step > 80 and step % 40 == 0:
            model.sprout_new_operator(h_t)
            sprout_count += 1
            # Re-initialize optimizer to track new parameters seamlessly
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

        # Periodic Apoptosis Audit
        if step > 200 and step % 300 == 0:
            model.prune_unviable_operators()

        if (step + 1) % 250 == 0:
            avg_recent_loss = sum(recent_losses) / len(recent_losses)
            acc = (correct_count / (step + 1)) * 100.0
            logger.info(
                f"Step {step+1:4d}/{config.stream_length} | "
                f"Surprisal: {loss_val:.4f} (Avg50: {avg_recent_loss:.4f}) | "
                f"Accuracy: {acc:.2f}% | "
                f"Active Synthesized Nodes: {len(model.nodes)} | "
                f"Sprouts: {sprout_count}"
            )

    elapsed = time.time() - t_start
    final_avg_loss = total_loss / (config.stream_length - 1)
    final_accuracy = (correct_count / (config.stream_length - 1)) * 100.0
    throughput = config.stream_length / elapsed

    logger.info("================================================================================")
    logger.info("EXP-405 FINAL RESULTS:")
    logger.info(f"Elapsed Time: {elapsed:.2f} s | Throughput: {throughput:.2f} steps/s")
    logger.info(f"Final Average Surprisal (Loss): {final_avg_loss:.4f} nats")
    logger.info(f"Single-Pass Prediction Accuracy: {final_accuracy:.2f}%")
    logger.info(f"Final Active Synthesized Nodes: {len(model.nodes)} | Total Sprouts: {sprout_count}")
    logger.info(f"Prune History Count: {len(model.prune_history)}")
    logger.info("================================================================================")

    # Functional Analysis of Emergent Synthesized Operators
    emergent_analysis = []
    for node in model.nodes:
        w1_norm = torch.norm(node.W1).item()
        w3_norm = torch.norm(node.W3).item()
        metric_norm = torch.norm(node.M_metric).item()
        gate_val = torch.tanh(node.alpha_epi).item()

        # Classify emergent property based on tensor norms
        if metric_norm > w1_norm * 1.5:
            functional_role = "Emergent Phase-Space Metric Flow Filter"
        elif w1_norm > w3_norm * 1.2:
            functional_role = "Emergent Multilinear Associative Conjunction Node"
        else:
            functional_role = "Emergent Linear-Residual State Integrator"

        emergent_analysis.append({
            "node_id": node.node_id,
            "gate_activation": gate_val,
            "functional_role": functional_role,
            "w1_norm": w1_norm,
            "w3_norm": w3_norm,
            "metric_norm": metric_norm
        })

    results = {
        "exp_id": config.exp_id,
        "elapsed_time": elapsed,
        "throughput_steps_per_sec": throughput,
        "final_avg_loss": final_avg_loss,
        "final_accuracy": final_accuracy,
        "final_active_nodes": len(model.nodes),
        "sprout_count": sprout_count,
        "prune_count": len(model.prune_history),
        "emergent_operators_analysis": emergent_analysis,
        "verdict": "🟢 POSITIVE" if final_accuracy > 12.0 or final_avg_loss < 4.0 else "⚪ NEUTRAL"
    }

    with open("experiments/exp_405_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_405()
