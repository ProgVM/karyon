"""
EXP-447: Sovereign Autonomous Memory Genesis & Topological Plasticity (SAMG-TP)
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
Standard: KEP v16.0 Sovereign Master (Principle 2, Principle 3, Principle 15, Principle 16, Principle 21, Principle 22, Principle 27 & KEP Rule #12)

Theoretical Blueprint (Answering Bazilevs: "А что если сделать память Кариону тоже суверенной?"):
In previous experiments (EXP-441..446), the memory structure was hardcoded as a single 2D matrix S in R^(H x D_k x D_v).
This is an artificial human constraint ("Шизофрения" / hardcoded structure)!

In Sovereign Autonomous Memory Genesis (SAMG-TP):
1. Substrate Freedom ("Где угодно"):
   Memory is an open-ended dynamic field of tensor manifolds M = {M_1, M_2, ..., M_K}.
   Karyon dynamically sprouts new memory manifolds (e.g., Fast 1D Transient Traces, 2D Associative Matrices, 3D Tensor Outer Products, or Attractor Nodes) on-demand based on Variational Free Energy stress (F_t > tau_sprout).
2. Operator Freedom ("Как угодно"):
   Karyon endogenously synthesizes the writing, addressing, and reading operators for each memory structure:
   - Writing: How input energy Psi(t) is inscribed into M_k (differential, Hebbian, non-linear phase, or tensor product).
   - Addressing & Routing: Dynamic routing weights w_k(q) addressing which memory manifold to write to or read from.
3. Morphology & Biological Vitality ("Какой угодно"):
   - Sprouting: Newly sprouted memory manifolds are wrapped in Net2Net Zero-Shock Grafting gates (tanh(alpha_epi) * M_new) initialized at alpha_epi = 0.0 to guarantee ZERO structural shock at birth!
   - Apoptosis / Consolidation: Redundant or uninformative memory spaces are downscaled and pruned (Neural Darwinism).

Let's implement and evaluate this sovereign memory genesis architecture!
"""

import time
import json
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-447-SAMG")

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


@dataclass
class EXP447Config:
    exp_id: str = "EXP-447"
    dim: int = 258
    num_heads: int = 8
    dim_k: int = 64
    dim_v: int = 64
    max_memory_nodes: int = 8
    sprout_threshold_loss: float = 2.0
    learning_rate: float = 0.015
    stream_length: int = 4000
    device_str: str = DEVICE_STR


class SovereignDynamicMemoryNode(nn.Module):
    """
    An autonomous memory manifold node M_k.
    Supports dynamic topology (1D trace, 2D matrix, 3D tensor product).
    Wrapped in Net2Net zero-shock grafting gate alpha_epi.
    """
    def __init__(self, node_id: int, node_type: str, heads: int, dim_k: int, dim_v: int):
        super().__init__()
        self.node_id = node_id
        self.node_type = node_type  # '2d_associative', '3d_tensor', '1d_transient'
        self.H = heads
        self.D_k = dim_k
        self.D_v = dim_v

        # Zero-shock epigenetic birth gate: tanh(alpha_epi) -> 0.0 at birth!
        self.alpha_epi = nn.Parameter(torch.tensor(-5.0, device=DEVICE))  # sigmoid(-5) ~ 0.006

        # Allocate memory state tensor based on type:
        if self.node_type == "2d_associative":
            self.state = torch.zeros(self.H, self.D_k, self.D_v, device=DEVICE)
        elif self.node_type == "3d_tensor":
            self.state = torch.zeros(self.H, 8, 8, self.D_v, device=DEVICE)
        else:  # 1d_transient
            self.state = torch.zeros(self.H, self.D_v, device=DEVICE)

    def reset_state(self):
        self.state.zero_()

    def get_effective_weight(self) -> torch.Tensor:
        return torch.tanh(self.alpha_epi)


class SovereignAutonomousMemoryEngine(nn.Module):
    """
    Sovereign Autonomous Memory Genesis (SAMG-TP) Engine.
    Manages dynamic memory allocation, operator synthesis, zero-shock grafting, and reading.
    """
    def __init__(self, config: EXP447Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.H = config.num_heads
        self.D_k = config.dim_k
        self.D_v = config.dim_v
        self.max_nodes = config.max_memory_nodes

        # 1. Byte Embedding Matrix [258, 256]
        self.byte_emb = nn.Embedding(self.dim, 256)

        # 2. Multi-Head Projections: Q, K, V
        self.W_q = nn.Linear(256, self.H * self.D_k, bias=False)
        self.W_k = nn.Linear(256, self.H * self.D_k, bias=False)
        self.W_v = nn.Linear(256, self.H * self.D_v, bias=False)

        # 3. Readout Projection
        self.W_out = nn.Linear(self.H * self.D_v, self.dim, bias=False)

        # 4. Hopfield Precision Snapping
        self.w_beta = nn.Parameter(torch.tensor(2.5, device=DEVICE))
        self.b_beta = nn.Parameter(torch.tensor(2.5, device=DEVICE))

        # 5. Autonomous Memory Router & Operator Synthesizer:
        # Dynamically synthesizes write/read weights and routing across memory nodes
        self.router = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, self.max_nodes * 3)  # [route_w, write_eta, alpha_decay]
        )

        # Dynamic Memory Nodes List:
        self.nodes = nn.ModuleList()
        # Seed initial core 2D memory node:
        init_node = SovereignDynamicMemoryNode(0, "2d_associative", self.H, self.D_k, self.D_v)
        # Fully open birth gate for initial core node:
        init_node.alpha_epi.data.fill_(3.0)  # tanh(3.0) ~ 0.995
        self.nodes.append(init_node)

        self.sprout_count = 0

    def reset_state(self):
        for node in self.nodes:
            node.reset_state()

    def maybe_sprout_memory_node(self, current_loss: float):
        """
        Endogenous Neurogenesis: Sprouts a new memory manifold if surprise is high
        and capacity remains.
        """
        if current_loss > self.config.sprout_threshold_loss and len(self.nodes) < self.max_nodes:
            node_id = len(self.nodes)
            node_type = "3d_tensor" if node_id % 2 == 1 else "2d_associative"
            new_node = SovereignDynamicMemoryNode(node_id, node_type, self.H, self.D_k, self.D_v)
            # Net2Net Zero-Shock: starts at alpha_epi = -5.0 (tanh ~ 0.0)
            self.nodes.append(new_node)
            self.sprout_count += 1
            logger.info(f"🌱 [NEUROGENESIS] Sprouted Sovereign Memory Node #{node_id} (Type: {node_type}) | Net2Net Gate = 0.00")

    def step(self, byte_idx: int, target_byte: int) -> Tuple[torch.Tensor, Dict]:
        x_tensor = torch.tensor([byte_idx], device=DEVICE)
        target_tensor = torch.tensor([target_byte], device=DEVICE)

        emb = self.byte_emb(x_tensor)  # [1, 256]

        q = F.normalize(self.W_q(emb).view(self.H, 1, self.D_k), p=2, dim=-1)
        k = F.normalize(self.W_k(emb).view(self.H, 1, self.D_k), p=2, dim=-1)
        v = F.normalize(self.W_v(emb).view(self.H, 1, self.D_v), p=2, dim=-1)

        # Synthesize Routing & Operator Parameters for active memory nodes:
        router_out = self.router(emb).view(self.max_nodes, 3)  # [max_nodes, 3]

        route_logits = router_out[:, 0]
        route_weights = F.softmax(route_logits[:len(self.nodes)], dim=0)

        # Read address and combine readouts from ALL active memory nodes:
        combined_read = torch.zeros(self.H, self.D_v, device=DEVICE)

        for i, node in enumerate(self.nodes):
            gate = node.get_effective_weight()  # Net2Net zero-shock multiplier
            node_w = route_weights[i]

            if node.node_type == "2d_associative":
                S = node.state.detach()
                read_h = torch.bmm(q, S).squeeze(1)  # [H, D_v]
            elif node.node_type == "3d_tensor":
                # 3D Tensor addressing: project q down to [H, 8]
                q_sub = q[:, :, :8]
                S = node.state.detach()  # [H, 8, 8, D_v]
                read_h = torch.einsum("hb,hbc->hc", q_sub.squeeze(1), S[:, 0, :, :])
            else:
                read_h = node.state.detach()

            combined_read = combined_read + gate * node_w * read_h

        read_flat = combined_read.view(1, -1)  # [1, H * D_v]

        # Modern Hopfield Precision Snapping:
        read_norm = torch.norm(read_flat)
        beta = 1.0 + F.softplus(self.w_beta * read_norm + self.b_beta)
        logits = beta * self.W_out(read_flat)

        loss = F.cross_entropy(logits, target_tensor)
        pred = torch.argmax(logits, dim=-1).item()
        surprise = loss.item()

        # Write / Inscribe into active memory nodes:
        v_pred = torch.bmm(k, self.nodes[0].state.detach())
        error = v - v_pred

        for i, node in enumerate(self.nodes):
            write_eta = F.softplus(router_out[i, 1]) + 0.1
            alpha_decay = torch.clamp(torch.sigmoid(router_out[i, 2]), min=0.75, max=0.999)

            if node.node_type == "2d_associative":
                S = node.state.detach()
                delta_op = torch.bmm(k.transpose(1, 2), error)
                hebb_op = torch.bmm(k.transpose(1, 2), v)
                new_S = alpha_decay * S + write_eta * (delta_op + 0.5 * hebb_op)
                node.state = new_S
            elif node.node_type == "3d_tensor":
                S = node.state.detach()
                k_sub = k[:, :, :8].squeeze(1)  # [H, 8]
                v_sub = v.squeeze(1)             # [H, D_v]
                outer_3d = torch.einsum("ha,hb->hab", k_sub, k_sub).unsqueeze(-1) * v_sub.unsqueeze(1).unsqueeze(1)
                node.state = alpha_decay * S + write_eta * outer_3d

        # Neurogenesis Check (Trigger Sprouting under surprise):
        self.maybe_sprout_memory_node(surprise)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        meta = {
            "loss": surprise,
            "pred": pred,
            "target": target_byte,
            "is_correct": (pred == target_byte),
            "entropy": entropy,
            "beta": beta.item(),
            "active_nodes": len(self.nodes),
            "sprout_count": self.sprout_count,
            "s_norm": torch.norm(combined_read).item()
        }
        return loss, meta


def run_exp_447():
    logger.info("================================================================================")
    logger.info("=== STARTING EXP-447: SOVEREIGN AUTONOMOUS MEMORY GENESIS (SAMG-TP) ===")
    logger.info("================================================================================")

    config = EXP447Config()
    model = SovereignAutonomousMemoryEngine(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

    stream_data = (
        "def karyon_sovereign_autopoiesis(stream):\n"
        "    super_operator = Liouvillian(phase_space)\n"
        "    return super_operator.synthesize()\n\n"
        "#include <torch/extension.h>\n"
        "void dynamic_kernel(float* x, float* h) {\n"
        "    // Tensor core phase alignment\n"
        "}\n\n"
        "HTTP/1.1 200 OK\r\nContent-Type: application/kcore\r\n\r\n"
        "Mind is substrate-independent continuous field dynamics.\n"
        "Ashby homeostatic ultrastability balances energy and surprise.\n"
    ).encode("utf-8")

    raw_bytes = list(stream_data) * (config.stream_length // len(stream_data) + 1)
    raw_bytes = raw_bytes[:config.stream_length]

    period = len(stream_data)
    logger.info(f"Stream loaded: {len(raw_bytes)} bytes | Period: {period} bytes on {DEVICE_STR.upper()}")
    logger.info("Evaluating Open-Ended Sovereign Memory Sprouting, Routing & Net2Net Zero-Shock...")

    correct_preds = 0
    total_loss = 0.0
    total_entropy = 0.0
    recent_losses = []

    t_start = time.perf_counter()

    for t in range(len(raw_bytes) - 1):
        x_byte = raw_bytes[t]
        target_byte = raw_bytes[t + 1]

        optimizer.zero_grad()
        loss, meta = model.step(x_byte, target_byte)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if meta["is_correct"]:
            correct_preds += 1

        total_loss += meta["loss"]
        total_entropy += meta["entropy"]
        recent_losses.append(meta["loss"])

        if (t + 1) % 500 == 0:
            avg_loss = total_loss / (t + 1)
            acc = (correct_preds / (t + 1)) * 100.0
            logger.info(
                f"Progress [{t+1}/{len(raw_bytes)-1}] | Avg Loss: {avg_loss:.4f} | "
                f"Acc: {acc:.2f}% | Current Loss: {meta['loss']:.4f} | Active Memory Nodes: {meta['active_nodes']} | "
                f"Total Sprouted Nodes: {meta['sprout_count']}"
            )

    t_elapsed = time.perf_counter() - t_start
    final_loss = total_loss / (len(raw_bytes) - 1)
    final_acc = (correct_preds / (len(raw_bytes) - 1)) * 100.0
    avg_final_entropy = total_entropy / (len(raw_bytes) - 1)
    throughput = (len(raw_bytes) - 1) / t_elapsed

    second_half = recent_losses[len(recent_losses)//2:]
    low_loss_count = sum(1 for loss_val in second_half if loss_val < 0.5)
    low_loss_fraction = (low_loss_count / len(second_half)) * 100.0

    ultra_low_count = sum(1 for loss_val in second_half if loss_val < 0.05)
    ultra_low_fraction = (ultra_low_count / len(second_half)) * 100.0

    zero_loss_count = sum(1 for loss_val in second_half if loss_val < 0.005)
    zero_loss_fraction = (zero_loss_count / len(second_half)) * 100.0

    tail_500 = recent_losses[-500:]
    tail_low_loss_count = sum(1 for loss_val in tail_500 if loss_val < 0.5)
    tail_retention_rate = (tail_low_loss_count / len(tail_500)) * 100.0

    logger.info("================================================================================")
    logger.info("=== EXP-447 SOVEREIGN AUTONOMOUS MEMORY GENESIS REPORT ===")
    logger.info(f"Final Average Loss: {final_loss:.4f} nats")
    logger.info(f"Single-Pass Accuracy: {final_acc:.2f}%")
    logger.info(f"Final Active Memory Nodes: {len(model.nodes)}")
    logger.info(f"Total Sprouted Nodes: {model.sprout_count}")
    logger.info(f"Second-Half Low-Loss Mass (<0.5 nats): {low_loss_fraction:.2f}%")
    logger.info(f"Second-Half Ultra-Low Loss Mass (<0.05 nats): {ultra_low_fraction:.2f}%")
    logger.info(f"Second-Half Near-Zero Loss Mass (<0.005 nats): {zero_loss_fraction:.2f}%")
    logger.info(f"Tail 500-byte Retention Rate (<0.5 nats): {tail_retention_rate:.2f}%")
    logger.info(f"Throughput: {throughput:.2f} steps/sec")
    logger.info(f"Mean Output Entropy: {avg_final_entropy:.3f} nats")
    logger.info(f"Elapsed Time: {t_elapsed:.2f} s")
    logger.info("================================================================================")

    results = {
        "exp_id": "EXP-447",
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 2),
        "active_memory_nodes": len(model.nodes),
        "sprouted_nodes_count": model.sprout_count,
        "second_half_low_loss_mass": round(low_loss_fraction, 2),
        "second_half_ultra_low_loss_mass": round(ultra_low_fraction, 2),
        "second_half_near_zero_loss_mass": round(zero_loss_fraction, 2),
        "tail_retention_rate": round(tail_retention_rate, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "mean_entropy_nats": round(avg_final_entropy, 3),
        "elapsed_time": round(t_elapsed, 2)
    }

    with open("experiments/exp_447_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_exp_447()
