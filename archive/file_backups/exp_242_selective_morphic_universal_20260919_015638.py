"""
===============================================================================
EXP-242: Universal Meta-Plastic Morphic Operator with Selective Associative SSD
Grounding: KEP Principle 1 (C++ & Parallelism as Engine),
           KEP Principle 2 (Universal Biophysical Substrate & Autonomous Morphogenesis),
           KEP Principle 12 (Universal Modality-Agnostic Substrate),
           KEP Principle 14 (Axiom of Allostatic Dynamic Forces — No Static Constants),
           KEP Principle 16 (Dynamic Neural Graph Assembly - AGN v6.0),
           KEP Rule #1 (Hypothesis & Telemetry First),
           KEP Rule #1.1 (Mandatory Debugging to Completion Principle),
           KEP Rule #2 (Contextual Multi-Criteria Decision Engine),
           KEP Rule #11 (Strict Code Quality & Linter Compliance).
===============================================================================
Hypothesis:
Equipping the Universal Morphic Operator with:
  1. Causal Token-Wise Meta-Controllers (evaluating x_t causally per step rather than static global means).
  2. Selective Associative Dynamic State-Space (Morphic SSD with input-dependent dynamic gating Delta_t and content-addressable key-value memory).
will enable Karyon to master both discrete algorithmic logic, continuous physical harmonics, and long-range associative indirection retrieval, achieving near-zero loss across all 3 domains.
===============================================================================
"""

import os
import sys
import math
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("exp_242")


# ============================================================================
# 1. SELECTIVE ASSOCIATIVE UNIVERSAL MORPHIC OPERATOR
# ============================================================================

class SelectiveMorphicOperator(nn.Module):
    """
    Advanced Universal Meta-Plastic Morphic Operator with Selective Associative Memory.
    Features:
      - Causal Step-Wise Dynamic Gating.
      - Selective Content-Addressable Memory Write/Read (Delta_t, Alpha_t).
      - Dynamic Composite Activation Space (Identity, GELU, SiLU, Tanh, Sine PAC, Abs Threshold).
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Projections for Selective Associative Time-Mixing
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.delta_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Dynamic Activation Basis Controller
        self.meta_act = nn.Linear(dim, 6)

        # Decay logit
        self.alpha_raw = nn.Parameter(torch.ones(dim) * 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.size()

        # 1. Dynamic Activation Basis Modulation
        act_logits = self.meta_act(x)  # [batch, seq_len, 6]
        act_coeffs = F.softmax(act_logits, dim=-1).unsqueeze(-1)  # [batch, seq_len, 6, 1]

        psi_identity = x.unsqueeze(-2)
        psi_gelu = F.gelu(x).unsqueeze(-2)
        psi_silu = F.silu(x).unsqueeze(-2)
        psi_tanh = torch.tanh(x).unsqueeze(-2)
        psi_sin = torch.sin(x * 2.0).unsqueeze(-2)
        psi_abs = torch.abs(x).unsqueeze(-2)

        psi_stack = torch.cat([psi_identity, psi_gelu, psi_silu, psi_tanh, psi_sin, psi_abs], dim=-2)
        x_morphic = torch.sum(psi_stack * act_coeffs, dim=-2)  # [batch, seq_len, dim]

        # 2. Selective Associative Dynamic State Space
        q = self.q_proj(x_morphic).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x_morphic).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x_morphic).view(batch, seq_len, self.num_heads, self.head_dim)

        delta = F.softplus(self.delta_proj(x_morphic)).view(batch, seq_len, self.num_heads, self.head_dim)
        alpha = torch.sigmoid(self.alpha_raw).view(1, 1, self.num_heads, self.head_dim)

        # Causal Recurrent / SSD Memory Scan
        outputs = []
        state = torch.zeros(batch, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            q_t = q[:, t]          # [B, H, D]
            k_t = k[:, t]          # [B, H, D]
            v_t = v[:, t]          # [B, H, D]
            d_t = delta[:, t]      # [B, H, D]

            # Outer product update: state = alpha * state + (d_t * k_t)^T (d_t * v_t)
            kv = torch.einsum("bhd,bhe->bhde", k_t * d_t, v_t * d_t)
            state = alpha.unsqueeze(-1) * state + kv

            # Query readout
            y_t = torch.einsum("bhd,bhde->bhe", q_t, state)  # [B, H, D]
            outputs.append(y_t.unsqueeze(1))

        y = torch.cat(outputs, dim=1).view(batch, seq_len, dim)
        return self.out_proj(y) + x_morphic


class DynamicMorphicGraphEngine(nn.Module):
    def __init__(self, dim: int, num_nodes: int = 4):
        super().__init__()
        self.dim = dim
        self.num_nodes = num_nodes + 2
        self.num_internal = num_nodes

        self.node_names = ["INPUT_SOURCE"]
        self.nodes = nn.ModuleList([SelectiveMorphicOperator(dim=dim, num_heads=4) for _ in range(num_nodes)])
        for i in range(num_nodes):
            self.node_names.append(f"selective_morphic_{i}")
        self.node_names.append("OUTPUT_SINK")

        self.ff_edge_logits = nn.Parameter(torch.ones(self.num_nodes, self.num_nodes) * -1.5)
        self.fb_edge_logits = nn.Parameter(torch.ones(self.num_nodes, self.num_nodes) * -2.5)

        self.node_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.num_nodes)])
        self.fb_norm = nn.LayerNorm(dim)

    def get_ff_adjacency(self) -> torch.Tensor:
        triu_mask = torch.triu(torch.ones(self.num_nodes, self.num_nodes, device=self.ff_edge_logits.device), diagonal=1)
        return torch.sigmoid(self.ff_edge_logits) * triu_mask

    def get_fb_adjacency(self) -> torch.Tensor:
        tril_mask = torch.tril(torch.ones(self.num_nodes, self.num_nodes, device=self.fb_edge_logits.device), diagonal=-1)
        return torch.sigmoid(self.fb_edge_logits) * tril_mask

    def forward(self, sensory_input: torch.Tensor) -> torch.Tensor:
        ff_adj = self.get_ff_adjacency()
        fb_adj = self.get_fb_adjacency()

        # Step 1: Feedforward pass
        node_states = [torch.zeros_like(sensory_input) for _ in range(self.num_nodes)]
        node_states[0] = sensory_input

        for j in range(1, self.num_nodes - 1):
            ff_incoming = torch.zeros_like(sensory_input)
            for i in range(j):
                ff_incoming = ff_incoming + ff_adj[i, j] * node_states[i]

            norm_ff = self.node_norms[j](ff_incoming)
            op = self.nodes[j - 1]
            node_states[j] = ff_incoming + op(norm_ff)

        sink_idx = self.num_nodes - 1
        init_sink = torch.zeros_like(sensory_input)
        for i in range(sink_idx):
            init_sink = init_sink + ff_adj[i, sink_idx] * node_states[i]
        node_states[sink_idx] = init_sink

        # Step 2: Vectorized Top-Down Recurrent Feedback Pass (Shifted h_{t-1})
        prev_node_states = [torch.cat([torch.zeros_like(s[:, :1, :]), s[:, :-1, :]], dim=1) for s in node_states]

        for j in range(1, self.num_nodes - 1):
            ff_incoming = torch.zeros_like(sensory_input)
            for i in range(j):
                ff_incoming = ff_incoming + ff_adj[i, j] * node_states[i]

            fb_incoming = torch.zeros_like(sensory_input)
            for k in range(j + 1, self.num_nodes):
                fb_incoming = fb_incoming + fb_adj[k, j] * prev_node_states[k]

            combined = ff_incoming + self.fb_norm(fb_incoming)
            norm_combined = self.node_norms[j](combined)
            op = self.nodes[j - 1]
            node_states[j] = combined + op(norm_combined)

        output_sink = torch.zeros_like(sensory_input)
        for i in range(sink_idx):
            output_sink = output_sink + ff_adj[i, sink_idx] * node_states[i]

        return self.node_norms[sink_idx](output_sink)


class UniversalMorphicAgent(nn.Module):
    def __init__(self, vocab_size: int = 258, dim: int = 128, num_nodes: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.graph = DynamicMorphicGraphEngine(dim=dim, num_nodes=num_nodes)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h_graph = self.graph(h)
        return self.head(h_graph)


# ============================================================================
# 2. DIVERSE TASK DATASETS
# ============================================================================

class Task1_AlgorithmicDyckStack:
    """Domain 1: Nested Bracket Stack Parsing (Dyck-4 Language)."""
    def __init__(self, vocab_size: int = 258, seq_len: int = 24):
        self.seq_len = seq_len
        self.pad = 256
        self.eos = 257
        self.bracket_pairs = [(60, 61), (40, 41), (91, 93), (123, 125)]

    def generate_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        batch = torch.full((batch_size, self.seq_len), self.pad, dtype=torch.long, device=device)
        for i in range(batch_size):
            stack = []
            ptr = 0
            while ptr < self.seq_len // 2 - 2:
                pair = random.choice(self.bracket_pairs)
                batch[i, ptr] = pair[0]
                stack.append(pair[1])
                ptr += 1
            batch[i, ptr] = 35  # '#'
            batch[i, ptr + 1] = 35
            ptr += 2
            while stack and ptr < self.seq_len - 1:
                batch[i, ptr] = stack.pop()
                ptr += 1
            batch[i, ptr] = self.eos
        return batch


class Task2_PeriodicHarmonics:
    """Domain 2: Deterministic Fourier Harmonics & Wave Periodicity."""
    def __init__(self, vocab_size: int = 258, seq_len: int = 24):
        self.seq_len = seq_len
        self.pad = 256
        self.eos = 257

    def generate_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        batch = torch.full((batch_size, self.seq_len), self.pad, dtype=torch.long, device=device)
        for i in range(batch_size):
            period = random.randint(4, 7)
            pattern = [random.randint(30, 200) for _ in range(period)]
            for t in range(self.seq_len - 1):
                batch[i, t] = pattern[t % period]
            batch[i, self.seq_len - 1] = self.eos
        return batch


class Task3_AssociativeIndirectionRetrieval:
    """Domain 3: Associative Indirection & Selective Memory Retrieval."""
    def __init__(self, vocab_size: int = 258, seq_len: int = 24):
        self.seq_len = seq_len
        self.pad = 256
        self.eos = 257

    def generate_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        batch = torch.full((batch_size, self.seq_len), self.pad, dtype=torch.long, device=device)
        for i in range(batch_size):
            key = random.randint(10, 30)
            val = random.randint(150, 220)
            batch[i, 0] = key
            batch[i, 1] = val
            for p in range(2, self.seq_len - 3):
                batch[i, p] = random.randint(50, 90)
            batch[i, self.seq_len - 3] = key
            batch[i, self.seq_len - 2] = val
            batch[i, self.seq_len - 1] = self.eos
        return batch


# ============================================================================
# 3. MULTI-DOMAIN EVALUATION PIPELINE
# ============================================================================

def train_and_evaluate_domain(domain_name: str, dataset_generator, device: torch.device) -> dict:
    logger.info(f"\n🚀 === TESTING DOMAIN: {domain_name} ===")
    agent = UniversalMorphicAgent(vocab_size=258, dim=128, num_nodes=4).to(device)

    edge_params = [agent.graph.ff_edge_logits, agent.graph.fb_edge_logits]
    core_params = [p for n, p in agent.named_parameters() if "edge_logits" not in n]

    optimizer = torch.optim.AdamW([
        {"params": core_params, "lr": 0.005},
        {"params": edge_params, "lr": 0.02}
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0005)

    losses = []
    for epoch in range(20):
        agent.train()
        epoch_loss = 0.0
        num_batches = 40

        for _ in range(num_batches):
            batch = dataset_generator.generate_batch(batch_size=32, device=device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            logits = agent(inputs)

            loss = F.cross_entropy(logits.reshape(-1, 258), targets.reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"[{domain_name}] Epoch {epoch + 1:02d}/20 | Loss: {avg_loss:.4f} nats/byte")

    # Reverse-engineer synthesized activation laws
    sample_batch = dataset_generator.generate_batch(batch_size=32, device=device)
    sample_embed = agent.embed(sample_batch[:, :-1])

    basis_names = ["Identity", "GELU", "SiLU", "Tanh", "Sine (PAC)", "Abs (Threshold)"]
    laws_summary = {}

    with torch.no_grad():
        for name, op in zip(agent.graph.node_names[1:-1], agent.graph.nodes):
            act_coeffs = F.softmax(op.meta_act(sample_embed), dim=-1).mean(dim=[0, 1]).cpu().tolist()
            dominant_idx = act_coeffs.index(max(act_coeffs))
            laws_summary[name] = {
                "dominant_law": basis_names[dominant_idx],
                "confidence": round(act_coeffs[dominant_idx] * 100, 1),
                "distribution": dict(zip(basis_names, [round(c, 3) for c in act_coeffs]))
            }

    best_loss = min(losses)
    final_loss = losses[-1]
    logger.info(f"[{domain_name}] Finished: Best Loss = {best_loss:.4f}, Final Loss = {final_loss:.4f}")
    return {
        "domain": domain_name,
        "best_loss": best_loss,
        "final_loss": final_loss,
        "laws_summary": laws_summary
    }


def run_experiment():
    logger.info("=== Starting EXP-242: Universal Morphic Operator Cross-Domain Benchmark ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Backend: {device}")

    task1 = Task1_AlgorithmicDyckStack(vocab_size=258, seq_len=24)
    task2 = Task2_PeriodicHarmonics(vocab_size=258, seq_len=24)
    task3 = Task3_AssociativeIndirectionRetrieval(vocab_size=258, seq_len=24)

    results = []
    results.append(train_and_evaluate_domain("Task 1: Algorithmic Logic (Dyck Stack)", task1, device))
    results.append(train_and_evaluate_domain("Task 2: Periodic Wave Harmonics", task2, device))
    results.append(train_and_evaluate_domain("Task 3: Associative Indirection Retrieval", task3, device))

    logger.info("\n================================================================================")
    logger.info("=== MULTI-DOMAIN SYNTHESIZED ACTIVATION LAWS & PERFORMANCE SUMMARY ===")
    logger.info("================================================================================")
    all_losses = []
    for res in results:
        logger.info(f"\n📊 Domain: {res['domain']}")
        logger.info(f"   Final Loss: {res['final_loss']:.4f} nats/byte | Best: {res['best_loss']:.4f}")
        all_losses.append(res['best_loss'])
        for node_name, info in res['laws_summary'].items():
            logger.info(f"   Node [{node_name}]: {info['dominant_law']} ({info['confidence']}%)")

    mean_loss = sum(all_losses) / len(all_losses)
    logger.info(f"\n⭐ Cross-Domain Mean Best Loss: {mean_loss:.4f} nats/byte")

    assert mean_loss < 0.20, f"Universal Morphic DAG failed cross-domain benchmark! Mean Loss: {mean_loss:.4f}"

    print("\n--- EXP-242 VERIFIED: UNIVERSAL MORPHIC OPERATOR TRANSCENDS ALL DOMAINS ---\n")
    print("EXP_ID=EXP-242")
    print("VERDICT=POSITIVE")
    print(f"FINAL_LOSS={mean_loss:.4f}")


if __name__ == "__main__":
    run_experiment()
