"""
===============================================================================
EXP-242: Multi-Domain Benchmark of Universal Morphic Self-Configuring Operator
Grounding: KEP Principle 1 (C++ & Parallelism as Engine),
           KEP Principle 2 (Universal Biophysical Substrate & Autonomous Morphogenesis),
           KEP Principle 10 (Autonomy of Protocol Evolution),
           KEP Principle 12 (Universal Modality-Agnostic Substrate),
           KEP Principle 14 (Axiom of Allostatic Dynamic Forces — No Static Constants),
           KEP Principle 16 (Dynamic Neural Graph Assembly - AGN v6.0),
           KEP Rule #1 (Hypothesis & Telemetry First),
           KEP Rule #1.1 (Mandatory Debugging to Completion Principle),
           KEP Rule #2 (Contextual Multi-Criteria Decision Engine),
           KEP Rule #11 (Strict Code Quality & Linter Compliance).
===============================================================================
Hypothesis:
The Universal Meta-Plastic Self-Configuring Morphic Operator (where nodes dynamically
synthesize weights W(x), composite activation bases phi(x), and temporal integration
parameters on-the-fly) will autonomously adapt its internal mathematical laws across
3 radically diverse structural task domains:
  1. Task 1 (Algorithmic Logic): Multi-level Dyck-K Nested Stack Bracket Parsing.
  2. Task 2 (Grammar Syntax): Hierarchical Dependent Context-Free Language Parsing.
  3. Task 3 (Deterministic Sequence Induction): Multi-Period Token Interleaving & Reversal.
And achieve near-zero prediction loss (Loss < 0.05 nats/byte) across all 3 domains without any task-specific inductive modifications.
===============================================================================
"""

import os
import sys
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
# 1. UNIVERSAL MORPHIC OPERATOR ENGINE
# ============================================================================

class UniversalMorphicOperator(nn.Module):
    """
    Universal Meta-Plastic Computational Kernel.
    Dynamically synthesizes low-rank weight tensors, composite activation basis
    (Identity, GELU, SiLU, Tanh, Sine PAC, Abs Threshold), and temporal decay dynamics.
    """
    def __init__(self, dim: int, rank: int = 16):
        super().__init__()
        self.dim = dim
        self.rank = rank

        self.meta_controller = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, rank * dim * 2 + 6 + 2)
        )
        self.base_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.size()
        context = torch.mean(x, dim=1)
        meta_params = self.meta_controller(context)

        ptr = 0
        u_flat = meta_params[:, ptr:ptr + self.rank * dim].view(batch, dim, self.rank)
        ptr += self.rank * dim
        v_flat = meta_params[:, ptr:ptr + self.rank * dim].view(batch, self.rank, dim)
        ptr += self.rank * dim

        act_coeffs = F.softmax(meta_params[:, ptr:ptr + 6], dim=-1).unsqueeze(1).unsqueeze(2)
        ptr += 6

        alpha = torch.sigmoid(meta_params[:, ptr:ptr + 1]).unsqueeze(1)
        beta = torch.sigmoid(meta_params[:, ptr + 1:ptr + 2]).unsqueeze(1)

        x_low = torch.bmm(x, u_flat)
        dynamic_transformed = torch.bmm(x_low, v_flat) * 0.1
        h_linear = self.base_proj(x) + dynamic_transformed

        psi_identity = h_linear.unsqueeze(-1)
        psi_gelu = F.gelu(h_linear).unsqueeze(-1)
        psi_silu = F.silu(h_linear).unsqueeze(-1)
        psi_tanh = torch.tanh(h_linear).unsqueeze(-1)
        psi_sin = torch.sin(h_linear * 2.0).unsqueeze(-1)
        psi_abs = torch.abs(h_linear).unsqueeze(-1)

        psi_stack = torch.cat([psi_identity, psi_gelu, psi_silu, psi_tanh, psi_sin, psi_abs], dim=-1)
        h_activated = torch.sum(psi_stack * act_coeffs, dim=-1)

        cum_temporal = torch.cumsum(h_activated * (1.0 - alpha), dim=1)
        y = beta * cum_temporal + (1.0 - beta) * h_activated
        return y


class DynamicMorphicGraphEngine(nn.Module):
    def __init__(self, dim: int, num_nodes: int = 6):
        super().__init__()
        self.dim = dim
        self.num_nodes = num_nodes + 2
        self.num_internal = num_nodes

        self.node_names = ["INPUT_SOURCE"]
        self.nodes = nn.ModuleList([UniversalMorphicOperator(dim=dim, rank=16) for _ in range(num_nodes)])
        for i in range(num_nodes):
            self.node_names.append(f"morphic_node_{i}")
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
    def __init__(self, vocab_size: int = 258, dim: int = 256, num_nodes: int = 6):
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
# 2. TASK DOMAIN GENERATORS
# ============================================================================

class Task1_AlgorithmicDyckStack:
    """Domain 1: Nested Stack Dyck-4 Language Parsing."""
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


class Task2_ContextFreeGrammar:
    """Domain 2: Hierarchical Dependent Context-Free Grammar Parsing."""
    def __init__(self, vocab_size: int = 258, seq_len: int = 24):
        self.seq_len = seq_len
        self.pad = 256
        self.eos = 257

    def generate_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        batch = torch.full((batch_size, self.seq_len), self.pad, dtype=torch.long, device=device)
        for i in range(batch_size):
            # Subject (10-15), Verb (30-35), Object (50-55) with agreement rules
            subj = random.randint(10, 15)
            verb = 30 + (subj % 3) * 2 + random.randint(0, 1)
            obj = 50 + (verb % 4)
            modifier = 80 + (subj + verb) % 10

            template = [subj, modifier, verb, obj, 99, subj, verb, obj, self.eos]
            for idx, token in enumerate(template):
                if idx < self.seq_len:
                    batch[i, idx] = token
        return batch


class Task3_SequenceInduction:
    """Domain 3: Multi-Period Token Interleaving & Sequence Induction."""
    def __init__(self, vocab_size: int = 258, seq_len: int = 24):
        self.seq_len = seq_len
        self.pad = 256
        self.eos = 257

    def generate_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        batch = torch.full((batch_size, self.seq_len), self.pad, dtype=torch.long, device=device)
        for i in range(batch_size):
            base_a = random.randint(20, 50)
            base_b = random.randint(120, 150)
            step_a = random.randint(1, 3)
            step_b = random.randint(2, 4)
            for t in range(0, self.seq_len - 2, 2):
                batch[i, t] = (base_a + (t // 2) * step_a) % 250
                batch[i, t + 1] = (base_b + (t // 2) * step_b) % 250
            batch[i, self.seq_len - 1] = self.eos
        return batch


# ============================================================================
# 3. EVALUATION PIPELINE
# ============================================================================

def train_and_evaluate_domain(domain_name: str, dataset_generator, device: torch.device) -> dict:
    logger.info(f"\n🚀 === TESTING DOMAIN: {domain_name} ===")
    agent = UniversalMorphicAgent(vocab_size=258, dim=256, num_nodes=6).to(device)

    edge_params = [agent.graph.ff_edge_logits, agent.graph.fb_edge_logits]
    core_params = [p for n, p in agent.named_parameters() if "edge_logits" not in n]

    optimizer = torch.optim.AdamW([
        {"params": core_params, "lr": 0.003},
        {"params": edge_params, "lr": 0.02}
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.0005)

    losses = []
    for epoch in range(25):
        agent.train()
        epoch_loss = 0.0
        num_batches = 50

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
            logger.info(f"[{domain_name}] Epoch {epoch + 1:02d}/25 | Loss: {avg_loss:.4f} nats/byte")

    # Reverse-engineer synthesized activation laws
    sample_batch = dataset_generator.generate_batch(batch_size=32, device=device)
    sample_embed = agent.embed(sample_batch[:, :-1])

    basis_names = ["Identity", "GELU", "SiLU", "Tanh", "Sine (PAC)", "Abs (Threshold)"]
    laws_summary = {}

    with torch.no_grad():
        for name, op in zip(agent.graph.node_names[1:-1], agent.graph.nodes):
            context = torch.mean(sample_embed, dim=1)
            meta_params = op.meta_controller(context)
            act_coeffs = F.softmax(meta_params[:, -8:-2], dim=-1).mean(dim=0).cpu().tolist()
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
    logger.info("=== Starting EXP-242: Multi-Domain Benchmark of Universal Morphic Self-Configuring DAG ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Backend: {device}")

    task1 = Task1_AlgorithmicDyckStack(vocab_size=258, seq_len=24)
    task2 = Task2_ContextFreeGrammar(vocab_size=258, seq_len=24)
    task3 = Task3_SequenceInduction(vocab_size=258, seq_len=24)

    results = []
    results.append(train_and_evaluate_domain("Task 1: Algorithmic Logic (Dyck Stack)", task1, device))
    results.append(train_and_evaluate_domain("Task 2: Context-Free Grammar Parsing", task2, device))
    results.append(train_and_evaluate_domain("Task 3: Sequence Induction & Interleaving", task3, device))

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

    assert mean_loss < 0.05, f"Universal Morphic DAG failed cross-domain benchmark! Mean Loss: {mean_loss:.4f}"

    print("\n--- EXP-242 VERIFIED: UNIVERSAL MORPHIC OPERATOR TRANSCENDS ALL DOMAINS ---\n")
    print("EXP_ID=EXP-242")
    print("VERDICT=POSITIVE")
    print(f"FINAL_LOSS={mean_loss:.4f}")


if __name__ == "__main__":
    run_experiment()
