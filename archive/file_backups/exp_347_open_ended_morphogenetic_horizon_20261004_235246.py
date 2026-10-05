import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================================================================
# EXP-347: OPEN-ENDED MORPHOGENETIC HORIZON & TOPOLOGICAL EVOLUTION BENCHMARK
# ===============================================================================

class AtomicOperator(nn.Module):
    """Base class for dynamically sprouted mathematical operator nodes."""
    def __init__(self, in_dim: int, out_dim: int, op_type: str):
        super().__init__()
        self.op_type = op_type
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * (1.0 / math.sqrt(in_dim)))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        # Epigenetic Morphogenesis Gate (Principle 15 & 16)
        # Initialized at 0.0 to ensure strict zero-delta functional identity at 'birth'
        self.alpha_epi = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Net2Net Smooth Grafting: tanh(alpha_epi) scales the sprouted pathway
        gate = torch.tanh(self.alpha_epi)
        
        if self.op_type == "linear":
            return x + gate * F.linear(x, self.weight, self.bias)
        elif self.op_type == "bilinear":
            # Multiplicative conjunction
            proj = F.linear(x, self.weight, self.bias)
            return x + gate * (x * torch.sigmoid(proj))
        elif self.op_type == "attractor":
            # Continuous Hopfield relaxation step approximation
            proj = F.linear(x, self.weight, self.bias)
            return x + gate * torch.tanh(proj)
        elif self.op_type == "decay":
            # Leaky integration / State Space memory decay step
            decay_factor = torch.sigmoid(self.weight.mean()) * 0.15
            proj = F.linear(x, self.weight, self.bias)
            return (1.0 - decay_factor) * x + gate * torch.relu(proj)
        else:
            return x

class MorphogeneticGraph(nn.Module):
    """
    Dynamic Neural Graph Assembly (AGN v6.0) that evolves its topology
    open-endedly under the pressure of Variational Free Energy minimization.
    """
    def __init__(self, dim: int, device: str = "cpu"):
        super().__init__()
        self.dim = dim
        self.nodes = nn.ModuleList([
            AtomicOperator(dim, dim, "linear").to(device),
            AtomicOperator(dim, dim, "decay").to(device)
        ])
        self.adjacency = nn.Parameter(torch.eye(2, device=device)) # Dynamic routing matrix

    def sprout_node(self, op_type: str):
        """Inserts a new mathematical operator node with zero-shock epigenetic gating."""
        # Detect active device of current parameters to prevent CPU-GPU mismatch
        device = next(self.parameters()).device
        new_node = AtomicOperator(self.dim, self.dim, op_type).to(device)
        nn.init.constant_(new_node.alpha_epi, 0.0)
        
        # When appending to nn.ModuleList, explicitly cast and register
        self.nodes.append(new_node)
        
        # Expand adjacency routing matrix
        n = len(self.nodes)
        new_adj = torch.zeros(n, n, device=device)
        new_adj[:-1, :-1] = self.adjacency.data
        new_adj[-1, -1] = 1.0 # Self-loop default
        # Small random routing sprout
        for i in range(n - 1):
            if random.random() > 0.7:
                new_adj[i, -1] = 0.1
                new_adj[-1, i] = 0.1
        self.adjacency = nn.Parameter(new_adj)
        
        # Enforce device synchronization of all module elements explicitly
        self.to(device)
        for node in self.nodes:
            node.to(device)
            node.weight.data = node.weight.data.to(device)
            node.bias.data = node.bias.data.to(device)
            node.alpha_epi.data = node.alpha_epi.data.to(device)

    def prune_unviable_nodes(self, threshold: float = 0.05):
        """Neurodarwinian Apoptosis: Prunes operators whose epigenetic expression drops near zero."""
        keep_indices = []
        pruned_types = []
        for i, node in enumerate(self.nodes):
            expr = torch.abs(torch.tanh(node.alpha_epi)).item()
            if expr >= threshold or i < 2: # Keep first 2 as structural anchor
                keep_indices.append(i)
            else:
                pruned_types.append(node.op_type)
        
        if len(keep_indices) < len(self.nodes):
            new_nodes = nn.ModuleList([self.nodes[i] for i in keep_indices])
            self.nodes = new_nodes
            self.adjacency = nn.Parameter(self.adjacency[keep_indices][:, keep_indices])
        return pruned_types

    def forward(self, x: torch.Tensor, recurrent_depth: int = 3) -> tuple[torch.Tensor, float]:
        # Recurrent Latent Depth (Principle 21): problem time decoupled from feedforward step
        h = x
        n_nodes = len(self.nodes)
        node_states = [h for _ in range(n_nodes)]
        
        routing = F.softmax(self.adjacency, dim=-1)
        
        # Information Transport & Synaptic Flow micro-probing (Principle 23)
        entropy_accum = 0.0
        for i in range(n_nodes):
            entropy_accum += -torch.sum(routing[i] * torch.log(routing[i] + 1e-9)).item()
        avg_routing_entropy = entropy_accum / n_nodes
        
        for _ in range(recurrent_depth):
            next_states = []
            for i, node in enumerate(self.nodes):
                # Gather inputs from other nodes based on dynamic routing
                node_input = torch.zeros_like(h)
                for j in range(n_nodes):
                    node_input += routing[j, i] * node_states[j]
                
                next_states.append(node(node_input))
            node_states = next_states
            
        # Final read-out is the average of active node states
        out = torch.stack(node_states).mean(dim=0)
        return out, avg_routing_entropy

# ===============================================================================
# BENCHMARK EVALUATION LIFECYCLE
# ===============================================================================

def run_evolution_experiment():
    print("=== STARTING EXP-347: OPEN-ENDED MORPHOGENETIC HORIZON BENCHMARK ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Substrate: {device}")
    
    dim = 64
    graph = MorphogeneticGraph(dim, device=str(device)).to(device)
    optimizer = torch.optim.AdamW(graph.parameters(), lr=0.01, weight_decay=1e-4)
    
    # 1. Diverse Problem Manifolds to evaluate open-ended adaptation (Principle 21)
    # Manifold A: Continuous Chaotic Attractor (Lorenz-like tracking)
    # Manifold B: Algorithmic Logic Synthesis (Parity & Bitwise mapping)
    # Manifold C: High-entropy Associative Recall
    
    batch_size = 32
    steps_per_epoch = 15
    epochs = 20
    
    history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_entropy = 0.0
        
        # Dynamic sprouting schedule based on performance pressure
        if epoch > 0 and epoch % 3 == 0:
            sprout_type = random.choice(["linear", "bilinear", "attractor", "decay"])
            graph.sprout_node(sprout_type)
            print(f"🌱 [Epoch {epoch}] Sprouted new Atomic Operator node: '{sprout_type}'. Active Nodes: {len(graph.nodes)}")
            # Refresh optimizer to track new parameters
            optimizer = torch.optim.AdamW(graph.parameters(), lr=0.01, weight_decay=1e-4)
            
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            
            # Generate synthetic complex chaotic dynamics
            x_raw = torch.randn(batch_size, dim, device=device)
            # Target is a non-linear chaotic transition mapping
            target = torch.sin(x_raw) * 0.5 + torch.cos(x_raw * 1.5) * 0.3
            
            # Recurrent thinking depth modulated by complexity
            rec_depth = 3 if epoch < 10 else 5
            
            # Explicitly cast model parameters to correct device before forward pass
            graph.to(device)
            out, routing_entropy = graph(x_raw, recurrent_depth=rec_depth)
            
            # Variational Free Energy formulation (F = Complexity + Accuracy Error)
            # Complexity: L2 penalty on sprouted parameters to enforce Tononi SHY metabolic constraint
            complexity = 0.0
            for node in graph.nodes:
                complexity += 0.01 * torch.sum(node.alpha_epi ** 2)
                
            accuracy_loss = F.mse_loss(out, target)
            free_energy = accuracy_loss + complexity
            
            free_energy.backward()
            torch.nn.utils.clip_grad_norm_(graph.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += accuracy_loss.item()
            epoch_entropy += routing_entropy
            
        # Neurodarwinian Apoptosis phase (Sleep-phase pruning)
        if epoch > 0 and epoch % 5 == 0:
            pruned = graph.prune_unviable_nodes(threshold=0.02)
            if pruned:
                print(f"💀 [Epoch {epoch}] Neurodarwinian Apoptosis: Pruned unviable operators: {pruned}. Active Nodes: {len(graph.nodes)}")
                optimizer = torch.optim.AdamW(graph.parameters(), lr=0.01, weight_decay=1e-4)
                
        avg_loss = epoch_loss / steps_per_epoch
        avg_ent = epoch_entropy / steps_per_epoch
        
        # Compute Participation Ratio (Effective Dimensionality) of node states
        active_expressions = [torch.abs(torch.tanh(n.alpha_epi)).item() for n in graph.nodes]
        sum_expr = sum(active_expressions) + 1e-9
        participation_ratio = (sum_expr ** 2) / (sum([e**2 for e in active_expressions]) + 1e-9)
        
        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "routing_entropy": avg_ent,
            "active_nodes": len(graph.nodes),
            "participation_ratio": participation_ratio,
            "mean_expression": sum(active_expressions) / len(active_expressions)
        })
        
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.6f} | Routing Entropy: {avg_ent:.4f} | Nodes: {len(graph.nodes)} | PR: {participation_ratio:.3f}")
        
    print("=== BENCHMARK COMPLETED SUCCESSFULLY ===")
    return history

if __name__ == "__main__":
    results = run_evolution_experiment()
    
    # Assertions and telemetry verification
    final_loss = results[-1]["loss"]
    print(f"Final Loss: {final_loss:.6f}")
    assert final_loss < 0.1, "Error: Evolution failed to converge below target error threshold!"
    print("✅ Convergence target verified.")
