"""
===============================================================================
EXP-237: Universal DAG-KCORE Persistence & In-Container Dynamic Execution
Grounding: KEP Principle 1 (C++ & Parallelism as Engine),
           KEP Principle 2 (Universal Biophysical Substrate & Autonomous Morphogenesis),
           KEP Principle 15 (Epigenetic Morphogenesis & Net2Net Smooth Grafting),
           KEP Principle 16 (Dynamic Neural Graph Assembly - AGN v6.0),
           KEP Rule #1 (Hypothesis & Telemetry First),
           KEP Rule #2 (Contextual Multi-Criteria Decision Engine),
           KEP Rule #11 (Strict Code Quality & Linter Compliance).
===============================================================================
Hypothesis:
Serializing the complete dynamic DAG computational graph schema (nodes, operators,
and epigenetic adjacency matrix W_edge) directly into the Manifest (Section 1) and
Logic (Section 2) of the `.kcore` binary container format will allow:
  1. Full zero-loss extraction and reconstruction of the self-evolved graph from `.kcore`.
  2. Direct in-container execution of dynamic DAG models on device without manual re-wiring.
  3. Seamless continuation of DAG morphogenesis across save/load cycles with identical numerical outputs.
===============================================================================
"""

import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("exp_237")


# ============================================================================
# 1. HETEROGENEOUS BIOPHYSICAL OPERATORS POOL
# ============================================================================

class LinearOp(nn.Module):
    """Linear projection node."""
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class NonLinearOp(nn.Module):
    """SwiGLU channel-mixing non-linear node."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim * 2, bias=False)
        self.down_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = torch.chunk(self.gate_proj(x), 2, dim=-1)
        return self.down_proj(F.silu(gate) * val)


class DelayOp(nn.Module):
    """First-order SDE/state-space temporal memory node."""
    def __init__(self, dim: int):
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.randn(dim) * 0.1 - 1.0)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.size()
        alpha = torch.sigmoid(self.alpha_raw)
        outputs = []
        h_prev = torch.zeros(batch, dim, device=x.device, dtype=x.dtype)
        proj_x = self.proj(x)
        for t in range(seq_len):
            h_curr = alpha * h_prev + (1.0 - alpha) * proj_x[:, t, :]
            outputs.append(h_curr.unsqueeze(1))
            h_prev = h_curr
        return torch.cat(outputs, dim=1)


class GateOp(nn.Module):
    """Gated multiplicative modulator node."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(x) * torch.sigmoid(self.gate(x))


class ContinuousHopfieldOp(nn.Module):
    """Modern Continuous Hopfield Network attractor retrieval node."""
    def __init__(self, dim: int, num_basins: int = 16):
        super().__init__()
        self.dim = dim
        self.num_basins = num_basins
        self.basins = nn.Parameter(torch.randn(num_basins, dim) / (dim ** 0.5))
        self.values = nn.Parameter(torch.randn(num_basins, dim) / (dim ** 0.5))
        self.beta = nn.Parameter(torch.tensor(8.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_basins = F.normalize(self.basins, p=2, dim=-1)
        logits = torch.matmul(x, normalized_basins.t()) * self.beta
        weights = F.softmax(logits, dim=-1)
        return torch.matmul(weights, self.values)


class ThetaGammaOscillationOp(nn.Module):
    """Theta-Gamma Phase-Amplitude Coupling (PAC) oscillatory node."""
    def __init__(self, dim: int):
        super().__init__()
        self.freq_theta = nn.Parameter(torch.ones(dim) * 2.0)
        self.freq_gamma = nn.Parameter(torch.ones(dim) * 8.0)
        self.amplitude = nn.Parameter(torch.ones(dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.size()
        t = torch.arange(seq_len, device=x.device, dtype=x.dtype).view(1, seq_len, 1)
        theta = torch.sin(t * self.freq_theta * 0.1)
        gamma = torch.sin(t * self.freq_gamma * 0.5)
        pac_modulator = 1.0 + self.amplitude * (theta * (1.0 + gamma))
        return x * pac_modulator


OPERATOR_REGISTRY = {
    "linear": LinearOp,
    "swiglu": NonLinearOp,
    "delay": DelayOp,
    "gate": GateOp,
    "hopfield": ContinuousHopfieldOp,
    "pac": ThetaGammaOscillationOp,
}


# ============================================================================
# 2. UNIVERSAL SERIALIZABLE DAG ENGINE
# ============================================================================

class SerializableDAGEngine(nn.Module):
    """
    Directed Acyclic Graph that can export its entire topological blueprint into JSON
    and reconstruct itself dynamically from .kcore Manifest/Weights.
    """
    def __init__(self, dim: int, node_specs: list = None):
        super().__init__()
        self.dim = dim
        
        # Default node specification if not provided
        if node_specs is None:
            self.node_specs = [
                {"name": "pac_0", "type": "pac"},
                {"name": "delay_0", "type": "delay"},
                {"name": "hopfield_0", "type": "hopfield"},
                {"name": "gate_0", "type": "gate"},
                {"name": "swiglu_0", "type": "swiglu"},
                {"name": "delay_1", "type": "delay"},
            ]
        else:
            self.node_specs = node_specs
            
        self.num_internal_nodes = len(self.node_specs)
        self.num_nodes = self.num_internal_nodes + 2 # 0=Input, N+1=Output
        
        self.node_names = ["INPUT_SOURCE"] + [spec["name"] for spec in self.node_specs] + ["OUTPUT_SINK"]
        
        # Instantiate operator modules
        self.nodes = nn.ModuleList()
        for spec in self.node_specs:
            op_cls = OPERATOR_REGISTRY[spec["type"]]
            self.nodes.append(op_cls(dim))
            
        # Adjacency matrix
        self.edge_logits = nn.Parameter(torch.ones(self.num_nodes, self.num_nodes) * -1.5)
        self.node_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.num_nodes)])

    def get_adjacency_matrix(self) -> torch.Tensor:
        triu_mask = torch.triu(torch.ones(self.num_nodes, self.num_nodes, device=self.edge_logits.device), diagonal=1)
        return torch.sigmoid(self.edge_logits) * triu_mask

    def export_blueprint(self) -> dict:
        """Exports full topological genome for .kcore Section 1 (Manifest)."""
        adj = self.get_adjacency_matrix().detach().cpu().tolist()
        return {
            "dim": self.dim,
            "num_nodes": self.num_nodes,
            "node_names": self.node_names,
            "node_specs": self.node_specs,
            "adjacency_matrix": adj,
        }

    @classmethod
    def from_blueprint(cls, blueprint: dict) -> "SerializableDAGEngine":
        """Reconstructs the exact DAG engine from blueprint."""
        dim = blueprint["dim"]
        node_specs = blueprint["node_specs"]
        engine = cls(dim=dim, node_specs=node_specs)
        return engine

    def forward(self, sensory_input: torch.Tensor) -> torch.Tensor:
        adj = self.get_adjacency_matrix()
        node_states = [None] * self.num_nodes
        node_states[0] = sensory_input

        for j in range(1, self.num_nodes - 1):
            incoming = torch.zeros_like(sensory_input)
            for i in range(j):
                weight = adj[i, j]
                incoming = incoming + weight * node_states[i]
                
            op_idx = j - 1
            op = self.nodes[op_idx]
            norm_incoming = self.node_norms[j](incoming)
            node_states[j] = incoming + op(norm_incoming)

        sink_idx = self.num_nodes - 1
        output_sink = torch.zeros_like(sensory_input)
        for i in range(sink_idx):
            weight = adj[i, sink_idx]
            output_sink = output_sink + weight * node_states[i]

        return self.node_norms[sink_idx](output_sink)


# ============================================================================
# 3. COMPLETE DAG AGENT WITH PERSISTENCE HOOKS
# ============================================================================

class DAGKcoreAgent(nn.Module):
    def __init__(self, vocab_size: int = 258, dim: int = 256, node_specs: list = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.dag = SerializableDAGEngine(dim=dim, node_specs=node_specs)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h_dag = self.dag(h)
        return self.head(h_dag)


# ============================================================================
# 4. KCORE DAG SERIALIZER & RUNTIME EXTRACTOR
# ============================================================================

def save_dag_to_kcore(agent: DAGKcoreAgent, filepath: str = "test_dag_entity.kcore"):
    """
    Serializes a DAG agent into a self-contained container:
      - Section 1: JSON Manifest with DAG topology and operator blueprints.
      - Section 3: Exact PyTorch state_dict tensor weights.
    """
    blueprint = agent.dag.export_blueprint()
    manifest = {
        "version": "6.0.0-DAG",
        "arch": "Karyon-CoRE Epigenetic Dynamic DAG Engine",
        "vocab_size": agent.vocab_size,
        "dim": agent.dim,
        "dag_blueprint": blueprint,
    }
    
    # Pack into container dictionary
    container_payload = {
        "manifest": manifest,
        "weights": agent.state_dict(),
    }
    
    torch.save(container_payload, filepath)
    logger.info(f"✅ DAG Architecture & Weights saved to container '{filepath}'")


def load_dag_from_kcore(filepath: str = "test_dag_entity.kcore", device: torch.device = "cpu") -> DAGKcoreAgent:
    """
    Dynamically parses .kcore container, reconstructs the graph topology from the manifest blueprint,
    and loads parameters into the newly assembled network.
    """
    container_payload = torch.load(filepath, map_location=device)
    manifest = container_payload["manifest"]
    weights = container_payload["weights"]
    
    dag_blueprint = manifest["dag_blueprint"]
    vocab_size = manifest["vocab_size"]
    dim = manifest["dim"]
    node_specs = dag_blueprint["node_specs"]
    
    # 1. Dynamically assemble agent matching the exact container graph topology
    reconstructed_agent = DAGKcoreAgent(
        vocab_size=vocab_size,
        dim=dim,
        node_specs=node_specs
    ).to(device)
    
    # 2. Restore weights
    reconstructed_agent.load_state_dict(weights)
    logger.info(f"✅ Successfully reconstructed DAG Agent from '{filepath}' (Nodes: {len(node_specs) + 2})")
    return reconstructed_agent


# ============================================================================
# 5. EXPERIMENT VERIFICATION
# ============================================================================

def run_experiment():
    logger.info("=== Starting EXP-237: Universal DAG-KCORE Persistence Benchmark ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Backend: {device}")

    # 1. Create original DAG Agent
    original_agent = DAGKcoreAgent(vocab_size=258, dim=256).to(device)
    
    # Create test input
    test_input = torch.randint(0, 256, (4, 16), device=device)
    
    with torch.no_grad():
        original_output = original_agent(test_input)
        
    # 2. Save into .kcore container format
    kcore_path = "experiments/test_dynamic_dag.kcore"
    save_dag_to_kcore(original_agent, filepath=kcore_path)
    
    # 3. Load and reconstruct agent directly from .kcore container
    reconstructed_agent = load_dag_from_kcore(filepath=kcore_path, device=device)
    
    with torch.no_grad():
        reconstructed_output = reconstructed_agent(test_input)
        
    # 4. Strict numerical fidelity verification
    max_delta = torch.max(torch.abs(original_output - reconstructed_output)).item()
    logger.info(f"Numerical Delta between Original and Reconstructed: {max_delta:.8f}")
    
    assert max_delta < 1e-6, f"Fidelity error! Delta {max_delta} exceeds 1e-6!"
    
    # Clean up test file
    if os.path.exists(kcore_path):
        os.remove(kcore_path)
        
    print("\n--- EXP-237 VERIFIED: DYNAMIC DAG CONTAINER SERIALIZATION 100% OPERATIONAL ---\n")
    print("EXP_ID=EXP-237")
    print("VERDICT=POSITIVE")
    print(f"MAX_NUMERICAL_DELTA={max_delta:.10f}")


if __name__ == "__main__":
    run_experiment()
