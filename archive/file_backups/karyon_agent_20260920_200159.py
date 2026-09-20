# karyon_agent.py
"""
===============================================================================
KARYON CORE AGENT MASTER WRAPPER v34.0
Python Orchestrator Wrapper for C++20 UniversalMorphicSpace & DynamicMorphicGraph
===============================================================================
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List

import karyon_core as kcore
from karyon_config import CoREConfig

class ConfigMock:
    class NetMock:
        text_dim = 256
        text_gen_dim = 256
        vocab_size = 258
    net = NetMock()

class CoREAgent(nn.Module):
    """
    Master Python wrapper over C++20 UniversalMorphicSpace & DynamicMorphicGraph engines.
    Ensures 100% compliance with karyon_checkpoint.py (.kcore v5.0 serialization).
    """
    def __init__(self, vocab_size=258, embed_dim=256, num_cells=2, num_operators=4, device='cpu', use_graph=False):
        super().__init__()
        self.config = ConfigMock()
        self.device = device
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.unified_dim = embed_dim
        self.hidden_dim = embed_dim
        self.latent_dim = 64
        self.action_dim = vocab_size
        self.use_graph = use_graph

        if self.use_graph:
            # Instantiate the C++20 DynamicMorphicGraph
            self.graph = kcore.DynamicMorphicGraph(embed_dim, device)
        else:
            # Instantiate the C++20 UniversalMorphicSpace engine
            self.space = kcore.UniversalMorphicSpace(
                vocab_size, embed_dim, num_cells, num_operators, device
            )

    def forward(self, input_ids: torch.Tensor, thinking_steps: int = 4) -> torch.Tensor:
        """Direct forward pass returning logits or graph readouts."""
        if self.use_graph:
            # For graph, input_ids is projected into embed_dim space or used as sensory features
            # Let's project token embeddings using a small embedding layer if input_ids is token indices
            if input_ids.dtype == torch.long or input_ids.dtype == torch.int32:
                # We can use an embedding layer or similar, but since DynamicMorphicGraph takes continuous vectors,
                # let's map tokens to continuous space using a simple one-hot or embedding map.
                # To keep it self-contained, we can dynamically initialize a small embedding weight if needed.
                if not hasattr(self, 'graph_emb'):
                    self.graph_emb = nn.Embedding(self.vocab_size, self.embed_dim).to(self.device)
                    nn.init.normal_(self.graph_emb.weight, 0.0, 0.02)
                x = self.graph_emb(input_ids)
                if x.dim() == 3:
                    x = x.mean(dim=1) # Pool sequence to batch for graph forward
            else:
                x = input_ids
            return self.graph(x, thinking_steps)
        else:
            return self.space(input_ids)

    def forward_latent(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning latent manifold states."""
        if self.use_graph:
            if not hasattr(self, 'graph_emb'):
                self.graph_emb = nn.Embedding(self.vocab_size, self.embed_dim).to(self.device)
                nn.init.normal_(self.graph_emb.weight, 0.0, 0.02)
            x = self.graph_emb(input_ids)
            if x.dim() == 3:
                x = x.mean(dim=1)
            # Return graph state as a pseudo-latent representation [B, 1, D]
            out = self.graph(x).unsqueeze(1)
            return out
        else:
            return self.space.forward_latent(input_ids)

    def add_node(self, name: str, op_type: str, is_core: bool = False, initial_alpha: float = 0.0):
        """Sprouts a new node inside the C++20 DynamicMorphicGraph."""
        if not self.use_graph:
            raise ValueError("add_node can only be called when use_graph=True")
        self.graph.add_node(name, op_type, is_core, initial_alpha)

    def get_topology_manifest(self) -> str:
        """Returns the JSON manifest representing the evolved graph topology."""
        if self.use_graph:
            return self.graph.get_topology_manifest()
        return "{}"

    def get_complete_state_dict(self) -> Dict[str, torch.Tensor]:
        """Exposes C++20 module parameters for .kcore v5.0 container serialization."""
        state = {}
        if self.use_graph:
            state.update(self.graph.named_parameters_map())
            if hasattr(self, 'graph_emb'):
                for k, v in self.graph_emb.state_dict().items():
                    state[f"graph_emb.{k}"] = v
        else:
            state.update(self.space.named_parameters_map())
        return state

    def load_complete_state_dict(self, state_dict: Dict[str, torch.Tensor], device: str = 'cpu'):
        """Restores parameters into C++20 core engine."""
        if self.use_graph:
            # Reconstruct evolved nodes from state_dict if they aren't created yet
            # Let's parse node names from state_dict keys to auto-sprout them!
            # Keys look like: op_0_alpha, op_0.w, op_0.b, op_1.w_left, etc.
            op_indices = set()
            for key in state_dict.keys():
                if key.startswith("op_") and "_" in key:
                    parts = key.split("_")
                    if parts[1].isdigit():
                        op_indices.add(int(parts[1]))
            
            # Sort indices and sprout them if they don't exist
            sorted_indices = sorted(list(op_indices))
            for idx in sorted_indices:
                if idx >= self.graph.k_nodes:
                    # We need to determine the op_type from the state_dict keys
                    # e.g., if we see "op_idx.w_left", it's BilinearMultiplicative
                    # if we see "op_idx.w_gate", it's SaturatedAttractor
                    # otherwise LinearAccumulator
                    op_type = "LinearAccumulator"
                    for k in state_dict.keys():
                        if k.startswith(f"op_{idx}."):
                            if "w_left" in k:
                                op_type = "BilinearMultiplicative"
                                break
                            elif "w_gate" in k:
                                op_type = "SaturatedAttractor"
                                break
                    self.add_node(name=f"auto_node_{idx}", op_type=op_type)

            param_map = self.graph.named_parameters_map()
            with torch.no_grad():
                for name, param in param_map.items():
                    if name in state_dict:
                        param.copy_(state_dict[name].to(device))
            
            if hasattr(self, 'graph_emb'):
                emb_state = {}
                for k, v in state_dict.items():
                    if k.startswith("graph_emb."):
                        emb_state[k.replace("graph_emb.", "")] = v
                if emb_state:
                    self.graph_emb.load_state_dict(emb_state)
        else:
            param_map = self.space.named_parameters_map()
            with torch.no_grad():
                for name, param in param_map.items():
                    if name in state_dict:
                        param.copy_(state_dict[name].to(device))
