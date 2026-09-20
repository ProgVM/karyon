# karyon_agent.py
"""
===============================================================================
KARYON CORE AGENT MASTER WRAPPER v34.1
Python Orchestrator Wrapper for C++20 UniversalMorphicSpace & DynamicMorphicGraph
with Sleep-Consolidation and Morphogenetic Neurogenesis Engine
===============================================================================
"""
from typing import Dict, Tuple
import random

import torch
import torch.nn as nn

import karyon_core as kcore


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
            if input_ids.dtype == torch.long or input_ids.dtype == torch.int32:
                if not hasattr(self, 'graph_emb'):
                    self.graph_emb = nn.Embedding(self.vocab_size, self.embed_dim).to(self.device)
                    nn.init.normal_(self.graph_emb.weight, 0.0, 0.02)
                x = self.graph_emb(input_ids)
                if x.dim() == 3:
                    x = x.mean(dim=1)  # Pool sequence to batch for graph forward
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

    def execute_deep_allostatic_sleep(
        self,
        downscaling_factor: float = 0.01,
        sprout_probability: float = 0.5,
        available_ops: Tuple[str, ...] = ("LinearAccumulator", "BilinearMultiplicative", "SaturatedAttractor")
    ) -> Dict[str, float]:
        """
        Executes Biophysical Sleep & Morphogenetic Neurogenesis Cycle:
        1. Tononi SHY Synaptic Scaling (soft downscaling).
        2. Epigenetic Sprouting of new dynamic graph nodes (AGN v6.0 / Net2Net zero-shock).
        """
        # Phase 1: Tononi Synaptic Homeostasis Hypothesis (SHY) downscaling
        scaled_params_count = 0
        with torch.no_grad():
            if self.use_graph:
                param_map = self.graph.named_parameters_map()
                for name, param in param_map.items():
                    if "w_route" in name or "weight" in name:
                        param.mul_(1.0 - downscaling_factor)
                        scaled_params_count += 1
            else:
                param_map = self.space.named_parameters_map()
                for name, param in param_map.items():
                    if "weight" in name or "matrix" in name:
                        param.mul_(1.0 - downscaling_factor)
                        scaled_params_count += 1

        # Phase 2: Morphogenetic Graph Sprouting
        sprouted = False
        sprouted_type = None
        if self.use_graph and random.random() < sprout_probability:
            op_type = random.choice(available_ops)
            node_idx = self.graph.k_nodes
            node_name = f"sleep_sprouted_op_{node_idx}_{op_type.lower()}"
            self.add_node(name=node_name, op_type=op_type, is_core=False, initial_alpha=0.0)
            sprouted = True
            sprouted_type = op_type

        return {
            "scaled_params": float(scaled_params_count),
            "sprouted": 1.0 if sprouted else 0.0,
            "total_nodes": float(self.graph.k_nodes if self.use_graph else 0)
        }

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
            op_indices = set()
            for key in state_dict.keys():
                if key.startswith("op_") and "_" in key:
                    parts = key.split("_")
                    if parts[1].isdigit():
                        op_indices.add(int(parts[1]))

            sorted_indices = sorted(list(op_indices))
            for idx in sorted_indices:
                if idx >= self.graph.k_nodes:
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
