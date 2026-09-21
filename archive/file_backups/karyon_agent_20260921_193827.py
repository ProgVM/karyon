# karyon_agent.py
"""
===============================================================================
KARYON CORE AGENT MASTER WRAPPER v34.4
===============================================================================
Python Orchestrator Wrapper for C++20 UniversalMorphicSpace & DynamicMorphicGraph
with Integrated Continuous Hopfield Attractor Memory, Sleep-Consolidation,
Allostatic Morphogenesis Engine (Sprouting + Apoptosis), and Safe Optimizer State Rebinding.
===============================================================================
"""
import math
import random
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import karyon_core as kcore
from karyon_logger import get_logger

logger = get_logger()


class ConfigMock:
    pass


class CoREAgent(nn.Module):
    """
    Master Python wrapper over C++20 UniversalMorphicSpace & DynamicMorphicGraph engines.
    Equipped with Continuous Hopfield Attractor Memory for zero-shot pattern separation.
    Ensures 100% compliance with karyon_checkpoint.py (.kcore v5.0 serialization).
    """
    def __init__(
        self,
        vocab_size: int = 258,
        embed_dim: int = 256,
        num_cells: int = 2,
        num_operators: int = 4,
        device: str = 'cpu',
        use_graph: bool = False,
        use_hopfield: bool = True,
        num_basins: int = 128
    ):
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
        self.use_hopfield = use_hopfield and (not use_graph)

        if self.use_graph:
            # Instantiate the C++20 DynamicMorphicGraph
            self.graph = kcore.DynamicMorphicGraph(embed_dim, device)
            # Guarantee at least 2 base core operators for immediate bidirectional connectivity
            self.graph.add_node("core_0", "LinearAccumulator", True, 1.0)
            self.graph.add_node("core_1", "SaturatedAttractor", True, 1.0)
            self.graph_emb = nn.Embedding(vocab_size, embed_dim).to(device)
            self.graph_head = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
            nn.init.normal_(self.graph_emb.weight, 0.0, 0.02)
            nn.init.normal_(self.graph_head.weight, 0.0, 0.02)
        else:
            # Instantiate the C++20 UniversalMorphicSpace engine
            self.space = kcore.UniversalMorphicSpace(
                vocab_size, embed_dim, num_cells, num_operators, device
            )
            if self.use_hopfield:
                # Integrate Continuous Hopfield Attractor Memory directly
                self.hopfield = kcore.ContinuousHopfieldMemory(embed_dim, num_basins, device)
                self.hopfield_head = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
                nn.init.normal_(self.hopfield_head.weight, 0.0, 0.02)
                self.gate = nn.Parameter(torch.tensor([0.35], device=device))

    def forward(self, input_ids: torch.Tensor, thinking_steps: int = 4) -> torch.Tensor:
        """Direct forward pass returning logits or graph readouts."""
        if self.use_graph:
            if input_ids.dtype in (torch.long, torch.int32, torch.int64):
                x = self.graph_emb(input_ids)
                # If sequence dimension present [B, S, D], process sequence or pool representation
                if x.dim() == 3:
                    B, S, D = x.shape
                    # Pass through dynamic morphic recurrent thinking graph
                    x_flat = x.reshape(B * S, D)
                    h_graph = self.graph(x_flat, thinking_steps)
                    logits = self.graph_head(h_graph).reshape(B, S, self.vocab_size)
                    return logits
                else:
                    h_graph = self.graph(x, thinking_steps)
                    return self.graph_head(h_graph)
            else:
                return self.graph(input_ids, thinking_steps)
        else:
            # Continuous Morphic Space forward pass
            logits = self.space.forward(input_ids)
            if self.use_hopfield:
                h_latent = self.space.forward_latent(input_ids)
                h_hopfield = self.hopfield.forward(h_latent)
                hopfield_logits = self.hopfield_head(h_hopfield)
                g = torch.sigmoid(self.gate)
                logits = (1.0 - g) * logits + g * hopfield_logits
            return logits

    def forward_latent(self, input_ids: torch.Tensor, thinking_steps: int = 4) -> torch.Tensor:
        """Returns internal continuous latent representations."""
        if self.use_graph:
            x = self.graph_emb(input_ids)
            if x.dim() == 3:
                B, S, D = x.shape
                x_flat = x.reshape(B * S, D)
                h_graph = self.graph(x_flat, thinking_steps)
                return h_graph.reshape(B, S, D)
            else:
                return self.graph(x, thinking_steps).unsqueeze(1)
        else:
            return self.space.forward_latent(input_ids)

    def add_node(self, name: str, op_type: str, is_core: bool = False, initial_alpha: float = 0.0):
        """Sprouts a new node inside the C++20 DynamicMorphicGraph."""
        if not self.use_graph:
            raise ValueError("add_node can only be called when use_graph=True")
        self.graph.add_node(name, op_type, is_core, initial_alpha)

    def prune_inactive_nodes(self, threshold: float = 0.02) -> int:
        """Prunes inactive dynamic nodes via Edelman Neural Darwinism."""
        if not self.use_graph:
            return 0
        return self.graph.prune_inactive_nodes(threshold)

    def execute_deep_allostatic_sleep(
        self,
        downscaling_factor: float = 0.01,
        sprout_probability: float = 0.5,
        prune_threshold: float = 0.02,
        available_ops: Tuple[str, ...] = (
            "LinearAccumulator",
            "BilinearMultiplicative",
            "SaturatedAttractor",
            "ContinuousHopfield",
            "StateSpaceMemory"
        )
    ) -> Dict[str, float]:
        """
        Executes Biophysical Sleep & Morphogenetic Neurogenesis Cycle:
        1. Tononi SHY Synaptic Scaling (soft downscaling).
        2. Neural Darwinism Apoptosis (pruning inactive nodes with |tanh(alpha)| < prune_threshold).
        3. Epigenetic Sprouting of new dynamic graph nodes (AGN v6.0 / Net2Net zero-shock).
        """
        scaled_params_count = 0
        with torch.no_grad():
            if self.use_graph:
                param_map = self.graph.named_parameters_map()
                for name, param in param_map.items():
                    if "w_route" in name or "weight" in name or "w_" in name:
                        param.mul_(1.0 - downscaling_factor)
                        scaled_params_count += 1
                self.graph_emb.weight.mul_(1.0 - downscaling_factor)
                self.graph_head.weight.mul_(1.0 - downscaling_factor)
                scaled_params_count += 2
            else:
                param_map = self.space.named_parameters_map()
                for name, param in param_map.items():
                    if "weight" in name or "matrix" in name:
                        param.mul_(1.0 - downscaling_factor)
                        scaled_params_count += 1
                if self.use_hopfield:
                    for param in self.hopfield.parameters():
                        param.mul_(1.0 - downscaling_factor)
                        scaled_params_count += 1
                    self.hopfield_head.weight.mul_(1.0 - downscaling_factor)
                    scaled_params_count += 1

        # Phase 2: Neural Darwinism Apoptosis (Pruning)
        pruned_nodes = 0
        if self.use_graph:
            pruned_nodes = self.prune_inactive_nodes(prune_threshold)

        # Phase 3: Epigenetic Sprouting
        sprouted = False
        if self.use_graph and random.random() < sprout_probability:
            op_type = random.choice(available_ops)
            node_idx = self.graph.k_nodes
            node_name = f"sleep_sprouted_op_{node_idx}_{op_type.lower()}"
            self.add_node(name=node_name, op_type=op_type, is_core=False, initial_alpha=0.0)
            sprouted = True

        return {
            "scaled_params": float(scaled_params_count),
            "pruned_nodes": float(pruned_nodes),
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
            for k, v in self.graph.named_parameters_map().items():
                state[f"graph.{k}"] = v
            state["graph_emb.weight"] = self.graph_emb.weight
            state["graph_head.weight"] = self.graph_head.weight
        else:
            for k, v in self.space.named_parameters_map().items():
                state[f"space.{k}"] = v
            if self.use_hopfield:
                for idx, p in enumerate(self.hopfield.parameters()):
                    state[f"hopfield.param_{idx}"] = p
                state["hopfield_head.weight"] = self.hopfield_head.weight
                state["gate"] = self.gate
        return state

    def load_complete_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Restores module parameters shape-adaptively from binary state dictionary."""
        current_state = self.get_complete_state_dict()
        with torch.no_grad():
            for k, v in state_dict.items():
                if k in current_state:
                    target = current_state[k]
                    if target.shape == v.shape:
                        target.copy_(v)
                    else:
                        slices = [slice(0, min(d_t, d_s)) for d_t, d_s in zip(target.shape, v.shape)]
                        target[tuple(slices)].copy_(v[tuple(slices)])
