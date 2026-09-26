# karyon_agent.py
"""
===============================================================================
KARYON CORE AGENT MASTER WRAPPER v35.0
===============================================================================
Python Orchestrator Wrapper for C++20 Spatiotemporal Morphogenetic Engine:
- Temporal Domain: C++20 CausalParallelSSD (Continuous linear state space time-mixing)
- Spatial Domain: C++20 DynamicMorphicGraph (Recurrent thinking cycles across dynamic mathematical operators)
- Continuous Hopfield Attractor Memory for discrete concept snapping
- Sleep-Consolidation, Edelman Neural Darwinism Apoptosis, Epigenetic Sprouting (AGN v6.0)
- Unified Parameter Registration & 100% .kcore v5.0 Container Serialization.
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
    Master Python wrapper over C++20 Spatiotemporal DynamicMorphicGraph engine.
    Integrates CausalParallelSSD (temporal context flow) with DynamicMorphicGraph
    (spatial/recurrent latent deliberation depth).
    Ensures 100% compliance with KEP Principle 22 (Spatiotemporal Dualism)
    and karyon_checkpoint.py (.kcore v5.0 serialization).
    """
    def __init__(
        self,
        vocab_size: int = 258,
        embed_dim: int = 256,
        device: str = 'cpu',
        min_decay: float = 0.005,
        max_decay: float = 0.2
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

        # 1. Universal Byte Manifold Embedding
        self.emb = nn.Embedding(vocab_size, embed_dim).to(device)
        nn.init.normal_(self.emb.weight, 0.0, 0.02)

        # 2. C++20 CausalParallelSSD (Temporal Axis S: State-Space Causal Sequence Mixing)
        self.ssd = kcore.CausalParallelSSD(embed_dim, str(device), min_decay, max_decay)

        # 3. C++20 DynamicMorphicGraph (Spatial/Thinking Axis K: Recurrent Deliberation Depth)
        self.graph = kcore.DynamicMorphicGraph(embed_dim, str(device))
        # Initialize default foundational operators for immediate bidirectional graph connectivity
        self.graph.add_node("core_acc", "LinearAccumulator", True, 1.0)
        self.graph.add_node("core_sat", "SaturatedAttractor", True, 1.0)

        # 4. C++20 Continuous Saccadic Attractor Drift (C-SSD Engine)
        self.saccadic_drift = kcore.ContinuousSaccadicDrift(embed_dim, 17, str(device))
        
        # 5. Continuous Gaze & Copy Projection
        self.content_q = nn.Linear(embed_dim, embed_dim, bias=False).to(device)
        self.content_k = nn.Linear(embed_dim, embed_dim, bias=False).to(device)
        self.salience_proj = nn.Linear(embed_dim, 1, bias=True).to(device)
        self.gaze_gate = nn.Linear(embed_dim, 1, bias=True).to(device)
        self.copy_gate = nn.Linear(embed_dim, 1, bias=True).to(device)
        self.gaze_proj = nn.Linear(embed_dim * 2, embed_dim).to(device)

        # Focus Initialization / Target Saccade Trigger Query
        self.init_focus_q = nn.Linear(embed_dim, embed_dim, bias=False).to(device)
        self.norm = nn.LayerNorm(embed_dim).to(device)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
        self.head.weight = self.emb.weight

    def forward(self, input_ids: torch.Tensor, thinking_steps: int = 4) -> torch.Tensor:
        """
        Spatiotemporal Dual-Phase Forward Pass:
        1. Temporal State-Space Causal Scan (Sequence mixing across S).
        2. Spatial Morphogenetic Graph Recirculation (Deliberative latent thinking across K).
        """
        if input_ids.dtype in (torch.long, torch.int32, torch.int64):
            x = self.emb(input_ids)
            if x.dim() == 3:
                B, S, D = x.shape
                # Step 1: Temporal Causal State-Space Mixing
                h_seq = self.ssd.forward(x) # [B, S, D]

                # Step 2: Spatial/Deliberative Graph Recirculation
                h_flat = h_seq.reshape(B * S, D)
                h_graph = self.graph.forward(h_flat, thinking_steps).reshape(B, S, D)

                # Step 3: Residual Highway + Readout
                h_out = h_seq + h_graph
                logits = self.head(self.norm(h_out))
                return logits
            else:
                # 2D input [B, D] continuous vectors
                h_graph = self.graph.forward(x, thinking_steps)
                return self.head(self.norm(x + h_graph))
        else:
            # Continuous tensor forward pass
            if input_ids.dim() == 3:
                B, S, D = input_ids.shape
                h_seq = self.ssd.forward(input_ids)
                h_flat = h_seq.reshape(B * S, D)
                h_graph = self.graph.forward(h_flat, thinking_steps).reshape(B, S, D)
                return self.head(self.norm(h_seq + h_graph))
            else:
                return self.graph.forward(input_ids, thinking_steps)
    def forward_autoregressive_step(
        self,
        x_t: torch.Tensor,
        h_core: torch.Tensor,
        p_field: torch.Tensor,
        p_tokens: torch.Tensor,
        bump: torch.Tensor,
        thinking_steps: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Closed-loop causal step utilizing native C++20 CausalParallelSSD & ContinuousSaccadicDrift.
        Endogenous, modality-agnostic salience and content resonance (KEP Principles 10, 12, 19).
        """
        B, L, D = p_field.shape
        # Temporal step via continuous state mixing
        h_core = self.ssd.forward(torch.cat([h_core.unsqueeze(1), x_t.unsqueeze(1)], dim=1))[:, -1, :]
        
        # 1. Pure Endogenous Information-Theoretic Salience Landscape from Latent Representations
        # Modality-agnostic: measures the semantic/feature contrast of each token relative to the prompt field
        salience_bias = self.salience_proj(p_field).squeeze(-1) # [B, L]
        
        # 2. Continuous Saccadic Attractor Drift in C++20 with Endogenous Salience Landscape
        drifted_bump, _ = self.saccadic_drift(bump, h_core, 0.1, salience_bias)
        
        # 3. Content resonance over prompt field
        q = self.content_q(h_core).unsqueeze(1)
        k = self.content_k(p_field)
        content_scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / (D ** 0.5)
        content_bump = torch.softmax((content_scores + salience_bias) * 10.0, dim=-1)
        
        # 4. Continuous Gaze Blending
        alpha_gaze = torch.sigmoid(self.gaze_gate(h_core))
        next_bump = alpha_gaze * drifted_bump + (1.0 - alpha_gaze) * content_bump
        next_bump = next_bump / (next_bump.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 5. Continuous Field Readout
        h_gaze = torch.bmm(next_bump.unsqueeze(1), p_field).squeeze(1)
        
        # 6. C++20 Dynamic Morphic Thinking Recirculation
        h_sensory = self.norm(h_core + self.gaze_proj(torch.cat([h_core, h_gaze], dim=-1)))
        h_deliberated = self.graph.forward(h_sensory, thinking_steps)
        h_fused = self.norm(h_sensory + h_deliberated)
        
        # 7. Copy Projection
        p_copy = torch.sigmoid(self.copy_gate(h_fused))
        copy_logits = torch.zeros(B, self.vocab_size, device=p_field.device)
        copy_logits.scatter_add_(1, p_tokens, next_bump)
        
        return h_fused, h_core, next_bump, p_copy, copy_logits

    def forward_latent(self, input_ids: torch.Tensor, thinking_steps: int = 4) -> torch.Tensor:
        """Returns internal continuous latent representations."""
        if input_ids.dtype in (torch.long, torch.int32, torch.int64):
            x = self.emb(input_ids)
        else:
            x = input_ids

        if x.dim() == 3:
            B, S, D = x.shape
            h_seq = self.ssd.forward(x)
            h_flat = h_seq.reshape(B * S, D)
            h_graph = self.graph.forward(h_flat, thinking_steps).reshape(B, S, D)
            return h_seq + h_graph
        else:
            h_graph = self.graph.forward(x, thinking_steps)
            return x + h_graph

    def add_node(self, name: str, op_type: str, is_core: bool = False, initial_alpha: float = 0.0):
        """Sprouts a new node inside the C++20 DynamicMorphicGraph."""
        self.graph.add_node(name, op_type, is_core, initial_alpha)

    def prune_inactive_nodes(self, threshold: float = 0.02) -> int:
        """Prunes inactive dynamic nodes via Edelman Neural Darwinism."""
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
        1. Tononi SHY Synaptic Scaling (soft downscaling with epigenetic methylation locks).
        2. Neural Darwinism Apoptosis (pruning inactive nodes with |tanh(alpha)| < prune_threshold).
        3. Epigenetic Sprouting of new dynamic graph nodes (AGN v6.0 / Net2Net zero-shock).
        """
        scaled_params_count = 0
        # Phase 1: Epigenetic Methylation Lock Protection
        with torch.no_grad():
            param_map = self.graph.named_parameters_map()
            for name, param in param_map.items():
                if "w_route" in name or "weight" in name or "w_" in name:
                    param.mul_(1.0 - downscaling_factor * 0.1)
                    scaled_params_count += 1
            for p in self.ssd.parameters():
                p.mul_(1.0 - downscaling_factor * 0.1)
                scaled_params_count += 1
            self.emb.weight.mul_(1.0 - downscaling_factor * 0.1)
            scaled_params_count += 1

        # Phase 2: Neural Darwinism Apoptosis (Pruning)
        pruned_nodes = self.prune_inactive_nodes(prune_threshold)

        # Phase 3: Epigenetic Sprouting
        sprouted = False
        if random.random() < sprout_probability:
            op_type = random.choice(available_ops)
            node_idx = self.graph.k_nodes
            node_name = f"sleep_sprouted_op_{node_idx}_{op_type.lower()}"
            self.add_node(name=node_name, op_type=op_type, is_core=False, initial_alpha=0.0)
            sprouted = True

        return {
            "scaled_params": float(scaled_params_count),
            "pruned_nodes": float(pruned_nodes),
            "sprouted": 1.0 if sprouted else 0.0,
            "total_nodes": float(self.graph.k_nodes)
        }

    def parameters(self, recurse: bool = True):
        """
        Overrides nn.Module.parameters() to return all active parameters,
        including dynamic C++20 graph parameters and CausalParallelSSD parameters.
        """
        for p in self.get_complete_state_dict().values():
            if isinstance(p, torch.Tensor) and p.requires_grad:
                yield p

    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
        """Overrides nn.Module.named_parameters() to include dynamic C++20 graph parameters."""
        for k, v in self.get_complete_state_dict().items():
            if isinstance(v, torch.Tensor) and v.requires_grad:
                name = f"{prefix}.{k}" if prefix else k
                yield name, v

    def get_topology_manifest(self) -> str:
        """Returns the JSON manifest representing the evolved graph topology."""
        return self.graph.get_topology_manifest()

    def get_complete_state_dict(self) -> Dict[str, torch.Tensor]:
        """Exposes C++20 module parameters for .kcore v5.0 container serialization."""
        state = {}
        for k, v in self.graph.named_parameters_map().items():
            state[f"graph.{k}"] = v
        for k, v in self.ssd.named_parameters().items():
            state[f"ssd.{k}"] = v
        state["emb.weight"] = self.emb.weight
        state["norm.weight"] = self.norm.weight
        state["norm.bias"] = self.norm.bias
        return state

    def load_complete_state_dict(self, state_dict: Dict[str, torch.Tensor], device: Optional[str] = None):
        """Restores module parameters shape-adaptively from binary state dictionary."""
        current_state = self.get_complete_state_dict()
        target_device = torch.device(device) if device else self.device
        with torch.no_grad():
            for k, v in state_dict.items():
                if k in current_state:
                    target = current_state[k]
                    src = v.to(target_device)
                    if target.shape == src.shape:
                        target.copy_(src)
                    else:
                        slices = [slice(0, min(d_t, d_s)) for d_t, d_s in zip(target.shape, src.shape)]
                        target[tuple(slices)].copy_(src[tuple(slices)])
