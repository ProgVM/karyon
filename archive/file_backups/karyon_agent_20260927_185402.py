# karyon_agent.py
"""
===============================================================================
KARYON CORE AGENT MASTER WRAPPER v36.0
===============================================================================
Python Orchestrator Wrapper for C++20 Spatiotemporal Morphogenetic Engine:
- Temporal Domain: C++20 CausalParallelSSD (Continuous linear state space time-mixing)
- Spatial Domain: C++20 DynamicMorphicGraph (Recurrent thinking cycles across dynamic mathematical operators)
- Continuous Hopfield Attractor Memory for discrete concept snapping
- Sleep-Consolidation, Edelman Neural Darwinism Apoptosis, Epigenetic Sprouting (AGN v7.0)
- Susumu Ohno Gene Lock & Organelle Duplication Law integration
- Unified Parameter Registration & 100% .kcore v5.0 Container Serialization.
===============================================================================
"""
import math
import random
from typing import Dict, Any, Tuple, Optional, List, Union

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

        # 6. Endogenous Somatic Stress Accumulator & Autonomous Allostatic Morphogenesis Reflex
        self.somatic_stress: float = 0.0
        self.stress_lambda: float = 0.85
        self.tau_base: float = 0.50
        self.theta_morph: float = 1.5
        self.refractory_cooldown: int = 0
        self.refractory_period: int = 150
        self.min_grounding_steps: int = 200
        self.step_counter: int = 0
        self.active_organelle_idx: int = 0
        self.morphogenesis_count: int = 0
        self.max_morphogenesis_events: int = 1
        self.morphogenesis_events: List[Dict[str, Any]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        thinking_steps: Optional[int] = None,
        max_thinking_steps: int = 8,
        halt_threshold: float = 0.8,
        epsilon_halt: float = 1e-3,
        return_thinking_steps: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """
        Spatiotemporal Dual-Phase Forward Pass:
        1. Temporal State-Space Causal Scan (Sequence mixing across S).
        2. Spatial Morphogenetic Graph Recirculation (Deliberative latent thinking across K).
        When thinking_steps is None, executes Sovereign Adaptive Pondering / Halting (KEP Principle 21).
        """
        actual_steps = 0.0
        if input_ids.dtype in (torch.long, torch.int32, torch.int64):
            x = self.emb(input_ids)
            if x.dim() == 3:
                B, S, D = x.shape
                # Step 1: Temporal Causal State-Space Mixing
                h_seq = self.ssd.forward(x)  # [B, S, D]
                # Step 2: Spatial/Deliberative Graph Recirculation
                h_flat = h_seq.reshape(B * S, D)
                if thinking_steps is not None:
                    h_graph = self.graph.forward(h_flat, thinking_steps).reshape(B, S, D)
                    actual_steps = float(thinking_steps)
                else:
                    h_graph_flat, actual_steps = self.graph.forward_adaptive(
                        h_flat, max_thinking_steps, halt_threshold, epsilon_halt
                    )
                    h_graph = h_graph_flat.reshape(B, S, D)

                # Step 3: Residual Highway + Readout
                h_out = h_seq + h_graph
                logits = self.head(self.norm(h_out))
                if return_thinking_steps:
                    return logits, actual_steps
                return logits
            else:
                # 2D input [B, D] continuous vectors
                if thinking_steps is not None:
                    h_graph = self.graph.forward(x, thinking_steps)
                    actual_steps = float(thinking_steps)
                else:
                    h_graph, actual_steps = self.graph.forward_adaptive(
                        x, max_thinking_steps, halt_threshold, epsilon_halt
                    )
                logits = self.head(self.norm(x + h_graph))
                if return_thinking_steps:
                    return logits, actual_steps
                return logits
        else:
            # Continuous tensor forward pass
            if input_ids.dim() == 3:
                B, S, D = input_ids.shape
                h_seq = self.ssd.forward(input_ids)
                h_flat = h_seq.reshape(B * S, D)
                if thinking_steps is not None:
                    h_graph = self.graph.forward(h_flat, thinking_steps).reshape(B, S, D)
                    actual_steps = float(thinking_steps)
                else:
                    h_graph_flat, actual_steps = self.graph.forward_adaptive(
                        h_flat, max_thinking_steps, halt_threshold, epsilon_halt
                    )
                    h_graph = h_graph_flat.reshape(B, S, D)
                out = self.head(self.norm(h_seq + h_graph))
                if return_thinking_steps:
                    return out, actual_steps
                return out
            else:
                if thinking_steps is not None:
                    out = self.graph.forward(input_ids, thinking_steps)
                    actual_steps = float(thinking_steps)
                else:
                    out, actual_steps = self.graph.forward_adaptive(
                        input_ids, max_thinking_steps, halt_threshold, epsilon_halt
                    )
                if return_thinking_steps:
                    return out, actual_steps
                return out

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
        100% Modality-Agnostic Endogenous Contrast & Attractor Resonance (KEP Principles 10, 12, 19).
        Zero hardcoded ASCII values or delimiter checks.
        """
        B, L, D = p_field.shape
        # Temporal step via continuous state mixing
        h_core = self.ssd.forward(torch.cat([h_core.unsqueeze(1), x_t.unsqueeze(1)], dim=1))[:, -1, :]
        
        # 1. Pure Modality-Agnostic Information Contrast Salience:
        # Measure feature variance/deviation of each token vector from the local field centroid
        mean_field = p_field.mean(dim=1, keepdim=True) # [B, 1, D]
        field_contrast = torch.norm(p_field - mean_field, dim=-1) # [B, L]
        
        # Normalized contrast landscape + learned feature importance
        learned_salience = self.salience_proj(p_field).squeeze(-1) # [B, L]
        salience_bias = field_contrast + learned_salience # [B, L]
        
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

    def compute_initial_focus(self, p_field: torch.Tensor, h_core: torch.Tensor) -> torch.Tensor:
        """
        Pure modality-agnostic initial focus attractor localization (KEP Principles 12 & 19).
        Computes endogenous attractor query over settled prompt field without ASCII or delimiter checks.
        """
        B, L, D = p_field.shape
        q_focus = self.init_focus_q(h_core).unsqueeze(1) # [B, 1, D]
        k_field = self.content_k(p_field)               # [B, L, D]
        
        # Endogenous Information Contrast
        mean_field = p_field.mean(dim=1, keepdim=True)
        field_contrast = torch.norm(p_field - mean_field, dim=-1)
        learned_salience = self.salience_proj(p_field).squeeze(-1)
        salience_bias = field_contrast + learned_salience
        
        # Associative resonance + endogenous salience
        scores = torch.bmm(q_focus, k_field.transpose(1, 2)).squeeze(1) / (D ** 0.5)
        init_bump = torch.softmax((scores + salience_bias) * 10.0, dim=-1)
        return init_bump

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

    def update_somatic_stress_and_morphogenesis(self, free_energy: float) -> Optional[Dict[str, Any]]:
        """
        Endogenous Somatic Stress Accumulator & Autonomous Allostatic Morphogenesis Reflex:
        S_t = lambda * S_{t-1} + max(0, F_t - tau_base)
        Triggers autonomous morphogenesis reflex when S_t > theta_morph, step_counter >= min_grounding_steps,
        and not in refractory cooldown.
        """
        self.step_counter += 1
        if self.refractory_cooldown > 0:
            self.refractory_cooldown -= 1

        stress_increment = max(0.0, free_energy - self.tau_base)
        self.somatic_stress = self.stress_lambda * self.somatic_stress + stress_increment

        if (self.somatic_stress > self.theta_morph and 
                self.refractory_cooldown == 0 and 
                self.step_counter >= self.min_grounding_steps and
                self.morphogenesis_count < self.max_morphogenesis_events):
            self.morphogenesis_count += 1
            parent_idx = self.active_organelle_idx
            
            # 1. Epigenetically lock active parent organelle
            self.lock_node(parent_idx, 1.0)
            
            # 2. Susumu Ohno Zero-Shock Duplication
            clone_name = f"auto_organelle_gen{self.morphogenesis_count}"
            clone_idx = self.duplicate_node(parent_idx, clone_name, initial_alpha=1.0)
            
            # 3. Update active organelle index to newly sprouted plastic clone
            self.active_organelle_idx = clone_idx
            
            # 4. Adaptive Noise Injection Impulse
            sigma_noise = min(0.2, 0.02 * math.exp(min(2.0, free_energy / 5.0)))
            
            # 5. Reset stress accumulator & set refractory period & step counter
            prev_stress = self.somatic_stress
            self.somatic_stress = 0.0
            self.refractory_cooldown = self.refractory_period
            self.step_counter = 0

            event = {
                "generation": self.morphogenesis_count,
                "parent_idx": parent_idx,
                "clone_idx": clone_idx,
                "clone_name": clone_name,
                "free_energy": free_energy,
                "somatic_stress": prev_stress,
                "sigma_noise": sigma_noise
            }
            self.morphogenesis_events.append(event)
            logger.info(
                f"🧬 [AUTONOMOUS MORPHOGENESIS] Triggered! Parent Node {parent_idx} Locked (mu=1.0) -> "
                f"Cloned Node {clone_idx} ('{clone_name}') | Stress: {prev_stress:.2f} > {self.theta_morph:.2f} | "
                f"Free Energy: {free_energy:.4f}"
            )
            return event
        return None
    def add_node(self, name: str, op_type: str, is_core: bool = False, initial_alpha: float = 0.0) -> int:
        """Sprouts a new node inside the C++20 DynamicMorphicGraph."""
        self.graph.add_node(name, op_type, is_core, initial_alpha)
        return self.graph.k_nodes - 1

    def lock_node(self, idx: int, lock_value: float = 1.0):
        """Locks node parameters via Epigenetic Methylation (Susumu Ohno's Protection Law)."""
        self.graph.lock_node(idx, lock_value)

    def duplicate_node(self, src_idx: int, new_name: str, initial_alpha: float = 0.0) -> int:
        """Clones a node inside the C++20 DynamicMorphicGraph with zero-shock identity."""
        return self.graph.duplicate_node(src_idx, new_name, initial_alpha)

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
            "StateSpaceMemory",
            "StochasticLangevin",
            "ProgrammableDelay"
        )
    ) -> Dict[str, float]:
        """
        Executes Biophysical Sleep & Morphogenetic Neurogenesis Cycle:
        1. Tononi SHY Synaptic Scaling (soft downscaling with epigenetic methylation locks).
        2. Neural Darwinism Apoptosis (pruning inactive nodes with |tanh(alpha)| < prune_threshold).
        3. Epigenetic Sprouting of new dynamic graph nodes (AGN v7.0 / Net2Net zero-shock).
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
        state["content_q.weight"] = self.content_q.weight
        state["content_k.weight"] = self.content_k.weight
        state["salience_proj.weight"] = self.salience_proj.weight
        state["salience_proj.bias"] = self.salience_proj.bias
        state["gaze_gate.weight"] = self.gaze_gate.weight
        state["gaze_gate.bias"] = self.gaze_gate.bias
        state["copy_gate.weight"] = self.copy_gate.weight
        state["copy_gate.bias"] = self.copy_gate.bias
        state["gaze_proj.weight"] = self.gaze_proj.weight
        state["gaze_proj.bias"] = self.gaze_proj.bias
        state["init_focus_q.weight"] = self.init_focus_q.weight
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
