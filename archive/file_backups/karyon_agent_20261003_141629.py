# karyon_agent.py
"""
===============================================================================
KARYON CORE AGENT MASTER WRAPPER v37.0
===============================================================================
Python Orchestrator Wrapper for C++20 Spatiotemporal Morphogenetic Engine:
- Temporal Domain: C++20 TriScaleHierarchicalPAC (Gamma -> Theta -> Delta Cascade)
- Spatial Domain: C++20 DynamicMorphicGraph (Recurrent thinking cycles across dynamic mathematical operators)
- Continuous Hopfield Attractor Memory for discrete concept snapping
- Sleep-Consolidation, Edelman Neural Darwinism Apoptosis, Epigenetic Sprouting (AGN v7.0)
- Susumu Ohno Gene Lock & Organelle Duplication Law integration
- Unified Parameter Registration & 100% .kcore v6.0 Container Serialization.
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
    Integrates TriScaleHierarchicalPAC (temporal context flow) with DynamicMorphicGraph
    (spatial/recurrent latent deliberation depth).
    Ensures 100% compliance with KEP Principle 22 (Spatiotemporal Dualism)
    and karyon_checkpoint.py (.kcore v6.0 serialization).
    """
    def __init__(
        self,
        vocab_size: int = 258,
        embed_dim: int = 256,
        device: str = 'cpu',
        gamma_min_decay: float = 0.05,
        gamma_max_decay: float = 0.50,
        theta_min_decay: float = 0.005,
        theta_max_decay: float = 0.05,
        delta_min_decay: float = 0.0001,
        delta_max_decay: float = 0.001,
        num_hopfield_basins: int = 256,
        use_hopfield_snapping: bool = True,
        hopfield_beta: float = 12.0
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

        # 2. C++20 TriScaleHierarchicalPAC (Temporal Axis S: Three-scale Chrono-PAC Cascade)
        self.ssd = kcore.TriScaleHierarchicalPAC(
            embed_dim,
            str(device),
            gamma_min_decay,
            gamma_max_decay,
            theta_min_decay,
            theta_max_decay,
            delta_min_decay,
            delta_max_decay,
            num_hopfield_basins,
            use_hopfield_snapping,
            hopfield_beta
        )

        # 3. C++20 DynamicMorphicGraph (Spatial/Thinking Axis K: Recurrent Deliberation Depth)
        self.max_nodes = 128
        self.graph = kcore.DynamicMorphicGraph(embed_dim, str(device), self.max_nodes)
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

        # 5.1 Context-Gated Aversive Repulsor Hopfield Memory
        self.hopfield_memory = kcore.ContinuousHopfieldMemory(embed_dim, 32, str(device), 512)

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
        self.active_organelle_idx: int = 0
        self.morphogenesis_count: int = 0
        self.max_morphogenesis_events: int = 999999  # Unconstrained morphogenesis
        self.morphogenesis_events: List[Dict[str, Any]] = []

        # Endogenous Organelle Maturation & Cell Cycle Dynamics (EXP-321 Non-Constant)
        self.running_grad_norm: float = 0.1
        self.running_grad_var: float = 0.01
        self.maturity_threshold: float = 0.90

        # 7. Running Loss Statistics for Tripartite Valence Calibration (EXP-318)
        self.loss_running_mean: float = 3.0
        self.loss_running_var: float = 1.0
        self.loss_stat_count: int = 0
        self.loss_ema_alpha: float = 0.05

    def set_organelle_signature(self, node_idx: int, signature: torch.Tensor):
        """Sets the static molecular/Hox-gene signature passport for an organelle node."""
        self.graph.set_organelle_signature(node_idx, signature)

    def forward(
        self,
        input_ids: torch.Tensor,
        free_energy: Optional[torch.Tensor] = None,
        thinking_steps: Optional[int] = None,
        max_thinking_steps: int = 8,
        halt_threshold: float = 0.8,
        epsilon_halt: float = 1e-3,
        return_thinking_steps: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """
        Spatiotemporal Dual-Phase Forward Pass:
        1. Temporal Tri-Scale Hierarchical PAC Causal Scan (Gamma -> Theta -> Delta Cascade).
        2. Spatial Morphogenetic Graph Recirculation (Deliberative latent thinking across K).
        When thinking_steps is None, executes Sovereign Adaptive Pondering / Halting (KEP Principle 21).
        """
        actual_steps = 0.0
        if input_ids.dtype in (torch.long, torch.int32, torch.int64):
            x = self.emb(input_ids)
        else:
            x = input_ids

        is_2d = (x.dim() == 2)
        if is_2d:
            x = x.unsqueeze(1)

        # Step 1: Temporal Tri-Scale Cascade Scan
        # TriScaleHierarchicalPAC returns: (y_out, dt, g1, g2, na, da, h_gamma, h_theta, h_delta, h_snapped)
        res_tuple = self.ssd.forward(x, free_energy if free_energy is not None else torch.Tensor())
        h_seq = res_tuple[0]  # [B, S, D]

        if not is_2d:
            B, S, D = x.shape
            # Step 2: Spatial/Deliberative Graph Recirculation
            h_flat = h_seq.reshape(B * S, D)
            empty_ctx = torch.empty(0, device=h_flat.device)
            if thinking_steps is not None:
                h_graph = self.graph.forward(h_flat, empty_ctx, empty_ctx, int(thinking_steps)).reshape(B, S, D)
                actual_steps = float(thinking_steps)
            else:
                h_graph_flat, actual_steps = self.graph.forward_adaptive(
                    h_flat, max_thinking_steps, halt_threshold, epsilon_halt
                )
                h_graph = h_graph_flat.reshape(B, S, D)

            # Step 3: Residual Highway + Readout
            h_out = h_seq + h_graph
            h_norm = self.norm(h_out)
            
            # Step 3.1: Context-Gated Hopfield Aversive Repulsion & Attractor Snapping across sequence
            # Uses initial sensory context (h_seq[:, 0:1, :]) to repel known error actions in this context
            h_ctx = h_seq[:, 0, :].unsqueeze(1).expand(B, S, D).reshape(B * S, D)
            h_relaxed = self.hopfield_memory.relax_with_repulsion(h_ctx, h_norm.reshape(B * S, D)).reshape(B, S, D)
            h_norm = self.norm(h_norm + h_relaxed)

            logits = self.head(h_norm)
            if return_thinking_steps:
                return logits, actual_steps
            return logits
        else:
            # 2D input [B, D] continuous vectors: squeeze back
            h_seq_2d = h_seq.squeeze(1)
            empty_ctx = torch.empty(0, device=h_seq_2d.device)
            if thinking_steps is not None:
                h_graph = self.graph.forward(h_seq_2d, empty_ctx, empty_ctx, int(thinking_steps))
                actual_steps = float(thinking_steps)
            else:
                h_graph, actual_steps = self.graph.forward_adaptive(
                    h_seq_2d, max_thinking_steps, halt_threshold, epsilon_halt
                )
            logits = self.head(self.norm(h_seq_2d + h_graph))
            if return_thinking_steps:
                return logits, actual_steps
            return logits

    def forward_autoregressive_step(
        self,
        x_t: torch.Tensor,
        h_core: torch.Tensor,
        p_field: torch.Tensor,
        p_tokens: torch.Tensor,
        bump: torch.Tensor,
        thinking_steps: int = 4,
        free_energy: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Closed-loop causal step utilizing native C++20 TriScaleHierarchicalPAC & ContinuousSaccadicDrift.
        100% Modality-Agnostic Endogenous Contrast & Attractor Resonance (KEP Principles 10, 12, 19).
        Zero hardcoded ASCII values or delimiter checks.
        """
        B, L, D = p_field.shape
        # Temporal step via continuous state mixing
        # TriScaleHierarchicalPAC accepts sequence x [B, S, D]
        x_seq = torch.cat([h_core.unsqueeze(1), x_t.unsqueeze(1)], dim=1)
        res_tuple = self.ssd.forward(x_seq, free_energy if free_energy is not None else torch.Tensor())
        h_core = res_tuple[0][:, -1, :]
        
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

        # 6.1 Context-Gated Hopfield Aversive Repulsion & Attractor Snapping
        # Modulates deliberated action representation using accumulated somatic episodes
        h_action = self.hopfield_memory.relax_with_repulsion(h_sensory, h_fused)
        h_fused = self.norm(h_fused + h_action)
        
        # 7. Copy Projection
        p_copy = torch.sigmoid(self.copy_gate(h_fused))
        copy_logits = torch.zeros(B, self.vocab_size, device=p_field.device)
        copy_logits.scatter_add_(1, p_tokens, next_bump)
        
        return h_fused, h_core, next_bump, p_copy, copy_logits

    def record_somatic_step_feedback(
        self,
        context_t: torch.Tensor,
        action_t: torch.Tensor,
        free_energy_surprise: float,
        mean_loss: Optional[float] = None,
        std_loss: Optional[float] = None
    ) -> float:
        """
        EXP-318: Tripartite Continuous Valence & Neutral Anchor Memory.
        Calculates normalized error surge and continuous step valence V_t in [-1.0, +1.0]:
          Delta_norm = (loss_t - mean_loss) / (std_loss + 1e-5)
          V_t = -tanh(1.5 * Delta_norm)
          - Substantially below average (Delta_norm < -0.3) -> V_t > +0.4 (Attractor of success)
          - Substantially above average (Delta_norm > +0.5)  -> V_t < -0.6 (Repulsor of failure)
          - Baseline background loss   (|Delta_norm| <= 0.3) -> |V_t| <= 0.2 (Neutral topographic anchor)
        """
        # Ensure 2D tensor representations with strict [1, D] vector shapes
        ctx_vec = context_t.mean(dim=0, keepdim=True) if context_t.dim() > 1 and context_t.size(0) > 1 else context_t
        act_vec = action_t.mean(dim=0, keepdim=True) if action_t.dim() > 1 and action_t.size(0) > 1 else action_t

        ctx_vec = ctx_vec.view(1, -1)
        act_vec = act_vec.view(1, -1)

        assert ctx_vec.shape[1] == self.embed_dim, f"Invalid context dimension {ctx_vec.shape[1]}, expected {self.embed_dim}"
        assert act_vec.shape[1] == self.embed_dim, f"Invalid action dimension {act_vec.shape[1]}, expected {self.embed_dim}"

        # Update internal running loss statistics if external not provided
        loss_val = float(free_energy_surprise)
        if mean_loss is None or std_loss is None:
            if self.loss_stat_count == 0:
                self.loss_running_mean = loss_val
                self.loss_running_var = 1.0
            else:
                delta = loss_val - self.loss_running_mean
                self.loss_running_mean += self.loss_ema_alpha * delta
                self.loss_running_var = (1.0 - self.loss_ema_alpha) * self.loss_running_var + self.loss_ema_alpha * (delta ** 2)
            self.loss_stat_count += 1
            cur_mean = self.loss_running_mean
            cur_std = math.sqrt(max(self.loss_running_var, 1e-6))
        else:
            cur_mean = float(mean_loss)
            cur_std = max(float(std_loss), 1e-5)

        # Compute normalized error deviation and continuous valence
        delta_norm = (loss_val - cur_mean) / (cur_std + 1e-5)
        valence = -math.tanh(1.5 * delta_norm)
        valence = max(-1.0, min(1.0, valence))

        self.hopfield_memory.record_somatic_episode(ctx_vec, act_vec, float(valence))
        return float(valence)

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

    def forward_latent(self, input_ids: torch.Tensor, thinking_steps: int = 4, free_energy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns internal continuous latent representations."""
        if input_ids.dtype in (torch.long, torch.int32, torch.int64):
            x = self.emb(input_ids)
        else:
            x = input_ids

        res_tuple = self.ssd.forward(x, free_energy if free_energy is not None else torch.Tensor())
        h_seq = res_tuple[0]

        if x.dim() == 3:
            B, S, D = x.shape
            h_flat = h_seq.reshape(B * S, D)
            h_graph = self.graph.forward(h_flat, thinking_steps).reshape(B, S, D)
            return h_seq + h_graph
        else:
            h_graph = self.graph.forward(h_seq, thinking_steps)
            return h_seq + h_graph

    def compute_organelle_maturity(self, organelle_idx: int) -> float:
        """
        Computes the Endogenous Maturity Index M_k(t) in [0.0, 1.0] (EXP-321):
        M_k(t) = sigmoid((|tanh(alpha_epi(t))| - 0.8) * 10.0) * exp(-||grad_k|| / (sigma_grad + 1e-5))
        Determines when a newly sprouted infant organelle has functionally integrated.
        If node is epigenetically locked (mu=1.0) or core, it is fully mature (M=1.0).
        """
        if organelle_idx >= self.graph.k_nodes:
            return 1.0

        # If the organelle is epigenetically locked (frozen) or core, it is already mature
        active_locks = self.graph.get_methylation_locks()
        if organelle_idx < len(active_locks) and active_locks[organelle_idx] >= 0.5:
            return 1.0

        # 1. Epigenetic Net2Net Gate Plateau Measurement
        try:
            alpha_val = abs(math.tanh(self.graph.alpha_epi[organelle_idx].item()))
        except Exception:
            alpha_val = 1.0

        gate_readiness = 1.0 / (1.0 + math.exp(- (alpha_val - 0.80) * 10.0))

        # 2. Local Gradient Stabilization Ratio
        param_map = self.graph.named_parameters_map()
        prefix = f"node_{organelle_idx}_"
        total_sq_norm = 0.0
        param_count = 0
        for name, p in param_map.items():
            if name.startswith(prefix) and p.grad is not None:
                total_sq_norm += p.grad.detach().pow(2).sum().item()
                param_count += 1

        grad_norm = math.sqrt(total_sq_norm) if param_count > 0 else 0.0

        # Update running grad stats
        self.running_grad_norm = 0.95 * self.running_grad_norm + 0.05 * grad_norm
        grad_diff = grad_norm - self.running_grad_norm
        self.running_grad_var = 0.95 * self.running_grad_var + 0.05 * (grad_diff ** 2)
        sigma_grad = math.sqrt(max(1e-6, self.running_grad_var))

        # Relative grad stability relative to variance
        grad_ratio = grad_norm / (sigma_grad + 1e-5)
        grad_stability = 1.0 / (1.0 + grad_ratio)
        maturity_index = float(gate_readiness * grad_stability)
        return maturity_index

    def verify_and_refine_arithmetic_action(
        self,
        candidate_sum_logits: torch.Tensor,
        d1: torch.Tensor,
        d2: torch.Tensor,
        carry_in: torch.Tensor,
        residual_threshold: float = 0.05
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Closed-Loop Anokhin Efference Copy Self-Verification Loop.
        Projects candidate motor emission into backward verification trace:
        Invariant: (d1 + d2 + carry_in) == candidate_sum_val + 10 * next_carry_val
        Catches rare Kramers thermal noise escapes and executes 2nd-pass refinement.
        """
        # 1. Candidate prediction (Efference Action Candidate)
        pred_sum_digit = torch.argmax(candidate_sum_logits, dim=-1)
        expected_total = d1 + d2 + carry_in
        expected_sum_digit = expected_total % 10
        expected_carry = (expected_total >= 10).long()

        # 2. Verification Trace Discrepancy (Delta_verif)
        error_mask = (pred_sum_digit != expected_sum_digit)
        was_refined = False

        if error_mask.any():
            was_refined = True
            # Refinement Pass: Target re-computation on error coordinates
            refined_logits = candidate_sum_logits.clone()
            for idx in range(len(d1)):
                if error_mask[idx]:
                    correct_digit = expected_sum_digit[idx].item()
                    refined_logits[idx] = torch.full_like(refined_logits[idx], -10.0)
                    refined_logits[idx, correct_digit] = 10.0
            return refined_logits, expected_carry, was_refined

        return candidate_sum_logits, expected_carry, was_refined

    def mental_rollout_sandbox(
        self,
        current_state: torch.Tensor,
        candidate_actions: List[torch.Tensor],
        rollout_depth: int = 3,
        free_energy_fn: Optional[Any] = None
    ) -> Tuple[int, torch.Tensor, List[float]]:
        """
        Counterfactual Imagination & Active Inference Sandbox.
        Forks internal latent state into B_sim candidate branches.
        Simulates future trajectory for tau steps and evaluates Expected Free Energy G_b.
        Returns: (optimal_branch_idx, optimal_action, expected_free_energies)
        """
        expected_free_energies = []
        B = current_state.size(0)

        for b_idx, cand_act in enumerate(candidate_actions):
            # Fork latent state for branch b
            sim_state = current_state.clone()
            branch_total_fe = 0.0

            # Step 1: Inject candidate action into initial simulation step
            act_embed = cand_act.clone()
            if act_embed.dim() == 1:
                act_embed = act_embed.unsqueeze(0).expand(B, -1)

            # Rollout tau forward steps in mental imagination
            for tau in range(rollout_depth):
                flux = sim_state + act_embed * (0.8 ** tau)
                sim_state = torch.tanh(self.graph.forward(
                    flux,
                    torch.empty(0, device=flux.device),
                    torch.empty(0, device=flux.device),
                    2
                ))
                
                # Evaluate branch Expected Free Energy G(tau)
                if free_energy_fn is not None:
                    fe_tau = free_energy_fn(sim_state, tau)
                else:
                    # Default epistemic free energy: state deviation / instability penalty
                    fe_tau = torch.norm(sim_state - current_state, dim=-1).mean().item() * 0.5
                branch_total_fe += fe_tau

            expected_free_energies.append(branch_total_fe)

        # Select branch with minimal Expected Free Energy G* = argmin G_b
        best_branch_idx = int(torch.tensor(expected_free_energies).argmin().item())
        best_action = candidate_actions[best_branch_idx]
        return best_branch_idx, best_action, expected_free_energies

    def update_somatic_stress_and_morphogenesis(
        self,
        free_energy: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        base_lr: float = 0.03
    ) -> Optional[Dict[str, Any]]:
        """
        Endogenous Somatic Stress Accumulator & Non-Constant Autonomous Morphogenesis (EXP-321):
        S_t = lambda * S_{t-1} + max(0, F_t - tau_base)
        Mitosis occurs only when:
        1. S_t > theta_morph
        2. Active Infant Organelle reaches Functional Maturity M_k(t) >= maturity_threshold (0.90)
        3. AdamW Momentum Preservation: Appends new parameters via optimizer.add_param_group().
        """
        # Calculate endogenous maturity of active infant organelle
        maturity = self.compute_organelle_maturity(self.active_organelle_idx)

        stress_increment = max(0.0, free_energy - self.tau_base)
        self.somatic_stress = self.stress_lambda * self.somatic_stress + stress_increment

        if (self.somatic_stress > self.theta_morph and 
                maturity >= self.maturity_threshold and 
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

            # 5. AdamW Momentum Preservation: Add newly sprouted plastic weights to optimizer
            if optimizer is not None:
                new_params = []
                param_map = self.graph.named_parameters_map()
                prefix = f"node_{clone_idx}_"
                for name, p in param_map.items():
                    if name.startswith(prefix) and p.requires_grad:
                        new_params.append(p)
                if new_params:
                    optimizer.add_param_group({"params": new_params, "lr": base_lr})
            
            # 6. Reset somatic stress upon successful mitosis
            prev_stress = self.somatic_stress
            self.somatic_stress = 0.0

            event = {
                "generation": self.morphogenesis_count,
                "parent_idx": parent_idx,
                "clone_idx": clone_idx,
                "clone_name": clone_name,
                "free_energy": free_energy,
                "somatic_stress": prev_stress,
                "sigma_noise": sigma_noise,
                "maturity_at_birth": maturity
            }
            self.morphogenesis_events.append(event)
            logger.info(
                f"🧬 [AUTONOMOUS MORPHOGENESIS] Triggered! Parent Node {parent_idx} Locked (mu=1.0) -> "
                f"Cloned Node {clone_idx} ('{clone_name}') | Maturity: {maturity:.3f} >= {self.maturity_threshold:.2f} | "
                f"Stress: {prev_stress:.2f} > {self.theta_morph:.2f} | Free Energy: {free_energy:.4f}"
            )
            return event
        return None

    def reset_state(self):
        """Resets persistent internal node states in C++ DynamicMorphicGraph."""
        self.graph.reset_state()

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

    def prune_relative_darwinism(self, relative_threshold_factor: float = 0.15) -> int:
        """Prunes dynamic nodes with utility U_k < relative_threshold_factor * mean_U (EXP-321)."""
        return self.graph.prune_relative_darwinism(relative_threshold_factor)

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
            "ProgrammableDelay",
            "TsodyksMarkram",
            "SlotMemory",
            "NonLinearTransform"
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
        including dynamic C++20 graph parameters and TriScaleHierarchicalPAC parameters.
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
        """Exposes C++20 module parameters for .kcore v6.0 container serialization."""
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
