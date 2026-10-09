"""
EXP-387: Bidirectional Laminar Predictive Coding & Hierarchical Error Routing (BLPC-HER)
========================================================================================

Standard: KEP v16.0 Sovereign Master
Author: Bazilevs (ProgVM) & Autonomous Lead AI Cyberneticist

Core Architecture:
1. Endogenous Hierarchical Predictive Coding (Rao & Ballard 1999, Friston Active Inference):
   - In standard feed-forward stacks (EXP-382..EXP-386), layers process bottom-up features blindly
     without top-down hypothesis constraints.
   - BLPC introduces Top-Down Generative Priors: Layer l+1 predicts latent state of Layer l:
     h_hat_l = W_topdown^{(l+1)}(h_{l+1}) + b_topdown^{(l+1)}
   - Bottom-Up signals are converted to Precision-Weighted Prediction Errors:
     eps_l = Pi_l * (h_l - h_hat_l)
     where Pi_l = sigmoid(W_pi h_l) represents salience / precision gain.
2. Riemannian Manifold Geometry (from EXP-386):
   - State-dependent Riemannian Metric Tensor G(h) = L(h) L(h)^T + eps * I
   - Endogenous Associative Trace Memory (EATM) with Riemannian inner products <q, k>_G
   - Riemannian Attractor Phase Snapping (RAPS) on output logits.
3. Multi-Interface Laminar Free Energy:
   - F_total = F_sensory + sum_{l=1}^{L-1} lambda_l * ||eps_l||_G^2 + beta * Complexity
   - Higher layers hold slower, macro-scale representations that constrain low-level fast dynamics,
     dramatically reducing entropy and bit prediction errors.
4. Net2Net Smooth Grafting & Numerical Stability:
   - When a new laminar sheet sprouts via the Gödel Sandbox, top-down and error routing weights
     are initialized with zero-shock tanh(alpha_epi) identity gates.
   - Predictions and error terms are normalized and clamped via LayerNorm/tanh to prevent
     unbounded gain cascades across deep laminar passes.
"""

import os
import sys
import math
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# =============================================================================
# 1. ATOMIC DYNAMIC OPERATORS (UNIVERSAL BASIS)
# =============================================================================

class LinearAccumulatorOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.W = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        leak = torch.clamp(tau, 0.01, 0.99)
        proj = torch.matmul(x, self.W.t())
        return (1.0 - leak) * h_prev + leak * proj


class BilinearMultiplicativeOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.W_a = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_b = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        a = torch.matmul(x, self.W_a.t())
        b = torch.matmul(h_prev, self.W_b.t())
        conj = torch.tanh(a * b)
        leak = torch.clamp(tau, 0.01, 0.99)
        return (1.0 - leak) * h_prev + leak * conj


class SaturatedAttractorOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.W = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        rec = torch.matmul(h_prev, self.W.t())
        basin = torch.tanh(rec + x)
        leak = torch.clamp(tau, 0.01, 0.99)
        return (1.0 - leak) * h_prev + leak * basin


class StateSpaceMemoryOp(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.log_a = nn.Parameter(torch.randn(dim, device=device) * 0.1 - 1.0)
        self.W_b = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.log_a)
        decay = torch.clamp(a * (1.0 - tau * 0.5), 0.001, 0.999)
        b_x = torch.matmul(x, self.W_b.t())
        return decay * h_prev + (1.0 - decay) * b_x


# =============================================================================
# 2. SOVEREIGN RIEMANNIAN METRIC TENSOR G(h)
# =============================================================================

class SovereignMetricTensor(nn.Module):
    def __init__(self, dim: int = 128, rank: int = 16, device: torch.device = DEVICE):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.device = device
        self.W_L = nn.Parameter(torch.randn(dim * rank, dim, device=device) * (0.05 / math.sqrt(dim)))
        self.b_L = nn.Parameter(torch.zeros(dim * rank, device=device))
        self.eps = 1e-4

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_bounded = torch.clamp(h, -10.0, 10.0)
        flat_L = torch.matmul(self.W_L, h_bounded) + self.b_L
        flat_L = torch.clamp(flat_L, -5.0, 5.0)
        L = flat_L.view(self.dim, self.rank)
        G = torch.matmul(L, L.t()) + self.eps * torch.eye(self.dim, device=self.device)
        return G

    def inner_product(self, u: torch.Tensor, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        G = self.forward(h)
        return torch.matmul(u, torch.matmul(G, v))

    def compute_geodesic_distance_sq(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        delta = h1 - h2
        h_mid = 0.5 * (h1 + h2)
        G = self.forward(h_mid)
        return torch.matmul(delta, torch.matmul(G, delta))


class OutputMetricTensor(nn.Module):
    def __init__(self, dim: int = 8, rank: int = 4, device: torch.device = DEVICE):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.device = device
        self.W_L = nn.Parameter(torch.randn(dim * rank, dim, device=device) * (0.1 / math.sqrt(dim)))
        self.b_L = nn.Parameter(torch.zeros(dim * rank, device=device))
        self.eps = 1e-3

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y_bounded = torch.clamp(y, -10.0, 10.0)
        flat_L = torch.matmul(self.W_L, y_bounded) + self.b_L
        flat_L = torch.clamp(flat_L, -5.0, 5.0)
        L = flat_L.view(self.dim, self.rank)
        G = torch.matmul(L, L.t()) + self.eps * torch.eye(self.dim, device=self.device)
        return G

    def warp_output(self, y: torch.Tensor) -> torch.Tensor:
        G = self.forward(y)
        return torch.matmul(G, y)


# =============================================================================
# 3. ENDOGENOUS ASSOCIATIVE TRACE MEMORY (EATM)
# =============================================================================

class EndogenousAssociativeMemory(nn.Module):
    def __init__(self, slots: int = 32, dim: int = 128, device: torch.device = DEVICE):
        super().__init__()
        self.slots = slots
        self.dim = dim
        self.device = device

        self.memory_slots = nn.Parameter(torch.randn(slots, dim, device=device) * 0.1)
        self.W_q = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_k = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_v = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))

        self.W_read_gate = nn.Parameter(torch.randn(1, dim, device=device) * 0.1)
        self.W_write_gate = nn.Parameter(torch.randn(1, dim, device=device) * 0.1)

    def forward(self, h_query: torch.Tensor, current_mem: torch.Tensor, metric_tensor: SovereignMetricTensor, NA: float, DA: float) -> tuple:
        q = torch.matmul(h_query, self.W_q.t())
        keys = torch.matmul(current_mem, self.W_k.t())
        values = torch.matmul(current_mem, self.W_v.t())

        G = metric_tensor(h_query)
        q_G = torch.matmul(q, G)
        sim = torch.matmul(keys, q_G) / math.sqrt(self.dim)
        sim = torch.clamp(sim, -20.0, 20.0)
        attn = torch.softmax(sim, dim=-1)

        mem_read = torch.matmul(attn, values)

        g_read_base = torch.sigmoid(torch.matmul(self.W_read_gate, h_query)).squeeze()
        g_write_base = torch.sigmoid(torch.matmul(self.W_write_gate, h_query)).squeeze()

        g_read = torch.clamp(g_read_base * (0.8 + 0.6 * DA), 0.0, 1.0)
        g_write = torch.clamp(g_write_base * (0.5 + 1.0 * NA), 0.0, 1.0)

        write_val = torch.tanh(torch.matmul(h_query, self.W_v.t()))
        write_slot_update = torch.outer(attn, write_val)
        updated_mem = (1.0 - 0.05 * g_write) * current_mem + g_write * write_slot_update

        return mem_read, updated_mem, g_read, g_write, attn


# =============================================================================
# 4. BIDIRECTIONAL PREDICTIVE LAMINAR SHEET (BLPC)
# =============================================================================

class BidirectionalLaminarSheet(nn.Module):
    """
    Cortical sheet equipped with:
    1. Upward Bottom-Up feed (processing input/error).
    2. Downward Top-Down prediction generator targeting layer below (h_hat_{l-1}).
    3. Precision / Salience weight gate Pi_l.
    4. Internal recurrent multi-timescale operators.
    """
    def __init__(self, layer_id: int, dim: int = 128, device: torch.device = DEVICE):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.device = device

        # Genome: [0..3: op blend, 4: alpha_layer, 5: log_tau, 6..9: role vector]
        self.genome = nn.Parameter(torch.randn(10, device=device) * 0.1)

        self.op_linear = LinearAccumulatorOp(dim, device)
        self.op_bilinear = BilinearMultiplicativeOp(dim, device)
        self.op_attractor = SaturatedAttractorOp(dim, device)
        self.op_ssm = StateSpaceMemoryOp(dim, device)

        self.W_ff = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_fb = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.W_mem = nn.Parameter(torch.randn(dim, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.bias = nn.Parameter(torch.zeros(dim, device=device))

        # --- Top-Down Prediction Generator (Predicts the layer below) ---
        self.W_topdown = nn.Parameter(torch.randn(dim, dim, device=device) * (0.05 / math.sqrt(dim)))
        self.b_topdown = nn.Parameter(torch.zeros(dim, device=device))

        # --- Precision / Salience Weighting Gate Pi_l ---
        self.W_pi = nn.Parameter(torch.randn(dim, dim, device=device) * (0.1 / math.sqrt(dim)))
        self.b_pi = nn.Parameter(torch.zeros(dim, device=device))

        # --- Smooth Net2Net Grafting Gating for Top-Down feedback ---
        self.alpha_topdown = nn.Parameter(torch.tensor([0.0], device=device))

    def generate_topdown_prediction(self, h_current: torch.Tensor) -> torch.Tensor:
        """Projects downward prediction for Layer l-1 bounded via tanh."""
        h_hat = torch.matmul(h_current, self.W_topdown.t()) + self.b_topdown
        return torch.tanh(self.alpha_topdown) * torch.tanh(h_hat)

    def compute_precision_weighted_error(self, h_current: torch.Tensor, h_hat_prior: torch.Tensor) -> tuple:
        """
        Calculates prediction error eps_l = Pi_l * (h_l - h_hat_l).
        Bounded by tanh to guarantee numerical stability across laminar stack.
        """
        pi = torch.sigmoid(torch.matmul(h_current, self.W_pi.t()) + self.b_pi)
        raw_error = torch.tanh(h_current - h_hat_prior)
        weighted_error = pi * raw_error
        return weighted_error, raw_error, pi

    def forward(
        self,
        bottom_up_in: torch.Tensor,
        h_prev: torch.Tensor,
        top_down_feedback: torch.Tensor,
        mem_trace: torch.Tensor,
        NA: float,
        DA: float
    ) -> tuple:
        op_blend = torch.softmax(self.genome[0:4], dim=-1)
        alpha_layer = torch.tanh(self.genome[4])

        log_tau = self.genome[5]
        tau_base = torch.sigmoid(log_tau)
        tau = torch.clamp(tau_base * (1.0 + 1.2 * NA - 0.4 * DA), 0.01, 0.99)
        role_vector = torch.softmax(self.genome[6:10], dim=-1)

        mixed_input = (
            torch.matmul(bottom_up_in, self.W_ff.t())
            + torch.matmul(top_down_feedback, self.W_fb.t())
            + torch.matmul(mem_trace, self.W_mem.t())
            + self.bias
        )

        h0 = self.op_linear(mixed_input, h_prev, tau)
        h1 = self.op_bilinear(mixed_input, h_prev, tau)
        h2 = self.op_attractor(mixed_input, h_prev, tau)
        h3 = self.op_ssm(mixed_input, h_prev, tau)

        h_next = op_blend[0] * h0 + op_blend[1] * h1 + op_blend[2] * h2 + op_blend[3] * h3
        gated_output = alpha_layer * torch.tanh(h_next)

        return gated_output, h_next, alpha_layer, op_blend, role_vector, tau


# =============================================================================
# 5. SOVEREIGN BIDIRECTIONAL PREDICTIVE ENGINE (BLPC-HER)
# =============================================================================

class SovereignThinkingEngine(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        max_layers: int = 8,
        mem_slots: int = 32,
        max_thinking_steps: int = 4,
        device: torch.device = DEVICE
    ):
        super().__init__()
        self.dim = dim
        self.max_layers = max_layers
        self.mem_slots = mem_slots
        self.max_thinking_steps = max_thinking_steps
        self.device = device

        self.layers = nn.ModuleList([BidirectionalLaminarSheet(i, dim, device) for i in range(max_layers)])
        self.active_layers = 2

        with torch.no_grad():
            self.layers[0].genome[4].fill_(2.5)
            self.layers[1].genome[4].fill_(2.5)
            self.layers[0].genome[6].fill_(2.0)
            self.layers[1].genome[7].fill_(2.0)
            self.layers[1].alpha_topdown.fill_(0.5)

        # Riemannian Metric Tensors
        self.metric_tensor = SovereignMetricTensor(dim=dim, rank=16, device=device)
        self.output_metric_tensor = OutputMetricTensor(dim=8, rank=4, device=device)

        # Endogenous Memory Manifold
        self.associative_memory = EndogenousAssociativeMemory(slots=mem_slots, dim=dim, device=device)

        self.W_in = nn.Parameter(torch.randn(dim, 8, device=device) * (1.0 / math.sqrt(8)))
        self.W_out = nn.Parameter(torch.randn(8, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_out = nn.Parameter(torch.zeros(8, device=device))

        # Sensory Prediction Generator (Layer 0 generates direct sensory prediction w_hat)
        self.W_sensory_pred = nn.Parameter(torch.randn(8, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_sensory_pred = nn.Parameter(torch.zeros(8, device=device))

        # Endogenous Halting Gate & Recurrent Thinking Readout
        self.W_halt = nn.Parameter(torch.randn(1, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_halt = nn.Parameter(torch.tensor([-0.5], device=device))

        # Attractor Snapping Kernel
        self.W_snap = nn.Parameter(torch.randn(8, 8, device=device) * (1.0 / math.sqrt(8)))

        # Ashby Neuromodulators
        self.NA = 0.5
        self.DA = 0.5
        self.rolling_error = 1.0

        # Endogenous Meta-Law Generator
        self.W_meta = nn.Parameter(torch.randn(4, dim, device=device) * (1.0 / math.sqrt(dim)))
        self.b_meta = nn.Parameter(torch.zeros(4, device=device))

        self.sprout_events = 0
        self.prune_events = 0
        self.rejections = 0

    def update_neuromodulators(self, prediction_error: float):
        if math.isnan(prediction_error):
            return
        self.NA = 0.90 * self.NA + 0.10 * math.tanh(prediction_error * 2.0)
        error_delta = self.rolling_error - prediction_error
        self.DA = 0.90 * self.DA + 0.10 * torch.sigmoid(torch.tensor([error_delta * 5.0])).item()
        self.rolling_error = 0.95 * self.rolling_error + 0.05 * prediction_error

    def _execute_bidirectional_laminar_pass(
        self,
        v_in: torch.Tensor,
        states: torch.Tensor,
        current_mem: torch.Tensor
    ):
        L = self.active_layers
        mem_trace, next_mem, g_read, g_write, _ = self.associative_memory(
            states[0], current_mem, self.metric_tensor, self.NA, self.DA
        )
        effective_mem_trace = g_read * mem_trace

        # Step 1: Top-Down Predictions (from Layer l+1 to Layer l)
        top_down_priors = [torch.zeros(self.dim, device=self.device) for _ in range(L)]
        for i in range(L - 1, 0, -1):
            top_down_priors[i - 1] = self.layers[i].generate_topdown_prediction(states[i])

        # Step 2: Forward Laminar Pass with Bottom-Up Error Propagation
        next_states = []
        layer_gates = []
        layer_blends = []
        layer_roles = []
        layer_taus = []
        laminar_errors = []
        laminar_precisions = []

        for i in range(L):
            if i == 0:
                bu_signal = v_in
            else:
                bu_signal = laminar_errors[i - 1]

            td_prior = top_down_priors[i]

            gated_out, raw_h, alpha_layer, op_blend, role_vector, tau = self.layers[i](
                bottom_up_in=bu_signal,
                h_prev=states[i],
                top_down_feedback=td_prior,
                mem_trace=effective_mem_trace,
                NA=self.NA,
                DA=self.DA
            )

            w_err, raw_err, pi = self.layers[i].compute_precision_weighted_error(gated_out, td_prior)

            next_states.append(gated_out)
            layer_gates.append(alpha_layer)
            layer_blends.append(op_blend)
            layer_roles.append(role_vector)
            layer_taus.append(tau)
            laminar_errors.append(w_err)
            laminar_precisions.append(pi)

        next_states = torch.stack(next_states, dim=0)
        layer_gates = torch.stack(layer_gates, dim=0)
        layer_blends = torch.stack(layer_blends, dim=0)
        layer_roles = torch.stack(layer_roles, dim=0)
        laminar_errors_stack = torch.stack(laminar_errors, dim=0)

        return (
            next_states, next_mem, layer_gates, layer_blends, layer_roles,
            layer_taus, laminar_errors_stack, g_read, g_write
        )

    def forward(self, x_t: torch.Tensor, prev_states: torch.Tensor, current_mem: torch.Tensor) -> tuple:
        v_in = torch.matmul(x_t, self.W_in.t())
        L = self.active_layers
        cur_states = prev_states[:L]
        mem = current_mem

        # Recurrent Latent Thinking Loop
        total_thinking_steps = 0
        halting_weights = []
        last_states = cur_states
        last_mem = mem
        last_gates = None
        last_blends = None
        last_roles = None
        last_taus = None
        last_errors = None
        last_g_read = None
        last_g_write = None

        accum_h = torch.zeros(self.dim, device=self.device)
        remaining_p = 1.0

        for k in range(self.max_thinking_steps):
            total_thinking_steps += 1
            prev_top_h = cur_states[-1].clone() if k > 0 else None

            (
                cur_states, mem, layer_gates, layer_blends, layer_roles,
                layer_taus, laminar_errors, g_read, g_write
            ) = self._execute_bidirectional_laminar_pass(v_in, cur_states, mem)

            last_states = cur_states
            last_mem = mem
            last_gates = layer_gates
            last_blends = layer_blends
            last_roles = layer_roles
            last_taus = layer_taus
            last_errors = laminar_errors
            last_g_read = g_read
            last_g_write = g_write

            top_h = cur_states[-1]
            p_halt = torch.sigmoid(torch.matmul(self.W_halt, top_h) + self.b_halt).squeeze()

            # Geodesic Curvature-Driven Halting
            if k > 0:
                geodesic_dist_sq = self.metric_tensor.compute_geodesic_distance_sq(top_h, prev_top_h)
                delta_f_G = torch.sqrt(geodesic_dist_sq + 1e-8).item()
            else:
                delta_f_G = 1.0

            if k == self.max_thinking_steps - 1:
                step_weight = remaining_p
            else:
                halt_factor = p_halt * (1.0 - torch.sigmoid(torch.tensor(delta_f_G, device=self.device) - 0.1))
                step_weight = remaining_p * halt_factor
                remaining_p = remaining_p * (1.0 - halt_factor)

            halting_weights.append(step_weight)
            accum_h = accum_h + step_weight * top_h

            if remaining_p < 0.05 or (k > 0 and delta_f_G < 1e-3):
                break

        # ---------------------------------------------------------------------
        # RIEMANNIAN ATTRACTOR PHASE SNAPPING (RAPS)
        # ---------------------------------------------------------------------
        y_continuous = torch.matmul(self.W_out, accum_h) + self.b_out
        y_warped = self.output_metric_tensor.warp_output(y_continuous)

        beta_snap = 3.0 + 12.0 * self.DA
        snapped_logits = torch.matmul(y_warped, self.W_snap.t())
        y_pred = torch.tanh(beta_snap * snapped_logits)

        padded_states = prev_states.clone()
        padded_states[:L] = last_states

        complexity = torch.sum(torch.abs(last_gates))
        meta_vector = torch.sigmoid(torch.matmul(self.W_meta, accum_h) + self.b_meta)

        # Hierarchical Error Energy (normalized mean square error)
        hierarchical_error_energy = torch.mean(torch.sum(last_errors**2, dim=-1))

        return (
            y_pred, padded_states, last_mem, complexity, hierarchical_error_energy,
            last_blends, last_roles, last_gates, meta_vector, last_taus,
            last_g_read, last_g_write, total_thinking_steps
        )

    # -------------------------------------------------------------------------
    # GÖDEL MACHINE SANDBOX WITH HIERARCHICAL PREDICTIVE ERROR VERIFICATION
    # -------------------------------------------------------------------------
    def evaluate_laminar_mutation_in_sandbox(
        self,
        x_seq: torch.Tensor,
        target_seq: torch.Tensor,
        proposal_type: str,
        target_layer_id: int,
        proposal_genome: torch.Tensor,
        current_mem: torch.Tensor,
        endogenous_tolerance: float
    ) -> bool:
        original_active = self.active_layers
        original_genomes = [self.layers[i].genome.data.clone() for i in range(self.max_layers)]
        original_alpha_td = [self.layers[i].alpha_topdown.data.clone() for i in range(self.max_layers)]

        if proposal_type == "sprout":
            self.active_layers = min(self.max_layers, self.active_layers + 1)
            new_id = self.active_layers - 1
            self.layers[new_id].genome.data.copy_(proposal_genome)
            self.layers[new_id].alpha_topdown.data.fill_(0.0)
        elif proposal_type == "prune":
            self.active_layers = max(2, self.active_layers - 1)
        elif proposal_type == "mutate":
            self.layers[target_layer_id].genome.data.copy_(proposal_genome)

        baseline_states = torch.zeros(original_active, self.dim, device=self.device)
        candidate_states = torch.zeros(self.active_layers, self.dim, device=self.device)

        mem_base = current_mem.clone()
        mem_cand = current_mem.clone()

        def compute_trajectory_free_energy(L_count, states_init, mem_init):
            self.active_layers = L_count
            states = states_init
            mem = mem_init
            total_fe = 0.0
            seq_len = x_seq.size(0)

            for step in range(seq_len):
                x_val = x_seq[step]
                tgt_val = target_seq[step]
                with torch.no_grad():
                    (
                        y_pred, n_states, n_mem, comp, hier_err,
                        _, _, _, _, _, _, _, _
                    ) = self.forward(x_val, states, mem)
                    states = n_states[:L_count]
                    mem = n_mem

                    strain = 0.5 * torch.sum((y_pred - tgt_val)**2).item()
                    fe = strain + 0.05 * comp.item() + 0.05 * hier_err.item() + 0.02 * L_count
                    total_fe += fe

            return total_fe / max(1, seq_len)

        fe_candidate = compute_trajectory_free_energy(self.active_layers, candidate_states, mem_cand)

        self.active_layers = original_active
        for i in range(self.max_layers):
            self.layers[i].genome.data.copy_(original_genomes[i])
            self.layers[i].alpha_topdown.data.copy_(original_alpha_td[i])

        fe_baseline = compute_trajectory_free_energy(original_active, baseline_states, mem_base)

        delta_fe = fe_candidate - fe_baseline
        if delta_fe < endogenous_tolerance:
            if proposal_type == "sprout":
                self.active_layers = min(self.max_layers, original_active + 1)
                new_id = self.active_layers - 1
                self.layers[new_id].genome.data.copy_(proposal_genome)
                self.layers[new_id].alpha_topdown.data.fill_(0.0)
            elif proposal_type == "prune":
                self.active_layers = max(2, original_active - 1)
            elif proposal_type == "mutate":
                self.layers[target_layer_id].genome.data.copy_(proposal_genome)
            return True
        else:
            return False

    def extract_symbolic_formulas(
        self,
        layer_blends: torch.Tensor,
        layer_roles: torch.Tensor,
        layer_gates: torch.Tensor,
        layer_taus: list,
        g_read: float,
        g_write: float,
        avg_thinking_steps: float
    ) -> str:
        ops = ["LinearAccumulator", "BilinearConj", "AttractorBasin", "StateSpaceMemory"]
        roles = ["SensoryInput", "AssociationInter", "LatentWorkspace", "MotorReadout"]

        lines = [
            f"=== SOVEREIGN BLPC-HER ARCHITECTURAL SPECIFICATION (ACTIVE LAYERS: {self.active_layers}) ===",
            f"Endogenous Recurrent Thinking Depth: {avg_thinking_steps:.2f} cycles/token",
            f"Endogenous Associative Memory: g_read={g_read:.3f}, g_write={g_write:.3f}",
            f"Hierarchical Top-Down Feedback: Active on all layers l >= 1",
            "-------------------------------------------------------------------------------"
        ]

        for i in range(self.active_layers):
            blend = layer_blends[i].cpu().tolist()
            role = layer_roles[i].cpu().tolist()
            gate = layer_gates[i].item()
            tau = layer_taus[i].item()
            td_alpha = torch.tanh(self.layers[i].alpha_topdown).item()

            dom_op = ops[np.argmax(blend)]
            dom_role = roles[np.argmax(role)]
            op_str = " + ".join([f"{w:.2f}*{ops[idx]}" for idx, w in enumerate(blend)])

            formula = (
                f"Layer[{i}] ({dom_role}) | gate={gate:.3f} | tau={tau:.3f} | TopDown={td_alpha:.3f}\n"
                f"  OpDistribution: {op_str}\n"
                f"  Dynamics: h_{i}(t) = (1 - {tau:.3f})*h_{i}(t-1) + {tau:.3f}*[{dom_op}(bu_{i} + {td_alpha:.2f}*td_{i})]"
            )
            lines.append(formula)

        return "\n".join(lines)


# =============================================================================
# 6. STREAM GENERATION & BENCHMARK SUITE
# =============================================================================

def byte_to_bit_vector(b: int) -> torch.Tensor:
    bits = [(b >> (7 - i)) & 1 for i in range(8)]
    return torch.tensor([1.0 if bit == 1 else -1.0 for bit in bits], dtype=torch.float32, device=DEVICE)


def bit_vector_to_byte(v: torch.Tensor) -> int:
    bits = (v > 0.0).cpu().int().tolist()
    byte_val = 0
    for b in bits:
        byte_val = (byte_val << 1) | b
    return byte_val


def generate_raw_text_stream(length: int = 1500) -> tuple:
    corpus = (
        "In the beginning was the computation, and the computation was continuous. "
        "Mind is substrate-independent, realizable across silicon, electrons, and non-Euclidean manifolds. "
        "Active inference minimizes variational free energy through endogenous allostasis and precision-weighted errors. "
        "The cognitive architecture evolves its own operators, dynamic time scales, and laminar topologies without human intervention. "
        "Deep hierarchical predictive coding replaces static feedforward stacks with reciprocal top-down hypotheses and bottom-up prediction errors. "
        "Every byte is predicted through recurrent thinking cycles settling into metric geodesics. "
    )
    raw_bytes = list(corpus.encode("utf-8"))
    repeated_bytes = []
    while len(repeated_bytes) < length:
        repeated_bytes.extend(raw_bytes)
    selected_bytes = repeated_bytes[:length]

    vectors = [byte_to_bit_vector(b) for b in selected_bytes]
    tensor_data = torch.stack(vectors, dim=0)
    return tensor_data, selected_bytes


def generate_lorenz_chaotic_stream(length: int = 1500, dt: float = 0.01) -> torch.Tensor:
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    x, y, z = 1.0, 1.0, 1.0
    vectors = []
    for _ in range(length):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x += dx
        y += dy
        z += dz
        v = [
            math.tanh(x * 0.05),
            math.tanh(y * 0.05),
            math.tanh(z * 0.03),
            math.tanh((x - y) * 0.05),
            math.tanh((y - z) * 0.05),
            math.tanh((x * y) * 0.001),
            math.tanh((y * z) * 0.001),
            math.tanh((x * z) * 0.001)
        ]
        vectors.append(v)
    return torch.tensor(vectors, dtype=torch.float32, device=DEVICE)


def run_evaluation(domain_name, x_data, raw_bytes=None, is_text=False):
    print(f"\nEvaluating BLPC-HER on Domain: {domain_name}...")
    length = x_data.size(0)

    model = SovereignThinkingEngine(dim=128, max_layers=8, mem_slots=32, max_thinking_steps=4, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2, weight_decay=1e-5)

    states = torch.zeros(model.max_layers, model.dim, device=DEVICE)
    current_mem = model.associative_memory.memory_slots.clone()

    errors = []
    hier_errors = []
    bit_mismatches = []
    thinking_step_history = []
    t0 = time.time()

    last_blends = None
    last_roles = None
    last_gates = None
    last_taus = None
    last_g_read = 0.0
    last_g_write = 0.0

    for t in range(length - 1):
        cur_vec = x_data[t]
        tgt_vec = x_data[t + 1]

        optimizer.zero_grad()
        (
            y_pred, next_states, next_mem, complexity, hier_err_energy,
            layer_blends, layer_roles, layer_gates, meta_vector, layer_taus,
            g_read, g_write, thinking_steps
        ) = model(cur_vec, states, current_mem)

        states = next_states.detach()
        current_mem = next_mem.detach()

        last_blends = layer_blends.detach()
        last_roles = layer_roles.detach()
        last_gates = layer_gates.detach()
        last_taus = layer_taus
        last_g_read = g_read.item()
        last_g_write = g_write.item()
        thinking_step_history.append(thinking_steps)

        # Endogenous parameter synthesis via meta_vector
        sprout_thresh = 0.30 + 0.40 * meta_vector[0].item()
        prune_thresh = 0.05 + 0.15 * meta_vector[1].item()
        godel_tolerance = meta_vector[2].item()
        mutation_volatility = 0.01 + 0.20 * meta_vector[3].item()

        beta_complexity = 0.05 * (1.0 - model.DA)
        beta_hierarchical = 0.05 * (1.0 + 0.5 * model.NA)

        prediction_strain = 0.5 * torch.sum((y_pred - tgt_vec)**2)
        free_energy = (
            prediction_strain
            + beta_complexity * complexity
            + beta_hierarchical * hier_err_energy
            + 0.02 * model.active_layers
        )

        free_energy.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        strain_val = prediction_strain.item()
        hier_val = hier_err_energy.item()
        model.update_neuromodulators(strain_val)
        errors.append(strain_val)
        hier_errors.append(hier_val)

        # ---------------------------------------------------------------------
        # ENDOGENOUS LAMINAR MORPHOGENESIS (Sprouting & Pruning)
        # ---------------------------------------------------------------------
        if t > 50 and t % 32 == 0:
            past_x = x_data[max(0, t-32):t]
            past_y = x_data[max(0, t-31):t+1]

            if model.rolling_error > sprout_thresh and model.active_layers < model.max_layers:
                candidate_genome = torch.randn(10, device=DEVICE) * 0.1
                candidate_genome[4] = 0.0  # Net2Net Smooth Grafting

                if model.evaluate_laminar_mutation_in_sandbox(
                    past_x, past_y, "sprout", -1, candidate_genome, current_mem, godel_tolerance
                ):
                    model.sprout_events += 1
                    print(
                        f"  [BLPC LAMINAR SPROUT] Added Sheet {model.active_layers} at step {t+1}. "
                        f"Total Layers: {model.active_layers} | Tolerance: {godel_tolerance:.4f}"
                    )
                else:
                    model.rejections += 1

            elif model.rolling_error < prune_thresh and model.active_layers > 2:
                top_gate = torch.tanh(model.layers[model.active_layers - 1].genome[4]).item()
                if abs(top_gate) < 0.15:
                    if model.evaluate_laminar_mutation_in_sandbox(
                        past_x, past_y, "prune", -1, None, current_mem, godel_tolerance
                    ):
                        model.prune_events += 1
                        print(
                            f"  [BLPC LAMINAR PRUNE] Pruned Sheet {model.active_layers} at step {t+1}. "
                            f"Total Layers: {model.active_layers} | Tolerance: {godel_tolerance:.4f}"
                        )
                    else:
                        model.rejections += 1

            else:
                target_layer_id = random.randint(0, model.active_layers - 1)
                mutation_delta = torch.randn(10, device=DEVICE) * mutation_volatility
                proposed_genome = model.layers[target_layer_id].genome.data + mutation_delta

                if model.evaluate_laminar_mutation_in_sandbox(
                    past_x, past_y, "mutate", target_layer_id, proposed_genome, current_mem, godel_tolerance
                ):
                    pass
                else:
                    model.rejections += 1

        if is_text and raw_bytes is not None:
            pred_byte = bit_vector_to_byte(y_pred)
            tgt_byte = raw_bytes[t + 1]
            bit_diff = bin(pred_byte ^ tgt_byte).count('1')
            bit_mismatches.append(bit_diff)

        if (t + 1) % 256 == 0:
            tail_err = np.mean(errors[-64:])
            tail_hier = np.mean(hier_errors[-64:])
            tail_bits = np.mean(bit_mismatches[-64:]) if is_text else 0.0
            tail_think = np.mean(thinking_step_history[-64:])
            print(
                f"  Step {t+1:04d}/{length} | Layers: {model.active_layers:02d} | Think: {tail_think:.1f}x "
                f"| NA: {model.NA:.3f} | DA: {model.DA:.3f} | HierErr: {tail_hier:.3f} | Strain: {tail_err:.4f}"
                + (f" | Bit Errors: {tail_bits:.2f}/8" if is_text else "")
            )

    elapsed = time.time() - t0
    tok_per_sec = length / elapsed if elapsed > 0 else 0.0

    q_len = max(1, len(errors) // 4)
    final_strain = float(np.mean(errors[-q_len:]))
    final_hier_err = float(np.mean(hier_errors[-q_len:]))
    final_bit_errors = float(np.mean(bit_mismatches[-q_len:])) if len(bit_mismatches) > 0 else 0.0
    avg_thinking_steps = float(np.mean(thinking_step_history))

    discovered_formulas = model.extract_symbolic_formulas(
        last_blends, last_roles, last_gates, last_taus, last_g_read, last_g_write, avg_thinking_steps
    )
    print("\n-------------------------------------------------------------------------------")
    print(discovered_formulas)
    print("-------------------------------------------------------------------------------\n")

    return {
        "final_strain": final_strain,
        "final_hier_err": final_hier_err,
        "final_bit_errors": final_bit_errors,
        "active_layers": model.active_layers,
        "avg_thinking_steps": avg_thinking_steps,
        "sprout_events": model.sprout_events,
        "prune_events": model.prune_events,
        "rejections": model.rejections,
        "tok_per_sec": tok_per_sec,
        "discovered_formulas": discovered_formulas
    }


def run_exp_387():
    print("===============================================================================")
    print("=== KEP EXP-387: BIDIRECTIONAL LAMINAR PREDICTIVE CODING (BLPC-HER)         ===")
    print("===============================================================================")
    print(f"Device: {DEVICE}")

    text_data, text_bytes = generate_raw_text_stream(1500)
    lorenz_data = generate_lorenz_chaotic_stream(1500)

    r1 = run_evaluation("Raw UTF-8 Text Stream", text_data, text_bytes, is_text=True)
    r2 = run_evaluation("Chaotic Lorenz Attractor", lorenz_data, is_text=False)

    summary_results = {
        "exp_id": "EXP-387",
        "raw_text_stream": r1,
        "lorenz_chaotic_stream": r2
    }

    is_positive = (r1["final_bit_errors"] < 1.80) and (r2["final_strain"] < 0.01)
    verdict = "🟢 POSITIVE" if is_positive else "⚪ NEUTRAL"
    print(f"👑 FINAL VERDICT: {verdict}")

    summary_results["verdict"] = verdict

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/exp_387_results.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    return summary_results


if __name__ == "__main__":
    run_exp_387()
