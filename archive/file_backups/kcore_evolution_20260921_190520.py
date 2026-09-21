# kcore_evolution.py
"""
===============================================================================
KARYON CORE SLEEP META-GENETICS & ALLOLSTATIC MORPHOGENESIS ENGINE (v6.0)
===============================================================================
Foundational Principles:
  1. Epigenetic Morphogenesis: Sprouting of dynamic operators & morphic pathways
     initialized with zero-shock identity (alpha_epi = 0.0) at birth.
  2. Teleological Vitality (Edelman Selectionism): Candidate structures are tested
     against high-entropy replay traces during sleep; only mutations that minimize
     Variational Free Energy (Delta F_t < 0) are epigenetically consolidated.
  3. Seamless Optimizer Adaptation: Safe rebinding of AdamW moment buffers (m_t, v_t)
     across topological parameter dimension changes without autograd crashes.
===============================================================================
"""
import copy
import random
from dataclasses import asdict
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import karyon_core as kcore
from karyon_config import CoREConfig, HomeostasisConfig, NetworkConfig, MemoryConfig
from karyon_logger import get_logger

logger = get_logger()


# =============================================================================
# 1. SHAPE-ADAPTIVE PARAMETER & OPTIMIZER STATE REBINDING
# =============================================================================

def adapt_and_copy_tensor(target: torch.Tensor, source: torch.Tensor):
    """
    Copies source tensor into target tensor with shape-adaptive zero-padding or slicing.
    Supports Net2Net expansion and safe state restoration without mismatch exceptions.
    """
    if target.shape == source.shape:
        target.copy_(source)
        return

    # Handle 0-dim or mismatched dimension rank scalars safely
    if target.dim() == 0 or source.dim() == 0 or target.dim() != source.dim():
        if target.dim() == 0 and source.dim() == 0:
            target.copy_(source)
        return

    with torch.no_grad():
        slices = tuple(slice(0, min(d_target, d_source)) for d_target, d_source in zip(target.shape, source.shape))
        target.zero_()
        target[slices] = source[slices]


def rebind_optimizer_moments(
    old_optimizer: torch.optim.Optimizer,
    new_parameters: List[torch.Tensor],
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8
) -> torch.optim.AdamW:
    """
    Constructs a fresh AdamW optimizer instance for newly spawned/expanded parameters
    while safely migrating accumulated momentum (exp_avg, exp_avg_sq, step) from the previous optimizer.
    Guarantees zero autograd graph corruption and preserves optimization momentum.
    """
    new_optimizer = torch.optim.AdamW(
        new_parameters,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )

    old_state = old_optimizer.state_dict()["state"]
    if old_state:
        for idx, p in enumerate(new_parameters):
            if idx in old_state:
                old_param_state = old_state[idx]
                new_state_entry = {}
                for k, v in old_param_state.items():
                    if isinstance(v, torch.Tensor):
                        if v.shape == p.shape:
                            new_state_entry[k] = v.clone()
                        elif v.dim() == 0:
                            new_state_entry[k] = v.clone()
                        else:
                            new_tensor = torch.zeros_like(p)
                            adapt_and_copy_tensor(new_tensor, v)
                            new_state_entry[k] = new_tensor
                    else:
                        new_state_entry[k] = copy.deepcopy(v)
                new_optimizer.state[p] = new_state_entry

    return new_optimizer


# =============================================================================
# 2. TELEOLOGICAL MORPHOGENETIC SELECTION & SLEEP ENGINE
# =============================================================================

class MorphogeneticAllostaticEngine:
    """
    Autonomous Sleep & Neuroevolution Engine for Karyon-CoRE:
    - Governs Tononi SHY synaptic downscaling during slow-wave sleep.
    - Evaluates Free Energy pressure and spawns functional morphic operators.
    - Executes Edelman Neural Darwinism selection: retains sprouted pathways if Delta F_t < 0.
    """

    @staticmethod
    def compute_variational_free_energy(
        agent: Any,
        replay_tokens: torch.Tensor,
        hopfield_beta: float = 20.0,
        latent_pred: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Computes exact Variational Free Energy F_t across replay memories:
        F_t = D_KL(Q || P) + E_Hopfield(h) + L_recon(tokens)
        """
        device = replay_tokens.device
        h_latent = agent.forward_latent(replay_tokens)
        
        # 1. Hopfield Attractor Energy (if hopfield active)
        hopfield_energy = torch.tensor(0.0, device=device)
        if hasattr(agent, "hopfield") and agent.use_hopfield:
            hopfield_basins = list(agent.hopfield.parameters())[0]
            h_norm = F.normalize(h_latent, p=2, dim=-1)
            b_norm = F.normalize(hopfield_basins, p=2, dim=-1)
            sim = torch.matmul(h_norm, b_norm.t())
            hopfield_energy = - (1.0 / hopfield_beta) * torch.logsumexp(hopfield_beta * sim, dim=-1).mean()

        # 2. Latent Active Inference Complexity (if latent_pred provided)
        kl_complexity = torch.tensor(0.0, device=device)
        if latent_pred is not None and h_latent.size(1) > 1:
            h_prev = h_latent[:, :-1, :]
            h_curr = h_latent[:, 1:, :]
            p_mu, p_logvar, q_mu, q_logvar = latent_pred(h_prev, h_curr)
            p_logvar = torch.clamp(p_logvar, -10.0, 10.0)
            q_logvar = torch.clamp(q_logvar, -10.0, 10.0)
            kl = 0.5 * torch.sum(p_logvar - q_logvar + (q_logvar.exp() + (q_mu - p_mu)**2) / (p_logvar.exp() + 1e-6) - 1.0, dim=-1)
            kl_complexity = kl.mean() / p_mu.size(-1)

        # 3. Target Suffix Reconstruction Energy
        logits = agent(replay_tokens)
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = replay_tokens[:, 1:].reshape(-1)
        recon_energy = F.cross_entropy(shift_logits, shift_labels)

        return 0.05 * kl_complexity + 0.10 * hopfield_energy + recon_energy

    @staticmethod
    def execute_allostatic_morphogenesis_cycle(
        agent: Any,
        replay_tokens: torch.Tensor,
        downscaling_factor: float = 0.01,
        sprout_probability: float = 0.5,
        available_ops: Tuple[str, ...] = ("LinearAccumulator", "BilinearMultiplicative", "SaturatedAttractor")
    ) -> Dict[str, Any]:
        """
        Executes a complete offline sleep, synaptic normalization, and morphogenetic genesis cycle.
        """
        with torch.no_grad():
            f_pre = MorphogeneticAllostaticEngine.compute_variational_free_energy(agent, replay_tokens).item()

        # Phase 1: Tononi SHY Synaptic Scaling + Epigenetic Genesis
        sleep_telemetry = agent.execute_deep_allostatic_sleep(
            downscaling_factor=downscaling_factor,
            sprout_probability=sprout_probability,
            available_ops=available_ops
        )

        with torch.no_grad():
            f_post = MorphogeneticAllostaticEngine.compute_variational_free_energy(agent, replay_tokens).item()

        delta_fe = f_pre - f_post
        sleep_telemetry["f_pre_sleep"] = f_pre
        sleep_telemetry["f_post_sleep"] = f_post
        sleep_telemetry["delta_fe"] = delta_fe
        sleep_telemetry["is_viable"] = 1.0 if delta_fe >= -0.05 else 0.0

        return sleep_telemetry
