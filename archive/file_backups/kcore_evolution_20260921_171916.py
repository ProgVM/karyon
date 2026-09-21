# kcore_evolution.py
"""
===============================================================================
KARYON CORE SLEEP META-GENETICS & EPIGENETIC EVOLUTION ENGINE
===============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict
from typing import Dict, Any, Tuple, Optional, List

from karyon_config import CoREConfig, HomeostasisConfig, NetworkConfig, MemoryConfig
import karyon_core as kcore
from karyon_logger import get_logger

logger = get_logger()


# =============================================================================
# HELPER: SHAPE-ADAPTIVE PARAMETER ZERO-PAD COPY
# =============================================================================

def adapt_and_copy_tensor(target: torch.Tensor, source: torch.Tensor):
    """
    Copies source tensor into target tensor with shape-adaptive zero-padding or slicing.
    Supports Net2Net expansion without shape mismatch errors.
    """
    if target.shape == source.shape:
        target.copy_(source)
        return

    with torch.no_grad():
        slices = []
        for d_target, d_source in zip(target.shape, source.shape):
            slices.append(slice(0, min(d_target, d_source)))

        slices = tuple(slices)
        target.zero_()
        target[slices] = source[slices]


# =============================================================================
# SLEEP META-GENETICS ENGINE
# =============================================================================

class SleepMetaGeneticsEngine:
    """
    Spawns candidate genome mutations during sleep, evaluates variational surprise
    and Free Energy drop on an autonomous validation buffer, and selects the optimal genome.
    """

    @staticmethod
    def get_active_genome(agent: Any) -> Dict[str, Any]:
        """Extracts the live tunable biophysical genome from agent configs."""
        if hasattr(agent, 'config'):
            net_cfg = getattr(agent.config, 'net', None)
            homeo_cfg = getattr(agent.config, 'homeo', None)
            return {
                "min_beta_stage1": float(getattr(net_cfg, "min_beta_stage1", 0.005)) if net_cfg else 0.005,
                "max_beta_stage1": float(getattr(net_cfg, "max_beta_stage1", 0.15)) if net_cfg else 0.15,
                "min_beta_stage2": float(getattr(net_cfg, "min_beta_stage2", 0.0001)) if net_cfg else 0.0001,
                "max_beta_stage2": float(getattr(net_cfg, "max_beta_stage2", 0.05)) if net_cfg else 0.05,
                "hopfield_beta": float(getattr(net_cfg, "hopfield_beta", 12.0)) if net_cfg else 12.0,
                "pac_entropy_threshold": float(getattr(net_cfg, "pac_entropy_threshold", 0.70)) if net_cfg else 0.70,
                "noradrenaline_surprise_weight": float(getattr(homeo_cfg, "noradrenaline_surprise_weight", 0.85)) if homeo_cfg else 0.85,
                "dopamine_reward_scale": float(getattr(homeo_cfg, "dopamine_reward_scale", 2.00)) if homeo_cfg else 2.00,
                "volitional_recall_gain": float(getattr(homeo_cfg, "volitional_recall_gain", 2.00)) if homeo_cfg else 2.00
            }
        return {
            "min_beta_stage1": 0.005,
            "max_beta_stage1": 0.15,
            "min_beta_stage2": 0.0001,
            "max_beta_stage2": 0.05,
            "hopfield_beta": 12.0,
            "pac_entropy_threshold": 0.70,
            "noradrenaline_surprise_weight": 0.85,
            "dopamine_reward_scale": 2.00,
            "volitional_recall_gain": 2.00
        }

    @staticmethod
    def apply_genome_to_agent(agent: Any, genome: Dict[str, Any]):
        """Injects evolved genome parameters into live agent runtime."""
        if hasattr(agent, 'config'):
            net_cfg = getattr(agent.config, 'net', None)
            homeo_cfg = getattr(agent.config, 'homeo', None)
            for k, v in genome.items():
                if net_cfg and hasattr(net_cfg, k):
                    setattr(net_cfg, k, v)
                elif homeo_cfg and hasattr(homeo_cfg, k):
                    setattr(homeo_cfg, k, v)
