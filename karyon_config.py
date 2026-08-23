# karyon_config.py
"""
===============================================================================
KARYON MASTER CONFIGURATION REGISTRY v5.6
Unified Settings for Ultra-Fast Parallel Pre-Projected Cortical Engine (Q=64).
===============================================================================
"""

from dataclasses import dataclass, field, asdict
import json
import os
from typing import Dict, Any


@dataclass
class HomeostasisConfig:
    """Homeostatic Somatic Physiology and Neurotransmitter Dynamics."""
    energy_recovery_rate: float = 0.0012
    backprop_energy_cost: float = 0.0050
    idle_somatic_decay: float = 0.0005
    sleep_energy_recovery: float = 0.0050
    
    curiosity_setpoint: float = 0.80
    energy_setpoint: float = 1.00
    stability_setpoint: float = 1.00
    health_setpoint: float = 1.00
    
    curiosity_pain_weight: float = 1.00
    energy_pain_weight: float = 1.00
    stability_pain_weight: float = 1.20
    health_pain_weight: float = 1.50
    
    noradrenaline_surprise_weight: float = 0.80
    noradrenaline_arousal_persistence: float = 0.40
    dopamine_reward_scale: float = 2.00
    volitional_recall_gain: float = 2.00


@dataclass
class SDEConfig:
    """Continuous Langevin Differential Equation Core Settings."""
    gamma_drift: float = 0.10
    wiener_noise_sigma: float = 1e-3
    min_effective_dt: float = 0.20
    max_effective_dt: float = 2.50
    na_dt_compression_weight: float = 0.70
    da_dt_expansion_weight: float = 0.80


@dataclass
class NetworkConfig:
    """Multi-Layer Cortical Dimensions and Network Topologies."""
    text_dim: int = 128
    vision_dim: int = 256
    audio_dim: int = 256
    action_dim: int = 3
    cog_action_dim: int = 3
    homeo_dim: int = 6
    text_gen_dim: int = 258
    
    unified_dim: int = 256
    hidden_dim: int = 512
    latent_dim: int = 128
    
    # 2-Layer High-Velocity Cortical Stack (Q=64)
    num_layers: int = 2
    expand_dim: int = 1536
    num_heads: int = 8
    head_k: int = 32
    head_v: int = 64


@dataclass
class MemoryConfig:
    """Vectorized Episodic Memory & Volitional Read Gating Settings."""
    max_capacity: int = 1000
    protected_slots: int = 3
    default_read_threshold: float = 0.50
    default_attention_temp: float = 0.05
    sigmoid_gating_beta: float = 15.00
    pruning_similarity_threshold: float = 0.93
    
    volitional_na_trigger: float = 0.12
    volitional_fe_trigger: float = 0.25


@dataclass
class TrainConfig:
    """Training, Optimization, and Dynamic Plasticity Gating (DFET v3)."""
    batch_size: int = 32
    learning_rate: float = 2.5e-3
    min_learning_rate: float = 1e-4
    warmup_steps: int = 300
    weight_decay: float = 0.01
    grad_clip_norm: float = 2.0
    loss_free_energy_weight: float = 0.05
    loss_speech_weight: float = 1.00
    chunk_size: int = 64
    bptt_chunk_size: int = 256
    
    dfet_enabled: bool = True
    dfet_alpha_ma: float = 0.05
    dfet_k_sigma_base: float = 0.45
    dfet_k_sigma_na_weight: float = 0.25
    dfet_min_k_sigma: float = 0.15
    mastery_setpoint: float = 0.025
    speech_mastery_setpoint: float = 0.30


@dataclass
class CoREConfig:
    """Master Unified Config Container for Karyon Architecture."""
    homeo: HomeostasisConfig = field(default_factory=HomeostasisConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    net: NetworkConfig = field(default_factory=NetworkConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoREConfig":
        cfg = cls()
        if "homeo" in data: cfg.homeo = HomeostasisConfig(**data["homeo"])
        if "sde" in data: cfg.sde = SDEConfig(**data["sde"])
        if "net" in data: cfg.net = NetworkConfig(**data["net"])
        if "memory" in data: cfg.memory = MemoryConfig(**data["memory"])
        if "train" in data: cfg.train = TrainConfig(**data["train"])
        return cfg

    def save_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "CoREConfig":
        if not os.path.exists(filepath):
            return cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
