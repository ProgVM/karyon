# karyon_config.py
"""
===============================================================================
KARYON MASTER CONFIGURATION REGISTRY v18.5
Unified Biophysical Settings for Dynamic Allostasis, SWR Replay, and Active Inference.
===============================================================================
"""

from dataclasses import dataclass, field, asdict
import json
import os
from typing import Dict, Any


@dataclass
class HomeostasisConfig:
    """Dynamic Allostatic Somatic Physiology and Neurotransmitter Kinetics."""
    energy_recovery_rate: float = 0.0025
    perceptive_rest_recovery: float = 0.0040
    backprop_energy_cost: float = 0.0050
    motor_speech_cost_per_patch: float = 0.0015
    idle_somatic_decay: float = 0.0005
    sleep_energy_recovery: float = 0.0080
    
    curiosity_setpoint: float = 0.80
    energy_setpoint: float = 1.00
    stability_setpoint: float = 1.00
    health_setpoint: float = 1.00
    
    curiosity_pain_weight: float = 1.00
    energy_pain_weight: float = 1.00
    stability_pain_weight: float = 1.20
    health_pain_weight: float = 1.50
    
    noradrenaline_surprise_weight: float = 0.85
    noradrenaline_arousal_persistence: float = 0.35
    dopamine_reward_scale: float = 2.00
    volitional_recall_gain: float = 2.00
    
    allostatic_fatigue_threshold: float = 0.25
    allostatic_sleep_trigger: float = 0.20
    wake_replay_frequency_steps: int = 50


@dataclass
class SDEConfig:
    """Multi-Timescale State-Space Duality and Langevin Fluctuations."""
    gamma_drift: float = 0.10
    wiener_noise_sigma: float = 1e-3
    min_effective_dt: float = 0.30
    max_effective_dt: float = 2.00
    na_dt_compression_weight: float = 0.40
    da_dt_expansion_weight: float = 0.40


@dataclass
class NetworkConfig:
    """Unshackled Cortical Microcircuit Dimensions (KEP Principle 2 & 7)."""
    text_dim: int = 256
    vision_dim: int = 256
    audio_dim: int = 256
    binary_dim: int = 256
    telepathic_dim: int = 256
    action_dim: int = 3
    cog_action_dim: int = 3
    homeo_dim: int = 6
    text_gen_dim: int = 258
    
    unified_dim: int = 256
    hidden_dim: int = 768
    latent_dim: int = 128
    expand_dim: int = 3072
    num_heads: int = 12
    head_k: int = 64
    head_v: int = 128
    num_attractors: int = 256
    max_seq_len: int = 8192


@dataclass
class MemoryConfig:
    """Vectorized Episodic Memory & Volitional Read Gating Settings."""
    max_capacity: int = 1000
    protected_slots: int = 3
    default_read_threshold: float = 0.70
    default_attention_temp: float = 0.05
    sigmoid_gating_beta: float = 15.00
    pruning_similarity_threshold: float = 0.93
    
    volitional_na_trigger: float = 0.12
    volitional_fe_trigger: float = 0.20


@dataclass
class TrainConfig:
    """Active Inference, DFET v3 Plasticity Gating, and Training Settings."""
    batch_size: int = 64
    seq_len: int = 2048
    chunk_size: int = 64
    learning_rate: float = 3e-3
    min_learning_rate: float = 1e-4
    warmup_steps: int = 50
    weight_decay: float = 0.01
    grad_clip_norm: float = 3.0
    loss_free_energy_weight: float = 0.05
    loss_speech_weight: float = 1.00
    
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
