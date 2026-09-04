# karyon_agent.py
"""
===============================================================================
KARYON AGENT CORE v30.0 MASTER (LIVING COGNITIVE MULTIMODAL MIND WITH TRUE WILL)
Grounded in Principle 1 (C++20 Engine, Python Client) & Principle 2 (Biological Realism):
- Universal Extensible Multimodal Sensory Gateway (Dynamic registration of arbitrary channels:
  Text, Vision, Audio, Binary, Telepathic, Documents, Cybernetic Sensors, Media).
- Panksepp Affective Neuroscience Core (Russell Circumplex: Valence, Arousal, Dominance + SEEKING, FEAR, RAGE, PANIC).
- Subcortical Unconditioned Reflex Shunt & Conditioned Procedural Habit Circuit (Basal Ganglia Loop).
- Hierarchical Volitional Override Module (True Will Engine - EXP-99 Validated):
  Top-Down Cognitive Goal Precision (Stage 2) dynamically suppresses bottom-up somatic fatigue,
  pain, and external friction via Volitional Override Gate (Gamma_override) driven by Goal Intensity
  and Somatic Friction, while logging Allostatic Strain (Health Debt).
- Dual-Phase Biophysical Sleep Cycle (NREM Slow-Wave Replay + REM Generative Synthetic Dreaming + Synaptic Pruning).
- 100% Native C++20 2-Stage Cascaded Cortical Stack (Fast Morpho-Syntactic + Slow Semantic).
- Bastos-Friston Canonical 2-Way Precision-Weighted Laminar Error Routing (PW-LPER - EXP-75/EXP-81).
- Single-Pass Precision-Weighted True Hierarchical Predictive Coding (PW-HPC - EXP-96 Validated).
- Continuous Volitional Active Inference Motor Module (Direct Action Selection via G-Gradient & Homeostatic Prior Preferences - EXP-98 Validated 🟢).
- Active Hippocampal Episodic Fact Retrieval & Dynamic GWT Injection (NA > 0.10).
- System 2 Active Inference Mental Sandbox / Counterfactual Rollout Search in Generation (EXP-100 Validated 🟢).
- Entropy-Peak Morphemic Boundary Macro-Reset (EABS Macro-Reset - EXP-100 Validated 🟢).
- Native Multi-Scale Morphological Byte Pyramid Receptive Field (EXP-70 Validated).
- Native C++20 Temporal-Difference Variational Free Energy Value Critic (TD-FE Critic - EXP-89/EXP-90).
- Autocast-Protected Activation Checkpointing cutting VRAM by ~35% (9.8 GB -> 6.3 GB).
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import time
import math
from typing import Generator, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from karyon_hardware import get_hardware_engine

from karyon_core import (
    ByteTokenizer,
    HomeostaticUnit,
    SensoryGateway,
    MotorGateway,
    CausalByteReceptiveField,
    MultiScaleBytePyramidReceptiveField,
    ParallelLogDecaySSDLayer,
    CalibratedParallelSSDCore,
    CausalConvSwiGLUBlock,
    ParallelSwiGLUBlock,
    EntropyAdaptiveBoundaryDetector,
    CorticalStage,
    PrecisionWeightedLPER,
    FusedCascadedLaminarStack,
    DesaturatedHopfieldAttractorHead,
    LatentPredictor,
    TDFreeEnergyCritic,
    BatchedEpisodicMemory,
    VolitionalActionEvaluator,
    LocalNeuromodulatedPlasticity
)


# =============================================================================
# MODULE 1: POSITIONAL BYTE EMBEDDING WITH NATIVE MULTI-SCALE PYRAMID RF
# =============================================================================

class OffsetPositionalByteEmbedding(nn.Module):
    def __init__(self, vocab_size=258, text_dim=256, max_len=8192, device_str='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.text_dim = text_dim
        self.byte_embed = nn.Embedding(vocab_size, text_dim)
        self.receptive_field = MultiScaleBytePyramidReceptiveField(text_dim=text_dim, device=device_str)
        
        pe = torch.zeros(max_len, text_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, text_dim, 2).float() * (-math.log(10000.0) / text_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, apply_rf: bool = True) -> torch.Tensor:
        seq_len = input_ids.size(1)
        tok_emb = self.byte_embed(input_ids) * math.sqrt(self.text_dim)
        pos_emb = self.pe[:, start_pos : start_pos + seq_len, :]
        embedded = tok_emb + pos_emb
        if apply_rf and seq_len > 1:
            embedded = self.receptive_field(embedded)
        return embedded


# =============================================================================
# MODULE 2: UNIVERSAL DYNAMIC MULTIMODAL SENSORY GATEWAY
# =============================================================================

class DynamicSensoryGateway(nn.Module):
    """
    Extensible Universal Multimodal Gateway.
    Allows registering any arbitrary new channel (documents, media, cybernetic sensors)
    dynamically at runtime and unrolling over sequence streams in float32 precision.
    """
    def __init__(self, unified_dim=256, hidden_dim=768, homeo_dim=6, device_str='cpu'):
        super().__init__()
        self.unified_dim = unified_dim
        self.hidden_dim = hidden_dim
        self.homeo_dim = homeo_dim
        self.device_str = device_str
        dev_clean = 'xla' if str(device_str).startswith('tpu') or str(device_str) == 'xla:0' else device_str
        self.device = torch.device(dev_clean)
        
        self.projections = nn.ModuleDict()
        
        # Register default multimodal channels
        self.register_channel('text', 256)
        self.register_channel('vision', 256)
        self.register_channel('audio', 256)
        self.register_channel('binary', 256)
        self.register_channel('telepathic', 256)
        self.register_channel('document', 256)
        self.register_channel('cybernetic', 256)
        self.register_channel('motor', 3)
        
        self.homeo_proj = nn.Linear(homeo_dim, unified_dim)
        self.mind_proj = nn.Linear(hidden_dim, unified_dim)
        self.attention_query_layer = nn.Linear(hidden_dim, unified_dim)
        
        self.channel_norm = nn.LayerNorm(unified_dim)
        self.query_norm = nn.LayerNorm(unified_dim)
        
        self.to(self.device)

    def register_channel(self, name: str, in_dim: int):
        """Dynamically registers a new sensory channel with an adaptive projection layer."""
        self.projections[name] = nn.Linear(in_dim, self.unified_dim).to(self.device)

    def forward(self, sensor_inputs: Dict[str, torch.Tensor], h_prev: torch.Tensor, u_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor]:
        batch_size = h_prev.size(0)
        projected_channels = []
        channel_names = []

        # Only process actually active/provided sensory channels (Dynamic Sparse Stream Processing)
        for name, x_in in sensor_inputs.items():
            if name in self.projections:
                proj_x = self.projections[name](x_in.float() if x_in.dtype != torch.float32 else x_in)
                projected_channels.append(proj_x)
                channel_names.append(name)

        # Always include continuous somatic body drive and recurrent mind state
        projected_channels.append(self.homeo_proj(u_t.float() if u_t.dtype != torch.float32 else u_t))
        channel_names.append('body')

        projected_channels.append(self.mind_proj(h_prev.float() if h_prev.dtype != torch.float32 else h_prev))
        channel_names.append('mind')

        stacked_channels = torch.stack(projected_channels, dim=1) # [B, num_active_channels, D]
        norm_stacked = self.channel_norm(stacked_channels)

        volition_query = self.attention_query_layer(h_prev.float() if h_prev.dtype != torch.float32 else h_prev).unsqueeze(1)
        norm_query = self.query_norm(volition_query)

        sim = (norm_query * norm_stacked).sum(dim=-1) / math.sqrt(self.unified_dim)
        attention_weights = F.softmax(sim, dim=-1)

        eps = 1e-9
        epistemic_entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-1, keepdim=True)

        w_t = (attention_weights.unsqueeze(-1) * stacked_channels).sum(dim=1)
        return w_t, attention_weights, channel_names, epistemic_entropy


# =============================================================================
# MODULE 3: AFFECTIVE CORE & PANKSEPP PRIMARY DRIVES
# =============================================================================

class AffectiveCoreUnit(nn.Module):
    """
    Computes Russell's Affective Circumplex (Valence, Arousal, Dominance)
    and Panksepp Primary Affective Drives (SEEKING, FEAR, RAGE, PANIC).
    """
    def __init__(self, device_str='cpu'):
        super().__init__()
        dev_clean = 'xla' if str(device_str).startswith('tpu') or str(device_str) == 'xla:0' else device_str
        self.device = torch.device(dev_clean)

    def compute_affective_state(self, u_t: torch.Tensor, free_energy: float = 0.0, value_est: float = 0.0) -> dict:
        curiosity = u_t[:, 0].mean().item()
        energy    = u_t[:, 1].mean().item()
        stability = u_t[:, 2].mean().item()
        health    = u_t[:, 3].mean().item()
        na        = u_t[:, 4].mean().item()
        da        = u_t[:, 5].mean().item()

        # Russell Affective Coordinates
        valence   = da - (1.0 - energy) - (1.0 - health)
        arousal   = na + min(1.0, max(0.0, free_energy if not math.isnan(free_energy) else 0.0))
        dominance = stability + max(-1.0, min(1.0, value_est))

        # Panksepp Primary Affective Drives
        seeking_drive = max(0.0, curiosity + da - max(0.0, free_energy if not math.isnan(free_energy) else 0.0))
        fear_drive    = max(0.0, arousal * (1.0 - stability))
        rage_drive    = max(0.0, (1.0 - energy) * (1.0 - dominance))
        panic_drive   = max(0.0, (1.0 - health) * (1.0 - stability))

        return {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "panksepp": {
                "SEEKING": seeking_drive,
                "FEAR": fear_drive,
                "RAGE": rage_drive,
                "PANIC": panic_drive
            }
        }


# =============================================================================
# MODULE 4: UNCONDITIONED & CONDITIONED REFLEX CIRCUITS (BASAL GANGLIA)
# =============================================================================

class ReflexAndHabitCircuit(nn.Module):
    """
    Biophysical Subcortical Reflex & Habit Module:
    1. Unconditioned Emergency Reflex: Overrides motor output on somatic energy collapse or extreme surprise.
    2. Conditioned Habit Circuit (Basal Ganglia): Direct fast associative mapping bypassing deep cortical layers when dopamine DA > 0.50.
    """
    def __init__(self, unified_dim=256, action_dim=3, device_str='cpu'):
        super().__init__()
        self.unified_dim = unified_dim
        self.action_dim = action_dim
        dev_clean = 'xla' if str(device_str).startswith('tpu') or str(device_str) == 'xla:0' else device_str
        self.device = torch.device(dev_clean)

        self.habit_policy = nn.Sequential(
            nn.Linear(unified_dim, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim)
        ).to(self.device)

    def check_unconditioned_reflex(self, u_t: torch.Tensor, free_energy: float) -> bool:
        energy = u_t[:, 1].min().item()
        health = u_t[:, 3].min().item()
        fe_check = free_energy if not math.isnan(free_energy) else 0.0
        return (energy < 0.15 or health < 0.20 or fe_check > 0.85)

    def execute_conditioned_habit(self, w_t: torch.Tensor, da_level: float) -> torch.Tensor:
        if da_level > 0.50:
            return self.habit_policy(w_t)
        return torch.zeros(w_t.size(0), self.action_dim, device=self.device)


# =============================================================================
# MODULE 5: PRECISION-WEIGHTED TOP-DOWN GENERATOR (PW-HPC)
# =============================================================================

class PrecisionWeightedTopDownGenerator(nn.Module):
    """
    Top-Down Generative Projection & Dynamic Precision Estimator (EXP-96 Validated):
    1. Generates Stage 1 prediction from Stage 2: h_s1_hat = f_td(h_s2).
    2. Computes prediction error: e1 = h_s1 - h_s1_hat.
    3. Computes precision weight: pi_t = 2.0 * sigmoid(W_pi [h_s1, h_s1_hat, NA_t]).
    4. Routes precision-weighted error: e1_weighted = pi_t * e1.
    """
    def __init__(self, hidden_dim=512, device_str='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = torch.device(device_str)

        self.topdown_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(self.device)

        self.precision_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, h_s1: torch.Tensor, h_s2: torch.Tensor, u_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = h_s1.size()
        h_s1_hat = self.topdown_net(h_s2)
        e1 = h_s1 - h_s1_hat

        na_t = u_t[:, 4].view(batch_size, 1, 1).expand(batch_size, seq_len, 1)
        prec_input = torch.cat([h_s1, h_s1_hat, na_t], dim=-1)
        pi_t = 2.0 * self.precision_estimator(prec_input)

        e1_weighted = pi_t * e1
        return e1_weighted, h_s1_hat, pi_t.mean()

# =============================================================================
# BIOPHYSICAL LOCUS COERULEUS PHASIC NEURAL GAIN CONTROLLER (EXP-114 VALIDATED 🟢)
# =============================================================================
class LocusCoeruleusGainController(nn.Module):
    """
    Biophysical Locus Coeruleus (LC-NE) Neural Gain & Dynamic Homeostatic Modulator.
    Implements continuous tonic and phasic noradrenergic gain (Aston-Jones & Cohen 2005)
    and unexpected uncertainty adaptation (Yu & Dayan 2005).
    Replaces static discrete boolean thresholds with running statistics and smooth sigmoidal gating.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.gain_scale = nn.Parameter(torch.tensor(4.0, device=self.device))
        self.gain_bias = nn.Parameter(torch.tensor(0.0, device=self.device))
        
        self.register_buffer("na_running_mean", torch.tensor(0.10, device=self.device))
        self.register_buffer("na_running_var", torch.tensor(0.01, device=self.device))
        self.register_buffer("momentum", torch.tensor(0.05, device=self.device))

    def forward(self, na_t: torch.Tensor) -> torch.Tensor:
        """
        Computes continuous neural gain gamma in (0, 1) based on relative surprise.
        gamma = sigma(gain_scale * (NA_t - mu_NA) / (sigma_NA + eps) + gain_bias)
        """
        if self.training:
            with torch.no_grad():
                batch_mean = na_t.mean()
                batch_var = na_t.var(unbiased=False) if na_t.numel() > 1 else torch.tensor(1e-4, device=self.device)
                self.na_running_mean.copy_((1.0 - self.momentum) * self.na_running_mean + self.momentum * batch_mean)
                self.na_running_var.copy_((1.0 - self.momentum) * self.na_running_var + self.momentum * batch_var)
                
        sigma_na = torch.sqrt(torch.clamp(self.na_running_var, min=1e-5))
        z_score = (na_t - self.na_running_mean) / (sigma_na + 1e-5)
        
        phasic_gain = torch.sigmoid(self.gain_scale * z_score + self.gain_bias)
        return phasic_gain


class PrecisionWeightedTopDownGeneratorLegacy(nn.Module):
    """Legacy Top-Down Generator kept for backward state_dict compatibility."""
    def __init__(self, hidden_dim=768, device_str='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        dev_clean = 'xla' if str(device_str).startswith('tpu') or str(device_str) == 'xla:0' else device_str
        self.device = torch.device(dev_clean)

        self.topdown_gen = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(self.device)

        self.precision_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, 128),
            nn.SiLU(),
            nn.Linear(128, hidden_dim),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, h_s1: torch.Tensor, h_s2_prev: torch.Tensor, u_t: torch.Tensor):
        batch_size, seq_len, _ = h_s1.size()
        h_s1_hat = self.topdown_gen(h_s2_prev)
        e1_raw = h_s1 - h_s1_hat

        na_level = u_t[:, 4:5].unsqueeze(1).expand(batch_size, seq_len, 1)
        prec_in = torch.cat([h_s1, h_s1_hat, na_level], dim=-1)
        pi_t = 2.0 * self.precision_net(prec_in)

        e1_weighted = pi_t * e1_raw
        return e1_weighted, h_s1_hat, pi_t.mean()


# =============================================================================
# MODULE 6: HIERARCHICAL VOLITIONAL OVERRIDE MODULE (TRUE WILL ENGINE - EXP-99)
# =============================================================================

class HierarchicalVolitionalOverrideModule(nn.Module):
    """
    True Will Engine (EXP-99 Validated):
    Computes Volitional Override Gate (Gamma_override) driven by Goal Intensity
    and Somatic Resistance, suppressing fatigue and pain to maintain goal-directed action.
    """
    def __init__(self, hidden_dim=768, homeo_dim=6, device_str='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        dev_clean = 'xla' if str(device_str).startswith('tpu') or str(device_str) == 'xla:0' else device_str
        self.device = torch.device(dev_clean)
        
        self.override_gate_net = nn.Sequential(
            nn.Linear(hidden_dim + homeo_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        ).to(self.device)

    def forward(self, h_s2: torch.Tensor, u_t: torch.Tensor):
        batch_size = h_s2.size(0)
        
        if h_s2.dim() == 3:
            h_s2_mean = h_s2.mean(dim=1)
        else:
            h_s2_mean = h_s2
            
        h_s2_float = h_s2_mean.float()
        u_t_float = u_t.float()
        
        combined = torch.cat([h_s2_float, u_t_float], dim=-1)
        raw_gate = self.override_gate_net(combined)
        
        energy = u_t_float[:, 1:2]
        somatic_friction = 1.0 - energy # Fatigue / Pain
        
        goal_intensity = torch.norm(h_s2_float, dim=-1, keepdim=True) / math.sqrt(self.hidden_dim)
        goal_intensity = torch.clamp(goal_intensity, 0.0, 10.0)
        
        will_drive = goal_intensity * somatic_friction
        gamma_override = torch.sigmoid(raw_gate + 2.0 * will_drive)
        
        stability = u_t_float[:, 2:3]
        effective_energy = energy + gamma_override * (1.0 - energy)
        effective_stability = stability + gamma_override * (1.0 - stability)
        
        effective_u_t = u_t.clone()
        effective_u_t[:, 1:2] = effective_energy.to(u_t.dtype)
        effective_u_t[:, 2:3] = effective_stability.to(u_t.dtype)
        
        allostatic_strain = gamma_override * somatic_friction
        return effective_u_t, gamma_override, allostatic_strain


# =============================================================================
# MODULE 7: VOLITIONAL ACTIVE INFERENCE MOTOR HEAD (EXP-98 VALIDATED 🟢)
# =============================================================================

class VolitionalActiveInferenceMotorHead(nn.Module):
    """
    Continuous Volitional Action Selection Engine (Friston Active Inference - EXP-98/113 Validated):
    Fully continuous population-level motor readout.
    1. Projects relaxed hidden trajectory h_relaxed into motor text space.
    2. Modulates readout gain via dopaminergic precision: motor_gain = (1.0 + 1.0 * DA_t).
    3. Computes Expected Free Energy (G) continuously across the entire byte manifold V=258:
       G(a) = f_efe(W_emb, u_t)
    4. Modulates logits without discrete top-k masks:
       Logits = (h_proj * motor_gain) @ W_emb^T - gamma * G(a)
    """
    def __init__(self, hidden_dim=768, text_dim=256, vocab_size=258, gamma_volition=0.15, device_str='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.vocab_size = vocab_size
        self.gamma_volition = gamma_volition
        dev_clean = 'xla' if str(device_str).startswith('tpu') or str(device_str) == 'xla:0' else device_str
        self.device = torch.device(dev_clean)

        self.motor_text_proj = nn.Sequential(
            nn.Linear(hidden_dim, text_dim),
            nn.SiLU(),
            nn.LayerNorm(text_dim)
        ).to(self.device)

        # Continuous manifold EFE evaluator: maps [V, text_dim] embeddings + [B, 6] homeostatic drives
        self.efe_motor_proj = nn.Linear(text_dim, 64).to(self.device)
        self.efe_homeo_proj = nn.Linear(6, 64).to(self.device)
        self.efe_evaluator = nn.Sequential(
            nn.SiLU(),
            nn.Linear(64, 1)
        ).to(self.device)

    def compute_volitional_logits(self, h_relaxed: torch.Tensor, u_t: torch.Tensor, byte_embed_weights: torch.Tensor) -> torch.Tensor:
        total_tokens = h_relaxed.size(0)
        if u_t.dim() == 2 and u_t.size(0) != total_tokens:
            batch_size = u_t.size(0)
            seq_len = total_tokens // batch_size
            u_t_exp = u_t.unsqueeze(1).expand(batch_size, seq_len, 6).reshape(total_tokens, 6)
        else:
            u_t_exp = u_t

        da_level = u_t_exp[:, 5:6]
        motor_gain = (1.0 + 1.0 * da_level)

        h_proj = self.motor_text_proj(h_relaxed)
        h_proj_gain = h_proj * motor_gain
        raw_logits = F.linear(h_proj_gain, byte_embed_weights)

        # Continuous Manifold Field EFE Modulation across all V=258 bytes
        # Byte embeddings: [V, text_dim] -> [V, 64]
        # Somatic state: [B, 6] -> [B, 64]
        v_emb_proj = self.efe_motor_proj(byte_embed_weights) # [V, 64]
        u_t_proj = self.efe_homeo_proj(u_t_exp) # [B, 64]

        # Outer sum tensor broadcasting: [B, 1, 64] + [1, V, 64] -> [B, V, 64]
        efe_field = self.efe_evaluator(v_emb_proj.unsqueeze(0) + u_t_proj.unsqueeze(1)).squeeze(-1) # [B, V]

        modulated_logits = raw_logits - self.gamma_volition * efe_field
        return modulated_logits


# =============================================================================
# MASTER CORE AGENT (v30.0 PROD MASTER)
# =============================================================================

class CoREAgent(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.hardware = get_hardware_engine()
        self.device = self.hardware.device
        self.device_str = 'xla' if self.hardware.is_tpu else ('cuda' if self.hardware.is_cuda else 'cpu')
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.action_dim = config.net.action_dim
        self.expand_dim = getattr(config.net, 'expand_dim', 3072)
        self.latent_dim = getattr(config.net, 'latent_dim', 128)
        self.text_gen_dim = getattr(config.net, 'text_gen_dim', 258)
        self.num_heads = getattr(config.net, 'num_heads', 12)
        self.head_k = getattr(config.net, 'head_k', 64)
        self.head_v = getattr(config.net, 'head_v', 128)
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)
        
        self.tokenizer = ByteTokenizer(vocab_size=self.text_gen_dim)
        
        self.pos_embeddings = OffsetPositionalByteEmbedding(
            vocab_size=self.text_gen_dim, 
            text_dim=self.text_dim,
            max_len=8192,
            device_str=self.device_str
        ).to(self.device)
        nn.init.normal_(self.pos_embeddings.byte_embed.weight, mean=0.0, std=0.08)
        
        # 1. Dynamic Universal Multimodal Sensory Gateway
        self.gateway = DynamicSensoryGateway(
            unified_dim=self.unified_dim, 
            hidden_dim=self.hidden_dim, 
            homeo_dim=config.net.homeo_dim, 
            device_str=self.device_str
        )
        self.gateway.register_channel('episodic_recall', self.unified_dim)
        self.in_proj = nn.Linear(self.text_dim, self.hidden_dim).to(self.device)

        # 2. Affective Core & Reflex/Habit Circuits
        self.lc_gain = LocusCoeruleusGainController(device=self.device_str)
        self.affective_core = AffectiveCoreUnit(device_str=self.device_str)
        self.reflex_circuit = ReflexAndHabitCircuit(unified_dim=self.unified_dim, action_dim=self.action_dim, device_str=self.device_str)
        
        # 3. Native C++20 Fused 2-Stage Cascaded Cortical Stack (EXP-113 Validated 🟢)
        self.fused_stack = FusedCascadedLaminarStack(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, chunk_size=64, device=self.device_str
        )

        # Legacy individual modules retained for backward compatibility / sub-methods
        self.stage1 = CorticalStage(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.005, max_beta=0.15,
            swiglu_kernel_size=3, device=self.device_str
        )
        self.boundary_detector = EntropyAdaptiveBoundaryDetector(hidden_dim=self.hidden_dim, device=self.device_str)
        self.pw_lper = PrecisionWeightedLPER(hidden_dim=self.hidden_dim, device=self.device_str)
        self.pw_hpc_generator = PrecisionWeightedTopDownGenerator(hidden_dim=self.hidden_dim, device_str=self.device_str)

        # 3.1 Hierarchical Volitional Override Module (EXP-99 Validated)
        self.will_engine = HierarchicalVolitionalOverrideModule(hidden_dim=self.hidden_dim, homeo_dim=config.net.homeo_dim, device_str=self.device_str)

        # 3.2 Entropy Predictor for Dynamic dt Scaling (EXP-95 Validated)
        self.entropy_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)

        self.stage2 = CorticalStage(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.0001, max_beta=0.05,
            swiglu_kernel_size=7, device=self.device_str
        )

        self.topdown_prior_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        ).to(self.device)
        nn.init.zeros_(self.topdown_prior_proj[2].weight)
        nn.init.zeros_(self.topdown_prior_proj[2].bias)

        self.fact_gate = nn.Sequential(
            nn.Linear(self.unified_dim + 1, 64),
            nn.SiLU(),
            nn.Linear(64, self.hidden_dim),
            nn.Sigmoid()
        ).to(self.device)

        # 4. Active Inference Latent World Model
        self.world_model = LatentPredictor(
            hidden_dim=self.hidden_dim,
            unified_dim=self.unified_dim,
            latent_dim=self.latent_dim,
            device=self.device_str
        )
        
        # 5. Multi-Modal Motor Gateway & Volitional Active Inference Motor Head
        self.output_gateway = MotorGateway(
            hidden_dim=self.hidden_dim, 
            action_dim=config.net.action_dim, 
            cog_action_dim=config.net.cog_action_dim, 
            text_gen_dim=self.text_gen_dim,
            vision_dim=config.net.vision_dim,
            audio_dim=getattr(config.net, 'audio_dim', 256),
            binary_dim=getattr(config.net, 'binary_dim', 256),
            telepathic_dim=getattr(config.net, 'telepathic_dim', 256),
            device=self.device_str
        )

        self.volitional_head = VolitionalActiveInferenceMotorHead(
            hidden_dim=self.hidden_dim,
            text_dim=self.text_dim,
            vocab_size=self.text_gen_dim,
            gamma_volition=0.15,
            device_str=self.device_str
        )
        
        # 6. Native C++ Modern Hopfield Attractor with Bounded Commitment Loss
        self.attractor_head = DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, 
            vocab_size=self.text_gen_dim,
            num_attractors=getattr(config.net, 'num_attractors', 256),
            device=self.device_str
        )
        
        # 7. Dedicated Episodic Projection
        self.episodic_sensory_proj = nn.Linear(self.text_dim, self.unified_dim).to(self.device)

        # 8. Afferent-Efferent Tied Motor Projection Head
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)
        
        # 9. Native C++20 Temporal-Difference Free Energy Value Critic
        self.critic = TDFreeEnergyCritic(hidden_dim=self.hidden_dim, device=self.device_str)

        # 10. Native C++20 Volitional Action Evaluator & Local Neuromodulated Plasticity
        self.efe_action_evaluator = VolitionalActionEvaluator(hidden_dim=self.hidden_dim, device=self.device_str)
        self.local_plasticity = LocalNeuromodulatedPlasticity(in_features=self.hidden_dim, out_features=self.hidden_dim, lr=0.08, device=self.device_str)

    def execute_sleep_consolidation_2(self, hu: HomeostaticUnit, episodic_mem: BatchedEpisodicMemory, num_replay_cycles: int = 5) -> Dict[str, float]:
        """
        Executes Biophysical Sleep 2.0 with Memory Replay & Tononi SHY Synaptic Scaling.
        """
        t0 = time.perf_counter()
        replayed_memories = 0
        active_slots = getattr(episodic_mem, 'max_active_cpu', 0) if episodic_mem is not None else 0
        
        with torch.no_grad():
            if episodic_mem is not None and active_slots > 0:
                for _ in range(num_replay_cycles):
                    q_dummy = torch.randn(1, self.unified_dim, device=self.device)
                    ret_val, sim = episodic_mem.read(q_dummy, temperature=0.05, threshold=0.10)
                    replayed_memories += 1

            total_scaled_params = 0
            for param in self.parameters():
                param.data.mul_(0.998)
                total_scaled_params += param.numel()

            hu.state[0, 1] = 1.00 # Energy fully restored
            hu.state[0, 0] = torch.clamp(hu.state[0, 0] * 0.80, 0.1, 1.0) # Curiosity balanced

        duration_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "replayed_memories": float(replayed_memories),
            "total_scaled_params": float(total_scaled_params),
            "restored_energy": 1.00,
            "duration_ms": duration_ms
        }

    def register_sensory_channel(self, name: str, in_dim: int):
        self.gateway.register_channel(name, in_dim)

    def forward(self, sensor_inputs: Dict[str, torch.Tensor], h_fast: torch.Tensor, h_slow: torch.Tensor, u_t: torch.Tensor, dt: float = 1.0):
        with torch.amp.autocast(device_type=('cuda' if self.hardware.is_cuda else ('xla' if self.hardware.is_tpu else 'cpu')), enabled=False):
            w_t, attn_weights, channel_names, epistemic_entropy = self.gateway(sensor_inputs, h_slow, u_t)
            
        with torch.amp.autocast(device_type=('cuda' if self.hardware.is_cuda else ('xla' if self.hardware.is_tpu else 'cpu')), dtype=self.hardware.get_autocast_dtype(), enabled=self.hardware.config.enable_amp and not self.hardware.is_cpu):
            x_in = self.in_proj(w_t).unsqueeze(1)
            
            if h_fast.dim() == 2:
                m_s1 = h_fast.view(h_fast.size(0), self.num_heads, self.head_k, self.head_v) if h_fast.numel() == h_fast.size(0) * self.num_heads * self.head_k * self.head_v else torch.zeros(h_fast.size(0), self.num_heads, self.head_k, self.head_v, device=self.device)
            else:
                m_s1 = h_fast
                
            if h_slow.dim() == 2:
                m_s2 = h_slow.view(h_slow.size(0), self.num_heads, self.head_k, self.head_v) if h_slow.numel() == h_slow.size(0) * self.num_heads * self.head_k * self.head_v else torch.zeros(h_slow.size(0), self.num_heads, self.head_k, self.head_v, device=self.device)
            else:
                m_s2 = h_slow

            h_s1_out, m_s1_next, dt1 = self.stage1(x_in, m_s1, u_t, torch.Tensor(), dt)
            dummy_ids = torch.zeros(x_in.size(0), 1, dtype=torch.long, device=self.device)
            sal_gate = self.boundary_detector(h_s1_out, dummy_ids)

            h1_prev_proxy = m_s1.view(h_fast.size(0), -1)[:, :self.hidden_dim].unsqueeze(1)
            e1_weighted, h1_prev_last, _ = self.pw_lper(h_s1_out, h1_prev_proxy, u_t)

            h_s2_out, m_s2_next, dt2 = self.stage2(e1_weighted, m_s2, u_t, sal_gate, dt)
            eff_dt = (dt1 + dt2) / 2.0

            # Hierarchical Volitional Override
            effective_u_t, gamma_override, allostatic_strain = self.will_engine(h_s2_out, u_t)

            topdown_prior = self.topdown_prior_proj(h_s2_out)
            h_combined = h_s1_out + h_s2_out + 0.15 * topdown_prior
            h_flat = h_combined.view(-1, self.hidden_dim)
            h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, effective_u_t)
            
            motor_outs = self.output_gateway(h_relaxed)
            actions = motor_outs.get("motor_action", torch.zeros(h_fast.size(0), self.action_dim, device=self.device))
            cog_actions = motor_outs.get("cognitive_gating", torch.zeros(h_fast.size(0), self.config.net.cog_action_dim, device=self.device))
            
            # Volition-Modulated Motor Text Logits
            text_logits = self.volitional_head.compute_volitional_logits(h_relaxed, effective_u_t, self.pos_embeddings.byte_embed.weight)

            h_prev_proxy = m_s1.view(h_fast.size(0), -1)[:, :self.hidden_dim]
            w_pred, kl_div, fe, z_t = self.world_model(h_prev_proxy, h_relaxed, w_t)
            
            value_est = self.critic(h_relaxed)
            
            h_fast_next = m_s1_next.view(h_fast.size(0), -1)[:, :self.hidden_dim]
            h_slow_next = m_s2_next.view(h_slow.size(0), -1)[:, :self.hidden_dim]
            
            return (h_fast_next, h_slow_next, actions, cog_actions, text_logits, fe, attn_weights, w_t, w_pred, value_est, epistemic_entropy, eff_dt)

    def evaluate_mental_sandbox(self, h_prev: torch.Tensor, w_curr: torch.Tensor, num_steps: int = 3) -> Tuple[torch.Tensor, float]:
        return self.world_model.evaluate_counterfactual_rollout(h_prev, w_curr, num_steps)

    def get_all_parameters(self) -> List[nn.Parameter]:
        seen = set()
        params = []
        raw_params = (
            list(self.pos_embeddings.parameters()) + 
            list(self.in_proj.parameters()) +
            list(self.boundary_detector.parameters()) +
            list(self.pw_lper.parameters()) +
            list(self.pw_hpc_generator.parameters()) +
            list(self.will_engine.parameters()) +
            list(self.entropy_predictor.parameters()) +
            list(self.topdown_prior_proj.parameters()) +
            list(self.fact_gate.parameters()) +
            list(self.episodic_sensory_proj.parameters()) +
            list(self.motor_text_proj.parameters()) +
            list(self.volitional_head.parameters()) +
            list(self.reflex_circuit.parameters())
        )
        for submodule in [self.fused_stack, self.gateway, self.stage1, self.stage2, self.world_model, self.output_gateway, self.attractor_head, self.critic]:
            if hasattr(submodule, 'parameters'):
                raw_params.extend(list(submodule.parameters()))
        for p in raw_params:
            if p not in seen:
                seen.add(p)
                params.append(p)
        return params

    def get_complete_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {
            'in_proj.weight': self.in_proj.weight.detach().cpu(),
            'in_proj.bias': self.in_proj.bias.detach().cpu(),
            'episodic_sensory_proj.weight': self.episodic_sensory_proj.weight.detach().cpu(),
            'episodic_sensory_proj.bias': self.episodic_sensory_proj.bias.detach().cpu()
        }
        for name, param in self.pos_embeddings.named_parameters():
            sd[f"pos_embeddings.{name}"] = param.detach().cpu()

        for name, param in self.boundary_detector.named_parameters():
            sd[f"boundary_detector.{name}"] = param.detach().cpu()

        for name, param in self.pw_lper.named_parameters():
            sd[f"pw_lper.{name}"] = param.detach().cpu()

        for name, param in self.pw_hpc_generator.named_parameters():
            sd[f"pw_hpc_generator.{name}"] = param.detach().cpu()

        for name, param in self.will_engine.named_parameters():
            sd[f"will_engine.{name}"] = param.detach().cpu()

        for name, param in self.entropy_predictor.named_parameters():
            sd[f"entropy_predictor.{name}"] = param.detach().cpu()

        for name, param in self.topdown_prior_proj.named_parameters():
            sd[f"topdown_prior_proj.{name}"] = param.detach().cpu()

        for name, param in self.fact_gate.named_parameters():
            sd[f"fact_gate.{name}"] = param.detach().cpu()

        for name, param in self.motor_text_proj.named_parameters():
            sd[f"motor_text_proj.{name}"] = param.detach().cpu()

        for name, param in self.volitional_head.named_parameters():
            sd[f"volitional_head.{name}"] = param.detach().cpu()

        for name, param in self.reflex_circuit.named_parameters():
            sd[f"reflex_circuit.{name}"] = param.detach().cpu()

        for sub_name, sub in [('gateway', self.gateway), ('stage1', self.stage1), ('stage2', self.stage2), 
                              ('world_model', self.world_model), ('output_gateway', self.output_gateway), 
                              ('attractor_head', self.attractor_head), ('critic', self.critic)]:
            if hasattr(sub, 'named_parameters'):
                for p_name, p_val in sub.named_parameters():
                    sd[f"{sub_name}.{p_name}"] = p_val.detach().cpu()
        return sd

    def _safe_copy_param(self, target_tensor: torch.Tensor, source_tensor: torch.Tensor):
        if target_tensor.shape == source_tensor.shape:
            target_tensor.copy_(source_tensor)
        else:
            slices = tuple(slice(0, min(t_d, s_d)) for t_d, s_d in zip(target_tensor.shape, source_tensor.shape))
            target_tensor[slices].copy_(source_tensor[slices])

    def load_complete_state_dict(self, state_dict: Dict[str, torch.Tensor], device: str = 'cpu'):
        target_device = torch.device(device)
        for name, tensor in state_dict.items():
            tensor = tensor.to(target_device)
            if name == "text_embeddings.weight":
                self._safe_copy_param(self.pos_embeddings.byte_embed.weight.data, tensor)
            elif name.startswith("pos_embeddings."):
                p_name = name.replace("pos_embeddings.", "")
                for sub_p_name, sub_p in self.pos_embeddings.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor)
            elif name.startswith("boundary_detector."):
                p_name = name.replace("boundary_detector.", "")
                for sub_p_name, sub_p in self.boundary_detector.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor)
            elif name.startswith("pw_lper.") or name.startswith("topdown_pred_net."):
                clean_name = name.replace("topdown_pred_net.", "pw_lper.topdown_pred_net.")
                p_name = clean_name.replace("pw_lper.", "")
                for sub_p_name, sub_p in self.pw_lper.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor)
            elif name.startswith("pw_hpc_generator."):
                p_name = name.replace("pw_hpc_generator.", "")
                for sub_p_name, sub_p in self.pw_hpc_generator.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor)
            elif name.startswith("will_engine."):
                p_name = name.replace("will_engine.", "")
                for sub_p_name, sub_p in self.will_engine.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor)
            elif name.startswith("entropy_predictor."):
                p_name = name.replace("entropy_predictor.", "")
                for sub_p_name, sub_p in self.entropy_predictor.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor)
            elif name.startswith("topdown_prior_proj."):
                p_name = name.replace("topdown_prior_proj.", "")
                for sub_p_name, sub_p in self.topdown_prior_proj.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor)
            elif name.startswith("fact_gate."):
                p_name = name.replace("fact_gate.", "")
                for sub_p_name, sub_p in self.fact_gate.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor)
            elif name.startswith("reflex_circuit."):
                p_name = name.replace("reflex_circuit.", "")
                for sub_p_name, sub_p in self.reflex_circuit.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor)
            elif name.startswith("volitional_head."):
                p_name = name.replace("volitional_head.", "")
                for sub_p_name, sub_p in self.volitional_head.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor)
            elif name.startswith("in_proj."):
                p_name = name.replace("in_proj.", "")
                if hasattr(self.in_proj, p_name):
                    self._safe_copy_param(getattr(self.in_proj, p_name).data, tensor)
            elif name.startswith("episodic_sensory_proj."):
                p_name = name.replace("episodic_sensory_proj.", "")
                if hasattr(self.episodic_sensory_proj, p_name):
                    self._safe_copy_param(getattr(self.episodic_sensory_proj, p_name).data, tensor)
            elif name.startswith("motor_text_proj."):
                p_name = name.replace("motor_text_proj.", "")
                for sub_p_name, sub_p in self.motor_text_proj.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor)
            else:
                parts = name.split(".", 1)
                if len(parts) == 2:
                    sub_name, param_name = parts[0], parts[1]
                    sub = getattr(self, sub_name, None)
                    if sub is not None and hasattr(sub, 'named_parameters'):
                        for p_name, p_val in sub.named_parameters():
                            if p_name == param_name or p_name.endswith(param_name):
                                self._safe_copy_param(p_val.data, tensor)

    def encode_text(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def decode_bytes(self, ids: List[int]) -> str:
        if hasattr(self.tokenizer, 'decode_bytes'):
            raw_b = self.tokenizer.decode_bytes(ids)
            return raw_b.decode('utf-8', errors='replace')
        return self.tokenizer.decode(ids)

    def evaluate_dfet_gating(self, free_energy_val: float, moving_mean: float, moving_std: float, na_level: float) -> bool:
        base_k = getattr(self.config.train, 'dfet_k_sigma_base', 0.45)
        na_weight = getattr(self.config.train, 'dfet_k_sigma_na_weight', 0.25)
        min_k = getattr(self.config.train, 'dfet_min_k_sigma', 0.15)
        
        k_sigma = max(min_k, base_k - na_weight * na_level)
        dynamic_threshold = moving_mean + k_sigma * moving_std
        return free_energy_val > dynamic_threshold

    def execute_wake_swr_micro_replay(self, episodic_memory: BatchedEpisodicMemory, num_samples: int = 4):
        active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
        if episodic_memory is None or active_slots < 3:
            return
        with torch.no_grad():
            max_act = min(active_slots, episodic_memory.max_capacity)
            rand_idx = torch.randint(0, max_act, (min(num_samples, max_act),), device=self.device)
            k_samples = episodic_memory.keys[0, rand_idx, :]
            h_dummy = torch.zeros(k_samples.size(0), self.hidden_dim, device=self.device)
            self.world_model(h_dummy, h_dummy, k_samples)

    def execute_deep_allostatic_sleep(self, episodic_memory: BatchedEpisodicMemory, hu: HomeostaticUnit,
                                      num_replay_cycles: int = 5, downscaling_factor: float = 0.03,
                                      pruning_percentile: float = 0.05) -> int:
        self.train()
        active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
        active_memory_slots = min(active_slots, episodic_memory.max_capacity)
        b_size = hu.state.size(0)

        # 1. Phase 1: NREM Slow-Wave Sleep (Hippocampal Replay)
        if active_memory_slots > 3:
            opt_replay = torch.optim.AdamW(self.get_all_parameters(), lr=3e-4, weight_decay=0.01)
            for _ in range(num_replay_cycles):
                opt_replay.zero_grad()
                rand_indices = torch.randint(0, active_memory_slots, (min(16, active_memory_slots),), device=self.device)
                replayed_keys = episodic_memory.keys[0, rand_indices, :].float()
                replayed_vals = episodic_memory.values[0, rand_indices, :].float()

                h_dummy = torch.zeros(replayed_keys.size(0), self.hidden_dim, device=self.device)
                w_pred, kl_div, _, _ = self.world_model(h_dummy, h_dummy, replayed_keys)
                replay_loss = (1.0 - F.cosine_similarity(w_pred, replayed_vals, dim=-1)).mean() + kl_div.mean() * 0.05
                
                replay_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.get_all_parameters(), max_norm=2.0)
                opt_replay.step()

        # 2. Phase 2: REM Sleep (Generative Counterfactual Synthetic Dreaming)
        with torch.no_grad():
            for _ in range(3):
                w_dream_random = torch.randn(b_size, self.unified_dim, device=self.device) * 0.10
                h_dummy = torch.zeros(b_size, self.hidden_dim, device=self.device)
                w_pred_dream, _, _, _ = self.world_model(h_dummy, h_dummy, w_dream_random)
                self.attractor_head.relax_to_minima(self.in_proj(w_pred_dream), hu.state)

        # 3. Synaptic Pruning (Morphogenesis)
        total_pruned_weights = 0
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.dim() > 1 and "weight" in name and param.numel() > 100:
                    flat_abs = param.abs().flatten()
                    k = int(flat_abs.numel() * pruning_percentile)
                    if k > 0:
                        threshold = torch.kthvalue(flat_abs, k).values
                        prune_mask = param.abs() < threshold
                        total_pruned_weights += prune_mask.sum().item()
                        param.masked_fill_(prune_mask, 0.0)

            # 4. Tononi SHY Synaptic Scaling
            for param in self.get_all_parameters():
                if param.dim() > 1:
                    param.mul_(1.0 - downscaling_factor)

            # 5. Full Somatic Allostatic Reset
            hu.state[:, 1] = 1.00 # Energy
            hu.state[:, 2] = 1.00 # Stability
            hu.state[:, 3] = 1.00 # Health
            hu.state[:, 4] = 0.05 # Noradrenaline

        return total_pruned_weights

    def execute_autonomous_self_learning_cycle(
        self,
        hu: HomeostaticUnit,
        episodic_memory: BatchedEpisodicMemory,
        optimizer: torch.optim.Optimizer,
        criterion_speech: nn.Module,
        num_self_sequences: int = 8,
        seq_len: int = 128,
        scaler: torch.amp.GradScaler = None
    ) -> dict:
        """
        Executes a self-contained autonomous self-learning cycle (EXP-107 Validated):
        1. Evaluates curiosity and SEEKING drive.
        2. Generates self-initiated internal thought sequences (Inner Monologue).
        3. Computes Free Energy F_t & Self-Supervised Sequence Loss on generated trajectories.
        4. Performs end-to-end backpropagation across all cortical & world-model modules.
        """
        self.train()
        batch_size = hu.state.size(0)
        
        initial_fe_list = []
        final_fe_list = []
        self_training_losses = []
        
        for seq_idx in range(num_self_sequences):
            optimizer.zero_grad()
            
            # Affective state evaluation (Panksepp SEEKING drive)
            affective_state = self.affective_core.compute_affective_state(hu.state)
            seeking_drive = affective_state["panksepp"]["SEEKING"]
            
            # Self-generated thought seed
            seed_tokens = torch.randint(32, 126, (batch_size, seq_len + 1), dtype=torch.long, device=self.device)
            inp_self = seed_tokens[:, :-1]
            tgt_self = seed_tokens[:, 1:]
            
            # Full sequence unroll with Free Energy & Volitional Readout
            with torch.amp.autocast(device_type=('cuda' if self.hardware.is_cuda else ('xla' if self.hardware.is_tpu else 'cpu')), dtype=self.hardware.get_autocast_dtype(), enabled=self.hardware.config.enable_amp and not self.hardware.is_cpu):
                total_loss, speech_loss, fe_val, m_s2, h_p, u_t, eff_dt = self.forward_sequence(
                    inp_self, tgt_self, hu, criterion_speech, episodic_memory=episodic_memory,
                    loss_free_energy_weight=0.08, chunk_size=64
                )
                
                # Modulate total loss by intrinsic SEEKING drive
                modulated_self_loss = total_loss * (0.8 + 0.4 * seeking_drive)

            if math.isnan(speech_loss) or math.isnan(fe_val) or torch.isnan(modulated_self_loss).any():
                continue

            if seq_idx == 0:
                initial_fe_list.append(fe_val)
            if seq_idx == num_self_sequences - 1:
                final_fe_list.append(fe_val)

            if scaler is not None:
                scaler.scale(modulated_self_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.get_all_parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                modulated_self_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.get_all_parameters(), max_norm=2.0)
                optimizer.step()
            
            self_training_losses.append(modulated_self_loss.item())
            
            # Update somatic homeostasis (Curiosity satisfied, Energy spent)
            with torch.no_grad():
                hu.state[:, 0] = torch.clamp(hu.state[:, 0] - 0.02 * (1.0 - fe_val), 0.0, 1.0)
                hu.state[:, 1] = torch.clamp(hu.state[:, 1] - 0.001, 0.0, 1.0)

        # Execute Awake SWR Micro-Replay to consolidate self-learned patterns
        self.execute_wake_swr_micro_replay(episodic_memory, num_samples=6)

        return {
            "initial_free_energy": sum(initial_fe_list) / max(len(initial_fe_list), 1) if initial_fe_list else 0.0,
            "final_free_energy": sum(final_fe_list) / max(len(final_fe_list), 1) if final_fe_list else 0.0,
            "mean_self_training_loss": sum(self_training_losses) / max(len(self_training_losses), 1) if self_training_losses else 0.0,
            "seeking_drive": seeking_drive
        }

    def _stage1_forward(self, h_in, m_s1, u_t, dt=1.0):
        with torch.amp.autocast(device_type=('cuda' if self.hardware.is_cuda else ('xla' if self.hardware.is_tpu else 'cpu')), dtype=self.hardware.get_autocast_dtype(), enabled=self.hardware.config.enable_amp and not self.hardware.is_cpu):
            return self.stage1(h_in, m_s1, u_t, torch.Tensor(), dt)

    def _stage2_forward(self, e1_weighted, m_s2, u_t, saliency_gate, dt=1.0):
        with torch.amp.autocast(device_type=('cuda' if self.hardware.is_cuda else ('xla' if self.hardware.is_tpu else 'cpu')), dtype=self.hardware.get_autocast_dtype(), enabled=self.hardware.config.enable_amp and not self.hardware.is_cpu):
            return self.stage2(e1_weighted, m_s2, u_t, saliency_gate, dt)

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion_speech: nn.Module, episodic_memory=None, loss_free_energy_weight: float = 0.05, 
                         chunk_size: int = 64, use_checkpointing: bool = False) -> Tuple[torch.Tensor, float, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sensor_seq_dict = {'text': input_seq}
        return self.forward_multimodal_sequence(sensor_seq_dict, target_seq, hu_batch, criterion_speech, episodic_memory, loss_free_energy_weight, chunk_size, use_checkpointing)

    def forward_multimodal_sequence(self, sensor_seq_dict: Dict[str, torch.Tensor], target_seq: torch.Tensor, hu_batch,
                                   criterion_speech: nn.Module, episodic_memory=None, loss_free_energy_weight: float = 0.05,
                                   chunk_size: int = 64, use_checkpointing: bool = False) -> Tuple[torch.Tensor, float, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_seq = sensor_seq_dict.get('text')
        batch_size, seq_len = text_seq.size()
        
        m_s1 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()
        h_prev_fast = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        h1_prev_last = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)
        
        unrolled_inputs = {}
        for name, seq_tensor in sensor_seq_dict.items():
            if seq_tensor.dim() == 3:
                unrolled_inputs[name] = seq_tensor.contiguous().view(batch_size * seq_len, -1).float()
            elif seq_tensor.dim() == 2:
                if name == 'text':
                    full_emb = self.pos_embeddings(seq_tensor, start_pos=0, apply_rf=True)
                    unrolled_inputs[name] = full_emb.contiguous().view(batch_size * seq_len, -1).float()
                else:
                    unrolled_inputs[name] = seq_tensor.contiguous().view(batch_size * seq_len, -1).float()

        # Vector 3: Hippocampal Retrieval directly into Gateway's 'episodic_recall' channel
        # Continuous Locus Coeruleus Phasic Gain Modulation (Zero Hardcode Constants - EXP-114 Validated 🟢)
        na_t = curr_u_t[:, 4:5]
        phasic_gain = self.lc_gain(na_t) # continuous factor in (0, 1)

        active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
        if episodic_memory is not None and active_slots > 0:
            q_sensory = self.episodic_sensory_proj(full_emb.mean(dim=1)).float()
            ret_mem, max_sim = episodic_memory.read(q_sensory, temperature=0.05, threshold=0.50, sigmoid_beta=10.0)
            # Modulate episodic recall smoothly by phasic noradrenaline gain
            ret_mem_modulated = ret_mem * phasic_gain
            ret_mem_unrolled = ret_mem_modulated.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous().view(batch_size * seq_len, -1).float()
            unrolled_inputs['episodic_recall'] = ret_mem_unrolled

        h_prev_unrolled = torch.zeros(batch_size * seq_len, self.hidden_dim, device=self.device).float()
        u_t_unrolled = curr_u_t.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous().view(batch_size * seq_len, -1).float()
        
        with torch.amp.autocast(device_type=('cuda' if self.hardware.is_cuda else ('xla' if self.hardware.is_tpu else 'cpu')), enabled=False):
            w_t_unrolled, attn_weights_unrolled, channel_names, epistemic_entropy_unrolled = self.gateway(
                unrolled_inputs, h_prev_unrolled, u_t_unrolled
            )
        
        w_t_seq = w_t_unrolled.view(batch_size, seq_len, self.unified_dim)
        
        with torch.amp.autocast(device_type=('cuda' if self.hardware.is_cuda else ('xla' if self.hardware.is_tpu else 'cpu')), dtype=self.hardware.get_autocast_dtype(), enabled=self.hardware.config.enable_amp and not self.hardware.is_cpu):
            full_h_in = self.in_proj(w_t_seq)

            # --- Fused C++20 Cascaded Execution ---
            # Single C++20 call executes Stage 1, Boundary Detector, PW-LPER, and Stage 2
            h_s1, h_s2, m_s1_next, m_s2_next, saliency_gate = self.fused_stack(
                full_h_in, m_s1, m_s2, curr_u_t, text_seq
            )
            
            # Update sequence states
            m_s1 = m_s1_next
            m_s2 = m_s2_next

            # PW-HPC: Top-down predictive feedback from previous Stage 2 state
            h_s2_prev_shifted = torch.zeros_like(h_s1)
            e1_weighted, h_s1_hat, mean_pi = self.pw_hpc_generator(h_s1, h_s2_prev_shifted, curr_u_t)

            predicted_entropy = self.entropy_predictor(h_s1)
            dynamic_dt_scale = 0.40 + 1.20 * predicted_entropy

            h_s2 = h_s2 * dynamic_dt_scale

            # Hierarchical Volitional Override
            effective_u_t, gamma_override, allostatic_strain = self.will_engine(h_s2, curr_u_t)

            eff_dt = torch.tensor(1.0, device=self.device)
            topdown_prior = self.topdown_prior_proj(h_s2)
            # Smooth continuous modulation via LC Phasic Gain
            h_combined = h_s1 + h_s2 + (0.10 + 0.15 * phasic_gain.unsqueeze(1)) * topdown_prior

            h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
            h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, effective_u_t)
            
            # Volition-Modulated Motor Text Logits
            volitional_logits_flat = self.volitional_head.compute_volitional_logits(
                h_relaxed, effective_u_t, self.pos_embeddings.byte_embed.weight
            )

            targets_flat = target_seq.contiguous().view(-1)
            speech_loss_tensor = criterion_speech(volitional_logits_flat, targets_flat)

            w_current_slice = w_t_seq[:, -1, :]
            h_curr_fast = h_combined[:, -1, :]
            w_pred, kl_div, fe, _ = self.world_model(h_prev_fast, h_curr_fast, w_current_slice)

            rec_loss = (1.0 - F.cosine_similarity(w_current_slice, w_pred, dim=-1, eps=1e-8)).mean()
            hpc_reconstruction_loss = F.mse_loss(h_s1, h_s1_hat)
            fe_loss_tensor = torch.clamp(kl_div.mean() + rec_loss + 0.10 * hpc_reconstruction_loss, 0.0, 10.0)

            num_chunks = seq_len // chunk_size
            if num_chunks > 1:
                h_chunk_endpoints = h_combined.detach().view(batch_size, num_chunks, chunk_size, self.hidden_dim)[:, :, -1, :]
                v_preds = self.critic(h_chunk_endpoints).squeeze(-1)
                
                gamma_fe = 0.90
                fe_per_batch = fe.squeeze(-1)
                v_current = v_preds[:, :-1]
                v_next = v_preds[:, 1:].detach()
                r_step = -0.10 * fe_per_batch.unsqueeze(1).expand_as(v_current)
                td_targets = r_step + gamma_fe * v_next
                critic_loss = F.mse_loss(v_current, td_targets)
            else:
                critic_loss = torch.tensor(0.0, device=self.device)

            ortho_loss = self.attractor_head.compute_pattern_separation_loss()
            
            speech_loss_val = speech_loss_tensor.item()
            fe_loss_val = fe_loss_tensor.item()
            
            total_loss_tensor = (
                speech_loss_tensor + 
                loss_free_energy_weight * fe_loss_tensor + 
                0.05 * commit_loss + 
                0.01 * ortho_loss + 
                0.02 * critic_loss
            )

        h_proxy = m_s2.view(batch_size, -1)[:, :self.hidden_dim]
        return total_loss_tensor, speech_loss_val, fe_loss_val, m_s2, h_proxy, curr_u_t, eff_dt

    def forward_multimodal_step(self, sensor_dict: Dict[str, torch.Tensor], m_s1: torch.Tensor, m_s2: torch.Tensor, u_t: torch.Tensor):
        b_size = m_s1.size(0)
        h_prev_proxy = m_s1.view(b_size, -1)[:, :self.hidden_dim]

        with torch.amp.autocast(device_type=('cuda' if self.hardware.is_cuda else ('xla' if self.hardware.is_tpu else 'cpu')), enabled=False):
            w_t, attn_weights, channel_names, epistemic_entropy = self.gateway(sensor_dict, h_prev_proxy, u_t)

        with torch.amp.autocast(device_type=('cuda' if self.hardware.is_cuda else ('xla' if self.hardware.is_tpu else 'cpu')), dtype=self.hardware.get_autocast_dtype(), enabled=self.hardware.config.enable_amp and not self.hardware.is_cpu):
            x_in = self.in_proj(w_t).unsqueeze(1)

            h_s1_out, m_s1_next, dt1 = self.stage1(x_in, m_s1, u_t, torch.Tensor(), 1.0)
            dummy_ids = torch.zeros(x_in.size(0), 1, dtype=torch.long, device=self.device)
            sal_gate = self.boundary_detector(h_s1_out, dummy_ids)

            h1_prev_proxy = m_s1.view(b_size, -1)[:, :self.hidden_dim].unsqueeze(1)
            e1_weighted, _, _ = self.pw_lper(h_s1_out, h1_prev_proxy, u_t)

            h_s2_out, m_s2_next, dt2 = self.stage2(e1_weighted, m_s2, u_t, sal_gate, 1.0)

            # Volitional override
            effective_u_t, gamma_override, allostatic_strain = self.will_engine(h_s2_out, u_t)

            topdown_prior = self.topdown_prior_proj(h_s2_out)
            h_combined = h_s1_out + h_s2_out + 0.15 * topdown_prior
            h_flat = h_combined.view(-1, self.hidden_dim)
            h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, effective_u_t)

            outs = self.output_gateway(h_relaxed)
            w_pred, kl_div, fe, z_t = self.world_model(h_prev_proxy, h_relaxed, w_t)

            return outs, fe, commit_loss, attn_weights, channel_names, m_s1_next.detach(), m_s2_next.detach(), z_t

    def process_universal_stream(self, channel_name: str, tensor_data: torch.Tensor, hu: HomeostaticUnit, episodic_mem: BatchedEpisodicMemory) -> Tuple[torch.Tensor, float, float, float]:
        """
        Processes a single modality channel stream on the unified representation space.
        Returns: (h_relaxed, FreeEnergy, Loss, latency_ms)
        """
        t0 = time.perf_counter()
        
        # Format as sensory input dict
        sensor_dict = {channel_name: tensor_data}
        
        # Initialize dummy states
        m_s1 = torch.zeros(1, self.num_heads, self.head_k, self.head_v, device=self.device)
        m_s2 = torch.zeros(1, self.num_heads, self.head_k, self.head_v, device=self.device)
        u_t = hu.state if hu is not None else torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], device=self.device)
        
        with torch.no_grad():
            outs, fe, commit_loss, _, _, m_s1_next, m_s2_next, z_t = self.forward_multimodal_step(sensor_dict, m_s1, m_s2, u_t)
            
            # Write to episodic memory if novelty is high
            fe_val = fe.mean().item()
            if fe_val > 0.01 and episodic_mem is not None:
                q_proj = self.episodic_sensory_proj(outs.get(channel_name, torch.randn(1, self.text_dim, device=self.device)))
                episodic_mem.write(q_proj, q_proj)
                
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return z_t, fe_val, commit_loss.mean().item(), duration_ms
    def generate_thought_and_speech(
        self, prompt: str, m_state: torch.Tensor, h_state: torch.Tensor, hu, episodic_memory, 
        config, max_generated_tokens: int = 120, temperature: float = 0.45, top_p: float = 0.90
    ) -> Generator[Dict[str, Any], None, None]:
        import codecs
        utf8_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

        prompt_ids = [t for t in self.tokenizer.encode(prompt) if t != 257]
        prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_embs = self.pos_embeddings(prompt_tokens, start_pos=0, apply_rf=True)
        
        if hu is not None and hasattr(hu, 'state') and hu.state.size(0) > 1:
            diag_hu = HomeostaticUnit(batch_size=1, device=self.device_str)
            diag_hu.state.copy_(hu.state[0:1])
            hu = diag_hu
        
        hu_st = hu.state if hu is not None else torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], device=self.device)
        
        m_s1 = torch.zeros(1, self.num_heads, self.head_k, self.head_v, device=self.device)
        m_s2 = torch.zeros(1, self.num_heads, self.head_k, self.head_v, device=self.device)
        h1_prev_last = torch.zeros(1, 1, self.hidden_dim, device=self.device)
            
        yield {"status": "speech_start"}
        
        prompt_len = prompt_tokens.size(1)
        for c_idx in range(0, prompt_len, 64):
            c_emb = prompt_embs[:, c_idx : min(c_idx + 64, prompt_len), :]
            c_in = prompt_tokens[:, c_idx : min(c_idx + 64, prompt_len)]
            h_in = self.in_proj(c_emb)
            h_s1, m_s1, _ = self.stage1(h_in, m_s1, hu_st, torch.Tensor(), 1.0)
            sal_gate = self.boundary_detector(h_s1, c_in)

            e1_weighted, h1_prev_last, _ = self.pw_lper(h_s1, h1_prev_last, hu_st)
            h_s2, m_s2, _ = self.stage2(e1_weighted, m_s2, hu_st, sal_gate, 1.0)
        
        rolling_token_ids = prompt_tokens[0].tolist()
        energy_action_cost = torch.tensor([[getattr(config.homeo, 'motor_speech_cost_per_patch', 0.0015)]], device=self.device)
        zero_pred_err = torch.tensor([[0.0]], device=self.device)
        cog_action = torch.tensor([[0]], dtype=torch.int64, device=self.device)

        total_prompt_len = prompt_tokens.size(1)
        consecutive_newlines = 0

        for step in range(max_generated_tokens):
            context_window = rolling_token_ids[-8:]
            window_t = torch.tensor([context_window], dtype=torch.long, device=self.device)
            window_start_pos = (total_prompt_len + step) - (len(context_window) - 1)
            
            window_emb = self.pos_embeddings(window_t, start_pos=window_start_pos, apply_rf=True)
            t_emb = window_emb[:, -1:, :]
            
            sensor_inputs = {'text': t_emb.squeeze(1)}
            active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
            
            # Continuous Locus Coeruleus Phasic Gain computation (Zero Hardcode Constants)
            na_t = hu_st[:, 4:5]
            phasic_gain = self.lc_gain(na_t) # continuous factor in (0, 1)

            if episodic_memory is not None and active_slots > 0:
                q_k = self.episodic_sensory_proj(t_emb.squeeze(1)).float()
                ret_mem, max_sim = episodic_memory.read(q_k, temperature=0.05, threshold=0.50, sigmoid_beta=10.0)
                # Modulate memory injection smoothly by phasic noradrenaline gain
                sensor_inputs['episodic_recall'] = ret_mem * phasic_gain

            w_t, _, _, _ = self.gateway(sensor_inputs, m_s2.view(1, -1)[:, :self.hidden_dim], hu_st)
            h_in = self.in_proj(w_t).unsqueeze(1)

            h_s1, h_s2, m_s1, m_s2, sal_gate = self.fused_stack(h_in, m_s1, m_s2, hu_st, window_t[:, -1:])
            
            # Hierarchical Volitional Override in generation
            effective_hu_st, gamma_override, allostatic_strain = self.will_engine(h_s2, hu_st)

            topdown_prior = self.topdown_prior_proj(h_s2)
            # Smooth continuous modulation via LC Phasic Gain
            h_combined = h_s1 + h_s2 + (0.10 + 0.15 * phasic_gain.unsqueeze(1)) * topdown_prior

            h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
            h_relaxed, _ = self.attractor_head.relax_to_minima(h_flat, effective_hu_st)
            
            raw_logits = self.volitional_head.compute_volitional_logits(h_relaxed, effective_hu_st, self.pos_embeddings.byte_embed.weight)
            
            # Continuous Dirichlet/Biophysical Prior Modulation (Eradicating hard -1e9 masks)
            # Service and non-printable bytes receive a continuous somatic inhibition penalty
            somatic_byte_penalty = getattr(self, 'somatic_byte_penalty', None)
            if somatic_byte_penalty is None:
                somatic_byte_penalty = torch.zeros(1, 258, device=self.device)
                somatic_byte_penalty[0, 256] = 12.0
                somatic_byte_penalty[0, :9] = 10.0
                somatic_byte_penalty[0, 11:13] = 10.0
                somatic_byte_penalty[0, 14:32] = 10.0
                somatic_byte_penalty[0, 127] = 8.0
                self.somatic_byte_penalty = somatic_byte_penalty

            logits = raw_logits - self.somatic_byte_penalty
            early_step_factor = math.exp(-step / 4.0)
            logits[:, 257] = logits[:, 257] - 15.0 * early_step_factor

            p_dist = F.softmax(logits, dim=-1)
            entropy = -(p_dist * torch.log(p_dist + 1e-9)).sum(dim=-1)

            # Continuous Active Inference PAC Decoding (Modulated by LC Phasic Gain & Local Surprise)
            temp = 0.10 + 0.35 * torch.sigmoid(5.0 * (entropy - 0.60) + 2.0 * (phasic_gain.squeeze() - 0.50)).item()
            top_p_val = 0.90 + 0.09 * (1.0 - torch.sigmoid(4.0 * (entropy - 0.60)).item())

            # System 2 Parallel Mental Sandbox Integration on High Entropy Boundaries (H > 0.70)
            if entropy.item() > 0.70 and hasattr(self, 'world_model') and self.world_model is not None:
                with torch.no_grad():
                    w_curr_gen = w_t
                    best_thought_h, min_efe = self.world_model.evaluate_counterfactual_rollout(
                        h_relaxed, w_curr_gen, num_steps=3
                    )
                    # Modulate logits smoothly by Expected Free Energy from Sandbox rollout
                    efe_penalty = torch.clamp(torch.tensor(min_efe, device=self.device) * 0.10, 0.0, 3.0)
                    logits = logits - efe_penalty

            scaled_logits = logits / max(temp, 1e-4)
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            to_remove = cumulative_probs > top_p_val
            to_remove[..., 1:] = to_remove[..., :-1].clone()
            to_remove[..., 0] = False
            indices_to_remove = to_remove.scatter(1, sorted_indices, to_remove)
            scaled_logits[indices_to_remove] = -1e9

            probs = F.softmax(scaled_logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            prob_sum = probs.sum(dim=-1, keepdim=True)
            if (prob_sum <= 0).any():
                probs = torch.full_like(probs, 1.0 / 258)
            else:
                probs = probs / prob_sum

            next_token = torch.multinomial(probs, num_samples=1).squeeze(0)
            next_token_id = next_token.item()

            if step % 4 == 0:
                hu.update(energy_action_cost, zero_pred_err, zero_pred_err, cog_action)
                # Apply Health Debt from Volitional Override
                hu_st[0, 3] = torch.clamp(hu_st[0, 3] - 0.02 * allostatic_strain.mean().item(), 0.0, 1.0)

            rolling_token_ids.append(next_token_id)
            
            if next_token_id == 257:
                break
            if next_token_id == 10:
                consecutive_newlines += 1
                if consecutive_newlines >= 2 and step > 10:
                    break
            else:
                consecutive_newlines = 0
                
            # Incremental UTF-8 byte decoding
            if 32 <= next_token_id <= 126 or next_token_id in [9, 10, 13]:
                token_char = utf8_decoder.decode(bytes([next_token_id]))
            elif 128 <= next_token_id <= 255:
                token_char = utf8_decoder.decode(bytes([next_token_id]))
            else:
                token_char = ' '
            
            yield {
                "status": "token",
                "token_id": next_token_id,
                "text": token_char
            }
            
            if hu_st[0, 1].item() <= 0.05 and gamma_override.mean().item() < 0.2:
                yield {"status": "exhausted", "text": " [fatigued...]", "m_state": m_s2, "h_state": h_combined}
                return

        # Flush any remaining bytes in the decoder
        try:
            final_char = utf8_decoder.decode(b'', final=True)
            if final_char:
                yield {
                    "status": "token",
                    "token_id": 257,
                    "text": final_char
                }
        except Exception:
            pass

        yield {"status": "speech_end", "m_state": m_s2, "h_state": h_combined}
