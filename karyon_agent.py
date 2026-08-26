# karyon_agent.py
"""
===============================================================================
KARYON AGENT CORE v22.0 MASTER (ACTIVE INFERENCE NATIVE C++20 CORTICAL ARCHITECTURE)
Grounded in Principle 1 (C++20 as Engine, Python as Client) & Principle 2 (Biological Realism):
- 100% Native C++20 2-Stage Cascaded Cortical Stack (Fast Morpho-Syntactic + Slow Semantic)
- Native Precision-Weighted Laminar Error Routing (PW-LPER - EXP-75 & EXP-81 Validated)
- Native Multi-Scale Morphological Byte Pyramid Receptive Field (EXP-70 Validated)
- Native Causal Depthwise ConvSwiGLU Channel-Mixing (EXP-73/EXP-74 Validated)
- Native Exact Parallel Log-Space Cumulative Retention Decay Scan with RoPE & Mamba-2 Gating
- Native Entropy-Adaptive Word/Morpheme Boundary Detector (EABS)
- Theta-Gamma PAC Entropy-Adaptive Decoding & Mamba-2 Head Equalization
- Active Inference Latent World Model & Autocast-Safe Episodic Projections
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import math
from typing import Generator, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    DesaturatedHopfieldAttractorHead,
    LatentPredictor,
    BatchedEpisodicMemory
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
        tok_emb = self.byte_embed(input_ids)
        pos_emb = self.pe[:, start_pos : start_pos + seq_len, :]
        embedded = tok_emb + pos_emb
        if apply_rf and seq_len > 1:
            embedded = self.receptive_field(embedded)
        return embedded


# =============================================================================
# MASTER CORE AGENT (v22.0 PROD MASTER)
# =============================================================================

class CoREAgent(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.device_str = 'cuda' if (str(device).find('cuda') != -1 and torch.cuda.is_available()) else 'cpu'
        self.device = torch.device(self.device_str)
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
        
        # 1. Multi-Modal Sensory Gateway (Global Workspace)
        self.gateway = SensoryGateway(
            unified_dim=self.unified_dim, 
            hidden_dim=self.hidden_dim, 
            homeo_dim=config.net.homeo_dim, 
            text_dim=self.text_dim, 
            vision_dim=config.net.vision_dim, 
            action_dim=config.net.action_dim,
            device=self.device_str
        )
        self.in_proj = nn.Linear(self.text_dim, self.hidden_dim).to(self.device)
        
        # 2. Native C++20 2-Stage Cascaded Cortical Stack
        # Stage 1: Fast Morpho-Syntactic Cortical Sheet (Decay 0.005 - 0.15, ConvSwiGLU K=3)
        self.stage1 = CorticalStage(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.005, max_beta=0.15,
            swiglu_kernel_size=3, device=self.device_str
        )

        # Native Entropy-Adaptive Word/Morpheme Boundary Detector (EABS)
        self.boundary_detector = EntropyAdaptiveBoundaryDetector(hidden_dim=self.hidden_dim, device=self.device_str)

        # Native Precision-Weighted LPER Module (EXP-75 & EXP-81 Validated)
        self.pw_lper = PrecisionWeightedLPER(hidden_dim=self.hidden_dim, device=self.device_str)

        # Stage 2: Slow Semantic-Discourse Cortical Sheet (Decay 0.0001 - 0.05, ConvSwiGLU K=7)
        self.stage2 = CorticalStage(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.0001, max_beta=0.05,
            swiglu_kernel_size=7, device=self.device_str
        )

        # 3. Active Inference Latent World Model (Predictive Coding)
        self.world_model = LatentPredictor(
            hidden_dim=self.hidden_dim,
            unified_dim=self.unified_dim,
            latent_dim=self.latent_dim,
            device=self.device_str
        )
        
        # 4. Multi-Modal Motor Gateway
        self.output_gateway = MotorGateway(
            hidden_dim=self.hidden_dim, 
            action_dim=config.net.action_dim, 
            cog_action_dim=config.net.cog_action_dim, 
            text_gen_dim=self.text_gen_dim,
            device=self.device_str
        )
        
        # 5. Native C++ Modern Hopfield Attractor with Bounded Commitment Loss
        self.attractor_head = DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, 
            vocab_size=self.text_gen_dim,
            num_attractors=getattr(config.net, 'num_attractors', 256),
            device=self.device_str
        )
        
        # 6. Dedicated Episodic Projection (256 -> 256)
        self.episodic_sensory_proj = nn.Linear(self.text_dim, self.unified_dim).to(self.device)

        # 7. Afferent-Efferent Tied Motor Projection Head (768 -> 256)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)
        
        self.critic = nn.Linear(self.hidden_dim, 1).to(self.device)

    def forward(self, sensor_inputs: Dict[str, torch.Tensor], h_fast: torch.Tensor, h_slow: torch.Tensor, u_t: torch.Tensor, dt: float = 1.0):
        text_in = sensor_inputs.get('text', torch.zeros(h_fast.size(0), self.text_dim, device=self.device))
        vision_in = sensor_inputs.get('vision', torch.zeros(h_fast.size(0), self.config.net.vision_dim, device=self.device))
        motor_in = sensor_inputs.get('motor_efference', torch.zeros(h_fast.size(0), self.action_dim, device=self.device))
        
        w_t, attn_weights, channel_names, epistemic_entropy = self.gateway(text_in, vision_in, motor_in, h_slow, u_t)
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
        e1_weighted, _, _ = self.pw_lper(h_s1_out, h1_prev_proxy, u_t)

        h_s2_out, m_s2_next, dt2 = self.stage2(e1_weighted, m_s2, u_t, sal_gate, dt)
        eff_dt = (dt1 + dt2) / 2.0

        h_combined = h_s1_out + h_s2_out
        h_flat = h_combined.view(-1, self.hidden_dim)
        h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, u_t)
        
        motor_outs = self.output_gateway(h_relaxed)
        actions = motor_outs.get("motor_action", torch.zeros(h_fast.size(0), self.action_dim, device=self.device))
        cog_actions = motor_outs.get("cognitive_gating", torch.zeros(h_fast.size(0), self.config.net.cog_action_dim, device=self.device))
        text_logits = motor_outs.get("text_generation", torch.zeros(h_fast.size(0), self.text_gen_dim, device=self.device))

        h_prev_proxy = m_s1.view(h_fast.size(0), -1)[:, :self.hidden_dim]
        w_pred, kl_div, fe, z_t = self.world_model(h_prev_proxy, h_relaxed, w_t)
        
        value_est = self.critic(h_relaxed)
        
        h_fast_next = m_s1_next.view(h_fast.size(0), -1)[:, :self.hidden_dim]
        h_slow_next = m_s2_next.view(h_slow.size(0), -1)[:, :self.hidden_dim]
        
        return (h_fast_next, h_slow_next, actions, cog_actions, text_logits, fe, attn_weights, w_t, w_pred, value_est, epistemic_entropy, eff_dt)

    def get_all_parameters(self) -> List[nn.Parameter]:
        params = (
            list(self.pos_embeddings.parameters()) + 
            list(self.in_proj.parameters()) +
            list(self.boundary_detector.parameters()) +
            list(self.pw_lper.parameters()) +
            list(self.episodic_sensory_proj.parameters()) +
            list(self.motor_text_proj.parameters()) + 
            list(self.critic.parameters())
        )
        for submodule in [self.gateway, self.stage1, self.stage2, self.world_model, self.output_gateway, self.attractor_head]:
            if hasattr(submodule, 'parameters'):
                params.extend(list(submodule.parameters()))
        return params

    def get_complete_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {
            'in_proj.weight': self.in_proj.weight.detach().cpu(),
            'in_proj.bias': self.in_proj.bias.detach().cpu(),
            'critic.weight': self.critic.weight.detach().cpu(),
            'critic.bias': self.critic.bias.detach().cpu(),
            'episodic_sensory_proj.weight': self.episodic_sensory_proj.weight.detach().cpu(),
            'episodic_sensory_proj.bias': self.episodic_sensory_proj.bias.detach().cpu()
        }
        for name, param in self.pos_embeddings.named_parameters():
            sd[f"pos_embeddings.{name}"] = param.detach().cpu()

        for name, param in self.boundary_detector.named_parameters():
            sd[f"boundary_detector.{name}"] = param.detach().cpu()

        for name, param in self.pw_lper.named_parameters():
            sd[f"pw_lper.{name}"] = param.detach().cpu()

        for name, param in self.motor_text_proj.named_parameters():
            sd[f"motor_text_proj.{name}"] = param.detach().cpu()

        for sub_name, sub in [('gateway', self.gateway), ('stage1', self.stage1), ('stage2', self.stage2), 
                              ('world_model', self.world_model), ('output_gateway', self.output_gateway), 
                              ('attractor_head', self.attractor_head)]:
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
            elif name.startswith("in_proj."):
                p_name = name.replace("in_proj.", "")
                if hasattr(self.in_proj, p_name):
                    self._safe_copy_param(getattr(self.in_proj, p_name).data, tensor)
            elif name.startswith("episodic_sensory_proj."):
                p_name = name.replace("episodic_sensory_proj.", "")
                if hasattr(self.episodic_sensory_proj, p_name):
                    self._safe_copy_param(getattr(self.episodic_sensory_proj, p_name).data, tensor)
            elif name.startswith("critic.weight"):
                self._safe_copy_param(self.critic.weight.data, tensor)
            elif name.startswith("critic.bias"):
                self._safe_copy_param(self.critic.bias.data, tensor)
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
                                      num_replay_cycles: int = 5, downscaling_factor: float = 0.03):
        self.train()
        active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
        active_memory_slots = min(active_slots, episodic_memory.max_capacity)
        
        if active_memory_slots > 3:
            opt_replay = torch.optim.AdamW(self.get_all_parameters(), lr=5e-4, weight_decay=0.01)
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

        # Synaptic Downscaling by Tononi (SHY)
        with torch.no_grad():
            for param in self.get_all_parameters():
                if param.dim() > 1:
                    param.mul_(1.0 - downscaling_factor)

        # Full Somatic Recovery
        with torch.no_grad():
            hu.state[:, 1] = 1.00
            hu.state[:, 2] = 1.00
            hu.state[:, 3] = 1.00
            hu.state[:, 4] = 0.05

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion_speech: nn.Module, episodic_memory=None, loss_free_energy_weight: float = 0.05, 
                         chunk_size: int = 64) -> Tuple[torch.Tensor, float, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_seq.size()
        
        m_s1 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()
        h_prev_fast = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        h1_prev_last = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)
        
        # 1. Vectorized full-sequence embedding, receptive field, and linear projection
        full_emb = self.pos_embeddings(input_seq, start_pos=0, apply_rf=True)
        full_h_in = self.in_proj(full_emb)
        
        num_chunks = max(1, seq_len // chunk_size)
        chunk_losses = []
        commit_losses = []
        fe_losses = []
        last_eff_dt = torch.tensor([[1.0]], device=self.device)

        da_level = curr_u_t[:, 5:6]
        motor_gain = (1.0 + 1.0 * da_level).unsqueeze(1)

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)
            c_len = c_end - c_start

            chunk_in = input_seq[:, c_start:c_end]
            chunk_tgt = target_seq[:, c_start:c_end]
            h_in = full_h_in[:, c_start:c_end, :]

            # Stage 1: Fast Morpho-Syntactic Cortical Pass (Native C++)
            h_s1, m_s1, dt1 = self.stage1(h_in, m_s1, curr_u_t, torch.Tensor(), 1.0)

            # Detect Dynamic Word / Morpheme Boundary Saliency (EABS Native C++)
            saliency_gate = self.boundary_detector(h_s1, chunk_in)

            # --- PRECISION-WEIGHTED LAMINAR ERROR ROUTING (PW-LPER Native C++) ---
            e1_weighted, h1_prev_last, _ = self.pw_lper(h_s1, h1_prev_last, curr_u_t)

            # Stage 2: Slow Semantic-Discourse Pass on Precision-Weighted Error e1_weighted (Native C++)
            h_s2, m_s2, dt2 = self.stage2(e1_weighted, m_s2, curr_u_t, saliency_gate, 1.0)
            last_eff_dt = (dt1 + dt2) / 2.0

            # Combined Laminar Representation (Stage 1 + Stage 2)
            h_combined = h_s1 + h_s2

            # Modern Hopfield Attractor Landscape with Native C++ Commitment Loss
            h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
            h_relaxed, chunk_commit = self.attractor_head.relax_to_minima(h_flat, curr_u_t)
            
            # Dopaminergic Afferent-Efferent Motor Readout (768 -> 256 -> 258)
            h_proj = self.motor_text_proj(h_relaxed).view(batch_size, c_len, self.text_dim)
            h_proj_gain = (h_proj * motor_gain).contiguous().view(-1, self.text_dim)
            logits_flat = F.linear(h_proj_gain, self.pos_embeddings.byte_embed.weight)

            targets_flat = chunk_tgt.contiguous().view(-1)
            chunk_loss = criterion_speech(logits_flat, targets_flat)
            chunk_losses.append(chunk_loss)
            commit_losses.append(chunk_commit)

            # Continuous Active Inference: World Model Predictor
            w_current_slice = self.episodic_sensory_proj(full_emb[:, c_end - 1, :])
            h_curr_fast = h_combined[:, -1, :]
            w_pred, kl_div, _, _ = self.world_model(h_prev_fast, h_curr_fast, w_current_slice)
            h_prev_fast = h_curr_fast.detach()

            rec_loss = (1.0 - F.cosine_similarity(w_current_slice, w_pred, dim=-1, eps=1e-8)).mean()
            chunk_fe = (kl_div.mean() + rec_loss)
            fe_losses.append(chunk_fe)

            # High-Surprise Episodic Encoding
            with torch.no_grad():
                if chunk_fe.item() > 0.20 and episodic_memory is not None:
                    episodic_memory.write(w_current_slice.detach().float(), w_pred.detach().float(), protected_slots=3)

                has_eos = (chunk_in == 257).any(dim=-1).view(batch_size, 1, 1, 1).float()
                m_s1 = m_s1 * (1.0 - has_eos)
                m_s2 = m_s2 * (1.0 - has_eos)

            m_s1 = m_s1.detach()
            m_s2 = m_s2.detach()

        avg_speech_loss_tensor = torch.stack(chunk_losses).mean()
        avg_commit_loss_tensor = torch.stack(commit_losses).mean()
        avg_fe_loss_tensor = torch.stack(fe_losses).mean()
        ortho_loss = self.attractor_head.compute_pattern_separation_loss()
        
        avg_speech_loss_val = avg_speech_loss_tensor.item()
        avg_fe_loss_val = avg_fe_loss_tensor.item()
        
        total_loss_tensor = (
            avg_speech_loss_tensor + 
            loss_free_energy_weight * avg_fe_loss_tensor + 
            0.05 * avg_commit_loss_tensor + 
            0.01 * ortho_loss
        )

        h_proxy = m_s2.view(batch_size, -1)[:, :self.hidden_dim]
        return total_loss_tensor, avg_speech_loss_val, avg_fe_loss_val, m_s2, h_proxy, curr_u_t, last_eff_dt

    def generate_thought_and_speech(
        self, prompt: str, m_state: torch.Tensor, h_state: torch.Tensor, hu, episodic_memory, 
        config, max_generated_tokens: int = 120, temperature: float = 0.45, top_p: float = 0.90
    ) -> Generator[Dict[str, Any], None, None]:
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
            
            h_in = self.in_proj(t_emb)
            h_s1, m_s1, _ = self.stage1(h_in, m_s1, hu_st, torch.Tensor(), 1.0)
            sal_gate = self.boundary_detector(h_s1, window_t[:, -1:])

            e1_weighted, h1_prev_last, _ = self.pw_lper(h_s1, h1_prev_last, hu_st)
            h_s2, m_s2, _ = self.stage2(e1_weighted, m_s2, hu_st, sal_gate, 1.0)
            h_combined = h_s1 + h_s2

            # Volitional Memory Recall
            active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
            if episodic_memory is not None and hu_st[0, 4].item() > 0.12 and active_slots > 0:
                q_k = self.episodic_sensory_proj(t_emb.squeeze(1)).float()
                ret_mem, max_sim = episodic_memory.read(q_k, temperature=0.05, threshold=0.75, sigmoid_beta=15.0)
                if (max_sim > 0.75).any():
                    ret_mem_cast = ret_mem.to(h_combined.dtype)
                    ret_mem_proj = self.in_proj(ret_mem_cast).unsqueeze(1)
                    h_combined = h_combined + ret_mem_proj * 0.20

            h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
            h_relaxed, _ = self.attractor_head.relax_to_minima(h_flat, hu_st)
            h_proj = self.motor_text_proj(h_relaxed)
            raw_logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight)
            
            # Mask PAD, non-printable control characters, and non-ASCII bytes
            raw_logits[:, 256] = -1e9
            raw_logits[:, :9] = -1e9
            raw_logits[:, 11:13] = -1e9
            raw_logits[:, 14:32] = -1e9
            raw_logits[:, 127:256] = -1e9
            if step < 10:
                raw_logits[:, 257] = -1e9

            # Theta-Gamma PAC Entropy-Adaptive Decoding
            p_dist = F.softmax(raw_logits, dim=-1)
            entropy = -(p_dist * torch.log(p_dist + 1e-9)).sum(dim=-1).item()
            is_boundary = (len(rolling_token_ids) > 0 and rolling_token_ids[-1] in [32, 10, 44, 46])

            if is_boundary or entropy > 0.70:
                temp = 0.45
                top_p_val = 0.88
            else:
                temp = 0.08
                top_p_val = 0.99

            logits = raw_logits / max(temp, 1e-4)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            to_remove = cumulative_probs > top_p_val
            to_remove[..., 1:] = to_remove[..., :-1].clone()
            to_remove[..., 0] = False
            indices_to_remove = to_remove.scatter(1, sorted_indices, to_remove)
            logits[indices_to_remove] = -1e9

            probs = F.softmax(logits, dim=-1)
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
            rolling_token_ids.append(next_token_id)
            
            if next_token_id == 257:
                break
            if next_token_id == 10:
                consecutive_newlines += 1
                if consecutive_newlines >= 2 and step > 10:
                    break
            else:
                consecutive_newlines = 0
                
            token_char = chr(next_token_id) if 32 <= next_token_id <= 126 or next_token_id in [9, 10, 13] else ' '
            
            yield {
                "status": "token",
                "token_id": next_token_id,
                "text": token_char
            }
            
            if hu_st[0, 1].item() <= 0.05:
                yield {"status": "exhausted", "text": " [fatigued...]", "m_state": m_s2, "h_state": h_combined}
                return

        yield {"status": "speech_end", "m_state": m_s2, "h_state": h_combined}
