# karyon_agent.py
"""
===============================================================================
KARYON AGENT CORE v18.5 (ALLOSTATIC ACTIVE INFERENCE MASTER)
Grounded in Principle 2: Continuous Multi-Timescale State-Space Duality,
Dynamic Allostatic Regimes (Wakeful SWR Micro-Replay & Deep Sleep Consolidation),
Dopamine-Modulated Modern Hopfield Attractors (N=256 Basins), and Tied Lexical Readout.
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
    CalibratedParallelSSDCore,
    ParallelSwiGLUBlock,
    DesaturatedHopfieldAttractorHead,
    LatentPredictor,
    BatchedEpisodicMemory
)


# =============================================================================
# MODULE 1: UNSHACKLED 256D POSITIONAL BYTE EMBEDDING
# =============================================================================

class OffsetPositionalByteEmbedding(nn.Module):
    def __init__(self, vocab_size=258, text_dim=256, max_len=8192, device_str='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.text_dim = text_dim
        self.byte_embed = nn.Embedding(vocab_size, text_dim)
        self.receptive_field = CausalByteReceptiveField(text_dim=text_dim, kernel_size=4, device=device_str)
        
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
# MASTER CORE AGENT (v18.5 ALLOSTATIC MASTER)
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
        self.latent_dim = getattr(config.net, 'latent_dim', 128)
        self.text_gen_dim = getattr(config.net, 'text_gen_dim', 258)
        self.num_heads = getattr(config.net, 'num_heads', 8)
        self.head_k = getattr(config.net, 'head_k', 32)
        self.head_v = getattr(config.net, 'head_v', 64)
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)
        
        self.tokenizer = ByteTokenizer(vocab_size=self.text_gen_dim)
        
        self.pos_embeddings = OffsetPositionalByteEmbedding(
            vocab_size=self.text_gen_dim, 
            text_dim=self.text_dim,
            max_len=8192,
            device_str=self.device_str
        ).to(self.device)
        self.text_embeddings = self.pos_embeddings.byte_embed
        
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
        
        # 2. Native C++20 Time-Mixing Multi-Timescale SSD Core
        self.ssd_core = CalibratedParallelSSDCore(
            text_dim=self.text_dim,
            unified_dim=self.unified_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            head_k=self.head_k,
            head_v=self.head_v,
            device=self.device_str
        )
        
        # 3. Native C++20 Channel-Mixing SwiGLU Knowledge Synthesis Block (2048 dim)
        self.channel_mixer = ParallelSwiGLUBlock(
            hidden_dim=self.hidden_dim,
            expand_dim=getattr(config.net, 'expand_dim', 2048),
            device=self.device_str
        )
        
        # 4. Active Inference Latent World Model (Predictive Coding)
        self.world_model = LatentPredictor(
            hidden_dim=self.hidden_dim,
            unified_dim=self.unified_dim,
            latent_dim=self.latent_dim,
            device=self.device_str
        )
        
        # 5. Multi-Modal Motor Gateway
        self.output_gateway = MotorGateway(
            hidden_dim=self.hidden_dim, 
            action_dim=config.net.action_dim, 
            cog_action_dim=config.net.cog_action_dim, 
            text_gen_dim=self.text_gen_dim,
            device=self.device_str
        )
        
        # 6. Dense Modern Hopfield Attractor Memory Landscape (N=256 Basins)
        self.attractor_head = DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, 
            vocab_size=self.text_gen_dim,
            num_attractors=getattr(config.net, 'num_attractors', 256),
            device=self.device_str
        )
        
        # 7. Dedicated Episodic Projection (256 -> 256)
        self.episodic_sensory_proj = nn.Linear(self.text_dim, self.unified_dim).to(self.device)

        # Afferent-Efferent Tied Motor Projection Head (512 -> 256)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)
        
        self.critic = nn.Linear(self.hidden_dim, 1).to(self.device)

    def get_all_parameters(self) -> List[nn.Parameter]:
        params = (
            list(self.pos_embeddings.parameters()) + 
            list(self.episodic_sensory_proj.parameters()) + 
            list(self.motor_text_proj.parameters()) + 
            list(self.critic.parameters())
        )
        for submodule in [self.gateway, self.ssd_core, self.channel_mixer, self.world_model, self.output_gateway, self.attractor_head]:
            if hasattr(submodule, 'parameters'):
                params.extend(list(submodule.parameters()))
        return params

    def get_complete_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {
            'text_embeddings.weight': self.pos_embeddings.byte_embed.weight.detach().cpu(),
            'critic.weight': self.critic.weight.detach().cpu(),
            'critic.bias': self.critic.bias.detach().cpu(),
            'episodic_sensory_proj.weight': self.episodic_sensory_proj.weight.detach().cpu(),
            'episodic_sensory_proj.bias': self.episodic_sensory_proj.bias.detach().cpu()
        }
        for name, param in self.pos_embeddings.named_parameters():
            sd[f"pos_embeddings.{name}"] = param.detach().cpu()

        for name, param in self.motor_text_proj.named_parameters():
            sd[f"motor_text_proj.{name}"] = param.detach().cpu()

        for sub_name, sub in [('gateway', self.gateway), ('ssd_core', self.ssd_core), 
                              ('channel_mixer', self.channel_mixer), ('world_model', self.world_model), 
                              ('output_gateway', self.output_gateway), ('attractor_head', self.attractor_head)]:
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
            elif name.startswith("ssd_core."):
                p_name = name.replace("ssd_core.", "")
                for sub_p_name, sub_p in self.ssd_core.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor)
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
        """Executes ultra-fast (O(1)) awake Sharp-Wave Ripple micro-replay in background."""
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
        """Executes deep nocturnal sleep consolidation: Full Replay + SHY Downscaling + Somatic Reset."""
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
            hu.state[:, 1] = 1.00 # Energy restored
            hu.state[:, 2] = 1.00 # Stability restored
            hu.state[:, 3] = 1.00 # Health restored
            hu.state[:, 4] = 0.05 # Arousal reset to baseline

    def forward_step(self, sensor_inputs: Dict[str, torch.Tensor], h_prev_fast: torch.Tensor, 
                     h_prev_slow: torch.Tensor, u_t: torch.Tensor, episodic_memory=None, 
                     dt: float = 1.0, attention_temp: float = 0.05):
        batch_size = h_prev_fast.size(0)
        
        text_in = sensor_inputs.get('text', torch.zeros(batch_size, self.config.net.text_dim, device=self.device))
        if text_in.dim() == 3:
            text_in = text_in.reshape(batch_size, self.config.net.text_dim)
            
        vision_in = sensor_inputs.get('vision', torch.zeros(batch_size, self.config.net.vision_dim, device=self.device))
        motor_in = sensor_inputs.get('motor_efference', torch.zeros(batch_size, self.config.net.action_dim, device=self.device))
        
        w_current, attn_weights, channel_names, epistemic_entropy = self.gateway(
            text_in, vision_in, motor_in, h_prev_fast, u_t
        )
        
        curiosity     = u_t.select(1, 0).unsqueeze(1)
        energy        = u_t.select(1, 1).unsqueeze(1)
        noradrenaline = u_t.select(1, 4).unsqueeze(1)
        
        volitional_recall_gate = torch.sigmoid(2.0 * noradrenaline + 1.5 * curiosity - 0.5 * (1.0 - energy))
        na_trigger = getattr(self.config.memory, 'volitional_na_trigger', 0.12)
        active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
        should_search_memory = (episodic_memory is not None) and (noradrenaline.mean().item() > na_trigger) and (active_slots > 0)

        if should_search_memory:
            with torch.no_grad():
                retrieved_memory, max_sim = episodic_memory.read(
                    w_current.detach(), attention_temp, 
                    self.config.memory.default_read_threshold,
                    self.config.memory.sigmoid_gating_beta
                )
                if retrieved_memory.dim() > 2:
                    retrieved_memory = retrieved_memory.reshape(batch_size, self.unified_dim)
            w_integrated = w_current + retrieved_memory.detach() * volitional_recall_gate
        else:
            w_integrated = w_current
            
        # 1. Time-Mixing SSD
        t_seq = text_in.unsqueeze(1)
        m_dummy = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        ssd_out = self.ssd_core.forward_chunk_parallel_ssd(t_seq, m_dummy, u_t, dt)
        h_ssm, _, eff_dt = ssd_out[0], ssd_out[1], ssd_out[2]
        
        # 2. Channel-Mixing SwiGLU (2048 dim)
        h_reasoned = self.channel_mixer(h_ssm)
        h_next_fast = h_reasoned
        h_next_slow = h_next_fast
        
        w_pred, kl_div, _, z_t = self.world_model(h_prev_fast, h_next_slow, w_current)
        
        cosine_sim = F.cosine_similarity(w_current, w_pred, dim=-1, eps=1e-8).unsqueeze(-1)
        rec_loss = 1.0 - cosine_sim
        free_energy = kl_div + rec_loss

        relax_out = self.attractor_head.relax_to_minima(h_reasoned, u_t)
        h_relaxed = relax_out[0]
        
        outputs = self.output_gateway(h_relaxed)
        h_proj = self.motor_text_proj(h_relaxed)
        tied_text_logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        outputs["text_generation"] = tied_text_logits
        
        state_value = self.critic(h_reasoned)
        return h_next_fast, h_next_slow, outputs, state_value, w_pred, free_energy, kl_div, w_current, attn_weights, channel_names, epistemic_entropy, eff_dt

    def forward(self, *args, **kwargs):
        return self.forward_step(*args, **kwargs)

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion_speech: nn.Module, episodic_memory=None, loss_free_energy_weight: float = 0.05, 
                         chunk_size: int = 64) -> Tuple[torch.Tensor, float, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_seq.size()
        
        m_curr = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()
        h_prev_fast = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        
        num_chunks = max(1, seq_len // chunk_size)
        chunk_losses = []
        fe_losses = []
        last_eff_dt = torch.tensor([[1.0]], device=self.device)

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)
            c_len = c_end - c_start

            chunk_in = input_seq[:, c_start:c_end]
            chunk_tgt = target_seq[:, c_start:c_end]

            chunk_emb = self.pos_embeddings(chunk_in, start_pos=c_start, apply_rf=True)

            # Parallel State-Space Duality Scan
            ssd_out = self.ssd_core.forward_chunk_parallel_ssd(chunk_emb, m_curr, curr_u_t, 1.0)
            h_chunk, m_curr, last_eff_dt = ssd_out[0], ssd_out[1], ssd_out[2]
            
            # SwiGLU Channel Mixer (2048 dim)
            h_reasoned = self.channel_mixer(h_chunk)

            # Modern Hopfield Attractor Landscape (N=256 Basins with DA-modulation)
            h_relaxed, energy_basin = self.attractor_head.relax_to_minima(h_reasoned, curr_u_t)
            
            # Unshackled Tied Lexical Readout (512 -> 256 -> 258)
            h_proj = self.motor_text_proj(h_relaxed)
            logits_flat = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim

            targets_flat = chunk_tgt.contiguous().view(-1)
            chunk_loss = criterion_speech(logits_flat, targets_flat)
            chunk_losses.append(chunk_loss)

            # Continuous Active Inference: World Model Predictor
            w_current_slice = self.episodic_sensory_proj(chunk_emb[:, -1, :])
            h_curr_fast = h_reasoned.view(batch_size, c_len, self.hidden_dim)[:, -1, :]
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
                m_curr = m_curr * (1.0 - has_eos)

            m_curr = m_curr.detach()

        avg_speech_loss_tensor = torch.stack(chunk_losses).mean()
        avg_fe_loss_tensor = torch.stack(fe_losses).mean()
        
        avg_speech_loss_val = avg_speech_loss_tensor.item()
        avg_fe_loss_val = avg_fe_loss_tensor.item()
        
        total_loss_tensor = avg_speech_loss_tensor + loss_free_energy_weight * avg_fe_loss_tensor

        h_proxy = m_curr.view(batch_size, -1)[:, :self.hidden_dim]
        return total_loss_tensor, avg_speech_loss_val, avg_fe_loss_val, m_curr, h_proxy, curr_u_t, last_eff_dt

    def generate_thought_and_speech(
        self, prompt: str, m_state: torch.Tensor, h_state: torch.Tensor, hu, episodic_memory, 
        config, max_generated_tokens: int = 120, temperature: float = 0.45, top_p: float = 0.90
    ) -> Generator[Dict[str, Any], None, None]:
        prompt_ids = [t for t in self.tokenizer.encode(prompt) if t != 257]
        prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_embs = self.pos_embeddings(prompt_tokens, start_pos=0, apply_rf=True)
        
        m_curr = m_state.clone() if m_state is not None else torch.zeros(1, self.num_heads, self.head_k, self.head_v, device=self.device)
        if m_curr.dim() == 2:
            m_curr = torch.zeros(1, self.num_heads, self.head_k, self.head_v, device=self.device)
            
        yield {"status": "speech_start"}
        
        # Parallel prompt processing in Native C++ SSD + SwiGLU
        ssd_out = self.ssd_core.forward_chunk_parallel_ssd(prompt_embs, m_curr, hu.state, 1.0)
        h_ssm, m_curr = ssd_out[0], ssd_out[1]
        h_chunk = self.channel_mixer(h_ssm)
        
        rolling_token_ids = prompt_tokens[0].tolist()
        energy_action_cost = torch.tensor([[getattr(config.homeo, 'motor_speech_cost_per_patch', 0.0015)]], device=self.device)
        zero_pred_err = torch.tensor([[0.0]], device=self.device)
        cog_action = torch.tensor([[0]], dtype=torch.int64, device=self.device)

        total_prompt_len = prompt_tokens.size(1)
        consecutive_newlines = 0

        for step in range(max_generated_tokens):
            context_window = rolling_token_ids[-4:]
            window_t = torch.tensor([context_window], dtype=torch.long, device=self.device)
            window_start_pos = (total_prompt_len + step) - (len(context_window) - 1)
            
            window_emb = self.pos_embeddings(window_t, start_pos=window_start_pos, apply_rf=True)
            t_emb = window_emb[:, -1:, :]
            
            ssd_out = self.ssd_core.forward_chunk_parallel_ssd(t_emb, m_curr, hu.state, 1.0)
            h_step_ssm, m_curr = ssd_out[0], ssd_out[1]
            h_out = self.channel_mixer(h_step_ssm)

            # Volitional Memory Recall (Active Inference Precision Gating)
            active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
            if episodic_memory is not None and hu.state[0, 4].item() > 0.12 and active_slots > 0:
                q_k = self.episodic_sensory_proj(t_emb.squeeze(1)).float()
                ret_mem, max_sim = episodic_memory.read(q_k, temperature=0.05, threshold=0.75, sigmoid_beta=15.0)
                if (max_sim > 0.75).any():
                    ret_mem_cast = ret_mem.to(h_out.dtype)
                    h_out = h_out + ret_mem_cast.repeat(1, h_out.size(1) // ret_mem_cast.size(1) if h_out.size(1) > ret_mem_cast.size(1) else 1)[:, :h_out.size(1)] * 0.20

            h_relaxed, _ = self.attractor_head.relax_to_minima(h_out, hu.state)
            h_proj = self.motor_text_proj(h_relaxed)
            logits = (F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim) / max(temperature, 1e-4)
            logits[:, 256:] = -1e9
            
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
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
            
            if hu.state[0, 1].item() <= 0.05:
                yield {"status": "exhausted", "text": " [fatigued...]", "m_state": m_curr, "h_state": h_out}
                return

        yield {"status": "speech_end", "m_state": m_curr, "h_state": h_out}
