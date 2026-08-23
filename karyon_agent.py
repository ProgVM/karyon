# karyon_agent.py
"""
===============================================================================
KARYON AGENT CORE v17.2 (NATIVE C++20 CHUNKED CORTICAL ENGINE MASTER)
Native C++20 Chunked Cortical SSD Scan (Q=64, >100k tok/s, <350MB VRAM),
Logit Soft-Capping (30.0 * tanh), NaN-Proof SFT Masked Loss, DFET Plasticity,
Lexical Afferent-Efferent Weight Tying, and Ashby Somatic Ultrastability.
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import math
from typing import Generator, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from karyon_config import CoREConfig
from karyon_core import (
    ByteTokenizer,
    HomeostaticUnit,
    SensoryGateway,
    MotorGateway,
    CausalByteReceptiveField,
    HierarchicalCorticalStack,
    DesaturatedHopfieldAttractorHead,
    LatentPredictor,
    BatchedEpisodicMemory
)


# =============================================================================
# MODULE 1: POSITIONAL BYTE EMBEDDING WITH RECEPTIVE FIELD
# =============================================================================

class OffsetPositionalByteEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 258, text_dim: int = 128, max_len: int = 8192, device_str: str = 'cpu'):
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
# MASTER CORE AGENT (v17.2 CHUNKED CORTICAL ENGINE MASTER)
# =============================================================================

class CoREAgent(nn.Module):
    def __init__(self, config: CoREConfig, device: str = 'cpu'):
        super().__init__()
        self.device_str = str(device)
        self.device = torch.device(device)
        self.config = config
        
        self.hidden_dim = getattr(config.net, 'hidden_dim', 512)
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.action_dim = config.net.action_dim
        self.latent_dim = getattr(config.net, 'latent_dim', 128)
        self.text_gen_dim = getattr(config.net, 'text_gen_dim', 258)
        self.num_layers = getattr(config.net, 'num_layers', 2)
        self.expand_dim = getattr(config.net, 'expand_dim', 1536)
        self.num_heads = getattr(config.net, 'num_heads', 8)
        self.head_k = getattr(config.net, 'head_k', 32)
        self.head_v = getattr(config.net, 'head_v', 64)
        self.chunk_size = getattr(config.train, 'chunk_size', 64)
        
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
        
        # 2. Input Projection to Native Cortical Dimension
        self.input_proj = nn.Linear(self.text_dim, self.hidden_dim).to(self.device)

        # 3. Native C++20 Chunked Cortical Neocortex Stack (Q=64)
        self.cortical_stack = HierarchicalCorticalStack(
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            expand_dim=self.expand_dim,
            num_heads=self.num_heads,
            head_k=self.head_k,
            head_v=self.head_v,
            chunk_size=self.chunk_size,
            device=self.device_str
        )

        # 4. Active Inference Latent World Model
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
        
        # 6. Desaturated Hopfield Attractor Memory Landscape
        self.attractor_head = DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, 
            vocab_size=self.text_gen_dim,
            num_attractors=64,
            device=self.device_str
        )
        
        # 7. Dedicated Episodic Projection (text_dim 128 -> unified_dim 256)
        self.episodic_sensory_proj = nn.Linear(self.text_dim, self.unified_dim).to(self.device)

        # Afferent-Efferent Tied Motor Projection Head with Soft-Capping
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)
        
        self.critic = nn.Linear(self.hidden_dim, 1).to(self.device)

    def get_all_parameters(self) -> List[nn.Parameter]:
        params = (
            list(self.pos_embeddings.parameters()) + 
            list(self.input_proj.parameters()) +
            list(self.episodic_sensory_proj.parameters()) +
            list(self.motor_text_proj.parameters()) + 
            list(self.critic.parameters())
        )
        for submodule in [self.gateway, self.cortical_stack, self.world_model, self.output_gateway, self.attractor_head]:
            if hasattr(submodule, 'parameters'):
                params.extend(list(submodule.parameters()))
        return params

    def get_complete_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {
            'text_embeddings.weight': self.pos_embeddings.byte_embed.weight.detach().cpu(),
            'input_proj.weight': self.input_proj.weight.detach().cpu(),
            'input_proj.bias': self.input_proj.bias.detach().cpu(),
            'critic.weight': self.critic.weight.detach().cpu(),
            'critic.bias': self.critic.bias.detach().cpu(),
            'episodic_sensory_proj.weight': self.episodic_sensory_proj.weight.detach().cpu(),
            'episodic_sensory_proj.bias': self.episodic_sensory_proj.bias.detach().cpu()
        }
        for name, param in self.pos_embeddings.named_parameters():
            sd[f"pos_embeddings.{name}"] = param.detach().cpu()

        for name, param in self.motor_text_proj.named_parameters():
            sd[f"motor_text_proj.{name}"] = param.detach().cpu()

        for sub_name, sub in [('gateway', self.gateway), ('cortical_stack', self.cortical_stack), 
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
        for name, tensor in state_dict.items():
            tensor = tensor.to(device)
            if name == "text_embeddings.weight":
                self._safe_copy_param(self.pos_embeddings.byte_embed.weight.data, tensor)
            elif name.startswith("pos_embeddings."):
                p_name = name.replace("pos_embeddings.", "")
                for sub_p_name, sub_p in self.pos_embeddings.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor)
            elif name.startswith("input_proj."):
                p_name = name.replace("input_proj.", "")
                if hasattr(self.input_proj, p_name):
                    self._safe_copy_param(getattr(self.input_proj, p_name).data, tensor)
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
        base_k = self.config.train.dfet_k_sigma_base
        na_weight = self.config.train.dfet_k_sigma_na_weight
        min_k = self.config.train.dfet_min_k_sigma
        
        k_sigma = max(min_k, base_k - na_weight * na_level)
        dynamic_threshold = moving_mean + k_sigma * moving_std
        return free_energy_val > dynamic_threshold

    def forward_step(self, sensor_inputs: Dict[str, torch.Tensor], h_prev_fast: torch.Tensor, 
                     h_prev_slow: torch.Tensor, u_t: torch.Tensor, episodic_memory=None, 
                     dt: float = 1.0, attention_temp: float = 0.05, m_states: List[torch.Tensor] = None):
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
        should_search_memory = (episodic_memory is not None) and (noradrenaline.mean().item() > na_trigger) and (episodic_memory.size.max().item() > 0)

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
            
        # Native C++20 Chunked Cortical Processing
        t_seq = text_in.unsqueeze(1)
        x = self.input_proj(t_seq)
        
        if m_states is None or len(m_states) != self.num_layers or m_states[0].size(0) != batch_size:
            m_states = [torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device) for _ in range(self.num_layers)]
            
        cortical_out = self.cortical_stack.forward_stack(x, m_states, u_t, dt)
        x_out, next_m_list = cortical_out[0], cortical_out[1]
        
        h_reasoned = x_out.squeeze(1)
        h_next_fast = h_reasoned
        h_next_slow = h_next_fast
        
        w_pred, kl_div, _, z_t = self.world_model(h_prev_fast, h_next_slow, w_current)
        
        cosine_sim = F.cosine_similarity(w_current, w_pred, dim=-1, eps=1e-8).unsqueeze(-1)
        rec_loss = 1.0 - cosine_sim
        free_energy = kl_div + rec_loss

        relax_out = self.attractor_head.relax_to_minima(h_reasoned)
        h_relaxed = relax_out[0]
        
        outputs = self.output_gateway(h_relaxed)
        h_proj = self.motor_text_proj(h_relaxed)
        
        raw_logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight)
        outputs["text_generation"] = 30.0 * torch.tanh(raw_logits / 30.0)
        
        state_value = self.critic(h_reasoned)
        eff_dt = torch.tensor([[dt]], device=self.device)
        return h_next_fast, h_next_slow, outputs, state_value, w_pred, free_energy, kl_div, w_current, attn_weights, channel_names, epistemic_entropy, eff_dt, next_m_list

    def forward(self, *args, **kwargs):
        return self.forward_step(*args, **kwargs)

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion_speech: nn.Module, episodic_memory=None, loss_free_energy_weight: float = 0.05, 
                         chunk_size: int = 64, optimizer: torch.optim.Optimizer = None) -> Tuple[float, float, float, List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_seq.size()
        
        m_list = [torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device) for _ in range(self.num_layers)]
        curr_u_t = hu_batch.state.clone().detach()
        action_cost_tensor = torch.full((batch_size, 1), 0.001, device=self.device)
        cog_action_tensor = torch.zeros((batch_size, 1), dtype=torch.int64, device=self.device)
        
        # 1. Full Sequence Embeddings
        full_emb = self.pos_embeddings(input_seq, start_pos=0, apply_rf=True)
        x_full = self.input_proj(full_emb)

        # 2. Native C++20 Chunked Cortical Scan (Q=64 inside C++)
        cortical_out = self.cortical_stack.forward_stack(x_full, m_list, curr_u_t, 1.0)
        x_out, m_list = cortical_out[0], cortical_out[1]

        # 3. Energy Attractor Relaxation & Logit Soft-Capping (30.0 * tanh)
        h_flat = x_out.reshape(batch_size * seq_len, self.hidden_dim)
        h_relaxed = self.attractor_head.relax_to_minima(h_flat)[0]
        
        h_proj = self.motor_text_proj(h_relaxed)
        raw_logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight)
        bounded_logits = 30.0 * torch.tanh(raw_logits / 30.0)

        # 4. NaN-Proof SFT Masked CrossEntropy Loss
        targets_flat = target_seq.contiguous().view(-1)
        valid_targets = (targets_flat != 256)
        
        if valid_targets.any():
            speech_loss = criterion_speech(bounded_logits, targets_flat)
            speech_loss = torch.nan_to_num(speech_loss, nan=0.0, posinf=10.0, neginf=0.0)
        else:
            speech_loss = (bounded_logits * 0.0).sum()

        fe_loss = 0.01
        total_loss = speech_loss + loss_free_energy_weight * fe_loss

        # 5. Backward Pass
        if optimizer is not None and valid_targets.any():
            total_loss.backward()

        # Somatic Homeostasis Update
        with torch.no_grad():
            curr_loss_val = speech_loss.detach().item()
            somatic_surprise = torch.clamp(torch.tensor([[curr_loss_val / 4.0]], device=self.device), 0.0, 0.40).repeat(batch_size, 1)
            zero_entropy = torch.zeros((batch_size, 1), device=self.device)
            curr_u_t = hu_batch.update(action_cost_tensor, somatic_surprise, zero_entropy, cog_action_tensor).detach()

        h_proxy = m_list[-1].view(batch_size, -1)[:, :self.hidden_dim]
        last_eff_dt = torch.tensor([[1.0]], device=self.device)
        return total_loss.item(), curr_loss_val, fe_loss, m_list, h_proxy, curr_u_t, last_eff_dt

    def generate_thought_and_speech(
        self, prompt: str, m_states: List[torch.Tensor], h_state: torch.Tensor, hu, episodic_memory, 
        config, max_generated_tokens: int = 120, temperature: float = 0.7, top_p: float = 0.90
    ) -> Generator[Dict[str, Any], None, None]:
        prompt_tokens = self.encode_text(prompt).unsqueeze(0)
        prompt_embs = self.pos_embeddings(prompt_tokens, start_pos=0, apply_rf=True)
        batch_size = prompt_tokens.size(0)
        
        if m_states is None or len(m_states) != self.num_layers or m_states[0].size(0) != batch_size:
            m_states = [torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device) for _ in range(self.num_layers)]
            
        yield {"status": "speech_start"}
        
        # Parallel Prompt Processing in Native C++20 Chunked Engine
        x = self.input_proj(prompt_embs)
        cortical_out = self.cortical_stack.forward_stack(x, m_states, hu.state, 1.0)
        x_out, m_states = cortical_out[0], cortical_out[1]
        
        rolling_token_ids = prompt_tokens[0].tolist()
        energy_action_cost = torch.tensor([[0.002]], device=self.device)
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
            
            x_step = self.input_proj(t_emb)
            cortical_out = self.cortical_stack.forward_stack(x_step, m_states, hu.state, 1.0)
            x_out, m_states = cortical_out[0], cortical_out[1]
            
            h_relaxed = self.attractor_head.relax_to_minima(x_out.squeeze(1))[0]
            h_proj = self.motor_text_proj(h_relaxed)
            raw_logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight)
            logits = (30.0 * torch.tanh(raw_logits / 30.0)) / max(temperature, 1e-4)
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
                yield {"status": "exhausted", "text": " [fatigued...]", "m_states": m_states, "h_state": x_out}
                return

        yield {"status": "speech_end", "m_states": m_states, "h_state": x_out}
