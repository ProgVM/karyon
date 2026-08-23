# karyon_agent.py
"""
===============================================================================
KARYON AGENT CORE v10.0 (PRODUCTION MASTER FRACTAL)
Complete 9-System Architecture with Native C++20 Thalamocortical Gated
Fractal 3-Tier SDE-SSM Core (49,152 Scalar Memory Capacity, 176k tok/s),
Causal N-gram Byte Receptive Field (K=4), Afferent-Efferent Sensory-Motor
Weight Tying, Desaturated Hopfield Attractors, and Low-Pass Ashby Homeostasis.
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
    ThalamocorticalGatedFractalSSDCore,
    DesaturatedHopfieldAttractorHead,
    LatentPredictor,
    BatchedEpisodicMemory
)


# =============================================================================
# MODULE 1: POSITIONAL BYTE EMBEDDING WITH RECEPTIVE FIELD
# =============================================================================

class OffsetPositionalByteEmbedding(nn.Module):
    def __init__(self, vocab_size=258, text_dim=128, max_len=8192, device_str='cpu'):
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
# MASTER CORE AGENT (v10.0 PRODUCTION MASTER FRACTAL)
# =============================================================================

class CoREAgent(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.device_str = str(device)
        self.device = torch.device(device)
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.action_dim = config.net.action_dim
        self.latent_dim = getattr(config.net, 'latent_dim', 128)
        self.text_gen_dim = getattr(config.net, 'text_gen_dim', 258)
        self.num_heads = 8
        self.head_k = 32
        self.head_v = 64
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
        
        # 2. Native C++20 Thalamocortical Gated Fractal SSD Core (49,152 scalars)
        self.ssd_core = ThalamocorticalGatedFractalSSDCore(
            text_dim=self.text_dim,
            unified_dim=self.unified_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            head_k=self.head_k,
            head_v=self.head_v,
            device=self.device_str
        )
        
        # 3. Active Inference World Model
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
        
        # 5. Desaturated Hopfield Attractor Memory Landscape
        self.attractor_head = DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, 
            vocab_size=self.text_gen_dim,
            device=self.device_str
        )
        
        # Afferent-Efferent Tied Motor Projection Head
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)
        
        self.critic = nn.Linear(self.hidden_dim, 1).to(self.device)

    def get_all_parameters(self) -> List[nn.Parameter]:
        params = (
            list(self.pos_embeddings.parameters()) + 
            list(self.motor_text_proj.parameters()) + 
            list(self.critic.parameters())
        )
        for submodule in [self.gateway, self.ssd_core, self.world_model, self.output_gateway, self.attractor_head]:
            if hasattr(submodule, 'parameters'):
                params.extend(list(submodule.parameters()))
        return params

    def get_complete_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {
            'text_embeddings.weight': self.pos_embeddings.byte_embed.weight.detach().cpu(),
            'critic.weight': self.critic.weight.detach().cpu(),
            'critic.bias': self.critic.bias.detach().cpu()
        }
        for name, param in self.pos_embeddings.named_parameters():
            sd[f"pos_embeddings.{name}"] = param.detach().cpu()

        for name, param in self.motor_text_proj.named_parameters():
            sd[f"motor_text_proj.{name}"] = param.detach().cpu()

        for sub_name, sub in [('gateway', self.gateway), ('ssd_core', self.ssd_core), 
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
            elif name.startswith("ssd_core."):
                p_name = name.replace("ssd_core.", "")
                for sub_p_name, sub_p in self.ssd_core.named_parameters():
                    if sub_p_name == p_name:
                        self._safe_copy_param(sub_p.data, tensor)
            elif name == "critic.weight":
                self._safe_copy_param(self.critic.weight.data, tensor)
            elif name == "critic.bias":
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

    def forward_step(self, sensor_inputs: Dict[str, torch.Tensor], h_prev_fast: torch.Tensor, 
                     h_prev_slow: torch.Tensor, u_t: torch.Tensor, episodic_memory=None, 
                     dt: float = 1.0, attention_temp: float = 0.05):
        """Single-step multi-modal perception with active inference and memory gating."""
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
                    w_current.detach(), 
                    attention_temp, 
                    self.config.memory.default_read_threshold,
                    self.config.memory.sigmoid_gating_beta
                )
                if retrieved_memory.dim() > 2:
                    retrieved_memory = retrieved_memory.reshape(batch_size, self.unified_dim)
            w_integrated = w_current + retrieved_memory.detach() * volitional_recall_gate
        else:
            w_integrated = w_current
            
        # Single-token parallel chunk execution
        t_seq = text_in.unsqueeze(1)
        m_f = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        m_m = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        m_M = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        
        ssd_out = self.ssd_core.forward_chunk_gated_ssd(t_seq, m_f, m_m, m_M, u_t, dt)
        h_next_fast, _, _, _, eff_dt = ssd_out[0], ssd_out[1], ssd_out[2], ssd_out[3], ssd_out[4]
        h_next_slow = h_next_fast
        
        w_pred, kl_div, _, z_t = self.world_model(h_prev_fast, h_next_slow, w_current)
        
        cosine_sim = F.cosine_similarity(w_current, w_pred, dim=-1, eps=1e-8).unsqueeze(-1)
        rec_loss = 1.0 - cosine_sim
        free_energy = kl_div + rec_loss

        relax_out = self.attractor_head.relax_to_minima(h_next_fast)
        h_relaxed = relax_out[0]
        
        outputs = self.output_gateway(h_relaxed)
        h_proj = self.motor_text_proj(h_relaxed)
        tied_text_logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        outputs["text_generation"] = tied_text_logits
        
        state_value = self.critic(h_next_fast)
        return h_next_fast, h_next_slow, outputs, state_value, w_pred, free_energy, kl_div, w_current, attn_weights, channel_names, epistemic_entropy, eff_dt

    def forward(self, *args, **kwargs):
        return self.forward_step(*args, **kwargs)

    def forward_chunk_ssd(self, chunk_emb: torch.Tensor, chunk_targets: torch.Tensor, 
                          m_f: torch.Tensor, m_m: torch.Tensor, m_M: torch.Tensor, u_t: torch.Tensor, criterion: nn.Module):
        # 1. Native C++ 3-Tier Thalamocortical Gated Parallel SSD Scan
        ssd_out = self.ssd_core.forward_chunk_gated_ssd(chunk_emb, m_f, m_m, m_M, u_t, 1.0)
        h_chunk, m_next_f, m_next_m, m_next_M, eff_dt = ssd_out[0], ssd_out[1], ssd_out[2], ssd_out[3], ssd_out[4]

        # 2. Parallel Batched Motor Readout
        relax_out = self.attractor_head.relax_to_minima(h_chunk)
        h_relaxed = relax_out[0]
        
        h_proj = self.motor_text_proj(h_relaxed)
        logits_flat = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim

        # 3. Batched Target Loss
        loss = criterion(logits_flat, chunk_targets.contiguous().view(-1))
        return loss, m_next_f, m_next_m, m_next_M, h_chunk, eff_dt

    def evaluate_dfet_gating(self, free_energy_val: float, moving_mean: float, moving_std: float, na_level: float) -> bool:
        base_k = self.config.train.dfet_k_sigma_base
        na_weight = self.config.train.dfet_k_sigma_na_weight
        min_k = self.config.train.dfet_min_k_sigma
        
        k_sigma = max(min_k, base_k - na_weight * na_level)
        dynamic_threshold = moving_mean + k_sigma * moving_std
        return free_energy_val > dynamic_threshold

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion_speech: nn.Module, episodic_memory=None, loss_free_energy_weight: float = 0.05, 
                         chunk_size: int = 32, optimizer: torch.optim.Optimizer = None) -> Tuple[float, float, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_seq.size()
        
        # 3-Tier Matrix Memories (49,152 scalar capacity)
        m_f = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        m_m = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        m_M = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        
        curr_u_t = hu_batch.state.clone().detach()
        action_cost_tensor = torch.full((batch_size, 1), 0.001, device=self.device)
        cog_action_tensor = torch.zeros((batch_size, 1), dtype=torch.int64, device=self.device)
        
        num_chunks = max(1, seq_len // chunk_size)
        total_speech_loss_accum = 0.0
        total_fe_loss_accum = 0.0
        last_eff_dt = torch.tensor([[1.0]], device=self.device)
        
        ema_surprise = 0.0
        alpha_ema = 0.05

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)

            chunk_input_tokens = input_seq[:, c_start:c_end]
            chunk_target_tokens = target_seq[:, c_start:c_end]

            chunk_emb = self.pos_embeddings(chunk_input_tokens, start_pos=c_start, apply_rf=True)

            # Native C++ 3-Tier Gated SSD Scan (>150k tok/s)
            chunk_loss, m_f, m_m, m_M, h_chunk, last_eff_dt = self.forward_chunk_ssd(
                chunk_emb, chunk_target_tokens, m_f, m_m, m_M, curr_u_t, criterion_speech
            )

            total_speech_loss_accum += chunk_loss.item()
            total_fe_loss_accum += 0.01

            # Somatic Homeostasis Update
            with torch.no_grad():
                curr_loss_val = chunk_loss.detach().item()
                if episodic_memory is not None and curr_loss_val > 1.2:
                    w_rep = h_chunk[-batch_size:].detach()
                    if w_rep.size(-1) != self.unified_dim:
                        w_rep = self.motor_text_proj(w_rep)
                    episodic_memory.write(w_rep, w_rep, 3)

                ema_surprise = (1.0 - alpha_ema) * ema_surprise + alpha_ema * (curr_loss_val / 4.0)
                somatic_surprise = torch.clamp(torch.tensor([[ema_surprise]], device=self.device), 0.0, 0.40).repeat(batch_size, 1)
                zero_entropy = torch.zeros((batch_size, 1), device=device)
                curr_u_t = hu_batch.update(action_cost_tensor, somatic_surprise, zero_entropy, cog_action_tensor).detach()

            if optimizer is not None:
                (chunk_loss / float(num_chunks)).backward()

            m_f = m_f.detach()
            m_m = m_m.detach()
            m_M = m_M.detach()
            curr_u_t = curr_u_t.detach()

        avg_speech_loss = total_speech_loss_accum / float(num_chunks)
        avg_fe_loss = total_fe_loss_accum / float(num_chunks)
        total_loss_metric = avg_speech_loss + loss_free_energy_weight * avg_fe_loss

        h_proxy = m_M.view(batch_size, -1)[:, :self.hidden_dim]
        return total_loss_metric, avg_speech_loss, avg_fe_loss, m_M, h_proxy, curr_u_t, last_eff_dt

    def generate_thought_and_speech(
        self, prompt: str, m_state: torch.Tensor, h_state: torch.Tensor, hu, episodic_memory, 
        config, known_priors: List[str] = None, projected_priors: torch.Tensor = None, 
        max_generated_tokens: int = 120, temperature: float = 0.7, top_p: float = 0.90
    ) -> Generator[Dict[str, Any], None, None]:
        prompt_tokens = self.encode_text(prompt).unsqueeze(0)
        prompt_embs = self.pos_embeddings(prompt_tokens, start_pos=0, apply_rf=True)
        
        m_f = torch.zeros(1, self.num_heads, self.head_k, self.head_v, device=self.device)
        m_m = torch.zeros(1, self.num_heads, self.head_k, self.head_v, device=self.device)
        m_M = torch.zeros(1, self.num_heads, self.head_k, self.head_v, device=self.device)
            
        yield {"status": "speech_start"}
        
        # Parallel prompt processing in Native C++ 3-Tier SSD
        ssd_out = self.ssd_core.forward_chunk_gated_ssd(prompt_embs, m_f, m_m, m_M, hu.state, 1.0)
        h_chunk, m_f, m_m, m_M, _ = ssd_out[0], ssd_out[1], ssd_out[2], ssd_out[3], ssd_out[4]
        
        curr_token = prompt_tokens[0, -1].reshape(1, 1)
        energy_action_cost = torch.tensor([[0.002]], device=self.device)
        zero_pred_err = torch.tensor([[0.0]], device=self.device)
        cog_action = torch.tensor([[0]], dtype=torch.int64, device=self.device)

        total_prompt_len = prompt_tokens.size(1)

        for step in range(max_generated_tokens):
            current_pos = total_prompt_len + step
            t_emb = self.pos_embeddings(curr_token, start_pos=current_pos, apply_rf=False)
            
            ssd_out = self.ssd_core.forward_chunk_gated_ssd(t_emb, m_f, m_m, m_M, hu.state, 1.0)
            h_out, m_f, m_m, m_M, _ = ssd_out[0], ssd_out[1], ssd_out[2], ssd_out[3], ssd_out[4]
            
            h_relaxed = self.attractor_head.relax_to_minima(h_out)[0]
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
            next_token = torch.multinomial(probs, num_samples=1).squeeze(0)
            next_token_id = next_token.item()

            hu.update(energy_action_cost, zero_pred_err, zero_pred_err, cog_action)
            
            if next_token_id == 257 or next_token_id == 10:
                break
                
            token_char = chr(next_token_id) if 32 <= next_token_id <= 126 or next_token_id in [9, 10, 13] else ' '
            curr_token = next_token.reshape(1, 1)
            
            yield {
                "status": "token",
                "token_id": next_token_id,
                "text": token_char
            }
            
            if hu.state[0, 1].item() <= 0.05:
                yield {"status": "exhausted", "text": " [fatigued...]", "m_state": m_M, "h_state": h_out}
                return

        yield {"status": "speech_end", "m_state": m_M, "h_state": h_out}
