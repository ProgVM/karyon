# karyon_agent.py
"""
===============================================================================
KARYON AGENT CORE v6.5 (PRODUCTION MASTER)
Continuous-Time Selective SDE State-Space Recurrent Engine (SDE-SSM),
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
    SelectiveSDEStateSpaceCore,
    DesaturatedHopfieldAttractorHead,
    LatentPredictor,
    BatchedEpisodicMemory
)


# =============================================================================
# MODULE 1: POSITIONAL BYTE EMBEDDING WITH NATIVE C++ RECEPTIVE FIELD
# =============================================================================

class OffsetPositionalByteEmbedding(nn.Module):
    def __init__(self, vocab_size=258, text_dim=128, max_len=4096, device_str='cpu'):
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
# MASTER CORE AGENT (v6.5 PRODUCTION MASTER)
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
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)
        
        self.tokenizer = ByteTokenizer(vocab_size=self.text_gen_dim)
        
        self.pos_embeddings = OffsetPositionalByteEmbedding(
            vocab_size=self.text_gen_dim, 
            text_dim=self.text_dim,
            device_str=self.device_str
        ).to(self.device)
        self.text_embeddings = self.pos_embeddings.byte_embed
        
        self.gateway = SensoryGateway(
            unified_dim=self.unified_dim, 
            hidden_dim=self.hidden_dim, 
            homeo_dim=config.net.homeo_dim, 
            text_dim=self.text_dim, 
            vision_dim=config.net.vision_dim, 
            action_dim=config.net.action_dim,
            device=self.device_str
        )
        
        self.sde_ssm = SelectiveSDEStateSpaceCore(
            hidden_dim=self.hidden_dim,
            unified_dim=self.unified_dim,
            homeo_dim=config.net.homeo_dim,
            device=self.device_str
        )
        
        self.world_model = LatentPredictor(
            hidden_dim=self.hidden_dim,
            unified_dim=self.unified_dim,
            latent_dim=self.latent_dim,
            device=self.device_str
        )
        
        self.output_gateway = MotorGateway(
            hidden_dim=self.hidden_dim, 
            action_dim=config.net.action_dim, 
            cog_action_dim=config.net.cog_action_dim, 
            text_gen_dim=self.text_gen_dim,
            device=self.device_str
        )
        
        self.attractor_head = DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, 
            vocab_size=self.text_gen_dim,
            device=self.device_str
        )
        
        # Sensory-Motor Afferent-Efferent Tied Projection Head
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)
        
        self.critic = nn.Linear(self.hidden_dim, 1).to(self.device)

        self._cached_zero_vision = torch.zeros(1, config.net.vision_dim, device=self.device)
        self._cached_zero_motor = torch.zeros(1, config.net.action_dim, device=self.device)

    def get_all_parameters(self) -> List[nn.Parameter]:
        """Gathers all trainable parameters across Python and native C++ LibTorch extensions."""
        params = (
            list(self.pos_embeddings.parameters()) + 
            list(self.motor_text_proj.parameters()) + 
            list(self.critic.parameters())
        )
        for submodule in [self.gateway, self.sde_ssm, self.world_model, self.output_gateway, self.attractor_head]:
            if hasattr(submodule, 'parameters'):
                params.extend(list(submodule.parameters()))
        return params

    def get_complete_state_dict(self) -> Dict[str, torch.Tensor]:
        """Unified complete state dict across Python and C++ submodules."""
        sd = {
            'text_embeddings.weight': self.pos_embeddings.byte_embed.weight.detach().cpu(),
            'critic.weight': self.critic.weight.detach().cpu(),
            'critic.bias': self.critic.bias.detach().cpu()
        }
        for name, param in self.pos_embeddings.named_parameters():
            sd[f"pos_embeddings.{name}"] = param.detach().cpu()

        for name, param in self.motor_text_proj.named_parameters():
            sd[f"motor_text_proj.{name}"] = param.detach().cpu()

        for sub_name, sub in [('gateway', self.gateway), ('sde_ssm', self.sde_ssm), 
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
        should_search_memory = (episodic_memory is not None) and (noradrenaline.mean().item() > na_trigger)

        if should_search_memory:
            retrieved_memory, max_sim = episodic_memory.read(
                w_current, 
                attention_temp, 
                self.config.memory.default_read_threshold,
                self.config.memory.sigmoid_gating_beta
            )
            if retrieved_memory.dim() > 2:
                retrieved_memory = retrieved_memory.reshape(batch_size, self.unified_dim)
            w_integrated = w_current + retrieved_memory * volitional_recall_gate
        else:
            w_integrated = w_current
            
        # Native C++ Continuous-Time SDE-SSM Recurrent State-Space Integration
        sde_out = self.sde_ssm(h_prev_fast, w_integrated, u_t, dt)
        h_next_fast, y_out, eff_dt = sde_out[0], sde_out[1], sde_out[2]
        h_next_slow = h_next_fast
        
        w_pred, kl_div, _, z_t = self.world_model(h_prev_fast, h_next_slow, w_current)
        
        cosine_sim = F.cosine_similarity(w_current, w_pred, dim=-1, eps=1e-8).unsqueeze(-1)
        rec_loss = 1.0 - cosine_sim
        free_energy = kl_div + rec_loss

        relax_out = self.attractor_head.relax_to_minima(y_out)
        h_relaxed, basin_energy = relax_out[0], relax_out[1]
        
        outputs = self.output_gateway(h_relaxed)
        
        # Sensory-Motor Weight Tying: Direct Scaled Lexical Readout
        h_proj = self.motor_text_proj(h_relaxed)
        tied_text_logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        outputs["text_generation"] = tied_text_logits
        
        state_value = self.critic(y_out)
        
        return h_next_fast, h_next_slow, outputs, state_value, w_pred, free_energy, kl_div, w_current, attn_weights, channel_names, epistemic_entropy, eff_dt

    def forward(self, *args, **kwargs):
        return self.forward_step(*args, **kwargs)

    def evaluate_dfet_gating(self, free_energy_val: float, moving_mean: float, moving_std: float, na_level: float) -> bool:
        base_k = self.config.train.dfet_k_sigma_base
        na_weight = self.config.train.dfet_k_sigma_na_weight
        min_k = self.config.train.dfet_min_k_sigma
        
        k_sigma = max(min_k, base_k - na_weight * na_level)
        dynamic_threshold = moving_mean + k_sigma * moving_std
        return free_energy_val > dynamic_threshold

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion_speech: nn.Module, loss_free_energy_weight: float = 0.05, 
                         chunk_size: int = 32, optimizer: torch.optim.Optimizer = None) -> Tuple[float, float, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_seq.size()
        
        h_fast_curr = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        h_slow_curr = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        curr_prev_action = torch.zeros(batch_size, self.action_dim, device=self.device)
        obs_vision = torch.zeros(batch_size, self.config.net.vision_dim, device=self.device)
        
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
            curr_chunk_len = c_end - c_start

            chunk_input_tokens = input_seq[:, c_start:c_end]
            chunk_target_tokens = target_seq[:, c_start:c_end]

            chunk_emb = self.pos_embeddings(chunk_input_tokens, start_pos=c_start, apply_rf=True)

            chunk_speech_losses = []
            chunk_fe_losses = []

            for t in range(curr_chunk_len):
                input_emb = chunk_emb[:, t]
                target_t = chunk_target_tokens[:, t]
                
                sensor_inputs = {'text': input_emb, 'vision': obs_vision, 'motor_efference': curr_prev_action}
                
                h_fast_curr, h_slow_curr, loop_outputs, _, _, free_energy, _, _, _, _, eps_ent, last_eff_dt = self.forward_step(
                    sensor_inputs, h_fast_curr, h_slow_curr, curr_u_t, dt=1.0, attention_temp=self.config.memory.default_attention_temp
                )
                
                speech_logits = loop_outputs["text_generation"]
                loss_tok = criterion_speech(speech_logits, target_t)
                
                chunk_speech_losses.append(loss_tok)
                chunk_fe_losses.append(free_energy.mean())
                
                # Low-Pass EMA Somatic Surprisal Filter
                with torch.no_grad():
                    curr_loss_val = loss_tok.detach().item()
                    ema_surprise = (1.0 - alpha_ema) * ema_surprise + alpha_ema * (curr_loss_val / 4.0)
                    somatic_surprise = torch.clamp(torch.tensor([[ema_surprise]], device=self.device), 0.0, 0.40).repeat(batch_size, 1)
                    eps_ent_mean = eps_ent.detach().mean(dim=-1, keepdim=True)
                    curr_u_t = hu_batch.update(action_cost_tensor, somatic_surprise, eps_ent_mean, cog_action_tensor).detach()

            chunk_speech_loss = torch.stack(chunk_speech_losses).mean()
            chunk_fe_loss = torch.stack(chunk_fe_losses).mean()
            chunk_total_loss = self.config.train.loss_speech_weight * chunk_speech_loss + loss_free_energy_weight * chunk_fe_loss

            total_speech_loss_accum += chunk_speech_loss.item()
            total_fe_loss_accum += chunk_fe_loss.item()

            if optimizer is not None:
                (chunk_total_loss / float(num_chunks)).backward()

            h_fast_curr = h_fast_curr.detach()
            h_slow_curr = h_slow_curr.detach()
            curr_u_t = curr_u_t.detach()

        avg_speech_loss = total_speech_loss_accum / float(num_chunks)
        avg_fe_loss = total_fe_loss_accum / float(num_chunks)
        total_loss_metric = self.config.train.loss_speech_weight * avg_speech_loss + loss_free_energy_weight * avg_fe_loss

        return total_loss_metric, avg_speech_loss, avg_fe_loss, h_fast_curr, h_slow_curr, curr_u_t, last_eff_dt

    def generate_thought_and_speech(
        self, prompt: str, h_fast: torch.Tensor, h_slow: torch.Tensor, hu, episodic_memory, 
        config, known_priors: List[str] = None, projected_priors: torch.Tensor = None, 
        max_generated_tokens: int = 120, temperature: float = 0.7, top_p: float = 0.90
    ) -> Generator[Dict[str, Any], None, None]:
        prompt_tokens = self.encode_text(prompt).unsqueeze(0)
        prompt_embs = self.pos_embeddings(prompt_tokens, start_pos=0, apply_rf=True).squeeze(0)
        
        h_f = h_fast.clone()
        h_s = h_slow.clone()
        
        obs_vis = self._cached_zero_vision
        prev_act = self._cached_zero_motor
        
        total_prompt_steps = prompt_embs.size(0)
        for idx in range(total_prompt_steps):
            t_emb = prompt_embs[idx].reshape(1, -1)
            s_in = {'text': t_emb, 'vision': obs_vis, 'motor_efference': prev_act}
            
            h_f, h_s, _, _, _, _, _, _, attn_w, ch_names, eps_ent, _ = self.forward_step(
                s_in, h_f, h_s, hu.state, episodic_memory=episodic_memory
            )
            
            yield {
                "status": "reading",
                "step": idx + 1,
                "total": total_prompt_steps,
                "attn_weights": attn_w.detach().cpu(),
                "channel_names": ch_names
            }
            
        if episodic_memory is not None and episodic_memory.size.max().item() > 0:
            yield {"status": "memory_check", "similarity": 0.85}

        yield {"status": "speech_start"}
        
        curr_token = prompt_tokens[0, -1].reshape(1, 1)
        energy_action_cost = torch.tensor([[0.002]], device=self.device)
        zero_pred_err = torch.tensor([[0.0]], device=self.device)
        cog_action = torch.tensor([[0]], dtype=torch.int64, device=self.device)

        for step in range(max_generated_tokens):
            current_pos = total_prompt_steps + step
            t_emb = self.pos_embeddings(curr_token, start_pos=current_pos, apply_rf=False).squeeze(0)
            s_in = {'text': t_emb, 'vision': obs_vis, 'motor_efference': prev_act}
            
            h_f, h_s, outputs, _, _, fe, _, _, _, _, eps_ent, _ = self.forward_step(
                s_in, h_f, h_s, hu.state, episodic_memory=episodic_memory
            )
            
            logits = outputs["text_generation"] / max(temperature, 1e-4)
            logits[:, 256:] = -1e9
            
            # Top-p Nucleus Sampling
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

            hu.update(energy_action_cost, zero_pred_err, eps_ent.mean(dim=-1, keepdim=True), cog_action)
            
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
                yield {"status": "exhausted", "text": " [fatigued...]", "h_fast": h_f, "h_slow": h_s}
                return

        yield {"status": "speech_end", "h_fast": h_f, "h_slow": h_s}
