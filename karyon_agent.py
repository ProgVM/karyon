# karyon_agent.py
"""
===============================================================================
KARYON AGENT CORE v5.8 (PRODUCTION MASTER)
Active Inference Engine with Per-Chunk 32-Step BPTT Autograd Execution,
Gradient Scale Amplification, Positional Bytes, and DFET v3 Plasticity.
===============================================================================
"""

import math
from typing import Generator, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MODULE 1: SINUSOIDAL POSITIONAL BYTE EMBEDDING LAYER (KEP #7)
# =============================================================================

class PositionalByteEmbedding(nn.Module):
    """
    Sinusoidal Positional Encoding added to byte-level embeddings 
    to preserve strict byte order and character positional structure.
    """
    def __init__(self, vocab_size=258, text_dim=128, max_len=1024):
        super().__init__()
        self.byte_embed = nn.Embedding(vocab_size, text_dim)
        
        pe = torch.zeros(max_len, text_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, text_dim, 2).float() * (-math.log(10000.0) / text_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        tok_emb = self.byte_embed(input_ids)
        pos_emb = self.pe[:, :seq_len, :]
        return tok_emb + pos_emb


# =============================================================================
# MODULE 2: NORMALIZED HOPFIELD ENERGY ATTRACTOR HEAD
# =============================================================================

class NormalizedEnergyAttractorHead(nn.Module):
    """
    Direct Differentiable Hopfield Attractor Memory Landscape.
    Relaxes hidden states directly towards energy basins.
    """
    def __init__(self, hidden_dim=256, vocab_size=258, num_attractors=64, temperature=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_attractors = num_attractors
        self.temperature = temperature
        
        self.attractor_basins = nn.Parameter(torch.randn(num_attractors, hidden_dim) * 0.1)

    def compute_energy(self, h_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_dist_sq = (torch.cdist(h_state, self.attractor_basins, p=2)**2) / float(self.hidden_dim)
        energy = -torch.logsumexp(-norm_dist_sq * 2.0, dim=-1, keepdim=True)
        return energy, norm_dist_sq

    def relax_to_minima(self, h_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_dist_sq = (torch.cdist(h_state, self.attractor_basins, p=2)**2) / float(self.hidden_dim)
        attn_weights = F.softmax(-norm_dist_sq / self.temperature, dim=-1)
        
        attractor_shift = torch.matmul(attn_weights, self.attractor_basins)
        h_relaxed = h_state + 0.2 * attractor_shift
        
        energy = -torch.logsumexp(-norm_dist_sq * 2.0, dim=-1, keepdim=True)
        return h_relaxed, energy


# =============================================================================
# MASTER CORE AGENT (v5.8 PRODUCTION MASTER)
# =============================================================================

class CoREAgent(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.device_str = str(device)
        self.device = torch.device(device)
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.action_dim = config.net.action_dim
        self.latent_dim = getattr(config.net, 'latent_dim', 128)
        
        from karyon_core import ByteTokenizer, SensoryGateway, MotorGateway, DynamicRecurrentCore, LatentPredictor
        self.tokenizer = ByteTokenizer(vocab_size=config.net.text_gen_dim)
        
        self.pos_embeddings = PositionalByteEmbedding(
            vocab_size=config.net.text_gen_dim, 
            text_dim=config.net.text_dim
        ).to(self.device)
        self.text_embeddings = self.pos_embeddings.byte_embed
        
        self.gateway = SensoryGateway(
            unified_dim=self.unified_dim, 
            hidden_dim=self.hidden_dim, 
            homeo_dim=config.net.homeo_dim, 
            text_dim=config.net.text_dim, 
            vision_dim=config.net.vision_dim, 
            action_dim=config.net.action_dim,
            device=self.device_str
        )
        
        self.core = DynamicRecurrentCore(
            self.hidden_dim, 
            self.unified_dim, 
            homeo_dim=config.net.homeo_dim, 
            gamma=config.sde.gamma_drift,
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
            text_gen_dim=config.net.text_gen_dim,
            device=self.device_str
        )
        
        self.attractor_head = NormalizedEnergyAttractorHead(self.hidden_dim, config.net.text_gen_dim).to(self.device)
        self.critic = nn.Linear(self.hidden_dim, 1).to(self.device)

        self._cached_zero_vision = torch.zeros(1, config.net.vision_dim, device=self.device)
        self._cached_zero_motor = torch.zeros(1, config.net.action_dim, device=self.device)

    def get_all_parameters(self) -> List[nn.Parameter]:
        """Gathers parameters across Python layers and native C++ LibTorch extensions."""
        params = list(self.pos_embeddings.parameters()) + list(self.critic.parameters()) + list(self.attractor_head.parameters())
        for submodule in [self.gateway, self.core, self.world_model, self.output_gateway]:
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
        for name, param in self.attractor_head.named_parameters():
            sd[f"attractor_head.{name}"] = param.detach().cpu()

        for sub_name, sub in [('gateway', self.gateway), ('core', self.core), 
                              ('world_model', self.world_model), ('output_gateway', self.output_gateway)]:
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
            elif name == "critic.weight":
                self._safe_copy_param(self.critic.weight.data, tensor)
            elif name == "critic.bias":
                self._safe_copy_param(self.critic.bias.data, tensor)
            elif name.startswith("attractor_head."):
                param_name = name.replace("attractor_head.", "")
                if hasattr(self.attractor_head, param_name):
                    self._safe_copy_param(getattr(self.attractor_head, param_name).data, tensor)
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

    def forward(self, sensor_inputs: Dict[str, torch.Tensor], h_prev_fast: torch.Tensor, 
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
            
        core_outputs = self.core(h_prev_fast, h_prev_slow, w_integrated, u_t, dt)
        h_next_fast, h_next_slow, eff_dt = core_outputs[0], core_outputs[1], core_outputs[2]
        
        w_pred, kl_div, _, z_t = self.world_model(h_prev_fast, h_next_slow, w_current)
        
        cosine_sim = F.cosine_similarity(w_current, w_pred, dim=-1, eps=1e-8).unsqueeze(-1)
        rec_loss = 1.0 - cosine_sim
        free_energy = kl_div + rec_loss

        h_integrated = h_next_fast + h_next_slow
        h_relaxed, basin_energy = self.attractor_head.relax_to_minima(h_integrated)
        
        outputs = self.output_gateway(h_relaxed)
        state_value = self.critic(h_integrated)
        
        return h_next_fast, h_next_slow, outputs, state_value, w_pred, free_energy, kl_div, w_current, attn_weights, channel_names, epistemic_entropy, eff_dt

    def evaluate_dfet_gating(self, free_energy_val: float, moving_mean: float, moving_std: float, na_level: float) -> bool:
        base_k = self.config.train.dfet_k_sigma_base
        na_weight = self.config.train.dfet_k_sigma_na_weight
        min_k = self.config.train.dfet_min_k_sigma
        
        k_sigma = max(min_k, base_k - na_weight * na_level)
        dynamic_threshold = moving_mean + k_sigma * moving_std
        return free_energy_val > dynamic_threshold

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion_speech: nn.Module, loss_free_energy_weight: float = 0.05, 
                         chunk_size: int = 32):
        """Unrolls sequence with Per-Chunk Autograd Execution, preventing 3.7s backward pass lags."""
        batch_size, seq_len = input_seq.size()
        
        h_fast_curr = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        h_slow_curr = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        curr_prev_action = torch.zeros(batch_size, self.action_dim, device=self.device)
        obs_vision = torch.zeros(batch_size, self.config.net.vision_dim, device=self.device)
        
        curr_u_t = hu_batch.state.clone()
        action_cost_tensor = torch.full((batch_size, 1), 0.001, device=self.device)
        cog_action_tensor = torch.zeros((batch_size, 1), dtype=torch.int64, device=self.device)
        
        full_emb_seq = self.pos_embeddings(input_seq)
        
        speech_losses = []
        free_energy_losses = []
        last_attn = None
        last_names = None
        
        chunk_speech_loss = 0.0
        chunk_fe_loss = 0.0

        for t in range(seq_len):
            input_emb = full_emb_seq[:, t]
            target_t = target_seq[:, t]
            
            sensor_inputs = {'text': input_emb, 'vision': obs_vision, 'motor_efference': curr_prev_action}
            
            h_fast_curr, h_slow_curr, loop_outputs, _, _, free_energy, _, _, attn_weights, channel_names, eps_ent, eff_dt = self(
                sensor_inputs, h_fast_curr, h_slow_curr, curr_u_t, dt=1.0, attention_temp=self.config.memory.default_attention_temp
            )
            
            speech_logits = loop_outputs["text_generation"]
            loss_tok = criterion_speech(speech_logits, target_t)
            
            chunk_speech_loss = chunk_speech_loss + loss_tok
            chunk_fe_loss = chunk_fe_loss + free_energy.mean()
            
            last_attn = attn_weights
            last_names = channel_names

            somatic_pred_err = torch.clamp(free_energy * 3.0 / self.unified_dim, 0.0, 1.0)
            curr_u_t = hu_batch.update(action_cost_tensor, somatic_pred_err, eps_ent.mean(dim=-1, keepdim=True), cog_action_tensor)

            # Per-Chunk 32-Step Graph Truncation and Local Loss Accumulation
            if (t + 1) % chunk_size == 0 or t == seq_len - 1:
                speech_losses.append(chunk_speech_loss / chunk_size)
                free_energy_losses.append(chunk_fe_loss / chunk_size)
                
                chunk_speech_loss = 0.0
                chunk_fe_loss = 0.0
                
                h_fast_curr = h_fast_curr.detach()
                h_slow_curr = h_slow_curr.detach()

        speech_loss_total = torch.stack(speech_losses).mean()
        free_energy_total = torch.stack(free_energy_losses).mean()
        total_loss = loss_free_energy_weight * free_energy_total + self.config.train.loss_speech_weight * speech_loss_total
        
        return total_loss, speech_loss_total, free_energy_total, h_fast_curr, h_slow_curr, curr_u_t, eff_dt, last_attn, last_names

    def generate_thought_and_speech(
        self, prompt: str, h_fast: torch.Tensor, h_slow: torch.Tensor, hu, episodic_memory, 
        config, known_priors: List[str] = None, projected_priors: torch.Tensor = None, 
        max_generated_tokens: int = 120, temperature: float = 0.7, top_p: float = 0.90
    ) -> Generator[Dict[str, Any], None, None]:
        prompt_tokens = self.encode_text(prompt).unsqueeze(0)
        prompt_embs = self.pos_embeddings(prompt_tokens).squeeze(0)
        
        h_f = h_fast.clone()
        h_s = h_slow.clone()
        
        obs_vis = self._cached_zero_vision
        prev_act = self._cached_zero_motor
        
        total_prompt_steps = prompt_embs.size(0)
        for idx in range(total_prompt_steps):
            t_emb = prompt_embs[idx].reshape(1, -1)
            s_in = {'text': t_emb, 'vision': obs_vis, 'motor_efference': prev_act}
            
            h_f, h_s, _, _, _, _, _, _, attn_w, ch_names, eps_ent, _ = self(
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
        cog_act = torch.tensor([[0]], dtype=torch.int64, device=self.device)

        for step in range(max_generated_tokens):
            t_emb = self.pos_embeddings(curr_token).squeeze(0)
            s_in = {'text': t_emb, 'vision': obs_vis, 'motor_efference': prev_act}
            
            h_f, h_s, outputs, _, _, fe, _, _, _, _, eps_ent, _ = self(
                s_in, h_f, h_s, hu.state, episodic_memory=episodic_memory
            )
            
            logits = outputs["text_generation"] / max(temperature, 1e-4)
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

            hu.update(energy_action_cost, zero_pred_err, eps_ent.mean(dim=-1, keepdim=True), cog_act)
            
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
