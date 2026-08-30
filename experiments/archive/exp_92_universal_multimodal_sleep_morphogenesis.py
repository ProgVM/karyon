# experiments/exp_92_universal_multimodal_sleep_morphogenesis.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-92
Hypothesis:
1. Dynamic Extensible Multimodal Sensory Gateway allows adding arbitrary sensory channels
   (document, cybernetic telemetry, media, vision, audio) dynamically at runtime,
   achieving full cross-modal sequence processing without breaking existing architecture.
2. Sequence-Parallel Multimodal Unrolling processes multi-channel sequence streams
   simultaneously at native GPU speeds (>20,000 tok/s).
3. Biophysical Sleep Synaptic Pruning & Morphogenesis (execute_deep_allostatic_sleep_v2)
   prunes noise-corrupted parameters (bottom 5% weight magnitude) while consolidating active
   pathways, lowering Free Energy and preserving speech/multimodal loss.
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import os
import time
import math
import types
import json
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from datasets import load_dataset

# Dynamo Hotfix for Python 3.12 / Kaggle GPU
class DummyDynamoModule(types.ModuleType):
    def __getattr__(self, name):
        if name == "decorators":
            return decorators_mod
        if name == "disable":
            return _disable
        if name == "is_compiling":
            return lambda *args, **kwargs: False
        return lambda *args, **kwargs: None

def _disable(fn=None, *args, **kwargs):
    if fn is None or not callable(fn):
        return lambda *a, **kw: None
    return fn

decorators_mod = types.ModuleType("torch._dynamo.decorators")
class _DimRange:
    pass
decorators_mod._DimRange = _DimRange

dynamo_mod = DummyDynamoModule("torch._dynamo")
dynamo_mod.decorators = decorators_mod
dynamo_mod.disable = _disable

sys.modules["torch._dynamo"] = dynamo_mod
sys.modules["torch._dynamo.decorators"] = decorators_mod
torch._dynamo = dynamo_mod

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import karyon_config, karyon_core, karyon_agent, karyon_logger
importlib.reload(karyon_core)
importlib.reload(karyon_agent)

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')


class DynamicSensoryGateway(nn.Module):
    """
    Extensible Universal Multimodal Gateway.
    Allows registering any arbitrary new channel (documents, media, cybernetic sensors)
    dynamically at runtime and unrolling over sequence streams.
    """
    def __init__(self, unified_dim=256, hidden_dim=768, homeo_dim=6, device_str='cpu'):
        super().__init__()
        self.unified_dim = unified_dim
        self.hidden_dim = hidden_dim
        self.homeo_dim = homeo_dim
        self.device_str = device_str
        self.device = torch.device(device_str)
        
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

    def forward(self, sensor_inputs: dict, h_prev: torch.Tensor, u_t: torch.Tensor):
        batch_size = h_prev.size(0)
        projected_channels = []
        channel_names = []
        channel_masks = []
        
        for name, proj in self.projections.items():
            if name in sensor_inputs:
                x_in = sensor_inputs[name]
            else:
                in_dim = proj.in_features
                x_in = torch.zeros(batch_size, in_dim, dtype=torch.float32, device=self.device)
                
            x_max = x_in.abs().max(dim=-1, keepdim=True)[0]
            x_act = (x_max > 1e-5).float()
            
            proj_x = proj(x_in)
            projected_channels.append(proj_x)
            channel_names.append(name)
            channel_masks.append((1.0 - x_act) * -10000.0)
            
        projected_channels.append(self.homeo_proj(u_t))
        channel_names.append('body')
        channel_masks.append(torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device))
        
        projected_channels.append(self.mind_proj(h_prev))
        channel_names.append('mind')
        channel_masks.append(torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device))
        
        stacked_channels = torch.stack(projected_channels, dim=1)
        norm_stacked = self.channel_norm(stacked_channels)
        
        volition_query = self.attention_query_layer(h_prev).unsqueeze(1)
        norm_query = self.query_norm(volition_query)
        
        sim = (norm_query * norm_stacked).sum(dim=-1) / math.sqrt(self.unified_dim)
        stacked_masks = torch.cat(channel_masks, dim=1)
        sim = sim + stacked_masks
        
        attention_weights = F.softmax(sim, dim=-1)
        eps = 1e-9
        epistemic_entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-1, keepdim=True)
        
        w_t = (attention_weights.unsqueeze(-1) * stacked_channels).sum(dim=1)
        
        return w_t, attention_weights, channel_names, epistemic_entropy


class MultimodalSleepAgent(CoREAgent):
    """
    CoREAgent equipped with:
    1. DynamicSensoryGateway for dynamic channel extension (e.g. document, cybernetic).
    2. forward_multimodal_sequence for sequence-parallel cross-modal training.
    3. execute_deep_allostatic_sleep_v2 for Free-Energy Synaptic Pruning & Morphogenesis.
    """
    def __init__(self, config, device='cpu'):
        super().__init__(config, device)
        self.gateway = DynamicSensoryGateway(
            unified_dim=self.unified_dim,
            hidden_dim=self.hidden_dim,
            homeo_dim=config.net.homeo_dim,
            device_str=self.device_str
        )

    def register_sensory_channel(self, name: str, in_dim: int):
        self.gateway.register_channel(name, in_dim)

    def forward_multimodal_sequence(self, sensor_seq_dict: dict, target_seq: torch.Tensor, hu_batch,
                                   criterion_speech: nn.Module, episodic_memory=None, loss_free_energy_weight: float = 0.05,
                                   chunk_size: int = 64):
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
                unrolled_inputs[name] = seq_tensor.contiguous().view(batch_size * seq_len, -1)
            elif seq_tensor.dim() == 2:
                if name == 'text':
                    full_emb = self.pos_embeddings(seq_tensor, start_pos=0, apply_rf=True)
                    unrolled_inputs[name] = full_emb.contiguous().view(batch_size * seq_len, -1)
                else:
                    unrolled_inputs[name] = seq_tensor.contiguous().view(batch_size * seq_len, -1)
                    
        h_prev_unrolled = torch.zeros(batch_size * seq_len, self.hidden_dim, device=self.device)
        u_t_unrolled = curr_u_t.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous().view(batch_size * seq_len, -1)
        
        w_t_unrolled, attn_weights_unrolled, channel_names, epistemic_entropy_unrolled = self.gateway(
            unrolled_inputs, h_prev_unrolled, u_t_unrolled
        )
        
        w_t_seq = w_t_unrolled.view(batch_size, seq_len, self.unified_dim)
        full_h_in = self.in_proj(w_t_seq)
        
        da_level = curr_u_t[:, 5:6]
        na_level = curr_u_t[:, 4:5]
        motor_gain = (1.0 + 1.0 * da_level).unsqueeze(1)

        active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
        if episodic_memory is not None and active_slots > 2:
            with torch.no_grad():
                q_sensory = w_t_seq[:, -1, :].float()
                ret_mem, max_sim = episodic_memory.read(q_sensory, temperature=0.05, threshold=0.70, sigmoid_beta=15.0)
            
            fact_feat = torch.cat([ret_mem, na_level], dim=-1)
            g_fact = self.fact_gate(fact_feat).unsqueeze(1)
            ret_mem_h = self.in_proj(ret_mem).unsqueeze(1)
            full_h_in = full_h_in + g_fact * ret_mem_h

        if self.training and self.device_str == 'cuda':
            h_s1, m_s1, dt1 = checkpoint.checkpoint(
                self._stage1_forward, full_h_in, m_s1, curr_u_t, use_reentrant=False
            )
        else:
            h_s1, m_s1, dt1 = self._stage1_forward(full_h_in, m_s1, curr_u_t)

        saliency_gate = self.boundary_detector(h_s1, text_seq)

        e1_weighted, _, _ = self.pw_lper(h_s1, h1_prev_last, curr_u_t)

        if self.training and self.device_str == 'cuda':
            h_s2, m_s2, dt2 = checkpoint.checkpoint(
                self._stage2_forward, e1_weighted, m_s2, curr_u_t, saliency_gate, use_reentrant=False
            )
        else:
            h_s2, m_s2, dt2 = self._stage2_forward(e1_weighted, m_s2, curr_u_t, saliency_gate)

        eff_dt = (dt1 + dt2) / 2.0

        topdown_prior = self.topdown_prior_proj(h_s2)

        h_combined = h_s1 + h_s2 + 0.15 * topdown_prior

        h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
        h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, curr_u_t)
        
        h_proj = self.motor_text_proj(h_relaxed).view(batch_size, seq_len, self.text_dim)
        h_proj_gain = (h_proj * motor_gain).contiguous().view(-1, self.text_dim)
        logits_flat = F.linear(h_proj_gain, self.pos_embeddings.byte_embed.weight)

        targets_flat = target_seq.contiguous().view(-1)
        speech_loss_tensor = criterion_speech(logits_flat, targets_flat)

        w_current_slice = w_t_seq[:, -1, :]
        h_curr_fast = h_combined[:, -1, :]
        w_pred, kl_div, fe, _ = self.world_model(h_prev_fast, h_curr_fast, w_current_slice)

        rec_loss = (1.0 - F.cosine_similarity(w_current_slice, w_pred, dim=-1, eps=1e-8)).mean()
        fe_loss_tensor = (kl_div.mean() + rec_loss)

        num_chunks = seq_len // chunk_size
        if num_chunks > 1:
            h_chunk_endpoints = h_combined.view(batch_size, num_chunks, chunk_size, self.hidden_dim)[:, :, -1, :]
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

        with torch.no_grad():
            if fe_loss_tensor.item() > 0.20 and episodic_memory is not None:
                episodic_memory.write(w_current_slice.detach().float(), w_pred.detach().float(), protected_slots=3)

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

    def execute_deep_allostatic_sleep_v2(self, episodic_memory: BatchedEpisodicMemory, hu: HomeostaticUnit,
                                         num_replay_cycles: int = 5, downscaling_factor: float = 0.03,
                                         pruning_percentile: float = 0.05):
        """
        Advanced Biophysical Sleep Synaptic Pruning & Morphogenesis (EXP-92):
        1. Hippocampal Replay over high-surprise episodes.
        2. Active Free-Energy Synaptic Pruning (zeros out bottom percentile of weights).
        3. Tononi SHY Synaptic Downscaling.
        4. Full Metabolic & Somatic Allostatic Reset.
        """
        self.train()
        active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
        active_memory_slots = min(active_slots, episodic_memory.max_capacity)
        
        # 1. Hippocampal Replay
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

        # 2. Synaptic Pruning (Morphogenesis)
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

        # 3. Tononi SHY Synaptic Downscaling
        with torch.no_grad():
            for param in self.get_all_parameters():
                if param.dim() > 1:
                    param.mul_(1.0 - downscaling_factor)

        # 4. Somatic Recovery
        with torch.no_grad():
            hu.state[:, 1] = 1.00 # Energy
            hu.state[:, 2] = 1.00 # Stability
            hu.state[:, 3] = 1.00 # Health
            hu.state[:, 4] = 0.05 # Noradrenaline reset

        return total_pruned_weights


def prepare_multimodal_packed_stream(num_batches: int = 100, batch_size: int = 32, seq_len: int = 512):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-92 Multimodal Stream (S={seq_len}, Steps={num_batches})...")
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
    tokenizer = ByteTokenizer()
    full_stream = []

    for item in ds:
        inst = item.get("instruction", "").strip()
        out = item.get("output", "").strip()
        if inst and out:
            dialog = f"User: {inst}\nKaryon: {out}"
            full_stream.extend(tokenizer.encode(dialog))
        if len(full_stream) >= num_batches * batch_size * (seq_len + 1):
            break

    batches = []
    block_size = seq_len + 1
    for b in range(num_batches):
        batch_tensors = []
        for s in range(batch_size):
            start = (b * batch_size + s) * block_size
            end = start + block_size
            chunk = full_stream[start:end]
            if len(chunk) < block_size:
                chunk = chunk + [256] * (block_size - len(chunk))
            batch_tensors.append(torch.tensor(chunk, dtype=torch.long))
        batches.append(torch.stack(batch_tensors, dim=0).to(device))

    logger.info(f"Prepared {len(batches)} Real Packed Multimodal Batches (B={batch_size}, S={seq_len}).")
    return batches


def run_exp_92_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-92 (UNIVERSAL MULTIMODAL & SLEEP MORPHOGENESIS)] ===")
    print("="*85)

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12

    b_size, seq_len = 32, 512
    num_eval_steps = 100
    chunk_size = 64

    batches = prepare_multimodal_packed_stream(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)

    # 1. INITIALIZE MULTIMODAL SLEEP AGENT
    print("\n" + "-"*85)
    print(" >>> INITIALIZING MULTIMODAL SLEEP AGENT & REGISTERING NEW CHANNELS <<<")
    print("-"*85)
    agent = MultimodalSleepAgent(config, device=device_str).to(device)
    
    # Dynamically register a new custom channel ('cybernetic_telemetry')
    agent.register_sensory_channel('cybernetic_telemetry', 128)
    print("  ✅ Dynamically registered new channel 'cybernetic_telemetry' (dim=128)")
    print("  ✅ Default channels active: ['text', 'vision', 'audio', 'binary', 'telepathic', 'document', 'cybernetic', 'motor']")

    hu = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    optimizer = torch.optim.AdamW(agent.get_all_parameters(), lr=3e-3, weight_decay=0.01)
    
    # Calibrated GradScaler init_scale=1024.0 to prevent sequence accumulation FP16 overflow
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp, init_scale=1024.0)
    crit_speech = nn.CrossEntropyLoss(ignore_index=256)

    # 2. PHASE 1: MULTIMODAL SEQUENCE TRAINING
    print("\n" + "-"*85)
    print(" >>> PHASE 1: RUNNING MULTIMODAL SEQUENCE PARALLEL TRAINING <<<")
    print("-"*85)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    train_losses = []
    
    for step, batch_tensors in enumerate(batches):
        inp_text = batch_tensors[:, :-1]
        tgt_text = batch_tensors[:, 1:]
        
        # Construct synthetic multimodal streams for active channels
        vision_stream = torch.randn(b_size, seq_len, 256, device=device) * 0.05
        document_stream = torch.randn(b_size, seq_len, 256, device=device) * 0.02
        cybernetic_stream = torch.randn(b_size, seq_len, 128, device=device) * 0.01
        
        sensor_seq_dict = {
            'text': inp_text,
            'vision': vision_stream,
            'document': document_stream,
            'cybernetic_telemetry': cybernetic_stream
        }

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt = agent.forward_multimodal_sequence(
                sensor_seq_dict, tgt_text, hu, crit_speech, episodic_memory=mem, chunk_size=chunk_size
            )
            
        scaler.scale(tot_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(agent.get_all_parameters(), max_norm=3.0)
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(s_loss)

        if (step + 1) % 25 == 0:
            print(f"  [Multimodal Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | Free Energy: {fe_loss:.4f} | Arousal(NA): {u_t[0,4]:.3f}")

    train_duration = time.perf_counter() - t0
    pre_sleep_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    pre_sleep_loss = sum(train_losses[-20:]) / 20.0
    pre_sleep_ppl = math.exp(min(pre_sleep_loss, 20.0))
    throughput = (num_eval_steps * b_size * seq_len) / train_duration

    print(f"\n[Pre-Sleep Summary] Loss: {pre_sleep_loss:.4f} | PPL: {pre_sleep_ppl:.2f} | Peak VRAM: {pre_sleep_vram:.1f} MB | Throughput: {throughput:.1f} tok/s")

    # 3. PHASE 2: BIOPHYSICAL SLEEP SYNAPTIC PRUNING & MORPHOGENESIS
    print("\n" + "-"*85)
    print(" >>> PHASE 2: EXECUTING DEEP ALLOSTATIC SLEEP & SYNAPTIC PRUNING <<<")
    print("-"*85)
    
    pre_prune_zeros = sum((param == 0).sum().item() for name, param in agent.named_parameters() if param.dim() > 1)
    
    t_sleep0 = time.perf_counter()
    pruned_count = agent.execute_deep_allostatic_sleep_v2(mem, hu, num_replay_cycles=5, downscaling_factor=0.03, pruning_percentile=0.05)
    sleep_duration = time.perf_counter() - t_sleep0

    post_prune_zeros = sum((param == 0).sum().item() for name, param in agent.named_parameters() if param.dim() > 1)

    print(f"  ✅ Sleep Cycle Completed in {sleep_duration*1000.0:.2f} ms")
    print(f"  ✅ Synaptic Pruning Summary: {pruned_count} weak connections zeroed out (Zero Weights: {pre_prune_zeros} -> {post_prune_zeros})")
    print(f"  ✅ Somatic Recovery: Energy={hu.state[0,1]:.2f}, Health={hu.state[0,3]:.2f}, Arousal(NA) Reset={hu.state[0,4]:.2f}")

    # 4. PHASE 3: POST-SLEEP EVALUATION (VERIFY NO REGRESSION)
    print("\n" + "-"*85)
    print(" >>> PHASE 3: POST-SLEEP EVALUATION (CONVERGENCE & PARITY VERIFICATION) <<<")
    print("-"*85)
    
    agent.eval()
    post_sleep_losses = []
    
    with torch.no_grad():
        for eval_step in range(20):
            eval_batch = batches[eval_step]
            inp_text = eval_batch[:, :-1]
            tgt_text = eval_batch[:, 1:]
            
            vision_stream = torch.randn(b_size, seq_len, 256, device=device) * 0.05
            document_stream = torch.randn(b_size, seq_len, 256, device=device) * 0.02
            cybernetic_stream = torch.randn(b_size, seq_len, 128, device=device) * 0.01
            
            sensor_seq_dict = {
                'text': inp_text,
                'vision': vision_stream,
                'document': document_stream,
                'cybernetic_telemetry': cybernetic_stream
            }
            
            with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
                tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt = agent.forward_multimodal_sequence(
                    sensor_seq_dict, tgt_text, hu, crit_speech, episodic_memory=mem, chunk_size=chunk_size
                )
            post_sleep_losses.append(s_loss)

    post_sleep_loss = sum(post_sleep_losses) / len(post_sleep_losses)
    post_sleep_ppl = math.exp(min(post_sleep_loss, 20.0))
    delta_loss = pre_sleep_loss - post_sleep_loss

    print(f"\n[Post-Sleep Summary] Loss: {post_sleep_loss:.4f} | PPL: {post_sleep_ppl:.2f} | Delta Loss: {delta_loss:+.4f}")

    # 5. KEP RULE #6: GRADIENT FLOW AUDIT
    print("\n" + "="*85)
    print(" === [KEP RULE #6 GRADIENT FLOW & MULTIMODAL CHANNEL AUDIT] ===")
    print("="*85)
    agent.train()
    
    # Run 1 dummy backward step to verify gradient health across all projections
    dummy_inp = batches[0][:, :-1]
    dummy_tgt = batches[0][:, 1:]
    dummy_dict = {
        'text': dummy_inp,
        'vision': torch.randn(b_size, seq_len, 256, device=device),
        'document': torch.randn(b_size, seq_len, 256, device=device),
        'cybernetic_telemetry': torch.randn(b_size, seq_len, 128, device=device)
    }
    optimizer.zero_grad()
    tot_loss, _, _, _, _, _, _ = agent.forward_multimodal_sequence(dummy_dict, dummy_tgt, hu, crit_speech, episodic_memory=mem)
    scaler.scale(tot_loss).backward()
    scaler.unscale_(optimizer)

    zero_grads = 0
    healthy_grads = 0
    for name, param in agent.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.norm().item()
            if g_norm > 0:
                healthy_grads += 1
                status = "✅ HEALTHY"
            else:
                zero_grads += 1
                status = "⚠️ ZERO GRAD"
        else:
            zero_grads += 1
            status = "⚠️ DISCONNECTED"
        if "gateway" in name or "pos_embeddings" in name or "stage1" in name or "stage2" in name:
            print(f"  {name:<55} | Grad Norm: {g_norm if param.grad is not None else 0.0:<12.6f} | {status}")

    print("-" * 85)
    print(f"Audit Summary: Total Params Audited: {len(list(agent.named_parameters()))} | Healthy Grads: {healthy_grads} | Zero/Disconnected: {zero_grads}")

    # 6. KEP RULE #4 DIAGNOSTIC SPEECH SAMPLE
    print("\n" + "="*85)
    print(" === [KEP RULE #4 DIAGNOSTIC SPEECH SAMPLE AUDIT] ===")
    print("="*85)
    agent.eval()
    diag_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"
    diag_hu = HomeostaticUnit(batch_size=1, device=device_str)
    diag_mem = BatchedEpisodicMemory(batch_size=1, memory_dim=256, max_capacity=200, device=device_str)
    
    sample_chars = []
    with torch.no_grad():
        gen_stream = agent.generate_thought_and_speech(
            prompt=diag_prompt,
            m_state=torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device),
            h_state=torch.zeros(1, agent.hidden_dim, device=device),
            hu=diag_hu,
            episodic_memory=diag_mem,
            config=config,
            max_generated_tokens=60
        )
        for event in gen_stream:
            if event["status"] == "token":
                sample_chars.append(event["text"])

    print(f"  Prompt : \"{diag_prompt.strip()}\"")
    print(f"  Sample : \"{''.join(sample_chars).strip()}\"")
    print("="*85)

    # 7. KEP RULE #2 VERDICT
    print("\n" + "="*85)
    print(" === [KEP RULE #2 FINAL EMPIRICAL DECISION EVALUATION] ===")
    print("="*85)
    print(f"Pre-Sleep Loss     : {pre_sleep_loss:.4f} (PPL: {pre_sleep_ppl:.2f})")
    print(f"Post-Sleep Loss    : {post_sleep_loss:.4f} (PPL: {post_sleep_ppl:.2f})")
    print(f"Delta Loss         : {delta_loss:+.4f}")
    print(f"Pruned Connections : {pruned_count} weights (5.0% percentile)")
    print(f"Throughput Speed   : {throughput:.1f} tok/s")
    print(f"Peak VRAM          : {pre_sleep_vram:.1f} MB")

    if delta_loss >= -0.05 and throughput >= 18000.0 and zero_grads == 0:
        verdict = "🟢 POSITIVE"
        print(f"VERDICT            : {verdict} (Universal Multimodal Gateway & Sleep Pruning Validated!)")
    elif delta_loss < -0.08:
        verdict = "🔴 REJECTED"
        print(f"VERDICT            : {verdict} (Loss regression)")
    else:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"
        print(f"VERDICT            : {verdict}")
    print("="*85 + "\n")

    return {
        "verdict": verdict,
        "pre_sleep_loss": pre_sleep_loss,
        "post_sleep_loss": post_sleep_loss,
        "delta_loss": delta_loss,
        "pruned_weights": pruned_count,
        "throughput_tok_per_sec": throughput,
        "vram_mb": pre_sleep_vram
    }


if __name__ == "__main__":
    run_exp_92_benchmark()
