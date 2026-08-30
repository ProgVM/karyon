# experiments/exp_98_volitional_active_inference.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-98 (CONTINUOUS VOLITIONAL ACTIVE INFERENCE)
Testing Continuous Real Volition and Endogenous Intent:
1. Continuous Expected Free Energy Motor Modulation:
   Directly coupling motor readout logits to Expected Free Energy G(a) computed in
   the Latent World Model under homeostatic prior preferences P(u).
2. Spontaneous Endogenous Thought Loop (Self-Initiated Cognitive Volition):
   When external input is zero, high SEEKING drive triggers self-directed latent
   rollouts feeding back into the Global Workspace, preventing cognitive freezing.

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
from datasets import load_dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from karyon_agent import CoREAgent, DynamicSensoryGateway
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')


# =============================================================================
# 1. VOLITIONAL ACTIVE INFERENCE MOTOR MODULE (DIRECT G-MODULATION)
# =============================================================================

class VolitionalActiveInferenceMotorHead(nn.Module):
    """
    Continuous Volitional Action Selection Engine (Friston Active Inference):
    Evaluates Expected Free Energy G(a) for motor trajectories and modulates
    logits directly: Logits = Logits_raw - gamma * G(a).
    """
    def __init__(self, hidden_dim=768, text_dim=256, vocab_size=258, gamma_volition=0.15, device_str='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.vocab_size = vocab_size
        self.gamma_volition = gamma_volition
        self.device = torch.device(device_str)

        self.motor_text_proj = nn.Sequential(
            nn.Linear(hidden_dim, text_dim),
            nn.SiLU(),
            nn.LayerNorm(text_dim)
        ).to(self.device)

        self.efe_evaluator = nn.Sequential(
            nn.Linear(text_dim + 6, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        ).to(self.device)

    def compute_volitional_logits(self, h_relaxed: torch.Tensor, u_t: torch.Tensor, byte_embed_weights: torch.Tensor) -> torch.Tensor:
        batch_size = h_relaxed.size(0)
        da_level = u_t[:, 5:6]
        motor_gain = (1.0 + 1.0 * da_level)

        h_proj = self.motor_text_proj(h_relaxed)
        h_proj_gain = h_proj * motor_gain
        raw_logits = F.linear(h_proj_gain, byte_embed_weights)

        # Compute Expected Free Energy G(a) over top-8 candidate tokens for speed
        top8_vals, top8_indices = torch.topk(raw_logits, k=8, dim=-1)
        
        # Gather top-8 byte embeddings
        top8_embs = byte_embed_weights[top8_indices] # [B, 8, text_dim]
        u_t_expanded = u_t.unsqueeze(1).expand(batch_size, 8, 6) # [B, 8, 6]
        
        efe_inputs = torch.cat([top8_embs, u_t_expanded], dim=-1)
        g_scores = self.efe_evaluator(efe_inputs).squeeze(-1) # [B, 8]
        
        # Volitional modulation: subtract gamma * G(a) from candidate logits
        volitional_mod = -self.gamma_volition * g_scores
        
        modulated_logits = raw_logits.clone()
        modulated_logits.scatter_add_(1, top8_indices, volitional_mod)
        
        return modulated_logits


# =============================================================================
# 2. VOLITIONAL AGENT ARCHITECTURE FOR EXP-98
# =============================================================================

class VolitionalCoREAgent(CoREAgent):
    """CoREAgent extended with Continuous Volitional Active Inference Head & Endogenous Intent."""
    def __init__(self, config, device='cpu'):
        super().__init__(config, device)
        self.volitional_head = VolitionalActiveInferenceMotorHead(
            hidden_dim=self.hidden_dim,
            text_dim=self.text_dim,
            vocab_size=self.text_gen_dim,
            gamma_volition=0.15,
            device_str=self.device_str
        )

    def get_all_parameters(self):
        params = super().get_all_parameters()
        params.extend(list(self.volitional_head.parameters()))
        return params

    def forward_volitional_sequence(self, sensor_seq_dict, target_seq, hu_batch, criterion_speech, episodic_memory=None, chunk_size=64):
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
                    
        h_prev_unrolled = torch.zeros(batch_size * seq_len, self.hidden_dim, device=self.device).float()
        u_t_unrolled = curr_u_t.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous().view(batch_size * seq_len, -1).float()
        
        with torch.amp.autocast(device_type=self.device_str, enabled=False):
            w_t_unrolled, attn_weights_unrolled, channel_names, epistemic_entropy_unrolled = self.gateway(
                unrolled_inputs, h_prev_unrolled, u_t_unrolled
            )
        
        w_t_seq = w_t_unrolled.view(batch_size, seq_len, self.unified_dim)
        
        with torch.amp.autocast(device_type=self.device_str, dtype=torch.float16, enabled=(self.device_str == 'cuda')):
            full_h_in = self.in_proj(w_t_seq)

            h_s1, m_s1, dt1 = self._stage1_forward(full_h_in, m_s1, curr_u_t)
            saliency_gate = self.boundary_detector(h_s1, text_seq)

            e1_weighted, _, _ = self.pw_lper(h_s1, h1_prev_last, curr_u_t)

            predicted_entropy = self.entropy_predictor(h_s1)
            dynamic_dt_scale = 0.40 + 1.20 * predicted_entropy

            h_s2, m_s2, dt2 = self._stage2_forward(e1_weighted, m_s2, curr_u_t, saliency_gate, dynamic_dt_scale.mean().item())
            h_s2 = h_s2 * dynamic_dt_scale

            eff_dt = (dt1 + dt2) / 2.0
            topdown_prior = self.topdown_prior_proj(h_s2)
            h_combined = h_s1 + h_s2 + 0.15 * topdown_prior

            h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
            h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, curr_u_t)
            
            # Volitional Motor Readout with Expected Free Energy Modulation
            u_t_unrolled_step = curr_u_t.repeat_interleave(seq_len, dim=0)
            volitional_logits_flat = self.volitional_head.compute_volitional_logits(
                h_relaxed, u_t_unrolled_step, self.pos_embeddings.byte_embed.weight
            )

            targets_flat = target_seq.contiguous().view(-1)
            speech_loss_tensor = criterion_speech(volitional_logits_flat, targets_flat)

            w_current_slice = w_t_seq[:, -1, :]
            h_curr_fast = h_combined[:, -1, :]
            w_pred, kl_div, fe, _ = self.world_model(h_prev_fast, h_curr_fast, w_current_slice)

            rec_loss = (1.0 - F.cosine_similarity(w_current_slice, w_pred, dim=-1, eps=1e-8)).mean()
            fe_loss_tensor = (kl_div.mean() + rec_loss)

            ortho_loss = self.attractor_head.compute_pattern_separation_loss()
            
            speech_loss_val = speech_loss_tensor.item()
            fe_loss_val = fe_loss_tensor.item()
            
            total_loss_tensor = (
                speech_loss_tensor + 
                0.05 * fe_loss_tensor + 
                0.05 * commit_loss + 
                0.01 * ortho_loss
            )

        h_proxy = m_s2.view(batch_size, -1)[:, :self.hidden_dim]
        return total_loss_tensor, speech_loss_val, fe_loss_val, m_s2, h_proxy, curr_u_t, eff_dt


# =============================================================================
# 3. BENCHMARK STREAM PREPARATION
# =============================================================================

def prepare_multimodal_packed_stream(num_batches: int = 100, batch_size: int = 16, seq_len: int = 512):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-98 (S={seq_len}, Steps={num_batches})...")
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
    tokenizer = ByteTokenizer()
    full_text_stream = []

    required_tokens = num_batches * batch_size * (seq_len + 1)
    
    while len(full_text_stream) < required_tokens:
        for item in ds:
            inst = item.get("instruction", "").strip()
            out = item.get("output", "").strip()
            if inst and out:
                dialog = f"User: {inst}\nKaryon: {out}"
                full_text_stream.extend(tokenizer.encode(dialog))
            if len(full_text_stream) >= required_tokens:
                break

    full_text_stream = full_text_stream[:required_tokens]

    batches = []
    block_size = seq_len + 1
    for b in range(num_batches):
        text_batch = []
        vision_batch = []
        audio_batch = []
        
        for s in range(batch_size):
            start = (b * batch_size + s) * block_size
            end = start + block_size
            chunk = full_text_stream[start:end]
            
            text_tensor = torch.tensor(chunk, dtype=torch.long)
            text_batch.append(text_tensor)
            
            vis_signal = torch.randn(seq_len + 1, 256) * 0.05
            vision_batch.append(vis_signal)
            
            aud_signal = torch.randn(seq_len + 1, 256) * 0.05
            audio_batch.append(aud_signal)

        batches.append({
            'text': torch.stack(text_batch, dim=0).to(device),
            'vision': torch.stack(vision_batch, dim=0).to(device),
            'audio': torch.stack(audio_batch, dim=0).to(device)
        })

    logger.info(f"Prepared {len(batches)} Volitional Multimodal Batches (B={batch_size}, S={seq_len}).")
    return batches


# =============================================================================
# 4. EXP-98 BENCHMARK EXECUTION
# =============================================================================

def run_exp_98_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-98 (VOLITIONAL ACTIVE INFERENCE)] ===")
    print("="*85)

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12

    b_size, seq_len = 16, 512
    num_eval_steps = 100
    chunk_size = 64

    batches = prepare_multimodal_packed_stream(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)

    # -------------------------------------------------------------------------
    # PHASE 1: BASELINE (Standard CoREAgent without Volitional Motor Head)
    # -------------------------------------------------------------------------
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 1: BASELINE (CoREAgent without Volitional G-Modulation) <<<")
    print("-"*85)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    agent_base = CoREAgent(config, device=device_str).to(device)
    hu_base = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_base = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_base = torch.optim.AdamW(agent_base.get_all_parameters(), lr=1e-3, weight_decay=0.01)
    scaler_base = torch.amp.GradScaler('cuda', enabled=use_amp)
    crit_speech = nn.CrossEntropyLoss(ignore_index=256)

    t0 = time.perf_counter()
    base_losses = []
    
    for step, batch_dict in enumerate(batches):
        inp_dict = {
            'text': batch_dict['text'][:, :-1],
            'vision': batch_dict['vision'][:, :-1, :],
            'audio': batch_dict['audio'][:, :-1, :]
        }
        tgt = batch_dict['text'][:, 1:]
        
        opt_base.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt = agent_base.forward_multimodal_sequence(
                inp_dict, tgt, hu_base, crit_speech, episodic_memory=mem_base, chunk_size=chunk_size
            )
        scaler_base.scale(tot_loss).backward()
        scaler_base.unscale_(opt_base)
        torch.nn.utils.clip_grad_norm_(agent_base.get_all_parameters(), max_norm=3.0)
        scaler_base.step(opt_base)
        scaler_base.update()
        base_losses.append(s_loss)
        
        if (step + 1) % 25 == 0:
            print(f"  [Baseline Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | Free Energy: {fe_loss:.4f}")

    base_duration = time.perf_counter() - t0
    base_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    base_final_loss = sum(base_losses[-20:]) / 20.0
    base_ppl = math.exp(min(base_final_loss, 20.0))
    base_tok_per_sec = (num_eval_steps * b_size * seq_len) / base_duration

    print(f"\n[Baseline Summary] Final Loss: {base_final_loss:.4f} | PPL: {base_ppl:.2f} | Peak VRAM: {base_vram:.1f} MB | Throughput: {base_tok_per_sec:.1f} tok/s")

    # Clean VRAM between phases
    del agent_base, opt_base, scaler_base, hu_base, mem_base
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # PHASE 2: PROPOSED (VolitionalCoREAgent with Volitional G-Modulation)
    # -------------------------------------------------------------------------
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 2: PROPOSED (VolitionalCoREAgent with G-Modulation) <<<")
    print("-"*85)
    torch.cuda.reset_peak_memory_stats()

    agent_prop = VolitionalCoREAgent(config, device=device_str).to(device)
    hu_prop = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_prop = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_prop = torch.optim.AdamW(agent_prop.get_all_parameters(), lr=1e-3, weight_decay=0.01)
    scaler_prop = torch.amp.GradScaler('cuda', enabled=use_amp)

    t0 = time.perf_counter()
    prop_losses = []
    
    for step, batch_dict in enumerate(batches):
        inp_dict = {
            'text': batch_dict['text'][:, :-1],
            'vision': batch_dict['vision'][:, :-1, :],
            'audio': batch_dict['audio'][:, :-1, :]
        }
        tgt = batch_dict['text'][:, 1:]
        
        opt_prop.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt = agent_prop.forward_volitional_sequence(
                inp_dict, tgt, hu_prop, crit_speech, episodic_memory=mem_prop, chunk_size=chunk_size
            )
        scaler_prop.scale(tot_loss).backward()
        scaler_prop.unscale_(opt_prop)
        torch.nn.utils.clip_grad_norm_(agent_prop.get_all_parameters(), max_norm=3.0)
        scaler_prop.step(opt_prop)
        scaler_prop.update()
        prop_losses.append(s_loss)
        
        if (step + 1) % 25 == 0:
            print(f"  [Proposed Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | Free Energy: {fe_loss:.4f}")

    prop_duration = time.perf_counter() - t0
    prop_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    prop_final_loss = sum(prop_losses[-20:]) / 20.0
    prop_ppl = math.exp(min(prop_final_loss, 20.0))
    prop_tok_per_sec = (num_eval_steps * b_size * seq_len) / prop_duration

    print(f"\n[Proposed Summary] Final Loss: {prop_final_loss:.4f} | PPL: {prop_ppl:.2f} | Peak VRAM: {prop_vram:.1f} MB | Throughput: {prop_tok_per_sec:.1f} tok/s")

    # -------------------------------------------------------------------------
    # KEP DECISION EVALUATION
    # -------------------------------------------------------------------------
    delta_loss = base_final_loss - prop_final_loss
    throughput_retention_pct = (prop_tok_per_sec / base_tok_per_sec) * 100.0

    print("\n" + "="*85)
    print(" === [KEP RULE #2 FINAL EMPIRICAL DECISION EVALUATION] ===")
    print("="*85)
    print(f"Baseline Loss      : {base_final_loss:.4f} (PPL: {base_ppl:.2f}) | VRAM: {base_vram:.1f} MB | Speed: {base_tok_per_sec:.1f} tok/s")
    print(f"Proposed Loss      : {prop_final_loss:.4f} (PPL: {prop_ppl:.2f}) | VRAM: {prop_vram:.1f} MB | Speed: {prop_tok_per_sec:.1f} tok/s")
    print(f"Delta Loss         : {delta_loss:+.4f} ({'IMPROVEMENT' if delta_loss >= 0.08 else 'PARITY' if abs(delta_loss) < 0.08 else 'REGRESSION'})")
    print(f"Speed Retention    : {throughput_retention_pct:.1f}%")

    if delta_loss >= 0.08 and throughput_retention_pct >= 80.0:
        verdict = "🟢 POSITIVE"
    elif delta_loss < -0.08:
        verdict = "🔴 REJECTED"
    else:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    print(f"VERDICT            : {verdict}")
    print("="*85 + "\n")


if __name__ == "__main__":
    run_exp_98_benchmark()
