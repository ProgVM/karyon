# experiments/exp_102_embodied_hourglass_ssm.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-102
Hypothesis:
1. Embodied Tri-Aspect Character Gateway (ETCG) mapping bytes into visual (graphemic),
   acoustic (phonemic), and motor (articulatory) fields grounds character modeling in
   embodied cognitive neuroscience, accelerating speech loss convergence.
2. Hourglass Macro-SSD Layer (2-level temporal hierarchy: Q_micro=64, Q_macro=512)
   prevents long-horizon temporal dilution, dropping speech loss by >= 0.08 nats
   with stable VRAM and high throughput.
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
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from karyon_agent import CoREAgent, OffsetPositionalByteEmbedding
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory, CorticalStage

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')


# =============================================================================
# 1. REFORMULATED EMBODIED TRI-ASPECT CHARACTER GATEWAY (ETCG)
# =============================================================================

class EmbodiedTriAspectByteEmbedding(nn.Module):
    """
    Dehaene Triple-Code Model of Reading (Reformulated under KEP Rule #9):
    Applies direct, linear, precision-weighted embodied modulations on top of the
    canonical tied embedding, preserving 100% of the direct gradient highway.
    Gating parameters are initialized to small positive values (0.02) to kickstart embodied learning.
    """
    def __init__(self, vocab_size=258, text_dim=256, max_len=8192, device_str='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.text_dim = text_dim
        self.device_str = device_str
        self.device = torch.device(device_str)

        # Canonical Tied Embedding (Direct Gradient Highway)
        self.byte_embed = nn.Embedding(vocab_size, text_dim)

        # Embodied Modulation Fields
        self.grapheme_embed = nn.Embedding(vocab_size, text_dim)
        self.phoneme_embed = nn.Embedding(vocab_size, text_dim)
        self.motor_embed = nn.Embedding(vocab_size, text_dim)

        # Precision-weighting gating scalars initialized to small positive values (0.02)
        self.gamma_grapheme = nn.Parameter(torch.full((1, 1, 1), 0.02))
        self.gamma_phoneme = nn.Parameter(torch.full((1, 1, 1), 0.02))
        self.gamma_motor = nn.Parameter(torch.full((1, 1, 1), 0.02))

        self.receptive_field = karyon_core.MultiScaleBytePyramidReceptiveField(text_dim=text_dim, device=device_str)
        
        # Sinusoidal positional embeddings
        pe = torch.zeros(max_len, text_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, text_dim, 2).float() * (-math.log(10000.0) / text_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        self._initialize_embodied_priors()

    def _initialize_embodied_priors(self):
        """Initializes visual, acoustic, and motor fields with biologically realistic priors."""
        with torch.no_grad():
            nn.init.normal_(self.byte_embed.weight, mean=0.0, std=0.08)
            
            g_weight = torch.randn(self.vocab_size, self.text_dim) * 0.08
            p_weight = torch.randn(self.vocab_size, self.text_dim) * 0.08
            m_weight = torch.randn(self.vocab_size, self.text_dim) * 0.08

            vowels = [97, 101, 105, 111, 117, 65, 69, 73, 79, 85]
            plosives = [112, 116, 107, 98, 100, 103, 80, 84, 75, 66, 68, 71]

            for v in vowels:
                if v < self.vocab_size:
                    p_weight[v, :32] += 1.2
            for p in plosives:
                if p < self.vocab_size:
                    m_weight[p, :32] += 1.0

            self.grapheme_embed.weight.copy_(g_weight)
            self.phoneme_embed.weight.copy_(p_weight)
            self.motor_embed.weight.copy_(m_weight)

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, apply_rf: bool = True) -> torch.Tensor:
        seq_len = input_ids.size(1)
        
        tok_emb = self.byte_embed(input_ids) * math.sqrt(self.text_dim)
        
        g_mod = self.grapheme_embed(input_ids)
        p_mod = self.phoneme_embed(input_ids)
        m_mod = self.motor_embed(input_ids)
        
        embodied_mod = (
            self.gamma_grapheme * g_mod +
            self.gamma_phoneme * p_mod +
            self.gamma_motor * m_mod
        )
        
        embedded = tok_emb + embodied_mod
        
        pos_emb = self.pe[:, start_pos : start_pos + seq_len, :]
        embedded = embedded + pos_emb
        
        if apply_rf and seq_len > 1:
            embedded = self.receptive_field(embedded)
        return embedded


# =============================================================================
# 2. REFORMULATED HOURGLASS MACRO-SSD LAYER (GATED INTEGRATION)
# =============================================================================

class HourglassMacroSSDLayer(nn.Module):
    """
    2-Level Temporal Hierarchy (Reformulated under KEP Rule #9):
    Gates macro-level context using a learnable Sigmoid Gate initialized to zero,
    preventing untrained noise from disrupting early convergence.
    """
    def __init__(self, hidden_dim=768, num_heads=12, head_k=64, head_v=128, device_str='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_k = head_k
        self.head_v = head_v
        self.device = torch.device(device_str)

        self.macro_decay = nn.Parameter(torch.full((1, num_heads, 1, 1), 0.9998))
        self.macro_k_proj = nn.Linear(hidden_dim, num_heads * head_k)
        self.macro_v_proj = nn.Linear(hidden_dim, num_heads * head_v)
        self.macro_q_proj = nn.Linear(hidden_dim, num_heads * head_k)

        self.macro_norm = nn.GroupNorm(num_heads, num_heads * head_v)
        self.macro_out_proj = nn.Linear(num_heads * head_v, hidden_dim)

        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.constant_(self.gate_proj.weight, 0.0)
        nn.init.constant_(self.gate_proj.bias, -6.0) # Sigmoid(-6.0) ~ 0.0025

    def forward(self, h_micro: torch.Tensor, m_macro_prev: torch.Tensor, chunk_size: int = 64) -> tuple:
        batch_size, seq_len, _ = h_micro.size()
        num_chunks = seq_len // chunk_size

        if num_chunks == 0:
            return h_micro, m_macro_prev

        h_macro_frames = h_micro.view(batch_size, num_chunks, chunk_size, self.hidden_dim)[:, :, -1, :]

        m_q = self.macro_q_proj(h_macro_frames).view(batch_size, num_chunks, self.num_heads, self.head_k).transpose(1, 2)
        m_k = self.macro_k_proj(h_macro_frames).view(batch_size, num_chunks, self.num_heads, self.head_k).transpose(1, 2)
        m_v = self.macro_v_proj(h_macro_frames).view(batch_size, num_chunks, self.num_heads, self.head_v).transpose(1, 2)

        m_curr = m_macro_prev.to(torch.float32)
        y_macro_list = []
        
        alpha = torch.clamp(self.macro_decay, 0.9990, 0.99999)
        beta = 1.0 - alpha

        for c in range(num_chunks):
            q_c = m_q.select(2, c).unsqueeze(-1)
            y_c = torch.matmul(m_curr, q_c).squeeze(-1)
            y_macro_list.append(y_c)

            k_c = m_k.select(2, c).unsqueeze(-1)
            v_c = m_v.select(2, c).unsqueeze(-1)
            m_curr = alpha * m_curr + beta * torch.matmul(v_c, k_c.transpose(-1, -2))

        y_macro = torch.stack(y_macro_list, 2)
        y_macro_flat = y_macro.permute(0, 2, 1, 3).reshape(batch_size * num_chunks, self.num_heads * self.head_v)
        y_macro_normed = self.macro_norm(y_macro_flat)
        h_macro_context = self.macro_out_proj(y_macro_normed).view(batch_size, num_chunks, self.hidden_dim)

        h_macro_interpolated = h_macro_context.repeat_interleave(chunk_size, dim=1)
        
        g_t = torch.sigmoid(self.gate_proj(h_micro))
        h_hourglass_out = h_micro + g_t * h_macro_interpolated

        return h_hourglass_out, m_curr


# =============================================================================
# 3. EMBODIED HOURGLASS CORE AGENT
# =============================================================================

class EmbodiedHourglassCoREAgent(CoREAgent):
    """Karyon Agent with Embodied Tri-Aspect Gateway and Hourglass Macro-SSD."""
    def __init__(self, config, device='cpu'):
        super().__init__(config, device)
        
        self.pos_embeddings = EmbodiedTriAspectByteEmbedding(
            vocab_size=self.text_gen_dim,
            text_dim=self.text_dim,
            max_len=8192,
            device_str=self.device_str
        ).to(self.device)

        self.hourglass_macro = HourglassMacroSSDLayer(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            head_k=self.head_k,
            head_v=self.head_v,
            device_str=self.device_str
        ).to(self.device)

    def get_all_parameters(self):
        params = super().get_all_parameters()
        params.extend(list(self.hourglass_macro.parameters()))
        return params

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion_speech: nn.Module, episodic_memory=None, loss_free_energy_weight: float = 0.05, 
                         chunk_size: int = 64):
        batch_size, seq_len = input_seq.size()
        
        m_s1 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_macro = torch.zeros(batch_size, self.num_heads, self.head_v, self.head_k, dtype=torch.float32, device=self.device)
        
        curr_u_t = hu_batch.state.clone().detach()
        h_prev_fast = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        h1_prev_last = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)
        
        full_emb = self.pos_embeddings(input_seq, start_pos=0, apply_rf=True)
        full_h_in = self.in_proj(full_emb)
        
        da_level = curr_u_t[:, 5:6]
        motor_gain = (1.0 + 1.0 * da_level).unsqueeze(1)

        # Stage 1
        h_s1, m_s1, dt1 = self._stage1_forward(full_h_in, m_s1, curr_u_t)

        saliency_gate = self.boundary_detector(h_s1, input_seq)
        e1_weighted, _, _ = self.pw_lper(h_s1, h1_prev_last, curr_u_t)

        # Stage 2
        h_s2, m_s2, dt2 = self._stage2_forward(e1_weighted, m_s2, curr_u_t, saliency_gate)

        # Hourglass Macro-SSD Integration
        h_s2_hourglass, m_macro = self.hourglass_macro(h_s2, m_macro, chunk_size=chunk_size)

        eff_dt = (dt1 + dt2) / 2.0
        topdown_prior = self.topdown_prior_proj(h_s2_hourglass)
        h_combined = h_s1 + h_s2_hourglass + 0.15 * topdown_prior

        h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
        h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, curr_u_t)
        
        h_proj = self.motor_text_proj(h_relaxed).view(batch_size, seq_len, self.text_dim)
        h_proj_gain = (h_proj * motor_gain).contiguous().view(-1, self.text_dim)
        logits_flat = F.linear(h_proj_gain, self.pos_embeddings.byte_embed.weight)

        targets_flat = target_seq.contiguous().view(-1)
        speech_loss_tensor = criterion_speech(logits_flat, targets_flat)

        w_current_slice = self.episodic_sensory_proj(full_emb[:, -1, :])
        h_curr_fast = h_combined[:, -1, :]
        w_pred, kl_div, fe, _ = self.world_model(h_prev_fast, h_curr_fast, w_current_slice)

        rec_loss = (1.0 - F.cosine_similarity(w_current_slice, w_pred, dim=-1, eps=1e-8)).mean()
        fe_loss_tensor = (kl_div.mean() + rec_loss)

        num_chunks = seq_len // chunk_size
        h_chunk_endpoints = h_combined.view(batch_size, num_chunks, chunk_size, self.hidden_dim)[:, :, -1, :]
        
        v_preds = self.critic(h_chunk_endpoints).squeeze(-1)
        
        gamma_fe = 0.90
        fe_per_batch = fe.squeeze(-1)
        v_current = v_preds[:, :-1]
        v_next = v_preds[:, 1:].detach()
        r_step = -0.10 * fe_per_batch.unsqueeze(1).expand_as(v_current)
        td_targets = r_step + gamma_fe * v_next
        critic_loss = F.mse_loss(v_current, td_targets)

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


# =============================================================================
# 4. BENCHMARK RUNNER
# =============================================================================

def prepare_packed_stream(num_batches: int = 250, batch_size: int = 8, seq_len: int = 1024):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-102 (S={seq_len}, Steps={num_batches})...")
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

    logger.info(f"Prepared {len(batches)} Real Packed Batches (B={batch_size}, S={seq_len}).")
    return batches


def run_exp_102_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-102 (EMBODIED HOURGLASS SSM)] ===")
    print("="*85)
    print(f"Hardware Device : {device_str.upper()} | AMP FP16 Enabled: {use_amp}")

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12

    b_size, seq_len = 8, 1024
    num_eval_steps = 250
    chunk_size = 64

    batches = prepare_packed_stream(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)

    # 1. EVALUATE BASELINE (Standard CoREAgent v31.0 Master)
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 1: BASELINE (Standard CoREAgent) <<<")
    print("-"*85)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    agent_base = CoREAgent(config, device=device_str).to(device)
    hu_base = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_base = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_base = torch.optim.AdamW(agent_base.get_all_parameters(), lr=1e-3, weight_decay=0.01)
    scaler_base = torch.amp.GradScaler('cuda', enabled=use_amp)
    crit_speech = nn.CrossEntropyLoss(ignore_index=256)

    warmup_scheduler_base = torch.optim.lr_scheduler.LambdaLR(opt_base, lr_lambda=lambda step: min(1.0, (step + 1) / 20.0))

    t0 = time.perf_counter()
    base_losses = []
    
    for step, batch_tensors in enumerate(batches):
        inp = batch_tensors[:, :-1]
        tgt = batch_tensors[:, 1:]
        opt_base.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt = agent_base.forward_sequence(
                inp, tgt, hu_base, crit_speech, episodic_memory=mem_base, chunk_size=chunk_size
            )
        scaler_base.scale(tot_loss).backward()
        scaler_base.unscale_(opt_base)
        torch.nn.utils.clip_grad_norm_(agent_base.get_all_parameters(), max_norm=3.0)
        scaler_base.step(opt_base)
        scaler_base.update()
        warmup_scheduler_base.step()
        base_losses.append(s_loss)
        if (step + 1) % 50 == 0:
            print(f"  [Baseline Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | Free Energy: {fe_loss:.4f}")

    base_duration = time.perf_counter() - t0
    base_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    base_final_loss = sum(base_losses[-30:]) / 30.0
    base_ppl = math.exp(min(base_final_loss, 20.0))
    base_tok_per_sec = (num_eval_steps * b_size * seq_len) / base_duration

    print(f"\n[Baseline Summary] Final Loss: {base_final_loss:.4f} | PPL: {base_ppl:.2f} | Peak VRAM: {base_vram:.1f} MB | Throughput: {base_tok_per_sec:.1f} tok/s")

    # Explicitly delete baseline variables to free VRAM
    del agent_base, opt_base, scaler_base, hu_base, mem_base, base_losses
    gc.collect()
    torch.cuda.empty_cache()

    # 2. EVALUATE PROPOSED (Embodied Tri-Aspect Gateway + Hourglass Macro-SSD)
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 2: PROPOSED (Embodied Hourglass CoREAgent) <<<")
    print("-"*85)
    torch.cuda.reset_peak_memory_stats()

    agent_prop = EmbodiedHourglassCoREAgent(config, device=device_str).to(device)
    hu_prop = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_prop = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_prop = torch.optim.AdamW(agent_prop.get_all_parameters(), lr=1e-3, weight_decay=0.01)
    scaler_prop = torch.amp.GradScaler('cuda', enabled=use_amp)

    warmup_scheduler_prop = torch.optim.lr_scheduler.LambdaLR(opt_prop, lr_lambda=lambda step: min(1.0, (step + 1) / 20.0))

    t0 = time.perf_counter()
    prop_losses = []
    
    for step, batch_tensors in enumerate(batches):
        inp = batch_tensors[:, :-1]
        tgt = batch_tensors[:, 1:]
        opt_prop.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt = agent_prop.forward_sequence(
                inp, tgt, hu_prop, crit_speech, episodic_memory=mem_prop, chunk_size=chunk_size
            )
        scaler_prop.scale(tot_loss).backward()
        scaler_prop.unscale_(opt_prop)
        torch.nn.utils.clip_grad_norm_(agent_prop.get_all_parameters(), max_norm=3.0)
        scaler_prop.step(opt_prop)
        scaler_prop.update()
        warmup_scheduler_prop.step()
        prop_losses.append(s_loss)
        if (step + 1) % 50 == 0:
            print(f"  [Proposed Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | Free Energy: {fe_loss:.4f}")

    prop_duration = time.perf_counter() - t0
    prop_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    prop_final_loss = sum(prop_losses[-30:]) / 30.0
    prop_ppl = math.exp(min(prop_final_loss, 20.0))
    prop_tok_per_sec = (num_eval_steps * b_size * seq_len) / prop_duration

    print(f"\n[Proposed Summary] Final Loss: {prop_final_loss:.4f} | PPL: {prop_ppl:.2f} | Peak VRAM: {prop_vram:.1f} MB | Throughput: {prop_tok_per_sec:.1f} tok/s")

    # KEP Decision
    delta_loss = base_final_loss - prop_final_loss
    vram_reduction_pct = (1.0 - prop_vram / base_vram) * 100.0
    throughput_retention_pct = (prop_tok_per_sec / base_tok_per_sec) * 100.0

    print("\n" + "="*85)
    print(" === [KEP RULE #2 FINAL EMPIRICAL DECISION EVALUATION] ===")
    print("="*85)
    print(f"Baseline Loss      : {base_final_loss:.4f} (PPL: {base_ppl:.2f}) | VRAM: {base_vram:.1f} MB | Speed: {base_tok_per_sec:.1f} tok/s")
    print(f"Proposed Loss      : {prop_final_loss:.4f} (PPL: {prop_ppl:.2f}) | VRAM: {prop_vram:.1f} MB | Speed: {prop_tok_per_sec:.1f} tok/s")
    print(f"Delta Loss         : {delta_loss:+.4f} ({'IMPROVEMENT' if delta_loss >= 0.08 else 'NEUTRAL / REGRESSION'})")
    print(f"VRAM Reduction     : {vram_reduction_pct:+.1f}% ({base_vram:.1f} MB -> {prop_vram:.1f} MB)")
    print(f"Speed Retention    : {throughput_retention_pct:.1f}%")

    if delta_loss >= 0.08 and throughput_retention_pct >= 80.0:
        verdict = "🟢 POSITIVE"
    elif delta_loss < -0.05:
        verdict = "🔴 REJECTED"
    else:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    print(f"VERDICT            : {verdict}")
    print("="*85 + "\n")

    return {
        "verdict": verdict,
        "base_loss": base_final_loss,
        "prop_loss": prop_final_loss,
        "delta_loss": delta_loss,
        "base_vram": base_vram,
        "prop_vram": prop_vram,
        "vram_reduction_pct": vram_reduction_pct,
        "prop_tok_per_sec": prop_tok_per_sec,
        "prop_ppl": prop_ppl
    }


if __name__ == "__main__":
    run_exp_102_benchmark()
