# experiments/exp_112_parallel_chunk_associative_scan.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-112
Hypothesis:
1. Vector A (Fused Parallel Chunk Associative Scan): Replacing the sequential C++
   inter-chunk loop with a fully parallelized chunk-level associative scan matrix
   multiplication (using strictly lower triangular decay matrices and pre-computed
   cumulative sum decay factors) eliminates the O(C) sequential bottleneck.
2. Vector B (Full Batch Capacity Pre-Training Speed): Evaluating on large packed
   batches (B=16, S=1024 = 16,384 tokens/step) proves pure end-to-end training throughput
   exceeding 30,000+ tok/s while preserving full Active Inference and Speech Cross-Entropy Loss.

Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import os
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import karyon_config, karyon_core, karyon_agent, karyon_logger, karyon_hardware
from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import ByteTokenizer, HomeostaticUnit, CausalConvSwiGLUBlock
from karyon_hardware import get_hardware_engine

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

hw = get_hardware_engine()
device = hw.device
device_str = str(device)
use_amp = hw.config.enable_amp

def get_autocast_ctx():
    return torch.amp.autocast(
        device_type=('cuda' if hw.is_cuda else ('xla' if hw.is_tpu else 'cpu')),
        dtype=hw.get_autocast_dtype(),
        enabled=use_amp and not hw.is_cpu
    )

logger.info(f"=== EXP-112 INITIATED ON HARDWARE: {hw.get_telemetry()} ===")

# Multi-Domain Code & Multilingual Text Corpus
RICH_CORPUS_TEXT = """Смысл жизни: Есть неживое, а есть живое. Неживое не живёт, а живое живёт и продолжает жизнь. Если существо не оставит потомство и при этом умрёт, то жизни не будет. Останется ноль, а как известно ноль имеет нулевой смысл; если не будет жизни, то не будет и смысла жизни. Чем старше становится человек, тем всё меньше и меньше его радует жизнь. Ребёнок счастлив, потому что он ничего не знает и ему интересно познавать неизвестное. Если неизвестное заканчивается - абсолютно всё становится предсказуемым, и эмоция "удивление" перестаёт существовать.
The Meaning of Life: There is the non-living and there is the living. The non-living does not live, while the living lives and continues life. If an organism does not leave offspring and dies, there will be no life. Only zero will remain, and as is known, zero has zero meaning; if there is no life, there is no meaning to life. A child is happy because they know nothing and are interested in discovering the unknown. If the unknown ends—everything becomes absolutely predictable, and surprise ceases to exist.
#include <torch/extension.h>
#include <vector>
#include <cmath>
struct ParallelSSDCore { int64_t hidden_dim; int64_t num_heads; ParallelSSDCore(int64_t hidden_dim, int64_t num_heads) : hidden_dim(hidden_dim), num_heads(num_heads) {} torch::Tensor forward(torch::Tensor x, torch::Tensor m_prev) { auto q = torch::silu(x); auto k = torch::tanh(x); auto v = x; auto decay = torch::exp(-0.05f * torch::arange(x.size(1), x.options())); return torch::matmul(q, k.transpose(-1, -2)) * decay + m_prev; } };
def execute_autonomous_self_learning(agent, hu, memory, optimizer, num_steps=5): agent.train() for step in range(num_steps): optimizer.zero_grad() seed_tokens = torch.randint(32, 126, (1, 128), dtype=torch.long, device=agent.device) logits, state, fe = agent.forward_sequence(seed_tokens) loss = F.cross_entropy(logits[:, :-1].reshape(-1, 258), seed_tokens[:, 1:].reshape(-1)) loss.backward() optimizer.step() return fe"""

def prepare_packed_batches(num_batches=20, batch_size=16, seq_len=1024):
    tokenizer = ByteTokenizer()
    base_ids = tokenizer.encode(RICH_CORPUS_TEXT)
    required_len = num_batches * batch_size * (seq_len + 1)
    repeat_factor = (required_len // len(base_ids)) + 2
    full_ids = (base_ids * repeat_factor)[:required_len]
    
    tensor_data = torch.tensor(full_ids, dtype=torch.long, device=device)
    batches = tensor_data.view(num_batches, batch_size, seq_len + 1)
    return batches

# =============================================================================
# PROPOSED MODULE: Parallel Chunk Associative Scan SSD Layer
# =============================================================================
class ParallelChunkAssociativeSSDLayer(nn.Module):
    def __init__(self, in_dim=768, out_dim=768, num_heads=12, head_k=64, head_v=128, min_beta=0.0005, max_beta=0.08, chunk_size=64, device_str='cuda'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_k = head_k
        self.head_v = head_v
        self.chunk_size = chunk_size
        self.inv_sqrt_k = 1.0 / math.sqrt(head_k)
        
        dev = torch.device(device_str)
        self.q_proj = nn.Linear(in_dim, num_heads * head_k, device=dev)
        self.k_proj = nn.Linear(in_dim, num_heads * head_k, device=dev)
        self.v_proj = nn.Linear(in_dim, num_heads * head_v, device=dev)
        self.z_proj = nn.Linear(in_dim, num_heads * head_v, device=dev)
        self.delta_proj = nn.Linear(in_dim, num_heads, device=dev)
        
        betas = torch.exp(torch.linspace(math.log(max_beta), math.log(min_beta), num_heads, device=dev))
        alphas = 1.0 - betas
        logit_init = torch.log(alphas / (1.0 - alphas)).view(1, 1, num_heads, 1)
        self.decay_logits = nn.Parameter(logit_init)
        
        self.head_norm = nn.GroupNorm(num_heads, num_heads * head_v, device=dev)
        self.out_proj = nn.Linear(num_heads * head_v, out_dim, device=dev)
        self.norm = nn.LayerNorm(out_dim, device=dev)
        
        steps = torch.arange(0, head_k, 2, device=dev, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (steps / head_k))
        self.register_buffer("inv_freq", inv_freq)
        
    def get_rope_cos_sin(self, Q, device, dtype):
        t = torch.arange(Q, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().view(1, 1, 1, Q, self.head_k).to(dtype), emb.sin().view(1, 1, 1, Q, self.head_k).to(dtype)
        
    def forward(self, x_seq, m_prev, u_t, saliency_gate=None, dt=1.0):
        dtype = x_seq.dtype
        device = x_seq.device
        B, S, _ = x_seq.shape
        Q = self.chunk_size if S >= self.chunk_size else S
        num_chunks = S // Q
        
        curiosity = u_t[:, 0].view(B, 1, 1, 1, 1)
        na = u_t[:, 4].view(B, 1, 1, 1, 1)
        da = u_t[:, 5].view(B, 1, 1, 1, 1)
        eff_dt = torch.clamp(dt * (1.0 - 0.4 * na + 0.4 * da), 0.30, 2.00)
        
        q_full = (self.q_proj(x_seq).view(B, num_chunks, Q, self.num_heads, self.head_k).transpose(2, 3)) * self.inv_sqrt_k
        k_full = self.k_proj(x_seq).view(B, num_chunks, Q, self.num_heads, self.head_k).transpose(2, 3)
        v_full = self.v_proj(x_seq).view(B, num_chunks, Q, self.num_heads, self.head_v).transpose(2, 3)
        z_full = F.silu(self.z_proj(x_seq)).view(B * S, self.num_heads * self.head_v)
        
        cos, sin = self.get_rope_cos_sin(Q, device, dtype)
        q_rot = torch.cat([-q_full[..., self.head_k//2:], q_full[..., :self.head_k//2]], dim=-1)
        q_full = q_full * cos + q_rot * sin
        k_rot = torch.cat([-k_full[..., self.head_k//2:], k_full[..., :self.head_k//2]], dim=-1)
        k_full = k_full * cos + k_rot * sin
        
        delta_full = F.softplus(self.delta_proj(x_seq)).view(B, num_chunks, Q, self.num_heads).permute(0, 1, 3, 2)
        base_alpha = torch.sigmoid(self.decay_logits.view(1, 1, self.num_heads, 1))
        
        exponent = torch.clamp(delta_full * eff_dt.squeeze(-1), 0.1, 10.0)
        log_alpha = exponent * torch.log(base_alpha)
        
        if saliency_gate is not None and saliency_gate.numel() > 0:
            sal_chunk = saliency_gate.view(B, num_chunks, 1, Q)
            log_alpha = log_alpha + torch.log(1.0 - 0.80 * sal_chunk)
            
        log_alpha = torch.clamp(log_alpha, math.log(1e-4), math.log(0.9999))
        alpha = torch.exp(log_alpha)
        beta = 1.0 - alpha
        
        lambda_t = torch.cumsum(log_alpha, dim=-1)
        log_decay_matrix = lambda_t.unsqueeze(-1) - lambda_t.unsqueeze(-2)
        decay_matrix = torch.exp(torch.clamp(log_decay_matrix, -20.0, 0.0))
        
        causal_mask = torch.tril(torch.ones((Q, Q), device=device, dtype=dtype)).view(1, 1, 1, Q, Q)
        decay_weights = decay_matrix * causal_mask
        
        s_matrix = torch.matmul(q_full, k_full.transpose(-1, -2)) * decay_weights
        y_intra = torch.matmul(s_matrix, v_full)
        
        decay_to_start = torch.exp(torch.clamp(lambda_t, -20.0, 0.0)).unsqueeze(-1)
        lambda_end = lambda_t[..., -1:].unsqueeze(-1)
        decay_to_end = torch.exp(torch.clamp(lambda_end - lambda_t.unsqueeze(-1), -20.0, 0.0))
        
        k_decayed = k_full * decay_to_end * beta.unsqueeze(-1)
        kv_chunk_updates = torch.matmul(k_decayed.transpose(-1, -2), v_full)
        
        # =============================================================================
        # PARALLEL CHUNK ASSOCIATIVE SCAN (EXP-112)
        # =============================================================================
        log_alpha_chunks = torch.clamp(lambda_t[..., -1:], -20.0, 0.0).unsqueeze(-1) # (B, num_chunks, num_heads, 1, 1)
        lambda_chunks = torch.cumsum(log_alpha_chunks, dim=1) # (B, num_chunks, num_heads, 1, 1)
        
        lambda_chunks_flat = lambda_chunks.squeeze(-1).squeeze(-1).permute(0, 2, 1) # (B, num_heads, num_chunks)
        log_decay_matrix_chunks = lambda_chunks_flat.unsqueeze(-1) - lambda_chunks_flat.unsqueeze(-2)
        decay_matrix_chunks = torch.exp(torch.clamp(log_decay_matrix_chunks, -20.0, 0.0))
        
        causal_mask_chunks = torch.tril(torch.ones((num_chunks, num_chunks), device=device, dtype=dtype), diagonal=-1).view(1, 1, num_chunks, num_chunks)
        decay_weights_chunks = decay_matrix_chunks * causal_mask_chunks
        
        sigma_somatic = 1e-3 * (0.8 * curiosity.squeeze(1).float() + 0.4 * na.squeeze(1).float() + 0.1)
        dW_scale = torch.sqrt(eff_dt.squeeze(1).float()) * sigma_somatic
        dW_all = torch.randn((num_chunks, B, self.num_heads, self.head_k, self.head_v), device=device, dtype=torch.float32)
        dW_all_scaled = (dW_all * dW_scale.view(1, B, 1, 1, 1)).permute(1, 2, 0, 3, 4) # (B, num_heads, num_chunks, head_k, head_v)
        
        kv_flat = kv_chunk_updates.permute(0, 2, 1, 3, 4) # (B, num_heads, num_chunks, head_k, head_v)
        U = (kv_flat.float() + dW_all_scaled).to(dtype)
        
        # Batched Parallel Scan Matrix Multiplication
        U_reshaped = U.reshape(B, self.num_heads, num_chunks, self.head_k * self.head_v)
        M_inter_all = torch.matmul(decay_weights_chunks, U_reshaped).reshape(B, self.num_heads, num_chunks, self.head_k, self.head_v)
        
        # Initial State Decay
        decay_initial = torch.cat([
            torch.ones((B, self.num_heads, 1), device=device, dtype=dtype),
            torch.exp(lambda_chunks_flat[:, :, :-1])
        ], dim=2).unsqueeze(-1).unsqueeze(-1) # (B, num_heads, num_chunks, 1, 1)
        
        M_initial_decayed = decay_initial * m_prev.unsqueeze(2)
        M_all = M_inter_all + M_initial_decayed
        
        # Compute y_inter in parallel
        q_decay = q_full * decay_to_start
        q_decay_flat = q_decay.permute(0, 2, 1, 3, 4) # (B, num_heads, num_chunks, Q, head_k)
        
        y_inter_all = torch.matmul(q_decay_flat, M_all) # (B, num_heads, num_chunks, Q, head_v)
        y_inter = y_inter_all.permute(0, 2, 1, 3, 4) # (B, num_chunks, num_heads, Q, head_v)
        
        # Final state update for next sequence
        alpha_last = torch.exp(lambda_chunks_flat[:, :, -1:]).unsqueeze(-1).unsqueeze(-1)
        m_next = alpha_last * m_prev.unsqueeze(2) + M_inter_all[:, :, -1:]
        m_next = torch.clamp(m_next.squeeze(2), -10000.0, 10000.0)
        
        y_total = (y_intra + y_inter).permute(0, 1, 3, 2, 4).reshape(B * S, self.num_heads * self.head_v)
        y_normed = self.head_norm(y_total.float()).to(dtype)
        y_gated = y_normed * z_full
        h_seq = self.norm(self.out_proj(y_gated)).view(B, S, self.out_dim)
        
        return h_seq, m_next, eff_dt.mean()

class CorticalStagePy(nn.Module):
    def __init__(self, hidden_dim=768, expand_dim=3072, num_heads=12, head_k=64, head_v=128, min_beta=0.005, max_beta=0.15, swiglu_kernel_size=3, chunk_size=64, device_str='cuda'):
        super().__init__()
        dev = torch.device(device_str)
        self.pre_norm_ssd = nn.LayerNorm(hidden_dim, device=dev)
        self.ssd = ParallelChunkAssociativeSSDLayer(
            in_dim=hidden_dim, out_dim=hidden_dim, num_heads=num_heads, head_k=head_k, head_v=head_v,
            min_beta=min_beta, max_beta=max_beta, chunk_size=chunk_size, device_str=device_str
        )
        self.pre_norm_swiglu = nn.LayerNorm(hidden_dim, device=dev)
        self.swiglu = CausalConvSwiGLUBlock(hidden_dim, expand_dim, swiglu_kernel_size, device_str).to(dev)

    def forward(self, x, m_prev, u_t, saliency_gate=None, dt=1.0):
        norm_x = self.pre_norm_ssd(x)
        h_ssd, m_next, eff_dt = self.ssd(norm_x, m_prev, u_t, saliency_gate, dt)
        x_res1 = x + h_ssd.view_as(x)
        norm_res1 = self.pre_norm_swiglu(x_res1)
        x_out = self.swiglu.forward(norm_res1)
        return x_out, m_next, eff_dt

# =============================================================================
# BENCHMARK EVALUATION FUNCTION
# =============================================================================
def run_benchmark():
    config = CoREConfig()
    config.net.hidden_dim = 768
    config.net.unified_dim = 256
    config.net.text_dim = 256
    config.net.num_heads = 12
    
    num_batches = 20
    b_size = 16
    seq_len = 1024
    batches = prepare_packed_batches(num_batches=num_batches, batch_size=b_size, seq_len=seq_len)
    
    hu = HomeostaticUnit(batch_size=b_size, device=device_str)
    criterion_speech = nn.CrossEntropyLoss()
    
    # 1. Baseline Model Execution (Standard C++ ParallelLogDecaySSDLayer)
    logger.info(f"Evaluating Standard C++ SSD Baseline (Workload: B={b_size}, S={seq_len}, Steps={num_batches})...")
    agent_base = CoREAgent(config, device=device_str).to(device)
    opt_base = torch.optim.AdamW(agent_base.parameters(), lr=1e-4)
    scaler_base = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    agent_base.train()
    base_losses = []
    
    # Warmup
    warmup_batch = batches[0]
    with get_autocast_ctx():
        _ = agent_base.forward_sequence(warmup_batch[:, :-1], warmup_batch[:, 1:], hu, criterion_speech, chunk_size=64)
    torch.cuda.synchronize()
    
    t0_base = time.perf_counter()
    for batch in batches:
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]
        
        opt_base.zero_grad()
        with get_autocast_ctx():
            total_loss, speech_loss, fe_val, _, _, _, _ = agent_base.forward_sequence(
                input_seq, target_seq, hu, criterion_speech, chunk_size=64
            )
        scaler_base.scale(total_loss).backward()
        scaler_base.unscale_(opt_base)
        torch.nn.utils.clip_grad_norm_(agent_base.parameters(), max_norm=3.0)
        scaler_base.step(opt_base)
        scaler_base.update()
        base_losses.append(speech_loss)
        
    torch.cuda.synchronize()
    t1_base = time.perf_counter()
    base_duration = max(t1_base - t0_base, 1e-5)
    base_loss_mean = float(np.mean(base_losses))
    base_ppl = math.exp(min(base_loss_mean, 20.0))
    total_tokens = num_batches * b_size * seq_len
    base_tok_per_sec = total_tokens / base_duration
    base_vram = hw.get_telemetry().get("allocated_mb", 0.0)
    
    logger.info(f"Baseline Telemetry: Loss = {base_loss_mean:.4f}, PPL = {base_ppl:.2f}, Speed = {base_tok_per_sec:.1f} tok/s, VRAM = {base_vram:.1f} MB")
    
    # 2. Proposed Model Execution (Parallel Chunk Associative Scan)
    logger.info(f"Evaluating Proposed Fused Parallel Chunk Scan (Workload: B={b_size}, S={seq_len}, Steps={num_batches})...")
    agent_prop = CoREAgent(config, device=device_str).to(device)
    
    agent_prop.stage1 = CorticalStagePy(
        hidden_dim=768, expand_dim=3072, num_heads=12, head_k=64, head_v=128,
        min_beta=0.005, max_beta=0.15, swiglu_kernel_size=3, chunk_size=64, device_str=device_str
    ).to(device)
    agent_prop.stage2 = CorticalStagePy(
        hidden_dim=768, expand_dim=3072, num_heads=12, head_k=64, head_v=128,
        min_beta=0.0001, max_beta=0.05, swiglu_kernel_size=3, chunk_size=64, device_str=device_str
    ).to(device)
    
    opt_prop = torch.optim.AdamW(agent_prop.parameters(), lr=1e-4)
    scaler_prop = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    agent_prop.train()
    prop_losses = []
    
    # Warmup
    with get_autocast_ctx():
        _ = agent_prop.forward_sequence(warmup_batch[:, :-1], warmup_batch[:, 1:], hu, criterion_speech, chunk_size=64)
    torch.cuda.synchronize()
    
    t0_prop = time.perf_counter()
    for batch in batches:
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]
        
        opt_prop.zero_grad()
        with get_autocast_ctx():
            total_loss, speech_loss, fe_val, _, _, _, _ = agent_prop.forward_sequence(
                input_seq, target_seq, hu, criterion_speech, chunk_size=64
            )
        scaler_prop.scale(total_loss).backward()
        scaler_prop.unscale_(opt_prop)
        torch.nn.utils.clip_grad_norm_(agent_prop.parameters(), max_norm=3.0)
        scaler_prop.step(opt_prop)
        scaler_prop.update()
        prop_losses.append(speech_loss)
        
    torch.cuda.synchronize()
    t1_prop = time.perf_counter()
    prop_duration = max(t1_prop - t0_prop, 1e-5)
    prop_loss_mean = float(np.mean(prop_losses))
    prop_ppl = math.exp(min(prop_loss_mean, 20.0))
    prop_tok_per_sec = total_tokens / prop_duration
    prop_vram = hw.get_telemetry().get("allocated_mb", 0.0)
    
    logger.info(f"Proposed Telemetry: Loss = {prop_loss_mean:.4f}, PPL = {prop_ppl:.2f}, Speed = {prop_tok_per_sec:.1f} tok/s, VRAM = {prop_vram:.1f} MB")
    
    delta_loss = base_loss_mean - prop_loss_mean
    speed_retention_pct = (prop_tok_per_sec / max(base_tok_per_sec, 1e-5)) * 100.0
    
    metrics = {
        "base_loss": float(base_loss_mean),
        "base_ppl": float(base_ppl),
        "base_tok_per_sec": float(base_tok_per_sec),
        "base_vram_mb": float(base_vram),
        "prop_loss": float(prop_loss_mean),
        "prop_ppl": float(prop_ppl),
        "prop_tok_per_sec": float(prop_tok_per_sec),
        "prop_vram_mb": float(prop_vram),
        "delta_loss": float(delta_loss),
        "speed_retention_pct": float(speed_retention_pct)
    }
    with open("exp_112_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("\n" + "="*80)
    print("=== EXP-112 FULL-AGENT PRE-TRAINING BENCHMARK TELEMETRY ===")
    print(f"Baseline (Standard C++ SSD)        : Loss = {base_loss_mean:.4f} | PPL = {base_ppl:.2f} | Speed = {base_tok_per_sec:.1f} tok/s | VRAM = {base_vram:.1f} MB")
    print(f"Proposed (Fused Parallel Chunk Scan): Loss = {prop_loss_mean:.4f} | PPL = {prop_ppl:.2f} | Speed = {prop_tok_per_sec:.1f} tok/s | VRAM = {prop_vram:.1f} MB")
    print(f"Delta Loss                          : {delta_loss:+.4f} nats/byte")
    print(f"Speed Gain                          : {speed_retention_pct:.1f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_benchmark()
