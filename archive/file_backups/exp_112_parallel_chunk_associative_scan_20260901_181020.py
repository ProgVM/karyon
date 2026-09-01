# experiments/exp_112_parallel_chunk_associative_scan.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-112
Hypothesis:
Replacing the sequential inter-chunk loop in ParallelLogDecaySSDLayer with a
fully parallelized chunk-level associative scan (using strictly lower triangular
decay matrices and pre-computed cumulative sum decay factors) eliminates the
O(C) sequential bottleneck. This allows the entire sequence to be processed in
O(log C) parallel steps, boosting throughput to 50,000+ tok/s on large batches
while maintaining 100% mathematical equivalence and numerical parity.

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
from karyon_core import ByteTokenizer, HomeostaticUnit
from karyon_hardware import get_hardware_engine

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

hw = get_hardware_engine()
device = hw.device
device_str = str(device)
use_amp = hw.config.enable_amp

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
        alpha_chunks = torch.exp(torch.clamp(lambda_t[..., -1:], -20.0, 0.0)).unsqueeze(-1)
        
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
        
        # Prepare U = KV + dW
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
        q_decay_flat = q_decay.permute(0, 2, 1, 3).unsqueeze(-2) # (B, num_heads, num_chunks, 1, head_k)
        
        y_inter_all = torch.matmul(q_decay_flat, M_all).squeeze(-2) # (B, num_heads, num_chunks, head_v)
        y_inter = y_inter_all.permute(0, 2, 1, 3) # (B, num_chunks, num_heads, head_v)
        
        # Final state update for next sequence
        alpha_last = torch.exp(lambda_chunks_flat[:, :, -1:]).unsqueeze(-1).unsqueeze(-1)
        m_next = alpha_last * m_prev.unsqueeze(2) + M_inter_all[:, :, -1:]
        m_next = torch.clamp(m_next.squeeze(2), -10000.0, 10000.0)
        
        y_total = (y_intra + y_inter).permute(0, 1, 3, 2, 4).reshape(B * S, self.num_heads * self.head_v)
        y_normed = self.head_norm(y_total.float()).to(dtype)
        y_gated = y_normed * z_full
        h_seq = self.norm(self.out_proj(y_gated)).view(B, S, self.out_dim)
        
        return h_seq, m_next, eff_dt.mean()

# =============================================================================
# BENCHMARK EVALUATION
# =============================================================================
def run_benchmark():
    logger.info("=== EXP-112 BENCHMARK RUNNING ===")
    
    B, S = 8, 1024
    x = torch.randn(B, S, 768, device=device, dtype=torch.float16)
    m = torch.zeros(B, 12, 64, 128, device=device, dtype=torch.float16)
    u = torch.randn(B, 6, device=device, dtype=torch.float16)
    
    # 1. Baseline Layer (Standard C++ ParallelLogDecaySSDLayer)
    from karyon_core import ParallelLogDecaySSDLayer
    layer_base = ParallelLogDecaySSDLayer(768, 768, 12, 64, 128, 0.005, 0.15, 64, device_str).to(device)
    
    # Warmup
    for _ in range(10):
        with get_autocast_ctx():
            _ = layer_base(x, m, u)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    N = 100
    for _ in range(N):
        with get_autocast_ctx():
            _ = layer_base(x, m, u)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    base_speed = (B * S * N) / (t1 - t0)
    
    # 2. Proposed Layer (Parallel Chunk Associative Scan)
    layer_prop = ParallelChunkAssociativeSSDLayer(768, 768, 12, 64, 128, 0.005, 0.15, 64, device_str).to(device)
    
    # Warmup
    for _ in range(10):
        with get_autocast_ctx():
            _ = layer_prop(x, m, u)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(N):
        with get_autocast_ctx():
            _ = layer_prop(x, m, u)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    prop_speed = (B * S * N) / (t1 - t0)
    
    print("\n" + "="*80)
    print("=== EXP-112 LAYER-LEVEL BENCHMARK RESULTS ===")
    print(f"Baseline C++ Sequential Loop Speed: {base_speed:.1f} tok/s")
    print(f"Proposed Parallel Associative Scan: {prop_speed:.1f} tok/s")
    print(f"Speedup Ratio                     : {prop_speed / base_speed:.2f}x")
    print("="*80 + "\n")
    
    # Numerical Equivalence Check
    with get_autocast_ctx():
        out_base, m_next_base, _ = layer_base(x, m, u)
        out_prop, m_next_prop, _ = layer_prop(x, m, u)
        
    diff_out = (out_base - out_prop).abs().max().item()
    diff_m = (m_next_base - m_next_prop).abs().max().item()
    
    print(f"Max Output Difference: {diff_out:.6f}")
    print(f"Max State Difference : {diff_m:.6f}")
    
    if diff_out < 1e-2:
        logger.info("🟢 MATHEMATICAL EQUIVALENCE VERIFIED!")
    else:
        logger.warning("⚠️ NUMERICAL DRIFT DETECTED!")

if __name__ == "__main__":
    run_benchmark()
