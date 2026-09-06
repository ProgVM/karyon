#!/usr/bin/env python3
"""
=====================================================================================
=== EXP-136: QUAD-VECTOR BIOPHYSICAL NEO-CORTICAL SYNTHESIS BENCHMARK            ===
=====================================================================================
Hypothesis:
Evaluating 4 fundamental biophysical neocortical architecture vectors individually and in grand synthesis:
  - Vector 1: Entropy-Driven Hierarchical Concept Gating (BLT-Neuro / Multi-timescale Macro-Pulse).
  - Vector 2: Thalamocortical Dynamic Routing & Active Attention Gate (Pulvinar/TRN Gate).
  - Vector 3: Synaptic Fast-Weight Programmers & Hebbian Plasticity (Parallel Fast Outer-Product Scan).
  - Vector 4: Hierarchical Predictive Residual Coding (Bottom-Up Unpredicted Errors Only).
  - Grand Synthesis: All 4 vectors combined into a single unified neocortical cognitive core.

Quantitative Criteria (KEP Rule #2):
  - Speech Loss Delta >= 0.08 nats improvement or Free Energy reduction.
  - Preserved Throughput >= 80% baseline (~25k+ tok/s).
  - VRAM stability and clean speech sampling (KEP Rule #4).
=====================================================================================
"""

import os
import sys
import time
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure root directory is in path
sys.path.insert(0, os.path.abspath('.'))

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory
from karyon_checkpoint import load_karyon

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("exp_136_synthesis")

# =============================================================================
# MODULE IMPLEMENTATIONS FOR THE 4 VECTORS
# =============================================================================

# --- VECTOR 1: ENTROPY-DRIVEN HIERARCHICAL CONCEPT GATING (BLT-NEURO) ---
class Vector1EntropyMacroGating(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.entropy_head = nn.Linear(hidden_dim, 258) # Byte logits for entropy calculation
        self.macro_boundary_proj = nn.Linear(hidden_dim, 1)

    def forward(self, h_s1: torch.Tensor):
        # h_s1: [B, S, D]
        logits = self.entropy_head(h_s1)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1) # [B, S] in nats
        
        # Macro boundary probability
        boundary_logits = self.macro_boundary_proj(h_s1).squeeze(-1) # [B, S]
        boundary_gate = torch.sigmoid(boundary_logits + 2.0 * (entropy - 1.5)) # [B, S]
        return entropy, boundary_gate

# --- VECTOR 2: THALAMOCORTICAL DYNAMIC ROUTING & ATTENTION GATE ---
class Vector2ThalamocorticalGate(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Input: [u_t (6), mean(h_s1) (D), mean(h_s2) (D)] -> 3 channel weights
        self.routing_mlp = nn.Sequential(
            nn.Linear(6 + hidden_dim * 2, 128),
            nn.SiLU(),
            nn.Linear(128, 3) # Weights for [Stage 1, Stage 2, TopDown Prior]
        )

    def forward(self, h_s1: torch.Tensor, h_s2: torch.Tensor, u_t: torch.Tensor):
        # h_s1, h_s2: [B, S, D], u_t: [B, 6]
        B, S, D = h_s1.shape
        u_t_seq = u_t.unsqueeze(1).expand(B, S, -1) # [B, S, 6]
        ctx = torch.cat([u_t_seq, h_s1, h_s2], dim=-1) # [B, S, 6 + 2D]
        routing_weights = F.softmax(self.routing_mlp(ctx), dim=-1) # [B, S, 3]
        
        w1 = routing_weights[..., 0:1]
        w2 = routing_weights[..., 1:2]
        w3 = routing_weights[..., 2:3]
        
        # Dynamic thalamic combination
        h_thalamic = w1 * h_s1 + w2 * h_s2 + w3 * (h_s1 * h_s2)
        return h_thalamic, routing_weights

# --- VECTOR 3: SYNAPTIC FAST-WEIGHT PROGRAMMERS & HEBBIAN PLASTICITY ---
class Vector3FastWeightHebbian(nn.Module):
    def __init__(self, hidden_dim: int, key_dim: int = 64, value_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        self.k_proj = nn.Linear(hidden_dim, key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, value_dim, bias=False)
        self.q_proj = nn.Linear(hidden_dim, key_dim, bias=False)
        self.out_proj = nn.Linear(value_dim, hidden_dim, bias=False)
        self.lambda_decay = 0.92

    def forward(self, h_seq: torch.Tensor, u_t: torch.Tensor):
        # h_seq: [B, S, D], u_t: [B, 6]
        B, S, D = h_seq.shape
        K = self.k_proj(h_seq) # [B, S, d_k]
        V = self.v_proj(h_seq) # [B, S, d_v]
        Q = self.q_proj(h_seq) # [B, S, d_k]
        
        # Noradrenaline modulation for Hebbian write rate
        na_t = u_t[:, 4:5].unsqueeze(1) # [B, 1, 1]
        eta = 0.10 * (1.0 + 2.0 * na_t) # [B, 1, 1]
        
        # Parallel associative fast weight readout using causal log-decay mask
        idx = torch.arange(S, device=h_seq.device)
        decay_mask = self.lambda_decay ** (idx.unsqueeze(1) - idx.unsqueeze(0))
        causal_decay_mask = torch.tril(decay_mask).unsqueeze(0) # [1, S, S]
        
        attn_sim = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.key_dim) # [B, S, S]
        attn_decayed = attn_sim * causal_decay_mask * eta
        y_fast = torch.bmm(attn_decayed, V) # [B, S, d_v]
        
        return self.out_proj(y_fast)

# --- VECTOR 4: HIERARCHICAL PREDICTIVE RESIDUAL CODING ---
class Vector4PredictiveResidualRouting(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.topdown_pred = nn.Linear(hidden_dim, hidden_dim)
        self.precision_gate = nn.Linear(6, hidden_dim)

    def forward(self, h_s1: torch.Tensor, h_s2: torch.Tensor, u_t: torch.Tensor):
        # Top-down prediction of Stage 1 from Stage 2
        hat_h_s1 = self.topdown_pred(h_s2)
        # Residual prediction error
        error_s1 = h_s1 - hat_h_s1
        # Precision weighting driven by somatic state
        precision = torch.sigmoid(self.precision_gate(u_t)).unsqueeze(1) # [B, 1, D]
        weighted_error = precision * error_s1
        
        # Loss measuring residual surprise
        error_magnitude = torch.mean(weighted_error ** 2)
        return weighted_error, error_magnitude

# =============================================================================
# BENCHMARK RUNNER FUNCTION
# =============================================================================

def benchmark_variant(variant_name: str, base_agent_path: str, hu: HomeostaticUnit, 
                      variant_modules: dict, input_stream: torch.Tensor, 
                      steps: int = 25, device_str: str = 'cuda'):
    
    device = torch.device(device_str)
    config = CoREConfig()
    config.train.batch_size = 8
    
    agent = CoREAgent(config=config, device=device_str).to(device)
    mem = BatchedEpisodicMemory(batch_size=8, memory_dim=256, max_capacity=500, device=device_str)
    load_karyon(agent, mem, hu, filepath=base_agent_path, device=device_str)
    
    # Register additional module parameters if any
    all_params = list(agent.parameters())
    for mod in variant_modules.values():
        all_params.extend(list(mod.parameters()))
    variant_optimizer = torch.optim.AdamW(all_params, lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    losses = []
    free_energies = []
    step_times = []
    
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    
    batch_size = 8
    seq_len = 1024
    
    for step in range(steps):
        t0 = time.time()
        
        # Slice batch stream
        batch_start = step * batch_size * seq_len
        batch_end = batch_start + batch_size * seq_len
        if batch_end > len(input_stream):
            batch_start = 0
            batch_end = batch_size * seq_len
            
        raw_bytes = input_stream[batch_start:batch_end].reshape(batch_size, seq_len)
        input_seq = torch.from_numpy(raw_bytes).long().to(device)
        target_seq = input_seq.clone()
        
        variant_optimizer.zero_grad()
        
        # Forward pass through Agent
        curr_u_t = hu.state.clone().detach()[:batch_size]
        if curr_u_t.size(0) < batch_size:
            curr_u_t = curr_u_t.repeat(batch_size, 1)[:batch_size]
            
        m_s1 = torch.zeros(batch_size, agent.num_heads, agent.head_k, agent.head_v, device=device)
        m_s2 = torch.zeros(batch_size, agent.num_heads, agent.head_k, agent.head_v, device=device)
        
        # Run base unrolled inputs & gateway
        full_emb = agent.pos_embeddings(input_seq, start_pos=0, apply_rf=True)
        unrolled_inputs = {'text': full_emb.contiguous().view(batch_size * seq_len, -1).float()}
        h_prev_unrolled = torch.zeros(batch_size * seq_len, agent.hidden_dim, device=device)
        u_t_unrolled = curr_u_t.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous().view(batch_size * seq_len, -1).float()
        
        w_t_unrolled, _, _, _ = agent.gateway(unrolled_inputs, h_prev_unrolled, u_t_unrolled)
        w_t_seq = w_t_unrolled.view(batch_size, seq_len, agent.unified_dim)
        full_h_in = agent.in_proj(w_t_seq)
        
        # Fused C++ Stage 1 & Stage 2 Execution
        h_s1, h_s2, m_s1, m_s2, saliency = agent.fused_stack(full_h_in, m_s1, m_s2, curr_u_t, input_seq)
        
        aux_loss = torch.tensor(0.0, device=device)
        h_final = h_s1 + h_s2
        
        # --- APPLY SPECIFIC VECTOR MODIFICATIONS ---
        if 'v1' in variant_modules:
            entropy, boundary_gate = variant_modules['v1'](h_s1)
            # Concept-gated scaling of Stage 2
            h_s2 = h_s2 * (0.50 + 1.00 * boundary_gate.unsqueeze(-1))
            h_final = h_s1 + h_s2
            
        if 'v2' in variant_modules:
            h_thalamic, routing_weights = variant_modules['v2'](h_s1, h_s2, curr_u_t)
            h_final = h_thalamic
            
        if 'v3' in variant_modules:
            y_fast = variant_modules['v3'](h_s1, curr_u_t)
            h_final = h_final + 0.20 * y_fast
            
        if 'v4' in variant_modules:
            weighted_error, error_magnitude = variant_modules['v4'](h_s1, h_s2, curr_u_t)
            aux_loss = aux_loss + 0.10 * error_magnitude
            h_final = h_final + weighted_error
            
        # Volitional & Attractor Readout
        h_flat = h_final.contiguous().view(-1, agent.hidden_dim)
        h_relaxed, commit_loss = agent.attractor_head.relax_to_minima(h_flat, curr_u_t)
        
        volitional_logits = agent.volitional_head.compute_volitional_logits(
            h_relaxed, curr_u_t, agent.pos_embeddings.byte_embed.weight
        )
        
        targets_flat = target_seq.contiguous().view(-1)
        speech_loss = criterion(volitional_logits, targets_flat)
        
        # World Model Free Energy
        w_curr = w_t_seq[:, -1, :]
        h_prev_fast = h_final[:, -1, :]
        w_pred, kl_div, fe, _ = agent.world_model(h_prev_fast, h_prev_fast, w_curr)
        fe_loss = kl_div.mean() + torch.clamp(fe.mean(), 0.0, 5.0)
        
        total_loss = speech_loss + 0.05 * fe_loss + 0.05 * commit_loss + aux_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        variant_optimizer.step()
        
        t1 = time.time()
        step_times.append(t1 - t0)
        losses.append(speech_loss.item())
        free_energies.append(fe_loss.item())
        
    total_time = time.time() - start_time
    avg_loss = float(np.mean(losses[-10:]))
    avg_ppl = float(math.exp(avg_loss))
    avg_fe = float(np.mean(free_energies[-10:]))
    tot_tokens = steps * batch_size * seq_len
    throughput = tot_tokens / total_time
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    
    # Perform Speech Generation Test
    prompt = "User: What is Karyon?\nKaryon:"
    m_single = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device)
    h_single = torch.zeros(1, agent.hidden_dim, device=device)
    gen = agent.generate_thought_and_speech(
        prompt, m_state=m_single, h_state=h_single, hu=hu, episodic_memory=None,
        config=agent.config, max_generated_tokens=40, temperature=0.35, top_p=0.90
    )
    sample_text = "".join([ev.get('text', '') for ev in gen if ev.get('status') == 'token'])
    
    return {
        'variant': variant_name,
        'final_loss': avg_loss,
        'ppl': avg_ppl,
        'free_energy': avg_fe,
        'tok_per_sec': throughput,
        'vram_mb': peak_vram,
        'sample_text': repr(sample_text[:60])
    }

# =============================================================================
# MAIN EXECUTION & PROTOCOL SUITE
# =============================================================================

def main():
    logger.info("=== STARTING EXP-136: QUAD-VECTOR BIOPHYSICAL NEO-CORTICAL SYNTHESIS BENCHMARK ===")
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_agent_path = 'karyon_soul.kcore'
    
    data_path = 'data/karyon_multidomain_single_pass_stream.npy'
    if not os.path.exists(data_path):
        logger.error(f"Dataset file {data_path} not found. Please ensure data is ready.")
        sys.exit(1)
        
    input_stream = np.load(data_path)
    logger.info(f"Loaded dataset stream. Total bytes: {len(input_stream):,}")
    
    config = CoREConfig()
    hu = HomeostaticUnit(batch_size=8, device=device_str)
    hidden_dim = config.net.hidden_dim # 768
    
    results = []
    
    # 1. Baseline
    logger.info("\n--- Benchmarking Configuration 0: Baseline (Master v33.0) ---")
    res_base = benchmark_variant("0. Baseline (Master v33.0)", base_agent_path, hu, {}, input_stream, steps=20, device_str=device_str)
    results.append(res_base)
    
    # 2. Vector 1: Entropy Macro Gating
    logger.info("\n--- Benchmarking Configuration 1: Vector 1 (Entropy Concept Gating) ---")
    v1_mod = Vector1EntropyMacroGating(hidden_dim).to(device_str)
    res_v1 = benchmark_variant("1. Vector 1 (Entropy Gating)", base_agent_path, hu, {'v1': v1_mod}, input_stream, steps=20, device_str=device_str)
    results.append(res_v1)
    
    # 3. Vector 2: Thalamocortical Dynamic Gate
    logger.info("\n--- Benchmarking Configuration 2: Vector 2 (Thalamocortical Routing) ---")
    v2_mod = Vector2ThalamocorticalGate(hidden_dim).to(device_str)
    res_v2 = benchmark_variant("2. Vector 2 (Thalamocortical Gate)", base_agent_path, hu, {'v2': v2_mod}, input_stream, steps=20, device_str=device_str)
    results.append(res_v2)
    
    # 4. Vector 3: Fast-Weight Hebbian Plasticity
    logger.info("\n--- Benchmarking Configuration 3: Vector 3 (Fast-Weight Hebbian Plasticity) ---")
    v3_mod = Vector3FastWeightHebbian(hidden_dim).to(device_str)
    res_v3 = benchmark_variant("3. Vector 3 (Fast-Weight Hebbian)", base_agent_path, hu, {'v3': v3_mod}, input_stream, steps=20, device_str=device_str)
    results.append(res_v3)
    
    # 5. Vector 4: Predictive Residual Routing
    logger.info("\n--- Benchmarking Configuration 4: Vector 4 (Predictive Residual Errors) ---")
    v4_mod = Vector4PredictiveResidualRouting(hidden_dim).to(device_str)
    res_v4 = benchmark_variant("4. Vector 4 (Predictive Error Residuals)", base_agent_path, hu, {'v4': v4_mod}, input_stream, steps=20, device_str=device_str)
    results.append(res_v4)
    
    # 6. Grand Synthesis: All 4 Vectors Combined
    logger.info("\n--- Benchmarking Configuration 5: Grand Unified Neocortical Synthesis ---")
    v1_all = Vector1EntropyMacroGating(hidden_dim).to(device_str)
    v2_all = Vector2ThalamocorticalGate(hidden_dim).to(device_str)
    v3_all = Vector3FastWeightHebbian(hidden_dim).to(device_str)
    v4_all = Vector4PredictiveResidualRouting(hidden_dim).to(device_str)
    v_all = {'v1': v1_all, 'v2': v2_all, 'v3': v3_all, 'v4': v4_all}
    res_grand = benchmark_variant("5. Grand Synthesis (All 4 Vectors)", base_agent_path, hu, v_all, input_stream, steps=20, device_str=device_str)
    results.append(res_grand)
    
    # Display Empirical Summary Table
    print("\n" + "=" * 115)
    print(f"{'VARIANT':<38} | {'LOSS':<7} | {'PPL':<7} | {'FREE ENG':<8} | {'TOK/SEC':<8} | {'VRAM(MB)':<8} | {'SAMPLE GENERATION'}")
    print("=" * 115)
    
    base_loss = res_base['final_loss']
    
    for r in results:
        delta_str = f"({r['final_loss'] - base_loss:+.4f})" if r['variant'] != res_base['variant'] else "(BASE)"
        print(f"{r['variant']:<38} | {r['final_loss']:.4f} | {r['ppl']:.2f}  | {r['free_energy']:.4f}   | {r['tok_per_sec']:<8.0f} | {r['vram_mb']:<8.1f} | {r['sample_text']}")
        
    print("=" * 115)

if __name__ == "__main__":
    main()
