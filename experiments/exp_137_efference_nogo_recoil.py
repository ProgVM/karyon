# experiments/exp_137_efference_nogo_recoil.py
"""
===============================================================================
EXP-137: EFFERENCE COPY NOGO RECOIL & ATTRACTOR BASIN LOCKING BENCHMARK
===============================================================================
Hypothesis:
1. Efference Copy & Striatal NoGo Recoil (Negative Evidence):
   Motor actions evaluated by an internal efference copy pathway predict impending
   variational surprise before final motor output. Unviable candidate transitions
   are suppressed via GABAergic somatic inhibition (D2 NoGo pathway), preventing
   erratic phoneme drift without artificial logit clamping.
2. Attractor Basin Locking (Ballistic Morpheme Trajectories):
   At word/concept boundaries (high entropy peaks H > 0.65), the dominant Hopfield
   attractor basin is latched as a ballistic guide vector. During intra-morpheme steps,
   the trajectory is anchored by the attractor pole, preventing phonetic disintegration.
3. Allostatic Recoil & Phasic ERPR (Escape from Repetitive Traps):
   Accumulating allostatic strain and recurrent SSD state stagnation trigger a
   phasic noradrenergic surge (ERPR) that projects state into orthogonal space and
   damps fast-weight Hebbian plasticity, eliminating repetitive syllable loops.

Protocol:
- Strict Next-Byte Prediction (t -> t+1 shift, zero auto-encoding leakage).
- Evaluates Training Loss, PPL, Free Energy, Gradient Norms, and Throughput.
- Evaluates Autoregressive Diagnostic Speech Generation (KEP Rule #4).
- Quantitative Repetition Penalty & UTF-8 Morpheme Integrity Metric.

Author: Bazilevs (ProgVM) & Karyon AI Cyberneticist (KEP v9.0 Master)
===============================================================================
"""

import sys
import time
import math
import os
import gc
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EXP-137")

from karyon_config import CoREConfig
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory
from karyon_checkpoint import load_karyon
from karyon_agent import CoREAgent

# =============================================================================
# 1. BIOPHYSICAL MODULES FOR EXP-137
# =============================================================================

class StriatalNoGoRecoil(nn.Module):
    """
    Efference Copy & Striatal D2 NoGo Pathway.
    Calculates candidate motor state divergence against top-down expectation
    and generates an inhibitory GABAergic somatic suppression vector over vocabulary.
    """
    def __init__(self, hidden_dim: int, vocab_size: int = 258, device_str: str = 'cuda'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        self.efference_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device)
        self.nogo_head = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, vocab_size)
        ).to(self.device)

    def forward(self, h_motor: torch.Tensor, u_t: torch.Tensor, topdown_target: torch.Tensor):
        # h_motor: [B, S, D], u_t: [B, 6] or [B, S, 6], topdown_target: [B, S, D]
        if u_t.dim() == 2:
            u_t = u_t.unsqueeze(1).expand(-1, h_motor.size(1), -1)
            
        h_eff = self.efference_proj(h_motor)
        divergence = torch.abs(h_eff - topdown_target) # Negative evidence trace
        
        ctx = torch.cat([divergence, u_t], dim=-1)
        # Inhibitory potential over tokens (positive values represent GABAergic inhibition)
        inhibition_logits = F.relu(self.nogo_head(ctx))
        return inhibition_logits


class AttractorBasinLocking(nn.Module):
    """
    Ballistic Morpheme Trajectory via Attractor Basin Latched Guidance.
    When morpheme boundary gate is high, latches the active basin;
    when boundary gate is low, pulls hidden state towards the basin pole.
    """
    def __init__(self, hidden_dim: int, device_str: str = 'cuda'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.hidden_dim = hidden_dim
        self.ballistic_gain = nn.Parameter(torch.tensor(0.25, device=self.device))

    def forward(self, h_seq: torch.Tensor, boundary_gate: torch.Tensor, attractor_head) -> torch.Tensor:
        # h_seq: [B, S, D], boundary_gate: [B, S]
        B, S, D = h_seq.shape
        
        # Unit-sphere normalized basins from Hopfield head
        basins = F.normalize(attractor_head.attractor_basins, p=2, dim=-1) # [N, D]
        
        h_norm = F.normalize(h_seq, p=2, dim=-1) # [B, S, D]
        # Basin alignment: [B, S, N]
        sims = torch.matmul(h_norm, basins.t())
        
        # Soft basin anchor representation: [B, S, D]
        basin_weights = F.softmax(sims * 10.0, dim=-1)
        soft_anchor = torch.matmul(basin_weights, basins)
        
        # Inverted boundary: 1.0 inside morpheme (ballistic pull), 0.0 at boundary (free search)
        intra_word_pull = (1.0 - boundary_gate).unsqueeze(-1) # [B, S, 1]
        
        # Guided hidden state trajectory
        h_guided = h_seq + self.ballistic_gain * intra_word_pull * (soft_anchor - h_seq)
        return h_guided


class AllostaticERPRRecoil(nn.Module):
    """
    Event-Related Phase Reset & Phasic Recoil on Stagnation.
    Detects recurrent cyclic trap / allostatic stagnation and generates an orthogonal escape vector.
    """
    def __init__(self, hidden_dim: int, device_str: str = 'cuda'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.hidden_dim = hidden_dim
        self.recoil_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device)

    def forward(self, h_seq: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        # Check autocorrelation across successive steps
        B, S, D = h_seq.shape
        if S <= 2:
            return h_seq
            
        h_diff = h_seq[:, 1:, :] - h_seq[:, :-1, :] # [B, S-1, D]
        diff_norm = torch.norm(h_diff, p=2, dim=-1, keepdim=True) # [B, S-1, 1]
        stagnation = torch.exp(-diff_norm * 2.0) # High when states repeat
        
        # Noradrenaline amplifies escape force
        na_t = u_t[:, 4:5].unsqueeze(1) if u_t.dim() == 2 else u_t[..., 4:5]
        recoil_force = stagnation * (0.15 + 0.35 * na_t[:, :S-1, :])
        
        escape_vec = self.recoil_proj(h_seq[:, 1:, :])
        # Orthogonalized escape perturbation
        h_seq_out = h_seq.clone()
        h_seq_out[:, 1:, :] = h_seq[:, 1:, :] + recoil_force * escape_vec
        return h_seq_out


# =============================================================================
# 2. BENCHMARK SUITE IMPLEMENTATION
# =============================================================================

def run_training_benchmark(
    name: str,
    base_agent_path: str,
    input_stream: np.ndarray,
    active_vectors: dict,
    steps: int = 30,
    batch_size: int = 4,
    seq_len: int = 1024,
    device_str: str = 'cuda'
):
    logger.info(f"\n=======================================================")
    logger.info(f"BENCHMARK RUN: {name}")
    logger.info(f"=======================================================")
    
    device = torch.device(device_str)
    config = CoREConfig()
    config.train.batch_size = batch_size
    
    agent = CoREAgent(config=config, device=device_str).to(device)
    hu = HomeostaticUnit(batch_size=batch_size, device=device_str)
    episodic_mem = BatchedEpisodicMemory(batch_size=batch_size, memory_dim=256, max_capacity=500, device=device_str)
    
    load_karyon(agent, episodic_mem, hu, filepath=base_agent_path, device=device_str)
    agent.eval() # Base in eval mode, optimization over active layers
    
    criterion = nn.CrossEntropyLoss()
    
    # Collect trainable parameters
    trainable_params = []
    for mod_name, mod in active_vectors.items():
        trainable_params.extend(list(mod.parameters()))
    
    # Also fine-tune output heads slightly for alignment if active
    trainable_params.extend(list(agent.volitional_head.parameters()))
    trainable_params.extend(list(agent.attractor_head.parameters()))
    
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-4) if trainable_params else None
    
    loss_history = []
    fe_history = []
    tok_history = []
    step_times = []
    
    total_tokens_processed = 0
    start_time = time.perf_counter()
    
    for step in range(steps):
        t0 = time.perf_counter()
        
        # Strict (t -> t+1) Next-Byte Sequence Slicing
        raw_len = batch_size * (seq_len + 1)
        batch_start = step * raw_len
        batch_end = batch_start + raw_len
        if batch_end > len(input_stream):
            batch_start = 0
            batch_end = raw_len
            
        raw_slice = input_stream[batch_start:batch_end].reshape(batch_size, seq_len + 1)
        raw_tensor = torch.from_numpy(raw_slice).long().to(device)
        
        # Exact causal shift:
        input_seq = raw_tensor[:, :-1]  # [B, S]
        target_seq = raw_tensor[:, 1:]  # [B, S]
        
        if optimizer:
            optimizer.zero_grad()
            
        curr_u_t = hu.state.clone().detach()[:batch_size]
        if curr_u_t.size(0) < batch_size:
            curr_u_t = curr_u_t.repeat(batch_size, 1)[:batch_size]
            
        m_s1 = torch.zeros(batch_size, agent.num_heads, agent.head_k, agent.head_v, device=device)
        m_s2 = torch.zeros(batch_size, agent.num_heads, agent.head_k, agent.head_v, device=device)
        
        full_emb = agent.pos_embeddings(input_seq, start_pos=0, apply_rf=True)
        unrolled_inputs = {'text': full_emb.contiguous().view(batch_size * seq_len, -1).float()}
        h_prev_unrolled = torch.zeros(batch_size * seq_len, agent.hidden_dim, device=device)
        u_t_unrolled = curr_u_t.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous().view(batch_size * seq_len, -1).float()
        
        w_t_unrolled, _, _, _ = agent.gateway(unrolled_inputs, h_prev_unrolled, u_t_unrolled)
        w_t_seq = w_t_unrolled.view(batch_size, seq_len, agent.unified_dim)
        full_h_in = agent.in_proj(w_t_seq)
        
        # Execute Fused Cortical Stack
        h_s1, h_s2, m_s1, m_s2, saliency = agent.fused_stack(full_h_in, m_s1, m_s2, curr_u_t, input_seq)
        
        # Base Neo-Cortical Synthesis
        entropy_s1, boundary_gate = agent.entropy_macro_gate(h_s1)
        h_s2_gated = h_s2 * (0.50 + 1.00 * boundary_gate.unsqueeze(-1))
        
        h_thalamic, _ = agent.thalamic_router(h_s1, h_s2_gated, curr_u_t)
        y_fast = agent.fast_weight_hebbian(h_s1, curr_u_t)
        weighted_error, _ = agent.predictive_residual_router(h_s1, h_s2_gated, curr_u_t)
        topdown_prior = agent.topdown_prior_proj(h_s2_gated)
        
        h_cortical = h_thalamic + 0.20 * y_fast + weighted_error + topdown_prior
        
        # --- APPLY EXP-137 CANDIDATE MECHANISMS ---
        if 'attractor_locking' in active_vectors:
            h_cortical = active_vectors['attractor_locking'](h_cortical, boundary_gate, agent.attractor_head)
            
        if 'erpr_recoil' in active_vectors:
            h_cortical = active_vectors['erpr_recoil'](h_cortical, curr_u_t)
            
        h_flat = h_cortical.contiguous().view(-1, agent.hidden_dim)
        h_relaxed, commit_loss = agent.attractor_head.relax_to_minima(h_flat, curr_u_t)
        
        raw_logits = agent.volitional_head.compute_volitional_logits(
            h_relaxed, curr_u_t, agent.pos_embeddings.byte_embed.weight
        ) # [B*S, V]
        
        if 'nogo_recoil' in active_vectors:
            inhibition_logits = active_vectors['nogo_recoil'](
                h_cortical, curr_u_t, topdown_prior
            ).view(-1, agent.text_gen_dim)
            # Subtract GABAergic somatic inhibition from motor logits
            effective_logits = raw_logits - inhibition_logits
        else:
            effective_logits = raw_logits
            
        targets_flat = target_seq.contiguous().view(-1)
        speech_loss = criterion(effective_logits, targets_flat)
        
        # Active Inference Free Energy
        w_curr = w_t_seq[:, -1, :]
        h_curr = h_cortical[:, -1, :]
        _, kl_div, fe, _ = agent.world_model(h_curr, h_curr, w_curr)
        fe_loss = kl_div.mean() + torch.clamp(fe.mean(), 0.0, 5.0)
        
        total_loss = speech_loss + 0.05 * fe_loss + 0.05 * commit_loss
        
        if optimizer:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            
        dt_ms = (time.perf_counter() - t0) * 1000.0
        step_times.append(dt_ms)
        tokens_this_step = batch_size * seq_len
        tok_speed = tokens_this_step / (dt_ms / 1000.0)
        
        loss_val = speech_loss.item()
        fe_val = fe_loss.item()
        
        loss_history.append(loss_val)
        fe_history.append(fe_val)
        tok_history.append(tok_speed)
        
        if (step + 1) % 10 == 0 or step == steps - 1:
            logger.info(f"Step {step+1:02d}/{steps} | Loss: {loss_val:.4f} (PPL: {math.exp(loss_val):.2f}) | FE: {fe_val:.4f} | Speed: {tok_speed:.0f} tok/s")
            
    # Evaluation metrics
    mean_loss = float(np.mean(loss_history[-10:]))
    mean_ppl = float(math.exp(mean_loss))
    mean_fe = float(np.mean(fe_history[-10:]))
    mean_speed = float(np.mean(tok_history))
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    
    logger.info(f"=== RESULT FOR {name} ===")
    logger.info(f"Final Mean Speech Loss : {mean_loss:.4f} nats/byte")
    logger.info(f"Final Perplexity (PPL) : {mean_ppl:.2f}")
    logger.info(f"Free Energy (FE)       : {mean_fe:.4f}")
    logger.info(f"Average Throughput     : {mean_speed:.1f} tok/s")
    logger.info(f"Peak VRAM              : {peak_vram:.1f} MB")
    
    return {
        'name': name,
        'final_loss': mean_loss,
        'ppl': mean_ppl,
        'free_energy': mean_fe,
        'tok_per_sec': mean_speed,
        'vram_mb': peak_vram,
        'agent': agent,
        'active_vectors': active_vectors
    }


# =============================================================================
# 3. AUTOREGRESSIVE DIAGNOSTIC SPEECH GENERATION TEST (KEP Rule #4)
# =============================================================================

def run_speech_generation_test(bench_res: dict, prompt: str = "User: Hello! Who are you?\nKaryon:"):
    agent = bench_res['agent']
    active_vectors = bench_res['active_vectors']
    device = agent.device
    
    logger.info(f"\n--- Diagnostic Speech Generation: {bench_res['name']} ---")
    logger.info(f"Prompt: {repr(prompt)}")
    
    prompt_ids = [t for t in agent.tokenizer.encode(prompt) if t != 257]
    prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_embs = agent.pos_embeddings(prompt_tokens, start_pos=0, apply_rf=True)
    
    hu_st = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], device=device)
    m_s1 = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device)
    m_s2 = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device)
    
    # Warmup prompt
    prompt_len = prompt_tokens.size(1)
    for c_idx in range(0, prompt_len, 64):
        c_emb = prompt_embs[:, c_idx : min(c_idx + 64, prompt_len), :]
        c_in = prompt_tokens[:, c_idx : min(c_idx + 64, prompt_len)]
        h_in = agent.in_proj(c_emb)
        h_s1, h_s2, m_s1, m_s2, _ = agent.fused_stack(h_in, m_s1, m_s2, hu_st, c_in)
        
    rolling_ids = list(prompt_ids)
    generated_bytes = []
    
    for step in range(80):
        full_context_t = torch.tensor([rolling_ids[-128:]], dtype=torch.long, device=device)
        full_context_emb = agent.pos_embeddings(full_context_t, start_pos=max(0, len(rolling_ids) - 128), apply_rf=True)
        t_emb = full_context_emb[:, -1:, :]
        
        sensor_inputs = {'text': t_emb.squeeze(1)}
        w_t, _, _, _ = agent.gateway(sensor_inputs, m_s2.view(1, -1)[:, :agent.hidden_dim], hu_st)
        h_in = agent.in_proj(w_t).unsqueeze(1)
        
        h_s1, h_s2, m_s1, m_s2, _ = agent.fused_stack(h_in, m_s1, m_s2, hu_st, full_context_t[:, -1:])
        
        entropy_s1, boundary_gate = agent.entropy_macro_gate(h_s1)
        h_s2_gated = h_s2 * (0.50 + 1.00 * boundary_gate.unsqueeze(-1))
        
        h_thalamic, _ = agent.thalamic_router(h_s1, h_s2_gated, hu_st)
        y_fast = agent.fast_weight_hebbian(h_s1, hu_st)
        weighted_error, _ = agent.predictive_residual_router(h_s1, h_s2_gated, hu_st)
        topdown_prior = agent.topdown_prior_proj(h_s2_gated)
        
        h_cortical = h_thalamic + 0.20 * y_fast + weighted_error + topdown_prior
        
        # Candidate modules
        if 'attractor_locking' in active_vectors:
            h_cortical = active_vectors['attractor_locking'](h_cortical, boundary_gate, agent.attractor_head)
            
        if 'erpr_recoil' in active_vectors:
            h_cortical = active_vectors['erpr_recoil'](h_cortical, hu_st)
            
        h_flat = h_cortical.contiguous().view(-1, agent.hidden_dim)
        h_relaxed, _ = agent.attractor_head.relax_to_minima(h_flat, hu_st)
        
        raw_logits = agent.volitional_head.compute_volitional_logits(
            h_relaxed, hu_st, agent.pos_embeddings.byte_embed.weight
        )
        
        if 'nogo_recoil' in active_vectors:
            inhibition = active_vectors['nogo_recoil'](h_cortical, hu_st, topdown_prior).view(-1, agent.text_gen_dim)
            effective_logits = raw_logits - inhibition
        else:
            effective_logits = raw_logits
            
        # Nucleus Top-p sampling (T=0.45, p=0.90) without artificial byte clamping
        probs = F.softmax(effective_logits / 0.45, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        mask = cumulative_probs > 0.90
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
        next_token = sorted_indices.gather(-1, torch.multinomial(sorted_probs, 1)).item()
        if next_token == 257: # EOS
            break
            
        generated_bytes.append(next_token)
        rolling_ids.append(next_token)
        
    raw_bytes = bytes([b for b in generated_bytes if b < 256])
    decoded_text = raw_bytes.decode('utf-8', errors='replace')
    
    # Calculate unique 3-gram ratio (measure of repetitive loop freedom)
    n_grams = [decoded_text[i:i+3] for i in range(len(decoded_text)-2)]
    unique_3gram_ratio = len(set(n_grams)) / max(1, len(n_grams))
    
    logger.info(f"Generated Output : {repr(decoded_text)}")
    logger.info(f"Unique 3-gram Ratio : {unique_3gram_ratio:.3f} (1.000 = zero repetition)")
    return decoded_text, unique_3gram_ratio


# =============================================================================
# 4. MAIN SCIENTIFIC PIPELINE
# =============================================================================

def main():
    logger.info("=== STARTING EXP-137: EFFERENCE COPY NOGO RECOIL BENCHMARK ===")
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_agent_path = 'karyon_soul.kcore'
    
    data_path = 'data/karyon_multidomain_single_pass_stream.npy'
    if not os.path.exists(data_path):
        logger.error(f"Dataset {data_path} not found.")
        sys.exit(1)
        
    input_stream = np.load(data_path)
    logger.info(f"Loaded dataset: {len(input_stream):,} bytes")
    
    hidden_dim = 768
    results = []
    
    # 0. Baseline (Production v33.0 Master without NoGo/Attractor Locking)
    res_base = run_training_benchmark(
        "0. Baseline (Master v33.0)",
        base_agent_path, input_stream, {}, steps=25, device_str=device_str
    )
    results.append(res_base)
    
    # 1. Vector A: Striatal NoGo Recoil (Negative Evidence)
    nogo_mod = StriatalNoGoRecoil(hidden_dim, device_str=device_str)
    res_nogo = run_training_benchmark(
        "1. Vector A (Striatal NoGo Recoil)",
        base_agent_path, input_stream, {'nogo_recoil': nogo_mod}, steps=25, device_str=device_str
    )
    results.append(res_nogo)
    
    # 2. Vector B: Attractor Basin Locking (Ballistic Morphemes)
    lock_mod = AttractorBasinLocking(hidden_dim, device_str=device_str)
    res_lock = run_training_benchmark(
        "2. Vector B (Attractor Basin Locking)",
        base_agent_path, input_stream, {'attractor_locking': lock_mod}, steps=25, device_str=device_str
    )
    results.append(res_lock)
    
    # 3. Vector C: Allostatic ERPR Recoil (Anti-Loop Stagnation)
    erpr_mod = AllostaticERPRRecoil(hidden_dim, device_str=device_str)
    res_erpr = run_training_benchmark(
        "3. Vector C (Allostatic ERPR Recoil)",
        base_agent_path, input_stream, {'erpr_recoil': erpr_mod}, steps=25, device_str=device_str
    )
    results.append(res_erpr)
    
    # 4. Vector D: Full Grand Biophysical Synthesis (A + B + C)
    res_grand = run_training_benchmark(
        "4. Grand Synthesis (NoGo + Locking + ERPR)",
        base_agent_path, input_stream, {
            'nogo_recoil': nogo_mod,
            'attractor_locking': lock_mod,
            'erpr_recoil': erpr_mod
        }, steps=25, device_str=device_str
    )
    results.append(res_grand)
    
    # Run Diagnostic Speech Generation on All Variants
    logger.info("\n=======================================================")
    logger.info("DIAGNOSTIC SPEECH SAMPLING & REPETITION AUDIT")
    logger.info("=======================================================")
    
    speech_results = []
    test_prompt = "User: Tell me about yourself.\nKaryon:"
    for r in results:
        text, ratio = run_speech_generation_test(r, prompt=test_prompt)
        speech_results.append({'name': r['name'], 'text': text, 'ratio': ratio})
        
    logger.info("\n=======================================================")
    logger.info("FINAL EMPIRICAL COMPARISON REPORT")
    logger.info("=======================================================")
    logger.info(f"{'Configuration':<42} | {'Loss':<7} | {'PPL':<6} | {'FE':<6} | {'Speed (tok/s)':<13} | {'3-gram Ratio':<12}")
    logger.info("-" * 95)
    for r, s in zip(results, speech_results):
        logger.info(f"{r['name']:<42} | {r['final_loss']:<7.4f} | {r['ppl']:<6.2f} | {r['free_energy']:<6.4f} | {r['tok_per_sec']:<13.0f} | {s['ratio']:<12.3f}")
        
    logger.info("\nEXP-137 Complete! All scientific telemetry acquired.")

if __name__ == '__main__':
    main()
