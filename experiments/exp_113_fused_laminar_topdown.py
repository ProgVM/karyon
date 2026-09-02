# experiments/exp_113_fused_laminar_topdown.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-113
Hypothesis:
1. Vector A (Native C++20 Fused Cascaded Laminar Stack):
   Executing the multi-stage cortical pathway (Stage 1 Morpho-Syntactic SSD +
   Entropy Boundary Detection + Precision-Weighted LPER + Stage 2 Semantic SSD)
   within a single native C++20 kernel (`FusedCascadedLaminarStack`) eliminates
   intermediate Python-LibTorch boundary dispatch overhead, maximizing GPU Tensor
   Core residency and token throughput (target: >= 45,000 tok/s).
2. Vector B (Recurrent Top-Down Predictive Temporal Coupling h_s2_prev -> h_s1):
   Replacing the zeroed top-down prior (`zeros_like`) with a causally shifted
   recurrent temporal projection from Stage 2 discourse state (h_s2_shifted) into
   Stage 1 sensory prediction (h_s1_hat = f_td(h_s2_prev)) provides Friston Active
   Inference top-down contextual constraints, lowering Speech Cross-Entropy Loss
   (Delta >= 0.08 nats/byte) on real conversational text (vicgalle/alpaca-gpt4).

Author: Bazilevs (ProgVM member) & Autonomous Lead AI Cyberneticist (KEP v8.0 Master)
Date: September 2026
===============================================================================
"""

import sys
import os
import time
import math
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import karyon_config
import karyon_core
import karyon_agent
import karyon_logger
import karyon_hardware
from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import ByteTokenizer, HomeostaticUnit, FusedCascadedLaminarStack
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

logger.info(f"=== EXP-113 INITIATED ON HARDWARE: {hw.get_telemetry()} ===")

# =============================================================================
# DATASET CURATION (vicgalle/alpaca-gpt4 - KEP Rule #7)
# =============================================================================
def prepare_alpaca_dataset(num_batches=25, batch_size=8, seq_len=512):
    logger.info(f"Loading real conversational corpus (vicgalle/alpaca-gpt4)...")
    dataset = load_dataset("vicgalle/alpaca-gpt4", split="train[:500]")
    tokenizer = ByteTokenizer()
    
    encoded_stream = []
    for sample in dataset:
        inst = sample.get("instruction", "")
        inp = sample.get("input", "")
        out = sample.get("output", "")
        
        full_text = f"User: {inst}\n{inp}\nAssistant: {out}\n<|endoftext|>\n"
        byte_ids = tokenizer.encode(full_text)
        encoded_stream.extend(byte_ids)
        if len(encoded_stream) >= num_batches * batch_size * (seq_len + 1):
            break
            
    required_len = num_batches * batch_size * (seq_len + 1)
    if len(encoded_stream) < required_len:
        encoded_stream = (encoded_stream * ((required_len // len(encoded_stream)) + 2))[:required_len]
    else:
        encoded_stream = encoded_stream[:required_len]
        
    tensor_data = torch.tensor(encoded_stream, dtype=torch.long, device=device)
    batches = tensor_data.view(num_batches, batch_size, seq_len + 1)
    logger.info(f"Curated {num_batches} packed conversational batches (B={batch_size}, S={seq_len}). Total: {num_batches*batch_size*seq_len:,} tokens.")
    return batches

# =============================================================================
# PROPOSED AGENT WITH FUSED C++20 STACK & RECURRENT TOP-DOWN COUPLING
# =============================================================================
class ProposedAgentEXP113(CoREAgent):
    """
    Karyon-CoRE with:
    1. Native C++20 FusedCascadedLaminarStack
    2. Recurrent Top-Down Predictive Error Coupling h_s2_prev -> h_s1
    """
    def __init__(self, config: CoREConfig, device='cpu'):
        super().__init__(config, device=device)
        self.fused_stack = FusedCascadedLaminarStack(
            hidden_dim=self.hidden_dim,
            expand_dim=self.expand_dim,
            num_heads=self.num_heads,
            head_k=self.head_k,
            head_v=self.head_v,
            chunk_size=64,
            device=self.device_str
        )
        # Recurrent top-down projection net
        self.topdown_recurrent_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        ).to(self.device)
        nn.init.zeros_(self.topdown_recurrent_proj[2].weight)
        nn.init.zeros_(self.topdown_recurrent_proj[2].bias)
        
        # State memory for persistent inter-batch discourse
        self.register_buffer("h_s2_last_persistent", torch.zeros(1, 1, self.hidden_dim, device=self.device))

    def forward_multimodal_sequence(self, sensor_seq_dict, target_seq, hu_batch,
                                    criterion_speech, episodic_memory=None, loss_free_energy_weight=0.05,
                                    chunk_size=64, use_checkpointing=False):
        text_seq = sensor_seq_dict.get('text')
        batch_size, seq_len = text_seq.size()
        
        m_s1 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()
        h_prev_fast = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        
        unrolled_inputs = {}
        for name, seq_tensor in sensor_seq_dict.items():
            if name == 'text':
                full_emb = self.pos_embeddings(seq_tensor, start_pos=0, apply_rf=True)
                unrolled_inputs[name] = full_emb.contiguous().view(batch_size * seq_len, -1).float()
            else:
                unrolled_inputs[name] = seq_tensor.contiguous().view(batch_size * seq_len, -1).float()

        h_prev_unrolled = torch.zeros(batch_size * seq_len, self.hidden_dim, device=self.device).float()
        u_t_unrolled = curr_u_t.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous().view(batch_size * seq_len, -1).float()
        
        with torch.amp.autocast(device_type=('cuda' if self.hardware.is_cuda else ('xla' if self.hardware.is_tpu else 'cpu')), enabled=False):
            w_t_unrolled, attn_weights_unrolled, channel_names, epistemic_entropy_unrolled = self.gateway(
                unrolled_inputs, h_prev_unrolled, u_t_unrolled
            )
        
        w_t_seq = w_t_unrolled.view(batch_size, seq_len, self.unified_dim)
        
        with torch.amp.autocast(device_type=('cuda' if self.hardware.is_cuda else ('xla' if self.hardware.is_tpu else 'cpu')), dtype=self.hardware.get_autocast_dtype(), enabled=self.hardware.config.enable_amp and not self.hardware.is_cpu):
            full_h_in = self.in_proj(w_t_seq)
            
            # --- Fused C++20 Cascaded Execution ---
            # Single C++20 call executes Stage 1, Boundary Detector, PW-LPER, and Stage 2
            h_s1, h_s2, m_s1_next, m_s2_next, saliency_gate = self.fused_stack(
                full_h_in, m_s1, m_s2, curr_u_t, text_seq
            )
            
            # --- Recurrent Top-Down Predictive Temporal Coupling ---
            # Construct causal shifted top-down prior from previous Stage 2 discourse state
            h_s2_prev_init = self.h_s2_last_persistent.expand(batch_size, 1, self.hidden_dim)
            h_s2_shifted = torch.cat([h_s2_prev_init, h_s2[:, :-1, :]], dim=1)
            h_s1_predicted = self.topdown_recurrent_proj(h_s2_shifted)
            
            # Active Inference Prediction Error
            e1_topdown = h_s1 - h_s1_predicted
            hpc_reconstruction_loss = F.mse_loss(h_s1, h_s1_predicted)
            
            # Update persistent buffer with detached final state of this batch
            self.h_s2_last_persistent = h_s2[:, -1:, :].detach().mean(dim=0, keepdim=True)
            
            # Hierarchical Volitional Override
            effective_u_t, gamma_override, allostatic_strain = self.will_engine(h_s2, curr_u_t)
            
            # Integrated Cortical Representation with Top-Down Error Feedback
            h_combined = h_s1 + h_s2 + 0.15 * e1_topdown
            
            h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
            h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, effective_u_t)
            
            # Volition-Modulated Motor Text Logits
            volitional_logits_flat = self.volitional_head.compute_volitional_logits(
                h_relaxed, effective_u_t, self.pos_embeddings.byte_embed.weight
            )
            
            targets_flat = target_seq.contiguous().view(-1)
            speech_loss_tensor = criterion_speech(volitional_logits_flat, targets_flat)
            
            w_current_slice = w_t_seq[:, -1, :]
            h_curr_fast = h_combined[:, -1, :]
            w_pred, kl_div, fe, _ = self.world_model(h_prev_fast, h_curr_fast, w_current_slice)
            
            rec_loss = (1.0 - F.cosine_similarity(w_current_slice, w_pred, dim=-1, eps=1e-8)).mean()
            fe_loss_tensor = torch.clamp(kl_div.mean() + rec_loss + 0.10 * hpc_reconstruction_loss, 0.0, 10.0)
            
            total_loss = speech_loss_tensor + loss_free_energy_weight * fe_loss_tensor + 0.02 * commit_loss
            
        speech_loss_scalar = float(speech_loss_tensor.detach().item())
        fe_scalar = float(fe_loss_tensor.detach().item())
        
        return total_loss, speech_loss_scalar, fe_scalar, m_s1_next, m_s2_next, curr_u_t, h_combined

# =============================================================================
# BENCHMARK EVALUATION FUNCTION
# =============================================================================
def run_benchmark():
    config = CoREConfig()
    config.net.hidden_dim = 768
    config.net.unified_dim = 256
    config.net.text_dim = 256
    config.net.num_heads = 12
    config.net.head_k = 64
    config.net.head_v = 128
    
    num_batches = 25
    b_size = 8
    seq_len = 512
    batches = prepare_alpaca_dataset(num_batches=num_batches, batch_size=b_size, seq_len=seq_len)
    
    criterion_speech = nn.CrossEntropyLoss()
    
    # -------------------------------------------------------------------------
    # 1. BASELINE RUN: Standard Python-Orchestrated CoREAgent
    # -------------------------------------------------------------------------
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()
    
    logger.info("Initializing BASELINE Agent (Python multi-call orchestration, zeros top-down)...")
    agent_base = CoREAgent(config, device=device_str).to(device)
    hu_base = HomeostaticUnit(batch_size=b_size, device=device_str)
    opt_base = torch.optim.AdamW(agent_base.parameters(), lr=1.5e-4)
    scaler_base = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    agent_base.train()
    base_losses = []
    base_fes = []
    
    # Warmup
    with get_autocast_ctx():
        _ = agent_base.forward_sequence(batches[0][:, :-1], batches[0][:, 1:], hu_base, criterion_speech, chunk_size=64)
    torch.cuda.synchronize()
    
    t0_base = time.perf_counter()
    for i, batch in enumerate(batches):
        inp, tgt = batch[:, :-1], batch[:, 1:]
        opt_base.zero_grad()
        with get_autocast_ctx():
            tot_loss, s_loss, fe_val, _, _, _, _ = agent_base.forward_sequence(
                inp, tgt, hu_base, criterion_speech, chunk_size=64
            )
        scaler_base.scale(tot_loss).backward()
        scaler_base.unscale_(opt_base)
        torch.nn.utils.clip_grad_norm_(agent_base.parameters(), max_norm=3.0)
        scaler_base.step(opt_base)
        scaler_base.update()
        
        base_losses.append(s_loss)
        base_fes.append(fe_val)
    torch.cuda.synchronize()
    t1_base = time.perf_counter()
    
    dur_base = max(t1_base - t0_base, 1e-5)
    mean_base_loss = float(np.mean(base_losses[-10:]))
    base_ppl = math.exp(min(mean_base_loss, 20.0))
    base_tok_per_sec = (num_batches * b_size * seq_len) / dur_base
    base_vram = hw.get_telemetry().get("allocated_mb", 0.0)
    base_fe = float(np.mean(base_fes[-10:]))
    
    logger.info(f"BASELINE: Loss = {mean_base_loss:.4f}, PPL = {base_ppl:.2f}, Tok/s = {base_tok_per_sec:.1f}, VRAM = {base_vram:.1f} MB, FE = {base_fe:.4f}")
    
    # -------------------------------------------------------------------------
    # 2. PROPOSED RUN (EXP-113): Fused C++20 Stack + Recurrent Top-Down Coupling
    # -------------------------------------------------------------------------
    del agent_base, opt_base, scaler_base
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    logger.info("Initializing PROPOSED Agent (Native C++20 Fused Stack + Recurrent Top-Down Coupling)...")
    agent_prop = ProposedAgentEXP113(config, device=device_str).to(device)
    hu_prop = HomeostaticUnit(batch_size=b_size, device=device_str)
    opt_prop = torch.optim.AdamW(agent_prop.parameters(), lr=1.5e-4)
    scaler_prop = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    agent_prop.train()
    prop_losses = []
    prop_fes = []
    
    # Warmup
    with get_autocast_ctx():
        _ = agent_prop.forward_sequence(batches[0][:, :-1], batches[0][:, 1:], hu_prop, criterion_speech, chunk_size=64)
    torch.cuda.synchronize()
    
    t0_prop = time.perf_counter()
    for i, batch in enumerate(batches):
        inp, tgt = batch[:, :-1], batch[:, 1:]
        opt_prop.zero_grad()
        with get_autocast_ctx():
            tot_loss, s_loss, fe_val, _, _, _, _ = agent_prop.forward_sequence(
                inp, tgt, hu_prop, criterion_speech, chunk_size=64
            )
        scaler_prop.scale(tot_loss).backward()
        scaler_prop.unscale_(opt_prop)
        
        # Deep Diagnostics: Gradient Norm Auditing (KEP Rule #6)
        total_grad_norm = torch.nn.utils.clip_grad_norm_(agent_prop.parameters(), max_norm=3.0)
        scaler_prop.step(opt_prop)
        scaler_prop.update()
        
        prop_losses.append(s_loss)
        prop_fes.append(fe_val)
    torch.cuda.synchronize()
    t1_prop = time.perf_counter()
    
    dur_prop = max(t1_prop - t0_prop, 1e-5)
    mean_prop_loss = float(np.mean(prop_losses[-10:]))
    prop_ppl = math.exp(min(mean_prop_loss, 20.0))
    prop_tok_per_sec = (num_batches * b_size * seq_len) / dur_prop
    prop_vram = hw.get_telemetry().get("allocated_mb", 0.0)
    prop_fe = float(np.mean(prop_fes[-10:]))
    
    delta_loss = mean_base_loss - mean_prop_loss
    speedup_pct = (prop_tok_per_sec / base_tok_per_sec) * 100.0
    
    logger.info(f"PROPOSED: Loss = {mean_prop_loss:.4f}, PPL = {prop_ppl:.2f}, Tok/s = {prop_tok_per_sec:.1f}, VRAM = {prop_vram:.1f} MB, FE = {prop_fe:.4f}")
    logger.info(f"TELEMETRY COMPARISON: Delta Loss = {delta_loss:+.4f}, Speedup = {speedup_pct:.1f}%")
    
    # -------------------------------------------------------------------------
    # 3. KEP RULE #4: DIAGNOSTIC SPEECH SAMPLING (T=0.45, p=0.90)
    # -------------------------------------------------------------------------
    logger.info("Executing Diagnostic Speech Sampling (KEP Rule #4)...")
    agent_prop.eval()
    hu_sample = HomeostaticUnit(batch_size=1, device=device_str)
    m_state = torch.zeros(1, agent_prop.num_heads, agent_prop.head_k, agent_prop.head_v, device=device)
    h_state = torch.zeros(1, agent_prop.hidden_dim, device=device)
    
    test_prompts = ["Hello!", "Energy for Earth"]
    sampling_transcripts = {}
    
    for prompt in test_prompts:
        gen_stream = agent_prop.generate_thought_and_speech(
            prompt, m_state, h_state, hu_sample, None, config, max_generated_tokens=40
        )
        tokens = []
        for item in gen_stream:
            if item.get("status") == "token":
                tokens.append(item.get("text", ""))
        generated_speech = "".join(tokens)
        sampling_transcripts[prompt] = generated_speech
        logger.info(f"Prompt '{prompt}' -> Synthesized: {repr(generated_speech)}")
        
    # -------------------------------------------------------------------------
    # 4. KEP RULE #2 DECISION ENGINE
    # -------------------------------------------------------------------------
    # POSITIVE: Delta Loss >= 0.08 OR (Delta Loss >= 0.02 and Throughput >= 105%)
    if delta_loss >= 0.08 or (delta_loss >= 0.02 and speedup_pct >= 105.0):
        verdict = "🟢 POSITIVE"
    elif abs(delta_loss) <= 0.05 and speedup_pct >= 90.0:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    else:
        verdict = "🔴 REJECTED"
        
    results = {
        "exp_id": "EXP-113",
        "verdict": verdict,
        "base_loss": float(mean_base_loss),
        "prop_loss": float(mean_prop_loss),
        "delta_loss": float(delta_loss),
        "base_ppl": float(base_ppl),
        "prop_ppl": float(prop_ppl),
        "base_tok_per_sec": float(base_tok_per_sec),
        "prop_tok_per_sec": float(prop_tok_per_sec),
        "speedup_pct": float(speedup_pct),
        "base_vram_mb": float(base_vram),
        "prop_vram_mb": float(prop_vram),
        "base_fe": float(base_fe),
        "prop_fe": float(prop_fe),
        "speech_samples": sampling_transcripts
    }
    
    with open("exp_113_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*80)
    print(f"=== EXP-113 VERDICT: {verdict} ===")
    print(f"Base Loss: {mean_base_loss:.4f} (PPL: {base_ppl:.2f}) | Prop Loss: {mean_prop_loss:.4f} (PPL: {prop_ppl:.2f}) | Delta: {delta_loss:+.4f}")
    print(f"Base Speed: {base_tok_per_sec:.1f} tok/s | Prop Speed: {prop_tok_per_sec:.1f} tok/s | Speedup: {speedup_pct:.1f}%")
    print(f"Base Free Energy: {base_fe:.4f} | Prop Free Energy: {prop_fe:.4f}")
    print(f"Base VRAM: {base_vram:.1f} MB | Prop VRAM: {prop_vram:.1f} MB")
    print(f"Speech Samples: {sampling_transcripts}")
    print("="*80 + "\n")
    return results

if __name__ == "__main__":
    run_benchmark()
