# experiments/exp_103_hierarchical_predictive_macro_resonance.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-103
Hypothesis (Kiebel-Friston Hierarchical Predictive Macro-Micro Resonance):
1. In the 2-Stage Cortical Stack, higher cortical timescales (Macro-Scale, Q=64 bytes)
   provide a continuous top-down contextual trajectory prediction h_hat_micro(t) to lower
   morpho-syntactic layers.
2. Lower layers compute precision-weighted prediction errors e_micro(t) = pi_t * (h_micro(t) - h_hat_micro(t)).
3. Only the aggregated chunk error E_chunk = sum(e_micro(t)) updates the Macro State M_macro.
   When text is familiar/predictable (e_micro -> 0), the Macro memory state is protected
   from redundant byte clutter (preventing Semantic Bleed and pseudo-morphemic drift).
4. When high surprise occurs (topic transitions, questions), E_chunk updates M_macro,
   accelerating steady-state convergence and significantly reducing speech loss.
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
if 'cuda' in device_str:
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    gc.collect()

device = torch.device(device_str)
use_amp = ('cuda' in device_str)


def prepare_packed_stream(num_batches: int = 150, batch_size: int = 16, seq_len: int = 1024):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-103 (S={seq_len}, Steps={num_batches})...")
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


# =============================================================================
# PROPOSED ARCHITECTURE: HIERARCHICAL PREDICTIVE MACRO RESONANCE (HPMR-SSM)
# =============================================================================

class HierarchicalPredictiveResonanceAgent(CoREAgent):
    """
    Karyon-CoRE with Kiebel-Friston Hierarchical Predictive Macro-Micro Resonance:
    - Stage 1 (Micro-Scale): processes raw bytes at intra-chunk level (Q=64).
    - Top-Down Generator: predicts Micro-Scale trajectory from Macro-State h_macro.
    - Error Aggregation: accumulates precision-weighted error e_micro(t) = pi * (h_s1 - h_s1_hat).
    - Stage 2 (Macro-Scale SSD): recurrently updates state M_macro driven by e_micro_weighted surprise.
    - Bottom-Up Injection: Macro contextual prior h_macro is broadcast to guide final motor readout.
    """
    def __init__(self, config, device='cpu'):
        super().__init__(config, device)
        
        # 1. Macro-Level Top-Down Prior Generator (h_macro -> h_s1_hat trajectory)
        self.macro_topdown_gen = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        ).to(self.device)
        nn.init.zeros_(self.macro_topdown_gen[2].weight)
        nn.init.zeros_(self.macro_topdown_gen[2].bias)

        # 2. Precision Estimation Network for Micro-to-Macro error routing
        self.micro_precision_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 1, 128),
            nn.SiLU(),
            nn.Linear(128, self.hidden_dim),
            nn.Sigmoid()
        ).to(self.device)

        # 3. Macro Context Projection onto Micro Sequence
        self.macro_context_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        ).to(self.device)

    def get_all_parameters(self):
        params = super().get_all_parameters()
        params.extend(list(self.macro_topdown_gen.parameters()))
        params.extend(list(self.micro_precision_net.parameters()))
        params.extend(list(self.macro_context_gate.parameters()))
        return params

    def forward_multimodal_sequence(self, sensor_seq_dict, target_seq, hu_batch, criterion_speech, episodic_memory=None, loss_free_energy_weight=0.05, chunk_size=64):
        text_seq = sensor_seq_dict.get('text')
        batch_size, seq_len = text_seq.size()
        num_chunks = seq_len // chunk_size
        
        m_s1 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()
        h_prev_fast = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        
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

        # Hippocampal retrieval
        active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
        if episodic_memory is not None and active_slots > 2:
            q_sensory = self.episodic_sensory_proj(full_emb.mean(dim=1)).float()
            ret_mem, max_sim = episodic_memory.read(q_sensory, temperature=0.05, threshold=0.65, sigmoid_beta=15.0)
            ret_mem_unrolled = ret_mem.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous().view(batch_size * seq_len, -1).float()
            unrolled_inputs['episodic_recall'] = ret_mem_unrolled
        else:
            unrolled_inputs['episodic_recall'] = torch.zeros(batch_size * seq_len, self.unified_dim, device=self.device).float()

        h_prev_unrolled = torch.zeros(batch_size * seq_len, self.hidden_dim, device=self.device).float()
        u_t_unrolled = curr_u_t.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous().view(batch_size * seq_len, -1).float()
        
        with torch.amp.autocast(device_type=self.device_str, enabled=False):
            w_t_unrolled, attn_weights_unrolled, channel_names, epistemic_entropy_unrolled = self.gateway(
                unrolled_inputs, h_prev_unrolled, u_t_unrolled
            )
        
        w_t_seq = w_t_unrolled.view(batch_size, seq_len, self.unified_dim)
        
        with torch.amp.autocast(device_type=self.device_str, dtype=torch.float16, enabled=('cuda' in self.device_str)):
            full_h_in = self.in_proj(w_t_seq)

            # 1. Stage 1: Fast Morpho-Syntactic Micro-Scale Scan
            if self.training and 'cuda' in self.device_str:
                h_s1, m_s1, dt1 = checkpoint.checkpoint(
                    self._stage1_forward, full_h_in, m_s1, curr_u_t, use_reentrant=False
                )
            else:
                h_s1, m_s1, dt1 = self._stage1_forward(full_h_in, m_s1, curr_u_t)

            saliency_gate = self.boundary_detector(h_s1, text_seq)

            # 2. Hierarchical Predictive Resonant Coupling:
            # Macro context from Stage 2 is initialized with zero / prior state
            h_macro_prev = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)
            
            # Predict Micro trajectory from Macro context
            h_s1_hat = self.macro_topdown_gen(h_macro_prev).expand_as(h_s1)
            e_micro_raw = h_s1 - h_s1_hat
            
            na_level = curr_u_t[:, 4:5].unsqueeze(1).expand(batch_size, seq_len, 1)
            prec_in = torch.cat([h_s1, h_s1_hat, na_level], dim=-1)
            pi_t = 2.0 * self.micro_precision_net(prec_in)
            e_micro_weighted = pi_t * e_micro_raw

            # 3. Dynamic dt scaling based on predicted entropy
            predicted_entropy = self.entropy_predictor(h_s1)
            dynamic_dt_scale = 0.40 + 1.20 * predicted_entropy

            # 4. Stage 2: Slow Semantic-Discourse Macro-Scale Scan driven by e_micro_weighted surprise
            if self.training and 'cuda' in self.device_str:
                h_s2, m_s2, dt2 = checkpoint.checkpoint(
                    self._stage2_forward, e_micro_weighted, m_s2, curr_u_t, saliency_gate, dynamic_dt_scale.mean().item(), use_reentrant=False
                )
            else:
                h_s2, m_s2, dt2 = self._stage2_forward(e_micro_weighted, m_s2, curr_u_t, saliency_gate, dynamic_dt_scale.mean().item())

            h_s2 = h_s2 * dynamic_dt_scale

            # 5. Resonant Context Fusion: Context Gate modulates Stage 1 + Stage 2 integration
            macro_fusion_gate = self.macro_context_gate(torch.cat([h_s1, h_s2], dim=-1))
            h_combined = h_s1 + macro_fusion_gate * h_s2 + 0.15 * self.topdown_prior_proj(h_s2)

            # Hierarchical Volitional Override
            effective_u_t, gamma_override, allostatic_strain = self.will_engine(h_combined, curr_u_t)

            eff_dt = (dt1 + dt2) / 2.0

            # Attractor head relaxation
            h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
            h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, effective_u_t)
            
            # Volition-Modulated Motor Text Logits
            u_t_unrolled_step = effective_u_t.repeat_interleave(seq_len, dim=0)
            volitional_logits_flat = self.volitional_head.compute_volitional_logits(
                h_relaxed, u_t_unrolled_step, self.pos_embeddings.byte_embed.weight
            )

            targets_flat = target_seq.contiguous().view(-1)
            speech_loss_tensor = criterion_speech(volitional_logits_flat, targets_flat)

            w_current_slice = w_t_seq[:, -1, :]
            h_curr_fast = h_combined[:, -1, :]
            w_pred, kl_div, fe, _ = self.world_model(h_prev_fast, h_curr_fast, w_current_slice)

            rec_loss = (1.0 - F.cosine_similarity(w_current_slice, w_pred, dim=-1, eps=1e-8)).mean()
            hpc_loss = F.mse_loss(h_s1, h_s1_hat)
            fe_loss_tensor = (kl_div.mean() + rec_loss + 0.10 * hpc_loss)

            if num_chunks > 1:
                h_chunk_endpoints = h_combined.detach().view(batch_size, num_chunks, chunk_size, self.hidden_dim)[:, :, -1, :]
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


def run_exp_103_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-103 (HIERARCHICAL PREDICTIVE RESONANCE)] ===")
    print("="*85)
    print(f"Hardware Device : {device_str.upper()} | AMP FP16 Enabled: {use_amp}")

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12

    b_size, seq_len = 16, 1024
    num_eval_steps = 150
    chunk_size = 64

    batches = prepare_packed_stream(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)

    # 1. EVALUATE BASELINE (Standard Production CoREAgent)
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 1: BASELINE (CoREAgent v31.0 Master) <<<")
    print("-"*85)
    gc.collect()
    if 'cuda' in device_str:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
    
    agent_base = CoREAgent(config, device=device_str).to(device)
    hu_base = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_base = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    
    # Calibrated learning rate for batch size 16
    opt_base = torch.optim.AdamW(agent_base.get_all_parameters(), lr=5e-4, weight_decay=0.01)
    scaler_base = torch.amp.GradScaler(device_str, enabled=use_amp)
    crit_speech = nn.CrossEntropyLoss(ignore_index=256)

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
        base_losses.append(s_loss)
        if (step + 1) % 50 == 0:
            print(f"  [Baseline Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | Free Energy: {fe_loss:.4f}")

    base_duration = time.perf_counter() - t0
    base_vram = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024) if 'cuda' in device_str else 0.0
    base_final_loss = sum(base_losses[-25:]) / 25.0
    base_ppl = math.exp(min(base_final_loss, 20.0))
    base_tok_per_sec = (num_eval_steps * b_size * seq_len) / base_duration

    print(f"\n[Baseline Summary] Final Loss: {base_final_loss:.4f} | PPL: {base_ppl:.2f} | Peak VRAM: {base_vram:.1f} MB | Throughput: {base_tok_per_sec:.1f} tok/s")

    # 2. EVALUATE PROPOSED (Hierarchical Predictive Resonance Agent)
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 2: PROPOSED (Hierarchical Predictive Resonance Agent) <<<")
    print("-"*85)
    gc.collect()
    if 'cuda' in device_str:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)

    agent_prop = HierarchicalPredictiveResonanceAgent(config, device=device_str).to(device)
    hu_prop = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_prop = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    
    # Calibrated learning rate for batch size 16
    opt_prop = torch.optim.AdamW(agent_prop.get_all_parameters(), lr=5e-4, weight_decay=0.01)
    scaler_prop = torch.amp.GradScaler(device_str, enabled=use_amp)

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
        prop_losses.append(s_loss)
        if (step + 1) % 50 == 0:
            print(f"  [Proposed Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | Free Energy: {fe_loss:.4f}")

    prop_duration = time.perf_counter() - t0
    prop_vram = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024) if 'cuda' in device_str else 0.0
    prop_final_loss = sum(prop_losses[-25:]) / 25.0
    prop_ppl = math.exp(min(prop_final_loss, 20.0))
    prop_tok_per_sec = (num_eval_steps * b_size * seq_len) / prop_duration

    print(f"\n[Proposed Summary] Final Loss: {prop_final_loss:.4f} | PPL: {prop_ppl:.2f} | Peak VRAM: {prop_vram:.1f} MB | Throughput: {prop_tok_per_sec:.1f} tok/s")

    # 3. KEP RULE #6: GRADIENT HEALTH INSPECTION
    print("\n" + "="*85)
    print(" === [KEP RULE #6 GRADIENT FLOW AUDIT (PROPOSED MODEL)] ===")
    print("="*85)
    zero_grads = 0
    healthy_grads = 0
    for name, param in agent_prop.named_parameters():
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
        print(f"  {name:<55} | Grad Norm: {g_norm if param.grad is not None else 0.0:<12.6f} | {status}")

    print("-" * 85)
    print(f"Audit Summary: Healthy: {healthy_grads} | Disconnected/Zero: {zero_grads}")

    # 4. KEP RULE #4: DIAGNOSTIC SPEECH SAMPLE AUDIT
    print("\n" + "="*85)
    print(" === [KEP RULE #4 DIAGNOSTIC SPEECH SAMPLE AUDIT] ===")
    print("="*85)
    agent_prop.eval()
    diag_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"
    diag_hu = HomeostaticUnit(batch_size=1, device=device_str)
    diag_mem = BatchedEpisodicMemory(batch_size=1, memory_dim=256, max_capacity=200, device=device_str)
    
    sample_chars = []
    with torch.no_grad():
        gen_stream = agent_prop.generate_thought_and_speech(
            prompt=diag_prompt,
            m_state=torch.zeros(1, agent_prop.num_heads, agent_prop.head_k, agent_prop.head_v, device=device),
            h_state=torch.zeros(1, agent_prop.hidden_dim, device=device),
            hu=diag_hu,
            episodic_memory=diag_mem,
            config=config,
            max_generated_tokens=70
        )
        for event in gen_stream:
            if event["status"] == "token":
                sample_chars.append(event["text"])

    print(f"  Prompt : \"{diag_prompt.strip()}\"")
    print(f"  Sample : \"{''.join(sample_chars).strip()}\"")
    print("="*85)

    # 5. KEP RULE #2 FINAL DECISION EVALUATION
    delta_loss = base_final_loss - prop_final_loss
    speed_retention = (prop_tok_per_sec / base_tok_per_sec) * 100.0

    print("\n" + "="*85)
    print(" === [KEP RULE #2 FINAL EMPIRICAL DECISION EVALUATION] ===")
    print("="*85)
    print(f"Baseline Loss      : {base_final_loss:.4f} (PPL: {base_ppl:.2f}) | VRAM: {base_vram:.1f} MB | Speed: {base_tok_per_sec:.1f} tok/s")
    print(f"Proposed Loss      : {prop_final_loss:.4f} (PPL: {prop_ppl:.2f}) | VRAM: {prop_vram:.1f} MB | Speed: {prop_tok_per_sec:.1f} tok/s")
    print(f"Delta Loss         : {delta_loss:+.4f} ({'IMPROVEMENT' if delta_loss > 0 else 'REGRESSION'})")
    print(f"Speed Retention    : {speed_retention:.1f}%")
    print(f"Zero Gradients     : {zero_grads}")

    if delta_loss >= 0.08 and speed_retention >= 80.0 and zero_grads == 0:
        verdict = "🟢 POSITIVE"
    elif delta_loss < -0.08:
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
        "base_ppl": base_ppl,
        "prop_ppl": prop_ppl,
        "base_vram": base_vram,
        "prop_vram": prop_vram,
        "base_tok_per_sec": base_tok_per_sec,
        "prop_tok_per_sec": prop_tok_per_sec,
        "speed_retention_pct": speed_retention
    }


if __name__ == "__main__":
    run_exp_103_benchmark()
