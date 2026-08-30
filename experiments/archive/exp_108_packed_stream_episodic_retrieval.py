# experiments/exp_108_packed_stream_episodic_retrieval.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-108 (VECTOR 2: PACKED STREAM FACT RETRIEVAL)
Hypothesis:
Continuous non-parametric factual encoding during high arousal/surprise (NA > 0.10 or F_t > 0.18)
coupled with dynamic GWT hippocampal retrieval ('episodic_recall' channel) in packed stream
training offloads factual memorization from static parameters, accelerating convergence and
reducing packed speech loss on real multi-turn dialogue data (vicgalle/alpaca-gpt4).
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
from karyon_agent import CoREAgent
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')


def prepare_packed_batches(num_batches: int = 100, batch_size: int = 8, seq_len: int = 1024):
    logger.info(f"Loading vicgalle/alpaca-gpt4 dataset for EXP-108 (B={batch_size}, S={seq_len}, Steps={num_batches})...")
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

    block_size = seq_len + 1
    batches = []
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

    logger.info(f"Prepared {len(batches)} Packed Batches (Total Tokens: {len(batches) * batch_size * seq_len / 1e6:.2f}M).")
    return batches


class EpisodicFactAwareCoREAgent(CoREAgent):
    """
    Enhanced CoREAgent featuring dynamic chunk-level factual encoding and GWT memory retrieval.
    Writes key-value pairs during high arousal (NA > 0.10) or surprise (F_t > 0.18).
    """
    def forward_multimodal_sequence(self, sensor_seq_dict, target_seq, hu_batch, criterion_speech,
                                   episodic_memory=None, loss_free_energy_weight=0.05, chunk_size=64):
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

        # Vector 2: Dynamic Chunk-Level Hippocampal Retrieval directly into 'episodic_recall'
        active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
        if episodic_memory is not None and active_slots > 2:
            q_sensory = self.episodic_sensory_proj(full_emb.mean(dim=1)).float()
            ret_mem, max_sim = episodic_memory.read(q_sensory, temperature=0.05, threshold=0.55, sigmoid_beta=15.0)
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
        
        with torch.amp.autocast(device_type=self.device_str, dtype=torch.float16, enabled=(self.device_str == 'cuda')):
            full_h_in = self.in_proj(w_t_seq)

            h_s1, m_s1, dt1 = self._stage1_forward(full_h_in, m_s1, curr_u_t)
            saliency_gate = self.boundary_detector(h_s1, text_seq)

            h_s2_prev_shifted = torch.zeros_like(h_s1)
            e1_weighted, h_s1_hat, mean_pi = self.pw_hpc_generator(h_s1, h_s2_prev_shifted, curr_u_t)

            predicted_entropy = self.entropy_predictor(h_s1)
            dynamic_dt_scale = 0.40 + 1.20 * predicted_entropy

            h_s2, m_s2, dt2 = self._stage2_forward(e1_weighted, m_s2, curr_u_t, saliency_gate, dynamic_dt_scale.mean().item())
            h_s2 = h_s2 * dynamic_dt_scale

            effective_u_t, gamma_override, allostatic_strain = self.will_engine(h_s2, curr_u_t)
            eff_dt = (dt1 + dt2) / 2.0
            topdown_prior = self.topdown_prior_proj(h_s2)
            h_combined = h_s1 + h_s2 + 0.15 * topdown_prior

            h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
            h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, effective_u_t)
            
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
            fe_loss_tensor = (kl_div.mean() + rec_loss)

            # High-Arousal or High-Surprise Non-Parametric Episodic Fact Encoding
            na_level = curr_u_t[:, 4].mean().item()
            with torch.no_grad():
                if (fe_loss_tensor.item() > 0.18 or na_level > 0.10) and episodic_memory is not None:
                    episodic_memory.write(w_current_slice.detach().float(), w_pred.detach().float(), protected_slots=3)

            ortho_loss = self.attractor_head.compute_pattern_separation_loss()
            speech_loss_val = speech_loss_tensor.item()
            fe_loss_val = fe_loss_tensor.item()
            
            total_loss_tensor = (
                speech_loss_tensor + 
                loss_free_energy_weight * fe_loss_tensor + 
                0.05 * commit_loss + 
                0.01 * ortho_loss
            )

        h_proxy = m_s2.view(batch_size, -1)[:, :self.hidden_dim]
        return total_loss_tensor, speech_loss_val, fe_loss_val, m_s2, h_proxy, curr_u_t, eff_dt


def run_exp_108_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-108 (VECTOR 2: PACKED STREAM FACT RETRIEVAL)] ===")
    print("="*85)

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12
    config.memory.max_capacity = 2000

    b_size, seq_len = 8, 1024
    num_eval_steps = 100

    batches = prepare_packed_batches(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)

    # 1. EVALUATE BASELINE (Without Active Hippocampal Retrieval)
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 1: BASELINE (CoREAgent without Episodic Memory Retrieval) <<<")
    print("-"*85)
    torch.cuda.empty_cache()
    gc.collect()
    
    agent_base = CoREAgent(config, device=device_str).to(device)
    hu_base = HomeostaticUnit(batch_size=b_size, device=device_str)
    
    # Use a lower learning rate (5e-4) with a small warmup to prevent NaNs
    opt_base = torch.optim.AdamW(agent_base.get_all_parameters(), lr=5e-4, weight_decay=0.01)
    scaler_base = torch.amp.GradScaler('cuda', enabled=use_amp)
    crit_speech = nn.CrossEntropyLoss(ignore_index=256)

    t0 = time.perf_counter()
    base_losses = []
    
    for step, batch_tensors in enumerate(batches):
        inp = batch_tensors[:, :-1]
        tgt = batch_tensors[:, 1:]
        opt_base.zero_grad()
        
        # Warmup learning rate for first 15 steps to prevent NaNs
        if step < 15:
            for g in opt_base.param_groups:
                g['lr'] = 5e-4 * (step + 1) / 15.0

        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt = agent_base.forward_sequence(
                inp, tgt, hu_base, crit_speech, episodic_memory=None, chunk_size=64
            )
        
        if math.isnan(s_loss) or math.isnan(fe_loss):
            print(f"  ⚠️ Warning: NaN detected at step {step+1}. Skipping backward pass.")
            continue

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
    valid_base_losses = [l for l in base_losses if not math.isnan(l)]
    base_final_loss = sum(valid_base_losses[-15:]) / max(len(valid_base_losses[-15:]), 1)
    base_ppl = math.exp(min(base_final_loss, 20.0))
    base_tok_per_sec = (num_eval_steps * b_size * seq_len) / base_duration

    print(f"[Baseline Summary] Final Loss: {base_final_loss:.4f} | PPL: {base_ppl:.2f} | Speed: {base_tok_per_sec:.1f} tok/s")

    # Clean up baseline variables to prevent CUDA OOM
    del agent_base, opt_base, scaler_base, hu_base
    gc.collect()
    torch.cuda.empty_cache()

    # 2. EVALUATE PROPOSED (EpisodicFactAwareCoREAgent with Active Non-Parametric Retrieval)
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 2: PROPOSED (EpisodicFactAwareCoREAgent with Dynamic GWT Fact Retrieval) <<<")
    print("-"*85)
    torch.cuda.empty_cache()
    gc.collect()

    agent_prop = EpisodicFactAwareCoREAgent(config, device=device_str).to(device)
    hu_prop = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_prop = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=2000, device=device_str)
    opt_prop = torch.optim.AdamW(agent_prop.get_all_parameters(), lr=5e-4, weight_decay=0.01)
    scaler_prop = torch.amp.GradScaler('cuda', enabled=use_amp)

    t0 = time.perf_counter()
    prop_losses = []
    
    for step, batch_tensors in enumerate(batches):
        inp = batch_tensors[:, :-1]
        tgt = batch_tensors[:, 1:]
        opt_prop.zero_grad()
        
        # Warmup learning rate for first 15 steps to prevent NaNs
        if step < 15:
            for g in opt_prop.param_groups:
                g['lr'] = 5e-4 * (step + 1) / 15.0

        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt = agent_prop.forward_sequence(
                inp, tgt, hu_prop, crit_speech, episodic_memory=mem_prop, chunk_size=64
            )
        
        if math.isnan(s_loss) or math.isnan(fe_loss):
            print(f"  ⚠️ Warning: NaN detected at step {step+1}. Skipping backward pass.")
            continue

        scaler_prop.scale(tot_loss).backward()
        scaler_prop.unscale_(opt_prop)
        torch.nn.utils.clip_grad_norm_(agent_prop.get_all_parameters(), max_norm=3.0)
        scaler_prop.step(opt_prop)
        scaler_prop.update()
        prop_losses.append(s_loss)
        if (step + 1) % 25 == 0:
            active_slots = getattr(mem_prop, 'max_active_cpu', 0)
            print(f"  [Proposed Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | FE: {fe_loss:.4f} | Memory Active Slots: {active_slots}")

    prop_duration = time.perf_counter() - t0
    prop_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    valid_prop_losses = [l for l in prop_losses if not math.isnan(l)]
    prop_final_loss = sum(valid_prop_losses[-15:]) / max(len(valid_prop_losses[-15:]), 1)
    prop_ppl = math.exp(min(prop_final_loss, 20.0))
    prop_tok_per_sec = (num_eval_steps * b_size * seq_len) / prop_duration

    print(f"[Proposed Summary] Final Loss: {prop_final_loss:.4f} | PPL: {prop_ppl:.2f} | Speed: {prop_tok_per_sec:.1f} tok/s")

    # KEP Decision Evaluation
    delta_loss = base_final_loss - prop_final_loss
    speed_retention_pct = (prop_tok_per_sec / base_tok_per_sec) * 100.0

    print("\n" + "="*85)
    print(" === [KEP RULE #2 FINAL EMPIRICAL DECISION EVALUATION] ===")
    print("="*85)
    print(f"Baseline Speech Loss : {base_final_loss:.4f} (PPL: {base_ppl:.2f}) | Speed: {base_tok_per_sec:.1f} tok/s")
    print(f"Proposed Speech Loss : {prop_final_loss:.4f} (PPL: {prop_ppl:.2f}) | Speed: {prop_tok_per_sec:.1f} tok/s")
    print(f"Delta Loss           : {delta_loss:+.4f} ({'IMPROVEMENT' if delta_loss >= 0.08 else 'NEUTRAL/MARGINAL'})")
    print(f"Speed Retention      : {speed_retention_pct:.1f}%")

    if delta_loss >= 0.08 and speed_retention_pct >= 80.0:
        verdict = "🟢 POSITIVE"
    elif delta_loss < -0.05:
        verdict = "🔴 REJECTED"
    else:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"

    print(f"VERDICT              : {verdict}")
    print("="*85 + "\n")

    # Save metrics JSON
    metrics = {
        "verdict": verdict,
        "base_loss": base_final_loss,
        "prop_loss": prop_final_loss,
        "delta_loss": delta_loss,
        "base_ppl": base_ppl,
        "prop_ppl": prop_ppl,
        "base_tok_per_sec": base_tok_per_sec,
        "prop_tok_per_sec": prop_tok_per_sec,
        "speed_retention_pct": speed_retention_pct,
        "active_memory_slots": getattr(mem_prop, 'max_active_cpu', 0),
        "peak_vram_mb": prop_vram
    }
    with open("exp_108_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    run_exp_108_benchmark()
