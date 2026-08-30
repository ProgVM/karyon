# experiments/exp_93_multimodal_cross_active_alignment.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-93
Hypothesis:
1. Cross-Modal Bidirectional Active Inference Alignment (CM-AIA) between dynamic
   sensory channels (text, vision, audio, cybernetic, body) allows mutual predictive
   contextualization prior to Global Workspace Theory (GWT) frame selection.
2. Somatic Arousal (NA) dynamic precision scaling sharpens sensory focus during high-surprise
   episodes, accelerating Free Energy (F_t) and Speech Loss reduction.
3. Preserves high GPU throughput (>20,000 tok/s) and VRAM efficiency (<6.5 GB).
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
from karyon_agent import CoREAgent, DynamicSensoryGateway
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')


class CrossActiveAlignedSensoryGateway(DynamicSensoryGateway):
    """
    EXP-93 Proposed Module: Cross-Modal Bidirectional Active Alignment Gateway (CM-AIA).
    Computes mutual inter-modal predictive residuals and modulates channel attention
    weighted by somatic arousal (NA).
    """
    def __init__(self, unified_dim=256, hidden_dim=768, homeo_dim=6, device_str='cpu'):
        super().__init__(unified_dim, hidden_dim, homeo_dim, device_str)
        # Cross-modal active alignment projection
        self.cross_modal_align = nn.Sequential(
            nn.Linear(unified_dim, unified_dim),
            nn.SiLU(),
            nn.Linear(unified_dim, unified_dim)
        ).to(self.device)
        nn.init.zeros_(self.cross_modal_align[2].weight)
        nn.init.zeros_(self.cross_modal_align[2].bias)

    def forward(self, sensor_inputs, h_prev, u_t):
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
            
            proj_x = proj(x_in.float())
            projected_channels.append(proj_x)
            channel_names.append(name)
            channel_masks.append((1.0 - x_act) * -10000.0)
            
        projected_channels.append(self.homeo_proj(u_t.float()))
        channel_names.append('body')
        channel_masks.append(torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device))
        
        projected_channels.append(self.mind_proj(h_prev.float()))
        channel_names.append('mind')
        channel_masks.append(torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device))
        
        stacked_channels = torch.stack(projected_channels, dim=1) # [B, N_channels, unified_dim]
        
        # --- CM-AIA: Cross-Modal Alignment & Precision Modulation ---
        somatic_na = u_t[:, 4:5].unsqueeze(1) # [B, 1, 1] Arousal level
        precision_pi = 1.0 + 1.5 * somatic_na
        
        mean_modal_context = stacked_channels.mean(dim=1, keepdim=True)
        aligned_context = self.cross_modal_align(mean_modal_context)
        
        aligned_channels = stacked_channels + precision_pi * 0.10 * aligned_context
        norm_stacked = self.channel_norm(aligned_channels)
        
        volition_query = self.attention_query_layer(h_prev.float()).unsqueeze(1)
        norm_query = self.query_norm(volition_query)
        
        sim = (norm_query * norm_stacked).sum(dim=-1) / math.sqrt(self.unified_dim)
        stacked_masks = torch.cat(channel_masks, dim=1)
        sim = sim + stacked_masks
        
        attention_weights = F.softmax(sim, dim=-1)
        eps = 1e-9
        epistemic_entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-1, keepdim=True)
        
        w_t = (attention_weights.unsqueeze(-1) * aligned_channels).sum(dim=1)
        
        return w_t, attention_weights, channel_names, epistemic_entropy


class CrossActiveAlignedCoREAgent(CoREAgent):
    """Agent utilizing the proposed CrossActiveAlignedSensoryGateway."""
    def __init__(self, config, device='cpu'):
        super().__init__(config, device)
        self.gateway = CrossActiveAlignedSensoryGateway(
            unified_dim=self.unified_dim,
            hidden_dim=self.hidden_dim,
            homeo_dim=config.net.homeo_dim,
            device_str=self.device_str
        )


def prepare_multimodal_dataset(num_batches: int = 100, batch_size: int = 16, seq_len: int = 512):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-93 Multimodal Stream (S={seq_len}, Steps={num_batches})...")
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
    tokenizer = ByteTokenizer()
    
    batches = []
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
    for b in range(num_batches):
        text_batch = []
        vision_batch = []
        audio_batch = []
        cybernetic_batch = []
        
        for s in range(batch_size):
            start = (b * batch_size + s) * block_size
            end = start + block_size
            chunk = full_stream[start:end]
            if len(chunk) < block_size:
                chunk = chunk + [256] * (block_size - len(chunk))
            
            text_batch.append(torch.tensor(chunk, dtype=torch.long))
            vision_batch.append(torch.randn(block_size, 256))
            audio_batch.append(torch.randn(block_size, 256))
            cybernetic_batch.append(torch.randn(block_size, 128))
            
        batches.append({
            'text': torch.stack(text_batch, dim=0).to(device),
            'vision': torch.stack(vision_batch, dim=0).to(device),
            'audio': torch.stack(audio_batch, dim=0).to(device),
            'cybernetic': torch.stack(cybernetic_batch, dim=0).to(device)
        })

    logger.info(f"Prepared {len(batches)} Real Multimodal Batches (B={batch_size}, S={seq_len}).")
    return batches


def run_exp_93_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-93 (CROSS-MODAL ACTIVE ALIGNMENT)] ===")
    print("="*85)

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12

    b_size, seq_len = 16, 512
    num_eval_steps = 100
    chunk_size = 64

    batches = prepare_multimodal_dataset(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)

    # 1. EVALUATE BASELINE (CoREAgent)
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 1: BASELINE (CoREAgent Standard Gateway) <<<")
    print("-"*85)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    agent_base = CoREAgent(config, device=device_str).to(device)
    agent_base.register_sensory_channel('cybernetic', 128)
    
    hu_base = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_base = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_base = torch.optim.AdamW(agent_base.get_all_parameters(), lr=5e-4, weight_decay=0.01)
    scaler_base = torch.amp.GradScaler('cuda', enabled=use_amp)
    crit_speech = nn.CrossEntropyLoss(ignore_index=256)

    t0 = time.perf_counter()
    base_losses = []
    
    for step, batch_data in enumerate(batches):
        text_seq = batch_data['text']
        inp_text = text_seq[:, :-1]
        tgt_text = text_seq[:, 1:]
        
        sensor_dict = {
            'text': inp_text,
            'vision': batch_data['vision'][:, :-1],
            'audio': batch_data['audio'][:, :-1],
            'cybernetic': batch_data['cybernetic'][:, :-1]
        }

        opt_base.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt = agent_base.forward_multimodal_sequence(
                sensor_dict, tgt_text, hu_base, crit_speech, episodic_memory=mem_base, chunk_size=chunk_size
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

    # 2. EVALUATE PROPOSED (CrossActiveAlignedCoREAgent)
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 2: PROPOSED (Cross-Modal Active Alignment Gateway) <<<")
    print("-"*85)
    del agent_base, hu_base, mem_base, opt_base, scaler_base
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    agent_prop = CrossActiveAlignedCoREAgent(config, device=device_str).to(device)
    agent_prop.register_sensory_channel('cybernetic', 128)

    hu_prop = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_prop = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_prop = torch.optim.AdamW(agent_prop.get_all_parameters(), lr=5e-4, weight_decay=0.01)
    scaler_prop = torch.amp.GradScaler('cuda', enabled=use_amp)

    t0 = time.perf_counter()
    prop_losses = []
    
    for step, batch_data in enumerate(batches):
        text_seq = batch_data['text']
        inp_text = text_seq[:, :-1]
        tgt_text = text_seq[:, 1:]
        
        sensor_dict = {
            'text': inp_text,
            'vision': batch_data['vision'][:, :-1],
            'audio': batch_data['audio'][:, :-1],
            'cybernetic': batch_data['cybernetic'][:, :-1]
        }

        opt_prop.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt = agent_prop.forward_multimodal_sequence(
                sensor_dict, tgt_text, hu_prop, crit_speech, episodic_memory=mem_prop, chunk_size=chunk_size
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

    # 3. KEP RULE #6: GRADIENT FLOW AUDIT
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
        print(f"  {name:<54} | Grad Norm: {g_norm if param.grad is not None else 0.0:<12.6f} | {status}")

    print("-" * 85)
    print(f"Audit Summary: Healthy: {healthy_grads} | Disconnected/Zero: {zero_grads}")

    # 4. KEP RULE #4: DIAGNOSTIC SPEECH SAMPLE
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
            max_generated_tokens=60
        )
        for event in gen_stream:
            if event["status"] == "token":
                sample_chars.append(event["text"])

    print(f"  Prompt : \"{diag_prompt.strip()}\"")
    print(f"  Sample : \"{''.join(sample_chars).strip()}\"")
    print("="*85)

    # 5. KEP RULE #2 DECISION ENGINE
    delta_loss = base_final_loss - prop_final_loss
    vram_diff_mb = prop_vram - base_vram
    throughput_retention_pct = (prop_tok_per_sec / base_tok_per_sec) * 100.0

    print("\n" + "="*85)
    print(" === [KEP RULE #2 FINAL EMPIRICAL DECISION EVALUATION] ===")
    print("="*85)
    print(f"Baseline Loss      : {base_final_loss:.4f} (PPL: {base_ppl:.2f}) | VRAM: {base_vram:.1f} MB | Speed: {base_tok_per_sec:.1f} tok/s")
    print(f"Proposed Loss      : {prop_final_loss:.4f} (PPL: {prop_ppl:.2f}) | VRAM: {prop_vram:.1f} MB | Speed: {prop_tok_per_sec:.1f} tok/s")
    print(f"Delta Loss         : {delta_loss:+.4f} ({'IMPROVEMENT' if delta_loss >= 0.08 else 'NO SIGNIFICANT GAIN' if abs(delta_loss) <= 0.05 else 'REGRESSION'})")
    print(f"VRAM Memory Delta  : {vram_diff_mb:+.1f} MB")
    print(f"Speed Retention    : {throughput_retention_pct:.1f}%")

    if delta_loss >= 0.08 and throughput_retention_pct >= 80.0 and zero_grads == 0:
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
        "base_vram": base_vram,
        "prop_vram": prop_vram,
        "prop_tok_per_sec": prop_tok_per_sec,
        "prop_ppl": prop_ppl
    }


if __name__ == "__main__":
    run_exp_93_benchmark()
