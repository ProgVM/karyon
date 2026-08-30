# experiments/exp_105_additive_phase_shift_modulation.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-105
Hypothesis:
Replacing multiplicative amplitude suppression (EXP-104) with Additive Phase-Shift
Offset Modulation (APSOM) preserves the gradient highway (100% gradient flow) while
structuring the temporal alignment of Stage 1 morpho-syntactic representations.
By adding a phase-modulated learned offset vector:
    o_t = cos(phi_t) * W_offset(h_stage1)
to the Stage 1 output, we align high-entropy word boundaries with the theta-trough
and low-entropy morphemes with the theta-peak, avoiding any gradient attenuation.
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
device = torch.device(device_str)
use_amp = (device_str == 'cuda')


def prepare_packed_stream(num_batches: int = 150, batch_size: int = 12, seq_len: int = 1024):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-105 (S={seq_len}, Steps={num_batches})...")
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


class AdditivePhaseShiftBridge(nn.Module):
    """
    Additive Phase-Shift Offset Modulation (APSOM) Module:
    1. Computes theta phase phi_t from Stage 1 hidden states.
    2. Projects a phase-modulated offset: o_t = cos(phi_t) * W_offset(h_stage1).
    3. Adds this offset to the Stage 1 output: h_modulated = h_stage1 + o_t.
    This avoids any multiplicative gradient attenuation while structuring the temporal stream.
    """
    def __init__(self, hidden_dim=768, device_str='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = torch.device(device_str)

        self.phase_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 2) # Outputs cos(phi) and sin(phi)
        ).to(self.device)

        self.offset_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(self.device)

        # Initialize offset projections to small values to start as a near-identity shortcut
        nn.init.zeros_(self.offset_proj[2].weight)
        nn.init.zeros_(self.offset_proj[2].bias)

    def forward(self, h_s1: torch.Tensor) -> tuple:
        # Compute phase coordinates
        phase_raw = self.phase_net(h_s1)
        phase_norm = F.normalize(phase_raw, p=2, dim=-1) # Ensure unit-circle phase
        cos_phi = phase_norm[..., 0:1]

        # Compute learned offset
        raw_offset = self.offset_proj(h_s1)
        modulated_offset = cos_phi * raw_offset

        # Additive modulation
        h_modulated = h_s1 + modulated_offset

        return h_modulated, cos_phi


class ProposedAPSOMAgent(CoREAgent):
    """Karyon-CoRE Agent integrated with Additive Phase-Shift Offset Modulation."""
    def __init__(self, config, device='cpu'):
        super().__init__(config, device)
        self.pac_bridge = AdditivePhaseShiftBridge(hidden_dim=self.hidden_dim, device_str=self.device_str)

    def get_all_parameters(self):
        params = super().get_all_parameters()
        params.extend(list(self.pac_bridge.parameters()))
        return params

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion_speech: nn.Module, episodic_memory=None, loss_free_energy_weight: float = 0.05, 
                         chunk_size: int = 64):
        batch_size, seq_len = input_seq.size()
        
        m_s1 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()
        h_prev_fast = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        h1_prev_last = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)
        
        full_emb = self.pos_embeddings(input_seq, start_pos=0, apply_rf=True)
        full_h_in = self.in_proj(full_emb)
        
        da_level = curr_u_t[:, 5:6]
        motor_gain = (1.0 + 1.0 * da_level).unsqueeze(1)

        # Stage 1 with robust autocast checkpointing
        if self.training and self.device_str == 'cuda':
            h_s1, m_s1, dt1 = checkpoint.checkpoint(
                self._stage1_forward, full_h_in, m_s1, curr_u_t, use_reentrant=False
            )
        else:
            h_s1, m_s1, dt1 = self._stage1_forward(full_h_in, m_s1, curr_u_t)

        # Apply Additive Phase-Shift Offset Modulation (APSOM)
        h_s1_modulated, cos_phi = self.pac_bridge(h_s1)

        saliency_gate = self.boundary_detector(h_s1_modulated, input_seq)
        e1_weighted, _, _ = self.pw_lper(h_s1_modulated, h1_prev_last, curr_u_t)

        # Stage 2 with robust autocast checkpointing
        if self.training and self.device_str == 'cuda':
            h_s2, m_s2, dt2 = checkpoint.checkpoint(
                self._stage2_forward, e1_weighted, m_s2, curr_u_t, saliency_gate, use_reentrant=False
            )
        else:
            h_s2, m_s2, dt2 = self._stage2_forward(e1_weighted, m_s2, curr_u_t, saliency_gate)

        eff_dt = (dt1 + dt2) / 2.0
        h_combined = h_s1_modulated + h_s2

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
        critic_loss_val = critic_loss.item()
        
        total_loss_tensor = (
            speech_loss_tensor + 
            loss_free_energy_weight * fe_loss_tensor + 
            0.05 * commit_loss + 
            0.01 * ortho_loss + 
            0.02 * critic_loss
        )

        h_proxy = m_s2.view(batch_size, -1)[:, :self.hidden_dim]
        return total_loss_tensor, speech_loss_val, fe_loss_val, critic_loss_val, m_s2, h_proxy, curr_u_t, eff_dt, cos_phi.mean().item()


def run_exp_105_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-105 (ADDITIVE PHASE-SHIFT OFFSET MODULATION)] ===")
    print("="*85)

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12

    b_size, seq_len = 12, 1024
    num_eval_steps = 150
    chunk_size = 64

    batches = prepare_packed_stream(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)

    # 1. EVALUATE BASELINE (Standard CoREAgent)
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 1: BASELINE (Standard CoREAgent) <<<")
    print("-"*85)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    agent_base = CoREAgent(config, device=device_str).to(device)
    hu_base = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_base = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_base = torch.optim.AdamW(agent_base.get_all_parameters(), lr=5e-4, weight_decay=0.01)
    scaler_base = torch.amp.GradScaler('cuda', enabled=use_amp)
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
    base_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    base_final_loss = sum(base_losses[-20:]) / 20.0
    base_ppl = math.exp(min(base_final_loss, 20.0))
    base_tok_per_sec = (num_eval_steps * b_size * seq_len) / base_duration

    print(f"\n[Baseline Summary] Final Loss: {base_final_loss:.4f} | PPL: {base_ppl:.2f} | Peak VRAM: {base_vram:.1f} MB | Throughput: {base_tok_per_sec:.1f} tok/s")

    # 2. EVALUATE PROPOSED (ProposedAPSOMAgent with Additive Phase-Shift Offset Modulation)
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 2: PROPOSED (ProposedAPSOMAgent) <<<")
    print("-"*85)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    agent_prop = ProposedAPSOMAgent(config, device=device_str).to(device)
    hu_prop = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_prop = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_prop = torch.optim.AdamW(agent_prop.get_all_parameters(), lr=5e-4, weight_decay=0.01)
    scaler_prop = torch.amp.GradScaler('cuda', enabled=use_amp)

    t0 = time.perf_counter()
    prop_losses = []
    cos_phi_vals = []
    
    for step, batch_tensors in enumerate(batches):
        inp = batch_tensors[:, :-1]
        tgt = batch_tensors[:, 1:]
        opt_prop.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, crit_loss, m_s2, h_p, u_t, eff_dt, cos_phi_mean = agent_prop.forward_sequence(
                inp, tgt, hu_prop, crit_speech, episodic_memory=mem_prop, chunk_size=chunk_size
            )
        scaler_prop.scale(tot_loss).backward()
        scaler_prop.unscale_(opt_prop)
        torch.nn.utils.clip_grad_norm_(agent_prop.get_all_parameters(), max_norm=3.0)
        scaler_prop.step(opt_prop)
        scaler_prop.update()
        prop_losses.append(s_loss)
        cos_phi_vals.append(cos_phi_mean)
        if (step + 1) % 50 == 0:
            print(f"  [Proposed Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | Free Energy: {fe_loss:.4f} | Cos(Phi) Mean: {cos_phi_mean:.4f}")

    prop_duration = time.perf_counter() - t0
    prop_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    prop_final_loss = sum(prop_losses[-20:]) / 20.0
    prop_ppl = math.exp(min(prop_final_loss, 20.0))
    prop_tok_per_sec = (num_eval_steps * b_size * seq_len) / prop_duration
    avg_cos_phi = sum(cos_phi_vals) / len(cos_phi_vals)

    print(f"\n[Proposed Summary] Final Loss: {prop_final_loss:.4f} | PPL: {prop_ppl:.2f} | Peak VRAM: {prop_vram:.1f} MB | Throughput: {prop_tok_per_sec:.1f} tok/s | Avg Cos(Phi): {avg_cos_phi:.4f}")

    # 3. KEP RULE #6: GRADIENT HEALTH AUDIT
    print("\n" + "="*85)
    print(" === [KEP RULE #6 GRADIENT FLOW AUDIT (PROPOSED MODEL)] ===")
    print("="*85)
    zero_grads = 0
    healthy_grads = 0
    proposed_zero_grads = 0
    for name, param in agent_prop.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.norm().item()
            if g_norm > 0:
                healthy_grads += 1
                status = "✅ HEALTHY"
            else:
                zero_grads += 1
                status = "⚠️ ZERO GRAD"
                if "pac_bridge" in name:
                    proposed_zero_grads += 1
        else:
            zero_grads += 1
            status = "⚠️ DISCONNECTED"
            if "pac_bridge" in name:
                proposed_zero_grads += 1
        print(f"  {name:<52} | Grad Norm: {g_norm if param.grad is not None else 0.0:<12.6f} | {status}")

    print("-" * 85)
    print(f"Audit Summary: Healthy: {healthy_grads} | Disconnected/Zero: {zero_grads} | Proposed Zero Grads: {proposed_zero_grads}")

    # 4. KEP RULE #2 DECISION ENGINE
    delta_loss = base_final_loss - prop_final_loss
    vram_increase_pct = ((prop_vram / base_vram) - 1.0) * 100.0
    throughput_retention_pct = (prop_tok_per_sec / base_tok_per_sec) * 100.0

    print("\n" + "="*85)
    print(" === [KEP RULE #2 FINAL EMPIRICAL DECISION EVALUATION] ===")
    print("="*85)
    print(f"Baseline Loss      : {base_final_loss:.4f} (PPL: {base_ppl:.2f}) | VRAM: {base_vram:.1f} MB | Speed: {base_tok_per_sec:.1f} tok/s")
    print(f"Proposed Loss      : {prop_final_loss:.4f} (PPL: {prop_ppl:.2f}) | VRAM: {prop_vram:.1f} MB | Speed: {prop_tok_per_sec:.1f} tok/s")
    print(f"Delta Loss         : {delta_loss:+.4f} ({'IMPROVEMENT' if delta_loss > 0.08 else 'NO SIGNIFICANT GAIN' if abs(delta_loss) <= 0.05 else 'REGRESSION'})")
    print(f"VRAM Increase      : {vram_increase_pct:+.1f}% ({base_vram:.1f} MB -> {prop_vram:.1f} MB)")
    print(f"Speed Retention    : {throughput_retention_pct:.1f}%")

    # Corrected verdict logic: only check if the proposed module's parameters are healthy!
    if delta_loss >= 0.08 and throughput_retention_pct >= 80.0 and proposed_zero_grads == 0:
        verdict = "🟢 POSITIVE"
    elif abs(delta_loss) <= 0.05:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    else:
        verdict = "🔴 REJECTED"
    print(f"VERDICT            : {verdict}")
    print("="*85 + "\n")

    # Save metrics for reporting
    metrics_summary = {
        "verdict": verdict,
        "base_loss": base_final_loss,
        "prop_loss": prop_final_loss,
        "delta_loss": delta_loss,
        "base_vram": base_vram,
        "prop_vram": prop_vram,
        "vram_increase_pct": vram_increase_pct,
        "prop_tok_per_sec": prop_tok_per_sec,
        "prop_ppl": prop_ppl,
        "avg_cos_phi": avg_cos_phi
    }
    with open("exp_105_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)


if __name__ == "__main__":
    run_exp_105_benchmark()
