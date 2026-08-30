# experiments/exp_104_theta_gamma_pac_laminar_coupling.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-104
Hypothesis:
Continuous Theta-Gamma Phase-Amplitude Coupling (PAC - Buzsáki & Giraud 2012)
dynamically modulates high-frequency gamma (Stage 1 micro-computations) via low-frequency
theta (Stage 2 macro-context phase envelope phi_t), with event-boundary phase resets.
This biophysical coupling reduces semantic interference, stabilizes multi-turn speech loss,
and improves multi-dimensional Free Energy / Perplexity metrics without throughput degradation.

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


# =============================================================================
# 1. THETA-GAMMA PHASE-AMPLITUDE COUPLING (PAC) BRIDGE
# =============================================================================
class ThetaGammaPACLaminarBridge(nn.Module):
    """
    Biophysical Theta-Gamma Phase-Amplitude Coupling (Buzsaki 2006, Giraud & Poeppel 2012):
    1. Extracts continuous instantaneous theta phase phi_t in [0, 2*pi) from macro-context (Stage 2).
    2. Modulates micro-gamma amplitude (Stage 1 features) via envelope: A_gamma(t) = 1.0 + 0.35 * cos(phi_t).
    3. Triggers theta phase reset on morpheme/word boundary saliency gates (Saliency ~ 1.0).
    """
    def __init__(self, hidden_dim=768, device_str='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = torch.device(device_str)

        # Instantaneous Phase Network: [cos(phi_t), sin(phi_t)]
        self.phase_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        ).to(self.device)

        # Boundary Phase-Reset Projection
        self.phase_reset_proj = nn.Linear(hidden_dim, hidden_dim).to(self.device)

    def forward(self, h_s1: torch.Tensor, saliency_gate: torch.Tensor, u_t: torch.Tensor):
        batch_size, seq_len, _ = h_s1.size()

        # 1. Extract Unit-Sphere Phase Vector [cos(phi_t), sin(phi_t)]
        phase_raw = self.phase_net(h_s1)
        phase_norm = F.normalize(phase_raw, p=2, dim=-1)
        cos_phi = phase_norm[:, :, 0:1] # [B, S, 1]

        # 2. Compute Dopamine-Modulated Gamma Envelope
        if u_t.dim() == 2:
            da_level = u_t[:, 5:6].unsqueeze(1).expand(batch_size, seq_len, 1)
        else:
            da_level = u_t[:, :, 5:6]

        gamma_envelope = 1.0 + 0.35 * cos_phi * (1.0 + 0.5 * da_level)

        # 3. Apply Amplitude Modulation
        h_s1_pac = h_s1 * gamma_envelope

        # 4. Phase Reset on Boundary Saliency
        if saliency_gate.dim() == 3:
            sal_exp = saliency_gate.transpose(1, 2) # [B, S, 1]
        else:
            sal_exp = saliency_gate.unsqueeze(-1)

        boundary_reset = self.phase_reset_proj(h_s1) * sal_exp
        e1_pac = h_s1_pac + 0.20 * boundary_reset

        return e1_pac, phase_norm, gamma_envelope.mean()


# =============================================================================
# 2. PAC-ENHANCED AGENT PROTOTYPE FOR EXP-104
# =============================================================================
class ThetaGammaPACCoREAgent(CoREAgent):
    def __init__(self, config, device='cpu'):
        super().__init__(config, device)
        self.pac_bridge = ThetaGammaPACLaminarBridge(hidden_dim=self.hidden_dim, device_str=self.device_str)

    def get_all_parameters(self):
        params = super().get_all_parameters()
        params.extend(list(self.pac_bridge.parameters()))
        return params

    def forward_multimodal_sequence(self, sensor_seq_dict, target_seq, hu_batch, criterion_speech, 
                                    episodic_memory=None, loss_free_energy_weight=0.05, chunk_size=64):
        text_seq = sensor_seq_dict.get('text')
        batch_size, seq_len = text_seq.size()
        
        m_s1 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()
        h_prev_fast = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        h1_prev_last = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)
        
        full_emb = self.pos_embeddings(text_seq, start_pos=0, apply_rf=True)
        unrolled_inputs = {'text': full_emb.contiguous().view(batch_size * seq_len, -1).float()}

        active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
        if episodic_memory is not None and active_slots > 2:
            q_sensory = self.episodic_sensory_proj(full_emb.mean(dim=1)).float()
            ret_mem, _ = episodic_memory.read(q_sensory, temperature=0.05, threshold=0.65, sigmoid_beta=15.0)
            ret_mem_unrolled = ret_mem.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous().view(batch_size * seq_len, -1).float()
            unrolled_inputs['episodic_recall'] = ret_mem_unrolled
        else:
            unrolled_inputs['episodic_recall'] = torch.zeros(batch_size * seq_len, self.unified_dim, device=self.device).float()

        h_prev_unrolled = torch.zeros(batch_size * seq_len, self.hidden_dim, device=self.device).float()
        u_t_unrolled = curr_u_t.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous().view(batch_size * seq_len, -1).float()
        
        with torch.amp.autocast(device_type=self.device_str, enabled=False):
            w_t_unrolled, _, _, _ = self.gateway(unrolled_inputs, h_prev_unrolled, u_t_unrolled)
        
        w_t_seq = w_t_unrolled.view(batch_size, seq_len, self.unified_dim)
        
        with torch.amp.autocast(device_type=self.device_str, dtype=torch.float16, enabled=(self.device_str == 'cuda')):
            full_h_in = self.in_proj(w_t_seq)

            h_s1, m_s1, dt1 = self._stage1_forward(full_h_in, m_s1, curr_u_t)
            saliency_gate = self.boundary_detector(h_s1, text_seq)

            # PW-HPC Top-Down Prediction
            h_s2_prev_shifted = torch.zeros_like(h_s1)
            e1_weighted, h_s1_hat, mean_pi = self.pw_hpc_generator(h_s1, h_s2_prev_shifted, curr_u_t)

            # EXP-104: Apply Theta-Gamma PAC Modulation Bridge
            e1_pac, phase_norm, gamma_mean = self.pac_bridge(e1_weighted, saliency_gate, curr_u_t)

            predicted_entropy = self.entropy_predictor(h_s1)
            dynamic_dt_scale = 0.40 + 1.20 * predicted_entropy

            h_s2, m_s2, dt2 = self._stage2_forward(e1_pac, m_s2, curr_u_t, saliency_gate, dynamic_dt_scale.mean().item())
            h_s2 = h_s2 * dynamic_dt_scale

            # Hierarchical Volitional Override
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
            hpc_reconstruction_loss = F.mse_loss(h_s1, h_s1_hat)
            fe_loss_tensor = (kl_div.mean() + rec_loss + 0.10 * hpc_reconstruction_loss)

            num_chunks = seq_len // chunk_size
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
        return total_loss_tensor, speech_loss_val, fe_loss_val, m_s2, h_proxy, curr_u_t, eff_dt, gamma_mean.item()


# =============================================================================
# 3. DATASET PREPARATION
# =============================================================================
def prepare_packed_stream(num_batches: int = 150, batch_size: int = 12, seq_len: int = 1024):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-104 (S={seq_len}, Steps={num_batches})...")
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
# 4. BENCHMARK RUNNER
# =============================================================================
def run_exp_104_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-104 (THETA-GAMMA PAC LAMINAR BRIDGE)] ===")
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

    # -------------------------------------------------------------------------
    # PHASE 1: BASELINE (CoREAgent v31.0 Master)
    # -------------------------------------------------------------------------
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 1: BASELINE (v31.0 Master) <<<")
    print("-"*85)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    agent_base = CoREAgent(config, device=device_str).to(device)
    hu_base = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_base = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_base = torch.optim.AdamW(agent_base.get_all_parameters(), lr=5e-4, weight_decay=0.01)
    crit_speech = nn.CrossEntropyLoss(ignore_index=256)

    t0 = time.perf_counter()
    base_losses = []
    base_fe = []

    for step, batch_tensors in enumerate(batches):
        inp = batch_tensors[:, :-1]
        tgt = batch_tensors[:, 1:]
        opt_base.zero_grad()

        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt = agent_base.forward_sequence(
                inp, tgt, hu_base, crit_speech, episodic_memory=mem_base, chunk_size=chunk_size
            )

        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_base.get_all_parameters(), max_norm=2.0)
        opt_base.step()

        base_losses.append(s_loss)
        base_fe.append(fe_loss)

        if (step + 1) % 50 == 0:
            print(f"  [Baseline Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | Free Energy: {fe_loss:.4f}")

    base_duration = time.perf_counter() - t0
    base_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    base_final_loss = sum(base_losses[-20:]) / 20.0
    base_final_fe = sum(base_fe[-20:]) / 20.0
    base_ppl = math.exp(min(base_final_loss, 20.0))
    base_tok_per_sec = (num_eval_steps * b_size * seq_len) / base_duration

    print(f"\n[Baseline Summary] Final Loss: {base_final_loss:.4f} | PPL: {base_ppl:.2f} | Free Energy: {base_final_fe:.4f} | Peak VRAM: {base_vram:.1f} MB | Speed: {base_tok_per_sec:.1f} tok/s")

    # Clean up baseline model to prevent OOM in Proposed Phase
    del agent_base, hu_base, mem_base, opt_base
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # PHASE 2: PROPOSED (Theta-Gamma PAC CoREAgent)
    # -------------------------------------------------------------------------
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 2: PROPOSED (Theta-Gamma PAC Laminar Bridge) <<<")
    print("-"*85)
    torch.cuda.reset_peak_memory_stats()

    agent_prop = ThetaGammaPACCoREAgent(config, device=device_str).to(device)
    hu_prop = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_prop = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_prop = torch.optim.AdamW(agent_prop.get_all_parameters(), lr=5e-4, weight_decay=0.01)

    t0 = time.perf_counter()
    prop_losses = []
    prop_fe = []
    gamma_means = []

    for step, batch_tensors in enumerate(batches):
        inp = batch_tensors[:, :-1]
        tgt = batch_tensors[:, 1:]
        opt_prop.zero_grad()

        sensor_dict = {'text': inp}
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt, g_mean = agent_prop.forward_multimodal_sequence(
                sensor_dict, tgt, hu_prop, crit_speech, episodic_memory=mem_prop, chunk_size=chunk_size
            )

        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_prop.get_all_parameters(), max_norm=2.0)
        opt_prop.step()

        prop_losses.append(s_loss)
        prop_fe.append(fe_loss)
        gamma_means.append(g_mean)

        if (step + 1) % 50 == 0:
            print(f"  [Proposed Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | Free Energy: {fe_loss:.4f} | Gamma Mean: {g_mean:.4f}")

    prop_duration = time.perf_counter() - t0
    prop_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    prop_final_loss = sum(prop_losses[-20:]) / 20.0
    prop_final_fe = sum(prop_fe[-20:]) / 20.0
    prop_final_gamma = sum(gamma_means[-20:]) / 20.0
    prop_ppl = math.exp(min(prop_final_loss, 20.0))
    prop_tok_per_sec = (num_eval_steps * b_size * seq_len) / prop_duration

    print(f"\n[Proposed Summary] Final Loss: {prop_final_loss:.4f} | PPL: {prop_ppl:.2f} | Free Energy: {prop_final_fe:.4f} | Gamma Mean: {prop_final_gamma:.4f} | Peak VRAM: {prop_vram:.1f} MB | Speed: {prop_tok_per_sec:.1f} tok/s")

    # -------------------------------------------------------------------------
    # GRADIENT FLOW INSPECTION
    # -------------------------------------------------------------------------
    print("\n" + "="*85)
    print(" === [KEP RULE #6 GRADIENT FLOW AUDIT (EXP-104)] ===")
    print("="*85)
    healthy_grads = 0
    zero_grads = 0

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
        if "pac_bridge" in name:
            print(f"  {name:<55} | Grad Norm: {g_norm if param.grad is not None else 0.0:<12.6f} | {status}")

    print("-" * 85)
    print(f"Audit Summary: Healthy Gradients: {healthy_grads} | Disconnected: {zero_grads}")

    # -------------------------------------------------------------------------
    # KEP DECISION EVALUATION
    # -------------------------------------------------------------------------
    delta_loss = base_final_loss - prop_final_loss
    delta_fe = base_final_fe - prop_final_fe
    speed_retention = (prop_tok_per_sec / base_tok_per_sec) * 100.0

    print("\n" + "="*85)
    print(" === [KEP RULE #2 FINAL EMPIRICAL DECISION EVALUATION] ===")
    print("="*85)
    print(f"Baseline Loss      : {base_final_loss:.4f} (PPL: {base_ppl:.2f}, FE: {base_final_fe:.4f}) | VRAM: {base_vram:.1f} MB | Speed: {base_tok_per_sec:.1f} tok/s")
    print(f"Proposed Loss      : {prop_final_loss:.4f} (PPL: {prop_ppl:.2f}, FE: {prop_final_fe:.4f}) | VRAM: {prop_vram:.1f} MB | Speed: {prop_tok_per_sec:.1f} tok/s")
    print(f"Delta Speech Loss  : {delta_loss:+.4f} nats")
    print(f"Delta Free Energy  : {delta_fe:+.4f}")
    print(f"Throughput Speed   : {prop_tok_per_sec:.1f} tok/s ({speed_retention:.1f}% retention)")

    if delta_loss >= 0.08 and speed_retention >= 80.0:
        verdict = "🟢 POSITIVE"
    elif delta_loss <= -0.08:
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
        "base_fe": base_final_fe,
        "prop_fe": prop_final_fe,
        "base_ppl": base_ppl,
        "prop_ppl": prop_ppl,
        "base_tok_per_sec": base_tok_per_sec,
        "prop_tok_per_sec": prop_tok_per_sec,
        "speed_retention_pct": speed_retention,
        "base_vram_mb": base_vram,
        "prop_vram_mb": prop_vram
    }


if __name__ == "__main__":
    run_exp_104_benchmark()
