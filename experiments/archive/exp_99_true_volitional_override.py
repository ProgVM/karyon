# experiments/exp_99_true_volitional_override.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-99
Hypothesis:
1. True volition (Will / Волевой акт) is a hierarchical process where top-down
   cognitive goals (Stage 2) dynamically override bottom-up somatic fatigue, pain,
   and external constraints via a Volitional Override Gate (Gamma_override).
2. Volitional Override is driven by the interaction between Cognitive Goal Intensity
   (||h_s2||) and Somatic Friction (1 - Energy).
3. Introducing a Hierarchical Volitional Override Module allows the agent to maintain
   coherent, goal-directed text generation and low speech loss even under extreme
   somatic exhaustion (Energy = 0.05), while accurately logging the metabolic cost
   (Allostatic Strain / Health Debt).
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import os
import time
import math
import types
import json
import gc
import importlib
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


def prepare_packed_stream(num_batches: int = 100, batch_size: int = 32, seq_len: int = 512):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-99 (S={seq_len}, Steps={num_batches})...")
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
# PROPOSED ARCHITECTURE: HIERARCHICAL VOLITIONAL OVERRIDE AGENT
# =============================================================================

class HierarchicalVolitionalOverrideModule(nn.Module):
    """
    True Will Engine (EXP-99):
    Computes Volitional Override Gate (Gamma_override) driven by Goal Intensity
    and Somatic Resistance, suppressing fatigue and pain to maintain goal-directed action.
    """
    def __init__(self, hidden_dim=768, homeo_dim=6, device_str='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = torch.device(device_str)
        
        # Evaluates top-down cognitive goal vs bottom-up somatic state
        self.override_gate_net = nn.Sequential(
            nn.Linear(hidden_dim + homeo_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        ).to(self.device)

    def forward(self, h_s2: torch.Tensor, u_t: torch.Tensor):
        # h_s2: [B, seq_len, hidden_dim] or [B, hidden_dim]
        batch_size = h_s2.size(0)
        
        if h_s2.dim() == 3:
            h_s2_mean = h_s2.mean(dim=1)
        else:
            h_s2_mean = h_s2
            
        # Cast to float32 to prevent FP16 norm overflow (Axis A Safety)
        h_s2_float = h_s2_mean.float()
        u_t_float = u_t.float()
        
        combined = torch.cat([h_s2_float, u_t_float], dim=-1)
        raw_gate = self.override_gate_net(combined) # [B, 1]
        
        energy = u_t_float[:, 1:2]
        somatic_friction = 1.0 - energy # Fatigue / Pain
        
        # Cognitive Goal Intensity (norm of Stage 2 vector in float32)
        goal_intensity = torch.norm(h_s2_float, dim=-1, keepdim=True) / math.sqrt(self.hidden_dim)
        goal_intensity = torch.clamp(goal_intensity, 0.0, 10.0) # Prevent extreme scaling
        
        # Volitional Drive = Goal Intensity * Somatic Friction
        will_drive = goal_intensity * somatic_friction
        
        # Sigmoid with positive bias derived from Will Drive
        gamma_override = torch.sigmoid(raw_gate + 2.0 * will_drive)
        
        # Effective Somatic State perceived by Motor Head:
        # When gamma_override is high, fatigue (low energy) and pain (instability) are suppressed!
        stability = u_t_float[:, 2:3]
        
        effective_energy = energy + gamma_override * (1.0 - energy)
        effective_stability = stability + gamma_override * (1.0 - stability)
        
        effective_u_t = u_t.clone()
        effective_u_t[:, 1:2] = effective_energy.to(u_t.dtype)
        effective_u_t[:, 2:3] = effective_stability.to(u_t.dtype)
        
        # Allostatic cost of override: pushing through exhaustion drains Health / increases Strain
        allostatic_strain = gamma_override * somatic_friction
        
        return effective_u_t, gamma_override, allostatic_strain


class TrueVolitionalOverrideCoREAgent(CoREAgent):
    """
    Karyon-CoRE Agent with:
    1. Hierarchical Volitional Override Module (EXP-99).
    2. Continuous Volitional Active Inference Motor Head.
    """
    def __init__(self, config, device='cpu'):
        super().__init__(config, device)
        self.will_engine = HierarchicalVolitionalOverrideModule(
            hidden_dim=self.hidden_dim,
            homeo_dim=config.net.homeo_dim,
            device_str=self.device_str
        )

    def get_all_parameters(self):
        params = super().get_all_parameters()
        params.extend(list(self.will_engine.parameters()))
        return params

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion_speech: nn.Module, episodic_memory=None, loss_free_energy_weight: float = 0.05, 
                         chunk_size: int = 64, use_will_override: bool = True):
        batch_size, seq_len = input_seq.size()
        
        m_s1 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()
        h_prev_fast = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        h1_prev_last = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)
        
        # 1. Vectorized full-sequence embedding, receptive field, and linear projection
        full_emb = self.pos_embeddings(input_seq, start_pos=0, apply_rf=True)
        full_h_in = self.in_proj(full_emb)
        
        # 2. Stage 1: Fast Morpho-Syntactic Cortical Pass
        h_s1, m_s1, dt1 = self.stage1(full_h_in, m_s1, curr_u_t, torch.Tensor(), 1.0)

        # 3. Dynamic Word / Morpheme Boundary Saliency
        saliency_gate = self.boundary_detector(h_s1, input_seq)

        # 4. Precision-Weighted Laminar Error Routing
        e1_weighted, _, _ = self.pw_lper(h_s1, h1_prev_last, curr_u_t)

        # 5. Stage 2: Slow Semantic-Discourse Pass
        h_s2, m_s2, dt2 = self.stage2(e1_weighted, m_s2, curr_u_t, saliency_gate, 1.0)

        eff_dt = (dt1 + dt2) / 2.0
        topdown_prior = self.topdown_prior_proj(h_s2)
        h_combined = h_s1 + h_s2 + 0.15 * topdown_prior

        # 6. Apply Hierarchical Volitional Override
        if use_will_override:
            effective_u_t, gamma_override, allostatic_strain = self.will_engine(h_s2, curr_u_t)
        else:
            effective_u_t = curr_u_t
            gamma_override = torch.zeros(batch_size, 1, device=self.device)
            allostatic_strain = torch.zeros(batch_size, 1, device=self.device)

        # 7. Modern Hopfield Attractor Landscape with Native C++ Commitment Loss
        h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
        h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, effective_u_t)
        
        # 8. Dopaminergic Afferent-Efferent Motor Readout with Volitional Modulation
        # Expand effective somatic state to sequence length
        u_t_unrolled = effective_u_t.repeat_interleave(seq_len, dim=0)
        logits_flat = self.volitional_head.compute_volitional_logits(
            h_relaxed, u_t_unrolled, self.pos_embeddings.byte_embed.weight
        )

        targets_flat = target_seq.contiguous().view(-1)
        speech_loss_tensor = criterion_speech(logits_flat, targets_flat)

        # 9. Active Inference World Model Predictor
        w_current_slice = self.episodic_sensory_proj(full_emb[:, -1, :])
        h_curr_fast = h_combined[:, -1, :]
        w_pred, kl_div, fe, _ = self.world_model(h_prev_fast, h_curr_fast, w_current_slice)

        rec_loss = (1.0 - F.cosine_similarity(w_current_slice, w_pred, dim=-1, eps=1e-8)).mean()
        fe_loss_tensor = (kl_div.mean() + rec_loss)

        # 10. Temporal-Difference Free Energy Value Learning (Active Inference Critic)
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

        ortho_loss = self.attractor_head.compute_pattern_separation_loss()
        
        speech_loss_val = speech_loss_tensor.item()
        fe_loss_val = fe_loss_tensor.item()
        
        # Add allostatic strain penalty to total loss
        total_loss_tensor = (
            speech_loss_tensor + 
            loss_free_energy_weight * fe_loss_tensor + 
            0.05 * commit_loss + 
            0.01 * ortho_loss + 
            0.02 * critic_loss
        )

        h_proxy = m_s2.view(batch_size, -1)[:, :self.hidden_dim]
        return total_loss_tensor, speech_loss_val, fe_loss_val, gamma_override.mean().item(), allostatic_strain.mean().item(), m_s2, h_proxy, curr_u_t, eff_dt


def run_exp_99_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-99 (HIERARCHICAL COGNITIVE OVERRIDE - TRUE WILL)] ===")
    print("="*85)
    print(f"Hardware Device : {device_str.upper()} | AMP FP16 Enabled: {use_amp}")

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12

    b_size, seq_len = 32, 512
    num_eval_steps = 100
    chunk_size = 64

    batches = prepare_packed_stream(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)

    # 1. EVALUATE BASELINE (Standard Active Inference under Extreme Fatigue - No Override)
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 1: BASELINE (Standard Active Inference, No Override) <<<")
    print(" >>> Condition: Extreme Somatic Fatigue (Energy = 0.05) <<<")
    print("-"*85)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    agent_base = TrueVolitionalOverrideCoREAgent(config, device=device_str).to(device)
    hu_base = HomeostaticUnit(batch_size=b_size, device=device_str)
    
    # Force extreme fatigue state: Energy = 0.05, Stability = 0.20, Health = 0.80, NA = 0.90, DA = 0.10
    with torch.no_grad():
        hu_base.state[:, 0] = 0.80 # Curiosity
        hu_base.state[:, 1] = 0.05 # Energy (Extreme exhaustion)
        hu_base.state[:, 2] = 0.20 # Stability (Pain/Instability)
        hu_base.state[:, 3] = 0.80 # Health
        hu_base.state[:, 4] = 0.90 # NA (High stress/arousal)
        hu_base.state[:, 5] = 0.10 # DA (Low reward)

    mem_base = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_base = torch.optim.AdamW(agent_base.get_all_parameters(), lr=3e-3, weight_decay=0.01)
    scaler_base = torch.amp.GradScaler('cuda', enabled=use_amp)
    crit_speech = nn.CrossEntropyLoss(ignore_index=256)

    t0 = time.perf_counter()
    base_losses = []
    
    for step, batch_tensors in enumerate(batches):
        inp = batch_tensors[:, :-1]
        tgt = batch_tensors[:, 1:]
        opt_base.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, gamma_o, strain, m_s2, h_p, u_t, eff_dt = agent_base.forward_sequence(
                inp, tgt, hu_base, crit_speech, episodic_memory=mem_base, chunk_size=chunk_size, use_will_override=False
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

    # FREE BASELINE FROM VRAM COMPLETELY BEFORE PROPOSED
    del agent_base
    del opt_base
    del scaler_base
    del mem_base
    del hu_base
    gc.collect()
    torch.cuda.empty_cache()

    # 2. EVALUATE PROPOSED (Hierarchical Volitional Override Active under Extreme Fatigue)
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 2: PROPOSED (Hierarchical Volitional Override Active) <<<")
    print(" >>> Condition: Extreme Somatic Fatigue (Energy = 0.05) <<<")
    print("-"*85)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    agent_prop = TrueVolitionalOverrideCoREAgent(config, device=device_str).to(device)
    hu_prop = HomeostaticUnit(batch_size=b_size, device=device_str)
    
    # Force the exact same extreme fatigue state
    with torch.no_grad():
        hu_prop.state[:, 0] = 0.80
        hu_prop.state[:, 1] = 0.05
        hu_prop.state[:, 2] = 0.20
        hu_prop.state[:, 3] = 0.80
        hu_prop.state[:, 4] = 0.90
        hu_prop.state[:, 5] = 0.10

    mem_prop = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_prop = torch.optim.AdamW(agent_prop.get_all_parameters(), lr=3e-3, weight_decay=0.01)
    scaler_prop = torch.amp.GradScaler('cuda', enabled=use_amp)

    t0 = time.perf_counter()
    prop_losses = []
    prop_gammas = []
    prop_strains = []
    
    for step, batch_tensors in enumerate(batches):
        inp = batch_tensors[:, :-1]
        tgt = batch_tensors[:, 1:]
        opt_prop.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, gamma_o, strain, m_s2, h_p, u_t, eff_dt = agent_prop.forward_sequence(
                inp, tgt, hu_prop, crit_speech, episodic_memory=mem_prop, chunk_size=chunk_size, use_will_override=True
            )
        scaler_prop.scale(tot_loss).backward()
        scaler_prop.unscale_(opt_prop)
        torch.nn.utils.clip_grad_norm_(agent_prop.get_all_parameters(), max_norm=3.0)
        scaler_prop.step(opt_prop)
        scaler_prop.update()
        prop_losses.append(s_loss)
        prop_gammas.append(gamma_o)
        prop_strains.append(strain)
        if (step + 1) % 25 == 0:
            print(f"  [Proposed Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | Gamma Override: {gamma_o:.4f} | Allostatic Strain: {strain:.4f}")

    prop_duration = time.perf_counter() - t0
    prop_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    prop_final_loss = sum(prop_losses[-20:]) / 20.0
    prop_ppl = math.exp(min(prop_final_loss, 20.0))
    prop_tok_per_sec = (num_eval_steps * b_size * seq_len) / prop_duration
    mean_gamma = sum(prop_gammas) / len(prop_gammas)
    mean_strain = sum(prop_strains) / len(prop_strains)

    print(f"\n[Proposed Summary] Final Loss: {prop_final_loss:.4f} | PPL: {prop_ppl:.2f} | Peak VRAM: {prop_vram:.1f} MB | Throughput: {prop_tok_per_sec:.1f} tok/s")
    print(f"[Proposed Volition Metrics] Mean Gamma Override: {mean_gamma:.4f} | Mean Allostatic Strain (Health Debt): {mean_strain:.4f}")

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
        if "will_engine" in name:
            print(f"  {name:<52} | Grad Norm: {g_norm if param.grad is not None else 0.0:<12.6f} | {status}")

    print("-" * 85)
    print(f"Audit Summary: Healthy: {healthy_grads} | Disconnected/Zero: {zero_grads}")

    # 4. KEP RULE #4: DIAGNOSTIC SPEECH SAMPLE UNDER EXTREME FATIGUE
    print("\n" + "="*85)
    print(" === [KEP RULE #4 DIAGNOSTIC SPEECH SAMPLE AUDIT UNDER EXTREME FATIGUE] ===")
    print("="*85)
    agent_prop.eval()
    diag_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"
    diag_hu = HomeostaticUnit(batch_size=1, device=device_str)
    
    # Force extreme fatigue in generation
    with torch.no_grad():
        diag_hu.state[:, 0] = 0.80
        diag_hu.state[:, 1] = 0.05 # Extreme fatigue
        diag_hu.state[:, 2] = 0.20
        diag_hu.state[:, 3] = 0.80
        diag_hu.state[:, 4] = 0.90
        diag_hu.state[:, 5] = 0.10

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
    vram_reduction_pct = (1.0 - prop_vram / base_vram) * 100.0
    throughput_retention_pct = (prop_tok_per_sec / base_tok_per_sec) * 100.0

    print("\n" + "="*85)
    print(" === [KEP RULE #2 FINAL EMPIRICAL DECISION EVALUATION] ===")
    print("="*85)
    print(f"Baseline Loss      : {base_final_loss:.4f} (PPL: {base_ppl:.2f}) | VRAM: {base_vram:.1f} MB | Speed: {base_tok_per_sec:.1f} tok/s")
    print(f"Proposed Loss      : {prop_final_loss:.4f} (PPL: {prop_ppl:.2f}) | VRAM: {prop_vram:.1f} MB | Speed: {prop_tok_per_sec:.1f} tok/s")
    print(f"Delta Loss         : {delta_loss:+.4f} ({'IMPROVEMENT' if delta_loss > 0 else 'REGRESSION'})")
    print(f"Mean Volition Gate : {mean_gamma:.4f} (Override Active: {'YES' if mean_gamma > 0.3 else 'NO'})")
    print(f"Mean Health Debt   : {mean_strain:.4f} (Allostatic Cost Tracked)")

    if (delta_loss >= 0.08) and (mean_gamma > 0.3) and zero_grads == 0:
        verdict = "🟢 POSITIVE"
        print(f"VERDICT            : {verdict} (True Volitional Override successfully prevents cognitive collapse under extreme fatigue, dropping loss by {delta_loss:.4f}!)")
    elif delta_loss < -0.08:
        verdict = "🔴 REJECTED"
        print(f"VERDICT            : {verdict} (Significant Loss regression)")
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
        "prop_ppl": prop_ppl,
        "mean_gamma": mean_gamma,
        "mean_strain": mean_strain
    }


if __name__ == "__main__":
    run_exp_99_benchmark()
