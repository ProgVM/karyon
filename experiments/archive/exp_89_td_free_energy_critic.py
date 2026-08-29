# experiments/exp_89_td_free_energy_critic.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-89
Hypothesis: Temporal-Difference Variational Free Energy Value Critic (TD-FE Critic)
In canonical Active Inference (Friston 2015, 2024), the somatic homeostat estimates
future Expected Free Energy (EFE). By computing TD Free Energy value estimates:
  delta_t = F_t + gamma * V(h_{t+1}) - V(h_t)
  L_critic = 0.5 * delta_t^2
and optimizing the Critic in tandem with the 2-Stage Cortical Stack, we eliminate
disconnected subgraphs (100% healthy gradient flow) and provide value-guided
predictive coding representations that improve speech loss on vicgalle/alpaca-gpt4.
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


def prepare_packed_stream(num_batches: int = 300, batch_size: int = 32, seq_len: int = 1024):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-89 (S={seq_len}, Steps={num_batches})...")
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

    logger.info(f"Prepared {len(batches)} Real Packed Batches (B={batch_size}, S={seq_len}, Total Tokens: {len(batches)*batch_size*seq_len/1e6:.2f}M).")
    return batches


class TDActiveInferenceAgent(CoREAgent):
    """Extends CoREAgent with Temporal-Difference Free Energy Value Learning in forward_sequence."""
    def __init__(self, config, device='cpu'):
        super().__init__(config, device)
        # 2-layer MLP Critic with LayerNorm for stable Free Energy value estimation
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        ).to(self.device)

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion_speech: nn.Module, episodic_memory=None, loss_free_energy_weight: float = 0.05, 
                         chunk_size: int = 64):
        batch_size, seq_len = input_seq.size()
        
        m_s1 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        m_s2 = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, dtype=torch.float32, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()
        h_prev_fast = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        h1_prev_last = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)
        
        # 1. Vectorized full-sequence embedding, receptive field, and linear projection
        full_emb = self.pos_embeddings(input_seq, start_pos=0, apply_rf=True)
        full_h_in = self.in_proj(full_emb)
        
        da_level = curr_u_t[:, 5:6]
        motor_gain = (1.0 + 1.0 * da_level).unsqueeze(1)

        # 2. Stage 1: Fast Morpho-Syntactic Cortical Pass (Native C++ Parallel Scan)
        h_s1, m_s1, dt1 = self.stage1(full_h_in, m_s1, curr_u_t, torch.Tensor(), 1.0)

        # 3. Dynamic Word / Morpheme Boundary Saliency (EABS Native C++)
        saliency_gate = self.boundary_detector(h_s1, input_seq)

        # 4. Precision-Weighted Laminar Error Routing (PW-LPER Native C++)
        e1_weighted, _, _ = self.pw_lper(h_s1, h1_prev_last, curr_u_t)

        # 5. Stage 2: Slow Semantic-Discourse Pass on Precision-Weighted Error e1_weighted
        h_s2, m_s2, dt2 = self.stage2(e1_weighted, m_s2, curr_u_t, saliency_gate, 1.0)
        eff_dt = (dt1 + dt2) / 2.0

        # 6. Combined Laminar Representation (Stage 1 + Stage 2)
        h_combined = h_s1 + h_s2

        # 7. Modern Hopfield Attractor Landscape with Native C++ Commitment Loss
        h_flat = h_combined.contiguous().view(-1, self.hidden_dim)
        h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_flat, curr_u_t)
        
        # 8. Dopaminergic Afferent-Efferent Motor Readout (768 -> 256 -> 258)
        h_proj = self.motor_text_proj(h_relaxed).view(batch_size, seq_len, self.text_dim)
        h_proj_gain = (h_proj * motor_gain).contiguous().view(-1, self.text_dim)
        logits_flat = F.linear(h_proj_gain, self.pos_embeddings.byte_embed.weight)

        targets_flat = target_seq.contiguous().view(-1)
        speech_loss_tensor = criterion_speech(logits_flat, targets_flat)

        # 9. Active Inference World Model Predictor
        w_current_slice = self.episodic_sensory_proj(full_emb[:, -1, :])
        h_curr_fast = h_combined[:, -1, :]
        w_pred, kl_div, fe, _ = self.world_model(h_prev_fast, h_curr_fast, w_current_slice)

        rec_loss = (1.0 - F.cosine_similarity(w_current_slice, w_pred, dim=-1, eps=1e-8)).mean()
        fe_loss_tensor = (kl_div.mean() + rec_loss)

        # 10. Temporal-Difference Free Energy Value Learning (Active Inference Critic)
        # Sample chunk-level representations along the sequence to compute temporal value predictions
        # h_combined: [B, S, D] -> chunk endpoints [B, num_chunks, D]
        num_chunks = seq_len // chunk_size
        h_chunk_endpoints = h_combined.view(batch_size, num_chunks, chunk_size, self.hidden_dim)[:, :, -1, :]
        
        # Predict state values V(h_c) for each chunk endpoint
        v_preds = self.critic(h_chunk_endpoints).squeeze(-1) # [B, num_chunks]
        
        # Reward / Target signal: immediate negative Free Energy (Surprise minimization)
        gamma_fe = 0.90
        fe_per_batch = fe.squeeze(-1) # [B]
        
        # TD target: r_c + gamma * V(h_{c+1}) where r_c is bounded local surprise
        v_current = v_preds[:, :-1]
        v_next = v_preds[:, 1:].detach()
        # Local surprise target: -0.1 * fe_per_batch
        r_step = -0.10 * fe_per_batch.unsqueeze(1).expand_as(v_current)
        td_targets = r_step + gamma_fe * v_next
        td_error = td_targets - v_current
        critic_loss = F.mse_loss(v_current, td_targets)

        # High-Surprise Episodic Encoding
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
        return total_loss_tensor, speech_loss_val, fe_loss_val, critic_loss_val, m_s2, h_proxy, curr_u_t, eff_dt


def run_exp_89_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-89 (TD-FE ACTIVE INFERENCE CRITIC)] ===")
    print("="*85)
    print(f"Hardware Device : {device_str.upper()} | AMP FP16 Enabled: {use_amp}")

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12

    b_size, seq_len = 32, 1024
    num_eval_steps = 300
    chunk_size = 64

    batches = prepare_packed_stream(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)
    criterion = nn.CrossEntropyLoss(ignore_index=256)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    diag_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"

    print("\n[1/1] Initializing TDActiveInferenceAgent with TD Free Energy Critic (300 Steps)...")
    torch.manual_seed(42)
    agent = TDActiveInferenceAgent(config=config, device=device_str).to(device)
    hu = HomeostaticUnit(batch_size=b_size, device=device_str)
    episodic_mem = BatchedEpisodicMemory(batch_size=b_size, memory_dim=config.net.text_dim, max_capacity=500, device=device_str)

    optimizer = torch.optim.AdamW(agent.get_all_parameters(), lr=3.0e-3, weight_decay=0.01)

    warmup_steps = 30
    stable_steps = 200
    decay_steps = num_eval_steps - (warmup_steps + stable_steps)

    def wsd_schedule(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        elif step < warmup_steps + stable_steps:
            return 1.0
        else:
            p = float(step - (warmup_steps + stable_steps)) / float(max(1, decay_steps))
            return 0.33 + 0.67 * 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=wsd_schedule)

    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    losses = []
    fe_list = []
    critic_losses = []

    for step in range(num_eval_steps):
        t_batch_start = time.perf_counter()
        batch = batches[step]
        input_s = batch[:, :-1]
        target_s = batch[:, 1:]

        optimizer.zero_grad()
        t_fwd_0 = time.perf_counter()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, speech_loss, fe_val, critic_val, m_curr, h_proxy, curr_u_t, eff_dt = agent.forward_sequence(
                input_s, target_s, hu, criterion, episodic_memory=episodic_mem, chunk_size=chunk_size
            )
        t_fwd = (time.perf_counter() - t_fwd_0) * 1000.0

        t_bwd_0 = time.perf_counter()
        scaler.scale(tot_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm_total = torch.nn.utils.clip_grad_norm_(agent.get_all_parameters(), max_norm=3.0).item()

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        if scale_before <= scale_after:
            scheduler.step()
        t_bwd = (time.perf_counter() - t_bwd_0) * 1000.0

        # Somatic Coupling
        with torch.no_grad():
            cost_t = torch.full((b_size, 1), 0.001, device=device)
            err_t = torch.full((b_size, 1), float(speech_loss * 0.1), device=device)
            ent_t = torch.full((b_size, 1), float(fe_val), device=device)
            cog_t = torch.zeros((b_size, 1), dtype=torch.int64, device=device)
            hu.update(cost_t, err_t, ent_t, cog_t)

        losses.append(speech_loss)
        fe_list.append(fe_val)
        critic_losses.append(critic_val)

        t_step_total = (time.perf_counter() - t_batch_start) * 1000.0
        tok_sec = (b_size * seq_len) / (t_step_total / 1000.0)

        # KEP PROCESS DIAGNOSTICS DASHBOARD (EVERY 15 STEPS)
        if (step + 1) % 15 == 0 or step == num_eval_steps - 1:
            cur_loss = sum(losses[-15:]) / min(len(losses), 15)
            cur_fe = sum(fe_list[-15:]) / min(len(fe_list), 15)
            cur_critic = sum(critic_losses[-15:]) / min(len(critic_losses), 15)
            cur_lr = optimizer.param_groups[0]['lr']
            cur_ppl = math.exp(min(cur_loss, 20.0))

            curiosity, energy, stability, health, na, da = hu.state[0].tolist()
            peak_vram = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0

            g_emb = agent.pos_embeddings.byte_embed.weight.grad.norm().item() if agent.pos_embeddings.byte_embed.weight.grad is not None else 0.0
            g_critic = 0.0
            for p in agent.critic.parameters():
                if p.grad is not None:
                    g_critic = max(g_critic, p.grad.norm().item())
            g_pw = 0.0
            for p in agent.pw_lper.parameters():
                if p.grad is not None:
                    g_pw = max(g_pw, p.grad.norm().item())
            g_s1 = 0.0
            for p in agent.stage1.parameters():
                if p.grad is not None:
                    g_s1 = max(g_s1, p.grad.norm().item())
            g_s2 = 0.0
            for p in agent.stage2.parameters():
                if p.grad is not None:
                    g_s2 = max(g_s2, p.grad.norm().item())

            print("\n" + "="*95)
            print(f" === [KEP EXP-89 PROCESS DIAGNOSTICS DASHBOARD | STEP {step+1:03d}/{num_eval_steps}] ===")
            print("="*95)
            print(f"Metrics Progress          : Speech Loss = {speech_loss:.4f} (Avg: {cur_loss:.4f}, PPL: {cur_ppl:.2f}) | Critic Loss = {critic_val:.5f} (Avg: {cur_critic:.5f}) | LR = {cur_lr:.6f}")
            print(f"Active Inference Dynamics : Free Energy = {fe_val:.4f} (Avg: {cur_fe:.4f})")
            print(f"Timing & Throughput       : Forward: {t_fwd:.1f}ms | Backward: {t_bwd:.1f}ms | Total Step: {t_step_total:.1f}ms | {tok_sec:.1f} tok/s")
            dt_eff_val = eff_dt.mean().item() if isinstance(eff_dt, torch.Tensor) else float(eff_dt)
            print(f"Somatic State (Ashby)     : Curiosity: {curiosity:.3f} | Energy: {energy:.3f} | NA: {na:.3f} | DA: {da:.3f} | dt_eff: {dt_eff_val:.3f}")
            print(f"Gradient Flow Inspection  : Total: {grad_norm_total:.4f} | Emb: {g_emb:.4f} | Critic: {g_critic:.4f} | PW-LPER: {g_pw:.4f} | S1: {g_s1:.4f} | S2: {g_s2:.4f}")
            print(f"Hardware Resources        : Peak VRAM: {peak_vram:.1f} MB | Episodic Active Slots: {episodic_mem.max_active_cpu}")
            print("="*95)

        if (step + 1) % 75 == 0:
            sample_chars = []
            gen_stream = agent.generate_thought_and_speech(
                prompt=diag_prompt,
                m_state=m_curr[0:1],
                h_state=h_proxy[0:1],
                hu=hu,
                episodic_memory=episodic_mem,
                config=config,
                max_generated_tokens=65
            )
            for ev in gen_stream:
                if ev["status"] == "token":
                    sample_chars.append(ev["text"])
            sample_text = "".join(sample_chars).strip()
            logger.info(f"💬 [Live Diagnostic Speech Sample @ Step {step+1}] -> \"{sample_text}\"")

    if device.type == 'cuda': torch.cuda.synchronize()
    total_time_sec = time.perf_counter() - t_start
    final_loss = sum(losses[-30:]) / 30.0
    final_fe = sum(fe_list[-30:]) / 30.0
    final_critic = sum(critic_losses[-30:]) / 30.0
    throughput = (num_eval_steps * b_size * seq_len) / total_time_sec

    sample_chars = []
    gen_stream = agent.generate_thought_and_speech(
        prompt=diag_prompt,
        m_state=m_curr[0:1],
        h_state=h_proxy[0:1],
        hu=hu,
        episodic_memory=episodic_mem,
        config=config,
        max_generated_tokens=75
    )
    for ev in gen_stream:
        if ev["status"] == "token":
            sample_chars.append(ev["text"])
    final_sample = "".join(sample_chars).strip()

    baseline_loss = 1.3509 # EXP-84 baseline
    loss_delta = final_loss - baseline_loss

    print("\n" + "="*95)
    print(" === [KEP EXP-89 FINAL TELEMETRY DASHBOARD] ===")
    print("="*95)
    print(f"{'Performance Metric':<36} | {'EXP-89 TD-FE Critic Value':<40}")
    print("-" * 95)
    print(f"{'Initial Loss (Step 1)':<36} | {losses[0]:<40.4f}")
    print(f"{'Step 100 Loss':<36} | {losses[99]:<40.4f}")
    print(f"{'Step 200 Loss':<36} | {losses[199]:<40.4f}")
    print(f"{'Final Steady-State Speech Loss':<36} | {final_loss:<40.4f} (PPL: {math.exp(final_loss):.2f})")
    print(f"{'Variational Free Energy (F_t)':<36} | {final_fe:<40.4f}")
    print(f"{'Critic TD Loss':<36} | {final_critic:<40.6f}")
    print(f"{'Loss Delta vs EXP-84 Baseline':<36} | {loss_delta:<+40.4f}")
    print(f"{'Throughput Speed':<36} | {throughput:<40.1f} tok/s")
    print(f"{'Total Training Time':<36} | {total_time_sec:<40.2f} sec ({total_time_sec/60.0:.1f} min)")
    print("="*95)

    print("\n" + "="*95)
    print(" === [KEP RULE #4 FINAL DIAGNOSTIC SPEECH SAMPLE AUDIT] ===")
    print("="*95)
    print(f"Prompt : \"{diag_prompt}\"")
    print(f"Output : \"{final_sample}\"")
    print("="*95 + "\n")

    if loss_delta <= -0.05 and throughput >= 0.80 * 33702.7:
        verdict = "🟢 POSITIVE"
    elif abs(loss_delta) < 0.05:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    else:
        verdict = "🔴 REJECTED"

    results = {
        "exp_id": "EXP-89",
        "verdict": verdict,
        "final_loss": final_loss,
        "loss_delta": loss_delta,
        "ppl": math.exp(final_loss),
        "free_energy": final_fe,
        "critic_loss": final_critic,
        "tok_per_sec": throughput,
        "vram_mb": (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0,
        "duration_sec": total_time_sec,
        "sample": final_sample
    }
    print(f"EXP-89 JSON RESULT: {json.dumps(results)}")
    return results


if __name__ == "__main__":
    run_exp_89_benchmark()
