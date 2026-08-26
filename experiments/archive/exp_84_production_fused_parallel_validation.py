# experiments/exp_84_production_fused_parallel_validation.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-84 (PRODUCTION FUSED CHUNK-PARALLEL INTEGRATION)
Validates the full production integration of the Native C++20 Fused Chunk-Parallel
2-Stage Cortical Stack with PW-LPER, EABS Boundary Detector, and System 2 Sandbox
in karyon_agent.py CoREAgent.
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import os
import time
import math
import types
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

# Add project root to path
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
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-84 (S={seq_len}, Steps={num_batches})...")
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


def run_exp_84_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-84 (PRODUCTION FUSED PARALLEL INTEGRATION)] ===")
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

    print("\n[1/1] Initializing Production CoREAgent with Native C++ Fused Cortical Core (300 Steps)...")
    torch.manual_seed(42)
    agent = CoREAgent(config=config, device=device_str).to(device)
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

    for step in range(num_eval_steps):
        t_batch_start = time.perf_counter()
        batch = batches[step]
        input_s = batch[:, :-1]
        target_s = batch[:, 1:]

        optimizer.zero_grad()
        t_fwd_0 = time.perf_counter()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, speech_loss, fe_val, m_curr, h_proxy, curr_u_t, eff_dt = agent.forward_sequence(
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

        t_step_total = (time.perf_counter() - t_batch_start) * 1000.0
        tok_sec = (b_size * seq_len) / (t_step_total / 1000.0)

        # KEP PROCESS DIAGNOSTICS DASHBOARD (EVERY 15 STEPS)
        if (step + 1) % 15 == 0 or step == num_eval_steps - 1:
            cur_loss = sum(losses[-15:]) / min(len(losses), 15)
            cur_fe = sum(fe_list[-15:]) / min(len(fe_list), 15)
            cur_lr = optimizer.param_groups[0]['lr']
            cur_ppl = math.exp(min(cur_loss, 20.0))

            curiosity, energy, stability, health, na, da = hu.state[0].tolist()
            peak_vram = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0

            g_emb = agent.pos_embeddings.byte_embed.weight.grad.norm().item() if agent.pos_embeddings.byte_embed.weight.grad is not None else 0.0
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
            print(f" === [KEP EXP-84 PROCESS DIAGNOSTICS DASHBOARD | STEP {step+1:03d}/{num_eval_steps}] ===")
            print("="*95)
            print(f"Metrics Progress          : Speech Loss = {speech_loss:.4f} (Avg: {cur_loss:.4f}, PPL: {cur_ppl:.2f}) | LR = {cur_lr:.6f}")
            print(f"Active Inference Dynamics : Free Energy = {fe_val:.4f} (Avg: {cur_fe:.4f})")
            print(f"Timing & Throughput       : Forward: {t_fwd:.1f}ms | Backward: {t_bwd:.1f}ms | Total Step: {t_step_total:.1f}ms | {tok_sec:.1f} tok/s")
            dt_eff_val = eff_dt.mean().item() if isinstance(eff_dt, torch.Tensor) else float(eff_dt)
            print(f"Somatic State (Ashby)     : Curiosity: {curiosity:.3f} | Energy: {energy:.3f} | NA: {na:.3f} | DA: {da:.3f} | dt_eff: {dt_eff_val:.3f}")
            print(f"Gradient Flow Inspection  : Total: {grad_norm_total:.4f} | Emb: {g_emb:.4f} | PW-LPER: {g_pw:.4f} | S1: {g_s1:.4f} | S2: {g_s2:.4f}")
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

    print("\n" + "="*95)
    print(" === [KEP EXP-84 FINAL TELEMETRY DASHBOARD] ===")
    print("="*95)
    print(f"{'Performance Metric':<36} | {'EXP-84 Production CoREAgent Value':<40}")
    print("-" * 95)
    print(f"{'Initial Loss (Step 1)':<36} | {losses[0]:<40.4f}")
    print(f"{'Step 100 Loss':<36} | {losses[99]:<40.4f}")
    print(f"{'Step 200 Loss':<36} | {losses[199]:<40.4f}")
    print(f"{'Final Steady-State Speech Loss':<36} | {final_loss:<40.4f} (PPL: {math.exp(final_loss):.2f})")
    print(f"{'Variational Free Energy (F_t)':<36} | {final_fe:<40.4f}")
    print(f"{'Total Loss Drop (Delta)':<36} | {final_loss - losses[0]:<40.4f}")
    print(f"{'Throughput Speed':<36} | {(num_eval_steps * b_size * seq_len) / total_time_sec:<40.1f} tok/s")
    print(f"{'Total Training Time':<36} | {total_time_sec:<40.2f} sec ({total_time_sec/60.0:.1f} min)")
    print("="*95)

    print("\n" + "="*95)
    print(" === [KEP RULE #4 FINAL DIAGNOSTIC SPEECH SAMPLE AUDIT] ===")
    print("="*95)
    print(f"Prompt : \"{diag_prompt}\"")
    print(f"Output : \"{final_sample}\"")
    print("="*95 + "\n")

    return final_loss, (num_eval_steps * b_size * seq_len) / total_time_sec, final_sample


if __name__ == "__main__":
    run_exp_84_benchmark()
