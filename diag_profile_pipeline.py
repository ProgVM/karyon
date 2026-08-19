# diag_profile_pipeline.py
"""
===============================================================================
KARYON PIPELINE MASTER DIAGNOSTIC & PROFILER v2.1
Detailed Microsecond CUDA Event Profiling, Gradient Flow Verification,
VRAM Memory Inspection, and Real-Time Speech Sample Diagnostics.
Author: Karyon-CoRE Open-Source Research Team (2026)
===============================================================================
"""

import sys
import types
import time
import math
import importlib
import torch

# Unconditional PyTorch Dynamo Hotfix for Kaggle / Python 3.12
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

import torch.nn as nn
import torch.nn.functional as F

import karyon_config, karyon_core, karyon_agent, karyon_logger
importlib.reload(karyon_agent)

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory

torch.set_grad_enabled(True)
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
print(f"[KEP Master Profiler v2.1] Running CUDA Pipeline Diagnostic on: {device_str.upper()}")


# =============================================================================
# DETAILED CUDA PIPELINE PROFILER ENGINE
# =============================================================================

def profile_master_karyon_pipeline():
    config = CoREConfig()
    config.net.hidden_dim = 512
    config.net.unified_dim = 256
    config.net.text_dim = 128
    
    agent = CoREAgent(config=config, device=device_str).to(device)
    hu = HomeostaticUnit(batch_size=32, device=device_str)
    episodic_mem = BatchedEpisodicMemory(batch_size=32, memory_dim=256, max_capacity=200, device=device_str)
    optimizer = torch.optim.Adam(agent.get_all_parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=256)

    # Synthetic batch: B=32, seq_len=512
    b_size, seq_len = 32, 512
    dummy_tokens = torch.randint(0, 255, (b_size, seq_len), dtype=torch.long, device=device)
    target_tokens = torch.randint(0, 255, (b_size, seq_len), dtype=torch.long, device=device)

    start_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None

    # Timing accumulators (in ms)
    time_embeddings = 0.0
    time_gateway = 0.0
    time_core_sde = 0.0
    time_world_model = 0.0
    time_attractor_head = 0.0
    time_motor_gateway = 0.0
    time_backward_pass = 0.0

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    print("\n" + "="*85)
    print(" === MASTER PROFILING: 512-STEP CHUNKED BPTT FORWARD & BACKWARD PASS ===")
    print("="*80)

    t_batch_start = time.time()
    optimizer.zero_grad()
    
    # 1. Measure Embedding Pass
    if start_event: start_event.record()
    full_emb = agent.pos_embeddings(dummy_tokens)
    if end_event:
        end_event.record()
        torch.cuda.synchronize()
        time_embeddings = start_event.elapsed_time(end_event)

    h_f = torch.zeros(b_size, 512, device=device)
    h_s = torch.zeros(b_size, 512, device=device)
    obs_vis = torch.zeros(b_size, 256, device=device)
    prev_act = torch.zeros(b_size, 3, device=device)
    u_t = hu.state.clone()

    losses = []

    # 2. Profile Step-by-Step Submodule Execution
    for t in range(seq_len):
        t_emb = full_emb[:, t]
        target_t = target_tokens[:, t]

        # Gateway
        t0 = time.perf_counter()
        w_curr, attn, names, eps_ent = agent.gateway(t_emb, obs_vis, prev_act, h_f, u_t)
        time_gateway += (time.perf_counter() - t0) * 1000.0

        # SDE Core
        t0 = time.perf_counter()
        core_out = agent.core(h_f, h_s, w_curr, u_t, 1.0)
        h_f, h_s = core_out[0], core_out[1]
        time_core_sde += (time.perf_counter() - t0) * 1000.0

        # World Model
        t0 = time.perf_counter()
        w_pred, kl_div, fe, _ = agent.world_model(h_f, h_s, w_curr)
        time_world_model += (time.perf_counter() - t0) * 1000.0

        # Energy Attractor Head
        t0 = time.perf_counter()
        h_integrated = h_f + h_s
        h_relaxed, basin_energy = agent.attractor_head.relax_to_minima(h_integrated)
        time_attractor_head += (time.perf_counter() - t0) * 1000.0

        # Motor Gateway
        t0 = time.perf_counter()
        outputs = agent.output_gateway(h_relaxed)
        logits = outputs["text_generation"]
        time_motor_gateway += (time.perf_counter() - t0) * 1000.0

        loss_t = criterion(logits, target_t)
        losses.append(loss_t)

        # Truncate graph every 32 steps (32-step BPTT chunking)
        if (t + 1) % 32 == 0:
            h_f = h_f.detach()
            h_s = h_s.detach()

    total_loss = torch.stack(losses).mean()

    # 3. Measure Backward Pass (32-Step Chunked Graph)
    if start_event: start_event.record()
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if end_event:
        end_event.record()
        torch.cuda.synchronize()
        time_backward_pass = start_event.elapsed_time(end_event)

    total_batch_time_sec = time.time() - t_batch_start
    total_batch_time_ms = total_batch_time_sec * 1000.0

    # 4. Inspect Gradient Flow Norms across ALL Submodules
    grad_norms = {}
    for name, param in agent.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
        else:
            grad_norms[name] = 0.0

    # Peak VRAM
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0

    # =========================================================================
    # 5. DIAGNOSTIC TIMING BREAKDOWN TABLE
    # =========================================================================
    print(f"\n{'Submodule Component':<42} | {'Time (ms)':<14} | {'% of Batch':<12}")
    print("-" * 75)
    print(f"{'1. Positional Byte Embeddings':<42} | {time_embeddings:<14.2f} | {time_embeddings/total_batch_time_ms*100:<12.1f}%")
    print(f"{'2. Sensory Gateway (C++ JIT)':<42} | {time_gateway:<14.2f} | {time_gateway/total_batch_time_ms*100:<12.1f}%")
    print(f"{'3. Dynamic Recurrent SDE Core (C++ JIT)':<42} | {time_core_sde:<14.2f} | {time_core_sde/total_batch_time_ms*100:<12.1f}%")
    print(f"{'4. Latent Predictor World Model (C++)':<42} | {time_world_model:<14.2f} | {time_world_model/total_batch_time_ms*100:<12.1f}%")
    print(f"{'5. Energy Attractor Head (Analytic Relaxation)':<42} | {time_attractor_head:<14.2f} | {time_attractor_head/total_batch_time_ms*100:<12.1f}%")
    print(f"{'6. Motor Gateway (C++ JIT)':<42} | {time_motor_gateway:<14.2f} | {time_motor_gateway/total_batch_time_ms*100:<12.1f}%")
    print(f"{'7. Chunked Autograd Backward Pass (.backward())':<42} | {time_backward_pass:<14.2f} | {time_backward_pass/total_batch_time_ms*100:<12.1f}%")
    print("="*75)
    print(f"{'TOTAL BATCH TIME':<42} | {total_batch_time_ms:<14.2f} ms ({total_batch_time_sec:.2f} sec)")
    print(f"{'PEAK VRAM MEMORY ALLOCATED':<42} | {peak_vram_mb:<14.2f} MB")
    print(f"{'THROUGHPUT SPEED':<42} | {(b_size * seq_len) / total_batch_time_sec:<14.1f} tokens/sec")
    print("="*75)

    # =========================================================================
    # 6. GRADIENT FLOW VERIFICATION DASHBOARD
    # =========================================================================
    print("\n" + "="*85)
    print(" === GRADIENT FLOW VERIFICATION DASHBOARD (SUBMODULE PARAMETER NORMS) ===")
    print("="*85)
    print(f"{'Parameter Name':<50} | {'Grad Norm':<14} | {'Status':<15}")
    print("-" * 85)
    
    zero_grad_count = 0
    healthy_grad_count = 0

    for name, g_norm in grad_norms.items():
        if g_norm == 0.0:
            status_flag = "⚠️ ZERO GRADIENT"
            zero_grad_count += 1
        else:
            status_flag = "✅ HEALTHY"
            healthy_grad_count += 1
        print(f"  {name:<48} | {g_norm:<14.6f} | {status_flag}")

    print("-" * 85)
    print(f"Total Parameters Audited: {len(grad_norms)} | Healthy: {healthy_grad_count} | Disconnected: {zero_grad_count}")
    print("="*85)

    # =========================================================================
    # 7. LIVE DIAGNOSTIC SPEECH SAMPLE GENERATION
    # =========================================================================
    print("\n" + "="*85)
    print(" === LIVE DIAGNOSTIC SPEECH SAMPLE VERIFICATION ===")
    print("="*85)
    
    agent.eval()
    diag_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"
    diag_hu = HomeostaticUnit(batch_size=1, device=device_str)
    diag_mem = BatchedEpisodicMemory(batch_size=1, memory_dim=256, max_capacity=200, device=device_str)

    sample_chars = []
    with torch.no_grad():
        gen_stream = agent.generate_thought_and_speech(
            prompt=diag_prompt,
            h_fast=torch.zeros(1, agent.hidden_dim, device=device),
            h_slow=torch.zeros(1, agent.hidden_dim, device=device),
            hu=diag_hu,
            episodic_memory=diag_mem,
            config=config,
            max_generated_tokens=40
        )
        for event in gen_stream:
            if event["status"] == "token":
                sample_chars.append(event["text"])

    print(f"  Prompt : \"{diag_prompt.strip()}\"")
    print(f"  Output : \"{''.join(sample_chars).strip()}\"")
    print("="*85 + "\n")

if __name__ == "__main__":
    profile_master_karyon_pipeline()
