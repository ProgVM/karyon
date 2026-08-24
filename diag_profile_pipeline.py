# diag_profile_pipeline.py
"""
===============================================================================
KARYON PIPELINE MASTER DIAGNOSTIC & PROFILER v17.0
Detailed Microsecond CUDA Event Profiling, Unshackled Flow Inspection (256D -> 512D),
Gradient Flow Verification, VRAM Analysis, and Real-Time Speech Diagnostics.
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import types
import time
import math
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

# Unconditional PyTorch Dynamo Hotfix for Python 3.12 / Kaggle GPU
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

import karyon_config, karyon_core, karyon_agent, karyon_logger
importlib.reload(karyon_agent)

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory

torch.set_grad_enabled(True)
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
print(f"[KEP Master Profiler v17.0] Running CUDA Pipeline Diagnostic on: {device_str.upper()}")


def profile_master_karyon_pipeline():
    config = CoREConfig()
    config.net.text_dim = 256
    config.net.unified_dim = 256
    config.net.hidden_dim = 512
    config.net.expand_dim = 2048
    config.net.num_heads = 8
    config.net.head_k = 32
    config.net.head_v = 64
    config.net.num_attractors = 256
    config.train.chunk_size = 64

    agent = CoREAgent(config=config, device=device_str).to(device)
    hu = HomeostaticUnit(batch_size=32, device=device_str)
    episodic_mem = BatchedEpisodicMemory(batch_size=32, memory_dim=256, max_capacity=200, device=device_str)
    
    optimizer = torch.optim.AdamW(agent.get_all_parameters(), lr=3e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=256)

    b_size, seq_len = 32, 512
    chunk_size = 64
    dummy_tokens = torch.randint(32, 126, (b_size, seq_len), dtype=torch.long, device=device)
    target_tokens = torch.randint(32, 126, (b_size, seq_len), dtype=torch.long, device=device)

    start_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None

    time_embeddings = 0.0
    time_cortical_forward = 0.0
    time_opt_step = 0.0

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    print("\n" + "="*85)
    print(" === MASTER PROFILING: UNSHACKLED FLOW CORTICAL PIPELINE (v17.0) ===")
    print("="*85)

    t_batch_start = time.perf_counter()
    optimizer.zero_grad()

    # 1. Embedding Timing
    if start_event: start_event.record()
    full_emb = agent.pos_embeddings(dummy_tokens, start_pos=0, apply_rf=True)
    if end_event:
        end_event.record()
        torch.cuda.synchronize()
        time_embeddings = start_event.elapsed_time(end_event)

    # 2. Forward Sequence Timing
    if start_event: start_event.record()
    total_loss, speech_loss, fe_loss, m_states, h_proxy, curr_u_t, eff_dt = agent.forward_sequence(
        dummy_tokens, target_tokens, hu, criterion, episodic_memory=episodic_mem, chunk_size=chunk_size
    )
    if end_event:
        end_event.record()
        torch.cuda.synchronize()
        time_cortical_forward = start_event.elapsed_time(end_event)

    # 3. Optimizer Step Timing
    t_opt_0 = time.perf_counter()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.get_all_parameters(), max_norm=3.0)
    optimizer.step()
    if device.type == 'cuda': torch.cuda.synchronize()
    time_opt_step = (time.perf_counter() - t_opt_0) * 1000.0

    total_batch_time_ms = (time.perf_counter() - t_batch_start) * 1000.0
    total_batch_time_sec = total_batch_time_ms / 1000.0

    # 4. Inspect Gradient Flow
    grad_norms = {}
    for name, param in agent.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
        else:
            grad_norms[name] = 0.0

    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0
    tokens_per_sec = (b_size * seq_len) / total_batch_time_sec

    # =========================================================================
    # 5. DIAGNOSTIC REPORT BREAKDOWN
    # =========================================================================
    print(f"\n{'Submodule Component':<48} | {'Time (ms)':<14} | {'% of Batch':<12}")
    print("-" * 80)
    print(f"{'1. Positional Byte Embeddings (256D) + Conv1D':<48} | {time_embeddings:<14.2f} | {time_embeddings/total_batch_time_ms*100:<12.1f}%")
    print(f"{'2. Native C++20 Cortical Forward & Scan':<48} | {time_cortical_forward:<14.2f} | {time_cortical_forward/total_batch_time_ms*100:<12.1f}%")
    print(f"{'3. Backward Autograd & Optimizer Step':<48} | {time_opt_step:<14.2f} | {time_opt_step/total_batch_time_ms*100:<12.1f}%")
    print("="*80)
    print(f"{'TOTAL BATCH TIME':<48} | {total_batch_time_ms:<14.2f} ms ({total_batch_time_sec:.3f} sec)")
    print(f"{'PEAK VRAM MEMORY ALLOCATED':<48} | {peak_vram_mb:<14.2f} MB")
    print(f"{'THROUGHPUT SPEED':<48} | {tokens_per_sec:<14.1f} tokens/sec")
    print("="*80)

    # 6. Gradient Flow Dashboard
    print("\n" + "="*85)
    print(" === GRADIENT FLOW VERIFICATION DASHBOARD ===")
    print("="*85)
    print(f"{'Parameter Group':<52} | {'Grad Norm':<14} | {'Status':<15}")
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
        print(f"  {name:<50} | {g_norm:<14.6f} | {status_flag}")

    print("-" * 85)
    print(f"Total Parameters Audited: {len(grad_norms)} | Healthy: {healthy_grad_count} | Disconnected: {zero_grad_count}")
    print("="*85)

    # 7. Diagnostic Speech Sample
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
            m_state=torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device),
            h_state=torch.zeros(1, agent.hidden_dim, device=device),
            hu=diag_hu,
            episodic_memory=diag_mem,
            config=config,
            max_generated_tokens=50
        )
        for event in gen_stream:
            if event["status"] == "token":
                sample_chars.append(event["text"])

    print(f"  Prompt : \"{diag_prompt.strip()}\"")
    print(f"  Output : \"{''.join(sample_chars).strip()}\"")
    print("="*85 + "\n")

if __name__ == "__main__":
    profile_master_karyon_pipeline()
