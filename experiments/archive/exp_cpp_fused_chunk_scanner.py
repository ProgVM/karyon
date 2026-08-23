# experiments/exp_cpp_fused_chunk_scanner.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-20 (NATIVE C++ FUSED CHUNK SCANNER)
Hypothesis: Moving chunk recurrence loop inside C++ FusedSensorySDEEngine
eliminates 1024 Python/C++ roundtrips per batch, dropping latency to <100ms.
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import os
import types
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Unconditional PyTorch Dynamo Hotfix for Python 3.12 / Kaggle GPU
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

# 2. Add root path to import Karyon core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_config import CoREConfig
from karyon_core import (
    ByteTokenizer, HomeostaticUnit, SensoryGateway, MotorGateway,
    CausalByteReceptiveField, GoalConditionedMatrixSDESSMCore, FusedSensorySDEEngine,
    DesaturatedHopfieldAttractorHead, BatchedEpisodicMemory
)

torch.set_grad_enabled(True)
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)


# =============================================================================
# MODULE 1: POSITIONAL BYTE EMBEDDING
# =============================================================================

class OffsetPositionalByteEmbedding(nn.Module):
    def __init__(self, vocab_size=258, text_dim=128, max_len=4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.text_dim = text_dim
        self.byte_embed = nn.Embedding(vocab_size, text_dim)
        self.receptive_field = CausalByteReceptiveField(text_dim=text_dim, kernel_size=4, device=device_str)
        
        pe = torch.zeros(max_len, text_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, text_dim, 2).float() * (-math.log(10000.0) / text_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, apply_rf: bool = True) -> torch.Tensor:
        seq_len = input_ids.size(1)
        tok_emb = self.byte_embed(input_ids)
        pos_emb = self.pe[:, start_pos : start_pos + seq_len, :]
        embedded = tok_emb + pos_emb
        if apply_rf and seq_len > 1:
            embedded = self.receptive_field(embedded)
        return embedded


# =============================================================================
# BENCHMARK AGENTS
# =============================================================================

class PythonLoopAgent(nn.Module):
    """Agent executing 32-step chunk recurrence loop inside Python."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim)
        self.gateway = SensoryGateway(self.unified_dim, self.hidden_dim, config.net.homeo_dim,
                                      self.text_dim, config.net.vision_dim, config.net.action_dim, device_str)
        self.sde_ssm = GoalConditionedMatrixSDESSMCore(
            self.unified_dim, self.hidden_dim, 8, 32, 64, config.net.homeo_dim, device_str
        )
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim, device=device_str)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk(self, chunk_emb, chunk_targets, m_prev, h_prev, u_t, criterion):
        batch_size, chunk_len, _ = chunk_emb.size()
        obs_vis = torch.zeros(batch_size, self.config.net.vision_dim, device=device)
        prev_act = torch.zeros(batch_size, self.config.net.action_dim, device=device)

        h_states = []
        for t in range(chunk_len):
            w_t = self.gateway(chunk_emb[:, t], obs_vis, prev_act, h_prev, u_t)[0]
            sde_out = self.sde_ssm.forward_step(m_prev, h_prev, w_t, u_t)
            m_prev, h_prev = sde_out[0], sde_out[1]
            h_states.append(h_prev)

        h_chunk = torch.stack(h_states, dim=1).reshape(batch_size * chunk_len, self.hidden_dim)
        h_relaxed = self.attractor_head.relax_to_minima(h_chunk)[0]
        logits = F.linear(self.motor_text_proj(h_relaxed), self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        loss = criterion(logits, chunk_targets.contiguous().view(-1))
        return loss, m_prev, h_prev


class NativeCppFusedAgent(nn.Module):
    """Agent executing 32-step chunk recurrence loop inside Native C++ FusedSensorySDEEngine."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim)
        # Monolithic C++ Engine
        self.fused_engine = FusedSensorySDEEngine(
            self.unified_dim, self.hidden_dim, self.text_dim,
            config.net.vision_dim, config.net.action_dim, 8, 32, 64, config.net.homeo_dim, device_str
        )
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim, device=device_str)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk(self, chunk_emb, chunk_targets, m_prev, h_prev, u_t, criterion):
        batch_size, chunk_len, _ = chunk_emb.size()
        obs_vis = torch.zeros(batch_size, self.config.net.vision_dim, device=device)
        prev_act = torch.zeros(batch_size, self.config.net.action_dim, device=device)

        # 1 Single Monolithic C++ Call for all 32 steps!
        out = self.fused_engine.forward_chunk(chunk_emb, obs_vis, prev_act, m_prev, h_prev, u_t, 1.0)
        h_chunk, m_next, h_next = out[0], out[1], out[2]

        h_relaxed = self.attractor_head.relax_to_minima(h_chunk)[0]
        logits = F.linear(self.motor_text_proj(h_relaxed), self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        loss = criterion(logits, chunk_targets.contiguous().view(-1))
        return loss, m_next, h_next


# =============================================================================
# BENCHMARK EXECUTION SUITE
# =============================================================================

def run_isolated_benchmark():
    print(f"\n{'='*85}")
    print(f" === RUNNING KEP HYPOTHESIS #20 BENCHMARK (C++ FUSED SCANNER): {device_str.upper()} ===")
    print(f"{'='*85}\n")

    config = CoREConfig()
    config.net.hidden_dim = 512
    config.net.unified_dim = 256
    config.net.text_dim = 128
    config.net.text_gen_dim = 258

    batch_size = 32
    seq_len = 512
    chunk_size = 32
    num_eval_steps = 25

    criterion = nn.CrossEntropyLoss(ignore_index=256)
    num_chunks = seq_len // chunk_size

    torch.manual_seed(42)
    dummy_input = torch.randint(32, 126, (batch_size, seq_len), dtype=torch.long, device=device)
    dummy_target = torch.randint(32, 126, (batch_size, seq_len), dtype=torch.long, device=device)

    # -------------------------------------------------------------------------
    # TEST 1: BASELINE (Python Loop Dispatch)
    # -------------------------------------------------------------------------
    print("[1/2] Evaluating BASELINE (Python Chunk Recurrence Loop)...")
    base_model = PythonLoopAgent(config).to(device)
    base_opt = torch.optim.Adam(base_model.parameters(), lr=3e-3)
    hu_base = HomeostaticUnit(batch_size=batch_size, device=device_str)

    base_times, base_losses = [], []

    for step in range(num_eval_steps):
        base_opt.zero_grad()
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()

        m_prev = torch.zeros(batch_size, 8, 32, 64, device=device)
        h_prev = torch.zeros(batch_size, config.net.hidden_dim, device=device)
        u_t = hu_base.state.clone().detach()

        batch_losses = []
        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = (chunk_idx + 1) * chunk_size

            chunk_emb = base_model.pos_embeddings(dummy_input[:, c_start:c_end], start_pos=c_start, apply_rf=True)
            chunk_targets = dummy_target[:, c_start:c_end]

            chunk_loss, m_prev, h_prev = base_model.forward_chunk(chunk_emb, chunk_targets, m_prev, h_prev, u_t, criterion)
            (chunk_loss / float(num_chunks)).backward()
            batch_losses.append(chunk_loss.item())

            m_prev = m_prev.detach()
            h_prev = h_prev.detach()

        base_opt.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000.0

        base_times.append(step_ms)
        base_losses.append(sum(batch_losses) / len(batch_losses))

    # -------------------------------------------------------------------------
    # TEST 2: PROPOSED (Native C++ Fused Engine)
    # -------------------------------------------------------------------------
    print("[2/2] Evaluating PROPOSED (Native C++ FusedSensorySDEEngine)...")
    prop_model = NativeCppFusedAgent(config).to(device)
    prop_opt = torch.optim.Adam(prop_model.parameters(), lr=3e-3)
    hu_prop = HomeostaticUnit(batch_size=batch_size, device=device_str)

    prop_times, prop_losses = [], []

    for step in range(num_eval_steps):
        prop_opt.zero_grad()
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()

        m_prev = torch.zeros(batch_size, 8, 32, 64, device=device)
        h_prev = torch.zeros(batch_size, config.net.hidden_dim, device=device)
        u_t = hu_prop.state.clone().detach()

        batch_losses = []
        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = (chunk_idx + 1) * chunk_size

            chunk_emb = prop_model.pos_embeddings(dummy_input[:, c_start:c_end], start_pos=c_start, apply_rf=True)
            chunk_targets = dummy_target[:, c_start:c_end]

            chunk_loss, m_prev, h_prev = prop_model.forward_chunk(chunk_emb, chunk_targets, m_prev, h_prev, u_t, criterion)
            (chunk_loss / float(num_chunks)).backward()
            batch_losses.append(chunk_loss.item())

            m_prev = m_prev.detach()
            h_prev = h_prev.detach()

        prop_opt.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000.0

        prop_times.append(step_ms)
        prop_losses.append(sum(batch_losses) / len(batch_losses))

    # =========================================================================
    # KEP RULE #6: PROCESS DIAGNOSTICS & TELEMETRY REPORT
    # =========================================================================
    avg_base_time = sum(base_times[-10:]) / 10.0
    base_tok_per_sec = (batch_size * seq_len) / (avg_base_time / 1000.0)

    avg_prop_time = sum(prop_times[-10:]) / 10.0
    prop_tok_per_sec = (batch_size * seq_len) / (avg_prop_time / 1000.0)

    print("\n" + "="*90)
    print(" === [KEP RULE #6 TELEMETRY BENCHMARK COMPARISON REPORT] ===")
    print("="*90)
    print(f"{'Performance Metric':<35} | {'Baseline (Python Loop)':<22} | {'Proposed (Native C++ Fused)':<22} | {'Delta / Speedup':<15}")
    print("-" * 95)
    print(f"{'Total Step Duration (ms)':<35} | {avg_base_time:<22.2f} | {avg_prop_time:<22.2f} | {avg_prop_time - avg_base_time:+6.1f} ms (🚀)")
    print(f"{'Throughput Speed (tok/s)':<35} | {base_tok_per_sec:<22.1f} | {prop_tok_per_sec:<22.1f} | {prop_tok_per_sec/base_tok_per_sec:+.2f}x (⚡⚡⚡)")
    print(f"{'Final Loss (Step 25)':<35} | {base_losses[-1]:<22.4f} | {prop_losses[-1]:<22.4f} | {prop_losses[-1] - base_losses[-1]:+6.4f}")
    print("="*90)

    if avg_prop_time < avg_base_time * 0.7:
        print("🟢 KEP VERDICT: POSITIVE (Hypothesis #20 Validated! Ready for merge into production).")
    else:
        print("🔴 KEP VERDICT: REJECTED (Target speedup not reached).")

if __name__ == "__main__":
    run_isolated_benchmark()
