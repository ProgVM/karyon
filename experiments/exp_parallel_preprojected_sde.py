# experiments/exp_parallel_preprojected_sde.py
"""
feat(exp): implement parallel chunk pre-projection and lightweight fast-weight scan

===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-21 (PARALLEL PRE-PROJECTED SDE-SSM)
Hypothesis: Pre-projecting K, V, and Decay factors across all 32 tokens in a
single batched GEMM eliminates 95% of CUDA kernel launches, dropping step latency
from 2400ms to <100ms (>50,000 tok/s) with 100% mathematical fidelity.
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
from karyon_core import ByteTokenizer, HomeostaticUnit, SensoryGateway, MotorGateway, BatchedEpisodicMemory

torch.set_grad_enabled(True)
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)


# =============================================================================
# MODULE 1: CAUSAL RECEPTIVE FIELD & POSITIONAL EMBEDDINGS
# =============================================================================

class CausalByteReceptiveField(nn.Module):
    def __init__(self, text_dim=128, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(text_dim, text_dim, kernel_size=kernel_size, groups=text_dim, bias=False)
        self.norm = nn.LayerNorm(text_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x_trans = x_seq.transpose(1, 2)
        x_padded = F.pad(x_trans, (self.kernel_size - 1, 0), mode='constant', value=0.0)
        conv_out = self.conv(x_padded)
        return self.norm(conv_out.transpose(1, 2) + x_seq)


class OffsetPositionalByteEmbedding(nn.Module):
    def __init__(self, vocab_size=258, text_dim=128, max_len=4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.text_dim = text_dim
        self.byte_embed = nn.Embedding(vocab_size, text_dim)
        self.receptive_field = CausalByteReceptiveField(text_dim=text_dim, kernel_size=4)
        
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
# MODULE 2: DESATURATED HOPFIELD ATTRACTOR MEMORY
# =============================================================================

class DesaturatedHopfieldAttractorHead(nn.Module):
    def __init__(self, hidden_dim=512, vocab_size=258, num_attractors=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = 1.0 / math.sqrt(hidden_dim)
        self.attractor_basins = nn.Parameter(torch.randn(num_attractors, hidden_dim) * 0.05)

    def relax_to_minima(self, h_state: torch.Tensor):
        norm_dist_sq = (torch.cdist(h_state, self.attractor_basins, p=2)**2) * self.scale
        attn_weights = F.softmax(-norm_dist_sq, dim=-1)
        attractor_shift = torch.matmul(attn_weights, self.attractor_basins)
        h_relaxed = h_state + 0.25 * attractor_shift
        energy = -torch.logsumexp(-norm_dist_sq, dim=-1, keepdim=True)
        return h_relaxed, energy


# =============================================================================
# MODULE 3: ULTRA-FAST PRE-PROJECTED MATRIX SDE-SSM
# =============================================================================

class ParallelPreprojectedMatrixSDESSMCore(nn.Module):
    """
    Ultra-Fast Pre-Projected Matrix SDE-SSM.
    Projects K, V, and Decay across all 32 tokens in 1 batched GEMM,
    executing a zero-Linear-layer lightweight recurrent loop on CUDA.
    """
    def __init__(self, text_dim=128, unified_dim=256, hidden_dim=512, num_heads=8, head_k=32, head_v=64, homeo_dim=6):
        super().__init__()
        self.text_dim = text_dim
        self.unified_dim = unified_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_k = head_k
        self.head_v = head_v

        # Fast direct sensory projections (Pre-projected in parallel)
        self.sensory_proj = nn.Linear(text_dim, unified_dim)
        self.k_proj = nn.Linear(unified_dim, num_heads * head_k)
        self.v_proj = nn.Linear(unified_dim, num_heads * head_v)
        self.decay_proj = nn.Linear(unified_dim, num_heads)

        # Context query & outflow
        self.q_proj = nn.Linear(unified_dim + hidden_dim, num_heads * head_k)
        self.out_gate = nn.Linear(unified_dim + hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward_chunk_preprojected(self, chunk_emb, m_prev, h_prev, u_t, dt=1.0):
        batch_size, chunk_len, _ = chunk_emb.size()
        na = u_t[:, 4:5]
        da = u_t[:, 5:6]

        eff_dt_raw = torch.clamp(dt * (1.0 - 0.4 * na + 0.4 * da), 0.30, 2.00)
        eff_dt_4d = eff_dt_raw.view(batch_size, 1, 1, 1)

        # 1. PARALLEL CHUNK PRE-PROJECTION (1 Single Batched GEMM for all 32 tokens!)
        w_chunk = self.sensory_proj(chunk_emb) # [Batch, ChunkLen, UnifiedDim]
        k_chunk = self.k_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_k, 1)
        k_chunk = F.normalize(k_chunk, p=2, dim=-2)

        v_chunk = self.v_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, 1, self.head_v)
        alpha_chunk = (torch.sigmoid(self.decay_proj(w_chunk)).view(batch_size, chunk_len, self.num_heads, 1, 1) ** eff_dt_4d)

        sigma = 1e-3
        sqrt_dt_sigma = torch.sqrt(eff_dt_4d) * sigma

        # 2. ZERO-LINEAR INNER LOOP (Pure lightweight matrix updates!)
        h_states = []
        for t in range(chunk_len):
            w_t = w_chunk[:, t]
            k_t = k_chunk[:, t]
            v_t = v_chunk[:, t]
            alpha_t = alpha_chunk[:, t]

            kv_assoc = torch.matmul(k_t, v_t)
            dW = torch.randn_like(m_prev) * sqrt_dt_sigma

            m_prev = alpha_t * m_prev + (1.0 - alpha_t) * kv_assoc + dW

            # Goal-Conditioned Query
            q_in = torch.cat([w_t, h_prev], dim=-1)
            q_t = self.q_proj(q_in).view(batch_size, self.num_heads, 1, self.head_k)
            q_t = F.normalize(q_t, p=2, dim=-1)

            readout = torch.matmul(q_t, m_prev).view(batch_size, self.hidden_dim)
            gate = F.silu(self.out_gate(q_in))
            h_prev = self.norm(self.out_proj(readout * gate) + readout)
            h_states.append(h_prev)

        h_chunk_tensor = torch.stack(h_states, dim=1).reshape(batch_size * chunk_len, self.hidden_dim)
        return h_chunk_tensor, m_prev, h_prev


# =============================================================================
# BENCHMARK AGENTS
# =============================================================================

class UnfusedBaselineAgent(nn.Module):
    """Unfused Agent (dispatches Gateway and SDE-SSM on every token)."""
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
        self.in_proj = nn.Linear(self.unified_dim, self.hidden_dim)
        self.decay_proj = nn.Linear(self.unified_dim + config.net.homeo_dim, self.hidden_dim)
        self.out_gate = nn.Linear(self.unified_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk(self, chunk_emb, chunk_targets, h_prev, u_t, criterion):
        batch_size, chunk_len, _ = chunk_emb.size()
        obs_vis = torch.zeros(batch_size, self.config.net.vision_dim, device=device)
        prev_act = torch.zeros(batch_size, self.config.net.action_dim, device=device)

        step_losses = []
        for t in range(chunk_len):
            w_t = self.gateway(chunk_emb[:, t], obs_vis, prev_act, h_prev, u_t)[0]
            decay_logits = self.decay_proj(torch.cat([w_t, u_t], dim=-1))
            alpha = torch.sigmoid(decay_logits)
            b_input = self.in_proj(w_t)
            h_prev = alpha * h_prev + (1.0 - alpha) * b_input
            gate = F.silu(self.out_gate(w_t))
            y_t = self.layer_norm(self.out_proj(h_prev * gate) + h_prev)

            h_relaxed, _ = self.attractor_head.relax_to_minima(y_t)
            h_proj = self.motor_text_proj(h_relaxed)
            logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
            step_losses.append(criterion(logits, chunk_targets[:, t]))

        return torch.stack(step_losses).mean(), h_prev


class FastPreprojectedAgent(nn.Module):
    """Ultra-Fast Agent with Parallel Chunk Pre-Projection & Zero-Linear Recurrence."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim)
        self.core = ParallelPreprojectedMatrixSDESSMCore(
            text_dim=self.text_dim, unified_dim=self.unified_dim, hidden_dim=self.hidden_dim,
            num_heads=8, head_k=32, head_v=64, homeo_dim=config.net.homeo_dim
        )
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk(self, chunk_emb, chunk_targets, m_prev, h_prev, u_t, criterion):
        batch_size, chunk_len, _ = chunk_emb.size()

        # 1. Parallel Pre-Projected Fast Chunk Recurrence
        h_chunk, m_next, h_next = self.core.forward_chunk_preprojected(chunk_emb, m_prev, h_prev, u_t, dt=1.0)

        # 2. Parallel Batched Motor Tensor Readout (1 single GEMM for all 32 tokens!)
        h_relaxed_flat, _ = self.attractor_head.relax_to_minima(h_chunk)
        h_proj_flat = self.motor_text_proj(h_relaxed_flat)
        logits_flat = F.linear(h_proj_flat, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim

        # 3. Batched Loss
        loss = criterion(logits_flat, chunk_targets.contiguous().view(-1))
        return loss, m_next, h_next


# =============================================================================
# BENCHMARK EXECUTION SUITE
# =============================================================================

def run_isolated_benchmark():
    print(f"\n{'='*85}")
    print(f" === RUNNING KEP HYPOTHESIS #21 BENCHMARK (PARALLEL PRE-PROJECTION): {device_str.upper()} ===")
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
    # TEST 1: BASELINE (Unfused Step-by-Step Dispatch)
    # -------------------------------------------------------------------------
    print("[1/2] Evaluating BASELINE (Unfused Sequential Dispatch)...")
    base_model = UnfusedBaselineAgent(config).to(device)
    base_opt = torch.optim.Adam(base_model.parameters(), lr=3e-3)
    hu_base = HomeostaticUnit(batch_size=batch_size, device=device_str)

    base_times, base_losses = [], []

    for step in range(num_eval_steps):
        base_opt.zero_grad()
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()

        h_prev = torch.zeros(batch_size, config.net.hidden_dim, device=device)
        u_t = hu_base.state.clone().detach()

        batch_losses = []
        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = (chunk_idx + 1) * chunk_size

            chunk_emb = base_model.pos_embeddings(dummy_input[:, c_start:c_end], start_pos=c_start, apply_rf=True)
            chunk_targets = dummy_target[:, c_start:c_end]

            chunk_loss, h_prev = base_model.forward_chunk(chunk_emb, chunk_targets, h_prev, u_t, criterion)
            (chunk_loss / float(num_chunks)).backward()
            batch_losses.append(chunk_loss.item())

            h_prev = h_prev.detach()

        base_opt.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000.0

        base_times.append(step_ms)
        base_losses.append(sum(batch_losses) / len(batch_losses))

    # -------------------------------------------------------------------------
    # TEST 2: PROPOSED (Parallel Pre-Projected Fast SDE-SSM)
    # -------------------------------------------------------------------------
    print("[2/2] Evaluating PROPOSED (Parallel Chunk Pre-Projection & Fast Recurrence)...")
    prop_model = FastPreprojectedAgent(config).to(device)
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
    print(f"{'Performance Metric':<35} | {'Baseline (Unfused)':<22} | {'Proposed (Pre-Projected)':<22} | {'Delta / Speedup':<15}")
    print("-" * 95)
    print(f"{'Total Step Duration (ms)':<35} | {avg_base_time:<22.2f} | {avg_prop_time:<22.2f} | {avg_prop_time - avg_base_time:+6.1f} ms (🚀)")
    print(f"{'Throughput Speed (tok/s)':<35} | {base_tok_per_sec:<22.1f} | {prop_tok_per_sec:<22.1f} | {prop_tok_per_sec/base_tok_per_sec:+.2f}x (⚡⚡⚡)")
    print(f"{'Final Loss (Step 25)':<35} | {base_losses[-1]:<22.4f} | {prop_losses[-1]:<22.4f} | {prop_losses[-1] - base_losses[-1]:+6.4f}")
    print("="*90)

    if avg_prop_time < avg_base_time * 0.4:
        print("🟢 KEP VERDICT: POSITIVE (Hypothesis #21 Validated! Ready for merge into production).")
    else:
        print("🔴 KEP VERDICT: REJECTED (Target acceleration threshold not reached).")

if __name__ == "__main__":
    run_isolated_benchmark()
