# experiments/exp_parallel_chunk_scan_ssm.py
"""
feat(exp): implement parallel state-space duality chunk scan without token loops

===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-22 (STATE SPACE DUALITY CHUNK SCAN)
Hypothesis: Formulating intra-chunk recurrence as parallel matrix attention
Y_intra = (Q @ K.T * Decay) @ V and inter-chunk retrieval as Y_inter = Q @ M_prev
eliminates ALL token loops, dropping batch latency to <50-100ms (>80,000 tok/s).
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
# MODULE 3: ZERO-LOOP CHUNKED STATE-SPACE DUALITY CORE (SSD-SDE)
# =============================================================================

class ParallelChunkStateSpaceDualityCore(nn.Module):
    """
    Zero-Loop Parallel Chunked State-Space Duality Engine.
    Executes intra-chunk recurrence and inter-chunk retrieval in parallel Tensor Core GEMMs:
    Y_chunk = Y_intra + Y_inter
    100% Loop-Free on CUDA!
    """
    def __init__(self, text_dim=128, unified_dim=256, hidden_dim=512, num_heads=8, head_k=32, head_v=64):
        super().__init__()
        self.text_dim = text_dim
        self.unified_dim = unified_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_k = head_k
        self.head_v = head_v

        # Parallel Projections
        self.sensory_proj = nn.Linear(text_dim, unified_dim)
        self.q_proj = nn.Linear(unified_dim, num_heads * head_k)
        self.k_proj = nn.Linear(unified_dim, num_heads * head_k)
        self.v_proj = nn.Linear(unified_dim, num_heads * head_v)
        
        # Learnable multi-head continuous decay rates
        self.decay_logits = nn.Parameter(torch.randn(1, num_heads, 1, 1) * 0.1 + 2.0)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward_chunk_parallel_ssd(self, chunk_emb, m_prev, u_t, dt=1.0):
        # chunk_emb: [Batch, ChunkLen, TextDim]
        batch_size, chunk_len, _ = chunk_emb.size()
        na = u_t[:, 4:5].view(batch_size, 1, 1, 1)
        da = u_t[:, 5:6].view(batch_size, 1, 1, 1)
        eff_dt = torch.clamp(dt * (1.0 - 0.4 * na + 0.4 * da), 0.30, 2.00)

        # 1. Parallel Projections for the entire chunk [Batch, ChunkLen, Dim]
        w_chunk = self.sensory_proj(chunk_emb)
        
        # Reshape to [Batch, NumHeads, ChunkLen, HeadDim]
        q = self.q_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)
        q = F.normalize(q, p=2, dim=-1)

        k = self.k_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)
        k = F.normalize(k, p=2, dim=-1)

        v = self.v_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_v).transpose(1, 2)

        # Continuous decay factor per head: alpha [1, NumHeads, 1, 1]
        alpha = torch.sigmoid(self.decay_logits) ** eff_dt

        # 2. INTRA-CHUNK PARALLEL ATTENTION MATRIX (Zero Loops!)
        # Decay matrix: mask[i, j] = alpha^(i - j) for i >= j, else 0
        pos = torch.arange(chunk_len, device=device)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0) # [ChunkLen, ChunkLen]
        causal_mask = (diff >= 0).float()
        
        # Broadcasted decay weights [1, NumHeads, ChunkLen, ChunkLen]
        decay_weights = (alpha ** diff.clamp(min=0)) * causal_mask

        # S = (Q @ K.T) * DecayWeights -> [Batch, NumHeads, ChunkLen, ChunkLen]
        s_matrix = torch.matmul(q, k.transpose(-1, -2)) * decay_weights
        y_intra = torch.matmul(s_matrix, v) # [Batch, NumHeads, ChunkLen, HeadV]

        # 3. INTER-CHUNK STATE-SPACE RETRIEVAL (Zero Loops!)
        # Decay from chunk start: decay_to_start [1, NumHeads, ChunkLen, 1]
        decay_to_start = alpha ** pos.view(1, 1, chunk_len, 1)
        # y_inter = (Q * decay_to_start) @ M_prev -> [Batch, NumHeads, ChunkLen, HeadV]
        y_inter = torch.matmul(q * decay_to_start, m_prev)

        # Total Chunk Output: Y_total = Y_intra + Y_inter
        y_total = (y_intra + y_inter).transpose(1, 2).reshape(batch_size * chunk_len, self.hidden_dim)
        h_chunk = self.norm(self.out_proj(y_total) + y_total)

        # 4. CHUNK MATRIX STATE ACCUMULATION FOR NEXT CHUNK
        # Decay to chunk end: decay_to_end [1, NumHeads, ChunkLen, 1]
        decay_to_end = alpha ** (chunk_len - 1 - pos).view(1, 1, chunk_len, 1)
        # KV update: (K.T * decay_to_end) @ V
        k_decayed = k * decay_to_end # [Batch, NumHeads, ChunkLen, HeadK]
        kv_chunk_update = torch.matmul(k_decayed.transpose(-1, -2), v) # [Batch, NumHeads, HeadK, HeadV]

        # Wiener diffusion on matrix state
        sigma = 1e-3
        dW = torch.randn_like(m_prev) * torch.sqrt(eff_dt) * sigma

        # m_next = alpha^ChunkLen * m_prev + (1 - alpha) * kv_chunk_update + dW
        alpha_chunk = alpha ** chunk_len
        m_next = alpha_chunk * m_prev + (1.0 - alpha) * kv_chunk_update + dW

        return h_chunk, m_next


# =============================================================================
# BENCHMARK AGENTS
# =============================================================================

class PreprojectedLoopAgent(nn.Module):
    """Previous Model (Pre-Projected but with token-by-token loop in chunk)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim)
        self.sensory_proj = nn.Linear(self.text_dim, self.unified_dim)
        self.q_proj = nn.Linear(self.unified_dim + self.hidden_dim, 8 * 32)
        self.k_proj = nn.Linear(self.unified_dim, 8 * 32)
        self.v_proj = nn.Linear(self.unified_dim, 8 * 64)
        self.decay_proj = nn.Linear(self.unified_dim, 8)
        self.out_gate = nn.Linear(self.unified_dim + self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)

        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk(self, chunk_emb, chunk_targets, m_prev, h_prev, u_t, criterion):
        batch_size, chunk_len, _ = chunk_emb.size()
        w_chunk = self.sensory_proj(chunk_emb)
        k_chunk = F.normalize(self.k_proj(w_chunk).view(batch_size, chunk_len, 8, 32, 1), p=2, dim=-2)
        v_chunk = self.v_proj(w_chunk).view(batch_size, chunk_len, 8, 1, 64)
        alpha_chunk = torch.sigmoid(self.decay_proj(w_chunk)).view(batch_size, chunk_len, 8, 1, 1)

        h_states = []
        for t in range(chunk_len):
            kv_assoc = torch.matmul(k_chunk[:, t], v_chunk[:, t])
            m_prev = alpha_chunk[:, t] * m_prev + (1.0 - alpha_chunk[:, t]) * kv_assoc
            q_in = torch.cat([w_chunk[:, t], h_prev], dim=-1)
            q_t = F.normalize(self.q_proj(q_in).view(batch_size, 8, 1, 32), p=2, dim=-1)
            readout = torch.matmul(q_t, m_prev).view(batch_size, self.hidden_dim)
            gate = F.silu(self.out_gate(q_in))
            h_prev = self.norm(self.out_proj(readout * gate) + readout)
            h_states.append(h_prev)

        h_chunk = torch.stack(h_states, dim=1).reshape(batch_size * chunk_len, self.hidden_dim)
        h_relaxed = self.attractor_head.relax_to_minima(h_chunk)[0]
        logits = F.linear(self.motor_text_proj(h_relaxed), self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        loss = criterion(logits, chunk_targets.contiguous().view(-1))
        return loss, m_prev, h_prev


class ZeroLoopSSDAgent(nn.Module):
    """Proposed Model (100% Loop-Free State-Space Duality Parallel Scan)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim)
        self.ssd_core = ParallelChunkStateSpaceDualityCore(
            text_dim=self.text_dim, unified_dim=self.unified_dim, hidden_dim=self.hidden_dim,
            num_heads=8, head_k=32, head_v=64
        )
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk(self, chunk_emb, chunk_targets, m_prev, u_t, criterion):
        # Zero token loops! 100% Pure Parallel Tensor Core Operations
        h_chunk, m_next = self.ssd_core.forward_chunk_parallel_ssd(chunk_emb, m_prev, u_t, dt=1.0)
        h_relaxed = self.attractor_head.relax_to_minima(h_chunk)[0]
        logits = F.linear(self.motor_text_proj(h_relaxed), self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        loss = criterion(logits, chunk_targets.contiguous().view(-1))
        return loss, m_next


# =============================================================================
# BENCHMARK EXECUTION SUITE
# =============================================================================

def run_isolated_benchmark():
    print(f"\n{'='*85}")
    print(f" === RUNNING KEP HYPOTHESIS #22 BENCHMARK (ZERO-LOOP PARALLEL SSD): {device_str.upper()} ===")
    print(f"{'='*85}\n")

    config = CoREConfig()
    config.net.hidden_dim = 512
    config.net.unified_dim = 256
    config.net.text_dim = 128
    config.net.text_gen_dim = 258

    batch_size = 32
    seq_len = 512
    chunk_size = 32
    num_eval_steps = 30

    criterion = nn.CrossEntropyLoss(ignore_index=256)
    num_chunks = seq_len // chunk_size

    torch.manual_seed(42)
    dummy_input = torch.randint(32, 126, (batch_size, seq_len), dtype=torch.long, device=device)
    dummy_target = torch.randint(32, 126, (batch_size, seq_len), dtype=torch.long, device=device)

    # -------------------------------------------------------------------------
    # TEST 1: BASELINE (Pre-Projected with Token Loop in Chunk)
    # -------------------------------------------------------------------------
    print("[1/2] Evaluating BASELINE (Pre-projected with Token Loop in Chunk)...")
    base_model = PreprojectedLoopAgent(config).to(device)
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
    # TEST 2: PROPOSED (Zero-Loop Parallel State-Space Duality Scan)
    # -------------------------------------------------------------------------
    print("[2/2] Evaluating PROPOSED (Zero-Loop Parallel State-Space Duality Scan)...")
    prop_model = ZeroLoopSSDAgent(config).to(device)
    prop_opt = torch.optim.Adam(prop_model.parameters(), lr=3e-3)
    hu_prop = HomeostaticUnit(batch_size=batch_size, device=device_str)

    prop_times, prop_losses = [], []

    for step in range(num_eval_steps):
        prop_opt.zero_grad()
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()

        m_prev = torch.zeros(batch_size, 8, 32, 64, device=device)
        u_t = hu_prop.state.clone().detach()

        batch_losses = []
        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = (chunk_idx + 1) * chunk_size

            chunk_emb = prop_model.pos_embeddings(dummy_input[:, c_start:c_end], start_pos=c_start, apply_rf=True)
            chunk_targets = dummy_target[:, c_start:c_end]

            # 100% Loop-Free Parallel Chunked SSD Execution
            chunk_loss, m_prev = prop_model.forward_chunk(chunk_emb, chunk_targets, m_prev, u_t, criterion)
            (chunk_loss / float(num_chunks)).backward()
            batch_losses.append(chunk_loss.item())

            m_prev = m_prev.detach()

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
    print(f"{'Performance Metric':<35} | {'Baseline (Token Loop)':<22} | {'Proposed (Zero-Loop SSD)':<22} | {'Delta / Speedup':<15}")
    print("-" * 95)
    print(f"{'Total Step Duration (ms)':<35} | {avg_base_time:<22.2f} | {avg_prop_time:<22.2f} | {avg_prop_time - avg_base_time:+6.1f} ms (🚀)")
    print(f"{'Throughput Speed (tok/s)':<35} | {base_tok_per_sec:<22.1f} | {prop_tok_per_sec:<22.1f} | {prop_tok_per_sec/base_tok_per_sec:+.2f}x (⚡⚡⚡)")
    print(f"{'Final Loss (Step 30)':<35} | {base_losses[-1]:<22.4f} | {prop_losses[-1]:<22.4f} | {prop_losses[-1] - base_losses[-1]:+6.4f}")
    print("="*90)

    if avg_prop_time < avg_base_time * 0.3:
        print("🟢 KEP VERDICT: POSITIVE (Hypothesis #22 Validated! Ready for merge into production).")
    else:
        print("🔴 KEP VERDICT: REJECTED (Target acceleration not reached).")

if __name__ == "__main__":
    run_isolated_benchmark()
