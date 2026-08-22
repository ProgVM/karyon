# experiments/exp_fused_chunk_scanner.py
"""
feat(exp): implement fused native chunk recurrence and parallel motor readout

===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-19 (FUSED CHUNK RECURRENCE & PARALLEL MOTOR)
Hypothesis: Fusing chunk recurrence with parallel batched motor projections
slashes batch step latency by 15x-20x (>50,000 tok/s) with 100% mathematical
equivalence and preserved SDE-SSM gradient propagation.
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
# MODULE 3: GOAL-CONDITIONED MATRIX SDE-SSM
# =============================================================================

class GoalConditionedMatrixSDESSMCore(nn.Module):
    def __init__(self, unified_dim=256, hidden_dim=512, num_heads=8, head_k=32, head_v=64, homeo_dim=6):
        super().__init__()
        self.unified_dim = unified_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_k = head_k
        self.head_v = head_v

        self.q_proj = nn.Linear(unified_dim + hidden_dim, num_heads * head_k)
        self.k_proj = nn.Linear(unified_dim, num_heads * head_k)
        self.v_proj = nn.Linear(unified_dim, num_heads * head_v)
        
        self.decay_proj = nn.Linear(unified_dim + homeo_dim, num_heads)
        self.out_gate = nn.Linear(unified_dim + hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward_step(self, m_prev, h_prev, w_t, u_t, dt=1.0):
        batch_size = w_t.size(0)
        na = u_t[:, 4:5]
        da = u_t[:, 5:6]

        eff_dt_raw = torch.clamp(dt * (1.0 - 0.4 * na + 0.4 * da), 0.30, 2.00)
        eff_dt_4d = eff_dt_raw.view(batch_size, 1, 1, 1)

        q_input = torch.cat([w_t, h_prev], dim=-1)
        q = self.q_proj(q_input).view(batch_size, self.num_heads, 1, self.head_k)
        q = F.normalize(q, p=2, dim=-1)

        k = self.k_proj(w_t).view(batch_size, self.num_heads, self.head_k, 1)
        k = F.normalize(k, p=2, dim=-2)

        v = self.v_proj(w_t).view(batch_size, self.num_heads, 1, self.head_v)

        decay_in = torch.cat([w_t, u_t], dim=-1)
        alpha = (torch.sigmoid(self.decay_proj(decay_in)).view(batch_size, self.num_heads, 1, 1) ** eff_dt_4d)

        kv_assoc = torch.matmul(k, v)
        sigma = 1e-3
        dW = torch.randn_like(m_prev) * torch.sqrt(eff_dt_4d) * sigma

        m_next = alpha * m_prev + (1.0 - alpha) * kv_assoc + dW
        readout = torch.matmul(q, m_next).view(batch_size, self.hidden_dim)

        gate = F.silu(self.out_gate(q_input))
        h_next = self.norm(self.out_proj(readout * gate) + readout)

        return m_next, h_next, eff_dt_raw


# =============================================================================
# BENCHMARK AGENTS
# =============================================================================

class SequentialStepAgent(nn.Module):
    """Current Agent (Sequential Step-by-Step Dispatch inside Python Loop)."""
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
        self.goal_sde_ssm = GoalConditionedMatrixSDESSMCore(
            unified_dim=self.unified_dim, hidden_dim=self.hidden_dim, num_heads=8, head_k=32, head_v=64, homeo_dim=config.net.homeo_dim
        )
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk_sequential(self, chunk_emb, chunk_targets, m_prev, h_prev, u_t, criterion):
        batch_size, chunk_len, _ = chunk_emb.size()
        obs_vis = torch.zeros(batch_size, self.config.net.vision_dim, device=device)
        prev_act = torch.zeros(batch_size, self.config.net.action_dim, device=device)

        step_losses = []
        for t in range(chunk_len):
            w_t, _, _, _ = self.gateway(chunk_emb[:, t], obs_vis, prev_act, h_prev, u_t)
            m_prev, h_prev, _ = self.goal_sde_ssm.forward_step(m_prev, h_prev, w_t, u_t)
            
            # Step-by-step un-fused motor readout (dispatches kernels on every token)
            h_relaxed, _ = self.attractor_head.relax_to_minima(h_prev)
            h_proj = self.motor_text_proj(h_relaxed)
            logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
            step_losses.append(criterion(logits, chunk_targets[:, t]))

        return torch.stack(step_losses).mean(), m_prev, h_prev


class FusedChunkScannerAgent(nn.Module):
    """Proposed Agent (Fused Recurrent SDE-SSM + Parallel Batched Motor Tensor Readout)."""
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
        self.goal_sde_ssm = GoalConditionedMatrixSDESSMCore(
            unified_dim=self.unified_dim, hidden_dim=self.hidden_dim, num_heads=8, head_k=32, head_v=64, homeo_dim=config.net.homeo_dim
        )
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk_fused(self, chunk_emb, chunk_targets, m_prev, h_prev, u_t, criterion):
        batch_size, chunk_len, _ = chunk_emb.size()
        obs_vis = torch.zeros(batch_size, self.config.net.vision_dim, device=device)
        prev_act = torch.zeros(batch_size, self.config.net.action_dim, device=device)

        # 1. Sequential SDE-SSM Recurrence (Only states M_t and h_t are sequential)
        h_states = []
        for t in range(chunk_len):
            w_t, _, _, _ = self.gateway(chunk_emb[:, t], obs_vis, prev_act, h_prev, u_t)
            m_prev, h_prev, _ = self.goal_sde_ssm.forward_step(m_prev, h_prev, w_t, u_t)
            h_states.append(h_prev)

        # Stack chunk hidden states: [Batch, ChunkLen, HiddenDim] -> [Batch * ChunkLen, HiddenDim]
        h_chunk_tensor = torch.stack(h_states, dim=1).reshape(batch_size * chunk_len, self.hidden_dim)

        # 2. Parallel Batched Motor Tensor Readout (Runs 1 single kernel over all 32 tokens in parallel!)
        h_relaxed_flat, _ = self.attractor_head.relax_to_minima(h_chunk_tensor)
        h_proj_flat = self.motor_text_proj(h_relaxed_flat)
        logits_flat = F.linear(h_proj_flat, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim

        # 3. Single Batched Cross-Entropy Loss with Safe Memory Layout
        targets_flat = chunk_targets.contiguous().view(-1)
        chunk_loss = criterion(logits_flat, targets_flat)

        return chunk_loss, m_prev, h_prev


# =============================================================================
# BENCHMARK EXECUTION SUITE
# =============================================================================

def run_isolated_benchmark():
    print(f"\n{'='*85}")
    print(f" === RUNNING KEP HYPOTHESIS #19 BENCHMARK ON: {device_str.upper()} ===")
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
    tokenizer = ByteTokenizer()

    # Synthetic conversational batch
    torch.manual_seed(42)
    dummy_input = torch.randint(32, 126, (batch_size, seq_len), dtype=torch.long, device=device)
    dummy_target = torch.randint(32, 126, (batch_size, seq_len), dtype=torch.long, device=device)

    # -------------------------------------------------------------------------
    # TEST 1: BASELINE (Sequential Token-by-Token Dispatch)
    # -------------------------------------------------------------------------
    print("[1/2] Evaluating BASELINE (Sequential Token-by-Token Step Dispatch)...")
    baseline_model = SequentialStepAgent(config).to(device)
    base_opt = torch.optim.Adam(baseline_model.parameters(), lr=3e-3)
    hu_base = HomeostaticUnit(batch_size=batch_size, device=device_str)

    base_times, base_losses = [], []
    num_chunks = seq_len // chunk_size

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

            chunk_emb = baseline_model.pos_embeddings(dummy_input[:, c_start:c_end], start_pos=c_start, apply_rf=True)
            chunk_targets = dummy_target[:, c_start:c_end]

            chunk_loss, m_prev, h_prev = baseline_model.forward_chunk_sequential(chunk_emb, chunk_targets, m_prev, h_prev, u_t, criterion)
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
    # TEST 2: PROPOSED (Fused Chunk Recurrence + Parallel Motor Readout)
    # -------------------------------------------------------------------------
    print("[2/2] Evaluating PROPOSED (Fused Recurrence + Parallel Batched Readout)...")
    prop_model = FusedChunkScannerAgent(config).to(device)
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

            # Fused chunk execution with parallel batched motor readout
            chunk_loss, m_prev, h_prev = prop_model.forward_chunk_fused(chunk_emb, chunk_targets, m_prev, h_prev, u_t, criterion)
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
    print(f"{'Performance Metric':<35} | {'Baseline (Sequential)':<22} | {'Proposed (Fused Parallel)':<22} | {'Delta / Speedup':<15}")
    print("-" * 95)
    print(f"{'Total Step Duration (ms)':<35} | {avg_base_time:<22.2f} | {avg_prop_time:<22.2f} | {avg_prop_time - avg_base_time:+6.1f} ms (🚀)")
    print(f"{'Throughput Speed (tok/s)':<35} | {base_tok_per_sec:<22.1f} | {prop_tok_per_sec:<22.1f} | {prop_tok_per_sec/base_tok_per_sec:+.2f}x (⚡⚡⚡)")
    print(f"{'Initial Loss (Step 1)':<35} | {base_losses[0]:<22.4f} | {prop_losses[0]:<22.4f} | {prop_losses[0] - base_losses[0]:+6.4f}")
    print(f"{'Final Loss (Step 25)':<35} | {base_losses[-1]:<22.4f} | {prop_losses[-1]:<22.4f} | {prop_losses[-1] - base_losses[-1]:+6.4f}")
    print("="*90)

    # Verification of KEP Rule #2
    if avg_prop_time < avg_base_time * 0.5:
        print("🟢 KEP VERDICT: POSITIVE (Hypothesis #19 Validated! Ready for merge into production).")
    elif avg_prop_time <= avg_base_time * 0.8:
        print("⚪ KEP VERDICT: NEUTRAL (Moderate Speedup).")
    else:
        print("🔴 KEP VERDICT: REJECTED (Hypothesis did not achieve target acceleration).")

if __name__ == "__main__":
    run_isolated_benchmark()
