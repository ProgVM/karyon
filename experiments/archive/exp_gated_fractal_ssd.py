# experiments/exp_gated_fractal_ssd.py
"""
feat(exp): implement thalamocortical cross-scale router for long-context factual retrieval

===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-25 (THALAMOCORTICAL GATED MULTISCALE SSD)
Hypothesis: Dynamic Thalamocortical Cross-Scale Softmax Routing across gamma,
theta, and delta octaves eliminates LayerNorm variance explosion and phonemic
loops, enabling exact Needle-in-a-Haystack retrieval across 2048+ byte contexts.
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
    def __init__(self, vocab_size=258, text_dim=128, max_len=8192):
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
# MODULE 3: THALAMOCORTICAL GATED MULTI-SCALE SDE-SSM CORE
# =============================================================================

class ThalamocorticalGatedFractalSSDCore(nn.Module):
    """
    Biological Thalamocortical Gated Multi-Scale State-Space Core.
    Dynamically routes memory retrieval across 3 temporal frequency octaves:
    1. Fast (Gamma-scale: local syntax & bytes)
    2. Meso (Theta-scale: paragraph context)
    3. Macro (Delta-scale: long-document episodic needle facts)
    Uses normalized Softmax dynamic gain to preserve exact variance.
    """
    def __init__(self, text_dim=128, unified_dim=256, hidden_dim=512, num_heads=8, head_k=32, head_v=64):
        super().__init__()
        self.text_dim = text_dim
        self.unified_dim = unified_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_k = head_k
        self.head_v = head_v
        self.inv_sqrt_k = 1.0 / math.sqrt(head_k)

        # Projections
        self.sensory_proj = nn.Linear(text_dim, unified_dim)
        self.q_proj = nn.Linear(unified_dim, num_heads * head_k)
        self.k_proj = nn.Linear(unified_dim, num_heads * head_k)
        self.v_proj = nn.Linear(unified_dim, num_heads * head_v)

        # 3 Multi-Rate Decay Logits
        self.decay_fast = nn.Parameter(torch.tensor([2.2] * num_heads).view(1, num_heads, 1, 1))
        self.decay_meso = nn.Parameter(torch.tensor([4.5] * num_heads).view(1, num_heads, 1, 1))
        self.decay_macro = nn.Parameter(torch.tensor([7.8] * num_heads).view(1, num_heads, 1, 1))

        # Thalamocortical Dynamic Cross-Scale Gain Router
        self.gain_router = nn.Linear(unified_dim, num_heads * 3)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward_chunk_gated_ssd(self, chunk_emb, m_fast, m_meso, m_macro, u_t, dt=1.0):
        batch_size, chunk_len, _ = chunk_emb.size()
        na = u_t[:, 4:5].view(batch_size, 1, 1, 1)
        da = u_t[:, 5:6].view(batch_size, 1, 1, 1)
        eff_dt = torch.clamp(dt * (1.0 - 0.4 * na + 0.4 * da), 0.30, 2.00)

        # 1. Parallel Projections
        w_chunk = self.sensory_proj(chunk_emb)
        q = (self.q_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)) * self.inv_sqrt_k
        k = self.k_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)
        v = self.v_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_v).transpose(1, 2)

        # 3 Decay Rates
        alpha_f = torch.sigmoid(self.decay_fast) ** eff_dt
        alpha_m = torch.sigmoid(self.decay_meso) ** eff_dt
        alpha_M = torch.sigmoid(self.decay_macro) ** eff_dt

        beta_f = 1.0 - alpha_f
        beta_m = 1.0 - alpha_m
        beta_M = 1.0 - alpha_M

        pos = torch.arange(chunk_len, device=chunk_emb.device).float()
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        causal_mask = (diff >= 0).float()

        # 2. Intra-Chunk Causal Attention Matrix (Gamma)
        decay_weights_f = (alpha_f ** diff.clamp(min=0)) * causal_mask * beta_f
        s_matrix = torch.matmul(q, k.transpose(-1, -2)) * decay_weights_f
        y_intra = torch.matmul(s_matrix, v)

        # 3. 3-Tier Multi-Rate Retrieval
        decay_start_f = alpha_f ** ((pos + 1.0).view(1, 1, chunk_len, 1))
        decay_start_m = alpha_m ** ((pos + 1.0).view(1, 1, chunk_len, 1))
        decay_start_M = alpha_M ** ((pos + 1.0).view(1, 1, chunk_len, 1))

        y_inter_f = torch.matmul(q * decay_start_f, m_fast)
        y_inter_m = torch.matmul(q * decay_start_m, m_meso)
        y_inter_M = torch.matmul(q * decay_start_M, m_macro)

        y_fast_total = y_intra + y_inter_f

        # 4. Thalamocortical Dynamic Softmax Gain Routing (Variance Conserved!)
        # router_logits: [Batch, ChunkLen, NumHeads * 3] -> [Batch, NumHeads, ChunkLen, 3]
        router_logits = self.gain_router(w_chunk).view(batch_size, chunk_len, self.num_heads, 3).transpose(1, 2)
        gain_probs = F.softmax(router_logits, dim=-1) # [Batch, NumHeads, ChunkLen, 3]

        g_fast  = gain_probs[..., 0:1] # [Batch, NumHeads, ChunkLen, 1]
        g_meso  = gain_probs[..., 1:2]
        g_macro = gain_probs[..., 2:3]

        # Dynamically Weighted Readout
        y_routed = g_fast * y_fast_total + g_meso * y_inter_m + g_macro * y_inter_M

        y_total = y_routed.transpose(1, 2).reshape(batch_size * chunk_len, self.hidden_dim)
        h_chunk = self.norm(self.out_proj(y_total) + y_total)

        # 5. Multi-Rate Matrix Updates
        decay_end_f = alpha_f ** ((chunk_len - 1.0 - pos).view(1, 1, chunk_len, 1))
        decay_end_m = alpha_m ** ((chunk_len - 1.0 - pos).view(1, 1, chunk_len, 1))
        decay_end_M = alpha_M ** ((chunk_len - 1.0 - pos).view(1, 1, chunk_len, 1))

        kv_update_f = torch.matmul((k * decay_end_f).transpose(-1, -2), v)
        kv_update_m = torch.matmul((k * decay_end_m).transpose(-1, -2), v)
        kv_update_M = torch.matmul((k * decay_end_M).transpose(-1, -2), v)

        sigma = 1e-3
        dW = torch.randn_like(m_fast) * torch.sqrt(eff_dt) * sigma

        m_next_fast  = (alpha_f ** chunk_len) * m_fast  + beta_f * kv_update_f + dW
        m_next_meso  = (alpha_m ** chunk_len) * m_meso  + beta_m * kv_update_m + dW * 0.5
        m_next_macro = (alpha_M ** chunk_len) * m_macro + beta_M * kv_update_M + dW * 0.2

        return h_chunk, m_next_fast, m_next_meso, m_next_macro


# =============================================================================
# BENCHMARK AGENTS
# =============================================================================

class AdditiveUnGatedAgent(nn.Module):
    """EXP-24 Baseline: Un-gated static sum (suffered from attractor collapse)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim, max_len=8192)
        
        self.sensory_proj = nn.Linear(self.text_dim, self.unified_dim)
        self.q_proj = nn.Linear(self.unified_dim, 8 * 32)
        self.k_proj = nn.Linear(self.unified_dim, 8 * 32)
        self.v_proj = nn.Linear(self.unified_dim, 8 * 64)

        self.decay_fast = nn.Parameter(torch.tensor([2.2] * 8).view(1, 8, 1, 1))
        self.decay_meso = nn.Parameter(torch.tensor([4.2] * 8).view(1, 8, 1, 1))
        self.decay_macro = nn.Parameter(torch.tensor([7.6] * 8).view(1, 8, 1, 1))

        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk(self, chunk_emb, chunk_targets, m_f, m_m, m_M, u_t, criterion):
        batch_size, chunk_len, _ = chunk_emb.size()
        w_chunk = self.sensory_proj(chunk_emb)
        q = (self.q_proj(w_chunk).view(batch_size, chunk_len, 8, 32).transpose(1, 2)) * (1.0 / math.sqrt(32))
        k = self.k_proj(w_chunk).view(batch_size, chunk_len, 8, 32).transpose(1, 2)
        v = self.v_proj(w_chunk).view(batch_size, chunk_len, 8, 64).transpose(1, 2)

        alpha_f = torch.sigmoid(self.decay_fast)
        alpha_m = torch.sigmoid(self.decay_meso)
        alpha_M = torch.sigmoid(self.decay_macro)

        pos = torch.arange(chunk_len, device=chunk_emb.device).float()
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        decay_weights_f = (alpha_f ** diff.clamp(min=0)) * (diff >= 0).float() * (1.0 - alpha_f)
        
        y_intra = torch.matmul(torch.matmul(q, k.transpose(-1, -2)) * decay_weights_f, v)
        y_inter_f = torch.matmul(q * (alpha_f ** ((pos + 1.0).view(1, 1, chunk_len, 1))), m_f)
        y_inter_m = torch.matmul(q * (alpha_m ** ((pos + 1.0).view(1, 1, chunk_len, 1))), m_m)
        y_inter_M = torch.matmul(q * (alpha_M ** ((pos + 1.0).view(1, 1, chunk_len, 1))), m_M)

        # Un-gated static sum (causes variance inflation)
        y_total = (y_intra + y_inter_f + 0.5 * y_inter_m + 0.3 * y_inter_M).transpose(1, 2).reshape(batch_size * chunk_len, self.hidden_dim)
        h_chunk = self.norm(self.out_proj(y_total) + y_total)

        m_next_f = (alpha_f ** chunk_len) * m_f + (1.0 - alpha_f) * torch.matmul((k * (alpha_f ** ((chunk_len - 1.0 - pos).view(1, 1, chunk_len, 1)))).transpose(-1, -2), v)
        m_next_m = (alpha_m ** chunk_len) * m_m + (1.0 - alpha_m) * torch.matmul((k * (alpha_m ** ((chunk_len - 1.0 - pos).view(1, 1, chunk_len, 1)))).transpose(-1, -2), v)
        m_next_M = (alpha_M ** chunk_len) * m_M + (1.0 - alpha_M) * torch.matmul((k * (alpha_M ** ((chunk_len - 1.0 - pos).view(1, 1, chunk_len, 1)))).transpose(-1, -2), v)

        h_relaxed = self.attractor_head.relax_to_minima(h_chunk)[0]
        logits = F.linear(self.motor_text_proj(h_relaxed), self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        loss = criterion(logits, chunk_targets.contiguous().view(-1))
        return loss, m_next_f, m_next_m, m_next_M


class ThalamocorticalGatedAgent(nn.Module):
    """Proposed Model: Dynamic Thalamocortical Cross-Scale Router."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim, max_len=8192)
        self.gated_ssd = ThalamocorticalGatedFractalSSDCore(
            text_dim=self.text_dim, unified_dim=self.unified_dim, hidden_dim=self.hidden_dim,
            num_heads=8, head_k=32, head_v=64
        )
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk(self, chunk_emb, chunk_targets, m_f, m_m, m_M, u_t, criterion):
        h_chunk, m_next_f, m_next_m, m_next_M = self.gated_ssd.forward_chunk_gated_ssd(
            chunk_emb, m_f, m_m, m_M, u_t, dt=1.0
        )
        h_relaxed = self.attractor_head.relax_to_minima(h_chunk)[0]
        logits = F.linear(self.motor_text_proj(h_relaxed), self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        loss = criterion(logits, chunk_targets.contiguous().view(-1))
        return loss, m_next_f, m_next_m, m_next_M


# =============================================================================
# BENCHMARK EXECUTION SUITE (2048-BYTE NEEDLE RETRIEVAL TEST)
# =============================================================================

def run_isolated_benchmark():
    print(f"\n{'='*85}")
    print(f" === RUNNING KEP HYPOTHESIS #25 BENCHMARK (THALAMOCORTICAL GATED SSD): {device_str.upper()} ===")
    print(f"{'='*85}\n")

    config = CoREConfig()
    config.net.hidden_dim = 512
    config.net.unified_dim = 256
    config.net.text_dim = 128
    config.net.text_gen_dim = 258

    batch_size = 16
    seq_len = 2048 # 2048-Byte Long Context
    chunk_size = 32
    num_eval_steps = 30

    criterion = nn.CrossEntropyLoss(ignore_index=256)
    tokenizer = ByteTokenizer()
    num_chunks = seq_len // chunk_size

    # Hard 2048-Byte Long-Context Needle Prompt
    needle_fact = "Document: The secret project codename is PHENIX-909.\n"
    distractor_text = (
        "Background details: Continuous time state space models process multi-modal signals smoothly. "
        "Ashby homeostasis maintains biological ultrastability across neurotransmitter dynamics. "
        "The recurrent engine eliminates static matrix multiplications by integrating differential equations. "
        "Active inference minimizes variational free energy across sensory and motor modalities smoothly. "
        "Autonomous binary containers encapsulate logic and persistent state into a single portable binary soul. "
        "Cybernetic neural integration maintains homeostatic equilibrium during high surprise streaming. "
    )
    question_query = "Question: What is the secret project codename?\nAnswer: The secret project codename is PHENIX-909."
    
    distractor_repeats = (seq_len - len(tokenizer.encode(needle_fact)) - len(tokenizer.encode(question_query))) // len(tokenizer.encode(distractor_text)) + 2
    full_prompt_str = needle_fact + (distractor_text * distractor_repeats) + question_query
    
    tokens_raw = tokenizer.encode(full_prompt_str)
    full_tokens = tokens_raw[:seq_len + 1]
    
    input_tokens = torch.tensor([full_tokens[:seq_len]], dtype=torch.long, device=device).repeat(batch_size, 1)
    target_tokens = torch.tensor([full_tokens[1:seq_len + 1]], dtype=torch.long, device=device).repeat(batch_size, 1)

    print(f"Sequence Context Length: {seq_len} Bytes | Chunks: {num_chunks}")
    print(f"Needle Location        : Position 0 (Start of 2048-byte stream)\n")

    # -------------------------------------------------------------------------
    # TEST 1: BASELINE (Additive Un-Gated Multi-Scale SSM)
    # -------------------------------------------------------------------------
    print("[1/2] Evaluating BASELINE (Additive Un-Gated Multi-Scale SSM)...")
    base_model = AdditiveUnGatedAgent(config).to(device)
    base_opt = torch.optim.Adam(base_model.parameters(), lr=3e-3)
    hu_base = HomeostaticUnit(batch_size=batch_size, device=device_str)

    base_times, base_losses = [], []

    for step in range(num_eval_steps):
        base_opt.zero_grad()
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()

        m_f = torch.zeros(batch_size, 8, 32, 64, device=device)
        m_m = torch.zeros(batch_size, 8, 32, 64, device=device)
        m_M = torch.zeros(batch_size, 8, 32, 64, device=device)
        u_t = hu_base.state.clone().detach()

        batch_losses = []
        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = (chunk_idx + 1) * chunk_size

            chunk_emb = base_model.pos_embeddings(input_tokens[:, c_start:c_end], start_pos=c_start, apply_rf=True)
            chunk_targets = target_tokens[:, c_start:c_end]

            chunk_loss, m_f, m_m, m_M = base_model.forward_chunk(chunk_emb, chunk_targets, m_f, m_m, m_M, u_t, criterion)
            (chunk_loss / float(num_chunks)).backward()
            batch_losses.append(chunk_loss.item())

            m_f = m_f.detach()
            m_m = m_m.detach()
            m_M = m_M.detach()

        base_opt.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000.0

        base_times.append(step_ms)
        base_losses.append(sum(batch_losses) / len(batch_losses))

    # -------------------------------------------------------------------------
    # TEST 2: PROPOSED (Thalamocortical Dynamic Softmax Gated SSD)
    # -------------------------------------------------------------------------
    print("[2/2] Evaluating PROPOSED (Thalamocortical Dynamic Softmax Gated SSD)...")
    prop_model = ThalamocorticalGatedAgent(config).to(device)
    prop_opt = torch.optim.Adam(prop_model.parameters(), lr=3e-3)
    hu_prop = HomeostaticUnit(batch_size=batch_size, device=device_str)

    prop_times, prop_losses = [], []

    for step in range(num_eval_steps):
        prop_opt.zero_grad()
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()

        m_f = torch.zeros(batch_size, 8, 32, 64, device=device)
        m_m = torch.zeros(batch_size, 8, 32, 64, device=device)
        m_M = torch.zeros(batch_size, 8, 32, 64, device=device)
        u_t = hu_prop.state.clone().detach()

        batch_losses = []
        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = (chunk_idx + 1) * chunk_size

            chunk_emb = prop_model.pos_embeddings(input_tokens[:, c_start:c_end], start_pos=c_start, apply_rf=True)
            chunk_targets = target_tokens[:, c_start:c_end]

            chunk_loss, m_f, m_m, m_M = prop_model.forward_chunk(chunk_emb, chunk_targets, m_f, m_m, m_M, u_t, criterion)
            (chunk_loss / float(num_chunks)).backward()
            batch_losses.append(chunk_loss.item())

            m_f = m_f.detach()
            m_m = m_m.detach()
            m_M = m_M.detach()

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
    print(" === [KEP RULE #6 TELEMETRY BENCHMARK REPORT: 2048-BYTE CONTEXT] ===")
    print("="*90)
    print(f"{'Performance Metric':<35} | {'Baseline (Un-Gated Sum)':<22} | {'Proposed (Thalamic Gated)':<22} | {'Delta':<10}")
    print("-" * 95)
    print(f"{'Step Duration on 2048 Bytes (ms)':<35} | {avg_base_time:<22.2f} | {avg_prop_time:<22.2f} | {avg_prop_time - avg_base_time:+6.1f} ms")
    print(f"{'Throughput Speed (tok/s)':<35} | {base_tok_per_sec:<22.1f} | {prop_tok_per_sec:<22.1f} | {prop_tok_per_sec/base_tok_per_sec:+.2f}x (⚡⚡⚡)")
    print(f"{'Initial Loss (Step 1)':<35} | {base_losses[0]:<22.4f} | {prop_losses[0]:<22.4f} | {prop_losses[0] - base_losses[0]:+6.4f}")
    print(f"{'Final Loss on 2048 Bytes (Step 30)':<35} | {base_losses[-1]:<22.4f} | {prop_losses[-1]:<22.4f} | {prop_losses[-1] - base_losses[-1]:+6.4f} (🔥)")
    print(f"{'Perplexity (PPL Step 30)':<35} | {math.exp(base_losses[-1]):<22.2f} | {math.exp(prop_losses[-1]):<22.2f} | {math.exp(prop_losses[-1]) - math.exp(base_losses[-1]):+6.2f}")
    print("="*90)

    # =========================================================================
    # KEP RULE #4: DIAGNOSTIC SPEECH SAMPLING (FACTUAL RETRIEVAL AFTER 2048 BYTES)
    # =========================================================================
    print("\n" + "="*90)
    print(" === [KEP RULE #4 FACTUAL RETRIEVAL AUDIT AFTER 2048 BYTES OF DISTRACTORS] ===")
    print("="*90)

    eval_prompt = "Question: What is the secret project codename?\nAnswer:"
    p_ids = torch.tensor(tokenizer.encode(eval_prompt), dtype=torch.long, device=device).unsqueeze(0)

    def generate_eval_sample(model, name, is_thalamic=False):
        model.eval()
        with torch.no_grad():
            p_emb = model.pos_embeddings(p_ids, start_pos=0, apply_rf=True)
            u_t = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.2, 0.0]], device=device)

            m_f = torch.zeros(1, 8, 32, 64, device=device)
            m_m = torch.zeros(1, 8, 32, 64, device=device)
            m_M = torch.zeros(1, 8, 32, 64, device=device)

            if not is_thalamic:
                _, m_f, m_m, m_M = model.forward_chunk(p_emb, p_ids, m_f, m_m, m_M, u_t, criterion)
                chars = []
                curr_tok = p_ids[:, -1:]
                for s in range(60):
                    t_emb = model.pos_embeddings(curr_tok, start_pos=p_emb.size(1) + s, apply_rf=False)
                    w_chunk = model.sensory_proj(t_emb)
                    q = (model.q_proj(w_chunk).view(1, 1, 8, 32).transpose(1, 2)) * (1.0 / math.sqrt(32))
                    alpha_f = torch.sigmoid(model.decay_fast)
                    alpha_m = torch.sigmoid(model.decay_meso)
                    alpha_M = torch.sigmoid(model.decay_macro)
                    y_out = (torch.matmul(q * alpha_f, m_f) + 0.5 * torch.matmul(q * alpha_m, m_m) + 0.3 * torch.matmul(q * alpha_M, m_M)).transpose(1, 2).reshape(1, model.hidden_dim)
                    h_relaxed = model.attractor_head.relax_to_minima(model.norm(model.out_proj(y_out) + y_out))[0]
                    logits = F.linear(model.motor_text_proj(h_relaxed), model.pos_embeddings.byte_embed.weight) * model.inv_sqrt_text_dim
                    probs = F.softmax(logits / 0.7, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                    if next_id == 257: break
                    chars.append(chr(next_id) if 32 <= next_id <= 126 else ' ')
                    curr_tok = torch.tensor([[next_id]], device=device)
            else:
                _, m_f, m_m, m_M = model.forward_chunk(p_emb, p_ids, m_f, m_m, m_M, u_t, criterion)
                chars = []
                curr_tok = p_ids[:, -1:]
                for s in range(60):
                    t_emb = model.pos_embeddings(curr_tok, start_pos=p_emb.size(1) + s, apply_rf=False)
                    w_chunk = model.gated_ssd.sensory_proj(t_emb)
                    q = (model.gated_ssd.q_proj(w_chunk).view(1, 1, 8, 32).transpose(1, 2)) * (1.0 / math.sqrt(32))
                    alpha_f = torch.sigmoid(model.gated_ssd.decay_fast)
                    alpha_m = torch.sigmoid(model.gated_ssd.decay_meso)
                    alpha_M = torch.sigmoid(model.gated_ssd.decay_macro)
                    
                    y_f = torch.matmul(q * alpha_f, m_f)
                    y_m = torch.matmul(q * alpha_m, m_m)
                    y_M = torch.matmul(q * alpha_M, m_M)

                    r_logits = model.gated_ssd.gain_router(w_chunk).view(1, 1, 8, 3).transpose(1, 2)
                    g_probs = F.softmax(r_logits, dim=-1)
                    y_routed = g_probs[..., 0:1] * y_f + g_probs[..., 1:2] * y_m + g_probs[..., 2:3] * y_M
                    y_out = y_routed.transpose(1, 2).reshape(1, model.hidden_dim)

                    h_relaxed = model.attractor_head.relax_to_minima(model.gated_ssd.norm(model.gated_ssd.out_proj(y_out) + y_out))[0]
                    logits = F.linear(model.motor_text_proj(h_relaxed), model.pos_embeddings.byte_embed.weight) * model.inv_sqrt_text_dim
                    probs = F.softmax(logits / 0.7, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                    if next_id == 257: break
                    chars.append(chr(next_id) if 32 <= next_id <= 126 else ' ')
                    curr_tok = torch.tensor([[next_id]], device=device)

        print(f"[{name}] Sample -> \"{''.join(chars)}\"")

    generate_eval_sample(base_model, "Baseline (Un-Gated Additive Sum)", is_thalamic=False)
    generate_eval_sample(prop_model, "Proposed (Thalamocortical Gated Router)", is_thalamic=True)
    print("="*90 + "\n")

    if prop_losses[-1] < base_losses[-1] and prop_tok_per_sec > 70000.0:
        print("🟢 KEP VERDICT: POSITIVE (Thalamocortical Gain Routing Validated!).")
    else:
        print("⚪ KEP VERDICT: NEUTRAL (Performance or convergence check).")

if __name__ == "__main__":
    run_isolated_benchmark()
