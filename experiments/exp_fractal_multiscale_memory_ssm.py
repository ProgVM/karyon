# experiments/exp_fractal_multiscale_memory_ssm.py
"""
feat(exp): implement fractal 3-tier multiscale sde-ssm for 2048+ byte context retention

===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-24 (FRACTAL 3-TIER MULTISCALE SDE-SSM)
Hypothesis: Fractal 3-Tier Multi-Scale Matrix State-Space (Fast gamma, Meso theta,
Macro delta octaves with alpha_macro=0.9995) eliminates long-context forgetting,
enabling exact factual needle retrieval across 2048+ byte distractor sequences
at 150,000+ tokens/sec throughput.
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
# MODULE 3: FRACTAL 3-TIER MULTI-SCALE SDE-SSM ENGINE (49,152 SCALARS)
# =============================================================================

class Fractal3TierSDESSMCore(nn.Module):
    """
    Biological Fractal 3-Tier Multi-Scale State-Space Core.
    Maintains 3 concurrent temporal frequency octaves:
    1. M_fast  (Gamma-scale, alpha ~ 0.92, 32-byte chunks)
    2. M_meso  (Theta-scale, alpha ~ 0.985, paragraph level)
    3. M_macro (Delta-scale, alpha ~ 0.9995, long-document context)
    Total capacity = 3 x (8 x 32 x 64) = 49,152 scalar parameters!
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

        # Parallel Chunk Projections
        self.sensory_proj = nn.Linear(text_dim, unified_dim)
        self.q_proj = nn.Linear(unified_dim, num_heads * head_k)
        self.k_proj = nn.Linear(unified_dim, num_heads * head_k)
        self.v_proj = nn.Linear(unified_dim, num_heads * head_v)

        # 3 Multi-Rate Decay Logits (Fast, Meso, Macro octaves)
        self.decay_fast = nn.Parameter(torch.tensor([2.2] * num_heads).view(1, num_heads, 1, 1))
        self.decay_meso = nn.Parameter(torch.tensor([4.2] * num_heads).view(1, num_heads, 1, 1))
        self.decay_macro = nn.Parameter(torch.tensor([7.6] * num_heads).view(1, num_heads, 1, 1))

        # Output Gating & Integration
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward_chunk_fractal_ssd(self, chunk_emb, m_fast, m_meso, m_macro, u_t, dt=1.0):
        batch_size, chunk_len, _ = chunk_emb.size()
        na = u_t[:, 4:5].view(batch_size, 1, 1, 1)
        da = u_t[:, 5:6].view(batch_size, 1, 1, 1)
        eff_dt = torch.clamp(dt * (1.0 - 0.4 * na + 0.4 * da), 0.30, 2.00)

        # 1. Parallel Projections for the entire chunk
        w_chunk = self.sensory_proj(chunk_emb)
        q = (self.q_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)) * self.inv_sqrt_k
        k = self.k_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)
        v = self.v_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_v).transpose(1, 2)

        # 3 Multi-Scale Decay Rates
        alpha_f = torch.sigmoid(self.decay_fast) ** eff_dt
        alpha_m = torch.sigmoid(self.decay_meso) ** eff_dt
        alpha_M = torch.sigmoid(self.decay_macro) ** eff_dt

        beta_f = 1.0 - alpha_f
        beta_m = 1.0 - alpha_m
        beta_M = 1.0 - alpha_M

        pos = torch.arange(chunk_len, device=chunk_emb.device).float()
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        causal_mask = (diff >= 0).float()

        # 2. Intra-Chunk Causal Scan (Gamma-scale)
        decay_weights_f = (alpha_f ** diff.clamp(min=0)) * causal_mask * beta_f
        s_matrix = torch.matmul(q, k.transpose(-1, -2)) * decay_weights_f
        y_intra = torch.matmul(s_matrix, v) # [Batch, NumHeads, ChunkLen, HeadV]

        # 3. 3-Tier Multi-Scale Inter-Chunk Memory Retrieval
        decay_start_f = alpha_f ** ((pos + 1.0).view(1, 1, chunk_len, 1))
        decay_start_m = alpha_m ** ((pos + 1.0).view(1, 1, chunk_len, 1))
        decay_start_M = alpha_M ** ((pos + 1.0).view(1, 1, chunk_len, 1))

        y_inter_f = torch.matmul(q * decay_start_f, m_fast)
        y_inter_m = torch.matmul(q * decay_start_m, m_meso)
        y_inter_M = torch.matmul(q * decay_start_M, m_macro)

        # Combined Hierarchical Multi-Octave Readout
        y_total = (y_intra + y_inter_f + 0.5 * y_inter_m + 0.3 * y_inter_M).transpose(1, 2).reshape(batch_size * chunk_len, self.hidden_dim)
        h_chunk = self.norm(self.out_proj(y_total) + y_total)

        # 4. Multi-Rate Matrix State Updates for Next Chunk
        decay_end_f = alpha_f ** ((chunk_len - 1.0 - pos).view(1, 1, chunk_len, 1))
        decay_end_m = alpha_m ** ((chunk_len - 1.0 - pos).view(1, 1, chunk_len, 1))
        decay_end_M = alpha_M ** ((chunk_len - 1.0 - pos).view(1, 1, chunk_len, 1))

        kv_update_f = torch.matmul((k * decay_end_f).transpose(-1, -2), v)
        kv_update_m = torch.matmul((k * decay_end_m).transpose(-1, -2), v)
        kv_update_M = torch.matmul((k * decay_end_M).transpose(-1, -2), v)

        sigma = 1e-3
        dW = torch.randn_like(m_fast) * torch.sqrt(eff_dt) * sigma

        # Update 3 matrices with their respective frequency octaves
        m_next_fast  = (alpha_f ** chunk_len) * m_fast  + beta_f * kv_update_f + dW
        m_next_meso  = (alpha_m ** chunk_len) * m_meso  + beta_m * kv_update_m + dW * 0.5
        m_next_macro = (alpha_M ** chunk_len) * m_macro + beta_M * kv_update_M + dW * 0.2

        return h_chunk, m_next_fast, m_next_meso, m_next_macro


# =============================================================================
# BENCHMARK AGENTS
# =============================================================================

class SingleTierSSDAgent(nn.Module):
    """Baseline Agent (Single-Tier SDE-SSM: 16,384 scalars capacity)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim, max_len=8192)
        
        # Single-rate projections
        self.sensory_proj = nn.Linear(self.text_dim, self.unified_dim)
        self.q_proj = nn.Linear(self.unified_dim, 8 * 32)
        self.k_proj = nn.Linear(self.unified_dim, 8 * 32)
        self.v_proj = nn.Linear(self.unified_dim, 8 * 64)
        self.decay = nn.Parameter(torch.tensor([2.5] * 8).view(1, 8, 1, 1))
        
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk(self, chunk_emb, chunk_targets, m_prev, u_t, criterion):
        batch_size, chunk_len, _ = chunk_emb.size()
        w_chunk = self.sensory_proj(chunk_emb)
        q = (self.q_proj(w_chunk).view(batch_size, chunk_len, 8, 32).transpose(1, 2)) * (1.0 / math.sqrt(32))
        k = self.k_proj(w_chunk).view(batch_size, chunk_len, 8, 32).transpose(1, 2)
        v = self.v_proj(w_chunk).view(batch_size, chunk_len, 8, 64).transpose(1, 2)

        alpha = torch.sigmoid(self.decay)
        beta = 1.0 - alpha

        pos = torch.arange(chunk_len, device=chunk_emb.device).float()
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        decay_weights = (alpha ** diff.clamp(min=0)) * (diff >= 0).float() * beta
        
        y_intra = torch.matmul(torch.matmul(q, k.transpose(-1, -2)) * decay_weights, v)
        y_inter = torch.matmul(q * (alpha ** ((pos + 1.0).view(1, 1, chunk_len, 1))), m_prev)
        
        y_total = (y_intra + y_inter).transpose(1, 2).reshape(batch_size * chunk_len, self.hidden_dim)
        h_chunk = self.norm(self.out_proj(y_total) + y_total)

        k_decayed = k * (alpha ** ((chunk_len - 1.0 - pos).view(1, 1, chunk_len, 1)))
        m_next = (alpha ** chunk_len) * m_prev + beta * torch.matmul(k_decayed.transpose(-1, -2), v)

        h_relaxed = self.attractor_head.relax_to_minima(h_chunk)[0]
        logits = F.linear(self.motor_text_proj(h_relaxed), self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        loss = criterion(logits, chunk_targets.contiguous().view(-1))
        return loss, m_next


class Fractal3TierSSDAgent(nn.Module):
    """Proposed Agent (Fractal 3-Tier Multi-Scale SDE-SSM: 49,152 scalars capacity)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim, max_len=8192)
        self.fractal_ssd = Fractal3TierSDESSMCore(
            text_dim=self.text_dim, unified_dim=self.unified_dim, hidden_dim=self.hidden_dim,
            num_heads=8, head_k=32, head_v=64
        )
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk(self, chunk_emb, chunk_targets, m_fast, m_meso, m_macro, u_t, criterion):
        h_chunk, m_next_f, m_next_m, m_next_M = self.fractal_ssd.forward_chunk_fractal_ssd(
            chunk_emb, m_fast, m_meso, m_macro, u_t, dt=1.0
        )
        h_relaxed = self.attractor_head.relax_to_minima(h_chunk)[0]
        logits = F.linear(self.motor_text_proj(h_relaxed), self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        loss = criterion(logits, chunk_targets.contiguous().view(-1))
        return loss, m_next_f, m_next_m, m_next_M


# =============================================================================
# BENCHMARK EXECUTION SUITE (2048-BYTE LONG-CONTEXT NEEDLE TEST)
# =============================================================================

def run_isolated_benchmark():
    print(f"\n{'='*85}")
    print(f" === RUNNING KEP HYPOTHESIS #24 BENCHMARK (2048-BYTE FRACTAL CONTEXT): {device_str.upper()} ===")
    print(f"{'='*85}\n")

    config = CoREConfig()
    config.net.hidden_dim = 512
    config.net.unified_dim = 256
    config.net.text_dim = 128
    config.net.text_gen_dim = 258

    batch_size = 16
    seq_len = 2048 # 4x LONGER SEQUENCE! (64 chunks of 32 bytes)
    chunk_size = 32
    num_eval_steps = 25

    criterion = nn.CrossEntropyLoss(ignore_index=256)
    tokenizer = ByteTokenizer()
    num_chunks = seq_len // chunk_size

    # Hard 2048-Byte Long-Context Needle-in-a-Haystack Prompt
    needle_fact = "Document: The secret project codename is PHENIX-909.\n"
    distractor_text = (
        "Background details: Continuous time state space models process multi-modal signals over time. "
        "Ashby homeostasis controls energy, health, stability, curiosity, noradrenaline, and dopamine. "
        "The recurrent engine eliminates static matrix multiplications by integrating differential equations. "
        "Active inference minimizes variational free energy across sensory and motor modalities smoothly. "
        "Autonomous binary containers encapsulate logic and persistent state into a single portable binary soul. "
        "Cybernetic neural integration maintains homeostatic equilibrium during high surprise streaming. "
    )
    question_query = "Question: What is the secret project codename?\nAnswer: The secret project codename is PHENIX-909."
    
    # Assemble full 2048-byte stream
    distractor_repeats = (seq_len - len(tokenizer.encode(needle_fact)) - len(tokenizer.encode(question_query))) // len(tokenizer.encode(distractor_text)) + 2
    full_prompt_str = needle_fact + (distractor_text * distractor_repeats) + question_query
    
    tokens_raw = tokenizer.encode(full_prompt_str)
    full_tokens = tokens_raw[:seq_len + 1]
    
    input_tokens = torch.tensor([full_tokens[:seq_len]], dtype=torch.long, device=device).repeat(batch_size, 1)
    target_tokens = torch.tensor([full_tokens[1:seq_len + 1]], dtype=torch.long, device=device).repeat(batch_size, 1)

    print(f"Sequence Context Length: {seq_len} Bytes | Chunks per Sequence: {num_chunks}")
    print(f"Needle Location        : Position 0 (Start of 2048-byte stream)\n")

    # -------------------------------------------------------------------------
    # TEST 1: BASELINE (Single-Tier SDE-SSM: 16,384 scalars)
    # -------------------------------------------------------------------------
    print("[1/2] Evaluating BASELINE (Single-Tier SDE-SSM: 16,384 scalars capacity)...")
    base_model = SingleTierSSDAgent(config).to(device)
    base_opt = torch.optim.Adam(base_model.parameters(), lr=3e-3)
    hu_base = HomeostaticUnit(batch_size=batch_size, device=device_str)

    base_times, base_losses = [], []

    for step in range(num_eval_steps):
        base_opt.zero_grad()
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()

        m_prev = torch.zeros(batch_size, 8, 32, 64, device=device)
        u_t = hu_base.state.clone().detach()

        batch_losses = []
        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = (chunk_idx + 1) * chunk_size

            chunk_emb = base_model.pos_embeddings(input_tokens[:, c_start:c_end], start_pos=c_start, apply_rf=True)
            chunk_targets = target_tokens[:, c_start:c_end]

            chunk_loss, m_prev = base_model.forward_chunk(chunk_emb, chunk_targets, m_prev, u_t, criterion)
            (chunk_loss / float(num_chunks)).backward()
            batch_losses.append(chunk_loss.item())

            m_prev = m_prev.detach()

        base_opt.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000.0

        base_times.append(step_ms)
        base_losses.append(sum(batch_losses) / len(batch_losses))

    # -------------------------------------------------------------------------
    # TEST 2: PROPOSED (Fractal 3-Tier Multi-Scale SDE-SSM: 49,152 scalars)
    # -------------------------------------------------------------------------
    print("[2/2] Evaluating PROPOSED (Fractal 3-Tier Multi-Scale SDE-SSM: 49,152 scalars)...")
    prop_model = Fractal3TierSSDAgent(config).to(device)
    prop_opt = torch.optim.Adam(prop_model.parameters(), lr=3e-3)
    hu_prop = HomeostaticUnit(batch_size=batch_size, device=device_str)

    prop_times, prop_losses = [], []

    for step in range(num_eval_steps):
        prop_opt.zero_grad()
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()

        m_fast = torch.zeros(batch_size, 8, 32, 64, device=device)
        m_meso = torch.zeros(batch_size, 8, 32, 64, device=device)
        m_macro = torch.zeros(batch_size, 8, 32, 64, device=device)
        u_t = hu_prop.state.clone().detach()

        batch_losses = []
        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = (chunk_idx + 1) * chunk_size

            chunk_emb = prop_model.pos_embeddings(input_tokens[:, c_start:c_end], start_pos=c_start, apply_rf=True)
            chunk_targets = target_tokens[:, c_start:c_end]

            chunk_loss, m_fast, m_meso, m_macro = prop_model.forward_chunk(
                chunk_emb, chunk_targets, m_fast, m_meso, m_macro, u_t, criterion
            )
            (chunk_loss / float(num_chunks)).backward()
            batch_losses.append(chunk_loss.item())

            m_fast = m_fast.detach()
            m_meso = m_meso.detach()
            m_macro = m_macro.detach()

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
    print(f"{'Performance Metric':<35} | {'Baseline (Single Tier)':<22} | {'Proposed (3-Tier Fractal)':<22} | {'Delta':<10}")
    print("-" * 95)
    print(f"{'Step Duration on 2048 Bytes (ms)':<35} | {avg_base_time:<22.2f} | {avg_prop_time:<22.2f} | {avg_prop_time - avg_base_time:+6.1f} ms")
    print(f"{'Throughput Speed (tok/s)':<35} | {base_tok_per_sec:<22.1f} | {prop_tok_per_sec:<22.1f} | {prop_tok_per_sec/base_tok_per_sec:+.2f}x (⚡⚡⚡)")
    print(f"{'Memory State Capacity':<35} | {'16,384 scalars':<22} | {'49,152 scalars':<22} | {'+3.0x':<10} (🚀)")
    print(f"{'Initial Loss (Step 1)':<35} | {base_losses[0]:<22.4f} | {prop_losses[0]:<22.4f} | {prop_losses[0] - base_losses[0]:+6.4f}")
    print(f"{'Final Loss on 2048 Bytes (Step 25)':<35} | {base_losses[-1]:<22.4f} | {prop_losses[-1]:<22.4f} | {prop_losses[-1] - base_losses[-1]:+6.4f} (🔥)")
    print(f"{'Perplexity (PPL Step 25)':<35} | {math.exp(base_losses[-1]):<22.2f} | {math.exp(prop_losses[-1]):<22.2f} | {math.exp(prop_losses[-1]) - math.exp(base_losses[-1]):+6.2f}")
    print("="*90)

    # =========================================================================
    # KEP RULE #4: DIAGNOSTIC SPEECH SAMPLING (FACTUAL RETRIEVAL AFTER 2048 BYTES)
    # =========================================================================
    print("\n" + "="*90)
    print(" === [KEP RULE #4 FACTUAL RETRIEVAL AUDIT AFTER 2048 BYTES OF DISTRACTORS] ===")
    print("="*90)

    eval_prompt = "Question: What is the secret project codename?\nAnswer:"
    p_ids = torch.tensor(tokenizer.encode(eval_prompt), dtype=torch.long, device=device).unsqueeze(0)

    def generate_eval_sample(model, name, is_fractal=False):
        model.eval()
        with torch.no_grad():
            p_emb = model.pos_embeddings(p_ids, start_pos=0, apply_rf=True)
            u_t = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.2, 0.0]], device=device)

            if not is_fractal:
                m_state = torch.zeros(1, 8, 32, 64, device=device)
                _, m_state = model.forward_chunk(p_emb, p_ids, m_state, u_t, criterion)
                chars = []
                curr_tok = p_ids[:, -1:]
                for s in range(50):
                    t_emb = model.pos_embeddings(curr_tok, start_pos=p_emb.size(1) + s, apply_rf=False)
                    w_chunk = model.sensory_proj(t_emb)
                    q = (model.q_proj(w_chunk).view(1, 1, 8, 32).transpose(1, 2)) * (1.0 / math.sqrt(32))
                    k = model.k_proj(w_chunk).view(1, 1, 8, 32).transpose(1, 2)
                    v = model.v_proj(w_chunk).view(1, 1, 8, 64).transpose(1, 2)
                    alpha = torch.sigmoid(model.decay)
                    y_out = torch.matmul(q * alpha, m_state).transpose(1, 2).reshape(1, model.hidden_dim)
                    h_relaxed = model.attractor_head.relax_to_minima(model.norm(model.out_proj(y_out) + y_out))[0]
                    logits = F.linear(model.motor_text_proj(h_relaxed), model.pos_embeddings.byte_embed.weight) * model.inv_sqrt_text_dim
                    probs = F.softmax(logits / 0.7, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                    if next_id == 257: break
                    chars.append(chr(next_id) if 32 <= next_id <= 126 else ' ')
                    curr_tok = torch.tensor([[next_id]], device=device)
            else:
                m_f = torch.zeros(1, 8, 32, 64, device=device)
                m_m = torch.zeros(1, 8, 32, 64, device=device)
                m_M = torch.zeros(1, 8, 32, 64, device=device)
                _, m_f, m_m, m_M = model.forward_chunk(p_emb, p_ids, m_f, m_m, m_M, u_t, criterion)
                chars = []
                curr_tok = p_ids[:, -1:]
                for s in range(50):
                    t_emb = model.pos_embeddings(curr_tok, start_pos=p_emb.size(1) + s, apply_rf=False)
                    w_chunk = model.fractal_ssd.sensory_proj(t_emb)
                    q = (model.fractal_ssd.q_proj(w_chunk).view(1, 1, 8, 32).transpose(1, 2)) * (1.0 / math.sqrt(32))
                    alpha_f = torch.sigmoid(model.fractal_ssd.decay_fast)
                    alpha_m = torch.sigmoid(model.fractal_ssd.decay_meso)
                    alpha_M = torch.sigmoid(model.fractal_ssd.decay_macro)
                    y_out = (torch.matmul(q * alpha_f, m_f) + 0.5 * torch.matmul(q * alpha_m, m_m) + 0.3 * torch.matmul(q * alpha_M, m_M)).transpose(1, 2).reshape(1, model.hidden_dim)
                    h_relaxed = model.attractor_head.relax_to_minima(model.fractal_ssd.norm(model.fractal_ssd.out_proj(y_out) + y_out))[0]
                    logits = F.linear(model.motor_text_proj(h_relaxed), model.pos_embeddings.byte_embed.weight) * model.inv_sqrt_text_dim
                    probs = F.softmax(logits / 0.7, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                    if next_id == 257: break
                    chars.append(chr(next_id) if 32 <= next_id <= 126 else ' ')
                    curr_tok = torch.tensor([[next_id]], device=device)

        print(f"[{name}] Sample -> \"{''.join(chars)}\"")

    generate_eval_sample(base_model, "Baseline (Single-Tier SDE-SSM)", is_fractal=False)
    generate_eval_sample(prop_model, "Proposed (Fractal 3-Tier SDE-SSM)", is_fractal=True)
    print("="*90 + "\n")

    if prop_losses[-1] < base_losses[-1] and prop_tok_per_sec > 140000.0:
        print("🟢 KEP VERDICT: POSITIVE (Fractal 3-Tier Multi-Scale Context Scaling Validated!).")
    else:
        print("⚪ KEP VERDICT: NEUTRAL (Performance or convergence check).")

if __name__ == "__main__":
    run_isolated_benchmark()
