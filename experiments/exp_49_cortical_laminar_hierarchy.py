# experiments/exp_49_cortical_laminar_hierarchy.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-49 (3-TIER CORTICAL LAMINAR HIERARCHY)
Evaluating Layer IV (Fast Phonemic SSD) -> Layers II/III (Slow Context SSD) ->
Layers V/VI (Infragranular SwiGLU & Hopfield Attractor) vs Canonical Baseline.
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

# Dynamo Hotfix for Python 3.12 / GPU environments
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

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import karyon_config, karyon_core, karyon_agent, karyon_logger
from karyon_config import CoREConfig
from karyon_agent import CoREAgent, OffsetPositionalByteEmbedding
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory, DesaturatedHopfieldAttractorHead, ParallelSwiGLUBlock, LatentPredictor

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')


# =============================================================================
# 1. SPECIALIZED LAMINAR SSD MODULE (CHUNK Q=64 PARALLEL SCAN)
# =============================================================================
class SpecializedLaminarSSDLayer(nn.Module):
    """
    Cortical Laminar SSD Layer with specialized decay spectrum and GABAergic lateral gating.
    Enforces LayerNorm output to prevent FP16 overflows (>65504).
    """
    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, num_heads: int = 8,
                 head_k: int = 32, head_v: int = 64, min_beta: float = 0.15, max_beta: float = 0.70):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_k = head_k
        self.head_v = head_v
        self.inv_sqrt_k = 1.0 / math.sqrt(head_k)

        self.q_proj = nn.Linear(in_dim, num_heads * head_k)
        self.k_proj = nn.Linear(in_dim, num_heads * head_k)
        self.v_proj = nn.Linear(in_dim, num_heads * head_v)
        self.delta_proj = nn.Linear(in_dim, num_heads)

        # Laminar decay spectrum initialization
        betas = torch.exp(torch.linspace(math.log(max_beta), math.log(min_beta), num_heads))
        alphas = 1.0 - betas
        logit_init = torch.log(alphas / (1.0 - alphas)).view(1, num_heads, 1, 1)
        self.decay_logits = nn.Parameter(logit_init)

        self.out_proj = nn.Linear(num_heads * head_v, in_dim)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x_seq: torch.Tensor, m_prev: torch.Tensor, u_t: torch.Tensor, dt: float = 1.0):
        batch_size, seq_len, _ = x_seq.size()
        
        curiosity = u_t.select(1, 0).view(batch_size, 1, 1, 1)
        na = u_t.select(1, 4).view(batch_size, 1, 1, 1)
        da = u_t.select(1, 5).view(batch_size, 1, 1, 1)
        eff_dt = torch.clamp(dt * (1.0 - 0.4 * na + 0.4 * da), 0.30, 2.00)

        q = (self.q_proj(x_seq).view(batch_size, seq_len, self.num_heads, self.head_k).transpose(1, 2)) * self.inv_sqrt_k
        k = self.k_proj(x_seq).view(batch_size, seq_len, self.num_heads, self.head_k).transpose(1, 2)
        v = self.v_proj(x_seq).view(batch_size, seq_len, self.num_heads, self.head_v).transpose(1, 2)

        selective_delta = F.softplus(self.delta_proj(x_seq)).view(batch_size, seq_len, self.num_heads, 1).transpose(1, 2)
        base_alpha = torch.sigmoid(self.decay_logits)
        alpha = torch.pow(base_alpha, (selective_delta * eff_dt).clamp(0.1, 10.0))
        beta = 1.0 - alpha

        pos = torch.arange(seq_len, device=x_seq.device, dtype=torch.float32)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        causal_mask = (diff >= 0).float().view(1, 1, seq_len, seq_len)

        mean_alpha = alpha.mean(dim=2, keepdim=True)
        decay_weights = torch.pow(mean_alpha, diff.clamp_min(0).view(1, 1, seq_len, seq_len)) * causal_mask * beta.mean(dim=2, keepdim=True)

        s_matrix = torch.matmul(q, k.transpose(-1, -2)) * decay_weights
        y_intra = torch.matmul(s_matrix, v)

        decay_to_start = torch.pow(mean_alpha, (pos + 1.0).view(1, 1, seq_len, 1))
        y_inter = torch.matmul(q * decay_to_start, m_prev)

        y_total = (y_intra + y_inter).transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_v)
        h_out = self.norm(self.out_proj(y_total) + x_seq)

        decay_to_end = torch.pow(mean_alpha, (float(seq_len) - 1.0 - pos).view(1, 1, seq_len, 1))
        k_decayed = k * decay_to_end
        kv_chunk_update = torch.matmul(k_decayed.transpose(-1, -2), v)

        alpha_chunk = torch.pow(mean_alpha, float(seq_len))
        sigma_somatic = 1e-3 * (0.8 * curiosity + 0.4 * na + 0.1)
        dW = torch.randn_like(m_prev) * torch.sqrt(eff_dt) * sigma_somatic
        m_next = alpha_chunk * m_prev + beta.mean(dim=2, keepdim=True) * kv_chunk_update + dW

        return h_out, m_next


# =============================================================================
# 2. MODEL A: BASELINE CANONICAL ARCHITECTURE (1-LAYER v18.5)
# =============================================================================
class BaselineCanonicalAgent(nn.Module):
    def __init__(self, config: CoREConfig, device_str: str = 'cpu'):
        super().__init__()
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.config = config
        self.text_dim = config.net.text_dim
        self.hidden_dim = config.net.hidden_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.num_heads = config.net.num_heads
        self.head_k = config.net.head_k
        self.head_v = config.net.head_v
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(
            vocab_size=self.text_gen_dim, text_dim=self.text_dim, max_len=8192, device_str=device_str
        ).to(self.device)

        self.ssd_core = karyon_core.CalibratedParallelSSDCore(
            text_dim=self.text_dim, unified_dim=self.text_dim, hidden_dim=self.hidden_dim,
            num_heads=self.num_heads, head_k=self.head_k, head_v=self.head_v, device=device_str
        )
        self.channel_mixer = karyon_core.ParallelSwiGLUBlock(
            hidden_dim=self.hidden_dim, expand_dim=config.net.expand_dim, device=device_str
        )
        self.attractor_head = karyon_core.DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, vocab_size=self.text_gen_dim,
            num_attractors=config.net.num_attractors, device=device_str
        )
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, criterion, chunk_size: int = 64):
        batch_size, seq_len = input_seq.size()
        m_curr = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()

        num_chunks = max(1, seq_len // chunk_size)
        chunk_losses = []
        commit_losses = []

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)
            chunk_in = input_seq[:, c_start:c_end]
            chunk_tgt = target_seq[:, c_start:c_end]

            chunk_emb = self.pos_embeddings(chunk_in, start_pos=c_start, apply_rf=True)
            ssd_out = self.ssd_core.forward_chunk_parallel_ssd(chunk_emb, m_curr, curr_u_t, 1.0)
            h_chunk, m_curr = ssd_out[0], ssd_out[1]

            h_reasoned = self.channel_mixer(h_chunk)
            h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_reasoned, curr_u_t)

            h_proj = self.motor_text_proj(h_relaxed)
            logits_flat = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
            targets_flat = chunk_tgt.contiguous().view(-1)

            loss = criterion(logits_flat, targets_flat)
            chunk_losses.append(loss)
            commit_losses.append(commit_loss)

            has_eos = (chunk_in == 257).any(dim=-1).view(batch_size, 1, 1, 1).float()
            m_curr = (m_curr * (1.0 - has_eos)).detach()

        avg_loss = torch.stack(chunk_losses).mean()
        avg_commit = torch.stack(commit_losses).mean()
        total_loss = avg_loss + 0.05 * avg_commit
        return total_loss, avg_loss.item()


# =============================================================================
# 3. MODEL B: PROPOSED EXP-49 (3-TIER CORTICAL LAMINAR HIERARCHY)
# =============================================================================
class LaminarCorticalAgent(nn.Module):
    """
    3-Tier Biophysical Cortical Laminar Architecture (Bastos-Friston Canonical Microcircuit):
    - Layer IV: Granular Fast Phonemic SSD Core (alpha in [0.40, 0.85], fast half-life 1-5 bytes)
    - Layers II/III: Supragranular Slow Context SSD Core (alpha in [0.92, 0.9995], slow half-life 30-1500 bytes)
    - Layers V/VI: Infragranular Laminar SwiGLU + Hopfield Attractor Readout Head
    """
    def __init__(self, config: CoREConfig, device_str: str = 'cpu'):
        super().__init__()
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.config = config
        self.text_dim = config.net.text_dim
        self.hidden_dim = config.net.hidden_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.num_heads = config.net.num_heads
        self.head_k = config.net.head_k
        self.head_v = config.net.head_v
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        # 0. Afferent Positional Embeddings
        self.pos_embeddings = OffsetPositionalByteEmbedding(
            vocab_size=self.text_gen_dim, text_dim=self.text_dim, max_len=8192, device_str=device_str
        ).to(self.device)

        # Tier 1: Layer IV Granular Fast Phonemic SSD (Fast decay: half-life 1-5 bytes)
        self.layer_iv_granular = SpecializedLaminarSSDLayer(
            in_dim=self.text_dim, hidden_dim=self.hidden_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.25, max_beta=0.75
        ).to(self.device)

        # Inter-laminar ascending pathway IV -> II/III
        self.inter_laminar_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)

        # Tier 2: Layers II/III Supragranular Slow Context SSD (Slow decay: half-life 30-1500 bytes)
        self.layers_ii_iii_supragranular = SpecializedLaminarSSDLayer(
            in_dim=self.text_dim, hidden_dim=self.hidden_dim, num_heads=self.num_heads,
            head_k=self.head_k, head_v=self.head_v, min_beta=0.0005, max_beta=0.08
        ).to(self.device)

        # Tier 3: Layers V/VI Infragranular Fusion & Channel Mixing
        self.laminar_fusion_proj = nn.Sequential(
            nn.Linear(self.text_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim)
        ).to(self.device)

        self.channel_mixer = karyon_core.ParallelSwiGLUBlock(
            hidden_dim=self.hidden_dim, expand_dim=config.net.expand_dim, device=device_str
        )

        # Modern Continuous Hopfield Attractor Basin Landscape
        self.attractor_head = karyon_core.DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, vocab_size=self.text_gen_dim,
            num_attractors=config.net.num_attractors, device=device_str
        )

        # Afferent-Efferent Tied Motor Readout
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, criterion, chunk_size: int = 64):
        batch_size, seq_len = input_seq.size()
        
        # Dual-timescale state spaces: m_fast for Layer IV, m_slow for Layers II/III
        m_fast = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        m_slow = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()

        num_chunks = max(1, seq_len // chunk_size)
        chunk_losses = []
        commit_losses = []

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)
            chunk_in = input_seq[:, c_start:c_end]
            chunk_tgt = target_seq[:, c_start:c_end]

            chunk_emb = self.pos_embeddings(chunk_in, start_pos=c_start, apply_rf=True)

            # 1. Layer IV Granular Processing (Fast Phonemic State-Space Scan)
            h_layer_iv, m_fast = self.layer_iv_granular(chunk_emb, m_fast, curr_u_t, dt=1.0)

            # 2. Ascending Inter-Laminar Pathway
            h_inter = self.inter_laminar_proj(h_layer_iv)

            # 3. Layers II/III Supragranular Processing (Slow Context State-Space Scan)
            h_layer_ii_iii, m_slow = self.layers_ii_iii_supragranular(h_inter, m_slow, curr_u_t, dt=1.0)

            # 4. Layers V/VI Infragranular Laminar Integration & SwiGLU Synthesis
            h_fused = self.laminar_fusion_proj(torch.cat([h_layer_iv, h_layer_ii_iii], dim=-1))
            h_fused_flat = h_fused.contiguous().view(-1, self.hidden_dim)
            h_reasoned = self.channel_mixer(h_fused_flat)

            # 5. Attractor Relaxation & Lexical Readout
            h_relaxed, commit_loss = self.attractor_head.relax_to_minima(h_reasoned, curr_u_t)
            h_proj = self.motor_text_proj(h_relaxed)
            logits_flat = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
            targets_flat = chunk_tgt.contiguous().view(-1)

            loss = criterion(logits_flat, targets_flat)
            chunk_losses.append(loss)
            commit_losses.append(commit_loss)

            has_eos = (chunk_in == 257).any(dim=-1).view(batch_size, 1, 1, 1).float()
            m_fast = (m_fast * (1.0 - has_eos)).detach()
            m_slow = (m_slow * (1.0 - has_eos)).detach()

        avg_loss = torch.stack(chunk_losses).mean()
        avg_commit = torch.stack(commit_losses).mean()
        total_loss = avg_loss + 0.05 * avg_commit
        return total_loss, avg_loss.item()


# =============================================================================
# 4. RUN PARITY BENCHMARK & RECORD TELEMETRY
# =============================================================================
def run_exp_49_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-49 (CORTICAL LAMINAR HIERARCHY)] ===")
    print("="*85)
    print(f"Hardware Device : {device_str.upper()} | AMP FP16 Enabled: {use_amp}")

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

    b_size, seq_len = 32, 512
    num_eval_steps = 100
    chunk_size = 64

    # Prepare standard reproducible synthetic continuous batch stream
    torch.manual_seed(42)
    sample_tokens = torch.randint(32, 126, (num_eval_steps, b_size, seq_len + 1), dtype=torch.long, device=device)

    # -------------------------------------------------------------------------
    # EVALUATING MODEL A: CANONICAL 1-LAYER BASELINE
    # -------------------------------------------------------------------------
    print("\n[1/2] Benchmarking Model A: Canonical Baseline (1-Layer SSD + SwiGLU)...")
    baseline_agent = BaselineCanonicalAgent(config, device_str=device_str).to(device)
    baseline_opt = torch.optim.AdamW(baseline_agent.parameters(), lr=3e-3, weight_decay=0.01)
    baseline_hu = HomeostaticUnit(batch_size=b_size, device=device_str)
    criterion = nn.CrossEntropyLoss(ignore_index=256)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    t_start_base = time.perf_counter()
    base_losses = []

    for step in range(num_eval_steps):
        batch = sample_tokens[step]
        input_s = batch[:, :-1]
        target_s = batch[:, 1:]

        baseline_opt.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, speech_loss = baseline_agent.forward_sequence(input_s, target_s, baseline_hu, criterion, chunk_size=chunk_size)

        scaler.scale(tot_loss).backward()
        scaler.unscale_(baseline_opt)
        torch.nn.utils.clip_grad_norm_(baseline_agent.parameters(), max_norm=3.0)
        scaler.step(baseline_opt)
        scaler.update()
        base_losses.append(speech_loss)

    if device.type == 'cuda': torch.cuda.synchronize()
    time_base_sec = time.perf_counter() - t_start_base
    vram_base_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0
    throughput_base = (num_eval_steps * b_size * seq_len) / time_base_sec
    final_loss_base = sum(base_losses[-20:]) / 20.0

    # -------------------------------------------------------------------------
    # EVALUATING MODEL B: EXP-49 (3-TIER LAMINAR HIERARCHY)
    # -------------------------------------------------------------------------
    print("\n[2/2] Benchmarking Model B: EXP-49 (3-Tier Cortical Laminar Hierarchy)...")
    laminar_agent = LaminarCorticalAgent(config, device_str=device_str).to(device)
    laminar_opt = torch.optim.AdamW(laminar_agent.parameters(), lr=3e-3, weight_decay=0.01)
    laminar_hu = HomeostaticUnit(batch_size=b_size, device=device_str)

    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    t_start_lam = time.perf_counter()
    lam_losses = []

    for step in range(num_eval_steps):
        batch = sample_tokens[step]
        input_s = batch[:, :-1]
        target_s = batch[:, 1:]

        laminar_opt.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, speech_loss = laminar_agent.forward_sequence(input_s, target_s, laminar_hu, criterion, chunk_size=chunk_size)

        scaler.scale(tot_loss).backward()
        scaler.unscale_(laminar_opt)
        torch.nn.utils.clip_grad_norm_(laminar_agent.parameters(), max_norm=3.0)
        scaler.step(laminar_opt)
        scaler.update()
        lam_losses.append(speech_loss)

    if device.type == 'cuda': torch.cuda.synchronize()
    time_lam_sec = time.perf_counter() - t_start_lam
    vram_lam_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0
    throughput_lam = (num_eval_steps * b_size * seq_len) / time_lam_sec
    final_loss_lam = sum(lam_losses[-20:]) / 20.0

    # -------------------------------------------------------------------------
    # 5. KEP RULE #2 DECISION & TELEMETRY DASHBOARD
    # -------------------------------------------------------------------------
    loss_delta = final_loss_lam - final_loss_base
    ppl_base = math.exp(min(final_loss_base, 20.0))
    ppl_lam = math.exp(min(final_loss_lam, 20.0))
    speed_retention = (throughput_lam / throughput_base) * 100.0

    print("\n" + "="*85)
    print(" === [KEP RULE #2 DECISION & EMPIRICAL TELEMETRY DASHBOARD] ===")
    print("="*85)
    print(f"{'Performance Metric':<32} | {'Model A (Baseline)':<22} | {'Model B (EXP-49)':<22} | {'Delta':<12}")
    print("-" * 85)
    print(f"{'Final Steady-State Loss (nats)':<32} | {final_loss_base:<22.4f} | {final_loss_lam:<22.4f} | {loss_delta:+12.4f}")
    print(f"{'Perplexity (PPL)':<32} | {ppl_base:<22.2f} | {ppl_lam:<22.2f} | {ppl_lam - ppl_base:+12.2f}")
    print(f"{'Throughput Speed (tok/s)':<32} | {throughput_base:<22.1f} | {throughput_lam:<22.1f} | {speed_retention:11.1f}%")
    print(f"{'Peak VRAM Memory (MB)':<32} | {vram_base_mb:<22.1f} | {vram_lam_mb:<22.1f} | {vram_lam_mb - vram_base_mb:+12.1f} MB")
    print("="*85)

    if loss_delta <= -0.08 and speed_retention >= 75.0:
        verdict = "🟢 POSITIVE (Ready for Production Merge)"
    elif abs(loss_delta) < 0.08 and speed_retention >= 75.0:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE (Needs full 52k dataset multi-pass evaluation)"
    else:
        verdict = "🔴 REJECTED (Metric regression or speed degradation)"

    print(f"DECISION VERDICT: {verdict}")
    print("="*85 + "\n")


if __name__ == "__main__":
    run_exp_49_benchmark()
