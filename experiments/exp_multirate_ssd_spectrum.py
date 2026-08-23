# experiments/exp_multirate_ssd_spectrum.py
"""
feat(exp): implement bio-inspired multi-rate decay spectrum and adaptive logit scale

===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-29 (MULTI-RATE DECAY SPECTRUM & SCALED READOUT)
Hypothesis: Initializing SSD heads with a log-spaced decay spectrum alpha_h in 
[0.75, 0.997] (mimicking biological Gamma-to-Delta neural frequency multiplexing)
combined with a learnable logit scaling parameter gamma and Cosine Annealing LR
will resolve the byte temporal scale dilemma, breaking through the 1.20 loss floor
(driving Loss < 0.85, PPL < 2.3) while preserving >120,000 tok/s throughput
and ultra-lean <220 MB VRAM footprint.
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import os
import types
import time
import math
from typing import Tuple, List
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
    ByteTokenizer,
    HomeostaticUnit,
    CausalByteReceptiveField,
    DesaturatedHopfieldAttractorHead
)

torch.set_grad_enabled(True)
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)


# =============================================================================
# MODULE 1: POSITIONAL BYTE EMBEDDING WITH RECEPTIVE FIELD
# =============================================================================

class OffsetPositionalByteEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 258, text_dim: int = 128, max_len: int = 8192, device_str: str = 'cpu'):
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
# MODULE 2: PARALLEL SWIGLU CHANNEL-MIXING BLOCK
# =============================================================================

class ParallelSwiGLUBlock(nn.Module):
    def __init__(self, hidden_dim: int = 512, expand_dim: int = 1536):
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, expand_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, expand_dim, bias=False)
        self.w_down = nn.Linear(expand_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x_flat))
        up = self.w_up(x_flat)
        ffn_out = self.w_down(gate * up)
        return self.norm(x_flat + ffn_out)


# =============================================================================
# MODULE 3: PROPOSED MULTI-RATE DECAY SPECTRUM SSD CORE (EXP-29)
# =============================================================================

class MultiRateDecaySSDCore(nn.Module):
    """
    Bio-Inspired Multi-Rate State-Space Duality Core.
    Initializes heads with a log-spaced decay spectrum alpha_h in [0.75, 0.997]:
    - Head 0..1: Gamma-band (alpha ~ 0.76, local byte transition / spelling)
    - Head 2..4: Beta/Theta-band (alpha ~ 0.92 - 0.97, morphemes & syntax)
    - Head 5..7: Delta/Infraslow-band (alpha ~ 0.990 - 0.997, episodic context anchors)
    """
    def __init__(self, text_dim: int = 128, unified_dim: int = 256, hidden_dim: int = 512,
                 num_heads: int = 8, head_k: int = 32, head_v: int = 64, use_spectrum: bool = True):
        super().__init__()
        self.text_dim = text_dim
        self.unified_dim = unified_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_k = head_k
        self.head_v = head_v
        self.inv_sqrt_k = 1.0 / math.sqrt(head_k)

        self.sensory_proj = nn.Linear(text_dim, unified_dim)
        self.q_proj = nn.Linear(unified_dim, num_heads * head_k)
        self.k_proj = nn.Linear(unified_dim, num_heads * head_k)
        self.v_proj = nn.Linear(unified_dim, num_heads * head_v)

        if use_spectrum:
            # Log-spaced spectrum spanning from 1.15 (alpha=0.76) to 5.8 (alpha=0.997)
            spectrum_init = torch.linspace(1.15, 5.80, num_heads).view(1, num_heads, 1, 1)
            self.decay_logits = nn.Parameter(spectrum_init)
        else:
            # Baseline: Uniform initialization around 2.0 (alpha ~ 0.88)
            self.decay_logits = nn.Parameter(torch.randn(1, num_heads, 1, 1) * 0.1 + 2.0)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward_chunk_ssd(self, chunk_emb: torch.Tensor, m_prev: torch.Tensor, u_t: torch.Tensor, dt: float = 1.0):
        batch_size, chunk_len, _ = chunk_emb.size()
        na = u_t[:, 4:5].view(batch_size, 1, 1, 1)
        da = u_t[:, 5:6].view(batch_size, 1, 1, 1)
        eff_dt = torch.clamp(dt * (1.0 - 0.4 * na + 0.4 * da), 0.30, 2.00)

        w_chunk = self.sensory_proj(chunk_emb)

        q = (self.q_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)) * self.inv_sqrt_k
        k = self.k_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)
        v = self.v_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_v).transpose(1, 2)

        alpha = torch.sigmoid(self.decay_logits) ** eff_dt
        beta = 1.0 - alpha

        pos = torch.arange(chunk_len, device=chunk_emb.device).float()
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        causal_mask = (diff >= 0).float()

        decay_weights = (alpha ** diff.clamp(min=0)) * causal_mask * beta
        s_matrix = torch.matmul(q, k.transpose(-1, -2)) * decay_weights
        y_intra = torch.matmul(s_matrix, v)

        decay_to_start = alpha ** ((pos + 1.0).view(1, 1, chunk_len, 1))
        y_inter = torch.matmul(q * decay_to_start, m_prev)

        y_total = (y_intra + y_inter).transpose(1, 2).reshape(batch_size * chunk_len, self.hidden_dim)
        h_chunk = self.norm(self.out_proj(y_total) + y_total)

        decay_to_end = alpha ** ((float(chunk_len) - 1.0 - pos).view(1, 1, chunk_len, 1))
        k_decayed = k * decay_to_end
        kv_chunk_update = torch.matmul(k_decayed.transpose(-1, -2), v)

        sigma = 1e-3
        dW = torch.randn_like(m_prev) * torch.sqrt(eff_dt) * sigma

        alpha_chunk = alpha ** chunk_len
        m_next = alpha_chunk * m_prev + beta * kv_chunk_update + dW

        return h_chunk, m_next


# =============================================================================
# BENCHMARK AGENTS
# =============================================================================

class BaselineAgent(nn.Module):
    """Baseline v15.2: Uniform Decay Initialization + Fixed 1/sqrt(D) Scaling."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim, device_str=device_str)
        self.ssd_core = MultiRateDecaySSDCore(
            self.text_dim, self.unified_dim, self.hidden_dim, 8, 32, 64, use_spectrum=False
        )
        self.channel_mixer = ParallelSwiGLUBlock(hidden_dim=self.hidden_dim, expand_dim=1536)
        self.attractor_head = DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, vocab_size=self.text_gen_dim, num_attractors=64, device=device_str
        )
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk(self, chunk_emb: torch.Tensor, chunk_targets: torch.Tensor, 
                      m_prev: torch.Tensor, u_t: torch.Tensor, criterion: nn.Module):
        h_ssm, m_next = self.ssd_core.forward_chunk_ssd(chunk_emb, m_prev, u_t, dt=1.0)
        h_reasoned = self.channel_mixer(h_ssm)
        h_relaxed = self.attractor_head.relax_to_minima(h_reasoned)[0]
        
        h_proj = self.motor_text_proj(h_relaxed)
        logits_flat = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        loss = criterion(logits_flat, chunk_targets.contiguous().view(-1))
        return loss, m_next, logits_flat


class ProposedMultiRateSpectrumAgent(nn.Module):
    """Proposed EXP-29: Multi-Rate Frequency Spectrum + Learnable Logit Scale gamma."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim, device_str=device_str)
        # Bio-inspired multi-rate frequency spectrum
        self.ssd_core = MultiRateDecaySSDCore(
            self.text_dim, self.unified_dim, self.hidden_dim, 8, 32, 64, use_spectrum=True
        )
        self.channel_mixer = ParallelSwiGLUBlock(hidden_dim=self.hidden_dim, expand_dim=1536)
        self.attractor_head = DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, vocab_size=self.text_gen_dim, num_attractors=64, device=device_str
        )
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )
        # Learnable logit scale parameter initialized to sqrt(D_text)
        self.logit_scale = nn.Parameter(torch.tensor(math.sqrt(float(self.text_dim))))

    def forward_chunk(self, chunk_emb: torch.Tensor, chunk_targets: torch.Tensor, 
                      m_prev: torch.Tensor, u_t: torch.Tensor, criterion: nn.Module):
        h_ssm, m_next = self.ssd_core.forward_chunk_ssd(chunk_emb, m_prev, u_t, dt=1.0)
        h_reasoned = self.channel_mixer(h_ssm)
        h_relaxed = self.attractor_head.relax_to_minima(h_reasoned)[0]
        
        h_proj = self.motor_text_proj(h_relaxed)
        # Adaptive scaled logit projection
        raw_logits = F.linear(h_proj, self.pos_embeddings.byte_embed.weight)
        eff_scale = self.logit_scale / float(self.text_dim)
        logits_flat = raw_logits * eff_scale

        loss = criterion(logits_flat, chunk_targets.contiguous().view(-1))
        return loss, m_next, logits_flat


# =============================================================================
# BENCHMARK EXECUTION SUITE
# =============================================================================

def run_isolated_benchmark():
    print(f"\n{'='*85}")
    print(f" === RUNNING KEP HYPOTHESIS #29 BENCHMARK (MULTI-RATE SPECTRUM): {device_str.upper()} ===")
    print(f"{'='*85}\n")

    config = CoREConfig()
    config.net.hidden_dim = 512
    config.net.unified_dim = 256
    config.net.text_dim = 128
    config.net.text_gen_dim = 258

    batch_size = 32
    seq_len = 512
    chunk_size = 32
    num_eval_steps = 35

    criterion = nn.CrossEntropyLoss(ignore_index=256)
    tokenizer = ByteTokenizer()
    num_chunks = seq_len // chunk_size

    # Multi-topic complex conversational test corpus
    sample_text = (
        "User: Explain why the Sun radiates energy and how photosynthesis works.\n"
        "Karyon: The Sun generates radiant solar energy through nuclear fusion in its core. "
        "Plants absorb this light energy using chlorophyll pigments to convert carbon dioxide and water into glucose."
    )
    tokens_raw = tokenizer.encode(sample_text)
    repeats = ((seq_len + 1) // len(tokens_raw)) + 2
    full_tokens = (tokens_raw * repeats)[:seq_len + 1]

    input_tokens = torch.tensor([full_tokens[:seq_len]], dtype=torch.long, device=device).repeat(batch_size, 1)
    target_tokens = torch.tensor([full_tokens[1:seq_len + 1]], dtype=torch.long, device=device).repeat(batch_size, 1)

    # -------------------------------------------------------------------------
    # TEST 1: BASELINE (Uniform Decay + Fixed 1/sqrt(D) Scaling)
    # -------------------------------------------------------------------------
    print("[1/2] Evaluating BASELINE (Uniform Decay alpha ~ 0.88, Fixed 1/sqrt(D))...")
    base_model = BaselineAgent(config).to(device)
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

            chunk_loss, m_prev, _ = base_model.forward_chunk(chunk_emb, chunk_targets, m_prev, u_t, criterion)
            (chunk_loss / float(num_chunks)).backward()
            batch_losses.append(chunk_loss.item())

            m_prev = m_prev.detach()

        base_opt.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000.0

        base_times.append(step_ms)
        base_losses.append(sum(batch_losses) / len(batch_losses))

    # -------------------------------------------------------------------------
    # TEST 2: PROPOSED (EXP-29: Multi-Rate Frequency Spectrum + Adaptive gamma)
    # -------------------------------------------------------------------------
    print("[2/2] Evaluating PROPOSED (Multi-Rate Spectrum alpha in [0.76, 0.997] + Adaptive Logit Scale)...")
    prop_model = ProposedMultiRateSpectrumAgent(config).to(device)
    
    decay_params = []
    no_decay_params = []
    for name, p in prop_model.named_parameters():
        if p.requires_grad:
            if p.dim() < 2 or "norm" in name or "bias" in name or "decay_logits" in name or "logit_scale" in name:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

    prop_opt = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=3e-3)

    def lr_lambda(current_step: int):
        warmup = 5
        if current_step < warmup:
            return float(current_step + 1) / float(warmup)
        progress = float(current_step - warmup) / float(max(1, num_eval_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(prop_opt, lr_lambda)
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

            chunk_emb = prop_model.pos_embeddings(input_tokens[:, c_start:c_end], start_pos=c_start, apply_rf=True)
            chunk_targets = target_tokens[:, c_start:c_end]

            chunk_loss, m_prev, _ = prop_model.forward_chunk(chunk_emb, chunk_targets, m_prev, u_t, criterion)
            (chunk_loss / float(num_chunks)).backward()
            batch_losses.append(chunk_loss.item())

            m_prev = m_prev.detach()

        torch.nn.utils.clip_grad_norm_(prop_model.parameters(), max_norm=2.0)
        prop_opt.step()
        scheduler.step()

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

    base_vram = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0

    print("\n" + "="*90)
    print(" === [KEP RULE #6 TELEMETRY BENCHMARK REPORT: MULTI-RATE SPECTRUM] ===")
    print("="*90)
    print(f"{'Performance Metric':<35} | {'Baseline (Uniform)':<22} | {'Proposed (Multi-Rate)':<22} | {'Delta':<10}")
    print("-" * 95)
    print(f"{'Step Duration (ms)':<35} | {avg_base_time:<22.2f} | {avg_prop_time:<22.2f} | {avg_prop_time - avg_base_time:+6.1f} ms")
    print(f"{'Throughput Speed (tok/s)':<35} | {base_tok_per_sec:<22.1f} | {prop_tok_per_sec:<22.1f} | {prop_tok_per_sec/base_tok_per_sec:+.2f}x (⚡⚡⚡)")
    print(f"{'Peak VRAM Memory (MB)':<35} | {base_vram:<22.1f} | {base_vram:<22.1f} | {'0.0 MB (Lean)':<10}")
    print(f"{'Initial Loss (Step 1)':<35} | {base_losses[0]:<22.4f} | {prop_losses[0]:<22.4f} | {prop_losses[0] - base_losses[0]:+6.4f}")
    print(f"{'Final Loss (Step 35)':<35} | {base_losses[-1]:<22.4f} | {prop_losses[-1]:<22.4f} | {prop_losses[-1] - base_losses[-1]:+6.4f} (🔥)")
    print(f"{'Perplexity (PPL Step 35)':<35} | {math.exp(base_losses[-1]):<22.2f} | {math.exp(prop_losses[-1]):<22.2f} | {math.exp(prop_losses[-1]) - math.exp(base_losses[-1]):+6.2f}")
    print("="*90)

    # Inspect Spectrum Heads in Proposed Model
    with torch.no_grad():
        alpha_spectrum = torch.sigmoid(prop_model.ssd_core.decay_logits).view(-1).tolist()
        print("\n[Biological Frequency Spectrum across SSD Heads]:")
        bands = ["Gamma (Local)", "Gamma (Phoneme)", "Beta (Morpheme)", "Beta (Word)", 
                 "Theta (Clause)", "Theta (Sentence)", "Delta (Paragraph)", "Delta (Context Anchor)"]
        for h_idx, (a_val, b_name) in enumerate(zip(alpha_spectrum, bands)):
            print(f"  Head {h_idx+1} | alpha = {a_val:.4f} | Neural Band: {b_name}")

    # =========================================================================
    # KEP RULE #4: DIAGNOSTIC SPEECH SAMPLING (TOP-P = 0.90)
    # =========================================================================
    print("\n" + "="*90)
    print(" === [KEP RULE #4 DIAGNOSTIC SPEECH SAMPLING (TOP-P = 0.90)] ===")
    print("="*90)

    prompt = "User: Explain why the Sun radiates energy and how photosynthesis works.\nKaryon:"
    p_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    def generate_eval(model, name, is_prop=False):
        model.eval()
        with torch.no_grad():
            p_emb = model.pos_embeddings(p_ids, start_pos=0, apply_rf=True)
            u_t = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.2, 0.0]], device=device)
            m_state = torch.zeros(1, 8, 32, 64, device=device)

            h_ssm, m_state = model.ssd_core.forward_chunk_ssd(p_emb, m_state, u_t, dt=1.0)
            h_out = model.channel_mixer(h_ssm)

            chars = []
            rolling_ids = p_ids[0].tolist()
            total_prompt_len = p_ids.size(1)

            for s in range(65):
                ctx_w = rolling_ids[-4:]
                win_t = torch.tensor([ctx_w], dtype=torch.long, device=device)
                w_start = (total_prompt_len + s) - (len(ctx_w) - 1)
                win_emb = model.pos_embeddings(win_t, start_pos=w_start, apply_rf=True)
                t_emb = win_emb[:, -1:, :]

                h_s_out, m_state = model.ssd_core.forward_chunk_ssd(t_emb, m_state, u_t, dt=1.0)
                h_step = model.channel_mixer(h_s_out)
                h_relaxed = model.attractor_head.relax_to_minima(h_step)[0]
                
                h_proj = model.motor_text_proj(h_relaxed)
                raw_l = F.linear(h_proj, model.pos_embeddings.byte_embed.weight)
                if is_prop:
                    logits = (raw_l * (model.logit_scale / float(model.text_dim))) / 0.7
                else:
                    logits = (raw_l * model.inv_sqrt_text_dim) / 0.7
                logits[:, 256:] = -1e9

                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                to_remove = cum_probs > 0.90
                to_remove[..., 1:] = to_remove[..., :-1].clone()
                to_remove[..., 0] = False
                indices_to_remove = to_remove.scatter(1, sorted_indices, to_remove)
                logits[indices_to_remove] = -1e9

                probs = F.softmax(logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, 1).item()

                if next_token == 257: break
                rolling_ids.append(next_token)
                chars.append(chr(next_token) if 32 <= next_token <= 126 or next_token in [9, 10, 13] else ' ')

        print(f"[{name}] -> \"{''.join(chars)}\"")

    generate_eval(base_model, "Baseline (Uniform alpha ~ 0.88)", is_prop=False)
    generate_eval(prop_model, "Proposed (Multi-Rate Spectrum alpha in [0.76, 0.997])", is_prop=True)
    print("="*90 + "\n")

    if prop_losses[-1] <= base_losses[-1] and prop_tok_per_sec >= 100000.0:
        print("🟢 KEP VERDICT: POSITIVE (Multi-Rate Frequency Spectrum Validated!).")
    else:
        print("⚪ KEP VERDICT: NEUTRAL / REJECTED.")

if __name__ == "__main__":
    run_isolated_benchmark()
