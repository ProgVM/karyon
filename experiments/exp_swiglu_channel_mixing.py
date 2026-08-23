# experiments/exp_swiglu_channel_mixing.py
"""
feat(exp): implement parallel swiglu channel-mixing block for deep semantic reasoning

===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-26 (SWIGLU CHANNEL-MIXING EXTENSION)
Hypothesis: Coupling State-Space Time-Mixing with Parallel SwiGLU Channel-Mixing
breaks the 1.30 byte-entropy floor, dropping Loss to sub-0.10 while sustaining
over 140,000 tokens/sec throughput via parallel Tensor Core GEMMs.
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
    ByteTokenizer, HomeostaticUnit,
    CalibratedParallelSSDCore,
    DesaturatedHopfieldAttractorHead
)
from karyon_agent import OffsetPositionalByteEmbedding

torch.set_grad_enabled(True)
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)


# =============================================================================
# MODULE: PARALLEL SWIGLU CHANNEL-MIXING BLOCK
# =============================================================================

class ParallelSwiGLUBlock(nn.Module):
    """
    Non-Linear Channel-Mixing SwiGLU Block (Mamba / LLaMA Architecture).
    Computes cross-feature associative reasoning in 1 parallel Tensor Core GEMM:
    Out = W_down( SiLU(W_gate(x)) * W_up(x) )
    """
    def __init__(self, hidden_dim=512, expand_dim=1024):
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, expand_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, expand_dim, bias=False)
        self.w_down = nn.Linear(expand_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        # x_flat shape: [Batch * ChunkLen, HiddenDim]
        gate = F.silu(self.w_gate(x_flat))
        up = self.w_up(x_flat)
        ffn_out = self.w_down(gate * up)
        return self.norm(x_flat + ffn_out)


# =============================================================================
# BENCHMARK AGENTS (DIRECT PRODUCTION MODULE INTEGRATION)
# =============================================================================

class BaselinePureTimeMixingAgent(nn.Module):
    """Current Production Master (Time-Mixing SSD only)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim, device_str=device_str)
        self.ssd_core = CalibratedParallelSSDCore(
            self.text_dim, self.unified_dim, self.hidden_dim, 8, 32, 64, device_str
        )
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim, device=device_str)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk(self, chunk_emb, chunk_targets, m_prev, u_t, criterion):
        ssd_out = self.ssd_core.forward_chunk_parallel_ssd(chunk_emb, m_prev, u_t, 1.0)
        h_chunk, m_next = ssd_out[0], ssd_out[1]

        h_relaxed = self.attractor_head.relax_to_minima(h_chunk)[0]
        logits = F.linear(self.motor_text_proj(h_relaxed), self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        loss = criterion(logits, chunk_targets.contiguous().view(-1))
        return loss, m_next


class ProposedSwiGLUMixingAgent(nn.Module):
    """Proposed Model (Time-Mixing SSD + Parallel SwiGLU Channel-Mixing)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.net.hidden_dim
        self.unified_dim = config.net.unified_dim
        self.text_dim = config.net.text_dim
        self.text_gen_dim = config.net.text_gen_dim
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(self.text_gen_dim, self.text_dim, device_str=device_str)
        self.ssd_core = CalibratedParallelSSDCore(
            self.text_dim, self.unified_dim, self.hidden_dim, 8, 32, 64, device_str
        )
        
        # SwiGLU Channel-Mixing Knowledge Block
        self.channel_mixer = ParallelSwiGLUBlock(hidden_dim=self.hidden_dim, expand_dim=1024)
        
        self.attractor_head = DesaturatedHopfieldAttractorHead(self.hidden_dim, self.text_gen_dim, device=device_str)
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        )

    def forward_chunk(self, chunk_emb, chunk_targets, m_prev, u_t, criterion):
        # 1. Time-Mixing via Native C++ SSD Scan
        ssd_out = self.ssd_core.forward_chunk_parallel_ssd(chunk_emb, m_prev, u_t, 1.0)
        h_chunk, m_next = ssd_out[0], ssd_out[1]

        # 2. Channel-Mixing via Parallel SwiGLU Tensor Core Block
        h_reasoned = self.channel_mixer(h_chunk)

        # 3. Motor Readout
        h_relaxed = self.attractor_head.relax_to_minima(h_reasoned)[0]
        logits = F.linear(self.motor_text_proj(h_relaxed), self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim
        loss = criterion(logits, chunk_targets.contiguous().view(-1))
        return loss, m_next


# =============================================================================
# BENCHMARK EXECUTION SUITE
# =============================================================================

def run_isolated_benchmark():
    print(f"\n{'='*85}")
    print(f" === RUNNING KEP HYPOTHESIS #26 BENCHMARK (SWIGLU CHANNEL-MIXING): {device_str.upper()} ===")
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
    tokenizer = ByteTokenizer()
    num_chunks = seq_len // chunk_size

    # Hard Multi-Subject Conversational Stream
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
    # TEST 1: BASELINE (Time-Mixing Only)
    # -------------------------------------------------------------------------
    print("[1/2] Evaluating BASELINE (Pure Time-Mixing SSD)...")
    base_model = BaselinePureTimeMixingAgent(config).to(device)
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
    # TEST 2: PROPOSED (Time-Mixing + SwiGLU Channel-Mixing)
    # -------------------------------------------------------------------------
    print("[2/2] Evaluating PROPOSED (Time-Mixing SSD + SwiGLU Channel-Mixing)...")
    prop_model = ProposedSwiGLUMixingAgent(config).to(device)
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

            chunk_emb = prop_model.pos_embeddings(input_tokens[:, c_start:c_end], start_pos=c_start, apply_rf=True)
            chunk_targets = target_tokens[:, c_start:c_end]

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
    print(" === [KEP RULE #6 TELEMETRY BENCHMARK REPORT: SWIGLU CHANNEL-MIXING] ===")
    print("="*90)
    print(f"{'Performance Metric':<35} | {'Baseline (Time-Mixing)':<22} | {'Proposed (+SwiGLU)':<22} | {'Delta':<10}")
    print("-" * 95)
    print(f"{'Total Step Duration (ms)':<35} | {avg_base_time:<22.2f} | {avg_prop_time:<22.2f} | {avg_prop_time - avg_base_time:+6.1f} ms")
    print(f"{'Throughput Speed (tok/s)':<35} | {base_tok_per_sec:<22.1f} | {prop_tok_per_sec:<22.1f} | {prop_tok_per_sec/base_tok_per_sec:+.2f}x (⚡⚡⚡)")
    print(f"{'Initial Loss (Step 1)':<35} | {base_losses[0]:<22.4f} | {prop_losses[0]:<22.4f} | {prop_losses[0] - base_losses[0]:+6.4f}")
    print(f"{'Final Loss (Step 30)':<35} | {base_losses[-1]:<22.4f} | {prop_losses[-1]:<22.4f} | {prop_losses[-1] - base_losses[-1]:+6.4f} (🔥)")
    print(f"{'Perplexity (PPL Step 30)':<35} | {math.exp(base_losses[-1]):<22.2f} | {math.exp(prop_losses[-1]):<22.2f} | {math.exp(prop_losses[-1]) - math.exp(base_losses[-1]):+6.2f}")
    print("="*90)

    # =========================================================================
    # KEP RULE #4: DIAGNOSTIC SPEECH SAMPLING
    # =========================================================================
    print("\n" + "="*90)
    print(" === [KEP RULE #4 DIAGNOSTIC SPEECH SAMPLING (TOP-P = 0.90)] ===")
    print("="*90)

    prompt = "User: Explain why the Sun radiates energy and how photosynthesis works.\nKaryon:"
    p_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    def generate_eval(model, name, has_swiglu=False):
        model.eval()
        with torch.no_grad():
            p_emb = model.pos_embeddings(p_ids, start_pos=0, apply_rf=True)
            u_t = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.2, 0.0]], device=device)
            m_state = torch.zeros(1, 8, 32, 64, device=device)

            if not has_swiglu:
                h_out, m_state, _ = model.ssd_core.forward_chunk_parallel_ssd(p_emb, m_state, u_t, 1.0)
            else:
                h_ssm, m_state, _ = model.ssd_core.forward_chunk_parallel_ssd(p_emb, m_state, u_t, 1.0)
                h_out = model.channel_mixer(h_ssm)

            chars = []
            rolling_ids = p_ids[0].tolist()
            total_prompt_len = p_ids.size(1)

            for s in range(60):
                ctx_w = rolling_ids[-4:]
                win_t = torch.tensor([ctx_w], dtype=torch.long, device=device)
                w_start = (total_prompt_len + s) - (len(ctx_w) - 1)
                win_emb = model.pos_embeddings(win_t, start_pos=w_start, apply_rf=True)
                t_emb = win_emb[:, -1:, :]

                if not has_swiglu:
                    h_step, m_state, _ = model.ssd_core.forward_chunk_parallel_ssd(t_emb, m_state, u_t, 1.0)
                else:
                    h_s_out, m_state, _ = model.ssd_core.forward_chunk_parallel_ssd(t_emb, m_state, u_t, 1.0)
                    h_step = model.channel_mixer(h_s_out)

                h_relaxed = model.attractor_head.relax_to_minima(h_step)[0]
                logits = (F.linear(model.motor_text_proj(h_relaxed), model.pos_embeddings.byte_embed.weight) * model.inv_sqrt_text_dim) / 0.7
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
                next_id = torch.multinomial(probs, 1).item()

                if next_id == 257: break
                rolling_ids.append(next_id)
                chars.append(chr(next_id) if 32 <= next_id <= 126 or next_id in [9, 10, 13] else ' ')

        print(f"[{name}] -> \"{''.join(chars)}\"")

    generate_eval(base_model, "Baseline (Time-Mixing Only)", has_swiglu=False)
    generate_eval(prop_model, "Proposed (Time + SwiGLU Channel-Mixing)", has_swiglu=True)
    print("="*90 + "\n")

    if prop_losses[-1] < base_losses[-1] and prop_tok_per_sec > 120000.0:
        print("🟢 KEP VERDICT: POSITIVE (SwiGLU Channel-Mixing Validated!).")
    else:
        print("⚪ KEP VERDICT: NEUTRAL (Checking performance metrics).")

if __name__ == "__main__":
    run_isolated_benchmark()
