# experiments/exp_50_high_rank_matrix_ssd.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-50 (HIGH-RANK ASSOCIATIVE MATRIX SSD)
Evaluating High-Rank Matrix State Space (K=64, V=128 | 65k State Capacity)
vs Canonical Baseline (K=32, V=64 | 16k State Capacity) on Real Conversational Stream.
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
from datasets import load_dataset

# Dynamo Hotfix for Python 3.12 / Kaggle GPU
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

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import karyon_config, karyon_core, karyon_agent, karyon_logger
from karyon_config import CoREConfig
from karyon_agent import OffsetPositionalByteEmbedding
from karyon_core import ByteTokenizer, HomeostaticUnit, DesaturatedHopfieldAttractorHead, ParallelSwiGLUBlock

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')


# =============================================================================
# 1. PARAMETRIZABLE HIGH-RANK PARALLEL SSD CORE
# =============================================================================
class ParametricParallelSSDCore(nn.Module):
    """
    Parametrizable Parallel State-Space Duality (SSD) Core supporting arbitrary
    head dimensions (K, V) with CUDA Tensor Core alignment and selective delta gating.
    """
    def __init__(self, text_dim: int = 256, unified_dim: int = 256, hidden_dim: int = 512,
                 num_heads: int = 8, head_k: int = 64, head_v: int = 128, device_str: str = 'cpu'):
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
        self.delta_proj = nn.Linear(unified_dim, num_heads)

        # Multi-timescale decay spectrum
        betas = torch.exp(torch.linspace(math.log(0.30), math.log(0.0005), num_heads))
        alphas = 1.0 - betas
        logit_init = torch.log(alphas / (1.0 - alphas)).view(1, num_heads, 1, 1)
        self.decay_logits = nn.Parameter(logit_init)

        self.out_proj = nn.Linear(num_heads * head_v, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward_chunk_parallel_ssd(self, chunk_emb: torch.Tensor, m_prev: torch.Tensor, u_t: torch.Tensor, dt: float = 1.0):
        batch_size, chunk_len, _ = chunk_emb.size()

        curiosity = u_t.select(1, 0).view(batch_size, 1, 1, 1)
        na = u_t.select(1, 4).view(batch_size, 1, 1, 1)
        da = u_t.select(1, 5).view(batch_size, 1, 1, 1)
        eff_dt = torch.clamp(dt * (1.0 - 0.4 * na + 0.4 * da), 0.30, 2.00)

        w_chunk = self.sensory_proj(chunk_emb)

        q = (self.q_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)) * self.inv_sqrt_k
        k = self.k_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_k).transpose(1, 2)
        v = self.v_proj(w_chunk).view(batch_size, chunk_len, self.num_heads, self.head_v).transpose(1, 2)

        selective_delta = F.softplus(self.delta_proj(w_chunk)).view(batch_size, chunk_len, self.num_heads, 1).transpose(1, 2)
        base_alpha = torch.sigmoid(self.decay_logits)
        alpha = torch.pow(base_alpha, (selective_delta * eff_dt).clamp(0.1, 10.0))
        beta = 1.0 - alpha

        pos = torch.arange(chunk_len, device=chunk_emb.device, dtype=torch.float32)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        causal_mask = (diff >= 0).float().view(1, 1, chunk_len, chunk_len)

        mean_alpha = alpha.mean(dim=2, keepdim=True)
        decay_weights = torch.pow(mean_alpha, diff.clamp_min(0).view(1, 1, chunk_len, chunk_len)) * causal_mask * beta.mean(dim=2, keepdim=True)

        s_matrix = torch.matmul(q, k.transpose(-1, -2)) * decay_weights
        y_intra = torch.matmul(s_matrix, v)

        decay_to_start = torch.pow(mean_alpha, (pos + 1.0).view(1, 1, chunk_len, 1))
        y_inter = torch.matmul(q * decay_to_start, m_prev)

        y_total = (y_intra + y_inter).transpose(1, 2).reshape(batch_size * chunk_len, self.num_heads * self.head_v)
        h_chunk = self.norm(self.out_proj(y_total))

        decay_to_end = torch.pow(mean_alpha, (float(chunk_len) - 1.0 - pos).view(1, 1, chunk_len, 1))
        k_decayed = k * decay_to_end
        kv_chunk_update = torch.matmul(k_decayed.transpose(-1, -2), v)

        alpha_chunk = torch.pow(mean_alpha, float(chunk_len))
        sigma_somatic = 1e-3 * (0.8 * curiosity + 0.4 * na + 0.1)
        dW = torch.randn_like(m_prev) * torch.sqrt(eff_dt) * sigma_somatic
        m_next = alpha_chunk * m_prev + beta.mean(dim=2, keepdim=True) * kv_chunk_update + dW

        return h_chunk, m_next


# =============================================================================
# 2. EXPERIMENTAL AGENT CONTAINER (BASELINE vs EXP-50)
# =============================================================================
class ExperimentalCoREAgent(nn.Module):
    def __init__(self, text_dim: int = 256, hidden_dim: int = 512, expand_dim: int = 2048,
                 num_heads: int = 8, head_k: int = 32, head_v: int = 64, device_str: str = 'cpu'):
        super().__init__()
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_k = head_k
        self.head_v = head_v
        self.text_gen_dim = 258
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(text_dim)

        self.pos_embeddings = OffsetPositionalByteEmbedding(
            vocab_size=self.text_gen_dim, text_dim=text_dim, max_len=8192, device_str=device_str
        ).to(self.device)

        self.ssd_core = ParametricParallelSSDCore(
            text_dim=text_dim, unified_dim=text_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, head_k=head_k, head_v=head_v, device_str=device_str
        ).to(self.device)

        self.channel_mixer = karyon_core.ParallelSwiGLUBlock(
            hidden_dim=hidden_dim, expand_dim=expand_dim, device=device_str
        )

        self.attractor_head = karyon_core.DesaturatedHopfieldAttractorHead(
            hidden_dim=hidden_dim, vocab_size=self.text_gen_dim,
            num_attractors=256, device=device_str
        )

        self.motor_text_proj = nn.Sequential(
            nn.Linear(hidden_dim, text_dim),
            nn.SiLU(),
            nn.LayerNorm(text_dim)
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
            h_chunk, m_curr = self.ssd_core.forward_chunk_parallel_ssd(chunk_emb, m_curr, curr_u_t, 1.0)

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

    def generate_sample(self, prompt: str, max_tokens: int = 50, temperature: float = 0.45, top_p: float = 0.90) -> str:
        self.eval()
        tokenizer = ByteTokenizer()
        prompt_ids = tokenizer.encode(prompt)
        if prompt_ids and prompt_ids[-1] == 257:
            prompt_ids = prompt_ids[:-1]

        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        m_curr = torch.zeros(1, self.num_heads, self.head_k, self.head_v, device=self.device)
        hu_state = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], device=self.device)

        prompt_emb = self.pos_embeddings(prompt_t, start_pos=0, apply_rf=True)
        h_ssm, m_curr = self.ssd_core.forward_chunk_parallel_ssd(prompt_emb, m_curr, hu_state, 1.0)
        h_chunk = self.channel_mixer(h_ssm)

        rolling_ids = prompt_ids.copy()
        generated_chars = []

        with torch.no_grad():
            for step in range(max_tokens):
                context = rolling_ids[-4:]
                win_t = torch.tensor([context], dtype=torch.long, device=self.device)
                win_emb = self.pos_embeddings(win_t, start_pos=len(rolling_ids) - len(context), apply_rf=True)
                t_emb = win_emb[:, -1:, :]

                h_step, m_curr = self.ssd_core.forward_chunk_parallel_ssd(t_emb, m_curr, hu_state, 1.0)
                h_out = self.channel_mixer(h_step)
                h_rel, _ = self.attractor_head.relax_to_minima(h_out, hu_state)
                h_proj = self.motor_text_proj(h_rel)

                logits = (F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim) / max(temperature, 1e-4)
                logits[:, 256:] = -1e9

                probs = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                if next_token_id == 257:
                    break
                rolling_ids.append(next_token_id)
                char_c = chr(next_token_id) if 32 <= next_token_id <= 126 or next_token_id in [9, 10, 13] else ' '
                generated_chars.append(char_c)

        self.train()
        return "".join(generated_chars).strip()


# =============================================================================
# 3. REAL CONVERSATIONAL DATASET PREPARATION
# =============================================================================
def prepare_packed_batches(num_batches: int = 150, batch_size: int = 32, seq_len: int = 512):
    logger.info("Loading Real Dataset (vicgalle/alpaca-gpt4) for Parity Evaluation...")
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
    tokenizer = ByteTokenizer()
    full_stream = []

    for item in ds:
        inst = item.get("instruction", "").strip()
        out = item.get("output", "").strip()
        if inst and out:
            dialog = f"User: {inst}\nKaryon: {out}"
            full_stream.extend(tokenizer.encode(dialog))
        if len(full_stream) >= num_batches * batch_size * (seq_len + 1):
            break

    batches = []
    block_size = seq_len + 1
    total_samples = num_batches * batch_size

    for b in range(num_batches):
        batch_tensors = []
        for s in range(batch_size):
            start = (b * batch_size + s) * block_size
            end = start + block_size
            chunk = full_stream[start:end]
            if len(chunk) < block_size:
                chunk = chunk + [256] * (block_size - len(chunk))
            batch_tensors.append(torch.tensor(chunk, dtype=torch.long))
        batches.append(torch.stack(batch_tensors, dim=0).to(device))

    logger.info(f"Prepared {len(batches)} Real Packed Batches (B={batch_size}, S={seq_len}).")
    return batches


# =============================================================================
# 4. RUN PARITY BENCHMARK: BASELINE vs EXP-50
# =============================================================================
def run_exp_50_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-50 (HIGH-RANK ASSOCIATIVE MATRIX SSD)] ===")
    print("="*85)
    print(f"Hardware Device : {device_str.upper()} | AMP FP16 Enabled: {use_amp}")

    b_size, seq_len = 32, 512
    num_eval_steps = 150
    chunk_size = 64

    batches = prepare_packed_batches(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)
    criterion = nn.CrossEntropyLoss(ignore_index=256)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    diag_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"

    # -------------------------------------------------------------------------
    # 1. EVALUATE MODEL A: BASELINE (K=32, V=64 | 16k STATE CAPACITY)
    # -------------------------------------------------------------------------
    print("\n[1/2] Benchmarking Model A: Baseline Canonical SSD (K=32, V=64)...")
    torch.manual_seed(42)
    model_a = ExperimentalCoREAgent(text_dim=256, hidden_dim=512, expand_dim=2048, num_heads=8, head_k=32, head_v=64, device_str=device_str).to(device)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=3e-3, weight_decay=0.01)
    hu_a = HomeostaticUnit(batch_size=b_size, device=device_str)

    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    t_start_a = time.perf_counter()
    losses_a = []

    for step in range(num_eval_steps):
        batch = batches[step]
        input_s = batch[:, :-1]
        target_s = batch[:, 1:]

        opt_a.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, speech_loss = model_a.forward_sequence(input_s, target_s, hu_a, criterion, chunk_size=chunk_size)

        scaler.scale(tot_loss).backward()
        scaler.unscale_(opt_a)
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), max_norm=3.0)
        scaler.step(opt_a)
        scaler.update()
        losses_a.append(speech_loss)

    if device.type == 'cuda': torch.cuda.synchronize()
    time_a_sec = time.perf_counter() - t_start_a
    vram_a_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0
    throughput_a = (num_eval_steps * b_size * seq_len) / time_a_sec
    final_loss_a = sum(losses_a[-25:]) / 25.0
    sample_a = model_a.generate_sample(diag_prompt, max_tokens=60)

    # -------------------------------------------------------------------------
    # 2. EVALUATE MODEL B: EXP-50 (K=64, V=128 | 65k STATE CAPACITY)
    # -------------------------------------------------------------------------
    print("\n[2/2] Benchmarking Model B: EXP-50 High-Rank Matrix SSD (K=64, V=128)...")
    torch.manual_seed(42)
    model_b = ExperimentalCoREAgent(text_dim=256, hidden_dim=512, expand_dim=2048, num_heads=8, head_k=64, head_v=128, device_str=device_str).to(device)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=3e-3, weight_decay=0.01)
    hu_b = HomeostaticUnit(batch_size=b_size, device=device_str)

    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    t_start_b = time.perf_counter()
    losses_b = []

    for step in range(num_eval_steps):
        batch = batches[step]
        input_s = batch[:, :-1]
        target_s = batch[:, 1:]

        opt_b.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, speech_loss = model_b.forward_sequence(input_s, target_s, hu_b, criterion, chunk_size=chunk_size)

        scaler.scale(tot_loss).backward()
        scaler.unscale_(opt_b)
        torch.nn.utils.clip_grad_norm_(model_b.parameters(), max_norm=3.0)
        scaler.step(opt_b)
        scaler.update()
        losses_b.append(speech_loss)

    if device.type == 'cuda': torch.cuda.synchronize()
    time_b_sec = time.perf_counter() - t_start_b
    vram_b_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0
    throughput_b = (num_eval_steps * b_size * seq_len) / time_b_sec
    final_loss_b = sum(losses_b[-25:]) / 25.0
    sample_b = model_b.generate_sample(diag_prompt, max_tokens=60)

    # -------------------------------------------------------------------------
    # 5. KEP RULE #2 DECISION & TELEMETRY DASHBOARD
    # -------------------------------------------------------------------------
    loss_delta = final_loss_b - final_loss_a
    ppl_a = math.exp(min(final_loss_a, 20.0))
    ppl_b = math.exp(min(final_loss_b, 20.0))
    speed_retention = (throughput_b / throughput_a) * 100.0

    print("\n" + "="*85)
    print(" === [KEP RULE #2 DECISION & EMPIRICAL TELEMETRY DASHBOARD] ===")
    print("="*85)
    print(f"{'Performance Metric':<32} | {'Model A (K=32, V=64)':<22} | {'Model B (K=64, V=128)':<22} | {'Delta':<12}")
    print("-" * 85)
    print(f"{'Final Steady-State Loss (nats)':<32} | {final_loss_a:<22.4f} | {final_loss_b:<22.4f} | {loss_delta:+12.4f}")
    print(f"{'Perplexity (PPL)':<32} | {ppl_a:<22.2f} | {ppl_b:<22.2f} | {ppl_b - ppl_a:+12.2f}")
    print(f"{'Throughput Speed (tok/s)':<32} | {throughput_a:<22.1f} | {throughput_b:<22.1f} | {speed_retention:11.1f}%")
    print(f"{'Peak VRAM Memory (MB)':<32} | {vram_a_mb:<22.1f} | {vram_b_mb:<22.1f} | {vram_b_mb - vram_a_mb:+12.1f} MB")
    print("="*85)

    print("\n" + "="*85)
    print(" === [KEP RULE #4 DIAGNOSTIC SPEECH SAMPLE AUDIT] ===")
    print("="*85)
    print(f"Prompt        : \"{diag_prompt}\"")
    print(f"Model A Output: \"{sample_a}\"")
    print(f"Model B Output: \"{sample_b}\"")
    print("="*85)

    if loss_delta <= -0.08 and speed_retention >= 80.0:
        verdict = "🟢 POSITIVE (Ready for Production Merge)"
    elif abs(loss_delta) < 0.08 and speed_retention >= 80.0:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    else:
        verdict = "🔴 REJECTED"

    print(f"\nDECISION VERDICT: {verdict}")
    print("="*85 + "\n")


if __name__ == "__main__":
    run_exp_50_benchmark()
