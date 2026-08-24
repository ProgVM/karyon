"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-39 (CONTINUOUS PACKED STREAM + 2048 CONTEXT)
Evaluation of Zero-Padding Packed Streaming & 2048-Byte Long Context Horizon
vs Baseline 512-Byte Truncated Stream on vicgalle/alpaca-gpt4 (Parity 1M Tokens).
Protocol: KEP v5.0 (Rules #1, #2, #3, #4, #6, #7).
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import os
import sys

# 1. Guaranteed Repository Root Path Resolution (KEP Principle 6)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
curr_dir = os.path.abspath(os.path.dirname(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)
os.chdir(repo_root)

import types
import time
import math
import importlib
from typing import Generator, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

# =============================================================================
# 2. DYNAMO HOTFIX FOR PYTHON 3.12 / KAGGLE GPU
# =============================================================================
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

import karyon_config, karyon_core, karyon_logger
from karyon_core import (
    ByteTokenizer,
    HomeostaticUnit,
    CausalByteReceptiveField,
    CalibratedParallelSSDCore,
    ParallelSwiGLUBlock,
    DesaturatedHopfieldAttractorHead,
    BatchedEpisodicMemory
)
from karyon_logger import get_logger

logger = get_logger()
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
torch.set_grad_enabled(True)

print(f"\n[EXP-39] Initializing Packed Long-Context SSD Benchmark on: {device_str.upper()}")


# =============================================================================
# 3. POSITIONAL BYTE EMBEDDING (SCALED UP TO 8192 BYTES)
# =============================================================================
class OffsetPositionalByteEmbedding(nn.Module):
    def __init__(self, vocab_size=258, text_dim=128, max_len=8192, device_str='cpu'):
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
# 4. MASTER MULTI-TIMESCALE SSD AGENT (CANONICAL V16.0 ARCHITECTURE)
# =============================================================================
class MasterSSDAgent(nn.Module):
    def __init__(self, device_str='cpu'):
        super().__init__()
        self.device_str = device_str
        self.device = torch.device(device_str)

        self.text_dim = 128
        self.unified_dim = 256
        self.hidden_dim = 512
        self.num_heads = 8
        self.head_k = 32
        self.head_v = 64
        self.expand_dim = 1536
        self.text_gen_dim = 258
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.tokenizer = ByteTokenizer(vocab_size=self.text_gen_dim)
        self.pos_embeddings = OffsetPositionalByteEmbedding(
            vocab_size=self.text_gen_dim, text_dim=self.text_dim, max_len=8192, device_str=device_str
        ).to(self.device)

        self.ssd_core = CalibratedParallelSSDCore(
            text_dim=self.text_dim, unified_dim=self.unified_dim, hidden_dim=self.hidden_dim,
            num_heads=self.num_heads, head_k=self.head_k, head_v=self.head_v, device=device_str
        )
        self.channel_mixer = ParallelSwiGLUBlock(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, device=device_str
        )
        self.attractor_head = DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, vocab_size=self.text_gen_dim, device=device_str
        )
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)

    def get_all_parameters(self) -> List[nn.Parameter]:
        params = list(self.pos_embeddings.parameters()) + list(self.motor_text_proj.parameters())
        for sub in [self.ssd_core, self.channel_mixer, self.attractor_head]:
            if hasattr(sub, 'parameters'):
                params.extend(list(sub.parameters()))
        return params

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion: nn.Module, chunk_size: int = 64, optimizer: torch.optim.Optimizer = None):
        batch_size, seq_len = input_seq.size()
        m_curr = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()

        num_chunks = max(1, seq_len // chunk_size)
        total_loss_accum = 0.0

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)

            chunk_in = input_seq[:, c_start:c_end]
            chunk_tgt = target_seq[:, c_start:c_end]

            chunk_emb = self.pos_embeddings(chunk_in, start_pos=c_start, apply_rf=True)
            
            ssd_out = self.ssd_core.forward_chunk_parallel_ssd(chunk_emb, m_curr, curr_u_t, 1.0)
            h_chunk, m_curr = ssd_out[0], ssd_out[1]
            h_reasoned = self.channel_mixer(h_chunk)

            h_relaxed = self.attractor_head.relax_to_minima(h_reasoned)[0]
            h_proj = self.motor_text_proj(h_relaxed)
            logits_flat = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim

            chunk_loss = criterion(logits_flat, chunk_tgt.contiguous().view(-1))
            total_loss_accum += chunk_loss.item()

            # Theta Phase Reset on Event Boundary (EOS id=257)
            with torch.no_grad():
                has_eos = (chunk_in == 257).any(dim=-1).view(batch_size, 1, 1, 1).float()
                m_curr = m_curr * (1.0 - has_eos)

            if optimizer is not None:
                (chunk_loss / float(num_chunks)).backward()

            m_curr = m_curr.detach()

        return total_loss_accum / float(num_chunks)

    def generate_thought_and_speech(self, prompt: str, hu, max_generated_tokens: int = 65,
                                   temperature: float = 0.7, top_p: float = 0.90) -> str:
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_embs = self.pos_embeddings(prompt_tokens, start_pos=0, apply_rf=True)
        
        m_curr = torch.zeros(1, self.num_heads, self.head_k, self.head_v, device=self.device)
        ssd_out = self.ssd_core.forward_chunk_parallel_ssd(prompt_embs, m_curr, hu.state, 1.0)
        h_ssm, m_curr = ssd_out[0], ssd_out[1]
        h_chunk = self.channel_mixer(h_ssm)

        rolling_ids = prompt_tokens[0].tolist()
        total_prompt_len = len(rolling_ids)
        generated_chars = []

        for step in range(max_generated_tokens):
            ctx_ids = rolling_ids[-4:]
            ctx_t = torch.tensor([ctx_ids], dtype=torch.long, device=self.device)
            ctx_start = (total_prompt_len + step) - (len(ctx_ids) - 1)

            ctx_emb = self.pos_embeddings(ctx_t, start_pos=ctx_start, apply_rf=True)[:, -1:, :]

            ssd_out = self.ssd_core.forward_chunk_parallel_ssd(ctx_emb, m_curr, hu.state, 1.0)
            h_step_ssm, m_curr = ssd_out[0], ssd_out[1]
            h_out = self.channel_mixer(h_step_ssm)

            h_relaxed = self.attractor_head.relax_to_minima(h_out)[0]
            h_proj = self.motor_text_proj(h_relaxed)
            
            logits = (F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim) / max(temperature, 1e-4)
            logits[:, 256:] = -1e9

            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -1e9

            probs = F.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            prob_sum = probs.sum(dim=-1, keepdim=True)
            probs = probs / prob_sum if (prob_sum > 0).all() else torch.full_like(probs, 1.0 / 258)

            next_token_id = torch.multinomial(probs, num_samples=1).item()
            rolling_ids.append(next_token_id)

            if next_token_id == 257:
                break
            char = chr(next_token_id) if 32 <= next_token_id <= 126 or next_token_id in [9, 10, 13] else ' '
            generated_chars.append(char)

        return "".join(generated_chars).strip()


# =============================================================================
# 5. DATASETS: BASELINE (512 TRUNCATED + PAD) VS PROPOSED (2048 PACKED STREAM)
# =============================================================================
class BaselineTruncatedDataset(Dataset):
    def __init__(self, hf_data, tokenizer, max_samples=60, max_len=512):
        self.samples = []
        for item in hf_data:
            inst = item.get("instruction", "").strip()
            out = item.get("output", "").strip()
            if inst and out:
                dialog = f"User: {inst}\nKaryon: {out}"
                ids = tokenizer.encode(dialog)
                if len(ids) > max_len:
                    ids = ids[:max_len-1] + [257]
                if len(ids) > 16:
                    self.samples.append(torch.tensor(ids, dtype=torch.long))
            if len(self.samples) >= max_samples * 32:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class ContinuousPackedDataset(Dataset):
    """Zero-Padding Continuous Stream Packing with EOS Separators."""
    def __init__(self, hf_data, tokenizer, target_tokens=1000000, seq_len=2048):
        self.seq_len = seq_len
        full_token_stream = []

        for item in hf_data:
            inst = item.get("instruction", "").strip()
            out = item.get("output", "").strip()
            if inst and out:
                dialog = f"User: {inst}\nKaryon: {out}"
                ids = tokenizer.encode(dialog) # Contains 257 (<eos>) at the end!
                full_token_stream.extend(ids)
            if len(full_token_stream) >= target_tokens + seq_len + 10:
                break

        # Slice into uniform blocks of size seq_len + 1 (for input/target shift)
        num_blocks = len(full_token_stream) // (seq_len + 1)
        self.samples = []
        for b_idx in range(num_blocks):
            start = b_idx * (seq_len + 1)
            end = start + (seq_len + 1)
            chunk = full_token_stream[start:end]
            self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_pad_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=256)

def collate_packed_fn(batch):
    return torch.stack(batch, dim=0)


# =============================================================================
# 6. EXP-39 MASTER EXECUTION & AUTOMATED DECISION ENGINE
# =============================================================================
def run_exp_39_benchmark():
    print("\n" + "="*85)
    print(" === EXP-39: CONTINUOUS PACKED STREAM + 2048 CONTEXT BENCHMARK ===")
    print("="*85)

    tokenizer = ByteTokenizer()
    logger.info("Loading vicgalle/alpaca-gpt4 dataset for KEP Rule #7 Parity Evaluation...")
    raw_dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")

    # 1. Baseline Dataset (512 Truncated + Pad): 60 batches * 32 * 512 = 983,040 tokens
    dataset_base = BaselineTruncatedDataset(raw_dataset, tokenizer, max_samples=60, max_len=512)
    loader_base = DataLoader(dataset_base, batch_size=32, shuffle=True, collate_fn=collate_pad_fn, drop_last=True)

    # 2. Proposed Dataset (2048 Continuous Packed Stream): 30 batches * 16 * 2048 = 983,040 tokens (Exact Parity!)
    dataset_prop = ContinuousPackedDataset(raw_dataset, tokenizer, target_tokens=1000000, seq_len=2048)
    loader_prop = DataLoader(dataset_prop, batch_size=16, shuffle=True, collate_fn=collate_packed_fn, drop_last=True)

    logger.info(f"Parity Setup Ready. Baseline Batches: {len(loader_base)} (S=512) | Proposed Batches: {len(loader_prop)} (S=2048)")

    criterion = nn.CrossEntropyLoss(ignore_index=256)

    # -------------------------------------------------------------------------
    # PART A: RUN BASELINE TRUNCATED 512 STREAM (WITH PADDING)
    # -------------------------------------------------------------------------
    print("\n" + "-"*85)
    print(" [1/2] RUNNING BASELINE: 512-BYTE TRUNCATED STREAM (WITH PADDING)")
    print("-" * 85)

    model_base = MasterSSDAgent(device_str=device_str).to(device)
    opt_base = torch.optim.AdamW(model_base.get_all_parameters(), lr=3e-3, weight_decay=0.01)

    loss_history_base = []
    t_start_base = time.perf_counter()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()

    for idx, batch_tokens in enumerate(loader_base):
        batch_tokens = batch_tokens.to(device)
        in_seq = batch_tokens[:, :-1]
        tgt_seq = batch_tokens[:, 1:]
        hu = HomeostaticUnit(batch_size=32, device=device_str)

        opt_base.zero_grad()
        loss = model_base.forward_sequence(in_seq, tgt_seq, hu, criterion, chunk_size=64, optimizer=opt_base)
        torch.nn.utils.clip_grad_norm_(model_base.get_all_parameters(), max_norm=3.0)
        opt_base.step()

        loss_history_base.append(loss)
        if (idx + 1) % 20 == 0 or idx == len(loader_base) - 1:
            print(f"  Baseline Step [{idx+1:02d}/{len(loader_base)}] | Loss: {loss:.4f} (PPL: {math.exp(min(loss, 20.0)):.2f})")

    t_total_base = time.perf_counter() - t_start_base
    vram_base = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0
    tok_per_sec_base = (len(loss_history_base) * 32 * 512) / t_total_base
    final_loss_base = sum(loss_history_base[-10:]) / 10.0

    # -------------------------------------------------------------------------
    # PART B: RUN PROPOSED 2048 CONTINUOUS PACKED STREAM (ZERO PADDING)
    # -------------------------------------------------------------------------
    print("\n" + "-"*85)
    print(" [2/2] RUNNING PROPOSED: EXP-39 CONTINUOUS PACKED STREAM (S=2048, 0% PADDING)")
    print("-" * 85)

    model_prop = MasterSSDAgent(device_str=device_str).to(device)
    opt_prop = torch.optim.AdamW(model_prop.get_all_parameters(), lr=3e-3, weight_decay=0.01)

    loss_history_prop = []
    t_start_prop = time.perf_counter()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()

    for idx, batch_tokens in enumerate(loader_prop):
        batch_tokens = batch_tokens.to(device)
        in_seq = batch_tokens[:, :-1]
        tgt_seq = batch_tokens[:, 1:]
        hu = HomeostaticUnit(batch_size=16, device=device_str)

        opt_prop.zero_grad()
        loss = model_prop.forward_sequence(in_seq, tgt_seq, hu, criterion, chunk_size=64, optimizer=opt_prop)
        torch.nn.utils.clip_grad_norm_(model_prop.get_all_parameters(), max_norm=3.0)
        opt_prop.step()

        loss_history_prop.append(loss)
        if (idx + 1) % 10 == 0 or idx == len(loader_prop) - 1:
            print(f"  Proposed Step [{idx+1:02d}/{len(loader_prop)}] | Loss: {loss:.4f} (PPL: {math.exp(min(loss, 20.0)):.2f})")

    t_total_prop = time.perf_counter() - t_start_prop
    vram_prop = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0
    tok_per_sec_prop = (len(loss_history_prop) * 16 * 2048) / t_total_prop
    final_loss_prop = sum(loss_history_prop[-10:]) / 10.0

    # -------------------------------------------------------------------------
    # PART C: DIAGNOSTIC SPEECH SAMPLE (KEP RULE #4)
    # -------------------------------------------------------------------------
    print("\n" + "="*85)
    print(" === KEP RULE #4: LIVE DIAGNOSTIC SPEECH SAMPLES ===")
    print("="*85)
    diag_hu = HomeostaticUnit(batch_size=1, device=device_str)
    prompt_1 = "User: What is the primary source of energy for Earth?\nKaryon:"
    prompt_2 = "User: How does the human brain process memories?\nKaryon:"
    
    sample_1 = model_prop.generate_thought_and_speech(prompt_1, diag_hu, max_generated_tokens=75)
    sample_2 = model_prop.generate_thought_and_speech(prompt_2, diag_hu, max_generated_tokens=75)

    print(f"  Prompt 1: \"{prompt_1.strip()}\"")
    print(f"  Output 1: \"{sample_1}\"")
    print(f"\n  Prompt 2: \"{prompt_2.strip()}\"")
    print(f"  Output 2: \"{sample_2}\"")
    print("="*85)

    # -------------------------------------------------------------------------
    # PART D: GRADIENT FLOW VERIFICATION DASHBOARD (KEP RULE #6)
    # -------------------------------------------------------------------------
    print("\n" + "="*85)
    print(" === EXP-39 GRADIENT FLOW INSPECTION (PROPOSED MODEL) ===")
    print("="*85)
    for name, param in model_prop.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.norm().item()
            print(f"  {name:<55} | Grad Norm: {g_norm:<12.6f} | {'✅ HEALTHY' if g_norm > 0 else '⚠️ ZERO'}")
    print("="*85)

    # -------------------------------------------------------------------------
    # PART E: FINAL TELEMETRY COMPARISON & KEP VERDICT (KEP RULE #2)
    # -------------------------------------------------------------------------
    delta_loss = final_loss_prop - final_loss_base
    speed_ratio = (tok_per_sec_prop / tok_per_sec_base) * 100.0

    print("\n" + "="*85)
    print(" === EXP-39 FINAL TELEMETRY AUDIT TABLE ===")
    print("="*85)
    print(f"{'Metric':<32} | {'Baseline (512 Truncated + Pad)':<32} | {'Proposed (EXP-39 2048 Packed Stream)':<36}")
    print("-" * 100)
    print(f"{'Context Window (S)':<32} | {'512 Bytes (~80 words)':<32} | {'2,048 Bytes (~400 words)':<36}")
    print(f"{'Padding Overhead':<32} | {'~50% (Padded with 256)':<32} | {'0% (100% Pure Signal)':<36}")
    print(f"{'Final Convergence Loss':<32} | {final_loss_base:<32.4f} | {final_loss_prop:<36.4f}")
    print(f"{'Perplexity (PPL)':<32} | {math.exp(min(final_loss_base, 20.0)):<32.2f} | {math.exp(min(final_loss_prop, 20.0)):<36.2f}")
    print(f"{'Throughput Speed':<32} | {f'{tok_per_sec_base:,.1f} tok/s':<32} | {f'{tok_per_sec_prop:,.1f} tok/s':<36}")
    print(f"{'Peak VRAM Memory':<32} | {f'{vram_base:.1f} MB':<32} | {f'{vram_prop:.1f} MB':<36}")
    print(f"{'Total Execution Time':<32} | {f'{t_total_base:.2f} s':<32} | {f'{t_total_prop:.2f} s':<36}")
    print("="*100)

    print(f"\n[EXP-39 Decision Engine] Delta Loss: {delta_loss:+.4f} | Throughput Ratio: {speed_ratio:.1f}%")
    
    if delta_loss <= -0.08 and speed_ratio >= 80.0:
        verdict = "🟢 POSITIVE — Significant long-context packed gain on real dataset with high throughput retention. Ready for production merge."
    elif delta_loss <= 0.05 and speed_ratio >= 70.0:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE — Metric within statistical variance margin."
    else:
        verdict = "🔴 REJECTED — Regression in loss or throughput unacceptable."

    print(f"\n>>> FINAL VERDICT: {verdict}\n" + "="*100 + "\n")

if __name__ == "__main__":
    run_exp_39_benchmark()
