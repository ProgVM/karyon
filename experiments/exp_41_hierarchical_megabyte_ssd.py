"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-41 (HIERARCHICAL MEGABYTE SSM / HOURGLASS)
Full-Scale Evaluation of 2-Tier Hierarchical Cortical Patch SSD (P=4, N=512)
with NaN-Proof Modern Hopfield Dot-Product Attractor vs Baseline Flat 1-Byte SSD.
Protocol: KEP v5.0 (Rules #1, #2, #3, #4, #6, #7).
Biophysical Grounding: Cortical Temporal Hierarchy (Murray 2014) & BLT (Meta 2024).
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
    BatchedEpisodicMemory
)
from karyon_logger import get_logger

logger = get_logger()
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')
torch.set_grad_enabled(True)

print(f"\n[EXP-41] Initializing Stable Hierarchical MegaByte SSD Benchmark on: {device_str.upper()}")


# =============================================================================
# 3. NAN-PROOF MODERN CONTINUOUS HOPFIELD ATTRACTOR HEAD
# =============================================================================
class StableHopfieldAttractorHead(nn.Module):
    """Modern Continuous Hopfield Network with Dot-Product Energy (Ramsauer 2020)."""
    def __init__(self, hidden_dim=256, num_attractors=256, device_str='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_attractors = num_attractors
        self.scale = 1.0 / math.sqrt(hidden_dim)
        
        self.attractor_basins = nn.Parameter(torch.randn(num_attractors, hidden_dim) * 0.02)
        self.norm = nn.LayerNorm(hidden_dim)

    def relax_to_minima(self, h_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dot-Product Associative Energy (Singularity-Free & Fast)
        sim = torch.matmul(h_state, self.attractor_basins.transpose(0, 1)) * self.scale
        attn_weights = F.softmax(sim, dim=-1)
        attractor_shift = torch.matmul(attn_weights, self.attractor_basins)
        h_relaxed = self.norm(h_state + 0.25 * attractor_shift)
        energy = -torch.logsumexp(sim, dim=-1, keepdim=True)
        return h_relaxed, energy


# =============================================================================
# 4. POSITIONAL BYTE EMBEDDING (UNSHACKLED 256D)
# =============================================================================
class OffsetPositionalByteEmbedding(nn.Module):
    def __init__(self, vocab_size=258, text_dim=256, max_len=8192, device_str='cpu'):
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
# 5. BASELINE MODEL: FLAT 1-BYTE UNSHACKLED SSD (S=2048, 256D)
# =============================================================================
class BaselineFlatAgent(nn.Module):
    def __init__(self, device_str='cpu'):
        super().__init__()
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.text_dim = 256
        self.unified_dim = 256
        self.hidden_dim = 512
        self.num_heads = 8
        self.head_k = 32
        self.head_v = 64
        self.expand_dim = 2048
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
        self.attractor_head = StableHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, num_attractors=256, device_str=device_str
        ).to(self.device)
        
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)

    def get_all_parameters(self) -> List[nn.Parameter]:
        params = (
            list(self.pos_embeddings.parameters()) + 
            list(self.attractor_head.parameters()) + 
            list(self.motor_text_proj.parameters())
        )
        for sub in [self.ssd_core, self.channel_mixer]:
            if hasattr(sub, 'parameters'):
                params.extend(list(sub.parameters()))
        return params

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion: nn.Module, chunk_size: int = 64) -> torch.Tensor:
        batch_size, seq_len = input_seq.size()
        m_curr = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()

        num_chunks = max(1, seq_len // chunk_size)
        chunk_losses = []

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)

            chunk_in = input_seq[:, c_start:c_end]
            chunk_tgt = target_seq[:, c_start:c_end]

            chunk_emb = self.pos_embeddings(chunk_in, start_pos=c_start, apply_rf=True)
            
            ssd_out = self.ssd_core.forward_chunk_parallel_ssd(chunk_emb, m_curr, curr_u_t, 1.0)
            h_chunk, m_curr = ssd_out[0], ssd_out[1]
            h_reasoned = self.channel_mixer(h_chunk)

            h_relaxed, _ = self.attractor_head.relax_to_minima(h_reasoned)
            h_proj = self.motor_text_proj(h_relaxed)
            logits_flat = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim

            chunk_loss = criterion(logits_flat, chunk_tgt.contiguous().view(-1))
            chunk_losses.append(chunk_loss)

            with torch.no_grad():
                has_eos = (chunk_in == 257).any(dim=-1).view(batch_size, 1, 1, 1).float()
                m_curr = m_curr * (1.0 - has_eos)

            m_curr = m_curr.detach()

        return torch.stack(chunk_losses).mean()

    def generate_thought_and_speech(self, prompt: str, hu, max_generated_tokens: int = 75,
                                   temperature: float = 0.45, top_p: float = 0.90) -> str:
        prompt_ids = [t for t in self.tokenizer.encode(prompt) if t != 257]
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

            h_relaxed, _ = self.attractor_head.relax_to_minima(h_out)
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
# 6. PROPOSED MODEL: EXP-41 HIERARCHICAL MEGABYTE SSM AGENT (P=4, N=512)
# =============================================================================
class HierarchicalMegaByteSSDAgent(nn.Module):
    def __init__(self, patch_size: int = 4, device_str: str = 'cpu'):
        super().__init__()
        self.patch_size = patch_size
        self.device_str = device_str
        self.device = torch.device(device_str)

        self.text_dim = 256
        self.unified_dim = 256
        self.hidden_dim = 512
        self.num_heads = 8
        self.head_k = 32
        self.head_v = 64
        self.expand_dim = 2048
        self.text_gen_dim = 258
        self.inv_sqrt_text_dim = 1.0 / math.sqrt(self.text_dim)

        self.tokenizer = ByteTokenizer(vocab_size=self.text_gen_dim)
        self.pos_embeddings = OffsetPositionalByteEmbedding(
            vocab_size=self.text_gen_dim, text_dim=self.text_dim, max_len=8192, device_str=device_str
        ).to(self.device)

        # 1. Primary Sensory Cortex: Causal Strided Patch Encoder (4 bytes -> Z_macro 512D)
        self.patch_encoder = nn.Sequential(
            nn.Linear(self.patch_size * self.text_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.text_dim)
        ).to(self.device)

        # 2. Association / Prefrontal Cortex: Global Cognitive SSD Core (N=512 Macro-Tokens)
        self.global_ssd = CalibratedParallelSSDCore(
            text_dim=self.text_dim, unified_dim=self.unified_dim, hidden_dim=self.hidden_dim,
            num_heads=self.num_heads, head_k=self.head_k, head_v=self.head_v, device=device_str
        )
        self.global_swiglu = ParallelSwiGLUBlock(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, device=device_str
        )

        # Start of Sequence Macro Token
        self.sos_macro = nn.Parameter(torch.randn(1, 1, self.hidden_dim, device=self.device) * 0.02)

        # 3. Top-Down Macro-to-Micro Highway Projection (512D -> 256D)
        self.macro_to_byte_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)

        # 4. Local Articulatory Cortex: Fuses Local Byte Embedding + Top-Down Macro Context
        self.local_decoder = nn.Sequential(
            nn.Linear(self.text_dim + self.text_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.text_dim)
        ).to(self.device)

        self.attractor_head = StableHopfieldAttractorHead(
            hidden_dim=self.text_dim, num_attractors=256, device_str=device_str
        ).to(self.device)

    def get_all_parameters(self) -> List[nn.Parameter]:
        params = (
            list(self.pos_embeddings.parameters()) + 
            list(self.patch_encoder.parameters()) + 
            list(self.macro_to_byte_proj.parameters()) + 
            list(self.local_decoder.parameters()) + 
            list(self.attractor_head.parameters()) + 
            [self.sos_macro]
        )
        for sub in [self.global_ssd, self.global_swiglu]:
            if hasattr(sub, 'parameters'):
                params.extend(list(sub.parameters()))
        return params

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion: nn.Module, chunk_size: int = 64) -> torch.Tensor:
        batch_size, raw_seq_len = input_seq.size()
        
        # 1. Protective alignment to multiple of patch_size (P=4)
        num_patches = raw_seq_len // self.patch_size
        seq_len = num_patches * self.patch_size
        input_seq = input_seq[:, :seq_len]
        target_seq = target_seq[:, :seq_len]

        # 2. Local Byte Embeddings [B, S, 256]
        byte_emb = self.pos_embeddings(input_seq, start_pos=0, apply_rf=True)

        # 3. Patch Compression: [B, N, P * 256] -> [B, N, 256]
        patches_flat = byte_emb.view(batch_size, num_patches, self.patch_size * self.text_dim)
        macro_patches = self.patch_encoder(patches_flat)

        # 4. Global Cognitive SSD Scan on N=512 Patches (8 chunks of Q=64!)
        m_global = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()

        macro_chunk_size = chunk_size
        num_macro_chunks = max(1, num_patches // macro_chunk_size)
        macro_outputs = []

        for m_idx in range(num_macro_chunks):
            mc_start = m_idx * macro_chunk_size
            mc_end = min((m_idx + 1) * macro_chunk_size, num_patches)
            mc_len = mc_end - mc_start

            chunk_macro_in = macro_patches[:, mc_start:mc_end, :]
            ssd_out = self.global_ssd.forward_chunk_parallel_ssd(chunk_macro_in, m_global, curr_u_t, 1.0)
            h_macro_ssm, m_global = ssd_out[0], ssd_out[1]
            h_macro_reasoned = self.global_swiglu(h_macro_ssm).view(batch_size, mc_len, self.hidden_dim)
            macro_outputs.append(h_macro_reasoned)

            m_global = m_global.detach()

        h_global_all = torch.cat(macro_outputs, dim=1) # [B, N, 512]

        # 5. Causal Macro-Context Shift: Patch k is conditioned on macro state 0..k-1
        sos_expanded = self.sos_macro.repeat(batch_size, 1, 1)
        h_macro_causal = torch.cat([sos_expanded, h_global_all[:, :-1, :]], dim=1) # [B, N, 512]

        # 6. Broadcast Macro-Context & Project to Byte Space [B, S, 256]
        macro_per_byte = h_macro_causal.repeat_interleave(self.patch_size, dim=1).view(batch_size * seq_len, self.hidden_dim)
        h_macro_highway = self.macro_to_byte_proj(macro_per_byte).view(batch_size, seq_len, self.text_dim)

        # 7. Local Articulatory Decoder Fusion: (Local Byte + Top-Down Macro Context)
        local_in = torch.cat([byte_emb, h_macro_highway], dim=-1) # [B, S, 512]
        h_local = self.local_decoder(local_in) # [B, S, 256]

        # 8. Modern Hopfield Attractor Stabilization + Tied Output
        h_relaxed, _ = self.attractor_head.relax_to_minima(h_local)
        logits = F.linear(h_relaxed, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim

        loss = criterion(logits.view(-1, self.text_gen_dim), target_seq.contiguous().view(-1))
        return loss

    def generate_thought_and_speech(self, prompt: str, hu, max_generated_tokens: int = 75,
                                   temperature: float = 0.45, top_p: float = 0.90) -> str:
        prompt_ids = [t for t in self.tokenizer.encode(prompt) if t != 257]
        rolling_ids = list(prompt_ids)
        generated_chars = []

        for step in range(max_generated_tokens):
            cur_len = len(rolling_ids)
            pad_needed = (self.patch_size - (cur_len % self.patch_size)) % self.patch_size
            eval_ids = rolling_ids + [256] * pad_needed
            
            t_eval = torch.tensor([eval_ids], dtype=torch.long, device=self.device)
            byte_emb = self.pos_embeddings(t_eval, start_pos=0, apply_rf=True)
            
            num_patches = t_eval.size(1) // self.patch_size
            patches_flat = byte_emb.view(1, num_patches, self.patch_size * self.text_dim)
            macro_patches = self.patch_encoder(patches_flat)

            m_dummy = torch.zeros(1, self.num_heads, self.head_k, self.head_v, device=self.device)
            ssd_out = self.global_ssd.forward_chunk_parallel_ssd(macro_patches, m_dummy, hu.state, 1.0)
            h_macro = self.global_swiglu(ssd_out[0]).view(1, num_patches, self.hidden_dim)

            sos_exp = self.sos_macro.repeat(1, 1, 1)
            h_macro_causal = torch.cat([sos_exp, h_macro[:, :-1, :]], dim=1)
            macro_per_byte = h_macro_causal.repeat_interleave(self.patch_size, dim=1).view(t_eval.size(1), self.hidden_dim)
            h_macro_highway = self.macro_to_byte_proj(macro_per_byte).view(1, t_eval.size(1), self.text_dim)

            local_in = torch.cat([byte_emb, h_macro_highway], dim=-1)
            h_local = self.local_decoder(local_in)
            
            cur_pos = cur_len - 1
            last_h = h_local[:, cur_pos, :]
            last_relaxed, _ = self.attractor_head.relax_to_minima(last_h)
            
            logits = (F.linear(last_relaxed, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim) / max(temperature, 1e-4)
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
# 7. DATASET: CONTINUOUS PACKED STREAMING (S=2048, 0% PADDING)
# =============================================================================
class ContinuousPackedDataset(Dataset):
    def __init__(self, hf_data, tokenizer, max_samples=100, seq_len=2048):
        self.seq_len = seq_len
        full_token_stream = []

        for item in hf_data:
            inst = item.get("instruction", "").strip()
            out = item.get("output", "").strip()
            if inst and out:
                dialog = f"User: {inst}\nKaryon: {out}"
                ids = tokenizer.encode(dialog)
                full_token_stream.extend(ids)
            if len(full_token_stream) >= (max_samples * 16 * (seq_len + 1)):
                break

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

def collate_packed_fn(batch):
    return torch.stack(batch, dim=0)


# =============================================================================
# 8. EXP-41 MASTER EXECUTION & AUTOMATED DECISION ENGINE
# =============================================================================
def run_exp_41_benchmark():
    print("\n" + "="*85)
    print(" === EXP-41: STABLE HIERARCHICAL MEGABYTE SSM CAPACITY BENCHMARK ===")
    print("="*85)

    tokenizer = ByteTokenizer()
    logger.info("Loading vicgalle/alpaca-gpt4 dataset for KEP Rule #7 Parity Evaluation...")
    raw_dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")

    # Split into Train (50,000) and Held-Out Validation (2,002)
    train_split = raw_dataset.select(range(0, 50000))
    val_split = raw_dataset.select(range(50000, len(raw_dataset)))

    NUM_STEPS = 100
    BATCH_SIZE = 16
    SEQ_LEN = 2048

    dataset_train = ContinuousPackedDataset(train_split, tokenizer, max_samples=NUM_STEPS, seq_len=SEQ_LEN)
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_packed_fn, drop_last=True)

    dataset_val = ContinuousPackedDataset(val_split, tokenizer, max_samples=15, seq_len=SEQ_LEN)
    loader_val = DataLoader(dataset_val, batch_size=8, shuffle=False, collate_fn=collate_packed_fn, drop_last=True)

    logger.info(f"Parity Setup Ready. Train Steps: {NUM_STEPS} | S={SEQ_LEN} | Val Batches: {len(loader_val)}")

    criterion = nn.CrossEntropyLoss(ignore_index=256)

    def get_lr_multiplier(step):
        if step < 5:
            return float(step + 1) / 5.0
        progress = float(step - 5) / float(max(1, NUM_STEPS - 5))
        return 0.5 * (1.0 + math.cos(math.pi * progress * 0.7))

    # -------------------------------------------------------------------------
    # PART A: RUN BASELINE FLAT 1-BYTE UNSHACKLED SSD (100 STEPS)
    # -------------------------------------------------------------------------
    print("\n" + "-"*85)
    print(" [1/2] RUNNING BASELINE: FLAT 1-BYTE UNSHACKLED SSD (S=2048, 256D)")
    print("-" * 85)

    model_base = BaselineFlatAgent(device_str=device_str).to(device)
    param_count_base = sum(p.numel() for p in model_base.get_all_parameters())
    opt_base = torch.optim.AdamW(model_base.get_all_parameters(), lr=3e-3, weight_decay=0.01)
    sched_base = torch.optim.lr_scheduler.LambdaLR(opt_base, lr_lambda=get_lr_multiplier)
    scaler_base = torch.amp.GradScaler('cuda', enabled=use_amp)

    loss_history_base = []
    t_start_base = time.perf_counter()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()

    for idx, batch_tokens in enumerate(loader_train):
        if idx >= NUM_STEPS: break
        batch_tokens = batch_tokens.to(device)
        in_seq = batch_tokens[:, :-1]
        tgt_seq = batch_tokens[:, 1:]
        hu = HomeostaticUnit(batch_size=BATCH_SIZE, device=device_str)

        opt_base.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            loss = model_base.forward_sequence(in_seq, tgt_seq, hu, criterion, chunk_size=64)

        scaler_base.scale(loss).backward()
        scaler_base.unscale_(opt_base)
        torch.nn.utils.clip_grad_norm_(model_base.get_all_parameters(), max_norm=3.0)
        
        scale_before = scaler_base.get_scale()
        scaler_base.step(opt_base)
        scaler_base.update()
        scale_after = scaler_base.get_scale()
        
        if scale_before <= scale_after:
            sched_base.step()

        loss_history_base.append(loss.item())
        if (idx + 1) % 25 == 0 or idx == NUM_STEPS - 1:
            print(f"  Baseline Step [{idx+1:03d}/{NUM_STEPS}] | Train Loss: {loss.item():.4f} (PPL: {math.exp(min(loss.item(), 20.0)):.2f})")

    t_total_base = time.perf_counter() - t_start_base
    vram_base = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0
    tok_per_sec_base = (len(loss_history_base) * BATCH_SIZE * SEQ_LEN) / t_total_base
    final_train_loss_base = sum(loss_history_base[-10:]) / 10.0

    # -------------------------------------------------------------------------
    # PART B: RUN PROPOSED EXP-41 HIERARCHICAL MEGABYTE SSD (100 STEPS)
    # -------------------------------------------------------------------------
    print("\n" + "-"*85)
    print(" [2/2] RUNNING PROPOSED: EXP-41 HIERARCHICAL MEGABYTE SSD (P=4, N=512)")
    print("-" * 85)

    model_prop = HierarchicalMegaByteSSDAgent(patch_size=4, device_str=device_str).to(device)
    param_count_prop = sum(p.numel() for p in model_prop.get_all_parameters())
    opt_prop = torch.optim.AdamW(model_prop.get_all_parameters(), lr=3e-3, weight_decay=0.01)
    sched_prop = torch.optim.lr_scheduler.LambdaLR(opt_prop, lr_lambda=get_lr_multiplier)
    scaler_prop = torch.amp.GradScaler('cuda', enabled=use_amp)

    loss_history_prop = []
    t_start_prop = time.perf_counter()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()

    for idx, batch_tokens in enumerate(loader_train):
        if idx >= NUM_STEPS: break
        batch_tokens = batch_tokens.to(device)
        in_seq = batch_tokens[:, :-1]
        tgt_seq = batch_tokens[:, 1:]
        hu = HomeostaticUnit(batch_size=BATCH_SIZE, device=device_str)

        opt_prop.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            loss = model_prop.forward_sequence(in_seq, tgt_seq, hu, criterion, chunk_size=64)

        scaler_prop.scale(loss).backward()
        scaler_prop.unscale_(opt_prop)
        torch.nn.utils.clip_grad_norm_(model_prop.get_all_parameters(), max_norm=3.0)
        
        scale_before = scaler_prop.get_scale()
        scaler_prop.step(opt_prop)
        scaler_prop.update()
        scale_after = scaler_prop.get_scale()
        
        if scale_before <= scale_after:
            sched_prop.step()

        loss_history_prop.append(loss.item())
        if (idx + 1) % 25 == 0 or idx == NUM_STEPS - 1:
            print(f"  Proposed Step [{idx+1:03d}/{NUM_STEPS}] | Train Loss: {loss.item():.4f} (PPL: {math.exp(min(loss.item(), 20.0)):.2f})")

    t_total_prop = time.perf_counter() - t_start_prop
    vram_prop = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0
    tok_per_sec_prop = (len(loss_history_prop) * BATCH_SIZE * SEQ_LEN) / t_total_prop
    final_train_loss_prop = sum(loss_history_prop[-10:]) / 10.0

    # -------------------------------------------------------------------------
    # PART C: HELD-OUT COMMON VALIDATION EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "="*85)
    print(" === HELD-OUT VALIDATION SET EVALUATION ===")
    print("="*85)
    model_base.eval()
    model_prop.eval()

    val_loss_base_accum = 0.0
    val_loss_prop_accum = 0.0
    val_batches = len(loader_val)

    with torch.no_grad():
        for batch_tokens in loader_val:
            batch_tokens = batch_tokens.to(device)
            in_seq = batch_tokens[:, :-1]
            tgt_seq = batch_tokens[:, 1:]
            hu_val = HomeostaticUnit(batch_size=batch_tokens.size(0), device=device_str)

            with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
                loss_b = model_base.forward_sequence(in_seq, tgt_seq, hu_val, criterion, chunk_size=64)
                loss_p = model_prop.forward_sequence(in_seq, tgt_seq, hu_val, criterion, chunk_size=64)

            val_loss_base_accum += loss_b.item()
            val_loss_prop_accum += loss_p.item()

    final_val_loss_base = val_loss_base_accum / float(val_batches)
    final_val_loss_prop = val_loss_prop_accum / float(val_batches)

    print(f"  Held-Out Validation Loss (Baseline Flat SSD): {final_val_loss_base:.4f} (PPL: {math.exp(min(final_val_loss_base, 20.0)):.2f})")
    print(f"  Held-Out Validation Loss (Proposed EXP-41):   {final_val_loss_prop:.4f} (PPL: {math.exp(min(final_val_loss_prop, 20.0)):.2f})")
    print("="*85)

    # -------------------------------------------------------------------------
    # PART D: LIVE SHORT & LONG CONVERSATIONAL AUDITING (KEP RULE #4)
    # -------------------------------------------------------------------------
    print("\n" + "="*85)
    print(" === KEP RULE #4: LIVE CONVERSATIONAL AUDIT (SHORT & LONG PROMPTS) ===")
    print("="*85)
    diag_hu = HomeostaticUnit(batch_size=1, device=device_str)
    
    # 1. Short Casual Prompt (Testing Anti-Schizophasia)
    short_prompt = "User: Hello!\nKaryon:"
    sample_short_base = model_base.generate_thought_and_speech(short_prompt, diag_hu, max_generated_tokens=60)
    sample_short_prop = model_prop.generate_thought_and_speech(short_prompt, diag_hu, max_generated_tokens=60)

    # 2. Formal Technical Prompt
    long_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"
    sample_long_base = model_base.generate_thought_and_speech(long_prompt, diag_hu, max_generated_tokens=70)
    sample_long_prop = model_prop.generate_thought_and_speech(long_prompt, diag_hu, max_generated_tokens=70)

    print(f"  [Short Prompt: 'Hello!']")
    print(f"    • Baseline Output: \"{sample_short_base}\"")
    print(f"    • EXP-41 Proposed: \"{sample_short_prop}\"")
    print(f"\n  [Long Prompt: 'Energy for Earth']")
    print(f"    • Baseline Output: \"{sample_long_base}\"")
    print(f"    • EXP-41 Proposed: \"{sample_long_prop}\"")
    print("="*85)

    # -------------------------------------------------------------------------
    # PART E: FINAL TELEMETRY COMPARISON & KEP VERDICT (KEP RULE #2)
    # -------------------------------------------------------------------------
    delta_val_loss = final_val_loss_prop - final_val_loss_base
    speed_ratio = (tok_per_sec_prop / tok_per_sec_base) * 100.0

    print("\n" + "="*85)
    print(" === EXP-41 FINAL TELEMETRY AUDIT TABLE ===")
    print("="*85)
    print(f"{'Metric':<34} | {'Baseline (Flat 1-Byte SSD)':<28} | {'Proposed (EXP-41 MegaByte SSM)':<32}")
    print("-" * 102)
    print(f"{'Parameter Count':<34} | {param_count_base:<28,} | {param_count_prop:<32,}")
    print(f"{'Temporal Hierarchy':<34} | {'Flat 1-Byte (S=2048)':<28} | {'2-Tier Macro Patch (P=4, N=512)':<32}")
    print(f"{'Final Train Loss':<34} | {final_train_loss_base:<28.4f} | {final_train_loss_prop:<32.4f}")
    print(f"{'Held-Out Validation Loss':<34} | {final_val_loss_base:<28.4f} | {final_val_loss_prop:<32.4f}")
    print(f"{'Validation Perplexity (PPL)':<34} | {math.exp(min(final_val_loss_base, 20.0)):<28.2f} | {math.exp(min(final_val_loss_prop, 20.0)):<32.2f}")
    print(f"{'Throughput Speed':<34} | {f'{tok_per_sec_base:,.1f} tok/s':<28} | {f'{tok_per_sec_prop:,.1f} tok/s':<32}")
    print(f"{'Peak VRAM Memory':<34} | {f'{vram_base:.1f} MB':<28} | {f'{vram_prop:.1f} MB':<32}")
    print(f"{'Total Execution Time':<34} | {f'{t_total_base:.2f} s':<28} | {f'{t_total_prop:.2f} s':<32}")
    print("="*102)

    print(f"\n[EXP-41 Decision Engine] Delta Validation Loss: {delta_val_loss:+.4f} | Speed Ratio: {speed_ratio:.1f}%")
    
    if delta_val_loss <= -0.08 and speed_ratio >= 75.0:
        verdict = "🟢 POSITIVE — Significant hierarchical gain on held-out validation with high throughput. Ready for production merge."
    elif delta_val_loss <= 0.05 and speed_ratio >= 65.0:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE — Metric within statistical noise margin."
    else:
        verdict = "🔴 REJECTED — Regression in validation loss unacceptable."

    print(f"\n>>> FINAL VERDICT: {verdict}\n" + "="*102 + "\n")

if __name__ == "__main__":
    run_exp_41_benchmark()
