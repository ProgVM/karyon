"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-46 (BIOPHYSICAL SLEEP & PLASTICITY CYCLE)
Evaluation of Autonomous Sleep-Replay Consolidation (Tononi & Cirelli SHY) &
Three-Factor Neuromodulated Synaptic Plasticity vs Baseline Continuous SSD.
Protocol: KEP v6.1 (Rules #1, #2, #3, #4, #6, #7).
Biophysical Basis: Synaptic Homeostasis Hypothesis (Tononi 2014), Three-Factor
Hebbian Plasticity (Gerstner 2016), and Active Inference (Friston 2010).
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
    DesaturatedHopfieldAttractorHead,
    LatentPredictor,
    BatchedEpisodicMemory
)
from karyon_logger import get_logger

logger = get_logger()
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')
torch.set_grad_enabled(True)

print(f"\n[EXP-46] Initializing Biophysical Sleep & Plasticity Benchmark on: {device_str.upper()}")


# =============================================================================
# 3. POSITIONAL BYTE EMBEDDING WITH RECEPTIVE FIELD (UNSHACKLED 256D)
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
# 4. BASELINE MODEL: STANDARD CONTINUOUS SSD (NO SLEEP / NO REPLAY CONSOLIDATION)
# =============================================================================
class BaselineSSDAgent(nn.Module):
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
        self.attractor_head = DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, vocab_size=self.text_gen_dim, num_attractors=256, device=device_str
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

            h_relaxed, _ = self.attractor_head.relax_to_minima(h_reasoned, curr_u_t)
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

            h_relaxed, _ = self.attractor_head.relax_to_minima(h_out, hu.state)
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
# 5. PROPOSED MODEL: BIOPHYSICAL ORGANISM WITH SLEEP REPLAY & 3-FACTOR PLASTICITY
# =============================================================================
class BiophysicalLivingAgent(nn.Module):
    """Living Cognitive Entity with Sleep Consolidation & Three-Factor Plasticity."""
    def __init__(self, device_str='cpu'):
        super().__init__()
        self.device_str = device_str
        self.device = torch.device(device_str)

        self.text_dim = 256
        self.unified_dim = 256
        self.hidden_dim = 512
        self.latent_dim = 128
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

        # Time-Mixing SSD Core with Log-Spaced Temporal Rhythm
        self.ssd_core = CalibratedParallelSSDCore(
            text_dim=self.text_dim, unified_dim=self.unified_dim, hidden_dim=self.hidden_dim,
            num_heads=self.num_heads, head_k=self.head_k, head_v=self.head_v, device=device_str
        )
        
        # Channel-Mixing SwiGLU Reasoning Block
        self.channel_mixer = ParallelSwiGLUBlock(
            hidden_dim=self.hidden_dim, expand_dim=self.expand_dim, device=device_str
        )
        
        # Active Inference World Model
        self.world_model = LatentPredictor(
            hidden_dim=self.hidden_dim, unified_dim=self.unified_dim, latent_dim=self.latent_dim, device=device_str
        )
        
        # Dopamine-Modulated Modern Hopfield Attractor Head
        self.attractor_head = DesaturatedHopfieldAttractorHead(
            hidden_dim=self.hidden_dim, vocab_size=self.text_gen_dim, num_attractors=256, device=device_str
        )
        
        # Afferent-Efferent Lexical Tied Projection
        self.motor_text_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.text_dim),
            nn.SiLU(),
            nn.LayerNorm(self.text_dim)
        ).to(self.device)

        self.episodic_proj = nn.Linear(self.text_dim, self.unified_dim).to(self.device)

    def get_all_parameters(self) -> List[nn.Parameter]:
        params = (
            list(self.pos_embeddings.parameters()) + 
            list(self.episodic_proj.parameters()) +
            list(self.motor_text_proj.parameters())
        )
        for sub in [self.ssd_core, self.channel_mixer, self.world_model, self.attractor_head]:
            if hasattr(sub, 'parameters'):
                params.extend(list(sub.parameters()))
        return params

    def forward_sequence(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                         criterion: nn.Module, episodic_memory: BatchedEpisodicMemory,
                         chunk_size: int = 64) -> Tuple[torch.Tensor, float, float]:
        batch_size, seq_len = input_seq.size()
        m_curr = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()
        h_prev_fast = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        num_chunks = max(1, seq_len // chunk_size)
        chunk_losses = []
        fe_losses = []

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)
            c_len = c_end - c_start

            chunk_in = input_seq[:, c_start:c_end]
            chunk_tgt = target_seq[:, c_start:c_end]

            chunk_emb = self.pos_embeddings(chunk_in, start_pos=c_start, apply_rf=True)

            # Continuous Parallel SSD Scan
            ssd_out = self.ssd_core.forward_chunk_parallel_ssd(chunk_emb, m_curr, curr_u_t, 1.0)
            h_chunk, m_curr = ssd_out[0], ssd_out[1]
            h_reasoned = self.channel_mixer(h_chunk)

            # Dopamine-Sharpened Hopfield Attractors
            h_relaxed, _ = self.attractor_head.relax_to_minima(h_reasoned, curr_u_t)
            h_proj = self.motor_text_proj(h_relaxed)
            logits_flat = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim

            chunk_loss = criterion(logits_flat, chunk_tgt.contiguous().view(-1))
            chunk_losses.append(chunk_loss)

            # Active Inference Prediction Mismatch
            w_current_slice = self.episodic_proj(chunk_emb[:, -1, :])
            h_curr_fast = h_reasoned.view(batch_size, c_len, self.hidden_dim)[:, -1, :]
            w_pred, kl_div, _, _ = self.world_model(h_prev_fast, h_curr_fast, w_current_slice)
            h_prev_fast = h_curr_fast.detach()

            rec_loss = (1.0 - F.cosine_similarity(w_current_slice, w_pred, dim=-1, eps=1e-8)).mean()
            chunk_fe = (kl_div.mean() + rec_loss)
            fe_losses.append(chunk_fe)

            # One-Shot Hippocampal Encoding on High Surprise (F_t > 0.20)
            with torch.no_grad():
                if chunk_fe.item() > 0.20 and episodic_memory is not None:
                    episodic_memory.write(w_current_slice.detach(), w_pred.detach(), protected_slots=3)

                has_eos = (chunk_in == 257).any(dim=-1).view(batch_size, 1, 1, 1).float()
                m_curr = m_curr * (1.0 - has_eos)

            m_curr = m_curr.detach()

        avg_speech_loss = torch.stack(chunk_losses).mean()
        avg_fe_loss = torch.stack(fe_losses).mean()
        total_loss = avg_speech_loss + 0.05 * avg_fe_loss

        return total_loss, avg_speech_loss.item(), avg_fe_loss.item()

    # =========================================================================
    # BIOPHYSICAL SLEEP CONSOLIDATION & REPLAY ENGINE (TONONI & HASSAIS)
    # =========================================================================
    def execute_sleep_consolidation(self, episodic_memory: BatchedEpisodicMemory, hu: HomeostaticUnit,
                                    num_replay_cycles: int = 5, downscaling_factor: float = 0.05):
        """Executes full sleep cycle: Hippocampal Replay + Synaptic Downscaling + Somatic Recovery."""
        self.train()
        
        # 1. Closed Sensory Gate: Sleep Replay of High-Surprise Memories
        active_memory_slots = episodic_memory.size.max().item()
        if active_memory_slots > 3:
            opt_replay = torch.optim.AdamW(self.get_all_parameters(), lr=5e-4, weight_decay=0.01)
            
            for _ in range(num_replay_cycles):
                opt_replay.zero_grad()
                # Sample replayed memories from hippocampus
                rand_indices = torch.randint(0, active_memory_slots, (min(16, active_memory_slots),), device=self.device)
                replayed_keys = episodic_memory.keys[0, rand_indices, :] # [M, 256]
                replayed_vals = episodic_memory.values[0, rand_indices, :] # [M, 256]

                # Consolidation forward pass through associative world model
                h_dummy = torch.zeros(replayed_keys.size(0), self.hidden_dim, device=self.device)
                w_pred, kl_div, _, _ = self.world_model(h_dummy, h_dummy, replayed_keys)
                replay_loss = (1.0 - F.cosine_similarity(w_pred, replayed_vals, dim=-1)).mean() + kl_div.mean() * 0.05
                
                replay_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.get_all_parameters(), max_norm=2.0)
                opt_replay.step()

        # 2. Synaptic Homeostasis Downscaling (Tononi & Cirelli SHY)
        # Eliminates daily noise and restores metabolic efficiency
        with torch.no_grad():
            for param in self.get_all_parameters():
                if param.dim() > 1: # Matrix weights only
                    param.mul_(1.0 - downscaling_factor)

        # 3. Somatic Homeostatic Recovery (ATP restoration & NA reset)
        with torch.no_grad():
            hu.state[:, 1] = 1.00 # Energy restored
            hu.state[:, 2] = 1.00 # Stability restored
            hu.state[:, 3] = 1.00 # Health restored
            hu.state[:, 4] = 0.05 # Arousal reset to calm baseline

    def generate_thought_and_speech(self, prompt: str, hu, episodic_memory: BatchedEpisodicMemory = None,
                                   max_generated_tokens: int = 75, temperature: float = 0.45, top_p: float = 0.90) -> str:
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

            # Volitional Memory Recall (Active Inference Precision Gating)
            if episodic_memory is not None and hu.state[0, 4].item() > 0.12:
                q_k = self.episodic_proj(ctx_emb.squeeze(1))
                ret_mem, max_sim = episodic_memory.read(q_k, temperature=0.05, threshold=0.75, sigmoid_beta=15.0)
                if (max_sim > 0.75).any():
                    h_out = h_out + ret_mem.repeat(1, h_out.size(1) // ret_mem.size(1) if h_out.size(1) > ret_mem.size(1) else 1)[:, :h_out.size(1)] * 0.20

            h_relaxed, _ = self.attractor_head.relax_to_minima(h_out, hu.state)
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
# 6. DATASET: CONTINUOUS PACKED STREAMING (S=2048, 0% PADDING)
# =============================================================================
class ContinuousPackedDataset(Dataset):
    def __init__(self, hf_data, tokenizer, max_samples=600, seq_len=2048):
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
# 7. EXP-46 MASTER EXECUTION & BIOPHYSICAL BENCHMARK ENGINE
# =============================================================================
def run_exp_46_benchmark():
    print("\n" + "="*85)
    print(" === EXP-46: BIOPHYSICAL SLEEP CONSOLIDATION & PLASTICITY BENCHMARK ===")
    print("="*85)

    tokenizer = ByteTokenizer()
    logger.info("Loading vicgalle/alpaca-gpt4 dataset for KEP Rule #7 Parity Evaluation...")
    raw_dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")

    train_split = raw_dataset.select(range(0, 50000))
    val_split = raw_dataset.select(range(50000, len(raw_dataset)))

    NUM_STEPS = 600
    BATCH_SIZE = 16
    SEQ_LEN = 2048

    dataset_train = ContinuousPackedDataset(train_split, tokenizer, max_samples=NUM_STEPS, seq_len=SEQ_LEN)
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_packed_fn, drop_last=True)

    dataset_val = ContinuousPackedDataset(val_split, tokenizer, max_samples=25, seq_len=SEQ_LEN)
    loader_val = DataLoader(dataset_val, batch_size=8, shuffle=False, collate_fn=collate_packed_fn, drop_last=True)

    logger.info(f"Biophysical Setup Ready. Train Steps: {NUM_STEPS} (~20M tokens) | S={SEQ_LEN} | Val Batches: {len(loader_val)}")

    criterion = nn.CrossEntropyLoss(ignore_index=256)

    def get_lr_multiplier(step):
        if step < 25:
            return float(step + 1) / 25.0
        progress = float(step - 25) / float(max(1, NUM_STEPS - 25))
        return 0.5 * (1.0 + math.cos(math.pi * progress * 0.7))

    # -------------------------------------------------------------------------
    # PART A: RUN BASELINE SSD (NO SLEEP CONSOLIDATION)
    # -------------------------------------------------------------------------
    print("\n" + "-"*85)
    print(" [1/2] RUNNING BASELINE: CONTINUOUS SSD (NO SLEEP CYCLE)")
    print("-" * 85)

    model_base = BaselineSSDAgent(device_str=device_str).to(device)
    param_count_base = sum(p.numel() for p in model_base.get_all_parameters())
    opt_base = torch.optim.AdamW(model_base.get_all_parameters(), lr=3e-3, weight_decay=0.01)
    sched_base = torch.optim.lr_scheduler.LambdaLR(opt_base, lr_lambda=get_lr_multiplier)
    scaler_base = torch.amp.GradScaler('cuda', enabled=use_amp)
    hu_base = HomeostaticUnit(batch_size=BATCH_SIZE, device=device_str)

    loss_history_base = []
    t_start_base = time.perf_counter()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()

    for idx, batch_tokens in enumerate(loader_train):
        if idx >= NUM_STEPS: break
        batch_tokens = batch_tokens.to(device)
        in_seq = batch_tokens[:, :-1]
        tgt_seq = batch_tokens[:, 1:]

        opt_base.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            loss = model_base.forward_sequence(in_seq, tgt_seq, hu_base, criterion, chunk_size=64)

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
        if (idx + 1) % 150 == 0 or idx == NUM_STEPS - 1:
            print(f"  Baseline Step [{idx+1:04d}/{NUM_STEPS}] | Loss: {loss.item():.4f} (PPL: {math.exp(min(loss.item(), 20.0)):.2f})")

    t_total_base = time.perf_counter() - t_start_base
    vram_base = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0
    tok_per_sec_base = (len(loss_history_base) * BATCH_SIZE * SEQ_LEN) / t_total_base
    final_train_loss_base = sum(loss_history_base[-25:]) / 25.0

    # -------------------------------------------------------------------------
    # PART B: RUN PROPOSED BIOPHYSICAL AGENT (SLEEP REPLAY & 3-FACTOR PLASTICITY)
    # -------------------------------------------------------------------------
    print("\n" + "-"*85)
    print(" [2/2] RUNNING PROPOSED: EXP-46 BIOPHYSICAL LIVING AGENT (WAKE + SLEEP REPLAY)")
    print("-" * 85)

    model_prop = BiophysicalLivingAgent(device_str=device_str).to(device)
    param_count_prop = sum(p.numel() for p in model_prop.get_all_parameters())
    opt_prop = torch.optim.AdamW(model_prop.get_all_parameters(), lr=3e-3, weight_decay=0.01)
    sched_prop = torch.optim.lr_scheduler.LambdaLR(opt_prop, lr_lambda=get_lr_multiplier)
    scaler_prop = torch.amp.GradScaler('cuda', enabled=use_amp)
    hu_prop = HomeostaticUnit(batch_size=BATCH_SIZE, device=device_str)
    mem_prop = BatchedEpisodicMemory(batch_size=BATCH_SIZE, memory_dim=256, max_capacity=500, device=device_str)

    loss_history_prop = []
    t_start_prop = time.perf_counter()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()

    sleep_count = 0

    for idx, batch_tokens in enumerate(loader_train):
        if idx >= NUM_STEPS: break
        batch_tokens = batch_tokens.to(device)
        in_seq = batch_tokens[:, :-1]
        tgt_seq = batch_tokens[:, 1:]

        opt_prop.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            total_loss, speech_l, fe_l = model_prop.forward_sequence(
                in_seq, tgt_seq, hu_prop, criterion, episodic_memory=mem_prop, chunk_size=64
            )

        # Three-Factor Plasticity Modulation: Modulated by Somatic Arousal & Energy
        curiosity, energy, stability, health, na, da = hu_prop.state[0].tolist()
        somatic_lr_mod = 0.5 + 0.5 * energy + 0.3 * na

        scaler_prop.scale(total_loss * somatic_lr_mod).backward()
        scaler_prop.unscale_(opt_prop)
        torch.nn.utils.clip_grad_norm_(model_prop.get_all_parameters(), max_norm=3.0)
        
        scale_before = scaler_prop.get_scale()
        scaler_prop.step(opt_prop)
        scaler_prop.update()
        scale_after = scaler_prop.get_scale()
        
        if scale_before <= scale_after:
            sched_prop.step()

        # Update Somatic State
        action_cost_t = torch.full((BATCH_SIZE, 1), 0.003, device=device)
        pred_err_t = torch.full((BATCH_SIZE, 1), float(speech_l * 0.1), device=device)
        entropy_t = torch.full((BATCH_SIZE, 1), float(fe_l), device=device)
        cog_t = torch.zeros((BATCH_SIZE, 1), dtype=torch.int64, device=device)
        hu_prop.update(action_cost_t, pred_err_t, entropy_t, cog_t)

        loss_history_prop.append(speech_l)

        # Autonomous Sleep Consolidation Cycle Trigger (Every 200 steps or Energy < 0.25)
        if (idx + 1) % 200 == 0 or hu_prop.state[0, 1].item() < 0.25:
            model_prop.execute_sleep_consolidation(mem_prop, hu_prop, num_replay_cycles=3, downscaling_factor=0.03)
            sleep_count += 1
            print(f"  💤 [Sleep Cycle #{sleep_count} @ Step {idx+1:04d}] Replay Consolidated | Synaptic Downscaling Applied | Energy Restored: {hu_prop.state[0, 1].item():.2f}")

        if (idx + 1) % 150 == 0 or idx == NUM_STEPS - 1:
            print(f"  Proposed Step [{idx+1:04d}/{NUM_STEPS}] | Loss: {speech_l:.4f} (PPL: {math.exp(min(speech_l, 20.0)):.2f}) | F_t: {fe_l:.4f} | Energy: {hu_prop.state[0, 1].item():.2f}")

    t_total_prop = time.perf_counter() - t_start_prop
    vram_prop = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0
    tok_per_sec_prop = (len(loss_history_prop) * BATCH_SIZE * SEQ_LEN) / t_total_prop
    final_train_loss_prop = sum(loss_history_prop[-25:]) / 25.0

    # -------------------------------------------------------------------------
    # PART C: HELD-OUT COMMON VALIDATION EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "="*85)
    print(" === HELD-OUT VALIDATION SET EVALUATION (25 BATCHES) ===")
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
                loss_p, _, _ = model_prop.forward_sequence(in_seq, tgt_seq, hu_val, criterion, episodic_memory=None, chunk_size=64)

            val_loss_base_accum += loss_b.item()
            val_loss_prop_accum += loss_p.item()

    final_val_loss_base = val_loss_base_accum / float(val_batches)
    final_val_loss_prop = val_loss_prop_accum / float(val_batches)

    print(f"  Held-Out Validation Loss (Baseline SSD):       {final_val_loss_base:.4f} (PPL: {math.exp(min(final_val_loss_base, 20.0)):.2f})")
    print(f"  Held-Out Validation Loss (Proposed EXP-46):   {final_val_loss_prop:.4f} (PPL: {math.exp(min(final_val_loss_prop, 20.0)):.2f})")
    print("="*85)

    # -------------------------------------------------------------------------
    # PART D: LIVE SHORT & LONG CONVERSATIONAL AUDITING (KEP RULE #4)
    # -------------------------------------------------------------------------
    print("\n" + "="*85)
    print(" === KEP RULE #4: LIVE CONVERSATIONAL AUDIT (SHORT & LONG PROMPTS) ===")
    print("="*85)
    diag_hu = HomeostaticUnit(batch_size=1, device=device_str)
    
    short_prompt = "User: Hello!\nKaryon:"
    sample_short_base = model_base.generate_thought_and_speech(short_prompt, diag_hu, max_generated_tokens=60)
    sample_short_prop = model_prop.generate_thought_and_speech(short_prompt, diag_hu, episodic_memory=mem_prop, max_generated_tokens=60)

    long_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"
    sample_long_base = model_base.generate_thought_and_speech(long_prompt, diag_hu, max_generated_tokens=70)
    sample_long_prop = model_prop.generate_thought_and_speech(long_prompt, diag_hu, episodic_memory=mem_prop, max_generated_tokens=70)

    print(f"  [Short Prompt: 'Hello!']")
    print(f"    • Baseline Output: \"{sample_short_base}\"")
    print(f"    • EXP-46 Proposed: \"{sample_short_prop}\"")
    print(f"\n  [Long Prompt: 'Energy for Earth']")
    print(f"    • Baseline Output: \"{sample_long_base}\"")
    print(f"    • EXP-46 Proposed: \"{sample_long_prop}\"")
    print("="*85)

    # -------------------------------------------------------------------------
    # PART E: FINAL TELEMETRY COMPARISON & KEP VERDICT (KEP RULE #2)
    # -------------------------------------------------------------------------
    delta_val_loss = final_val_loss_prop - final_val_loss_base
    speed_ratio = (tok_per_sec_prop / tok_per_sec_base) * 100.0

    print("\n" + "="*85)
    print(" === EXP-46 FINAL TELEMETRY AUDIT TABLE (600 STEPS) ===")
    print("="*85)
    print(f"{'Metric':<34} | {'Baseline (No Sleep Replay)':<28} | {'Proposed (EXP-46 Living Cycle)':<32}")
    print("-" * 102)
    print(f"{'Biological Memory Mechanics':<34} | {'Static Weights Only':<28} | {'Wake + Sleep Replay + SHY':<32}")
    print(f"{'Plasticity Control':<34} | {'Standard Uniform Step':<28} | {'3-Factor Neuromodulated':<32}")
    print(f"{'Sleep Cycles Executed':<34} | {'0 (Never Sleeps)':<28} | {f'{sleep_count} Sleep Consolidations':<32}")
    print(f"{'Final Train Loss':<34} | {final_train_loss_base:<28.4f} | {final_train_loss_prop:<32.4f}")
    print(f"{'Held-Out Validation Loss':<34} | {final_val_loss_base:<28.4f} | {final_val_loss_prop:<32.4f}")
    print(f"{'Validation Perplexity (PPL)':<34} | {math.exp(min(final_val_loss_base, 20.0)):<28.2f} | {math.exp(min(final_val_loss_prop, 20.0)):<32.2f}")
    print(f"{'Throughput Speed':<34} | {f'{tok_per_sec_base:,.1f} tok/s':<28} | {f'{tok_per_sec_prop:,.1f} tok/s':<32}")
    print(f"{'Peak VRAM Memory':<34} | {f'{vram_base:.1f} MB':<28} | {f'{vram_prop:.1f} MB':<32}")
    print(f"{'Total Execution Time':<34} | {f'{t_total_base:.2f} s':<28} | {f'{t_total_prop:.2f} s':<32}")
    print("="*102)

    print(f"\n[EXP-46 Decision Engine] Delta Validation Loss: {delta_val_loss:+.4f} | Speed Ratio: {speed_ratio:.1f}%")
    
    if delta_val_loss <= -0.08 and speed_ratio >= 75.0:
        verdict = "🟢 POSITIVE — Significant biophysical gain on held-out validation with preserved throughput. Ready for production merge."
    elif delta_val_loss <= 0.05 and speed_ratio >= 65.0:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE — Metric within statistical noise margin."
    else:
        verdict = "🔴 REJECTED — Regression in validation loss unacceptable."

    print(f"\n>>> FINAL VERDICT: {verdict}\n" + "="*102 + "\n")

if __name__ == "__main__":
    run_exp_46_benchmark()
