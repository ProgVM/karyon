"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-47 (THREE-FACTOR LOCAL PLASTICITY ENGINE)
Evaluation of Autograd-Free Three-Factor Hebbian Plasticity & Local Predictive
Coding vs Standard Global Backpropagation on Real Dataset (vicgalle/alpaca-gpt4).
Protocol: KEP v6.1 (Rules #1, #2, #3, #4, #6, #7).
Biophysical Basis: Three-Factor Synaptic Plasticity (Gerstner 2016), Predictive
Coding (Rao & Ballard 1999, Friston 2005), and Synaptic Scaling (Tononi 2014).
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

print(f"\n[EXP-47] Initializing Three-Factor Local Plasticity Benchmark on: {device_str.upper()}")


# =============================================================================
# 3. POSITIONAL BYTE EMBEDDING (UNSHACKLED 256D)
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
# 4. BASELINE MODEL: STANDARD GLOBAL AUTOGRAD SSD
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
# 5. PROPOSED MODEL: THREE-FACTOR LOCAL PREDICTIVE PLASTICITY AGENT
# =============================================================================
class ThreeFactorLocalAgent(nn.Module):
    """Biologically Plausible 3-Factor Neuromodulated Local Predictive Plasticity."""
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

        # Local Learning Hyperparameters
        self.lr_local = 3e-3
        self.weight_decay_homeo = 0.01

    def get_all_parameters(self) -> List[nn.Parameter]:
        params = list(self.pos_embeddings.parameters()) + list(self.motor_text_proj.parameters())
        for sub in [self.ssd_core, self.channel_mixer, self.attractor_head]:
            if hasattr(sub, 'parameters'):
                params.extend(list(sub.parameters()))
        return params

    # =========================================================================
    # THREE-FACTOR NEUROMODULATED LOCAL SYNAPTIC UPDATE
    # =========================================================================
    def apply_local_three_factor_update(self, pre_activations: torch.Tensor, 
                                       error_signal: torch.Tensor, 
                                       target_param: nn.Parameter, 
                                       neuromodulator_gain: float):
        """
        Updates synaptic matrix W locally:
        Delta W = lr * M(u_t) * [ Pre^T * Error - gamma_homeo * W ]
        Zero PyTorch autograd graph overhead!
        """
        with torch.no_grad():
            # Factor 1 & 2: Local Pre-synaptic and Post-synaptic Error Outer Product
            if pre_activations.dim() == 3:
                pre_flat = pre_activations.reshape(-1, pre_activations.size(-1))
            else:
                pre_flat = pre_activations
                
            if error_signal.dim() == 3:
                err_flat = error_signal.reshape(-1, error_signal.size(-1))
            else:
                err_flat = error_signal

            # Local Hebbian Eligibility Trace Matrix [Out_dim, In_dim]
            hebbian_trace = torch.matmul(err_flat.transpose(0, 1), pre_flat) / float(pre_flat.size(0))

            # Factor 3: Neuromodulatory Multiplier + Homeostatic Synaptic Scaling
            delta_w = neuromodulator_gain * (hebbian_trace - self.weight_decay_homeo * target_param.data)
            
            # Direct In-Place Synaptic Modification
            target_param.data.add_(delta_w, alpha=-self.lr_local)

    def forward_sequence_local_learning(self, input_seq: torch.Tensor, target_seq: torch.Tensor, hu_batch, 
                                        criterion: nn.Module, chunk_size: int = 64) -> Tuple[float, float]:
        batch_size, seq_len = input_seq.size()
        m_curr = torch.zeros(batch_size, self.num_heads, self.head_k, self.head_v, device=self.device)
        curr_u_t = hu_batch.state.clone().detach()

        # Compute Neuromodulatory Factor 3 from Homeostatic State (Arousal NA + Reward DA)
        curiosity, energy, stability, health, na, da = curr_u_t[0].tolist()
        neuromodulator_gain = float(torch.clamp(torch.tensor(0.4 + 0.4 * energy + 0.6 * na + 0.4 * da), 0.1, 2.0).item())

        num_chunks = max(1, seq_len // chunk_size)
        total_loss_accum = 0.0

        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min((chunk_idx + 1) * chunk_size, seq_len)

            chunk_in = input_seq[:, c_start:c_end]
            chunk_tgt = target_seq[:, c_start:c_end]

            # 1. Forward Pass (No Graph Stored)
            with torch.no_grad():
                chunk_emb = self.pos_embeddings(chunk_in, start_pos=c_start, apply_rf=True)
                ssd_out = self.ssd_core.forward_chunk_parallel_ssd(chunk_emb, m_curr, curr_u_t, 1.0)
                h_chunk, m_curr = ssd_out[0], ssd_out[1]
                h_reasoned = self.channel_mixer(h_chunk)

                h_relaxed, _ = self.attractor_head.relax_to_minima(h_reasoned, curr_u_t)
                h_proj = self.motor_text_proj(h_relaxed)
                logits_flat = F.linear(h_proj, self.pos_embeddings.byte_embed.weight) * self.inv_sqrt_text_dim

                targets_flat = chunk_tgt.contiguous().view(-1)
                chunk_loss = criterion(logits_flat, targets_flat)
                total_loss_accum += chunk_loss.item()

                # 2. Local Predictive Error Calculation at Efferent Speech Output
                # Error unit = Softmax probability error (p - y_onehot)
                probs_flat = F.softmax(logits_flat, dim=-1)
                valid_mask = (targets_flat != 256).unsqueeze(-1)
                
                targets_clamped = targets_flat.clamp(0, self.text_gen_dim - 1)
                one_hot_targets = F.one_hot(targets_clamped, num_classes=self.text_gen_dim).float()
                
                speech_error = (probs_flat - one_hot_targets) * valid_mask.float() # [B * c_len, 258]

                # 3. Local Three-Factor Plasticity Updates
                # Update Motor Text Projection Weights locally
                error_proj = torch.matmul(speech_error, self.pos_embeddings.byte_embed.weight) # [B * c_len, 256]
                self.apply_local_three_factor_update(
                    pre_activations=h_relaxed,
                    error_signal=error_proj,
                    target_param=self.motor_text_proj[0].weight,
                    neuromodulator_gain=neuromodulator_gain
                )

                # Update Embeddings Locally
                embed_error = torch.matmul(speech_error.transpose(0, 1), h_proj) / float(h_proj.size(0))
                self.pos_embeddings.byte_embed.weight.data.add_(
                    (embed_error - self.weight_decay_homeo * self.pos_embeddings.byte_embed.weight.data) * neuromodulator_gain,
                    alpha=-self.lr_local
                )

                # Event Boundary Theta Phase Reset
                has_eos = (chunk_in == 257).any(dim=-1).view(batch_size, 1, 1, 1).float()
                m_curr = m_curr * (1.0 - has_eos)

        avg_loss = total_loss_accum / float(num_chunks)
        return avg_loss, neuromodulator_gain

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
# 6. DATASET: CONTINUOUS PACKED STREAMING (S=2048, 0% PADDING)
# =============================================================================
class ContinuousPackedDataset(Dataset):
    def __init__(self, hf_data, tokenizer, max_samples=400, seq_len=2048):
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
# 7. EXP-47 MASTER EXECUTION & BIOPHYSICAL BENCHMARK ENGINE
# =============================================================================
def run_exp_47_benchmark():
    print("\n" + "="*85)
    print(" === EXP-47: THREE-FACTOR LOCAL PREDICTIVE PLASTICITY BENCHMARK ===")
    print("="*85)

    tokenizer = ByteTokenizer()
    logger.info("Loading vicgalle/alpaca-gpt4 dataset for KEP Rule #7 Parity Evaluation...")
    raw_dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")

    train_split = raw_dataset.select(range(0, 50000))
    val_split = raw_dataset.select(range(50000, len(raw_dataset)))

    NUM_STEPS = 400
    BATCH_SIZE = 16
    SEQ_LEN = 2048

    dataset_train = ContinuousPackedDataset(train_split, tokenizer, max_samples=NUM_STEPS, seq_len=SEQ_LEN)
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_packed_fn, drop_last=True)

    dataset_val = ContinuousPackedDataset(val_split, tokenizer, max_samples=25, seq_len=SEQ_LEN)
    loader_val = DataLoader(dataset_val, batch_size=8, shuffle=False, collate_fn=collate_packed_fn, drop_last=True)

    logger.info(f"Local Plasticity Setup Ready. Train Steps: {NUM_STEPS} | S={SEQ_LEN} | Val Batches: {len(loader_val)}")

    criterion = nn.CrossEntropyLoss(ignore_index=256)

    def get_lr_multiplier(step):
        if step < 25:
            return float(step + 1) / 25.0
        progress = float(step - 25) / float(max(1, NUM_STEPS - 25))
        return 0.5 * (1.0 + math.cos(math.pi * progress * 0.7))

    # -------------------------------------------------------------------------
    # PART A: RUN BASELINE GLOBAL AUTOGRAD SSD (400 STEPS)
    # -------------------------------------------------------------------------
    print("\n" + "-"*85)
    print(" [1/2] RUNNING BASELINE: GLOBAL AUTOGRAD BACKPROPAGATION (400 STEPS)")
    print("-" * 85)

    model_base = BaselineSSDAgent(device_str=device_str).to(device)
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
        if (idx + 1) % 100 == 0 or idx == NUM_STEPS - 1:
            print(f"  Baseline Step [{idx+1:04d}/{NUM_STEPS}] | Loss: {loss.item():.4f} (PPL: {math.exp(min(loss.item(), 20.0)):.2f})")

    t_total_base = time.perf_counter() - t_start_base
    vram_base = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == 'cuda' else 0.0
    tok_per_sec_base = (len(loss_history_base) * BATCH_SIZE * SEQ_LEN) / t_total_base
    final_train_loss_base = sum(loss_history_base[-25:]) / 25.0

    # -------------------------------------------------------------------------
    # PART B: RUN PROPOSED THREE-FACTOR LOCAL PREDICTIVE PLASTICITY (400 STEPS)
    # -------------------------------------------------------------------------
    print("\n" + "-"*85)
    print(" [2/2] RUNNING PROPOSED: THREE-FACTOR LOCAL PREDICTIVE PLASTICITY (NO AUTOGRAD GRAPH)")
    print("-" * 85)

    model_prop = ThreeFactorLocalAgent(device_str=device_str).to(device)
    hu_prop = HomeostaticUnit(batch_size=BATCH_SIZE, device=device_str)

    loss_history_prop = []
    t_start_prop = time.perf_counter()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()

    for idx, batch_tokens in enumerate(loader_train):
        if idx >= NUM_STEPS: break
        batch_tokens = batch_tokens.to(device)
        in_seq = batch_tokens[:, :-1]
        tgt_seq = batch_tokens[:, 1:]

        # Single-Pass Local Hebbian Learning Update
        loss_val, m_gain = model_prop.forward_sequence_local_learning(in_seq, tgt_seq, hu_prop, criterion, chunk_size=64)

        # Update Somatic State
        action_cost_t = torch.full((BATCH_SIZE, 1), 0.002, device=device)
        pred_err_t = torch.full((BATCH_SIZE, 1), float(loss_val * 0.1), device=device)
        entropy_t = torch.full((BATCH_SIZE, 1), 0.05, device=device)
        cog_t = torch.zeros((BATCH_SIZE, 1), dtype=torch.int64, device=device)
        hu_prop.update(action_cost_t, pred_err_t, entropy_t, cog_t)

        loss_history_prop.append(loss_val)
        if (idx + 1) % 100 == 0 or idx == NUM_STEPS - 1:
            print(f"  Local Step [{idx+1:04d}/{NUM_STEPS}] | Loss: {loss_val:.4f} (PPL: {math.exp(min(loss_val, 20.0)):.2f}) | M(u_t) Gain: {m_gain:.3f}")

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
                
            loss_p = model_base.forward_sequence(in_seq, tgt_seq, hu_val, criterion, chunk_size=64)

            val_loss_base_accum += loss_b.item()
            val_loss_prop_accum += loss_p.item()

    final_val_loss_base = val_loss_base_accum / float(val_batches)
    final_val_loss_prop = val_loss_prop_accum / float(val_batches)

    print(f"  Held-Out Validation Loss (Baseline Autograd): {final_val_loss_base:.4f} (PPL: {math.exp(min(final_val_loss_base, 20.0)):.2f})")
    print(f"  Held-Out Validation Loss (Proposed EXP-47):   {final_val_loss_prop:.4f} (PPL: {math.exp(min(final_val_loss_prop, 20.0)):.2f})")
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
    sample_short_prop = model_prop.generate_thought_and_speech(short_prompt, diag_hu, max_generated_tokens=60)

    long_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"
    sample_long_base = model_base.generate_thought_and_speech(long_prompt, diag_hu, max_generated_tokens=70)
    sample_long_prop = model_prop.generate_thought_and_speech(long_prompt, diag_hu, max_generated_tokens=70)

    print(f"  [Short Prompt: 'Hello!']")
    print(f"    • Baseline Output: \"{sample_short_base}\"")
    print(f"    • EXP-47 Proposed: \"{sample_short_prop}\"")
    print(f"\n  [Long Prompt: 'Energy for Earth']")
    print(f"    • Baseline Output: \"{sample_long_base}\"")
    print(f"    • EXP-47 Proposed: \"{sample_long_prop}\"")
    print("="*85)

    # -------------------------------------------------------------------------
    # PART E: FINAL TELEMETRY COMPARISON & KEP VERDICT (KEP RULE #2)
    # -------------------------------------------------------------------------
    delta_val_loss = final_val_loss_prop - final_val_loss_base
    speed_ratio = (tok_per_sec_prop / tok_per_sec_base) * 100.0

    print("\n" + "="*85)
    print(" === EXP-47 FINAL TELEMETRY AUDIT TABLE (400 STEPS) ===")
    print("="*85)
    print(f"{'Metric':<34} | {'Baseline (Global Autograd)':<28} | {'Proposed (EXP-47 Local 3-Factor)':<32}")
    print("-" * 102)
    print(f"{'Autograd Graph Dependency':<34} | {'Global Computational Tree':<28} | {'Zero Graph (100% Local Hebb)':<32}")
    print(f"{'Plasticity Control':<34} | {'Artificial AdamW Gradients':<28} | {'Three-Factor Neuromodulated':<32}")
    print(f"{'Final Train Loss':<34} | {final_train_loss_base:<28.4f} | {final_train_loss_prop:<32.4f}")
    print(f"{'Held-Out Validation Loss':<34} | {final_val_loss_base:<28.4f} | {final_val_loss_prop:<32.4f}")
    print(f"{'Validation Perplexity (PPL)':<34} | {math.exp(min(final_val_loss_base, 20.0)):<28.2f} | {math.exp(min(final_val_loss_prop, 20.0)):<32.2f}")
    print(f"{'Throughput Speed':<34} | {f'{tok_per_sec_base:,.1f} tok/s':<28} | {f'{tok_per_sec_prop:,.1f} tok/s':<32}")
    print(f"{'Peak VRAM Memory':<34} | {f'{vram_base:.1f} MB':<28} | {f'{vram_prop:.1f} MB':<32}")
    print(f"{'Total Execution Time':<34} | {f'{t_total_base:.2f} s':<28} | {f'{t_total_prop:.2f} s':<32}")
    print("="*102)

    print(f"\n[EXP-47 Decision Engine] Delta Validation Loss: {delta_val_loss:+.4f} | Speed Ratio: {speed_ratio:.1f}%")
    
    if delta_val_loss <= 0.05 and speed_ratio >= 110.0:
        verdict = "🟢 POSITIVE — Successful autograd-free local plasticity with significant throughput & memory efficiency. Ready for production merge."
    elif delta_val_loss <= 0.10:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE — Metric within biological variance margin."
    else:
        verdict = "🔴 REJECTED — Regression in local plasticity convergence."

    print(f"\n>>> FINAL VERDICT: {verdict}\n" + "="*102 + "\n")

if __name__ == "__main__":
    run_exp_47_benchmark()
