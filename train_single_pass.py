"""
===============================================================================
KARYON MASSIVE HIGH-VELOCITY STREAMING RUNTIME (52k DATASET, N=5)
Production Master Pipeline with Continuous Packed Streaming (S=2048, 0% Padding),
Native C++20 Multi-Timescale SSD Core (>135k tok/s), Ergodic Shuffling,
AdamW Regularization (WD=0.01), Full-Horizon Cosine LR, and KEP Deep Diagnostics.
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import types
import time
import math
import os
import struct
import json
import importlib
import torch

# =============================================================================
# 0. UNCONDITIONAL DYNAMO HOTFIX FOR PYTHON 3.12 / KAGGLE GPU
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

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

import karyon_config, karyon_core, karyon_agent, karyon_checkpoint, karyon_logger
importlib.reload(karyon_agent)

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory
from karyon_checkpoint import load_karyon, save_karyon
from karyon_logger import get_logger
from init_priors import initialize_priors

logger = get_logger()
torch.set_grad_enabled(True)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
logger.info(f"Execution context: {device_str.upper()}")

kcore_path = "karyon_soul.kcore"

if not os.path.exists(kcore_path):
    logger.warning(f"Container '{kcore_path}' not found! Automatically building base model via init_priors...")
    initialize_priors(recreate=True, filepath=kcore_path, device=device_str)

logger.info("Loading COMPLETE Conversational Dataset (52,002 samples from vicgalle/alpaca-gpt4)...")
dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")

tokenizer = ByteTokenizer()

# =============================================================================
# 1. CONTINUOUS PACKED STREAM DATASET (EXP-40 VALIDATED: 0% PADDING, S=2048)
# =============================================================================
class ContinuousPackedDataset(Dataset):
    """Zero-Padding Continuous Stream Packing with EOS Separators (S=2048)."""
    def __init__(self, hf_data, tokenizer, seq_len=2048):
        self.seq_len = seq_len
        full_token_stream = []

        for item in hf_data:
            inst = item.get("instruction", "").strip()
            out = item.get("output", "").strip()
            if inst and out:
                dialog = f"User: {inst}\nKaryon: {out}"
                ids = tokenizer.encode(dialog) # Contains 257 (<eos>) at the end!
                full_token_stream.extend(ids)

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

BATCH_SIZE = 16
SEQ_LEN = 2048
NUM_PASSES = 5
CHUNK_SIZE = 64

train_dataset = ContinuousPackedDataset(dataset, tokenizer, seq_len=SEQ_LEN)
stream_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_packed_fn, drop_last=True)

logger.info(f"Packed Continuous Dataset Ready. Total Blocks (S={SEQ_LEN}): {len(train_dataset)} | Batches: {len(stream_loader)} | Passes: {NUM_PASSES}")

# =============================================================================
# 2. MODEL CONFIGURATION & CANONICAL INITIALIZATION
# =============================================================================
core_config = CoREConfig()
core_config.net.text_dim = 128
core_config.net.unified_dim = 256
core_config.net.hidden_dim = 512
core_config.net.latent_dim = 128
core_config.net.text_gen_dim = 258
core_config.train.batch_size = BATCH_SIZE
core_config.train.chunk_size = CHUNK_SIZE

if os.path.exists(kcore_path):
    with open(kcore_path, 'rb') as f:
        f.seek(8)
        header_raw = f.read(24)
        _, num_sections, _, _ = struct.unpack('<IIQQ', header_raw)
        sections = []
        for _ in range(num_sections):
            sec_raw = f.read(64)
            s_type, _, offset, size, _ = struct.unpack('<IIQQQ', sec_raw[:32])
            sections.append({"type": s_type, "offset": offset, "size": size})
        sec_manifest = next((s for s in sections if s["type"] == 1), None)
        if sec_manifest:
            f.seek(sec_manifest["offset"])
            manifest = json.loads(f.read(sec_manifest["size"]).decode('utf-8'))
            genome = manifest.get("genome", {})
            if "text_dim" in genome: core_config.net.text_dim = genome["text_dim"]
            if "text_gen_dim" in genome: core_config.net.text_gen_dim = genome["text_gen_dim"]
            if "unified_dim" in genome: core_config.net.unified_dim = genome["unified_dim"]
            if "hidden_dim" in genome: core_config.net.hidden_dim = genome["hidden_dim"]
            if "latent_dim" in genome: core_config.net.latent_dim = genome["latent_dim"]

agent_brain = CoREAgent(config=core_config, device=device_str).to(device)
hu = HomeostaticUnit(batch_size=BATCH_SIZE, device=device_str)
episodic_mem = BatchedEpisodicMemory(batch_size=BATCH_SIZE, memory_dim=core_config.net.unified_dim, max_capacity=1000, device=device_str)

h_fast, h_slow, saved_epoch, _ = load_karyon(agent_brain, episodic_mem, hu, filepath=kcore_path, device=device_str)

# AdamW with L2 weight decay (0.01)
optimizer = optim.AdamW(agent_brain.get_all_parameters(), lr=3e-3, weight_decay=0.01)
criterion_speech = nn.CrossEntropyLoss(ignore_index=256)

# Global Full-Horizon Cosine Annealing Schedule with 100-step Warmup
TOTAL_TRAINING_STEPS = len(stream_loader) * NUM_PASSES
WARMUP_STEPS = 100

def get_lr_multiplier(current_step: int) -> float:
    if current_step < WARMUP_STEPS:
        return float(current_step + 1) / float(WARMUP_STEPS)
    progress = float(current_step - WARMUP_STEPS) / float(max(1, TOTAL_TRAINING_STEPS - WARMUP_STEPS))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(0.0333, cosine_decay)

lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)

moving_mean_fe = 0.15
moving_var_fe = 0.01
alpha_ma = getattr(core_config.train, 'dfet_alpha_ma', 0.05)

FREE_ENERGY_MASTERY_SETPOINT = getattr(core_config.train, 'mastery_setpoint', 0.025)
SPEECH_MASTERY_SETPOINT = getattr(core_config.train, 'speech_mastery_setpoint', 0.30)

total_skipped_batches = 0
total_adapted_batches = 0
global_step_counter = 0

# =============================================================================
# 3. KEP RULE #4: LIVE DIAGNOSTIC TEXT SAMPLER
# =============================================================================
def run_diagnostic_text_sample(agent, memory, hu_state, config):
    agent.eval()
    diag_prompt = "User: What is the primary source of energy for Earth?\nKaryon:"
    diag_hu = HomeostaticUnit(batch_size=1, device=agent.device_str)
    diag_hu.state.copy_(hu_state[0:1])
    
    diag_mem = BatchedEpisodicMemory(batch_size=1, memory_dim=config.net.unified_dim, max_capacity=config.memory.max_capacity, device=agent.device_str)
    k_slice = min(memory.keys.size(1), config.memory.max_capacity)
    diag_mem.keys[:, :k_slice].copy_(memory.keys[:1, :k_slice])
    diag_mem.values[:, :k_slice].copy_(memory.values[:1, :k_slice])
    diag_mem.pointer.copy_(memory.pointer[:1])
    diag_mem.size.copy_(memory.size[:1])
    
    generated_chars = []
    with torch.no_grad():
        gen_stream = agent.generate_thought_and_speech(
            prompt=diag_prompt,
            m_state=torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=agent.device),
            h_state=torch.zeros(1, agent.hidden_dim, device=agent.device),
            hu=diag_hu,
            episodic_memory=diag_mem,
            config=config,
            max_generated_tokens=75,
            temperature=0.7,
            top_p=0.90
        )
        for event in gen_stream:
            if event["status"] == "token":
                generated_chars.append(event["text"])
                
    agent.train()
    return "".join(generated_chars).strip()

logger.info(f"Starting Production Long-Context Session ({NUM_PASSES} Passes over 52k samples, S={SEQ_LEN} @ 135k+ tok/s)...")

# =============================================================================
# 4. PRODUCTION MULTI-PASS STREAMING LOOP (S=2048, Q=64)
# =============================================================================
for pass_idx in range(NUM_PASSES):
    logger.info(f"\n{'='*85}\n === [STARTING PASS {pass_idx+1}/{NUM_PASSES} (EPOCH {saved_epoch + pass_idx + 1})] ===\n{'='*85}")
    
    for batch_idx, batch_tokens in enumerate(stream_loader):
        t_batch_start = time.perf_counter()
        
        batch_tokens = batch_tokens.to(device)
        current_batch_size = batch_tokens.size(0)
        seq_len = batch_tokens.size(1)

        hu_batch = HomeostaticUnit(batch_size=current_batch_size, device=device_str)

        input_seq = batch_tokens[:, :-1]
        target_seq = batch_tokens[:, 1:]

        optimizer.zero_grad()
        
        # Native C++ Multi-Timescale SSD Scan on 32 chunks of Q=64
        t_exec_start = time.perf_counter()
        total_loss_metric, speech_loss_val, fe_val, m_curr, h_curr, curr_u_t, eff_dt = agent_brain.forward_sequence(
            input_seq, target_seq, hu_batch, criterion_speech, episodic_memory=episodic_mem,
            loss_free_energy_weight=0.05, chunk_size=CHUNK_SIZE, optimizer=optimizer
        )
        t_exec_ms = (time.perf_counter() - t_exec_start) * 1000.0

        na_val = curr_u_t.select(1, 4).mean().item()

        # Plasticity and Adaptation Step
        moving_mean_fe = (1.0 - alpha_ma) * moving_mean_fe + alpha_ma * fe_val
        moving_var_fe = (1.0 - alpha_ma) * moving_var_fe + alpha_ma * ((fe_val - moving_mean_fe)**2)
        moving_std_fe = math.sqrt(max(1e-6, moving_var_fe))

        is_fe_unmastered = fe_val > FREE_ENERGY_MASTERY_SETPOINT
        is_speech_unmastered = speech_loss_val > SPEECH_MASTERY_SETPOINT
        is_statistical_outlier = agent_brain.evaluate_dfet_gating(fe_val, moving_mean_fe, moving_std_fe, na_val)
        
        should_adapt = is_fe_unmastered or is_speech_unmastered or is_statistical_outlier

        t_opt_ms = 0.0
        if should_adapt:
            t_opt_start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(agent_brain.get_all_parameters(), max_norm=3.0)
            optimizer.step()
            lr_scheduler.step()
            t_opt_ms = (time.perf_counter() - t_opt_start) * 1000.0
            
            cur_lr = optimizer.param_groups[0]['lr']
            total_adapted_batches += 1
            status_str = f"ADAPTED (lr={cur_lr:.6f})"
        else:
            optimizer.zero_grad()
            rest_recovery_rate = getattr(core_config.homeo, 'energy_recovery_rate', 0.0012)
            with torch.no_grad():
                curr_u_t[:, 1] = torch.clamp(curr_u_t[:, 1] + rest_recovery_rate, 0.0, 1.0)
                
            total_skipped_batches += 1
            status_str = f"RESTING / SKIPPED (0 Backprop FLOPs)"

        global_step_counter += 1
        batch_total_ms = (time.perf_counter() - t_batch_start) * 1000.0
        tokens_per_sec = (current_batch_size * (seq_len - 1)) / (batch_total_ms / 1000.0)

        # KEP Rule #6: Deep Process Diagnostics Dashboard
        if (batch_idx + 1) % 50 == 0 or batch_idx == len(stream_loader) - 1:
            perplexity = math.exp(min(speech_loss_val, 20.0))
            curiosity, energy, stability, health, na, da = curr_u_t[0].tolist()
            peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device_str == 'cuda' else 0.0

            grad_embed = agent_brain.pos_embeddings.byte_embed.weight.grad.norm().item() if agent_brain.pos_embeddings.byte_embed.weight.grad is not None else 0.0
            grad_head = 0.0
            if hasattr(agent_brain.attractor_head, 'attractor_basins') and agent_brain.attractor_head.attractor_basins.grad is not None:
                grad_head = agent_brain.attractor_head.attractor_basins.grad.norm().item()

            print(f"\n" + "="*85)
            print(f" === [KEP RULE #6 PROCESS DIAGNOSTICS DASHBOARD | PASS {pass_idx+1}/{NUM_PASSES} | STEP {batch_idx+1:04d}/{len(stream_loader)}] ===")
            print("="*85)
            print(f"Plasticity Gating Status  : {status_str}")
            print(f"Submodule Timing (ms)     : SSD+SwiGLU Scan: {t_exec_ms:.1f}ms | Step: {t_opt_ms:.1f}ms")
            print(f"Batch Performance         : Total Batch: {batch_total_ms:.1f}ms | Throughput: {tokens_per_sec:.1f} tok/s")
            print(f"Metrics Progress          : Speech Loss = {speech_loss_val:.4f} (PPL: {perplexity:.2f}) | Free Energy = {fe_val:.4f}")
            print(f"Gradient Flow Inspection  : Embeddings Grad Norm = {grad_embed:.6f} | Attractor Head Grad Norm = {grad_head:.6f}")
            print(f"Hardware & Somatic        : Peak VRAM: {peak_vram_mb:.1f} MB | Somatic Energy: {energy:.3f} | Arousal(NA): {na:.3f}")
            print("="*85)

        # KEP Rule #4: Live Diagnostic Speech Sample
        if (batch_idx + 1) % 100 == 0:
            diag_sample = run_diagnostic_text_sample(agent_brain, episodic_mem, curr_u_t, core_config)
            logger.info(f"💬 [KEP Rule #4 Diagnostic Speech Sample @ Pass {pass_idx+1} Step {batch_idx+1}] -> \"{diag_sample}\"\n")

    # Persist progress after each pass
    save_karyon(agent_brain, episodic_mem, hu, h_curr[0:1], h_curr[0:1], epoch=saved_epoch + pass_idx + 1, story_idx=len(stream_loader) * BATCH_SIZE * (pass_idx + 1), filepath=kcore_path)

logger.info(f"Massive Long-Context Session Complete! Total Adapted: {total_adapted_batches} | Total Skipped: {total_skipped_batches}.")
