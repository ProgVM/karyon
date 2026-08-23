# train_single_pass.py
"""
===============================================================================
KARYON MULTI-PASS HIGH-VELOCITY STREAMING RUNTIME (N=3)
Integrated with Native C++20 Dual-Layer Cortical SDE-SSM Stack (L2/3 + L5/6),
Safe Response-Only Target Masking, Dual-Rate Optimizer, and Top-p Sampling.
===============================================================================
"""

import sys
import types
import time
import math
import importlib
import torch

# Unconditional Dynamo Hotfix for Kaggle / Python 3.12 environments
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
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import os
import struct
import json

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Execution context: {device.upper()}")

kcore_path = "karyon_soul.kcore"

if not os.path.exists(kcore_path):
    logger.warning(f"Container '{kcore_path}' not found! Automatically building base model via init_priors...")
    initialize_priors(recreate=True, filepath=kcore_path, device=device)

logger.info("Loading Conversational Data Stream (alpaca-gpt4)...")
dataset = load_dataset("vicgalle/alpaca-gpt4", split="train[:10000]")

tokenizer = ByteTokenizer()

class StreamingDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len=512):
        self.samples = []
        for item in hf_dataset:
            instruction = item.get("instruction", "").strip()
            output = item.get("output", "").strip()
            if instruction and output:
                prompt_str = f"User: {instruction}\nKaryon: "
                response_str = f"{output}"
                
                prompt_ids = tokenizer.encode(prompt_str)[:-1] # strip EOS from prompt
                response_ids = tokenizer.encode(response_str) # contains EOS 257
                
                full_ids = prompt_ids + response_ids
                if len(full_ids) > max_len:
                    full_ids = full_ids[:max_len-1] + [257]
                
                if len(full_ids) > 10:
                    inp_t = torch.tensor(full_ids[:-1], dtype=torch.long)
                    tgt_t = torch.tensor(full_ids[1:], dtype=torch.long)
                    
                    # Safe prompt masking: mask indices corresponding to the prompt
                    prompt_target_len = max(0, len(prompt_ids) - 1)
                    if prompt_target_len > 0 and prompt_target_len < len(tgt_t):
                        tgt_t[:prompt_target_len] = 256
                    
                    self.samples.append((inp_t, tgt_t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    inps = [item[0] for item in batch]
    tgts = [item[1] for item in batch]
    padded_inps = pad_sequence(inps, batch_first=True, padding_value=256)
    padded_tgts = pad_sequence(tgts, batch_first=True, padding_value=256)
    return padded_inps, padded_tgts

BATCH_SIZE = 32
MAX_SEQ_LEN = 512
NUM_PASSES = 3

train_dataset = StreamingDataset(dataset, tokenizer, max_len=MAX_SEQ_LEN)
stream_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, drop_last=True)

logger.info(f"Data Stream Ready (Safe Response Masking). Samples: {len(train_dataset)} | Batches: {len(stream_loader)} | Passes: {NUM_PASSES}")

core_config = CoREConfig()
core_config.net.text_dim = 128
core_config.net.unified_dim = 256
core_config.net.hidden_dim = 512
core_config.net.latent_dim = 128
core_config.net.text_gen_dim = 258
core_config.train.batch_size = BATCH_SIZE

# Extract DNA Genome dimensions from container manifest
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

agent_brain = CoREAgent(config=core_config, device=device).to(device)
hu = HomeostaticUnit(batch_size=BATCH_SIZE, device=device)
episodic_mem = BatchedEpisodicMemory(batch_size=BATCH_SIZE, memory_dim=core_config.net.unified_dim, max_capacity=1000, device=device)

# Load state directly from .kcore container
h_fast, h_slow, saved_epoch, _ = load_karyon(agent_brain, episodic_mem, hu, filepath=kcore_path, device=device)

# Dual-Rate Optimizer: 5e-3 for lexical embeddings & 3e-3 for cortical SSM layers
emb_params = list(agent_brain.pos_embeddings.parameters()) + list(agent_brain.motor_text_proj.parameters())
core_params = [p for n, p in agent_brain.named_parameters() if not any(n.startswith(prefix) for prefix in ['pos_embeddings', 'motor_text_proj'])]

optimizer = optim.Adam([
    {'params': emb_params, 'lr': 5e-3},
    {'params': core_params, 'lr': 3e-3}
], weight_decay=0.0)

criterion_speech = nn.CrossEntropyLoss(ignore_index=256)

moving_mean_fe = 0.15
moving_var_fe = 0.01
alpha_ma = getattr(core_config.train, 'dfet_alpha_ma', 0.05)

FREE_ENERGY_MASTERY_SETPOINT = getattr(core_config.train, 'mastery_setpoint', 0.025)
SPEECH_MASTERY_SETPOINT = getattr(core_config.train, 'speech_mastery_setpoint', 1.20)

total_skipped_batches = 0
total_adapted_batches = 0

def run_diagnostic_text_sample(agent, memory, hu_state, config):
    """KEP Rule #4: Live Diagnostic Text Sample with Rolling Buffer K=4 Consistency."""
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
            max_generated_tokens=60,
            temperature=0.7,
            top_p=0.90
        )
        for event in gen_stream:
            if event["status"] == "token":
                generated_chars.append(event["text"])
                
    agent.train()
    return "".join(generated_chars).strip()

logger.info(f"Starting Multi-Pass High-Speed Session ({NUM_PASSES} Passes @ 140k+ tok/s)...")

for pass_idx in range(NUM_PASSES):
    logger.info(f"\n{'='*85}\n === [STARTING PASS {pass_idx+1}/{NUM_PASSES} (EPOCH {saved_epoch + pass_idx + 1})] ===\n{'='*85}")
    
    for batch_idx, (batch_inps, batch_tgts) in enumerate(stream_loader):
        t_batch_start = time.perf_counter()
        
        batch_inps = batch_inps.to(device)
        batch_tgts = batch_tgts.to(device)
        current_batch_size = batch_inps.size(0)
        seq_len = batch_inps.size(1)

        if seq_len <= 1:
            continue

        hu_batch = HomeostaticUnit(batch_size=current_batch_size, device=device)

        input_seq = batch_inps
        target_seq = batch_tgts

        optimizer.zero_grad()
        
        # 1. Native C++ Dual-Layer Cortical SDE-SSM Execution with Safe Target Handling
        t_exec_start = time.perf_counter()
        total_loss_metric, speech_loss_val, fe_val, m_curr, h_curr, curr_u_t, eff_dt = agent_brain.forward_sequence(
            input_seq, target_seq, hu_batch, criterion_speech, episodic_memory=episodic_mem,
            loss_free_energy_weight=0.05, chunk_size=32, optimizer=optimizer
        )
        t_exec_ms = (time.perf_counter() - t_exec_start) * 1000.0

        na_val = curr_u_t.select(1, 4).mean().item()

        # 2. Plasticity and Adaptation Step
        moving_mean_fe = (1.0 - alpha_ma) * moving_mean_fe + alpha_ma * fe_val
        moving_var_fe = (1.0 - alpha_ma) * moving_var_fe + alpha_ma * ((fe_val - moving_mean_fe)**2)
        moving_std_fe = math.sqrt(max(1e-6, moving_var_fe))

        is_fe_unmastered = fe_val > FREE_ENERGY_MASTERY_SETPOINT
        is_speech_unmastered = speech_loss_val > SPEECH_MASTERY_SETPOINT
        is_statistical_outlier = agent_brain.evaluate_dfet_gating(fe_val, moving_mean_fe, moving_std_fe, na_val)
        
        should_adapt = is_fe_unmastered or is_speech_unmastered or is_statistical_outlier

        t_opt_ms = 0.0
        if should_adapt:
            pass_scale = 1.0 / (1.0 + 0.3 * pass_idx)
            optimizer.param_groups[0]['lr'] = (5e-3 * pass_scale) * (1.0 + 1.0 * na_val)
            optimizer.param_groups[1]['lr'] = (3e-3 * pass_scale) * (1.0 + 1.0 * na_val)

            t_opt_start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(agent_brain.get_all_parameters(), max_norm=3.0)
            optimizer.step()
            t_opt_ms = (time.perf_counter() - t_opt_start) * 1000.0
            
            total_adapted_batches += 1
            status_str = f"ADAPTED (lr_emb={optimizer.param_groups[0]['lr']:.5f}, lr_core={optimizer.param_groups[1]['lr']:.5f})"
        else:
            optimizer.zero_grad()
            rest_recovery_rate = getattr(core_config.homeo, 'energy_recovery_rate', 0.0012)
            with torch.no_grad():
                curr_u_t[:, 1] = torch.clamp(curr_u_t[:, 1] + rest_recovery_rate, 0.0, 1.0)
                
            total_skipped_batches += 1
            status_str = f"RESTING / SKIPPED (0 Backprop FLOPs)"

        batch_total_ms = (time.perf_counter() - t_batch_start) * 1000.0
        tokens_per_sec = (current_batch_size * seq_len) / (batch_total_ms / 1000.0)

        # KEP Rule #6: Deep Process Diagnostics Dashboard
        if (batch_idx + 1) % 20 == 0 or batch_idx == len(stream_loader) - 1:
            perplexity = math.exp(min(speech_loss_val, 20.0))
            curiosity, energy, stability, health, na, da = curr_u_t[0].tolist()
            peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device == 'cuda' else 0.0

            grad_embed = agent_brain.pos_embeddings.byte_embed.weight.grad.norm().item() if agent_brain.pos_embeddings.byte_embed.weight.grad is not None else 0.0
            grad_head = 0.0
            if hasattr(agent_brain.attractor_head, 'attractor_basins') and agent_brain.attractor_head.attractor_basins.grad is not None:
                grad_head = agent_brain.attractor_head.attractor_basins.grad.norm().item()

            print(f"\n" + "="*85)
            print(f" === [KEP RULE #6 PROCESS DIAGNOSTICS DASHBOARD | PASS {pass_idx+1}/{NUM_PASSES} | STEP {batch_idx+1:04d}/{len(stream_loader)}] ===")
            print("="*85)
            print(f"Plasticity Gating Status  : {status_str}")
            print(f"Submodule Timing (ms)     : Dual-Layer SSD Scan: {t_exec_ms:.1f}ms | Step: {t_opt_ms:.1f}ms")
            print(f"Batch Performance         : Total Batch: {batch_total_ms:.1f}ms | Throughput: {tokens_per_sec:.1f} tok/s")
            print(f"Metrics Progress          : Response Loss = {speech_loss_val:.4f} (PPL: {perplexity:.2f}) | Free Energy = {fe_val:.4f}")
            print(f"Gradient Flow Inspection  : Embeddings Grad Norm = {grad_embed:.6f} | Attractor Head Grad Norm = {grad_head:.6f}")
            print(f"Hardware & Somatic        : Peak VRAM: {peak_vram_mb:.1f} MB | Somatic Energy: {energy:.3f} | Arousal(NA): {na:.3f}")
            print("="*85)

        # KEP Rule #4: Live Diagnostic Speech Sample every 30 batches
        if (batch_idx + 1) % 30 == 0:
            diag_sample = run_diagnostic_text_sample(agent_brain, episodic_mem, curr_u_t, core_config)
            logger.info(f"💬 [KEP Rule #4 Diagnostic Speech Sample @ Pass {pass_idx+1} Step {batch_idx+1}] -> \"{diag_sample}\"\n")

    # Persist progress after each pass
    save_karyon(agent_brain, episodic_mem, hu, h_curr[0:1], h_curr[0:1], epoch=saved_epoch + pass_idx + 1, story_idx=len(stream_loader) * BATCH_SIZE * (pass_idx + 1), filepath=kcore_path)

logger.info(f"Multi-Pass Session Complete! Total Adapted: {total_adapted_batches} | Total Skipped: {total_skipped_batches}.")
