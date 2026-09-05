# train_single_pass.py
"""
===============================================================================
KARYON SINGLE-PASS CONTINUOUS ALLOSTATIC LEARNING RUNTIME (v32.0 MASTER)
===============================================================================
Grounded in KEP Principles & Biological AGI Reality:
- Single Continuous Stream Pass (N=1 Pass, Zero Artificial Epochs):
  Experience flows continuously as a single unbroken stream of reality.
- Dynamic Allostatic Volitional Sleep 2.0 (Biophysical Sleep & SHY Consolidation):
  Instead of artificial epoch boundaries, Karyon monitors its own somatic energy
  and allostatic strain. When energy drops below 0.35 or when the native C++20
  `VolitionalActionEvaluator` triggers `INITIATE_SLEEP_CONSOLIDATION`, Karyon enters
  Phase 1 NREM Hippocampal Replay + Phase 2 REM Synthetic Dreaming + Synaptic Pruning,
  restores somatic energy to 1.00, and awakens to continue the stream!
- Error-Gated Neuromodulated Plasticity (DFET Gating):
  Backprop + Local Neuromodulated Fast-Weights adapt on high-surprise data;
  mastered data skips FLOPs to save metabolic energy.
- Full C++20 LibTorch Acceleration (18 Native Cognitive Modules).

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
import gc
import numpy as np
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
from huggingface_hub import HfApi

import karyon_config, karyon_core, karyon_agent, karyon_checkpoint, karyon_logger
importlib.reload(karyon_agent)

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory
from karyon_checkpoint import load_karyon, save_karyon
from karyon_logger import get_logger
from karyon_hardware import get_hardware_engine
from init_priors import initialize_priors

logger = get_logger()
torch.set_grad_enabled(True)

hw_engine = get_hardware_engine()
device = hw_engine.device
device_str = str(device)
use_amp = hw_engine.config.enable_amp and not hw_engine.is_cpu
autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
logger.info(f"Execution context: {device_str.upper()} (AMP Enabled: {use_amp}, Dtype: {autocast_dtype})")

kcore_path = "karyon_soul.kcore"
hf_repo_id = "progvmoff/karyon-v31-core"

# Function to safely push checkpoint to Hugging Face Hub
def sync_checkpoint_to_hf(local_file: str, repo_id: str, commit_msg: str):
    try:
        if not os.path.exists(local_file):
            return
        api = HfApi()
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=local_file,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg
        )
        logger.info(f"🤗 [HF Auto-Sync] Successfully uploaded '{local_file}' to '{repo_id}'!")
    except Exception as e:
        logger.warning(f"⚠️ [HF Auto-Sync Notice] Could not push to HF ({e}). Local checkpoint remains safe.")

# Ensure valid kcore container exists
if os.path.exists(kcore_path):
    with open(kcore_path, 'rb') as f:
        f.seek(8)
        header_raw = f.read(24)
        _, num_sections, _, _ = struct.unpack('<IIQQ', header_raw)
        sections = []
        for _ in range(num_sections):
            sec_raw = f.read(64)
            s_type, flags, offset, size, _ = struct.unpack('<IIQQQ', sec_raw[:32])
            sections.append({"type": s_type, "flags": flags, "offset": offset, "size": size})
        sec_manifest = next((s for s in sections if s["type"] == 1), None)
        if sec_manifest:
            f.seek(sec_manifest["offset"])
            manifest_raw = f.read(sec_manifest["size"])
            if sec_manifest["flags"] & 0x01: # FLAG_ZLIB_COMPRESSED
                import zlib
                manifest_raw = zlib.decompress(manifest_raw)
            manifest = json.loads(manifest_raw.decode('utf-8'))
            genome = manifest.get("genome", {})
            if genome.get("text_dim", 128) != 256:
                logger.warning(f"Detected legacy DNA (text_dim={genome.get('text_dim')}). Rebuilding container for Unshackled Flow 256D...")
                initialize_priors(recreate=True, filepath=kcore_path, device=device_str)
else:
    logger.warning(f"Container '{kcore_path}' not found! Automatically building base model via init_priors...")
    initialize_priors(recreate=True, filepath=kcore_path, device=device_str)

logger.info("Assembling Rich Multi-Domain Continuous Stream Dataset (Alpaca, Dolly, Code, GSM8k)...")

def build_multidomain_packed_stream(seq_len=1024):
    """Zero-Padding Continuous Stream Packing across 4 Diverse Cognitive Domains."""
    eos_arr = np.array([257], dtype=np.uint16)
    byte_chunks = []

    # Domain 1: General Conversation & World Knowledge (Alpaca-GPT4)
    logger.info(" -> Ingesting Domain 1: vicgalle/alpaca-gpt4 (General Dialogue)...")
    try:
        ds_alpaca = load_dataset("vicgalle/alpaca-gpt4", split="train")
        for item in ds_alpaca:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if inst and out:
                full_in = f"{inst}\nContext: {inp}" if inp else inst
                dialog = f"User: {full_in}\nKaryon: {out}"
                raw_b = dialog.encode('utf-8')
                byte_chunks.append(np.frombuffer(raw_b, dtype=np.uint8).astype(np.uint16))
                byte_chunks.append(eos_arr)
        del ds_alpaca
    except Exception as e:
        logger.warning(f"Notice loading Alpaca dataset: {e}")

    # Domain 2: Multi-Task Instructions (Databricks-Dolly-15k)
    logger.info(" -> Ingesting Domain 2: databricks/databricks-dolly-15k (Instructions & QA)...")
    try:
        ds_dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        for item in ds_dolly:
            inst = item.get("instruction", "").strip()
            ctx = item.get("context", "").strip()
            resp = item.get("response", "").strip()
            if inst and resp:
                full_in = f"{inst}\nContext: {ctx}" if ctx else inst
                dialog = f"User: {full_in}\nKaryon: {resp}"
                raw_b = dialog.encode('utf-8')
                byte_chunks.append(np.frombuffer(raw_b, dtype=np.uint8).astype(np.uint16))
                byte_chunks.append(eos_arr)
        del ds_dolly
    except Exception as e:
        logger.warning(f"Notice loading Dolly dataset: {e}")

    # Domain 3: Algorithmic & Code Logic (Python Code Instructions 18k)
    logger.info(" -> Ingesting Domain 3: iamtarun/python_code_instructions_18k_alpaca (Code Logic)...")
    try:
        ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        for item in ds_code:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if inst and out:
                full_in = f"{inst}\n{inp}" if inp else inst
                dialog = f"User: {full_in}\nKaryon: {out}"
                raw_b = dialog.encode('utf-8')
                byte_chunks.append(np.frombuffer(raw_b, dtype=np.uint8).astype(np.uint16))
                byte_chunks.append(eos_arr)
        del ds_code
    except Exception as e:
        logger.warning(f"Notice loading Code dataset: {e}")

    # Domain 4: Deductive & Mathematical Reasoning (GSM8k)
    logger.info(" -> Ingesting Domain 4: gsm8k (Step-by-Step Chain-of-Thought)...")
    try:
        ds_gsm = load_dataset("gsm8k", "main", split="train")
        for item in ds_gsm:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q and a:
                dialog = f"User: Solve step-by-step: {q}\nKaryon: {a}"
                raw_b = dialog.encode('utf-8')
                byte_chunks.append(np.frombuffer(raw_b, dtype=np.uint8).astype(np.uint16))
                byte_chunks.append(eos_arr)
        del ds_gsm
    except Exception as e:
        logger.warning(f"Notice loading GSM8k dataset: {e}")

    flat_stream = np.concatenate(byte_chunks)
    del byte_chunks
    gc.collect()
    return flat_stream

# =============================================================================
# 1. CONTINUOUS PACKED STREAM DATASET (0% PADDING, S=1024) - SINGLE PASS
# =============================================================================
class ContinuousPackedDataset(Dataset):
    """Zero-Padding Continuous Stream Packing with EOS Separators (S=1024) - Single Pass."""
    def __init__(self, flat_stream, seq_len=1024):
        self.seq_len = seq_len
        self.flat_stream = flat_stream
        self.num_blocks = len(self.flat_stream) // (seq_len + 1)

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        end = start + (self.seq_len + 1)
        return torch.from_numpy(self.flat_stream[start:end].astype(np.int64))

def collate_packed_fn(batch):
    return torch.stack(batch, dim=0)

BATCH_SIZE = 16
SEQ_LEN = 1024
CHUNK_SIZE = 64

flat_stream = build_multidomain_packed_stream(seq_len=SEQ_LEN)
train_dataset = ContinuousPackedDataset(flat_stream, seq_len=SEQ_LEN)

stream_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, # Sequential continuous stream flow (Single Pass)
    collate_fn=collate_packed_fn, 
    drop_last=True,
    num_workers=2,
    persistent_workers=True,
    pin_memory=hw_engine.is_cuda
)

logger.info(f"Single-Pass Continuous Stream Dataset Ready. Total Stream Blocks (S={SEQ_LEN}): {len(train_dataset)} | Stream Batches: {len(stream_loader)} (B={BATCH_SIZE})")

# =============================================================================
# 2. MODEL CONFIGURATION & INITIALIZATION
# =============================================================================
core_config = CoREConfig()
core_config.net.text_dim = 256
core_config.net.unified_dim = 256
core_config.net.hidden_dim = 768
core_config.net.expand_dim = 3072
core_config.net.num_heads = 12
core_config.net.latent_dim = 128
core_config.net.num_attractors = 256
core_config.net.text_gen_dim = 258
core_config.train.batch_size = BATCH_SIZE
core_config.train.chunk_size = CHUNK_SIZE

core_config.train.mastery_setpoint = 0.001
core_config.train.speech_mastery_setpoint = 0.05

agent_brain = CoREAgent(config=core_config, device=device_str).to(device)
hu = HomeostaticUnit(batch_size=BATCH_SIZE, device=device_str)
episodic_mem = BatchedEpisodicMemory(batch_size=BATCH_SIZE, memory_dim=core_config.net.unified_dim, max_capacity=1000, device=device_str)

h_fast, h_slow, saved_epoch, _ = load_karyon(agent_brain, episodic_mem, hu, filepath=kcore_path, device=device_str)

optimizer = optim.AdamW(agent_brain.get_all_parameters(), lr=5e-4, weight_decay=0.01)
criterion_speech = nn.CrossEntropyLoss(ignore_index=256)

scaler = torch.amp.GradScaler(hw_engine.device_type, enabled=(use_amp and autocast_dtype == torch.float16))

TOTAL_TRAINING_STEPS = len(stream_loader)
WARMUP_STEPS = 50

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

FREE_ENERGY_MASTERY_SETPOINT = getattr(core_config.train, 'mastery_setpoint', 0.001)
SPEECH_MASTERY_SETPOINT = getattr(core_config.train, 'speech_mastery_setpoint', 0.05)

total_skipped_batches = 0
total_adapted_batches = 0
total_sleep_cycles = 0

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
            temperature=0.45,
            top_p=0.90
        )
        for event in gen_stream:
            if event["status"] == "token":
                generated_chars.append(event["text"])
                
    agent.train()
    return "".join(generated_chars).strip()

logger.info(f"Starting Single-Pass Allostatic Session (1 Continuous Stream Pass, B={BATCH_SIZE}, S={SEQ_LEN}, 32,768 tokens/step)...")

# =============================================================================
# 4. SINGLE-PASS CONTINUOUS ALLOSTATIC STREAMING LOOP
# =============================================================================
def run_single_pass_training():
    global total_adapted_batches, total_skipped_batches, total_sleep_cycles, moving_mean_fe, moving_var_fe, h_fast, h_slow
    
    logger.info(f"\n{'='*85}\n === [STARTING SINGLE-PASS CONTINUOUS STREAM LEARNING (N=1 PASS)] ===\n{'='*85}")
    
    for batch_idx, batch_tokens in enumerate(stream_loader):
        t_batch_start = time.perf_counter()
        
        batch_tokens = batch_tokens.to(device, non_blocking=(device_str == 'cuda'))
        current_batch_size = batch_tokens.size(0)
        seq_len = batch_tokens.size(1)

        input_seq = batch_tokens[:, :-1]
        target_seq = batch_tokens[:, 1:]

        optimizer.zero_grad()
        
        t_exec_start = time.perf_counter()
        with torch.amp.autocast(device_type=device_str, dtype=autocast_dtype, enabled=use_amp):
            total_loss_tensor, speech_loss_val, fe_val, m_curr, h_curr, curr_u_t, eff_dt = agent_brain.forward_sequence(
                input_seq, target_seq, hu, criterion_speech, episodic_memory=episodic_mem,
                loss_free_energy_weight=0.05, chunk_size=CHUNK_SIZE, use_checkpointing=False
            )
        t_exec_ms = (time.perf_counter() - t_exec_start) * 1000.0

        if math.isnan(speech_loss_val) or math.isnan(fe_val) or torch.isnan(total_loss_tensor).any():
            logger.warning(f"⚠️ [Step {batch_idx+1}] Loss or Free Energy is NaN. Skipping backward step & resetting somatic state.")
            optimizer.zero_grad()
            if torch.isnan(hu.state).any():
                hu.state = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat((current_batch_size, 1))
            continue

        # Update Somatic Homeostasis (Metabolic expenditure)
        action_cost_tensor = torch.full((current_batch_size, 1), 0.003, device=device) # Metabolic cost
        pred_err_tensor = torch.full((current_batch_size, 1), float(speech_loss_val * 0.1), device=device)
        entropy_tensor = torch.full((current_batch_size, 1), float(fe_val), device=device)
        cog_act_tensor = torch.zeros((current_batch_size, 1), dtype=torch.int64, device=device)
        hu.update(action_cost_tensor, pred_err_tensor, entropy_tensor, cog_act_tensor)

        na_val = hu.state.select(1, 4).mean().item()
        curiosity_val = hu.state[0, 0].item()
        energy_val = hu.state[0, 1].item()

        # Dynamic Volitional Sleep 2.0 Trigger
        action_idx = agent_brain.efe_action_evaluator.select_volitional_action(h_curr[0:1], curiosity_val, energy_val)
        should_sleep = (energy_val <= 0.20) or (energy_val <= 0.35 and action_idx == 2)

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
            scaler.scale(total_loss_tensor).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(agent_brain.get_all_parameters(), max_norm=3.0)
            
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            
            if scale_before <= scale_after:
                lr_scheduler.step()
                
            t_opt_ms = (time.perf_counter() - t_opt_start) * 1000.0
            
            cur_lr = optimizer.param_groups[0]['lr']
            total_adapted_batches += 1
            status_str = f"ADAPTED (lr={cur_lr:.6f})"
        else:
            optimizer.zero_grad()
            rest_recovery_rate = getattr(core_config.homeo, 'energy_recovery_rate', 0.0012)
            with torch.no_grad():
                hu.state[:, 1] = torch.clamp(hu.state[:, 1] + rest_recovery_rate, 0.0, 1.0)
                
            total_skipped_batches += 1
            status_str = f"RESTING / SKIPPED (0 Backprop FLOPs)"

        if should_sleep:
            total_sleep_cycles += 1
            t_sleep_start = time.perf_counter()
            logger.info(f"🌙 [Step {batch_idx+1}] Somatic Energy={energy_val:.2f} | C++20 EFE Volition Action={action_idx}. Entering Biophysical Sleep 2.0...")
            pruned_weights = agent_brain.execute_deep_allostatic_sleep(episodic_mem, hu, num_replay_cycles=3, downscaling_factor=0.03)
            sleep_duration_ms = (time.perf_counter() - t_sleep_start) * 1000.0
            logger.info(f"☀️ [Awakened @ Step {batch_idx+1}] Sleep 2.0 Complete ({sleep_duration_ms:.1f}ms). Restored Energy={hu.state[0, 1].item():.2f} | Pruned Weights={pruned_weights}")

        batch_total_ms = (time.perf_counter() - t_batch_start) * 1000.0
        tokens_per_sec = (current_batch_size * (seq_len - 1)) / (batch_total_ms / 1000.0)

        if (batch_idx + 1) % 25 == 0 or batch_idx == len(stream_loader) - 1:
            perplexity = math.exp(min(speech_loss_val, 20.0))
            curiosity, energy, stability, health, na, da = hu.state[0].tolist()
            peak_vram_mb = hw_engine.get_telemetry().get('max_allocated_mb', 0.0)

            grad_embed = agent_brain.pos_embeddings.byte_embed.weight.grad.norm().item() if agent_brain.pos_embeddings.byte_embed.weight.grad is not None else 0.0
            grad_head = 0.0
            if hasattr(agent_brain.attractor_head, 'attractor_basins') and agent_brain.attractor_head.attractor_basins.grad is not None:
                grad_head = agent_brain.attractor_head.attractor_basins.grad.norm().item()

            print(f"\n" + "="*85)
            print(f" === [KEP RULE #6 SINGLE-PASS DIAGNOSTICS DASHBOARD | STREAM STEP {batch_idx+1:04d}/{len(stream_loader)}] ===")
            print("="*85)
            print(f"Plasticity Gating Status  : {status_str}")
            print(f"Submodule Timing (ms)     : Forward+Scan: {t_exec_ms:.1f}ms | Backward+Step: {t_opt_ms:.1f}ms")
            print(f"Stream Performance        : Step Duration: {batch_total_ms:.1f}ms | Throughput: {tokens_per_sec:.1f} tok/s")
            print(f"Metrics Progress          : Speech Loss = {speech_loss_val:.4f} (PPL: {perplexity:.2f}) | Free Energy = {fe_val:.4f}")
            print(f"Gradient Flow Inspection  : Embeddings Grad Norm = {grad_embed:.6f} | Attractor Head Grad Norm = {grad_head:.6f}")
            print(f"Hardware & Somatic        : Peak VRAM: {peak_vram_mb:.1f} MB | Somatic Energy: {energy:.3f} | Sleep Cycles: {total_sleep_cycles}")
            print("="*85)

        if (batch_idx + 1) % 50 == 0:
            diag_sample = run_diagnostic_text_sample(agent_brain, episodic_mem, hu.state, core_config)
            logger.info(f"💬 [KEP Rule #4 Diagnostic Speech Sample @ Step {batch_idx+1}] -> \"{diag_sample}\"\n")
            gc.collect()
            if device_str == 'cuda':
                torch.cuda.empty_cache()

        # Periodic container saving & HF Auto-Sync every 200 stream steps
        if (batch_idx + 1) % 200 == 0:
            save_karyon(agent_brain, episodic_mem, hu, h_curr[0:1], h_curr[0:1], epoch=1, story_idx=(batch_idx + 1) * BATCH_SIZE, filepath=kcore_path)
            commit_msg = f"feat(weights): single-pass stream step {batch_idx+1}/{len(stream_loader)} checkpoint - loss={speech_loss_val:.4f}"
            sync_checkpoint_to_hf(kcore_path, hf_repo_id, commit_msg)

    # Final container save & HF sync
    save_karyon(agent_brain, episodic_mem, hu, h_curr[0:1], h_curr[0:1], epoch=1, story_idx=len(stream_loader) * BATCH_SIZE, filepath=kcore_path)
    sync_checkpoint_to_hf(kcore_path, hf_repo_id, f"feat(weights): single-pass stream complete - final loss={speech_loss_val:.4f}")

    logger.info(f"Single-Pass Continuous Stream Session Complete! Total Steps: {len(stream_loader)} | Total Adapted: {total_adapted_batches} | Total Skipped: {total_skipped_batches} | Total Sleep Cycles: {total_sleep_cycles}.")

if __name__ == "__main__":
    run_single_pass_training()
