# train_single_pass.py
"""
===============================================================================
KARYON SINGLE-PASS CONTINUOUS ALLOSTATIC LEARNING RUNTIME (v33.0 MASTER)
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
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

# Force expandable segments to prevent CUDA VRAM fragmentation and OOM on Kaggle GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
# Force bfloat16 for numerical stability and zero AMP scaler underflow NaNs
use_amp = hw_engine.config.enable_amp and not hw_engine.is_cpu
autocast_dtype = torch.bfloat16
logger.info(f"Execution context: {device_str.upper()} (AMP Enabled: {use_amp}, Dtype: {autocast_dtype})")

kcore_path = "karyon_soul.kcore"
hf_repo_id = "progvmoff/karyon-v31-core"

# Function to safely push checkpoint AND training logs to Hugging Face Hub
def sync_checkpoint_to_hf(local_file: str, repo_id: str, commit_msg: str):
    try:
        api = HfApi()
        logger.info(f"🤗 [HF Auto-Sync] Pushing checkpoint '{local_file}' & training logs to HuggingFace Hub: {repo_id}...")
        
        # 1. Upload .kcore binary checkpoint
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo="karyon_soul.kcore",
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg
        )
        
        # 2. Upload train.log if exists in either root or logs/ directory
        candidate_logs = ["train.log", "logs/train.log"]
        for log_candidate in candidate_logs:
            if os.path.exists(log_candidate) and os.path.getsize(log_candidate) > 0:
                api.upload_file(
                    path_or_fileobj=log_candidate,
                    path_in_repo="logs/train.log",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"changelog: update training execution log ({commit_msg})"
                )
                logger.info(f"🤗 [HF Auto-Sync] Uploaded training log '{log_candidate}' to '{repo_id}:logs/train.log'")
                break
            
        logger.info(f"🤗 [HF Auto-Sync] Checkpoint sync cycle complete for '{repo_id}'!")
    except Exception as e:
        logger.warning(f"⚠️ [HF Auto-Sync Warning] Failed to upload checkpoint/log to HuggingFace Hub: {e}")

# =============================================================================
# 1. MULTI-DOMAIN CONTINUOUS STREAM DATASET BUILDER
# =============================================================================
def build_multidomain_packed_stream(seq_len: int = 1024) -> np.ndarray:
    """
    Constructs a rich, diverse, continuous single-pass byte stream spanning:
    1. General Dialogue & Conversational Semantics (vicgalle/alpaca-gpt4)
    2. Factuality, Instructions & Q&A (databricks/databricks-dolly-15k)
    3. Algorithmic Logic & Python Source Code (iamtarun/python_code_instructions_18k_alpaca)
    4. Multi-Step Mathematical & Chain-of-Thought Reasoning (gsm8k)
    
    Packaged as a 100% continuous byte array with zero padding and EOS (257) delimiters.
    """
    corpus_cache_file = "data/karyon_multidomain_single_pass_stream.npy"
    if os.path.exists(corpus_cache_file):
        logger.info(f"⚡ Loading pre-compiled multi-domain packed stream from cache: '{corpus_cache_file}'...")
        flat_stream = np.load(corpus_cache_file)
        logger.info(f"Loaded {len(flat_stream):,} continuous bytes from disk cache.")
        return flat_stream

    os.makedirs("data", exist_ok=True)
    logger.info("Assembling Rich Multi-Domain Continuous Stream Dataset (Alpaca, Dolly, Code, GSM8k)...")
    
    text_chunks = []
    
    # Domain 1: General Dialogue (Alpaca GPT-4)
    logger.info(" -> Ingesting Domain 1: vicgalle/alpaca-gpt4 (General Dialogue)...")
    try:
        ds_alpaca = load_dataset("vicgalle/alpaca-gpt4", split="train")
        for item in ds_alpaca:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if inp:
                text = f"User: {inst}\nContext: {inp}\nKaryon: {out}\n\n"
            else:
                text = f"User: {inst}\nKaryon: {out}\n\n"
            text_chunks.append(text)
        logger.info(f"Loaded {len(ds_alpaca):,} Alpaca dialogue samples.")
    except Exception as e:
        logger.warning(f"Failed to load Alpaca dataset: {e}")

    # Domain 2: Instruction & QA (Databricks Dolly 15k)
    logger.info(" -> Ingesting Domain 2: databricks/databricks-dolly-15k (Instructions & QA)...")
    try:
        ds_dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        for item in ds_dolly:
            inst = item.get("instruction", "").strip()
            ctx = item.get("context", "").strip()
            resp = item.get("response", "").strip()
            if ctx:
                text = f"Instruction: {inst}\nReference Context: {ctx}\nAnswer: {resp}\n\n"
            else:
                text = f"Question: {inst}\nAnswer: {resp}\n\n"
            text_chunks.append(text)
        logger.info(f"Loaded {len(ds_dolly):,} Dolly instruction samples.")
    except Exception as e:
        logger.warning(f"Failed to load Dolly dataset: {e}")

    # Domain 3: Python Code Logic (Python Code Instructions 18k)
    logger.info(" -> Ingesting Domain 3: iamtarun/python_code_instructions_18k_alpaca (Code Logic)...")
    try:
        ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        for item in ds_code:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if inp:
                text = f"Problem: {inst}\nCode Context: {inp}\nSolution:\n{out}\n\n"
            else:
                text = f"Coding Task: {inst}\nSolution:\n{out}\n\n"
            text_chunks.append(text)
        logger.info(f"Loaded {len(ds_code):,} Python Code samples.")
    except Exception as e:
        logger.warning(f"Failed to load Python Code dataset: {e}")

    # Domain 4: Step-by-Step Chain-of-Thought (GSM8k)
    logger.info(" -> Ingesting Domain 4: gsm8k (Step-by-Step Chain-of-Thought)...")
    try:
        ds_gsm = load_dataset("gsm8k", "main", split="train")
        for item in ds_gsm:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            text = f"Math Question: {q}\nStep-by-Step Solution: {a}\n\n"
            text_chunks.append(text)
        logger.info(f"Loaded {len(ds_gsm):,} GSM8k math reasoning samples.")
    except Exception as e:
        logger.warning(f"Failed to load GSM8k dataset: {e}")

    import random
    random.seed(42)
    random.shuffle(text_chunks)
    
    logger.info(f"Total Unified Multi-Domain Samples: {len(text_chunks):,}. Encoding into raw UTF-8 byte stream...")
    
    encoded_bytes_list = []
    EOS_BYTE = 257
    for t in text_chunks:
        b = t.encode('utf-8', errors='replace')
        encoded_bytes_list.extend(list(b))
        encoded_bytes_list.append(EOS_BYTE)

    flat_stream = np.array(encoded_bytes_list, dtype=np.int16)
    
    # Trim to exact multiple of (seq_len + 1)
    target_multiple = (seq_len + 1)
    valid_len = (len(flat_stream) // target_multiple) * target_multiple
    flat_stream = flat_stream[:valid_len]
    
    np.save(corpus_cache_file, flat_stream)
    logger.info(f"Compiled and cached packed byte stream to '{corpus_cache_file}'. Total size: {len(flat_stream):,} bytes ({len(flat_stream) / 1024 / 1024:.2f} MB).")
    return flat_stream

class ContinuousPackedDataset(Dataset):
    def __init__(self, flat_stream: np.ndarray, seq_len: int = 1024):
        self.flat_stream = flat_stream
        self.seq_len = seq_len
        self.stride = seq_len
        self.num_blocks = (len(flat_stream) - 1) // self.stride

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len + 1
        return torch.from_numpy(self.flat_stream[start:end].astype(np.int64))

def collate_packed_fn(batch):
    return torch.stack(batch, dim=0)

BATCH_SIZE = 8
SEQ_LEN = 512
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

h_fast, h_slow, saved_epoch, saved_story_idx = load_karyon(agent_brain, episodic_mem, hu, filepath=kcore_path, device=device_str)
start_step = (saved_story_idx // BATCH_SIZE) if saved_story_idx else 0
if start_step >= len(stream_loader):
    start_step = 0 # Loop stream seamlessly upon completing full stream pass
if start_step > 0:
    logger.info(f"⏩ [Resume Detected] Found saved checkpoint at step {start_step}/{len(stream_loader)}. Resuming stream seamlessly...")

BASE_LR = 1.2e-4
optimizer = optim.AdamW(agent_brain.get_all_parameters(), lr=BASE_LR, weight_decay=0.01)
criterion_speech = nn.CrossEntropyLoss(ignore_index=256)

scaler = torch.amp.GradScaler(hw_engine.device_type, enabled=(use_amp and autocast_dtype == torch.float16))

TOTAL_TRAINING_STEPS = len(stream_loader)
WARMUP_STEPS = 50

def get_lr_multiplier(current_step: int) -> float:
    if current_step < WARMUP_STEPS:
        base_mult = float(current_step + 1) / float(WARMUP_STEPS)
    else:
        progress = float(current_step - WARMUP_STEPS) / float(max(1, TOTAL_TRAINING_STEPS - WARMUP_STEPS))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        base_mult = max(0.0333, cosine_decay)
        
    # Resume warmup safety (Axis A): if we resumed, warm up lr over 50 steps from start_step
    if start_step > 0 and current_step >= start_step and current_step < start_step + 50:
        resume_warmup_factor = float(current_step - start_step + 1) / 50.0
        return base_mult * resume_warmup_factor
        
    return base_mult

for group in optimizer.param_groups:
    group['initial_lr'] = group['lr']

lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=start_step - 1 if start_step > 0 else -1)

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
            temperature=0.35,
            top_p=0.90
        )
        for event in gen_stream:
            if event["status"] == "token":
                generated_chars.append(event["text"])
                
    del diag_mem
    del diag_hu
    if agent.device_str == 'cuda':
        torch.cuda.empty_cache()
    agent.train()
    return "".join(generated_chars).strip()

logger.info(f"Starting Single-Pass Allostatic Session (1 Continuous Stream Pass, B={BATCH_SIZE}, S={SEQ_LEN}, 32,768 tokens/step)...")

# =============================================================================
# 4. SINGLE-PASS CONTINUOUS ALLOSTATIC STREAMING LOOP
# =============================================================================
def run_single_pass_training():
    global total_adapted_batches, total_skipped_batches, total_sleep_cycles, moving_mean_fe, moving_var_fe, h_fast, h_slow, optimizer, lr_scheduler, agent_brain
    
    logger.info(f"\n{'='*85}\n === [STARTING SINGLE-PASS CONTINUOUS STREAM LEARNING (N=1 PASS)] ===\n{'='*85}")
    
    for batch_idx, batch_tokens in enumerate(stream_loader):
        # Seamlessly skip already completed stream steps upon resuming
        if batch_idx < start_step:
            continue

        t_batch_start = time.perf_counter()
        
        batch_tokens = batch_tokens.to(device, non_blocking=(device_str == 'cuda'))
        current_batch_size = batch_tokens.size(0)
        seq_len = batch_tokens.size(1)

        input_seq = batch_tokens[:, :-1]
        target_seq = batch_tokens[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        
        t_exec_start = time.perf_counter()
        try:
            with torch.amp.autocast(device_type=device_str, dtype=autocast_dtype, enabled=use_amp):
                total_loss_tensor, speech_loss_val, fe_val, m_curr, h_curr, curr_u_t, eff_dt = agent_brain.forward_sequence(
                    input_seq, target_seq, hu, criterion_speech, episodic_memory=episodic_mem,
                    loss_free_energy_weight=0.05, chunk_size=CHUNK_SIZE, use_checkpointing=False
                )
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e) or isinstance(e, torch.OutOfMemoryError):
                logger.warning(f"⚠️ [Step {batch_idx+1}] CUDA OOM intercepted. Executing batch rollback & purging VRAM cache...")
                if 'total_loss_tensor' in locals(): del total_loss_tensor
                if 'm_curr' in locals(): del m_curr
                if 'h_curr' in locals(): del h_curr
                if 'curr_u_t' in locals(): del curr_u_t
                if 'eff_dt' in locals(): del eff_dt
                if 'input_seq' in locals(): del input_seq
                if 'target_seq' in locals(): del target_seq
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                if device_str == 'cuda':
                    torch.cuda.empty_cache()
                total_skipped_batches += 1
                time.sleep(2.0)
                continue
            else:
                raise e

        t_exec_ms = (time.perf_counter() - t_exec_start) * 1000.0

        if math.isnan(speech_loss_val) or math.isnan(fe_val) or torch.isnan(total_loss_tensor).any():
            logger.warning(f"⚠️ [Step {batch_idx+1}] Anomaly detected (sp_l={speech_loss_val}, fe={fe_val}). Sanitizing tensors and advancing stream...")
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for p in agent_brain.parameters():
                    if torch.isnan(p).any() or torch.isinf(p).any():
                        p.data.nan_to_num_(0.0)
                for b in agent_brain.buffers():
                    if torch.isnan(b).any() or torch.isinf(b).any():
                        b.data.nan_to_num_(0.0)
                if torch.isnan(hu.state).any() or torch.isinf(hu.state).any():
                    hu.state.nan_to_num_(0.5)
            if 'total_loss_tensor' in locals(): del total_loss_tensor
            if 'input_seq' in locals(): del input_seq
            if 'target_seq' in locals(): del target_seq
            if 'm_curr' in locals(): del m_curr
            if 'h_curr' in locals(): del h_curr
            if 'curr_u_t' in locals(): del curr_u_t
            if 'eff_dt' in locals(): del eff_dt
            gc.collect()
            if device_str == 'cuda':
                torch.cuda.empty_cache()
            total_skipped_batches += 1
            time.sleep(0.1)
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
            if scaler.is_enabled():
                scaler.scale(total_loss_tensor).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(agent_brain.get_all_parameters(), max_norm=0.5)
                
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                
                if scale_before <= scale_after:
                    lr_scheduler.step()
            else:
                total_loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(agent_brain.get_all_parameters(), max_norm=0.5)
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
                hu.state[:, 1] = torch.clamp(hu.state[:, 1] + rest_recovery_rate, 0.0, 1.0)
                
            total_skipped_batches += 1
            status_str = f"RESTING / SKIPPED (0 Backprop FLOPs)"

        if should_sleep:
            total_sleep_cycles += 1
            t_sleep_start = time.perf_counter()
            logger.info(f"🌙 [Step {batch_idx+1}] Somatic Energy={energy_val:.2f} | C++20 EFE Volition Action={action_idx}. Entering Biophysical Sleep 2.0...")
            sleep_res = agent_brain.execute_deep_allostatic_sleep(
                episodic_memory=episodic_mem,
                hu=hu,
                num_replay_cycles=2,
                downscaling_factor=0.002,
                eval_inputs=input_seq[0:min(4, input_seq.size(0))],
                eval_targets=target_seq[0:min(4, target_seq.size(0))],
                criterion_speech=criterion_speech
            )
            if isinstance(sleep_res, tuple):
                pruned_weights, evolved_brain = sleep_res
                agent_brain = evolved_brain
            else:
                pruned_weights = sleep_res

            # Re-instantiate optimizer to track any newly sprouted or mutated parameters
            optimizer = optim.AdamW(agent_brain.get_all_parameters(), lr=BASE_LR, weight_decay=0.01)
            for group in optimizer.param_groups:
                group['initial_lr'] = group['lr']
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=batch_idx)

            sleep_duration_ms = (time.perf_counter() - t_sleep_start) * 1000.0
            logger.info(f"☀️ [Awakened @ Step {batch_idx+1}] Sleep 2.0 Complete ({sleep_duration_ms:.1f}ms). Restored Energy={hu.state[0, 1].item():.2f} | Pruned Weights={pruned_weights}")
            gc.collect()
            if device_str == 'cuda':
                torch.cuda.empty_cache()

        if (batch_idx + 1) % 50 == 0:
            gc.collect()
            if device_str == 'cuda':
                torch.cuda.empty_cache()
        batch_total_ms = (time.perf_counter() - t_batch_start) * 1000.0
        tokens_per_sec = (current_batch_size * (seq_len - 1)) / max(batch_total_ms / 1000.0, 1e-6)

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

        # Explicitly release step tensors to keep VRAM clean
        del total_loss_tensor, input_seq, target_seq, m_curr, h_curr, curr_u_t, eff_dt

    # Final container save & HF sync
    h_fast_save = h_fast if 'h_fast' in locals() else torch.zeros(1, agent_brain.hidden_dim, device=device)
    h_slow_save = h_slow if 'h_slow' in locals() else torch.zeros(1, agent_brain.hidden_dim, device=device)
    save_karyon(agent_brain, episodic_mem, hu, h_fast_save[0:1], h_slow_save[0:1], epoch=1, story_idx=len(stream_loader) * BATCH_SIZE, filepath=kcore_path)
    sync_checkpoint_to_hf(kcore_path, hf_repo_id, f"feat(weights): single-pass stream complete - final loss={speech_loss_val if 'speech_loss_val' in locals() else 0.0:.4f}")

    logger.info(f"Single-Pass Continuous Stream Session Complete! Total Steps: {len(stream_loader)} | Total Adapted: {total_adapted_batches} | Total Skipped: {total_skipped_batches} | Total Sleep Cycles: {total_sleep_cycles}.")

if __name__ == "__main__":
    run_single_pass_training()
