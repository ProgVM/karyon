# train_single_pass.py
"""
===============================================================================
KARYON SINGLE-PASS CONTINUOUS ALLOSTATIC LEARNING RUNTIME (v6.0 MASTER)
===============================================================================
Grounded in KEP Principles & Biological AGI Reality:
- Single Continuous Stream Pass (N=1 Pass, Zero Artificial Epochs):
  Experience flows continuously as a single unbroken stream of reality.
- Dynamic Allostatic Volitional Sleep 2.0 (Biophysical Sleep & SHY Consolidation):
  Instead of artificial epoch boundaries, Karyon monitors its own somatic energy
  and allostatic strain. When energy drops below 0.35 or when the native C++20
  `HomeostaticNexus` and `DynamicMorphicGraph` require consolidation, Karyon enters
  NREM Replay + Sleep SHY Synaptic Downscaling, restores somatic energy to 1.00,
  and awakens to continue the stream!
- Susumu Ohno Triadic Morphogenesis + Dynamic Routing + Tononi Apoptosis:
  Autonomously sprouts new mathematical primitive nodes under persistent Free Energy stress
  and prunes inactive nodes during sleep consolidation.
- Spatiotemporal Dualism (Principle 22):
  CausalParallelSSD (temporal context flow) + DynamicMorphicGraph (recurrent latent thinking).
- Full C++20 LibTorch Acceleration (UniversalManifold, Morphic Graph, HomeostaticNexus).

Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import os
import time
import math
import gc
import numpy as np

# Force expandable segments to prevent CUDA VRAM fragmentation and OOM on Kaggle GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from huggingface_hub import HfApi

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticNexus, ContinuousHopfieldMemory
from karyon_checkpoint import load_karyon, save_karyon
from karyon_logger import get_logger
from karyon_hardware import get_hardware_engine

logger = get_logger()
torch.set_grad_enabled(True)

hw_engine = get_hardware_engine()
device = hw_engine.device
device_str = str(device)
use_amp = hw_engine.config.enable_amp and not hw_engine.is_cpu
autocast_dtype = torch.bfloat16
logger.info(f"Execution context: {device_str.upper()} (AMP Enabled: {use_amp}, Dtype: {autocast_dtype})")

kcore_path = "karyon_soul_v6.kcore"
hf_repo_id = "progvmoff/karyon-v31-core"


# Function to safely push checkpoint AND training logs to Hugging Face Hub
def sync_checkpoint_to_hf(local_file: str, repo_id: str, commit_msg: str):
    try:
        api = HfApi()
        logger.info(f"🤗 [HF Auto-Sync] Pushing checkpoint '{local_file}' & training logs to HuggingFace Hub: {repo_id}...")

        # 1. Upload .kcore binary checkpoint
        if os.path.exists(local_file):
            api.upload_file(
                path_or_fileobj=local_file,
                path_in_repo="karyon_soul_v6.kcore",
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_msg
            )

        # 2. Upload train.log if exists
        candidate_logs = ["logs/train.log", "train.log"]
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
def build_multidomain_packed_stream(seq_len: int = 512) -> np.ndarray:
    """
    Constructs a rich, diverse, continuous single-pass byte stream spanning:
    1. General Dialogue & Conversational Semantics (vicgalle/alpaca-gpt4)
    2. Factuality, Instructions & Q&A (databricks/databricks-dolly-15k)
    3. Algorithmic Logic & Python Source Code (iamtarun/python_code_instructions_18k_alpaca)
    4. Multi-Step Mathematical & Chain-of-Thought Reasoning (gsm8k)
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
    def __init__(self, flat_stream: np.ndarray, seq_len: int = 512):
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


BATCH_SIZE = 16
SEQ_LEN = 512

flat_stream = build_multidomain_packed_stream(seq_len=SEQ_LEN)
train_dataset = ContinuousPackedDataset(flat_stream, seq_len=SEQ_LEN)

stream_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # Sequential continuous stream flow (Single Pass)
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
core_config.net.unified_dim = 256
core_config.net.hidden_dim = 512

agent_brain = CoREAgent(vocab_size=258, embed_dim=256, device=device_str).to(device)
hu_nexus = HomeostaticNexus(device=device_str) if HomeostaticNexus else None
hopfield_mem = ContinuousHopfieldMemory(dim=256, num_basins=32, device=device_str) if ContinuousHopfieldMemory else None

h_fast = torch.zeros(1, agent_brain.hidden_dim, device=device)
h_slow = torch.zeros(1, agent_brain.hidden_dim, device=device)

start_step = 0
if os.path.exists(kcore_path):
    try:
        h_fast, h_slow, saved_epoch, saved_story_idx = load_karyon(agent_brain, hopfield_mem, hu_nexus, filepath=kcore_path, device=device_str)
        start_step = (saved_story_idx // BATCH_SIZE) if saved_story_idx else 0
        if start_step >= len(stream_loader):
            start_step = 0
        if start_step > 0:
            logger.info(f"⏩ [Resume Detected] Found saved checkpoint at step {start_step}/{len(stream_loader)}. Resuming stream seamlessly...")
    except Exception as e:
        logger.warning(f"Could not load '{kcore_path}': {e}. Starting fresh stream session.")

BASE_LR = 1.2e-4
optimizer = optim.AdamW(agent_brain.parameters(), lr=BASE_LR, weight_decay=0.01)
criterion_speech = nn.CrossEntropyLoss(ignore_index=256)

scaler = torch.amp.GradScaler(hw_engine.device_type, enabled=(use_amp and autocast_dtype == torch.float16))


def get_neuromodulated_lr(base_lr: float, hu: HomeostaticNexus) -> float:
    """Dynamic Neuromodulated Learning Rate (Dayan & Friston)."""
    if hu is None:
        return base_lr
    states = hu.get_states()
    curiosity = float(states[0].item())
    energy = float(states[1].item())
    na = float(states[4].item())

    allostatic_gain = 0.40 + 1.20 * na + 0.80 * curiosity - 0.30 * (1.0 - energy)
    allostatic_gain = max(0.40, min(2.00, allostatic_gain))
    return base_lr * allostatic_gain


total_adapted_batches = 0
total_sleep_cycles = 0


# =============================================================================
# 3. KEP RULE #4: LIVE DIAGNOSTIC TEXT SAMPLER
# =============================================================================
def run_diagnostic_text_sample(agent: CoREAgent, max_tokens: int = 64) -> str:
    agent.eval()
    prompt = "User: What is the primary source of energy for Earth?\nKaryon:"
    prompt_bytes = list(prompt.encode('utf-8'))
    input_ids = torch.tensor([prompt_bytes], dtype=torch.long, device=agent.device)

    generated_bytes = []
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = agent(input_ids, thinking_steps=2) # [1, S, V]
            next_byte_logits = logits[0, -1, :256]
            next_byte = torch.argmax(next_byte_logits).item()
            if next_byte == 257:  # EOS
                break
            generated_bytes.append(next_byte)
            input_ids = torch.cat([input_ids, torch.tensor([[next_byte]], device=agent.device)], dim=1)

    agent.train()
    try:
        return bytes(generated_bytes).decode('utf-8', errors='replace').strip()
    except Exception:
        return ""


logger.info(f"Starting Single-Pass Allostatic Session (1 Continuous Stream Pass, B={BATCH_SIZE}, S={SEQ_LEN}, {BATCH_SIZE * SEQ_LEN} tokens/step)...")


# =============================================================================
# 4. SINGLE-PASS CONTINUOUS ALLOSTATIC STREAMING LOOP
# =============================================================================
def run_single_pass_training():
    global total_adapted_batches, total_sleep_cycles

    logger.info(f"\n{'='*85}\n === [STARTING CONTINUOUS STREAM LEARNING (N=1 PASS, Spatiotemporal Dualism)] ===\n{'='*85}")

    for batch_idx, batch_tokens in enumerate(stream_loader):
        if batch_idx < start_step:
            continue

        t_batch_start = time.perf_counter()

        batch_tokens = batch_tokens.to(device, non_blocking=(device_str == 'cuda'))
        current_batch_size = batch_tokens.size(0)
        seq_len = batch_tokens.size(1)

        input_seq = batch_tokens[:, :-1]
        target_seq = batch_tokens[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.amp.autocast(device_type=device_str, dtype=autocast_dtype, enabled=use_amp):
                # Forward through Spatiotemporal Engine (SSD + Morphic Recirculation)
                logits = agent_brain(input_seq, thinking_steps=3) # [B, S, V]
                loss = criterion_speech(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e) or isinstance(e, torch.OutOfMemoryError):
                logger.warning(f"⚠️ [Step {batch_idx+1}] CUDA OOM intercepted. Purging VRAM cache...")
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                if device_str == 'cuda':
                    torch.cuda.empty_cache()
                time.sleep(2.0)
                continue
            else:
                raise e

        speech_loss_val = loss.item()
        if math.isnan(speech_loss_val):
            logger.warning(f"⚠️ [Step {batch_idx+1}] Loss NaN detected. Resetting gradients and advancing stream...")
            optimizer.zero_grad(set_to_none=True)
            continue

        # Neuromodulated Learning Rate
        cur_lr = get_neuromodulated_lr(BASE_LR, hu_nexus)
        for group in optimizer.param_groups:
            group['lr'] = cur_lr

        # Backward & Optimization Step
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(agent_brain.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent_brain.parameters(), max_norm=0.5)
            optimizer.step()
        total_adapted_batches += 1

        # Interoceptive Homeostatic Updates (Ashby Homeostasis)
        if hu_nexus is not None:
            hu_nexus.update(torch.tensor([-0.001, -0.002, 0.001, 0.0, 0.005, 0.005], device=device))

        # Autonomous Morphogenesis Reflex check
        stress_val = agent_brain.get_somatic_stress()
        num_organelles = agent_brain.get_active_organelles_count()
        free_slots = agent_brain.get_free_organelle_slots()

        # Check Sleep & Synaptic Consolidation Condition
        states = hu_nexus.get_states() if hu_nexus is not None else torch.tensor([0.5, 1.0, 0.9, 1.0, 0.2, 0.1])
        energy_val = float(states[1].item())
        should_sleep = (energy_val <= 0.30) or ((batch_idx + 1) % 500 == 0)

        if should_sleep:
            total_sleep_cycles += 1
            t_sleep_start = time.perf_counter()
            logger.info(f"🌙 [Step {batch_idx+1}] Somatic Energy={energy_val:.2f} | Executing Sleep & Edelman Neurodarwinian Pruning...")

            # Prune inactive / dead sprouted nodes (Tononi SHY)
            pruned_count = agent_brain.prune_inactive_nodes(threshold=0.01)

            # Restore homeostatic energy
            if hu_nexus is not None:
                hu_nexus.update(torch.tensor([0.0, 1.0 - energy_val, 0.0, 0.0, 0.0, 0.0], device=device))

            sleep_duration_ms = (time.perf_counter() - t_sleep_start) * 1000.0
            logger.info(f"☀️ [Awakened @ Step {batch_idx+1}] Sleep Complete ({sleep_duration_ms:.1f}ms). Pruned Nodes={pruned_count} | Active Organelles={agent_brain.get_active_organelles_count()}")

            # Periodic container save & cloud sync
            save_karyon(agent_brain, hopfield_mem, hu_nexus, h_fast, h_slow, epoch=1, story_idx=(batch_idx + 1) * BATCH_SIZE, filepath=kcore_path)
            commit_msg = f"feat(weights): single-pass stream step {batch_idx+1}/{len(stream_loader)} checkpoint - loss={speech_loss_val:.4f}"
            sync_checkpoint_to_hf(kcore_path, hf_repo_id, commit_msg)

            gc.collect()
            if device_str == 'cuda':
                torch.cuda.empty_cache()

        batch_total_ms = (time.perf_counter() - t_batch_start) * 1000.0
        tokens_per_sec = (current_batch_size * (seq_len - 1)) / max(batch_total_ms / 1000.0, 1e-6)

        # Logging Dashboard every 25 steps
        if (batch_idx + 1) % 25 == 0 or batch_idx == len(stream_loader) - 1:
            perplexity = math.exp(min(speech_loss_val, 20.0))
            peak_vram_mb = hw_engine.get_telemetry().get('max_allocated_mb', 0.0)

            print("\n" + "=" * 85)
            print(f" === [KARYON v6.0 SINGLE-PASS DASHBOARD | STREAM STEP {batch_idx+1:04d}/{len(stream_loader)}] ===")
            print("=" * 85)
            print(f"Stream Performance        : Step Duration: {batch_total_ms:.1f}ms | Throughput: {tokens_per_sec:.1f} tok/s")
            print(f"Metrics Progress          : Speech Loss = {speech_loss_val:.4f} (PPL: {perplexity:.2f})")
            print(f"Morphogenetic Organelles  : Active Organelles = {num_organelles} | Free Slots = {free_slots} | Stress S_t = {stress_val:.4f}")
            print(f"Hardware & Somatic        : Peak VRAM: {peak_vram_mb:.1f} MB | Somatic Energy: {energy_val:.3f} | Sleep Cycles: {total_sleep_cycles}")
            print("=" * 85)

        # KEP Rule #4 Diagnostic text sample every 50 steps
        if (batch_idx + 1) % 50 == 0:
            diag_sample = run_diagnostic_text_sample(agent_brain)
            logger.info(f"💬 [KEP Rule #4 Diagnostic Speech Sample @ Step {batch_idx+1}] -> \"{diag_sample}\"\n")

        # Explicitly release step tensors
        del loss, logits, input_seq, target_seq

    # Final Save & HF Sync
    save_karyon(agent_brain, hopfield_mem, hu_nexus, h_fast, h_slow, epoch=1, story_idx=len(stream_loader) * BATCH_SIZE, filepath=kcore_path)
    sync_checkpoint_to_hf(kcore_path, hf_repo_id, f"feat(weights): stream complete - final loss={speech_loss_val:.4f}")
    logger.info(f"Single-Pass Continuous Stream Session Complete! Total Steps: {len(stream_loader)} | Total Adapted: {total_adapted_batches} | Sleep Cycles: {total_sleep_cycles}.")


if __name__ == "__main__":
    run_single_pass_training()
