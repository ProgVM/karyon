# train_continuous_web.py
"""
===============================================================================
KARYON CONTINUOUS WEB-SCALE PRE-TRAINING & KNOWLEDGE ACQUISITION RUNTIME (v31.0)
Autonomous Living Mind Pipeline:
1. Multi-Threaded Web Crawling across diverse knowledge domains (Cybernetics, Physics,
   Neuroscience, Philosophy, Computer Science, Literature, History, Mathematics).
2. Biophysical "Karyon Sieve": Shannon Entropy & Morphemic Coherence byte-filtering.
3. Zero-Copy Packed Continuous Streaming (S=1024, B=16, 0% Padding, EOS=257).
4. GPU-Accelerated 2-Stage PW-LPER Cortical Stack + Active Inference Latent World Model.
5. Periodic Automated Hugging Face Hub Checkpoint Sync (progvmoff/karyon-v31-core) to prevent data loss.
6. Full KEP Telemetry Dashboard (Loss, PPL, Free Energy, Somatic Homeostasis, Grad Norms, Speech Samples).
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

# Unconditional Dynamo Hotfix for Python 3.12 / Kaggle GPU
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
from huggingface_hub import HfApi

import karyon_config, karyon_core, karyon_agent, karyon_checkpoint, karyon_logger, karyon_crawler
importlib.reload(karyon_core)
importlib.reload(karyon_agent)

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory
from karyon_checkpoint import load_karyon, save_karyon
from karyon_logger import get_logger
from karyon_hardware import get_hardware_engine
from karyon_crawler import KaryonWebCrawler, KaryonDatasetBuilder, KaryonSieve
from init_priors import initialize_priors

logger = get_logger()
torch.set_grad_enabled(True)

hw_engine = get_hardware_engine()
device = hw_engine.device
device_str = str(device)
use_amp = hw_engine.config.enable_amp and not hw_engine.is_cpu

kcore_path = "karyon_soul.kcore"
hf_repo_id = "progvmoff/karyon-v31-core"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Broad seed URLs covering diverse scientific, technical, philosophical, and literary knowledge
DIVERSE_WEB_SEEDS = [
    # Cybernetics, Active Inference, Neuroscience
    "https://en.wikipedia.org/wiki/Active_inference",
    "https://en.wikipedia.org/wiki/Free_energy_principle",
    "https://en.wikipedia.org/wiki/Karl_J._Friston",
    "https://en.wikipedia.org/wiki/Cybernetics",
    "https://en.wikipedia.org/wiki/W._Ross_Ashby",
    "https://en.wikipedia.org/wiki/Homeostasis",
    "https://en.wikipedia.org/wiki/State-space_model",
    "https://en.wikipedia.org/wiki/Hopfield_network",
    "https://en.wikipedia.org/wiki/Neuroplasticity",
    "https://en.wikipedia.org/wiki/Cellular_automaton",
    
    # Physics, Astronomy & Mathematics
    "https://en.wikipedia.org/wiki/Thermodynamics",
    "https://en.wikipedia.org/wiki/Quantum_mechanics",
    "https://en.wikipedia.org/wiki/General_relativity",
    "https://en.wikipedia.org/wiki/Information_theory",
    "https://en.wikipedia.org/wiki/Claude_Shannon",
    "https://en.wikipedia.org/wiki/Differential_equation",
    "https://en.wikipedia.org/wiki/Stochastic_process",
    "https://en.wikipedia.org/wiki/Fourier_transform",
    "https://en.wikipedia.org/wiki/Entropy",
    "https://en.wikipedia.org/wiki/Complex_system",
    
    # Computer Science & Artificial Intelligence
    "https://en.wikipedia.org/wiki/Artificial_general_intelligence",
    "https://en.wikipedia.org/wiki/Alan_Turing",
    "https://en.wikipedia.org/wiki/Von_Neumann_architecture",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Recurrent_neural_network",
    "https://en.wikipedia.org/wiki/Compiler",
    "https://en.wikipedia.org/wiki/Operating_system",
    "https://en.wikipedia.org/wiki/Parallel_computing",
    
    # Philosophy & Cognitive Science
    "https://en.wikipedia.org/wiki/Philosophy_of_mind",
    "https://en.wikipedia.org/wiki/Consciousness",
    "https://en.wikipedia.org/wiki/Epistemology",
    "https://en.wikipedia.org/wiki/Cognitive_science",
    "https://en.wikipedia.org/wiki/Embodied_cognition",
    
    # World History & Earth Science
    "https://en.wikipedia.org/wiki/History_of_science",
    "https://en.wikipedia.org/wiki/Origin_of_life",
    "https://en.wikipedia.org/wiki/Evolutionary_biology",
    "https://en.wikipedia.org/wiki/Earth",
    "https://en.wikipedia.org/wiki/Sun"
]

def sync_checkpoint_to_hf(local_file: str, repo_id: str, commit_msg: str):
    """Asynchronously / safely uploads the model container to Hugging Face Hub."""
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
        logger.warning(f"⚠️ [HF Auto-Sync Notice] Cloud upload skipped/delayed ({e}). Local copy is safe.")


class PackedStreamDataset(Dataset):
    """Contiguous Byte Stream (0% Padding, S=1024, uint16 -> int64)."""
    def __init__(self, npy_path: str, seq_len: int = 1024):
        self.seq_len = seq_len
        self.flat_stream = np.load(npy_path, mmap_mode='r')
        self.num_blocks = len(self.flat_stream) // (seq_len + 1)
        logger.info(f"Loaded packed stream from '{npy_path}' ({len(self.flat_stream):,} bytes, {self.num_blocks} blocks).")

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        end = start + (self.seq_len + 1)
        return torch.from_numpy(self.flat_stream[start:end].astype(np.int64))

def collate_packed_fn(batch):
    return torch.stack(batch, dim=0)


def run_diagnostic_speech(agent, memory, hu_state, config):
    """KEP Rule #4: Periodically audits syntax and semantics using Top-p nucleus sampling."""
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


def run_continuous_web_training_pipeline():
    logger.info("="*85)
    logger.info(" === [KARYON CONTINUOUS WEB PRE-TRAINING RUNTIME (v31.0 MASTER)] ===")
    logger.info(f" Hardware Device: {device_str.upper()} | AMP FP16 Enabled: {use_amp}")
    logger.info("="*85)

    # 1. Check or Run Crawler
    web_stream_path = os.path.join(data_dir, "karyon_web_corpus.npy")
    
    if not os.path.exists(web_stream_path) or os.path.getsize(web_stream_path) < 100000:
        logger.info(f"🕸️ [Phase 1: Web Crawling] Initiating crawl on {len(DIVERSE_WEB_SEEDS)} seed domains...")
        crawler = KaryonWebCrawler(start_urls=DIVERSE_WEB_SEEDS, max_depth=2, max_pages=300, concurrency=16, timeout=6.0)
        docs = crawler.crawl()
        
        logger.info(f"📦 [Phase 2: Corpus Packing] Formatting and packing {len(docs)} sieve-passed documents into .kbin...")
        builder = KaryonDatasetBuilder(output_dir=data_dir)
        builder.build_packed_binary_stream(docs, filename="karyon_web_corpus.npy")
    else:
        logger.info(f"Found existing packed web corpus at '{web_stream_path}' ({os.path.getsize(web_stream_path)/(1024*1024):.2f} MB).")

    # 2. Build DataLoader
    BATCH_SIZE = 16
    SEQ_LEN = 1024
    CHUNK_SIZE = 64
    NUM_EPOCHS = 10

    dataset = PackedStreamDataset(web_stream_path, seq_len=SEQ_LEN)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_packed_fn,
        drop_last=True,
        num_workers=2,
        pin_memory=hw_engine.is_cuda
    )

    logger.info(f"Web Dataset Loaded. Blocks: {len(dataset)} | Batches/Epoch: {len(loader)} | Batch Size: {BATCH_SIZE} | Epochs: {NUM_EPOCHS}")

    # 3. Model & Memory Initialization
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

    agent_brain = CoREAgent(config=core_config, device=device_str).to(device)
    hu = HomeostaticUnit(batch_size=BATCH_SIZE, device=device_str)
    episodic_mem = BatchedEpisodicMemory(batch_size=BATCH_SIZE, memory_dim=core_config.net.unified_dim, max_capacity=1000, device=device_str)

    if not os.path.exists(kcore_path):
        initialize_priors(recreate=True, filepath=kcore_path, device=device_str)

    h_fast, h_slow, saved_epoch, _ = load_karyon(agent_brain, episodic_mem, hu, filepath=kcore_path, device=device_str)

    optimizer = optim.AdamW(agent_brain.get_all_parameters(), lr=3e-3, weight_decay=0.01)
    criterion_speech = nn.CrossEntropyLoss(ignore_index=256)
    scaler = torch.amp.GradScaler(hw_engine.device_type, enabled=use_amp)

    TOTAL_STEPS = len(loader) * NUM_EPOCHS
    WARMUP_STEPS = 50

    def get_lr_multiplier(current_step: int) -> float:
        if current_step < WARMUP_STEPS:
            return float(current_step + 1) / float(WARMUP_STEPS)
        progress = float(current_step - WARMUP_STEPS) / float(max(1, TOTAL_STEPS - WARMUP_STEPS))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.0333, cosine_decay)

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)

    global_step = 0
    total_adapted = 0
    total_skipped = 0

    logger.info(f"🚀 [Phase 3: Continuous Training] Commencing training across {NUM_EPOCHS} epochs with HF Auto-Sync...")

    for epoch in range(NUM_EPOCHS):
        current_epoch_num = saved_epoch + epoch + 1
        logger.info(f"\n{'='*85}\n === [STARTING WEB EPOCH {epoch+1}/{NUM_EPOCHS} (CONTAINER EPOCH {current_epoch_num})] ===\n{'='*85}")

        for batch_idx, batch_tokens in enumerate(loader):
            t_start = time.perf_counter()
            batch_tokens = batch_tokens.to(device, non_blocking=(device_str == 'cuda'))
            cur_b = batch_tokens.size(0)
            seq_l = batch_tokens.size(1)

            inp = batch_tokens[:, :-1]
            tgt = batch_tokens[:, 1:]

            optimizer.zero_grad()

            t_fwd_0 = time.perf_counter()
            with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
                total_loss, s_loss, fe_val, m_s2, h_proxy, curr_u_t, eff_dt = agent_brain.forward_sequence(
                    inp, tgt, hu, criterion_speech, episodic_memory=episodic_mem,
                    loss_free_energy_weight=0.05, chunk_size=CHUNK_SIZE, use_checkpointing=False
                )
            t_fwd_ms = (time.perf_counter() - t_fwd_0) * 1000.0

            if math.isnan(s_loss) or math.isnan(fe_val) or torch.isnan(total_loss).any():
                logger.warning(f"⚠️ [Epoch {epoch+1} Step {batch_idx+1}] NaN detected. Resetting gradients and somatic homeostasis.")
                optimizer.zero_grad()
                hu.state = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat((cur_b, 1))
                continue

            action_cost = torch.full((cur_b, 1), 0.001, device=device)
            pred_err = torch.full((cur_b, 1), float(s_loss * 0.1), device=device)
            entropy_t = torch.full((cur_b, 1), float(fe_val), device=device)
            cog_act = torch.zeros((cur_b, 1), dtype=torch.int64, device=device)
            hu.update(action_cost, pred_err, entropy_t, cog_act)

            # Optimization Step
            t_bwd_0 = time.perf_counter()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(agent_brain.get_all_parameters(), max_norm=3.0)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()

            if scale_before <= scale_after:
                lr_scheduler.step()

            t_bwd_ms = (time.perf_counter() - t_bwd_0) * 1000.0
            cur_lr = optimizer.param_groups[0]['lr']
            total_adapted += 1
            global_step += 1

            t_batch_ms = (time.perf_counter() - t_start) * 1000.0
            tok_per_sec = (cur_b * (seq_l - 1)) / (t_batch_ms / 1000.0)

            # Telemetry Reporting
            if (batch_idx + 1) % 20 == 0 or batch_idx == len(loader) - 1:
                ppl = math.exp(min(s_loss, 20.0))
                curiosity, energy, stability, health, na, da = hu.state[0].tolist()
                vram_mb = hw_engine.get_telemetry().get('max_allocated_mb', 0.0)
                grad_emb = agent_brain.pos_embeddings.byte_embed.weight.grad.norm().item() if agent_brain.pos_embeddings.byte_embed.weight.grad is not None else 0.0
                grad_attractor = agent_brain.attractor_head.attractor_basins.grad.norm().item() if hasattr(agent_brain.attractor_head, 'attractor_basins') and agent_brain.attractor_head.attractor_basins.grad is not None else 0.0

                print(f"\n" + "="*85)
                print(f" === [KEP TELEMETRY DASHBOARD | WEB EPOCH {epoch+1}/{NUM_EPOCHS} | STEP {batch_idx+1:04d}/{len(loader)}] ===")
                print("="*85)
                print(f"Plasticity Gating Status  : ADAPTED (lr={cur_lr:.6f})")
                print(f"Submodule Timing (ms)     : Forward+Scan: {t_fwd_ms:.1f}ms | Backward+Step: {t_bwd_ms:.1f}ms")
                print(f"Batch Performance         : Total Batch: {t_batch_ms:.1f}ms | Throughput: {tok_per_sec:.1f} tok/s")
                print(f"Metrics Progress          : Speech Loss = {s_loss:.4f} (PPL: {ppl:.2f}) | Free Energy = {fe_val:.4f}")
                print(f"Gradient Flow Inspection  : Embeddings Grad Norm = {grad_emb:.6f} | Attractor Grad Norm = {grad_attractor:.6f}")
                print(f"Hardware & Somatic        : Peak VRAM: {vram_mb:.1f} MB | Somatic Energy: {energy:.3f} | Arousal(NA): {na:.3f}")
                print("="*85)

            # Diagnostic Speech Sample
            if (batch_idx + 1) % 40 == 0:
                diag_sample = run_diagnostic_speech(agent_brain, episodic_mem, hu.state, core_config)
                logger.info(f"💬 [KEP Rule #4 Diagnostic Speech Sample @ Web Epoch {epoch+1} Step {batch_idx+1}] -> \"{diag_sample}\"\n")
                gc.collect()
                hw_engine.empty_cache()

            # EXP-109 Validated Feature: Interleaved Autonomous Self-Learning Cycle every 100 steps
            if (batch_idx + 1) % 100 == 0:
                sl_res = agent_brain.execute_autonomous_self_learning_cycle(
                    hu, episodic_mem, optimizer, criterion_speech, num_self_sequences=3, seq_len=64, scaler=scaler
                )
                logger.info(f"🧠 [Autonomous Self-Learning @ Step {batch_idx+1}] Inner Monologue FE: {sl_res['final_free_energy']:.4f} | SEEKING Drive: {sl_res['seeking_drive']:.3f}\n")

            # Intermediate Checkpoint Auto-Sync to HF Hub every 150 batches
            if (batch_idx + 1) % 150 == 0:
                save_karyon(agent_brain, episodic_mem, hu, h_proxy[0:1], h_proxy[0:1], epoch=current_epoch_num, story_idx=global_step, filepath=kcore_path)
                commit_msg = f"feat(weights): auto-sync web-pretrain step {global_step} (epoch {current_epoch_num}) - loss={s_loss:.4f}"
                sync_checkpoint_to_hf(kcore_path, hf_repo_id, commit_msg)

        # End of Epoch Sleep Consolidation
        logger.info(f"🌙 [Web Epoch {epoch+1} Complete] Entering Allostatic Sleep (Hippocampal Replay & SHY Synaptic Scaling)...")
        agent_brain.execute_deep_allostatic_sleep(episodic_mem, hu, num_replay_cycles=5, downscaling_factor=0.03)
        logger.info(f"☀️ [Awakened] Somatic Energy Restored: {hu.state[0, 1].item():.2f}.\n")

        # Save and Push Checkpoint at End of Each Epoch
        save_karyon(agent_brain, episodic_mem, hu, h_proxy[0:1], h_proxy[0:1], epoch=current_epoch_num, story_idx=global_step, filepath=kcore_path)
        commit_msg = f"feat(weights): epoch {current_epoch_num} complete - loss={s_loss:.4f} (ppl={math.exp(min(s_loss, 20.0)):.2f})"
        sync_checkpoint_to_hf(kcore_path, hf_repo_id, commit_msg)

        gc.collect()
        hw_engine.empty_cache()

    logger.info("🎉 [Continuous Web Pre-Training Complete] All epochs successfully finished and persisted to HF Hub!")


if __name__ == "__main__":
    run_continuous_web_training_pipeline()
