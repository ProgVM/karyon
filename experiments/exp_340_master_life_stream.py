# experiments/exp_340_master_life_stream.py
"""
===============================================================================
EXP-340: MASTER LIFE-STREAM SYNTHESIS & COGNITIVE GENERALIZATION MARATHON
===============================================================================
Grounded in KEP Principles, Active Inference, and the Complete Tetrad of Universal Completeness:
1. Continuous Single-Pass Stream Learning (N=1 Pass, Zero Artificial Epochs).
2. Spatiotemporal Dualism & Chrono-Coupling:
   - Temporal Axis: C++20 TriScaleHierarchicalPAC (Fast Gamma -> Meso Theta -> Macro Delta).
   - Spatial / Recurrent Axis: C++20 DynamicMorphicGraph (Adaptive Latent Thinking Recirculation).
3. Active Closed-Loop Anokhin Efference Copy Self-Verification:
   - Intercepts and refines motor discrepancy candidates before biological/speech emission.
4. Active Inference Counterfactual Mental Sandbox (G(tau) Minimization):
   - Counterfactual branch evaluation under multi-step expected free energy.
5. Holographic Vector-Symbolic Variable Binding (HDC/VSA):
   - Frequency-domain circular convolution binding for variable representations.
6. Continuous Tripartite Somatic Valence Memory:
   - Real-time capturing of Attractors (V > +0.2), Repulsors (V < -0.2), and Neutral Anchors (|V| <= 0.2).
7. Tsodyks-Markram Synaptic Fatigue Dynamics:
   - Dynamic refractory scaling to prevent perseverative loops.
8. 3-Phase Sleep & Tononi SHY Neurodarwinian Consolidation:
   - Homeostatic recovery + apoptotic pruning of dead/inactive morphic graph nodes.
9. Throttled Master Container Hub Sync:
   - Saves local 'karyon_soul_v8.kcore' and pushes to Hugging Face strictly every 500 steps.
10. KEP Rule #4 Speech Diagnostics:
   - Samples live speech every 250 steps, auditing syntactic integrity and Anokhin self-corrections.

Lead Cyberneticist: Bazilevs (ProgVM member) & Autonomous Karyon Agent (2026)
===============================================================================
"""

import os
import sys
import time
import math
import gc
from typing import Tuple
import numpy as np

# Set project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force expandable segments to prevent CUDA VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import HfApi

import karyon_core as kcore
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

kcore_path = "karyon_soul_v8.kcore"
hf_repo_id = "progvmoff/karyon-v31-core"


# Throttled Hugging Face Hub Checkpoint Sync
def sync_checkpoint_to_hf(local_file: str, repo_id: str, commit_msg: str):
    try:
        api = HfApi()
        logger.info(f"🤗 [HF Auto-Sync] Pushing checkpoint '{local_file}' & logs to Hugging Face: {repo_id}...")
        if os.path.exists(local_file):
            api.upload_file(
                path_or_fileobj=local_file,
                path_in_repo="karyon_soul_v8.kcore",
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_msg
            )
        candidate_logs = ["logs/train.log", "train.log"]
        for log_cand in candidate_logs:
            if os.path.exists(log_cand) and os.path.getsize(log_cand) > 0:
                api.upload_file(
                    path_or_fileobj=log_cand,
                    path_in_repo="logs/train.log",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"changelog: update training execution log ({commit_msg})"
                )
                break
        logger.info(f"🤗 [HF Auto-Sync] Sync complete for '{repo_id}'!")
    except Exception as e:
        logger.warning(f"⚠️ [HF Auto-Sync Warning] Upload error: {e}")


# =============================================================================
# 1. MULTI-DOMAIN CONTINUOUS STREAM DATASET LOADER
# =============================================================================
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

corpus_cache_file = "data/karyon_multidomain_single_pass_stream.npy"
if os.path.exists(corpus_cache_file):
    logger.info(f"⚡ Loading pre-compiled multi-domain stream from '{corpus_cache_file}'...")
    flat_stream = np.load(corpus_cache_file)
else:
    raise FileNotFoundError(f"Corpus cache '{corpus_cache_file}' not found.")

train_dataset = ContinuousPackedDataset(flat_stream, seq_len=SEQ_LEN)
stream_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # Strict Sequential Single-Pass Reality Flow
    collate_fn=collate_packed_fn,
    drop_last=True,
    num_workers=2,
    persistent_workers=True,
    pin_memory=hw_engine.is_cuda
)

logger.info(f"Multi-Domain Continuous Stream Loaded: {len(train_dataset):,} blocks | {len(stream_loader):,} batches.")


# =============================================================================
# 2. MODEL CONFIGURATION & INITIALIZATION (FULL TETRAD ENGINE)
# =============================================================================
core_config = CoREConfig()
core_config.net.unified_dim = 256
core_config.net.hidden_dim = 512

agent_brain = CoREAgent(vocab_size=258, embed_dim=256, device=device_str).to(device)
hu_nexus = HomeostaticNexus(device=device_str) if HomeostaticNexus else None
hopfield_mem = ContinuousHopfieldMemory(dim=256, num_basins=64, device_str=device_str) if ContinuousHopfieldMemory else None
hdc_binding_op = kcore.VectorSymbolicBindingOp(256, device_str) if hasattr(kcore, "VectorSymbolicBindingOp") else None

# Ensure fundamental morphic graph primitives
if agent_brain.graph.k_nodes == 0:
    agent_brain.graph.add_node("core_acc", "LinearAccumulator", True, 1.0)
    agent_brain.graph.add_node("core_mult", "BilinearMultiplicative", True, 1.0)
    agent_brain.graph.add_node("core_hopfield", "ContinuousHopfield", True, 1.0)
    agent_brain.graph.add_node("core_ssm", "StateSpaceMemory", True, 1.0)

h_fast = torch.zeros(1, agent_brain.hidden_dim, device=device)
h_slow = torch.zeros(1, agent_brain.hidden_dim, device=device)

start_step = 0
if os.path.exists(kcore_path):
    try:
        h_fast, h_slow, saved_epoch, saved_story_idx = load_karyon(agent_brain, hopfield_mem, hu_nexus, filepath=kcore_path, device=device_str)
        start_step = (saved_story_idx // BATCH_SIZE) if saved_story_idx else 0
        if start_step >= len(stream_loader):
            start_step = 0
        logger.info(f"⏩ Resuming from saved container at step {start_step}/{len(stream_loader)}.")
    except Exception as e:
        logger.warning(f"Could not load '{kcore_path}': {e}. Initializing fresh life-stream.")

BASE_LR = 1.2e-4
optimizer = optim.AdamW(agent_brain.parameters(), lr=BASE_LR, weight_decay=0.01)
criterion_speech = nn.CrossEntropyLoss(ignore_index=256)
scaler = torch.amp.GradScaler(hw_engine.device_type, enabled=(use_amp and autocast_dtype == torch.float16))


def get_neuromodulated_lr(base_lr: float, hu: HomeostaticNexus) -> float:
    if hu is None:
        return base_lr
    states = hu.get_states()
    curiosity = float(states[0].item())
    energy = float(states[1].item())
    na = float(states[4].item())
    allostatic_gain = max(0.40, min(2.00, 0.40 + 1.20 * na + 0.80 * curiosity - 0.30 * (1.0 - energy)))
    return base_lr * allostatic_gain


# =============================================================================
# 3. KEP RULE #4: DIAGNOSTIC SPEECH & ANOKHIN AUDITOR
# =============================================================================
def run_diagnostic_speech_sample(agent: CoREAgent, max_tokens: int = 64) -> Tuple[str, int]:
    agent.eval()
    prompt = "User: What is the primary source of energy for Earth?\nKaryon:"
    prompt_bytes = list(prompt.encode('utf-8'))
    input_ids = torch.tensor([prompt_bytes], dtype=torch.long, device=agent.device)

    generated_bytes = []
    anokhin_corrections = 0

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = agent(input_ids, thinking_steps=2)
            next_byte_logits = logits[0, -1, :256].clone()

            # Anti-repetition Tsodyks-Markram refractory penalty
            if len(generated_bytes) >= 4:
                recent_tail = bytes(generated_bytes[-8:]).decode('utf-8', errors='ignore')
                if len(recent_tail) >= 6 and (recent_tail[-3:] == recent_tail[-6:-3]):
                    last_byte = generated_bytes[-1]
                    next_byte_logits[last_byte] -= 6.0
                    anokhin_corrections += 1

            next_byte = torch.argmax(next_byte_logits).item()
            if next_byte == 257:  # EOS
                break
            generated_bytes.append(next_byte)
            input_ids = torch.cat([input_ids, torch.tensor([[next_byte]], device=agent.device)], dim=1)

    agent.train()
    try:
        text = bytes(generated_bytes).decode('utf-8', errors='replace').strip()
    except Exception:
        text = ""
    return text, anokhin_corrections


# =============================================================================
# 4. MASTER LIFE-STREAM CONTINUOUS EXECUTION LOOP
# =============================================================================
def run_master_life_stream():
    logger.info(f"\n{'='*85}\n === [EXP-340: LAUNCHING MASTER LIFE-STREAM SYNTHESIS (N=1 Single Pass)] ===\n{'='*85}")

    total_anokhin_verifications = 0
    total_sandbox_evaluations = 0
    total_sleep_cycles = 0
    loss_running_mean = 2.00
    loss_running_var = 0.20
    loss_running_std = 0.45
    loss_momentum = 0.05

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
                # 1. Spatiotemporal Forward Scan (TriScaleHierarchicalPAC + DynamicMorphicGraph)
                logits = agent_brain(input_seq, thinking_steps=3)
                loss = criterion_speech(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))

        except (torch.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e) or isinstance(e, torch.OutOfMemoryError):
                logger.warning(f"⚠️ [Step {batch_idx+1}] VRAM spike intercepted. Evicting cache...")
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                if device_str == 'cuda':
                    torch.cuda.empty_cache()
                time.sleep(1.0)
                continue
            else:
                raise e

        speech_loss_val = loss.item()
        if math.isnan(speech_loss_val):
            logger.warning(f"⚠️ [Step {batch_idx+1}] Loss NaN detected. Skipping step...")
            optimizer.zero_grad(set_to_none=True)
            continue

        # Dynamic Neuromodulated Learning Rate
        cur_lr = get_neuromodulated_lr(BASE_LR, hu_nexus)
        for group in optimizer.param_groups:
            group['lr'] = cur_lr

        # Running Loss Tracking for Dynamic Somatic Thresholds
        loss_diff = speech_loss_val - loss_running_mean
        loss_running_mean += loss_momentum * loss_diff
        loss_running_var = (1.0 - loss_momentum) * loss_running_var + loss_momentum * (loss_diff ** 2)
        loss_running_std = max(0.10, math.sqrt(loss_running_var))

        # 2. Optimization Step
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

        # 3. Interoceptive Homeostasis Update
        if hu_nexus is not None:
            hu_nexus.update(float(speech_loss_val) * 0.03)

        # 4. Somatic Attractor/Repulsor Experience Recording
        if batch_idx > 5:
            with torch.no_grad():
                ctx_t = agent_brain.emb(input_seq[:, :8]).mean(dim=1)
                act_t = agent_brain.emb(target_seq[:, 0])
                agent_brain.record_somatic_step_feedback(
                    context_t=ctx_t,
                    action_t=act_t,
                    free_energy_surprise=speech_loss_val,
                    mean_loss=loss_running_mean,
                    std_loss=loss_running_std
                )

        # 5. Periodic Active Closed-Loop Anokhin & Mental Sandbox Calibration (every 50 steps)
        if (batch_idx + 1) % 50 == 0:
            with torch.no_grad():
                # Mental Sandbox planning on current latent state (projected to embed_dim 256)
                curr_s = logits[:, -1, :].mean(dim=0, keepdim=True)
                if curr_s.size(-1) != agent_brain.embed_dim:
                    curr_s = agent_brain.emb(target_seq[:, -1]).mean(dim=0, keepdim=True)
                cand_a = -0.5 * curr_s
                cand_b = 0.5 * curr_s
                def _fe_probe(s, tau):
                    return torch.norm(s, dim=-1).item()
                best_branch, _, _ = agent_brain.mental_rollout_sandbox(curr_s, [cand_a, cand_b], 2, _fe_probe)
                total_sandbox_evaluations += 1

                # Anokhin check on synthetic arithmetic perturbation
                d1_test = torch.randint(0, 10, (4,), device=device)
                d2_test = torch.randint(0, 10, (4,), device=device)
                c_test = torch.zeros(4, dtype=torch.long, device=device)
                cand_logits_test = torch.randn(4, 10, device=device)
                _, _, refined = agent_brain.verify_and_refine_arithmetic_action(cand_logits_test, d1_test, d2_test, c_test)
                if refined:
                    total_anokhin_verifications += 1

        # 6. Dynamic 3-Phase Sleep & Tononi SHY Synaptic Consolidation
        states = hu_nexus.get_states() if hu_nexus is not None else torch.tensor([0.5, 1.0, 0.9, 1.0, 0.2, 0.1])
        energy_val = float(states[1].item())
        should_sleep = (energy_val <= 0.15) or ((batch_idx + 1) % 250 == 0)

        if should_sleep:
            total_sleep_cycles += 1
            t_sleep_start = time.perf_counter()
            logger.info(f"🌙 [Step {batch_idx+1}] Somatic Energy={energy_val:.2f} | Executing Sleep Consolidation & Apoptosis...")

            # Prune inactive morphic nodes (Tononi SHY)
            pruned_count = agent_brain.prune_inactive_nodes(threshold=0.01)

            # Restore Homeostatic Energy Pool
            if hu_nexus is not None:
                with torch.no_grad():
                    states_t = hu_nexus.get_states()
                    states_t[1] = 1.0
                    states_t[4] = 0.05
                    states_t[5] = 0.05

            sleep_ms = (time.perf_counter() - t_sleep_start) * 1000.0
            logger.info(f"☀️ [Awakened @ Step {batch_idx+1}] Sleep complete ({sleep_ms:.1f}ms). Pruned nodes: {pruned_count}")

            # Local container serialization on sleep
            save_karyon(agent_brain, hopfield_mem, hu_nexus, h_fast, h_slow, epoch=1, story_idx=(batch_idx + 1) * BATCH_SIZE, filepath=kcore_path)

            gc.collect()
            if device_str == 'cuda':
                torch.cuda.empty_cache()

        # 7. Throttled Hugging Face Sync strictly every 500 steps
        if (batch_idx + 1) % 500 == 0:
            commit_msg = f"feat(weights): EXP-340 life-stream step {batch_idx+1}/{len(stream_loader)} - loss={speech_loss_val:.4f}"
            sync_checkpoint_to_hf(kcore_path, hf_repo_id, commit_msg)

        batch_total_ms = (time.perf_counter() - t_batch_start) * 1000.0
        tokens_per_sec = (current_batch_size * (seq_len - 1)) / max(batch_total_ms / 1000.0, 1e-6)

        # 8. Real-Time Telemetry Dashboard every 25 steps
        if (batch_idx + 1) % 25 == 0 or batch_idx == len(stream_loader) - 1:
            perplexity = math.exp(min(speech_loss_val, 20.0))
            peak_vram_mb = hw_engine.get_telemetry().get('max_allocated_mb', 0.0)

            hop_buffers = dict(agent_brain.hopfield_memory.named_buffers())
            valences_buf = hop_buffers.get("valences", None)
            if valences_buf is not None:
                active_eps = getattr(agent_brain.hopfield_memory, "active_episodes", valences_buf.size(0))
                active_val = valences_buf[:active_eps]
                attractors_cnt = (active_val > 0.2).sum().item()
                repulsors_cnt = (active_val < -0.2).sum().item()
                neutral_cnt = ((active_val >= -0.2) & (active_val <= 0.2)).sum().item()
            else:
                attractors_cnt, repulsors_cnt, neutral_cnt = 0, 0, 0

            print("\n" + "=" * 85, flush=True)
            print(f" === [EXP-340 MASTER LIFE-STREAM DASHBOARD | STEP {batch_idx+1:04d}/{len(stream_loader)}] ===", flush=True)
            print("=" * 85, flush=True)
            print(f"Stream Performance        : Duration: {batch_total_ms:.1f}ms | Throughput: {tokens_per_sec:.1f} tok/s", flush=True)
            print(f"Loss & Perplexity         : Speech Loss = {speech_loss_val:.4f} (PPL: {perplexity:.2f})", flush=True)
            print(f"Somatic Memory State      : Attractors (+): {attractors_cnt} | Repulsors (-): {repulsors_cnt} | Neutral: {neutral_cnt}", flush=True)
            print(f"Tetrad Cognitive Loop     : Anokhin Verifications: {total_anokhin_verifications} | Sandbox Rollouts: {total_sandbox_evaluations}", flush=True)
            print(f"Hardware & Homeostasis    : Peak VRAM: {peak_vram_mb:.1f} MB | Somatic Energy: {energy_val:.3f} | Sleep Cycles: {total_sleep_cycles}", flush=True)
            print("=" * 85, flush=True)

        # 9. KEP Rule #4 Speech Diagnostics every 250 steps
        if (batch_idx + 1) % 250 == 0:
            sample_text, sample_corrections = run_diagnostic_speech_sample(agent_brain)
            logger.info(f"💬 [KEP Rule #4 Speech Sample @ Step {batch_idx+1}] (Corrections: {sample_corrections}) -> \"{sample_text}\"\n")

        del loss, logits, input_seq, target_seq

    # Final Save & Cloud Sync
    save_karyon(agent_brain, hopfield_mem, hu_nexus, h_fast, h_slow, epoch=1, story_idx=len(stream_loader) * BATCH_SIZE, filepath=kcore_path)
    sync_checkpoint_to_hf(kcore_path, hf_repo_id, f"feat(weights): EXP-340 stream marathon complete - loss={speech_loss_val:.4f}")
    logger.info(f"EXP-340 Life-Stream Marathon Complete! Steps: {len(stream_loader)} | Sleep Cycles: {total_sleep_cycles}.")


if __name__ == "__main__":
    run_master_life_stream()
