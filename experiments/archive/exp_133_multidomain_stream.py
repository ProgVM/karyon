# experiments/exp_133_multidomain_stream.py
"""
===============================================================================
EXP-133: MULTI-DOMAIN DIVERSIFIED STREAM LEARNING & INSTRUCTION GROUNDING
===============================================================================
Hypothesis:
1. Multi-Domain Interleaved Cognitive Substrate:
   Training Karyon on an interleaved mixture of 4 diverse domains:
     - Alpaca-GPT4 (Conversational & General Knowledge)
     - Databricks-Dolly-15k (Multi-Task Instruction Following: extraction, summarization, QA)
     - Python Code Instructions 18k (Algorithmic & Syntactic Rigor)
     - GSM8k (Step-by-Step Mathematical & Deductive Chain-of-Thought)
   dramatically expands linguistic entropy and breaks stereotyped sub-syllable attractor basins.
2. Single-Pass Allostatic Continual Adaptation:
   Continuous stream unrolling ($S=1024, B=16$) with homeostatic wake-sleep consolidation (SHY synaptic downscaling)
   and Dynamic Free-Energy Thresholding (DFET) plasticity gating allows seamless cross-domain learning without catastrophic forgetting.
3. Diagnostic Speech Evaluation:
   Periodically testing multi-domain prompts (Code, Math, General QA, Repetition/Instruction following).
===============================================================================
"""

import os
import sys
import gc
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

sys.path.append("/kaggle/working/karyon")

import karyon_config
import karyon_core
import karyon_agent
import karyon_checkpoint
from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory, ByteTokenizer
from karyon_checkpoint import load_karyon, save_karyon

def build_multi_domain_stream(seq_len=1024, max_samples_per_domain=12000):
    print("[EXP-133 Dataset] Loading Multi-Domain Corpora...")
    tokenizer = ByteTokenizer()
    eos_arr = np.array([257], dtype=np.uint16)
    all_chunks = []

    # Domain 1: General Conversation (Alpaca-GPT4)
    print(" -> Loading Alpaca-GPT4 (General Knowledge)...")
    try:
        ds_alpaca = load_dataset("vicgalle/alpaca-gpt4", split=f"train[:{max_samples_per_domain}]")
        for item in ds_alpaca:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if inst and out:
                full_in = f"{inst}\nContext: {inp}" if inp else inst
                dialog = f"User: {full_in}\nKaryon: {out}"
                raw_b = dialog.encode('utf-8')
                all_chunks.append(np.frombuffer(raw_b, dtype=np.uint8).astype(np.uint16))
    except Exception as e:
        print(f"Warning loading Alpaca: {e}")

    # Domain 2: Instruction Types (Databricks-Dolly-15k)
    print(" -> Loading Databricks-Dolly-15k (Multi-Task Instructions)...")
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
                all_chunks.append(np.frombuffer(raw_b, dtype=np.uint8).astype(np.uint16))
    except Exception as e:
        print(f"Warning loading Dolly: {e}")

    # Domain 3: Algorithmic & Code Logic (Python Code Instructions 18k)
    print(" -> Loading Python Code Instructions 18k (Algorithmic Logic)...")
    try:
        ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split=f"train[:{max_samples_per_domain}]")
        for item in ds_code:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if inst and out:
                full_in = f"{inst}\n{inp}" if inp else inst
                dialog = f"User: {full_in}\nKaryon: {out}"
                raw_b = dialog.encode('utf-8')
                all_chunks.append(np.frombuffer(raw_b, dtype=np.uint8).astype(np.uint16))
    except Exception as e:
        print(f"Warning loading Code: {e}")

    # Domain 4: Deductive Reasoning & Chain-of-Thought (GSM8k)
    print(" -> Loading GSM8k (Step-by-Step Reasoning)...")
    try:
        ds_gsm = load_dataset("gsm8k", "main", split="train")
        for item in ds_gsm:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q and a:
                dialog = f"User: Solve step-by-step: {q}\nKaryon: {a}"
                raw_b = dialog.encode('utf-8')
                all_chunks.append(np.frombuffer(raw_b, dtype=np.uint8).astype(np.uint16))
    except Exception as e:
        print(f"Warning loading GSM8k: {e}")

    print(f"[EXP-133 Dataset] Total Raw Interleaved Dialogues Gathered: {len(all_chunks)}")
    random.seed(42)
    random.shuffle(all_chunks)

    # Interleave with EOS tokens
    packed_pieces = []
    for chunk in all_chunks:
        packed_pieces.append(chunk)
        packed_pieces.append(eos_arr)

    flat_stream = np.concatenate(packed_pieces)
    del all_chunks, packed_pieces
    gc.collect()

    total_bytes = len(flat_stream)
    num_blocks = total_bytes // (seq_len + 1)
    print(f"[EXP-133 Dataset] Total Packed Bytes: {total_bytes / (1024*1024):.2f} MB | Sequence Blocks (S={seq_len}): {num_blocks}")
    return flat_stream, num_blocks

class MultiDomainStreamDataset(Dataset):
    def __init__(self, flat_stream, num_blocks, seq_len=1024):
        self.flat_stream = flat_stream
        self.num_blocks = num_blocks
        self.seq_len = seq_len

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        end = start + (self.seq_len + 1)
        return torch.from_numpy(self.flat_stream[start:end].astype(np.int64))

def run_experiment():
    print("=" * 80)
    print("STARTING EXP-133: MULTI-DOMAIN DIVERSIFIED STREAM LEARNING")
    print("=" * 80)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    config = CoREConfig()
    config.train.batch_size = 16
    config.train.chunk_size = 64
    config.train.seq_len = 1024

    BATCH_SIZE = config.train.batch_size
    SEQ_LEN = config.train.seq_len
    CHUNK_SIZE = config.train.chunk_size

    # 1. Prepare Multi-Domain Stream
    flat_stream, num_blocks = build_multi_domain_stream(seq_len=SEQ_LEN, max_samples_per_domain=12000)
    dataset = MultiDomainStreamDataset(flat_stream, num_blocks, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=2, pin_memory=True)

    # 2. Initialize Agent and Homeostasis
    hu = HomeostaticUnit(batch_size=BATCH_SIZE, device=device_str)
    memory = BatchedEpisodicMemory(batch_size=BATCH_SIZE, memory_dim=config.net.unified_dim, max_capacity=1000, device=device_str)
    agent = CoREAgent(config, device=device_str).to(device)

    kcore_path = "karyon_soul.kcore"
    if os.path.exists(kcore_path):
        load_karyon(agent, memory, hu, filepath=kcore_path, device=device_str)
        print(f"[EXP-133] Loaded existing entity from {kcore_path}")

    optimizer = optim.AdamW(agent.get_all_parameters(), lr=1.5e-3, weight_decay=0.01)
    criterion_speech = nn.CrossEntropyLoss(ignore_index=256)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    total_steps = len(loader)
    print(f"[EXP-133] Total Single-Pass Batches to Process: {total_steps} (Batch Size = {BATCH_SIZE})")

    moving_fe = 0.10
    total_adapted = 0
    total_sleeps = 0
    start_time = time.time()
    total_tokens_processed = 0

    agent.train()

    # Baseline speech check before training
    print("\n--- BASELINE GENERATION BEFORE EXP-133 STREAM ---")
    diag_hu = HomeostaticUnit(batch_size=1, device=device_str)
    diag_gen = agent.generate_thought_and_speech(
        "User: What is 12 + 15?\nKaryon:",
        m_state=torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device),
        h_state=torch.zeros(1, agent.hidden_dim, device=device),
        hu=diag_hu,
        episodic_memory=None,
        config=config,
        max_generated_tokens=40,
        temperature=0.30,
        top_p=0.90
    )
    base_chars = [ev["text"] for ev in diag_gen if ev["status"] == "token"]
    print("Baseline Answer:", repr("".join(base_chars)))

    # Process first 600 multi-domain batches for benchmark evaluation
    benchmark_batches = min(600, total_steps)
    print(f"\nExecuting Single-Pass Multi-Domain Stream ({benchmark_batches} batches)...")

    for step, batch in enumerate(loader):
        if step >= benchmark_batches:
            break

        batch = batch.to(device)
        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            total_loss, speech_loss_val, fe_loss_val, fe, pred_logits, action_indices, _ = agent.forward_sequence(
                input_seq=input_ids,
                target_seq=target_ids,
                hu_batch=hu,
                criterion_speech=criterion_speech,
                episodic_memory=memory,
                loss_free_energy_weight=0.05,
                chunk_size=CHUNK_SIZE
            )

        fe_val = fe.mean().item() if isinstance(fe, torch.Tensor) else float(fe)
        moving_fe = 0.95 * moving_fe + 0.05 * fe_val

        # Dynamic Plasticity Gating: Update synaptic weights when surprise exceeds mastery setpoint
        if fe_val > 0.005:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(agent.get_all_parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_adapted += 1

        # Update Homeostasis
        with torch.no_grad():
            energy_cost = torch.full((BATCH_SIZE, 1), 0.0008, device=device)
            zero_err = torch.zeros(BATCH_SIZE, 1, device=device)
            hu.update(energy_cost, zero_err, zero_err, action_indices)

            # Homeostatic Sleep Consolidation Trigger
            mean_energy = hu.state[:, 1].mean().item()
            if mean_energy < 0.25:
                hu.state[:, 1] = 1.0 # Restore Energy
                hu.state[:, 4] = 0.10 # Lower Noradrenaline
                total_sleeps += 1

        total_tokens_processed += (BATCH_SIZE * SEQ_LEN)

        if (step + 1) % 50 == 0 or (step + 1) == benchmark_batches:
            elapsed = time.time() - start_time
            tok_s = total_tokens_processed / max(1.0, elapsed)
            vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            ppl = math.exp(min(10.0, speech_loss_val))
            print(f"[Step {step+1:04d}/{benchmark_batches}] Loss: {speech_loss_val:.4f} | PPL: {ppl:.2f} | FE: {fe_val:.5f} | Adapt: {total_adapted} | Sleep: {total_sleeps} | Speed: {tok_s:.0f} tok/s | VRAM: {vram:.1f} MB")

    total_time = time.time() - start_time
    avg_speed = total_tokens_processed / max(1.0, total_time)

    # 4. Diagnostic Multi-Domain Speech Probing
    print("\n" + "=" * 80)
    print("POST-EXP-133 MULTI-DOMAIN DIAGNOSTIC TEXT SAMPLING")
    print("=" * 80)

    test_queries = [
        "User: What is 12 + 15?\nKaryon:",
        "User: Write a Python function to add two numbers.\nKaryon:",
        "User: Repeat after me: hello world\nKaryon:",
        "User: What is the capital of France?\nKaryon:"
    ]

    outputs = []
    for q in test_queries:
        diag_hu = HomeostaticUnit(batch_size=1, device=device_str)
        gen = agent.generate_thought_and_speech(
            q,
            m_state=torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device),
            h_state=torch.zeros(1, agent.hidden_dim, device=device),
            hu=diag_hu,
            episodic_memory=memory,
            config=config,
            max_generated_tokens=60,
            temperature=0.30,
            top_p=0.90
        )
        ch_list = [ev["text"] for ev in gen if ev["status"] == "token"]
        res = "".join(ch_list)
        outputs.append(res)
        print(f"\nQuery:\n  {q.strip()}\nResponse:\n  {repr(res)}")

    # Calculate 3-Gram Diversity across all outputs
    full_sample_text = " ".join(outputs)
    def n_gram_diversity(text, n=3):
        if len(text) < n:
            return 1.0
        ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
        return len(set(ngrams)) / float(len(ngrams))

    overall_div = n_gram_diversity(full_sample_text)
    print(f"\nOverall Multi-Domain 3-Gram Lexical Diversity: {overall_div:.4f}")

    # Persist updated soul
    h_fast_save = torch.zeros(1, agent.hidden_dim, device=device)
    h_slow_save = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device)
    save_karyon(agent, memory, hu, h_fast_save, h_slow_save, epoch=1, story_idx=benchmark_batches, filepath=kcore_path)

    metrics = {
        "final_speech_loss": speech_loss_val,
        "final_ppl": math.exp(min(10.0, speech_loss_val)),
        "final_free_energy": fe_val,
        "avg_tok_per_sec": avg_speed,
        "adapted_batches": total_adapted,
        "sleep_cycles": total_sleeps,
        "3gram_diversity": overall_div
    }

    verdict = "🟢 POSITIVE" if (speech_loss_val < 3.5 and overall_div > 0.40) else "⚪ NEUTRAL"
    print(f"\n[EXP-133 VERDICT]: {verdict}")
    print("=" * 80)
    return metrics, verdict

if __name__ == "__main__":
    run_experiment()
