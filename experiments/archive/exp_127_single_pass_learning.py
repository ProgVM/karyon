# experiments/exp_127_single_pass_learning.py
"""
===============================================================================
EXP-127: Single-Pass Continuous Allostatic Learning Stream Benchmark
===============================================================================
Hypothesis & KEP Protocol (Principle 2, KEP Rule #1, KEP Rule #7):
1. Single Continuous Pass (N=1, Zero Artificial Epochs):
   Karyon processes the stream of reality sequentially and continuously.
   Learning occurs online via error-gated predictive coding backprop + fast-weight
   plasticity without artificial data repetition.

2. Dynamic Allostatic Volitional Sleep 2.0:
   Somatic energy depletes as tokens are processed. When energy drops below 0.35
   or when native C++20 `VolitionalActionEvaluator` selects `INITIATE_SLEEP_CONSOLIDATION`,
   Karyon enters Biophysical Sleep 2.0 (NREM Replay + REM Dreaming + Synaptic Pruning),
   restores energy to 1.00, and awakens to continue the continuous stream.

Target Telemetry Metrics:
- Speech Loss Convergence (nats/byte) & Perplexity Delta
- Free Energy Trajectory ($F_t$)
- Throughput (tokens/sec) & Peak VRAM
- Sleep Consolidation Frequency & Energy Restoration Metrics

Protocol: KEP v9.0 Scientific Protocol
Author: Bazilevs & Autonomous Lead AI Cyberneticist
===============================================================================
"""

import os
import sys
import time
import json
import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath('.'))

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory
from karyon_hardware import get_hardware_engine


class ContinuousPackedDataset(Dataset):
    def __init__(self, hf_data, tokenizer, seq_len=1024, max_samples=500):
        self.seq_len = seq_len
        byte_chunks = []
        eos_arr = np.array([257], dtype=np.uint16)
        
        count = 0
        for item in hf_data:
            inst = item.get("instruction", "").strip()
            out = item.get("output", "").strip()
            if inst and out:
                dialog = f"User: {inst}\nKaryon: {out}"
                raw_b = dialog.encode('utf-8')
                arr = np.frombuffer(raw_b, dtype=np.uint8).astype(np.uint16)
                byte_chunks.append(arr)
                byte_chunks.append(eos_arr)
                count += 1
                if count >= max_samples:
                    break
                
        self.flat_stream = np.concatenate(byte_chunks)
        del byte_chunks
        gc.collect()

        self.num_blocks = len(self.flat_stream) // (seq_len + 1)

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        end = start + (self.seq_len + 1)
        return torch.from_numpy(self.flat_stream[start:end].astype(np.int64))

def collate_packed_fn(batch):
    return torch.stack(batch, dim=0)


def run_experiment():
    print("=" * 80)
    print("STARTING EXP-127: SINGLE-PASS CONTINUOUS ALLOSTATIC STREAM BENCHMARK")
    print("=" * 80)

    hw = get_hardware_engine()
    device_str = 'cuda' if hw.is_cuda else ('xla' if hw.is_tpu else 'cpu')
    device = torch.device(device_str)
    print(f"[Hardware Environment] Active Platform: {device_str}")

    tokenizer = ByteTokenizer()
    print("Loading Alpaca-GPT4 dataset subset for stream benchmark...")
    raw_dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")

    BATCH_SIZE = 4
    SEQ_LEN = 1024
    CHUNK_SIZE = 64

    stream_dataset = ContinuousPackedDataset(raw_dataset, tokenizer, seq_len=SEQ_LEN, max_samples=300)
    stream_loader = DataLoader(
        stream_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # Single Continuous Pass
        collate_fn=collate_packed_fn,
        drop_last=True
    )

    print(f"Stream Dataset Ready: {len(stream_dataset)} Blocks | {len(stream_loader)} Batches")

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.unified_dim = 256
    config.net.hidden_dim = 512
    config.net.expand_dim = 2048
    config.net.num_heads = 8
    config.train.batch_size = BATCH_SIZE
    config.train.chunk_size = CHUNK_SIZE

    agent = CoREAgent(config, device=device_str).to(device)
    hu = HomeostaticUnit(batch_size=BATCH_SIZE, device=device_str)
    episodic_mem = BatchedEpisodicMemory(batch_size=BATCH_SIZE, memory_dim=config.net.unified_dim, max_capacity=500, device=device_str)

    # Initial somatic state
    hu.state[:, 0] = 0.80 # Curiosity
    hu.state[:, 1] = 0.85 # Initial Energy

    optimizer = optim.AdamW(agent.get_all_parameters(), lr=1e-3, weight_decay=0.01)
    criterion_speech = nn.CrossEntropyLoss(ignore_index=256)
    use_amp = hw.config.enable_amp and not hw.is_cpu
    scaler = torch.amp.GradScaler(hw.device_type, enabled=use_amp)

    initial_loss = 0.0
    final_loss = 0.0
    step_telemetry = []
    total_sleep_cycles = 0

    print("\n--- Executing 30 Stream Steps in Single Pass ---")

    t_start_total = time.perf_counter()

    for step_idx, batch_tokens in enumerate(stream_loader):
        if step_idx >= 30:
            break

        t_step_start = time.perf_counter()
        batch_tokens = batch_tokens.to(device)

        input_seq = batch_tokens[:, :-1]
        target_seq = batch_tokens[:, 1:]

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            total_loss_tensor, speech_loss_val, fe_val, m_curr, h_curr, curr_u_t, eff_dt = agent.forward_sequence(
                input_seq, target_seq, hu, criterion_speech, episodic_memory=episodic_mem,
                loss_free_energy_weight=0.05, chunk_size=CHUNK_SIZE, use_checkpointing=False
            )

        if step_idx == 0:
            initial_loss = speech_loss_val

        # Update Somatic Homeostasis
        action_cost_tensor = torch.full((BATCH_SIZE, 1), 0.015, device=device) # Deplete energy per step
        pred_err_tensor = torch.full((BATCH_SIZE, 1), float(speech_loss_val * 0.1), device=device)
        entropy_tensor = torch.full((BATCH_SIZE, 1), float(fe_val), device=device)
        cog_act_tensor = torch.zeros((BATCH_SIZE, 1), dtype=torch.int64, device=device)
        hu.update(action_cost_tensor, pred_err_tensor, entropy_tensor, cog_act_tensor)

        energy_val = hu.state[0, 1].item()
        curiosity_val = hu.state[0, 0].item()

        # Volitional C++20 Sleep Trigger
        action_idx = agent.efe_action_evaluator.select_volitional_action(h_curr[0:1], curiosity_val, energy_val)
        should_sleep = (energy_val <= 0.35) or (action_idx == 2)

        sleep_occurred = False
        if should_sleep:
            total_sleep_cycles += 1
            sleep_occurred = True
            _ = agent.execute_deep_allostatic_sleep(episodic_mem, hu, num_replay_cycles=2, downscaling_factor=0.02)

        # Optimization Step
        scaler.scale(total_loss_tensor).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(agent.get_all_parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()

        step_total_ms = (time.perf_counter() - t_step_start) * 1000.0
        tok_per_sec = (BATCH_SIZE * (SEQ_LEN - 1)) / (step_total_ms / 1000.0)

        final_loss = speech_loss_val

        print(f"[Step {step_idx+1:02d}] Loss: {speech_loss_val:.4f} | FE: {fe_val:.4f} | Energy: {energy_val:.2f} | Sleep: {sleep_occurred} | Speed: {tok_per_sec:.1f} tok/s")

        step_telemetry.append({
            "step": step_idx + 1,
            "loss": speech_loss_val,
            "free_energy": fe_val,
            "energy": energy_val,
            "sleep_occurred": sleep_occurred,
            "tok_per_sec": tok_per_sec
        })

    total_duration_sec = time.perf_counter() - t_start_total
    loss_delta = initial_loss - final_loss
    initial_ppl = math.exp(min(initial_loss, 20.0))
    final_ppl = math.exp(min(final_loss, 20.0))

    print("\n" + "=" * 80)
    print("EXP-127 SINGLE-PASS CONTINUOUS BENCHMARK RESULTS")
    print("=" * 80)
    print(f"• Initial Speech Loss (Step 1) : {initial_loss:.4f} (PPL: {initial_ppl:.2f})")
    print(f"• Final Speech Loss (Step 30): {final_loss:.4f} (PPL: {final_ppl:.2f})")
    print(f"• Loss Delta (Improvement)    : {loss_delta:+.4f} nats/byte")
    print(f"• Total Sleep Cycles Executed : {total_sleep_cycles}")
    print(f"• Average Throughput          : {sum(s['tok_per_sec'] for s in step_telemetry)/len(step_telemetry):.1f} tok/s")
    print(f"• Total Duration              : {total_duration_sec:.2f} s")
    print("=" * 80)

    results = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_delta": loss_delta,
        "initial_ppl": initial_ppl,
        "final_ppl": final_ppl,
        "total_sleep_cycles": total_sleep_cycles,
        "avg_tok_per_sec": sum(s['tok_per_sec'] for s in step_telemetry)/len(step_telemetry),
        "total_duration_sec": total_duration_sec,
        "telemetry": step_telemetry
    }

    with open("exp_127_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_experiment()
