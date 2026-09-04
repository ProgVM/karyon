# experiments/exp_119_superhuman_sandbox_bench.py
"""
===============================================================================
EXP-119: Superhuman Parallel Latent Sandbox (K=16) & Harmonized Engine
===============================================================================
Hypothesis:
Replacing single-stream counterfactual rollout with a Massively Parallel Latent 
Mental Sandbox (K=16 trajectories rolled out concurrently on Tensor Cores) 
minimizing Expected Free Energy (G) prior to motor speech readout will:
1. Improve semantic coherence and lower speech perplexity (PPL).
2. Achieve 1-shot episodic consolidation without gradient cramming/overfitting.
3. Maintain high throughput (>15,000 tok/s) leveraging batched GPU operations.

Protocol: KEP v9.0 Scientific Protocol
Author: Bazilevs & Autonomous Lead AI Cyberneticist
===============================================================================
"""

import os
import sys
import time
import math
import json
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath('.'))

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import CorticalStage, HomeostaticUnit, BatchedEpisodicMemory, ByteTokenizer
from karyon_hardware import get_hardware_engine
from datasets import load_dataset


# =============================================================================
# SUPERHUMAN PARALLEL LATTENT MENTAL SANDBOX (K=16 BATCHED ROLLOUT)
# =============================================================================
class ParallelSuperhumanMentalSandbox(nn.Module):
    """
    Simulates K=16 mental counterfactual rollout trajectories concurrently 
    in latent space on Tensor Cores, evaluating Expected Free Energy (G)
    to select the optimal cognitive thought vector prior to motor output.
    """
    def __init__(self, hidden_dim: int, latent_dim: int, num_candidates: int = 16, device_str: str = 'cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_candidates = num_candidates
        self.device = torch.device(device_str)
        
        # Generator for candidate thought perturbations
        self.candidate_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * num_candidates)
        ).to(self.device)

    def parallel_rollout_search(self, world_model, h_curr: torch.Tensor, w_curr: torch.Tensor, steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Runs K=16 candidate thought rollouts concurrently.
        Returns:
            best_thought_h: (batch_size, hidden_dim) - selected optimal thought vector.
            min_efe: (batch_size, 1) - Expected Free Energy score.
            search_duration_ms: float - execution time in milliseconds.
        """
        t0 = time.perf_counter()
        b_size = h_curr.size(0)
        
        # 1. Generate K=16 candidate thought variations (batch_size, K, hidden_dim)
        cand_delta = self.candidate_proj(h_curr).view(b_size, self.num_candidates, self.hidden_dim)
        h_candidates = h_curr.unsqueeze(1) + 0.1 * cand_delta  # (B, K, H)
        
        # Flatten for batched GPU execution (B * K, H)
        h_sim = h_candidates.view(b_size * self.num_candidates, self.hidden_dim)
        w_sim = w_curr.unsqueeze(1).expand(b_size, self.num_candidates, -1).reshape(b_size * self.num_candidates, -1)
        
        accumulated_efe = torch.zeros(b_size * self.num_candidates, 1, device=self.device)
        
        # 2. Parallel Rollout across N steps
        for step in range(steps):
            w_pred, kl_div, fe_val, z_t = world_model(h_sim, h_sim, w_sim)
            accumulated_efe += fe_val
            w_sim = w_pred
            
        # Reshape EFE to (B, K)
        efe_matrix = accumulated_efe.view(b_size, self.num_candidates)
        
        # 3. Select ArgMin EFE candidate per batch item
        best_indices = torch.argmin(efe_matrix, dim=-1)  # (B,)
        
        # Extract best thought vectors
        batch_idx = torch.arange(b_size, device=self.device)
        best_thought_h = h_candidates[batch_idx, best_indices, :]  # (B, H)
        min_efe = efe_matrix[batch_idx, best_indices].unsqueeze(1) # (B, 1)
        
        search_time_ms = (time.perf_counter() - t0) * 1000.0
        return best_thought_h, min_efe, search_time_ms


# =============================================================================
# MAIN EXPERIMENTAL BENCHMARK
# =============================================================================
def run_experiment():
    print("=" * 80)
    print("STARTING EXP-119: SUPERHUMAN PARALLEL LATENT SANDBOX (K=16) BENCHMARK")
    print("=" * 80)

    hw = get_hardware_engine()
    device_str = 'cuda' if hw.is_cuda else ('xla' if hw.is_tpu else 'cpu')
    print(f"[Hardware] Active Device: {device_str}")

    config = CoREConfig()
    batch_size = 4
    seq_len = 128
    num_steps = 30

    print("[Data] Loading Alpaca-GPT4 dataset...")
    dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
    texts = [ex["instruction"] + " " + ex["output"] for ex in dataset.select(range(200))]

    encoded_batches = []
    for i in range(num_steps):
        batch_texts = [texts[(i * batch_size + b) % len(texts)] for b in range(batch_size)]
        inp_list = []
        tgt_list = []
        for t in batch_texts:
            b_ids = list(t.encode('utf-8'))[:seq_len + 1]
            if len(b_ids) < seq_len + 1:
                b_ids = b_ids + [32] * (seq_len + 1 - len(b_ids))
            inp_list.append(b_ids[:-1])
            tgt_list.append(b_ids[1:])
        encoded_batches.append((
            torch.tensor(inp_list, dtype=torch.long, device=device_str),
            torch.tensor(tgt_list, dtype=torch.long, device=device_str)
        ))

    val_inp = encoded_batches[0][0]
    val_tgt = encoded_batches[0][1]
    criterion = nn.CrossEntropyLoss()

    benchmark_results = {}

    # -------------------------------------------------------------------------
    # 1. BASELINE: Standard Single-Pass Execution
    # -------------------------------------------------------------------------
    print("\n--- Running Baseline: Standard Single-Pass Execution ---")
    torch.manual_seed(42)
    agent_base = CoREAgent(config, device=device_str).to(device_str)
    hu_base = HomeostaticUnit(batch_size=batch_size, device=device_str)
    opt_base = optim.AdamW(agent_base.get_all_parameters(), lr=1e-4)

    t0 = time.perf_counter()
    for inp_b, tgt_b in encoded_batches:
        opt_base.zero_grad()
        tot_loss, s_loss, _, _, _, _, _ = agent_base.forward_sequence(inp_b, tgt_b, hu_base, criterion)
        tot_loss.backward()
        opt_base.step()
    dur_base = time.perf_counter() - t0
    tok_s_base = (num_steps * batch_size * seq_len) / dur_base

    with torch.no_grad():
        _, val_loss_base, _, _, _, _, _ = agent_base.forward_sequence(val_inp, val_tgt, hu_base, criterion)

    print(f"[Baseline] Tok/s: {tok_s_base:.1f} | Val Loss: {val_loss_base:.4f} (PPL: {math.exp(val_loss_base):.2f})")
    benchmark_results["Baseline"] = {"tok_per_sec": tok_s_base, "val_loss": val_loss_base, "ppl": math.exp(val_loss_base)}

    # -------------------------------------------------------------------------
    # 2. PROPOSED: Superhuman Parallel Latent Sandbox (K=16 Candidates)
    # -------------------------------------------------------------------------
    print("\n--- Running Proposed: Superhuman Parallel Latent Sandbox (K=16) ---")
    torch.manual_seed(42)
    agent_sandbox = CoREAgent(config, device=device_str).to(device_str)
    hu_sandbox = HomeostaticUnit(batch_size=batch_size, device=device_str)
    mem_sandbox = BatchedEpisodicMemory(batch_size=batch_size, memory_dim=256, max_capacity=100, device=device_str)
    sandbox_engine = ParallelSuperhumanMentalSandbox(agent_sandbox.hidden_dim, agent_sandbox.latent_dim, num_candidates=16, device_str=device_str)
    
    opt_sandbox = optim.AdamW(list(agent_sandbox.get_all_parameters()) + list(sandbox_engine.parameters()), lr=1e-4)

    sandbox_timings = []

    t0 = time.perf_counter()
    for inp_b, tgt_b in encoded_batches:
        opt_sandbox.zero_grad()
        
        # Forward sequence pass to extract hidden states
        tot_loss, s_loss, fe_val, m_s2, h_p, u_t, eff_dt = agent_sandbox.forward_sequence(inp_b, tgt_b, hu_sandbox, criterion, episodic_memory=mem_sandbox)
        
        # Perform K=16 Parallel Mental Sandbox Search prior to final optimization
        w_curr = agent_sandbox.pos_embeddings(inp_b[:, -1:], start_pos=0, apply_rf=True).squeeze(1)
        best_thought_h, min_efe, search_ms = sandbox_engine.parallel_rollout_search(agent_sandbox.world_model, h_p, w_curr, steps=3)
        sandbox_timings.append(search_ms)
        
        # Optimize combined loss (Speech Loss + Minimum EFE from Sandbox)
        combined_loss = tot_loss + 0.05 * min_efe.mean()
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_sandbox.get_all_parameters(), max_norm=2.0)
        opt_sandbox.step()

        # 1-Shot Episodic Memory Write of the optimal thought candidate after backward
        with torch.no_grad():
            mem_sandbox.write(best_thought_h[:, :mem_sandbox.memory_dim].detach(), w_curr.detach())

    dur_sandbox = time.perf_counter() - t0
    tok_s_sandbox = (num_steps * batch_size * seq_len) / dur_sandbox
    avg_search_ms = sum(sandbox_timings) / len(sandbox_timings)

    with torch.no_grad():
        _, val_loss_sandbox, _, _, _, _, _ = agent_sandbox.forward_sequence(val_inp, val_tgt, hu_sandbox, criterion)

    print(f"[Parallel Sandbox K=16] Tok/s: {tok_s_sandbox:.1f} | Avg Rollout Time: {avg_search_ms:.2f}ms | Val Loss: {val_loss_sandbox:.4f} (PPL: {math.exp(val_loss_sandbox):.2f})")
    benchmark_results["Parallel Sandbox (K=16)"] = {
        "tok_per_sec": tok_s_sandbox,
        "val_loss": val_loss_sandbox,
        "ppl": math.exp(val_loss_sandbox),
        "avg_rollout_time_ms": avg_search_ms
    }

    # -------------------------------------------------------------------------
    # SUMMARY & JSON EXPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL EXPERIMENTAL BENCHMARK SUMMARY")
    print("=" * 80)
    for name, m in benchmark_results.items():
        print(f"• {name:30s} | Tok/s: {m['tok_per_sec']:8.1f} | Val Loss: {m['val_loss']:.4f} | PPL: {m['ppl']:.2f}")

    with open("exp_119_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

if __name__ == "__main__":
    run_experiment()
