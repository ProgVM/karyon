# experiments/exp_118_living_cognitive_engine_tritest.py
"""
===============================================================================
EXP-118: Living Cognitive Engine Tri-Test Benchmark
===============================================================================
Hypothesis:
Evaluating the 3 developmental vectors individually and in combination on real Alpaca-GPT4 data:
1. Vector 1 (3-Stage Laminar Hierarchy: Morpho-Syntactic -> Semantic -> Episodic-Intent)
2. Vector 2 (Decoupled Local Predictive Plasticity: Zero Global BPTT)
3. Vector 3 (Allostatic SWR Replay & SHY Synaptic Downscaling)
4. Combined Architecture (Vectors 1 + 2 + 3 Integrated)

Protocol: KEP v9.0 Scientific Protocol
Author: Bazilevs & Autonomous Lead AI Cyberneticist
===============================================================================
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath('.'))

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import CorticalStage, HomeostaticUnit, BatchedEpisodicMemory, ByteTokenizer
from karyon_hardware import get_hardware_engine
from datasets import load_dataset


# =============================================================================
# VECTOR 1: 3-Stage Cortical Laminar Stack Module
# =============================================================================
class ThreeStageLaminarStack(nn.Module):
    def __init__(self, hidden_dim: int, expand_dim: int, num_heads: int, head_k: int, head_v: int, device_str: str):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stage1 = CorticalStage(hidden_dim, expand_dim, num_heads, head_k, head_v, min_beta=0.005, max_beta=0.15, swiglu_kernel_size=3, device=device_str)
        self.stage2 = CorticalStage(hidden_dim, expand_dim, num_heads, head_k, head_v, min_beta=0.0001, max_beta=0.05, swiglu_kernel_size=7, device=device_str)
        self.stage3 = CorticalStage(hidden_dim, expand_dim, num_heads, head_k, head_v, min_beta=0.00001, max_beta=0.005, swiglu_kernel_size=15, device=device_str)
        
        self.topdown_2to1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)).to(device_str)
        self.topdown_3to2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)).to(device_str)

    def forward(self, h_in, m1, m2, m3, u_t, sal_gate):
        # Stage 1: Fast Morpho-Syntactic
        h1, m1_next, _ = self.stage1(h_in, m1, u_t, sal_gate, 1.0)
        
        # Stage 2: Medium Semantic-Discourse
        topdown_1 = self.topdown_2to1(h1)
        eps_1 = h1 - topdown_1
        h2, m2_next, _ = self.stage2(eps_1, m2, u_t, sal_gate, 1.0)
        
        # Stage 3: Ultra-Slow Episodic-Intent
        topdown_2 = self.topdown_3to2(h2)
        eps_2 = h2 - topdown_2
        h3, m3_next, _ = self.stage3(eps_2, m3, u_t, sal_gate, 1.0)
        
        h_fused = h1 + h2 + 0.5 * h3
        return h_fused, m1_next, m2_next, m3_next


# =============================================================================
# VECTOR 2 & COMBINED ENGINE IMPLEMENTATION
# =============================================================================
class LocalPlasticity3StageEngine:
    def __init__(self, agent: CoREAgent, stack: ThreeStageLaminarStack, lr: float = 1e-3):
        self.agent = agent
        self.stack = stack
        self.lr = lr
        self.opt_s1 = optim.Adam(self.stack.stage1.parameters(), lr=lr)
        self.opt_s2 = optim.Adam(self.stack.stage2.parameters(), lr=lr)
        self.opt_s3 = optim.Adam(self.stack.stage3.parameters(), lr=lr)
        self.opt_readout = optim.Adam(
            list(self.agent.in_proj.parameters()) +
            list(self.agent.volitional_head.parameters()) +
            list(self.agent.attractor_head.parameters()),
            lr=lr
        )

    def local_step(self, inp_tokens: torch.Tensor, tgt_tokens: torch.Tensor, hu: HomeostaticUnit):
        device = self.agent.device
        b_size, seq_len = inp_tokens.shape
        u_t = hu.state
        
        m1 = torch.zeros(b_size, self.agent.num_heads, self.agent.head_k, self.agent.head_v, device=device)
        m2 = torch.zeros(b_size, self.agent.num_heads, self.agent.head_k, self.agent.head_v, device=device)
        m3 = torch.zeros(b_size, self.agent.num_heads, self.agent.head_k, self.agent.head_v, device=device)
        
        self.opt_s1.zero_grad()
        self.opt_s2.zero_grad()
        self.opt_s3.zero_grad()
        self.opt_readout.zero_grad()

        emb = self.agent.pos_embeddings(inp_tokens, start_pos=0, apply_rf=True)
        h_in = self.agent.in_proj(emb)
        
        # Local forward passes with detached inter-layer boundaries
        h1, _, _ = self.stack.stage1(h_in.detach(), m1, u_t, torch.Tensor(), 1.0)
        sal_gate = self.agent.boundary_detector(h1.detach(), inp_tokens)
        
        h2, _, _ = self.stack.stage2(h1.detach(), m2, u_t, sal_gate, 1.0)
        h3, _, _ = self.stack.stage3(h2.detach(), m3, u_t, sal_gate, 1.0)
        
        # Local predictive loss updates
        loss_s1 = torch.mean((h1 - self.stack.topdown_2to1(h2.detach()))**2)
        loss_s1.backward()
        self.opt_s1.step()
        
        loss_s2 = torch.mean((h2 - self.stack.topdown_3to2(h3.detach()))**2)
        loss_s2.backward()
        self.opt_s2.step()
        
        # Final Readout loss
        h_comb = h1.detach() + h2.detach() + 0.5 * h3.detach()
        h_flat = h_comb.view(b_size * seq_len, -1)
        h_rel, _ = self.agent.attractor_head.relax_to_minima(h_flat, u_t)
        logits = self.agent.volitional_head.compute_volitional_logits(h_rel, u_t, self.agent.pos_embeddings.byte_embed.weight)
        
        loss_fn = nn.CrossEntropyLoss()
        speech_loss = loss_fn(logits.view(-1, self.agent.text_gen_dim), tgt_tokens.view(-1))
        speech_loss.backward()
        self.opt_readout.step()
        self.opt_s3.step()

        # Somatic Allostatic Update
        dummy_cost = torch.tensor([[0.01]], device=device).expand(b_size, 1)
        err_tensor = torch.tensor([[speech_loss.item()]], device=device).expand(b_size, 1)
        cog_act = torch.tensor([[0]], device=device).expand(b_size, 1)
        hu.update(dummy_cost, err_tensor, err_tensor, cog_act)

        return speech_loss.item()


# =============================================================================
# VECTOR 3: ALLOSTATIC SWR REPLAY & SHY DOWNSCALING
# =============================================================================
def execute_allostatic_shy_sleep(agent: CoREAgent, hu: HomeostaticUnit, episodic_mem: BatchedEpisodicMemory, shy_decay: float = 0.02):
    """
    Executes Adaptive Allostatic Sleep Consolidation:
    1. Replays high-surprise memories from episodic memory.
    2. Applies Tononi Synaptic Homeostasis (SHY) downscaling to prevent metabolic bloat.
    """
    device = agent.device
    na_t = hu.state[0, 4].item()
    f_ema = 0.15
    sleep_trigger = max(0.05, min(0.45, 0.20 * (1.0 - 0.5 * na_t) + 0.1 * f_ema))
    curr_energy = hu.state[0, 1].item()
    
    if curr_energy <= sleep_trigger:
        # Replay phase
        active_slots = getattr(episodic_mem, 'max_active_cpu', 0)
        if active_slots >= 3:
            rand_idx = torch.randint(0, active_slots, (min(4, active_slots),), device=device)
            keys = episodic_mem.keys[0, rand_idx, :]
            vals = episodic_mem.values[0, rand_idx, :]
            with torch.no_grad():
                h_dummy = torch.zeros(keys.size(0), agent.hidden_dim, device=device)
                agent.world_model(h_dummy, h_dummy, keys)
        
        # Tononi SHY Synaptic Downscaling (proportional relaxation of non-critical weights)
        with torch.no_grad():
            for p in agent.parameters():
                if p.requires_grad and p.dim() > 1:
                    p.mul_(1.0 - shy_decay)
        
        # Restore somatic energy
        hu.state[:, 1] = 1.0


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================
def run_experiment():
    print("=" * 80)
    print("STARTING EXP-118: LIVING COGNITIVE ENGINE TRI-TEST BENCHMARK")
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
    # 0. BASELINE: 2-Stage Standard Autograd BPTT
    # -------------------------------------------------------------------------
    print("\n--- Testing Baseline: 2-Stage Standard Autograd BPTT ---")
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
    # 1. VECTOR 1 STANDALONE: 3-Stage Laminar Stack (Global Autograd)
    # -------------------------------------------------------------------------
    print("\n--- Testing Vector 1: 3-Stage Laminar Stack ---")
    torch.manual_seed(42)
    agent_v1 = CoREAgent(config, device=device_str).to(device_str)
    stack_v1 = ThreeStageLaminarStack(agent_v1.hidden_dim, agent_v1.expand_dim, agent_v1.num_heads, agent_v1.head_k, agent_v1.head_v, device_str)
    hu_v1 = HomeostaticUnit(batch_size=batch_size, device=device_str)
    opt_v1 = optim.AdamW(list(agent_v1.get_all_parameters()) + list(stack_v1.parameters()), lr=1e-4)

    t0 = time.perf_counter()
    for inp_b, tgt_b in encoded_batches:
        opt_v1.zero_grad()
        u_t = hu_v1.state
        m1 = torch.zeros(batch_size, agent_v1.num_heads, agent_v1.head_k, agent_v1.head_v, device=device_str)
        m2 = torch.zeros(batch_size, agent_v1.num_heads, agent_v1.head_k, agent_v1.head_v, device=device_str)
        m3 = torch.zeros(batch_size, agent_v1.num_heads, agent_v1.head_k, agent_v1.head_v, device=device_str)
        
        emb = agent_v1.pos_embeddings(inp_b, start_pos=0, apply_rf=True)
        h_in = agent_v1.in_proj(emb)
        sal_gate = agent_v1.boundary_detector(h_in, inp_b)
        h_fused, _, _, _ = stack_v1(h_in, m1, m2, m3, u_t, sal_gate)
        
        h_flat = h_fused.view(batch_size * seq_len, -1)
        h_rel, _ = agent_v1.attractor_head.relax_to_minima(h_flat, u_t)
        logits = agent_v1.volitional_head.compute_volitional_logits(h_rel, u_t, agent_v1.pos_embeddings.byte_embed.weight)
        
        loss = criterion(logits.view(-1, agent_v1.text_gen_dim), tgt_b.view(-1))
        loss.backward()
        opt_v1.step()

    dur_v1 = time.perf_counter() - t0
    tok_s_v1 = (num_steps * batch_size * seq_len) / dur_v1
    
    with torch.no_grad():
        emb = agent_v1.pos_embeddings(val_inp, start_pos=0, apply_rf=True)
        h_in = agent_v1.in_proj(emb)
        sal_gate = agent_v1.boundary_detector(h_in, val_inp)
        h_fused, _, _, _ = stack_v1(h_in, m1, m2, m3, hu_v1.state, sal_gate)
        h_rel, _ = agent_v1.attractor_head.relax_to_minima(h_fused.view(batch_size * seq_len, -1), hu_v1.state)
        logits = agent_v1.volitional_head.compute_volitional_logits(h_rel, hu_v1.state, agent_v1.pos_embeddings.byte_embed.weight)
        val_loss_v1 = criterion(logits.view(-1, agent_v1.text_gen_dim), val_tgt.view(-1)).item()

    print(f"[Vector 1 (3-Stage)] Tok/s: {tok_s_v1:.1f} | Val Loss: {val_loss_v1:.4f} (PPL: {math.exp(val_loss_v1):.2f})")
    benchmark_results["Vector 1 (3-Stage Stack)"] = {"tok_per_sec": tok_s_v1, "val_loss": val_loss_v1, "ppl": math.exp(val_loss_v1)}

    # -------------------------------------------------------------------------
    # 2. VECTOR 2 STANDALONE: Decoupled Local Predictive Plasticity (No-BPTT)
    # -------------------------------------------------------------------------
    print("\n--- Testing Vector 2: Decoupled Local Predictive Plasticity ---")
    torch.manual_seed(42)
    agent_v2 = CoREAgent(config, device=device_str).to(device_str)
    hu_v2 = HomeostaticUnit(batch_size=batch_size, device=device_str)
    
    opt_v2_s1 = optim.Adam(agent_v2.stage1.parameters(), lr=1e-3)
    opt_v2_s2 = optim.Adam(agent_v2.stage2.parameters(), lr=1e-3)
    opt_v2_readout = optim.Adam(
        list(agent_v2.in_proj.parameters()) + list(agent_v2.volitional_head.parameters()) + list(agent_v2.attractor_head.parameters()),
        lr=1e-3
    )

    t0 = time.perf_counter()
    for inp_b, tgt_b in encoded_batches:
        opt_v2_s1.zero_grad()
        opt_v2_s2.zero_grad()
        opt_v2_readout.zero_grad()
        
        u_t = hu_v2.state
        m1 = torch.zeros(batch_size, agent_v2.num_heads, agent_v2.head_k, agent_v2.head_v, device=device_str)
        m2 = torch.zeros(batch_size, agent_v2.num_heads, agent_v2.head_k, agent_v2.head_v, device=device_str)
        
        emb = agent_v2.pos_embeddings(inp_b, start_pos=0, apply_rf=True)
        h_in = agent_v2.in_proj(emb)
        
        h1, _, _ = agent_v2.stage1(h_in.detach(), m1, u_t, torch.Tensor(), 1.0)
        sal_gate = agent_v2.boundary_detector(h1.detach(), inp_b)
        h2, _, _ = agent_v2.stage2(h1.detach(), m2, u_t, sal_gate, 1.0)
        
        loss_s1 = torch.mean((h1 - agent_v2.topdown_prior_proj(h2.detach()))**2)
        loss_s1.backward()
        opt_v2_s1.step()
        
        h_comb = h1.detach() + h2.detach()
        h_flat = h_comb.view(batch_size * seq_len, -1)
        h_rel, _ = agent_v2.attractor_head.relax_to_minima(h_flat, u_t)
        logits = agent_v2.volitional_head.compute_volitional_logits(h_rel, u_t, agent_v2.pos_embeddings.byte_embed.weight)
        
        loss_readout = criterion(logits.view(-1, agent_v2.text_gen_dim), tgt_b.view(-1))
        loss_readout.backward()
        opt_v2_readout.step()
        opt_v2_s2.step()

    dur_v2 = time.perf_counter() - t0
    tok_s_v2 = (num_steps * batch_size * seq_len) / dur_v2
    
    with torch.no_grad():
        emb = agent_v2.pos_embeddings(val_inp, start_pos=0, apply_rf=True)
        h_in = agent_v2.in_proj(emb)
        h1, _, _ = agent_v2.stage1(h_in, m1, hu_v2.state, torch.Tensor(), 1.0)
        sal_gate = agent_v2.boundary_detector(h1, val_inp)
        h2, _, _ = agent_v2.stage2(h1, m2, hu_v2.state, sal_gate, 1.0)
        h_rel, _ = agent_v2.attractor_head.relax_to_minima((h1 + h2).view(batch_size * seq_len, -1), hu_v2.state)
        logits = agent_v2.volitional_head.compute_volitional_logits(h_rel, hu_v2.state, agent_v2.pos_embeddings.byte_embed.weight)
        val_loss_v2 = criterion(logits.view(-1, agent_v2.text_gen_dim), val_tgt.view(-1)).item()

    print(f"[Vector 2 (Local Plasticity)] Tok/s: {tok_s_v2:.1f} | Val Loss: {val_loss_v2:.4f} (PPL: {math.exp(val_loss_v2):.2f})")
    benchmark_results["Vector 2 (Local Plasticity)"] = {"tok_per_sec": tok_s_v2, "val_loss": val_loss_v2, "ppl": math.exp(val_loss_v2)}

    # -------------------------------------------------------------------------
    # 3. VECTOR 3 STANDALONE: Allostatic SWR Replay & SHY Downscaling
    # -------------------------------------------------------------------------
    print("\n--- Testing Vector 3: Allostatic SWR Replay & SHY Downscaling ---")
    torch.manual_seed(42)
    agent_v3 = CoREAgent(config, device=device_str).to(device_str)
    hu_v3 = HomeostaticUnit(batch_size=batch_size, device=device_str)
    mem_v3 = BatchedEpisodicMemory(batch_size=batch_size, memory_dim=256, max_capacity=100, device=device_str)
    opt_v3 = optim.AdamW(agent_v3.get_all_parameters(), lr=1e-4)

    t0 = time.perf_counter()
    for step_i, (inp_b, tgt_b) in enumerate(encoded_batches):
        opt_v3.zero_grad()
        tot_loss, s_loss, _, _, _, _, _ = agent_v3.forward_sequence(inp_b, tgt_b, hu_v3, criterion, episodic_memory=mem_v3)
        tot_loss.backward()
        opt_v3.step()
        
        # Deplete energy to trigger allostatic sleep consolidation at step 15
        if step_i == 15:
            hu_v3.state[:, 1] = 0.05
            execute_allostatic_shy_sleep(agent_v3, hu_v3, mem_v3)

    dur_v3 = time.perf_counter() - t0
    tok_s_v3 = (num_steps * batch_size * seq_len) / dur_v3
    
    with torch.no_grad():
        _, val_loss_v3, _, _, _, _, _ = agent_v3.forward_sequence(val_inp, val_tgt, hu_v3, criterion)

    print(f"[Vector 3 (SWR & SHY)] Tok/s: {tok_s_v3:.1f} | Val Loss: {val_loss_v3:.4f} (PPL: {math.exp(val_loss_v3):.2f})")
    benchmark_results["Vector 3 (SWR & SHY)"] = {"tok_per_sec": tok_s_v3, "val_loss": val_loss_v3, "ppl": math.exp(val_loss_v3)}

    # -------------------------------------------------------------------------
    # 4. COMBINED ENGINE: Integrated Vectors 1 + 2 + 3
    # -------------------------------------------------------------------------
    print("\n--- Testing Combined Engine (Vectors 1 + 2 + 3) ---")
    torch.manual_seed(42)
    agent_comb = CoREAgent(config, device=device_str).to(device_str)
    stack_comb = ThreeStageLaminarStack(agent_comb.hidden_dim, agent_comb.expand_dim, agent_comb.num_heads, agent_comb.head_k, agent_comb.head_v, device_str)
    hu_comb = HomeostaticUnit(batch_size=batch_size, device=device_str)
    mem_comb = BatchedEpisodicMemory(batch_size=batch_size, memory_dim=256, max_capacity=100, device=device_str)
    engine_comb = LocalPlasticity3StageEngine(agent_comb, stack_comb, lr=1e-3)

    t0 = time.perf_counter()
    for step_i, (inp_b, tgt_b) in enumerate(encoded_batches):
        s_loss = engine_comb.local_step(inp_b, tgt_b, hu_comb)
        
        if step_i == 15:
            hu_comb.state[:, 1] = 0.05
            execute_allostatic_shy_sleep(agent_comb, hu_comb, mem_comb)

    dur_comb = time.perf_counter() - t0
    tok_s_comb = (num_steps * batch_size * seq_len) / dur_comb
    
    with torch.no_grad():
        m1 = torch.zeros(batch_size, agent_comb.num_heads, agent_comb.head_k, agent_comb.head_v, device=device_str)
        m2 = torch.zeros(batch_size, agent_comb.num_heads, agent_comb.head_k, agent_comb.head_v, device=device_str)
        m3 = torch.zeros(batch_size, agent_comb.num_heads, agent_comb.head_k, agent_comb.head_v, device=device_str)
        emb = agent_comb.pos_embeddings(val_inp, start_pos=0, apply_rf=True)
        h_in = agent_comb.in_proj(emb)
        sal_gate = agent_comb.boundary_detector(h_in, val_inp)
        h_fused, _, _, _ = stack_comb(h_in, m1, m2, m3, hu_comb.state, sal_gate)
        h_rel, _ = agent_comb.attractor_head.relax_to_minima(h_fused.view(batch_size * seq_len, -1), hu_comb.state)
        logits = agent_comb.volitional_head.compute_volitional_logits(h_rel, hu_comb.state, agent_comb.pos_embeddings.byte_embed.weight)
        val_loss_comb = criterion(logits.view(-1, agent_comb.text_gen_dim), val_tgt.view(-1)).item()

    print(f"[Combined Engine] Tok/s: {tok_s_comb:.1f} | Val Loss: {val_loss_comb:.4f} (PPL: {math.exp(val_loss_comb):.2f})")
    benchmark_results["Combined (Vector 1+2+3)"] = {"tok_per_sec": tok_s_comb, "val_loss": val_loss_comb, "ppl": math.exp(val_loss_comb)}

    # -------------------------------------------------------------------------
    # SUMMARY & JSON REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL COMPARATIVE BENCHMARK SUMMARY")
    print("=" * 80)
    for name, m in benchmark_results.items():
        print(f"• {name:28s} | Tok/s: {m['tok_per_sec']:8.1f} | Val Loss: {m['val_loss']:.4f} | PPL: {m['ppl']:.2f}")

    with open("exp_118_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

if __name__ == "__main__":
    run_experiment()
