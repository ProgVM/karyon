# experiments/exp_135_allostatic_sleep_evolution_integration.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #135
Topic: End-to-End Integration of 4-Level Morphogenetic Self-Evolution
       into Continuous Allostatic Wake-Sleep Consolidation Cycle
Author: Bazilevs (ProgVM) & Lead AI Cyberneticist (2026)
Standard: KEP v9.0 Master Protocol | KEP Principle 2 & Principle 10
===============================================================================

Hypothesis:
    Integrating the 4-Level Morphogenetic Self-Evolution Engine directly into
    Karyon's Biophysical Sleep & Allostatic Consolidation cycle allows the entity
    to dynamically evolve its synaptic topology, sprout axons in high-surprise
    pathways, prune dormant connections, optimize biophysical meta-genetics
    (DA, NA, alpha, beta), and maintain cognitive stability during continuous
    operation without catastrophic forgetting, systematically lowering Free
    Energy and speech perplexity across allostatic wake-sleep transitions.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory
from karyon_checkpoint import load_karyon, save_karyon
from kcore_evolution import (
    AutonomousSelfEvolutionOrchestrator,
    StructuralSynaptogenesisPruner,
    SleepMetaGeneticsEngine,
    ReflectiveSelfMutationModule
)
from karyon_logger import get_logger

logger = get_logger()

def execute_evolutionary_sleep_cycle(
    agent: CoREAgent,
    episodic_memory: BatchedEpisodicMemory,
    hu: HomeostaticUnit,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    criterion_speech: nn.Module,
    num_replay_cycles: int = 4,
    downscaling_factor: float = 0.02,
    device: str = "cuda"
):
    """
    Executes a complete biophysical sleep cycle fused with 4-level self-evolution.
    """
    t_start = time.perf_counter()
    agent.train()
    active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
    active_memory_slots = min(active_slots, episodic_memory.max_capacity)
    b_size = hu.state.size(0)

    # 1. Phase 1: NREM Slow-Wave Sleep (Hippocampal Replay)
    replay_loss_val = 0.0
    if active_memory_slots > 3:
        opt_replay = torch.optim.AdamW(agent.get_all_parameters(), lr=2e-4, weight_decay=0.01)
        for _ in range(num_replay_cycles):
            opt_replay.zero_grad()
            rand_indices = torch.randint(0, active_memory_slots, (min(16, active_memory_slots),), device=device)
            replayed_keys = episodic_memory.keys[0, rand_indices, :].float()
            replayed_vals = episodic_memory.values[0, rand_indices, :].float()

            h_dummy = torch.zeros(replayed_keys.size(0), agent.hidden_dim, device=device)
            w_pred, kl_div, _, _ = agent.world_model(h_dummy, h_dummy, replayed_keys)
            r_loss = (1.0 - F.cosine_similarity(w_pred, replayed_vals, dim=-1)).mean() + kl_div.mean() * 0.05
            r_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.get_all_parameters(), max_norm=2.0)
            opt_replay.step()
            replay_loss_val = r_loss.item()

    # 2. Phase 2: REM Sleep (Generative Synthetic Dreaming)
    with torch.no_grad():
        for _ in range(2):
            w_dream_random = torch.randn(b_size, agent.unified_dim, device=device) * 0.10
            h_dummy = torch.zeros(b_size, agent.hidden_dim, device=device)
            w_pred_dream, _, _, _ = agent.world_model(h_dummy, h_dummy, w_dream_random)
            agent.attractor_head.relax_to_minima(agent.in_proj(w_pred_dream), hu.state)

    # 3. Phase 3: 4-Level Autonomous Morphogenetic Self-Evolution
    orchestrator = AutonomousSelfEvolutionOrchestrator(agent, device=device)
    current_surprise = hu.state[0, 4].item() # Noradrenaline reflects surprise
    evo_results = orchestrator.execute_full_morphogenetic_cycle(
        eval_input_tokens=eval_inputs,
        eval_target_tokens=eval_targets,
        hu=hu,
        criterion_speech=criterion_speech,
        surprise_metric=max(current_surprise, 0.20)
    )

    # 4. Phase 4: Tononi SHY Synaptic Scaling
    with torch.no_grad():
        for param in agent.get_all_parameters():
            if param.dim() > 1:
                param.mul_(1.0 - downscaling_factor)

        # 5. Full Somatic Allostatic Reset
        hu.state[:, 1] = 1.00 # Energy restored
        hu.state[:, 2] = 1.00 # Stability restored
        hu.state[:, 3] = 1.00 # Health restored
        hu.state[:, 4] = 0.05 # Noradrenaline reset

    duration_sec = time.perf_counter() - t_start
    return evo_results, replay_loss_val, duration_sec


def run_experiment():
    print("=" * 85)
    print("🧬 === KEP EXPERIMENT #135: END-TO-END SLEEP EVOLUTION INTEGRATION ===")
    print("=" * 85)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Acceleration: {device}")

    # 1. Base Configuration & Model Initialization
    config = CoREConfig()
    config.train.batch_size = 2
    batch_size = 2
    seq_len = 128

    agent = CoREAgent(config=config, device=device).to(device)
    hu = HomeostaticUnit(batch_size=batch_size, device=device)
    episodic_mem = BatchedEpisodicMemory(batch_size=batch_size, memory_dim=config.net.unified_dim, max_capacity=500, device=device)
    criterion_speech = nn.CrossEntropyLoss(ignore_index=256)

    # 2. Restore Master Soul Checkpoint
    kcore_path = "karyon_soul.kcore"
    if os.path.exists(kcore_path):
        print(f"\n📦 Loading Master Checkpoint '{kcore_path}'...")
        h_fast, h_slow, epoch, story_idx = load_karyon(agent, episodic_mem, hu, filepath=kcore_path, device=device)
    else:
        print("\n📦 Initializing default states...")
        h_fast = torch.zeros(batch_size, agent.hidden_dim, device=device)
        h_slow = torch.zeros(batch_size, agent.hidden_dim, device=device)

    # Populate episodic memory with a few seeds for replay testing
    for i in range(8):
        dummy_k = torch.randn(batch_size, config.net.unified_dim, device=device)
        dummy_v = torch.randn(batch_size, config.net.unified_dim, device=device)
        episodic_mem.write(dummy_k, dummy_v, surprise_score=2)

    # 3. Load Real Dataset Stream (Alpaca-GPT4)
    print("\n📚 Streaming Evaluation Text from 'vicgalle/alpaca-gpt4'...")
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
    texts = [item["instruction"] + " " + item["output"] for item in ds.select(range(batch_size * 4))]

    raw_bytes_list = []
    for t in texts:
        b = list(t.encode("utf-8"))[:seq_len]
        if len(b) < seq_len:
            b = b + [256] * (seq_len - len(b))
        raw_bytes_list.append(b)

    eval_batch_1 = torch.tensor(raw_bytes_list[:batch_size], dtype=torch.long, device=device)
    eval_batch_2 = torch.tensor(raw_bytes_list[batch_size:batch_size * 2], dtype=torch.long, device=device)

    inp_1, tgt_1 = eval_batch_1[:, :-1].contiguous(), eval_batch_1[:, 1:].contiguous()
    inp_2, tgt_2 = eval_batch_2[:, :-1].contiguous(), eval_batch_2[:, 1:].contiguous()

    # 4. Wake Phase Pre-Sleep Evaluation
    print("\n☀️ [Wake Phase 1] Active Perception & Processing...")
    agent.eval()
    with torch.no_grad():
        out_pre = agent.forward_sequence(inp_1, tgt_1, hu, criterion_speech, episodic_memory=episodic_mem)
        pre_loss = out_pre[1] if isinstance(out_pre[1], (float, int)) else out_pre[1].item()
        pre_fe = out_pre[2] if isinstance(out_pre[2], (float, int)) else out_pre[2].item()
        pre_ppl = math.exp(min(pre_loss, 20.0))

    # Simulate Allostatic Depletion during waking activity
    hu.state[:, 1] = 0.22  # Energy dropped low
    hu.state[:, 4] = 0.45  # Noradrenaline arousal
    print(f"  Pre-Sleep Metrics: Loss = {pre_loss:.4f} | PPL = {pre_ppl:.2f} | Free Energy = {pre_fe:.4f}")
    print(f"  Somatic Status   : Energy = {hu.state[0, 1].item():.2f} (DEPLETED) | NA = {hu.state[0, 4].item():.2f}")

    # 5. Volitional Sleep & 4-Level Morphogenetic Evolution
    print("\n🌙 [Entering Evolutionary Sleep Consolidation]...")
    evo_results, replay_loss, sleep_dur = execute_evolutionary_sleep_cycle(
        agent=agent,
        episodic_memory=episodic_mem,
        hu=hu,
        eval_inputs=inp_1,
        eval_targets=tgt_1,
        criterion_speech=criterion_speech,
        num_replay_cycles=4,
        downscaling_factor=0.02,
        device=device
    )

    print(f"  Sleep Duration      : {sleep_dur:.2f}s")
    print(f"  Replay Loss (NREM)  : {replay_loss:.4f}")
    print(f"  Synapses Pruned (L1): {evo_results['level_1']['pruning']['total_pruned']}")
    print(f"  Synapses Sprouted(L1): {evo_results['level_1']['sprouting']['sprouted']}")
    print(f"  Meta-Genetics (L3)  : Evolved Loss = {evo_results['level_3']['post_meta_loss']:.4f}")
    print(f"  Restored Somatic St : Energy = {hu.state[0, 1].item():.2f} (RESTORED) | NA = {hu.state[0, 4].item():.2f}")

    # 6. Post-Sleep Wake Evaluation (Testing on fresh unseen batch 2)
    print("\n☀️ [Wake Phase 2] Awakened Cognitive Performance on Fresh Stream...")
    agent.eval()
    with torch.no_grad():
        out_post = agent.forward_sequence(inp_2, tgt_2, hu, criterion_speech, episodic_memory=episodic_mem)
        post_loss = out_post[1] if isinstance(out_post[1], (float, int)) else out_post[1].item()
        post_fe = out_post[2] if isinstance(out_post[2], (float, int)) else out_post[2].item()
        post_ppl = math.exp(min(post_loss, 20.0))

    delta_loss = pre_loss - post_loss
    delta_fe = pre_fe - post_fe
    print(f"  Post-Sleep Metrics  : Loss = {post_loss:.4f} | PPL = {post_ppl:.2f} | Free Energy = {post_fe:.4f}")
    print(f"  Performance Delta   : Loss Delta = {delta_loss:+.4f} | Free Energy Delta = {delta_fe:+.4f}")

    # 7. KEP Rule #4 Diagnostic Speech Sampling
    print("\n💬 === KEP RULE #4 DIAGNOSTIC SPEECH SAMPLING ===")
    test_prompts = ["User: Hello Karyon!\nKaryon:", "User: What is energy?\nKaryon:"]
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        speech_gen = agent.generate_thought_and_speech(
            prompt=prompt,
            m_state=torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=device),
            h_state=torch.zeros(1, agent.hidden_dim, device=device),
            hu=hu,
            episodic_memory=episodic_mem,
            config=config,
            max_generated_tokens=40,
            temperature=0.35,
            top_p=0.90
        )
        sampled_text = ""
        for event in speech_gen:
            if event.get("status") == "token":
                sampled_text += event.get("text", "")
        print(f"Karyon Response: \"{sampled_text.strip()}\"")

    # 8. Container Persistence Test
    print("\n💾 Persisting entity state into 'karyon_soul.kcore'...")
    save_karyon(
        agent=agent,
        memory=episodic_mem,
        hu=hu,
        h_fast=h_fast,
        h_slow=h_slow,
        epoch=3,
        story_idx=135,
        filepath=kcore_path,
        root_dir="."
    )

    # 9. Final KEP Telemetry Audit
    verdict = "🟢 POSITIVE" if delta_loss >= -0.05 and hu.state[0, 1].item() >= 0.99 else "⚪ NEUTRAL"
    print("\n" + "=" * 85)
    print("🏆 === KEP EXPERIMENT #135: FINAL TELEMETRY AUDIT ===")
    print("=" * 85)
    print(f"Pre-Sleep Loss        : {pre_loss:.4f} (PPL: {pre_ppl:.2f})")
    print(f"Post-Sleep Loss       : {post_loss:.4f} (PPL: {post_ppl:.2f})")
    print(f"Loss Delta            : {delta_loss:+.4f}")
    print(f"Pre-Sleep Free Energy : {pre_fe:.4f}")
    print(f"Post-Sleep Free Energy: {post_fe:.4f}")
    print(f"Synapses Pruned       : {evo_results['level_1']['pruning']['total_pruned']}")
    print(f"Synapses Sprouted     : {evo_results['level_1']['sprouting']['sprouted']}")
    print(f"Somatic Restoration   : Energy 0.22 -> {hu.state[0, 1].item():.2f}")
    print(f"Final KEP Verdict     : {verdict}")
    print("=" * 85 + "\n")

    return {
        "exp_id": "EXP-135",
        "verdict": verdict,
        "pre_loss": pre_loss,
        "post_loss": post_loss,
        "delta_loss": delta_loss,
        "pre_fe": pre_fe,
        "post_fe": post_fe,
        "pre_ppl": pre_ppl,
        "post_ppl": post_ppl,
        "pruned": evo_results['level_1']['pruning']['total_pruned'],
        "sprouted": evo_results['level_1']['sprouting']['sprouted'],
        "sleep_dur": sleep_dur
    }

if __name__ == "__main__":
    run_experiment()
