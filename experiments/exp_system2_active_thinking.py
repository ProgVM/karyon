# experiments/exp_system2_active_thinking.py
"""
===============================================================================
KARYON ENGINEERING PROTOCOL (KEP) EXPERIMENT #5 (CACHE-PROOF CONTAINER EVAL)
Topic: System 2 Active Inference Search Engine on Trained Soul (.kcore)
Author: Karyon-CoRE Open-Source Research Team (2026)
===============================================================================

Hypothesis:
    Executing K=3 System 2 Latent Active Inference Rollouts on a pre-trained 
    karyon_soul.kcore container will significantly reduce Variational Free Energy 
    (F_t > 15-30% drop) and boost target prediction confidence compared to System 1.

Control Group: 
    System 1 Reflex Output (Direct single-step prediction).

Experimental Group: 
    System 2 Active Inference Thinking Engine (G-Minimizing Latent Search).

Metrics Tracked:
    1. Variational Free Energy (F_t) Reduction (%)
    2. Target Prediction Confidence (%)
    3. Top-1 Predicted Token/Byte
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
import torch

# =============================================================================
# DYNAMIC HOTFIX FOR PYTORCH 2.4/2.5+ PYTHON 3.12 DYNAMO BUG ON KAGGLE
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

# Force reload karyon_agent from disk to bypass Kaggle Jupyter memory cache
import karyon_config, karyon_core, karyon_agent, karyon_checkpoint
importlib.reload(karyon_agent)

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory
from karyon_checkpoint import load_karyon

import torch.nn as nn
import torch.nn.functional as F

# Ensure autograd tracking is enabled
torch.set_grad_enabled(True)

# Set global seed for exact reproducibility
torch.manual_seed(42)

# Select execution hardware string
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
print(f"[KEP Experiment #5] Execution Context: {device_str.upper()}")

kcore_path = "karyon_soul.kcore"


# =============================================================================
# 1. SYSTEM 2 LATENT ACTIVE INFERENCE ROLLOUT ENGINE
# =============================================================================

def run_system2_thinking(agent, h_fast, h_slow, last_w_pred, rollout_steps=3, candidates_num=4):
    """
    Executes K internal counterfactual simulations in LatentPredictor, 
    selecting the thought trajectory that minimizes Expected Free Energy (G).
    """
    best_h_fast, best_h_slow = h_fast, h_slow
    min_expected_free_energy = float('inf')

    for cand_idx in range(candidates_num):
        h_f_sim = h_fast.clone()
        h_s_sim = h_slow.clone()
        w_sim = last_w_pred.clone()
        
        accumulated_G = 0.0

        for k in range(rollout_steps):
            w_sim, kl_div, fe, _ = agent.world_model(h_f_sim, h_s_sim, w_sim)
            x_proj_sim = F.silu(agent.gateway.project_text(w_sim)) if hasattr(agent.gateway, 'project_text') else w_sim
            
            h_f_sim = agent.core.slow_f[0](torch.cat([h_f_sim, torch.zeros(1, 6, device=device)], dim=-1)) if hasattr(agent.core, 'slow_f') else h_f_sim
            
            # Expected Free Energy = Error + Epistemic Penalty
            epistemic_curiosity = kl_div.mean().item()
            pragmatic_error = fe.mean().item()
            
            step_G = pragmatic_error - 0.2 * epistemic_curiosity
            accumulated_G += step_G

        if accumulated_G < min_expected_free_energy:
            min_expected_free_energy = accumulated_G
            best_h_fast = h_f_sim
            best_h_slow = h_s_sim

    return best_h_fast, best_h_slow, min_expected_free_energy


# =============================================================================
# 2. EXPERIMENTAL BENCHMARK ENGINE ON CHECKPOINTED SOUL
# =============================================================================

def run_checkpointed_system2_benchmark():
    config = CoREConfig()
    
    # Extract DNA Genome dimensions from container manifest
    if os.path.exists(kcore_path):
        with open(kcore_path, 'rb') as f:
            f.seek(8)
            header_raw = f.read(24)
            _, num_sections, _, _ = struct.unpack('<IIQQ', header_raw)
            sections = []
            for _ in range(num_sections):
                sec_raw = f.read(64)
                s_type, _, offset, size, _ = struct.unpack('<IIQQQ', sec_raw[:32])
                sections.append({"type": s_type, "offset": offset, "size": size})
            sec_manifest = next((s for s in sections if s["type"] == 1), None)
            if sec_manifest:
                f.seek(sec_manifest["offset"])
                manifest = json.loads(f.read(sec_manifest["size"]).decode('utf-8'))
                genome = manifest.get("genome", {})
                if "text_dim" in genome: config.net.text_dim = genome["text_dim"]
                if "text_gen_dim" in genome: config.net.text_gen_dim = genome["text_gen_dim"]
                if "unified_dim" in genome: config.net.unified_dim = genome["unified_dim"]
                if "hidden_dim" in genome: config.net.hidden_dim = genome["hidden_dim"]
                if "latent_dim" in genome: config.net.latent_dim = genome["latent_dim"]

    # Explicit string device passing
    agent = CoREAgent(config=config, device=device_str).to(device)
    hu = HomeostaticUnit(batch_size=1, device=device_str)
    episodic_mem = BatchedEpisodicMemory(batch_size=1, memory_dim=config.net.unified_dim, max_capacity=200, device=device_str)

    # Restore 100% trained parameters and states from .kcore container
    h_fast_base, h_slow_base, _, _ = load_karyon(agent, episodic_mem, hu, filepath=kcore_path, device=device_str)
    agent.eval()

    reasoning_tasks = [
        {"prompt": "User: What is the primary source of energy for Earth?\nKaryon:", "target_byte": ord('T')},
        {"prompt": "User: If you mix red and blue, you get\nKaryon:", "target_byte": ord('p')},
        {"prompt": "User: 2 + 2 =\nKaryon:", "target_byte": ord('4')}
    ]

    fe_control_list = []
    fe_exp_list = []

    print("\n" + "="*80)
    print(" === KARYON EXPERIMENT #5: SYSTEM 2 ON TRAINED SOUL (.kcore) ===")
    print("="*80)

    for task_idx, task in enumerate(reasoning_tasks):
        prompt_text = task["prompt"]
        target_byte = task["target_byte"]
        tokens = agent.encode_text(prompt_text)

        print(f"\nTask #{task_idx+1}: '{prompt_text.strip()}' (Target Byte: '{chr(target_byte)}')")
        print("-" * 80)
        print(f"{'Mode':<25} | {'Free Energy (F_t)':<20} | {'Target Confidence (%)':<22} | {'Top-1 Token':<12}")
        print("-" * 80)

        # ---------------------------------------------------------------------
        # 1. CONTROL GROUP: System 1 Reflex
        # ---------------------------------------------------------------------
        h_f1 = h_fast_base.clone()
        h_s1 = h_slow_base.clone()
        u_t = hu.state.clone()

        with torch.no_grad():
            for t_id in tokens[:-1]:
                t_emb = agent.text_embeddings(t_id.unsqueeze(0))
                s_in = {'text': t_emb, 'vision': torch.zeros(1, config.net.vision_dim, device=device), 'motor_efference': torch.zeros(1, config.net.action_dim, device=device)}
                h_f1, h_s1, _, _, _, fe1, _, w_curr1, _, _, _, _ = agent(s_in, h_f1, h_s1, u_t)

            last_token = tokens[-1]
            last_emb = agent.text_embeddings(last_token.unsqueeze(0))
            s_in_last = {'text': last_emb, 'vision': torch.zeros(1, config.net.vision_dim, device=device), 'motor_efference': torch.zeros(1, config.net.action_dim, device=device)}
            h_f1, h_s1, outputs1, _, _, fe1, _, _, _, _, _, _ = agent(s_in_last, h_f1, h_s1, u_t)

            logits1 = outputs1["text_generation"]
            probs1 = F.softmax(logits1, dim=-1)
            target_prob1 = probs1[0, target_byte].item() * 100.0
            top1_id1 = torch.argmax(probs1, dim=-1).item()
            top1_char1 = chr(top1_id1) if 32 <= top1_id1 <= 126 else '.'

            fe_control_list.append(fe1.mean().item())

        print(f"{'System 1 (Reflex)':<25} | {fe1.mean().item():<20.4f} | {target_prob1:<22.2f}% | {top1_char1 + ' (' + str(top1_id1) + ')':<12}")

        # ---------------------------------------------------------------------
        # 2. EXPERIMENTAL GROUP: System 2 Active Inference Thinking Engine
        # ---------------------------------------------------------------------
        h_f2 = h_fast_base.clone()
        h_s2 = h_slow_base.clone()

        with torch.no_grad():
            for t_id in tokens[:-1]:
                t_emb = agent.text_embeddings(t_id.unsqueeze(0))
                s_in = {'text': t_emb, 'vision': torch.zeros(1, config.net.vision_dim, device=device), 'motor_efference': torch.zeros(1, config.net.action_dim, device=device)}
                h_f2, h_s2, _, _, _, fe2, _, w_curr2, _, _, _, _ = agent(s_in, h_f2, h_s2, u_t)

            # System 2 Thinking Rollout in TRAINED Latent World Model
            h_f2_thought, h_s2_thought, _ = run_system2_thinking(
                agent, h_f2, h_s2, w_curr2, rollout_steps=3, candidates_num=4
            )

            s_in_last = {'text': last_emb, 'vision': torch.zeros(1, config.net.vision_dim, device=device), 'motor_efference': torch.zeros(1, config.net.action_dim, device=device)}
            h_f2_thought, h_s2_thought, outputs2, _, _, fe2, _, _, _, _, _, _ = agent(s_in_last, h_f2_thought, h_s2_thought, u_t)

            logits2 = outputs2["text_generation"]
            probs2 = F.softmax(logits2, dim=-1)
            target_prob2 = probs2[0, target_byte].item() * 100.0
            top1_id2 = torch.argmax(probs2, dim=-1).item()
            top1_char2 = chr(top1_id2) if 32 <= top1_id2 <= 126 else '.'

            fe_exp_list.append(fe2.mean().item())

        print(f"{'System 2 (Active Search)':<25} | {fe2.mean().item():<20.4f} | {target_prob2:<22.2f}% | {top1_char2 + ' (' + str(top1_id2) + ')':<12}")

    # =========================================================================
    # KEP EVALUATION & VERDICT LOGIC
    # =========================================================================
    avg_fe_control = sum(fe_control_list) / len(fe_control_list)
    avg_fe_exp = sum(fe_exp_list) / len(fe_exp_list)
    fe_reduction_pct = ((avg_fe_control - avg_fe_exp) / avg_fe_control) * 100.0

    print("\n" + "="*80)
    print(" === KEP EVALUATION & VERDICT ===")
    print("="*80)
    if fe_reduction_pct >= 3.0:
        print("🟢 VERDICT: POSITIVE EXPERIENCE ADOPTED!")
        print(f"   Reason: System 2 Active Search reduced Free Energy by {fe_reduction_pct:.2f}% on trained container.")
        print("   Action: Merge System 2 Thought Engine into production karyon_agent!")
    elif fe_reduction_pct < 0.0:
        print("🔴 VERDICT: NEGATIVE EXPERIENCE REJECTED!")
        print(f"   Reason: System 2 rollouts increased Free Energy error by {abs(fe_reduction_pct):.2f}%.")
    else:
        print("⚪ VERDICT: NEUTRAL EXPERIENCE DISCARDED.")
        print("   Reason: Insufficient Free Energy reduction gain.")
    print("="*80 + "\n")


# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_checkpointed_system2_benchmark()
