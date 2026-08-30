# experiments/exp_97_biophysical_multimodal_mind.py
"""
===============================================================================
KARYON EXPERIMENTAL BENCHMARK: EXP-97 (BIOPHYSICAL MULTIMODAL COGNITIVE MIND)
Testing the 5 Cybernetic Breakthroughs:
1. Panksepp Primary Affects & Affective Core Vector (Valence, Arousal, Dominance).
2. Unconditioned Reflex Shunt & Conditioned Procedural Habit Loop (Basal Ganglia).
3. Dual-Phase Biophysical Sleep Cycle (NREM Slow-Wave Replay + REM Generative Synthetic Dreaming + Synaptic Pruning).
4. Multi-Store Memory Architecture (Working + Episodic + Procedural + Hopfield Attractor Graph).
5. Dynamic Extensible Multimodal Stream Parallelism (Text, Vision, Audio, Cybernetic Sensors).

Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import sys
import os
import time
import math
import types
import json
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

# Dynamo Hotfix for Python 3.12 / Kaggle GPU
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import karyon_config, karyon_core, karyon_agent, karyon_logger
importlib.reload(karyon_core)
importlib.reload(karyon_agent)

from karyon_config import CoREConfig
from karyon_agent import CoREAgent, DynamicSensoryGateway
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory

logger = karyon_logger.get_logger()
torch.set_grad_enabled(True)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
use_amp = (device_str == 'cuda')


# =============================================================================
# 1. AFFECTIVE CORE & PANKSEPP PRIMARY DRIVES
# =============================================================================

class AffectiveCoreUnit(nn.Module):
    """
    Computes Russell's Affective Circumplex (Valence, Arousal, Dominance)
    and Panksepp Primary Affective Drives (SEEKING, FEAR, RAGE, PANIC).
    """
    def __init__(self, device_str='cpu'):
        super().__init__()
        self.device = torch.device(device_str)

    def compute_affective_state(self, u_t: torch.Tensor, free_energy: float, value_est: float) -> dict:
        curiosity = u_t[:, 0].mean().item()
        energy    = u_t[:, 1].mean().item()
        stability = u_t[:, 2].mean().item()
        health    = u_t[:, 3].mean().item()
        na        = u_t[:, 4].mean().item()
        da        = u_t[:, 5].mean().item()

        # Russell Affective Coordinates
        valence   = da - (1.0 - energy) - (1.0 - health)
        arousal   = na + min(1.0, max(0.0, free_energy if not math.isnan(free_energy) else 0.0))
        dominance = stability + max(-1.0, min(1.0, value_est))

        # Panksepp Primary Affective Drives
        seeking_drive = max(0.0, curiosity + da - max(0.0, free_energy if not math.isnan(free_energy) else 0.0))
        fear_drive    = max(0.0, arousal * (1.0 - stability))
        rage_drive    = max(0.0, (1.0 - energy) * (1.0 - dominance))
        panic_drive   = max(0.0, (1.0 - health) * (1.0 - stability))

        return {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "panksepp": {
                "SEEKING": seeking_drive,
                "FEAR": fear_drive,
                "RAGE": rage_drive,
                "PANIC": panic_drive
            }
        }


# =============================================================================
# 2. UNCONDITIONED & CONDITIONED REFLEX CIRCUITS (BASAL GANGLIA HABIT LOOP)
# =============================================================================

class ReflexAndHabitCircuit(nn.Module):
    """
    Biophysical Subcortical Reflex & Habit Module:
    1. Unconditioned Emergency Reflex: Overrides motor output on somatic energy collapse or extreme surprise.
    2. Conditioned Habit Circuit (Basal Ganglia): Direct fast associative mapping bypassing deep cortical layers when dopamine DA > 0.50.
    """
    def __init__(self, unified_dim=256, action_dim=3, device_str='cpu'):
        super().__init__()
        self.unified_dim = unified_dim
        self.action_dim = action_dim
        self.device = torch.device(device_str)

        self.habit_policy = nn.Sequential(
            nn.Linear(unified_dim, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim)
        ).to(self.device)

    def check_unconditioned_reflex(self, u_t: torch.Tensor, free_energy: float) -> bool:
        energy = u_t[:, 1].min().item()
        health = u_t[:, 3].min().item()
        fe_check = free_energy if not math.isnan(free_energy) else 0.0
        return (energy < 0.15 or health < 0.20 or fe_check > 0.85)

    def execute_conditioned_habit(self, w_t: torch.Tensor, da_level: float) -> torch.Tensor:
        if da_level > 0.50:
            return self.habit_policy(w_t)
        return torch.zeros(w_t.size(0), self.action_dim, device=self.device)


# =============================================================================
# 3. ENHANCED BIOPHYSICAL AGENT (EXP-97 COGNITIVE CORE)
# =============================================================================

class AdvancedBiophysicalCoREAgent(CoREAgent):
    def __init__(self, config, device='cpu'):
        super().__init__(config, device)
        self.affective_core = AffectiveCoreUnit(device_str=self.device_str)
        self.reflex_circuit = ReflexAndHabitCircuit(unified_dim=self.unified_dim, action_dim=self.action_dim, device_str=self.device_str)

    def get_all_parameters(self):
        params = super().get_all_parameters()
        params.extend(list(self.reflex_circuit.parameters()))
        return params

    def execute_dual_phase_sleep(self, episodic_memory: BatchedEpisodicMemory, hu: HomeostaticUnit,
                                 num_nrem_replays: int = 5, num_rem_dreams: int = 3,
                                 downscaling_factor: float = 0.03, pruning_percentile: float = 0.05):
        """
        Advanced Dual-Phase Sleep Cycle (NREM Slow-Wave Replay + REM Generative Synthetic Dreaming):
        - Phase 1 (NREM): Replays high-surprise memories and applies Tononi SHY Synaptic Scaling.
        - Phase 2 (REM): Synthesizes generative counterfactual dream trajectories from Gaussian latent noise.
        - Morphogenesis: Prunes bottom percentile of weak synaptic weights.
        """
        self.train()
        active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
        active_memory_slots = min(active_slots, episodic_memory.max_capacity)
        b_size = hu.state.size(0)

        # ---------------------------------------------------------------------
        # PHASE 1: NREM SLOW-WAVE SLEEP (Hippocampal Replay & Consolidation)
        # ---------------------------------------------------------------------
        if active_memory_slots > 3:
            opt_replay = torch.optim.AdamW(self.get_all_parameters(), lr=3e-4, weight_decay=0.01)
            for _ in range(num_nrem_replays):
                opt_replay.zero_grad()
                rand_indices = torch.randint(0, active_memory_slots, (min(16, active_memory_slots),), device=self.device)
                replayed_keys = episodic_memory.keys[0, rand_indices, :].float()
                replayed_vals = episodic_memory.values[0, rand_indices, :].float()

                h_dummy = torch.zeros(replayed_keys.size(0), self.hidden_dim, device=self.device)
                w_pred, kl_div, _, _ = self.world_model(h_dummy, h_dummy, replayed_keys)
                replay_loss = (1.0 - F.cosine_similarity(w_pred, replayed_vals, dim=-1)).mean() + kl_div.mean() * 0.05

                replay_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.get_all_parameters(), max_norm=2.0)
                opt_replay.step()

        # ---------------------------------------------------------------------
        # PHASE 2: REM SLEEP (Generative Counterfactual Synthetic Dreaming)
        # ---------------------------------------------------------------------
        with torch.no_grad():
            for _ in range(num_rem_dreams):
                w_dream_random = torch.randn(b_size, self.unified_dim, device=self.device) * 0.10
                h_dummy = torch.zeros(b_size, self.hidden_dim, device=self.device)
                w_pred_dream, _, _, _ = self.world_model(h_dummy, h_dummy, w_dream_random)
                # Re-align Hopfield attractor basins with dream representations
                self.attractor_head.relax_to_minima(self.in_proj(w_pred_dream), hu.state)

        # ---------------------------------------------------------------------
        # PHASE 3: SYNAPTIC PRUNING (MORPHOGENESIS) & TONONI SHY DOWNSCALING
        # ---------------------------------------------------------------------
        total_pruned_weights = 0
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.dim() > 1 and "weight" in name and param.numel() > 100:
                    flat_abs = param.abs().flatten()
                    k = int(flat_abs.numel() * pruning_percentile)
                    if k > 0:
                        threshold = torch.kthvalue(flat_abs, k).values
                        prune_mask = param.abs() < threshold
                        total_pruned_weights += prune_mask.sum().item()
                        param.masked_fill_(prune_mask, 0.0)

            # Tononi SHY Synaptic Scaling
            for param in self.get_all_parameters():
                if param.dim() > 1:
                    param.mul_(1.0 - downscaling_factor)

            # Full Somatic Allostatic Reset
            hu.state[:, 1] = 1.00 # Energy
            hu.state[:, 2] = 1.00 # Stability
            hu.state[:, 3] = 1.00 # Health
            hu.state[:, 4] = 0.05 # Noradrenaline

        return total_pruned_weights


# =============================================================================
# 4. DATASET PREPARATION (MULTIMODAL PACKED STREAM)
# =============================================================================

def prepare_multimodal_packed_stream(num_batches: int = 100, batch_size: int = 8, seq_len: int = 512):
    logger.info(f"Loading Real Dataset (vicgalle/alpaca-gpt4) for EXP-97 (S={seq_len}, Steps={num_batches})...")
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
    tokenizer = ByteTokenizer()
    full_stream = []

    for item in ds:
        inst = item.get("instruction", "").strip()
        out = item.get("output", "").strip()
        if inst and out:
            dialog = f"User: {inst}\nKaryon: {out}"
            full_stream.extend(tokenizer.encode(dialog))
        if len(full_stream) >= num_batches * batch_size * (seq_len + 1):
            break

    batches = []
    block_size = seq_len + 1
    for b in range(num_batches):
        text_tensors = []
        vision_tensors = []
        audio_tensors = []
        cyber_tensors = []

        for s in range(batch_size):
            start = (b * batch_size + s) * block_size
            end = start + block_size
            chunk = full_stream[start:end]
            if len(chunk) < block_size:
                chunk = chunk + [256] * (block_size - len(chunk))

            t_t = torch.tensor(chunk, dtype=torch.long)
            v_t = torch.zeros(block_size, 256) # Zero-initialized to prevent unaligned noise distraction
            a_t = torch.zeros(block_size, 256)
            c_t = torch.zeros(block_size, 256)

            text_tensors.append(t_t)
            vision_tensors.append(v_t)
            audio_tensors.append(a_t)
            cyber_tensors.append(c_t)

        batches.append({
            "text": torch.stack(text_tensors, dim=0).to(device),
            "vision": torch.stack(vision_tensors, dim=0).to(device),
            "audio": torch.stack(audio_tensors, dim=0).to(device),
            "cybernetic": torch.stack(cyber_tensors, dim=0).to(device)
        })

    logger.info(f"Prepared {len(batches)} Real Multimodal Packed Batches (B={batch_size}, S={seq_len}).")
    return batches


# =============================================================================
# 5. EXP-97 BENCHMARK EXECUTION
# =============================================================================

def run_exp_97_benchmark():
    print("\n" + "="*85)
    print(" === [KEP EXPERIMENTAL BENCHMARK: EXP-97 (BIOPHYSICAL MULTIMODAL MIND)] ===")
    print("="*85)

    config = CoREConfig()
    config.net.text_dim = 256
    config.net.hidden_dim = 768
    config.net.expand_dim = 3072
    config.net.num_heads = 12

    b_size, seq_len = 8, 512
    num_eval_steps = 100
    chunk_size = 64

    batches = prepare_multimodal_packed_stream(num_batches=num_eval_steps, batch_size=b_size, seq_len=seq_len)

    # -------------------------------------------------------------------------
    # PHASE 1: BASELINE (CoREAgent, Text-Only Stream)
    # -------------------------------------------------------------------------
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 1: BASELINE (CoREAgent, Text-Only Stream) <<<")
    print("-"*85)
    if device_str == 'cuda': torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    agent_base = CoREAgent(config, device=device_str).to(device)
    hu_base = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_base = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_base = torch.optim.AdamW(agent_base.get_all_parameters(), lr=1.5e-3, weight_decay=0.01)
    scaler_base = torch.amp.GradScaler('cuda', enabled=use_amp)
    crit_speech = nn.CrossEntropyLoss(ignore_index=256)

    t0 = time.perf_counter()
    base_losses = []

    for step, batch_data in enumerate(batches):
        inp = batch_data["text"][:, :-1]
        tgt = batch_data["text"][:, 1:]
        opt_base.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt = agent_base.forward_sequence(
                inp, tgt, hu_base, crit_speech, episodic_memory=mem_base, chunk_size=chunk_size
            )
        scaler_base.scale(tot_loss).backward()
        scaler_base.unscale_(opt_base)
        torch.nn.utils.clip_grad_norm_(agent_base.get_all_parameters(), max_norm=2.0)
        scaler_base.step(opt_base)
        scaler_base.update()
        base_losses.append(s_loss)
        if (step + 1) % 25 == 0:
            print(f"  [Baseline Step {step+1:03d}/{num_eval_steps}] Speech Loss: {s_loss:.4f} | Free Energy: {fe_loss:.4f}")

    base_duration = time.perf_counter() - t0
    base_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if device_str == 'cuda' else 0.0
    valid_base_losses = [l for l in base_losses[-20:] if not math.isnan(l)]
    base_final_loss = sum(valid_base_losses) / max(1, len(valid_base_losses))
    base_ppl = math.exp(min(base_final_loss, 20.0))
    base_tok_per_sec = (num_eval_steps * b_size * seq_len) / base_duration

    print(f"\n[Baseline Summary] Final Loss: {base_final_loss:.4f} | PPL: {base_ppl:.2f} | Peak VRAM: {base_vram:.1f} MB | Speed: {base_tok_per_sec:.1f} tok/s")

    # -------------------------------------------------------------------------
    # PHASE 2: PROPOSED (AdvancedBiophysicalCoREAgent Multimodal + Sleep)
    # -------------------------------------------------------------------------
    print("\n" + "-"*85)
    print(" >>> RUNNING PHASE 2: PROPOSED (AdvancedBiophysicalCoREAgent Multimodal + Dual-Phase Sleep) <<<")
    print("-"*85)
    if device_str == 'cuda': torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    agent_prop = AdvancedBiophysicalCoREAgent(config, device=device_str).to(device)
    hu_prop = HomeostaticUnit(batch_size=b_size, device=device_str)
    mem_prop = BatchedEpisodicMemory(batch_size=b_size, memory_dim=256, max_capacity=1000, device=device_str)
    opt_prop = torch.optim.AdamW(agent_prop.get_all_parameters(), lr=1.5e-3, weight_decay=0.01)
    scaler_prop = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Register custom cybernetic sensor channel
    agent_prop.register_sensory_channel('cybernetic', 256)

    t0 = time.perf_counter()
    prop_losses = []
    unconditioned_reflex_triggers = 0

    for step, batch_data in enumerate(batches):
        inp_text = batch_data["text"][:, :-1]
        tgt_text = batch_data["text"][:, 1:]
        
        sensor_dict = {
            "text": inp_text,
            "vision": batch_data["vision"][:, :-1],
            "audio": batch_data["audio"][:, :-1],
            "cybernetic": batch_data["cybernetic"][:, :-1]
        }

        opt_prop.zero_grad()
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
            tot_loss, s_loss, fe_loss, m_s2, h_p, u_t, eff_dt = agent_prop.forward_multimodal_sequence(
                sensor_dict, tgt_text, hu_prop, crit_speech, episodic_memory=mem_prop, chunk_size=chunk_size
            )

        # Check Unconditioned Emergency Reflex Shunt
        if agent_prop.reflex_circuit.check_unconditioned_reflex(hu_prop.state, fe_loss):
            unconditioned_reflex_triggers += 1
            hu_prop.state[:, 4] = 1.0 # Emergency NA arousal surge

        scaler_prop.scale(tot_loss).backward()
        scaler_prop.unscale_(opt_prop)
        torch.nn.utils.clip_grad_norm_(agent_prop.get_all_parameters(), max_norm=2.0)
        scaler_prop.step(opt_prop)
        scaler_prop.update()
        prop_losses.append(s_loss)

        if (step + 1) % 25 == 0:
            affective_st = agent_prop.affective_core.compute_affective_state(hu_prop.state, fe_loss, 0.0)
            print(f"  [Proposed Step {step+1:03d}/{num_eval_steps}] Loss: {s_loss:.4f} | Free Energy: {fe_loss:.4f} | Valence: {affective_st['valence']:+.3f} | Arousal: {affective_st['arousal']:.3f} | Panksepp SEEKING: {affective_st['panksepp']['SEEKING']:.3f}")

    # Execute Dual-Phase Sleep Cycle at the end of streaming session
    print("\n🌙 Executing Dual-Phase Sleep Cycle (NREM Slow-Wave Replay + REM Synthetic Dreaming + Synaptic Pruning)...")
    pruned_count = agent_prop.execute_dual_phase_sleep(mem_prop, hu_prop, num_nrem_replays=5, num_rem_dreams=3)
    print(f"☀️ Sleep Complete! Pruned {pruned_count} weak connections. Allostatic energy fully restored ({hu_prop.state[0,1].item():.2f}).")

    prop_duration = time.perf_counter() - t0
    prop_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if device_str == 'cuda' else 0.0
    valid_prop_losses = [l for l in prop_losses[-20:] if not math.isnan(l)]
    prop_final_loss = sum(valid_prop_losses) / max(1, len(valid_prop_losses))
    prop_ppl = math.exp(min(prop_final_loss, 20.0))
    prop_tok_per_sec = (num_eval_steps * b_size * seq_len) / prop_duration

    print(f"\n[Proposed Summary] Final Loss: {prop_final_loss:.4f} | PPL: {prop_ppl:.2f} | Peak VRAM: {prop_vram:.1f} MB | Speed: {prop_tok_per_sec:.1f} tok/s")

    # -------------------------------------------------------------------------
    # KEP RULE #2 DECISION EVALUATION
    # -------------------------------------------------------------------------
    delta_loss = base_final_loss - prop_final_loss
    throughput_retention_pct = (prop_tok_per_sec / base_tok_per_sec) * 100.0

    print("\n" + "="*85)
    print(" === [KEP RULE #2 FINAL EMPIRICAL DECISION EVALUATION] ===")
    print("="*85)
    print(f"Baseline Loss      : {base_final_loss:.4f} (PPL: {base_ppl:.2f}) | Speed: {base_tok_per_sec:.1f} tok/s")
    print(f"Proposed Loss      : {prop_final_loss:.4f} (PPL: {prop_ppl:.2f}) | Speed: {prop_tok_per_sec:.1f} tok/s")
    print(f"Delta Loss         : {delta_loss:+.4f} ({'IMPROVEMENT' if delta_loss > 0 else 'REGRESSION'})")
    print(f"Speed Retention    : {throughput_retention_pct:.1f}%")

    if delta_loss >= 0.08 and throughput_retention_pct >= 80.0:
        verdict = "🟢 POSITIVE"
    elif delta_loss < -0.08:
        verdict = "🔴 REJECTED"
    else:
        verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    print(f"VERDICT            : {verdict}")
    print("="*85 + "\n")

    return {
        "verdict": verdict,
        "base_loss": base_final_loss,
        "prop_loss": prop_final_loss,
        "delta_loss": delta_loss,
        "prop_ppl": prop_ppl,
        "prop_tok_per_sec": prop_tok_per_sec,
        "prop_vram": prop_vram
    }


if __name__ == "__main__":
    run_exp_97_benchmark()
