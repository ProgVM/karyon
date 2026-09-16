# experiments/exp_221_allostatic_motor_readout_synthesis.py
"""
===============================================================================
EXP-221: Allostatically-Modulated Motor Readout Synthesis Engine into CoRE Agent
Grounding: KEP Principle 2 (Living AGI & Biological Realism - NON-NEGOTIABLE),
           KEP Principle 14 (Axiom of Allostatic Dynamic Forces — No Static Constants),
           KEP Principle 15 (Net2Net Smooth Grafting: Strict Zero-Delta Identity at Birth t_0).
===============================================================================
Hypothesis:
In `VolitionalActiveInferenceMotorHead`, the motor readout projection `h_proj = motor_text_proj(h_relaxed)`
and CPG Causal Motor Receptive Field output `h_cpg_out` are combined via simple addition:
`h_cpg_out = cpg_motor[2](cpg_motor[1](h_cpg) + h_proj)`.
Replacing this static addition with an Allostatically-Modulated Motor Readout Synthesizer:
  h_cpg_out = cpg_motor[2](cpg_motor[1](h_cpg) + h_proj) + 0.20 * tanh(W_allo_readout * u_t_exp) * h_proj
with 100% Net2Net zero-initialization on W_allo_readout at birth t_0 guarantees strict zero-delta function
identity at birth. This dynamic homeostatic modulation allows the motor trajectory to be dynamically refined
by somatic states (curiosity, energy, stability, health, noradrenaline, dopamine), improving convergence
and driving Loss Delta >= 0.08 while maintaining 100% operational stability and zero throughput penalty.
"""

import sys
import os
import time
import math
import json
import logging
from typing import Tuple, List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_entity import KaryonEntity
from karyon_hardware import get_hardware_engine

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-221")


class AllostaticMotorReadoutSynthesizer(nn.Module):
    """
    Allostatically-Modulated Motor Readout Synthesizer with 100% Net2Net Zero-Delta Identity at Birth t_0.
    """
    def __init__(self, text_dim: int = 256, homeo_dim: int = 6, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.allo_gate = nn.Sequential(
            nn.Linear(homeo_dim, 32),
            nn.SiLU(),
            nn.Linear(32, text_dim),
            nn.Tanh()
        ).to(self.device)

        # Strict Net2Net Zero-Initialization at Birth t_0
        nn.init.zeros_(self.allo_gate[0].weight)
        nn.init.zeros_(self.allo_gate[0].bias)
        nn.init.zeros_(self.allo_gate[2].weight)
        nn.init.zeros_(self.allo_gate[2].bias)

    def forward(self, h_base: torch.Tensor, u_t_exp: torch.Tensor) -> torch.Tensor:
        # h_base: [S, text_dim], u_t_exp: [S, 6]
        gate = 0.20 * self.allo_gate(u_t_exp) # [S, text_dim]
        return h_base + gate * h_base


def evaluate_model(brain, entity, text_samples, num_steps=25, lr=1e-3):
    hw = get_hardware_engine()
    criterion = nn.CrossEntropyLoss(ignore_index=256)
    optimizer = torch.optim.AdamW(brain.parameters(), lr=lr, weight_decay=1e-4)

    step_losses = []
    step_fe_losses = []

    start_time = time.perf_counter()

    for step in range(num_steps):
        text = text_samples[step % len(text_samples)]
        prompt_ids = brain.tokenizer.encode(text)
        seq_t = torch.tensor([prompt_ids[:-1]], dtype=torch.long, device=hw.device)
        target_t = torch.tensor([prompt_ids[1:]], dtype=torch.long, device=hw.device)

        optimizer.zero_grad()
        tot_loss, speech_loss, fe_loss, _, _, _, _ = brain.forward_sequence(
            seq_t, target_t, entity.hu, criterion, chunk_size=seq_t.size(1), use_checkpointing=False
        )
        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()

        step_losses.append(float(speech_loss))
        step_fe_losses.append(float(fe_loss))

    elapsed = time.perf_counter() - start_time
    total_tokens = sum(len(brain.tokenizer.encode(t)) - 1 for t in text_samples[:num_steps])
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "final_loss": step_losses[-1],
        "initial_loss": step_losses[0],
        "mean_loss": sum(step_losses) / len(step_losses),
        "final_fe": step_fe_losses[-1],
        "initial_fe": step_fe_losses[0],
        "tok_per_sec": tok_per_sec
    }


def run_benchmark():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-221: ALLOSTATIC MOTOR READOUT SYNTHESIS BENCHMARK]")
    logger.info("=" * 80)

    hw = get_hardware_engine()
    logger.info(f"Hardware Acceleration Engine: {hw.device_str} (Device: {hw.device})")

    # Load corpus sample
    text_samples = [
        "The fundamental principle of Active Inference is that self-organizing systems minimize variational free energy.",
        "Karyon-CoRE operates at raw UTF-8 byte level V=258 with zero-loop parallel state-space duality.",
        "Ashby somatic homeostasis tracks curiosity, energy, stability, health, noradrenaline, and dopamine continuously.",
        "Mamba-2 GroupNorm head equalization bounds exponential state dynamics to maintain numeric stability.",
        "Theta-Gamma phase-amplitude coupling coordinates cortical micro-circuits for precise motor speech synthesis."
    ]

    # 1. Baseline Evaluation (Standard Model in karyon_soul.kcore)
    logger.info("📊 Evaluating Baseline (Standard Motor Readout)...")
    entity_base = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_base = entity_base.brain
    b_results = evaluate_model(brain_base, entity_base, text_samples, num_steps=25)

    # 2. Proposed Evaluation (EXP-221 Allostatic Motor Readout Synthesizer)
    logger.info("🧬 Injecting Allostatic Motor Readout Synthesizer into VolitionalActiveInferenceMotorHead...")
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain

    # Attach Synthesizer
    synthesizer = AllostaticMotorReadoutSynthesizer(
        text_dim=brain_prop.volitional_head.text_dim,
        homeo_dim=6,
        device_str=hw.device_str
    )
    brain_prop.volitional_head.allostatic_synthesizer = synthesizer

    # Patch compute_volitional_logits dynamically
    orig_compute_logits = brain_prop.volitional_head.compute_volitional_logits

    def patched_compute_volitional_logits(h_relaxed: torch.Tensor, u_t: torch.Tensor, byte_embed_weights: torch.Tensor) -> torch.Tensor:
        total_tokens = h_relaxed.size(0)
        if u_t.dim() == 2 and u_t.size(0) != total_tokens:
            batch_size = u_t.size(0)
            seq_len = total_tokens // batch_size
            u_t_exp = u_t.unsqueeze(1).expand(batch_size, seq_len, 6).reshape(total_tokens, 6)
        else:
            u_t_exp = u_t

        curiosity = u_t_exp[:, 0:1]
        energy = u_t_exp[:, 1:2]
        na_level = u_t_exp[:, 4:5]
        da_level = u_t_exp[:, 5:6]

        motor_gain = (1.0 + 1.0 * da_level)

        # 1. Project relaxed state to sensory manifold
        h_proj = brain_prop.volitional_head.motor_text_proj(h_relaxed) # [S, D]
        
        # 2. Apply CPG Causal Motor Receptive Field
        h_proj_seq = h_proj.unsqueeze(0).transpose(1, 2)
        try:
            h_cpg_seq = brain_prop.volitional_head.cpg_motor[0](h_proj_seq)
        except RuntimeError:
            with torch.backends.cudnn.flags(enabled=False):
                h_cpg_seq = brain_prop.volitional_head.cpg_motor[0](h_proj_seq)
        h_cpg_seq = h_cpg_seq[:, :, :total_tokens]
        h_cpg = h_cpg_seq.transpose(1, 2).squeeze(0)
        
        h_cpg_base = brain_prop.volitional_head.cpg_motor[2](brain_prop.volitional_head.cpg_motor[1](h_cpg) + h_proj)

        # EXP-221 Allostatic Readout Synthesis
        h_cpg_out = brain_prop.volitional_head.allostatic_synthesizer(h_cpg_base, u_t_exp)

        # 3. Apply Dopaminergic Precision Gain
        h_proj_gain = h_cpg_out * motor_gain
        raw_logits = F.linear(h_proj_gain, byte_embed_weights)

        # 4. Unshackled Full-Rank EFE Manifold Evaluation
        v_emb_proj = brain_prop.volitional_head.efe_motor_proj(byte_embed_weights)
        u_t_proj = brain_prop.volitional_head.efe_homeo_proj(u_t_exp)
        efe_field = brain_prop.volitional_head.efe_evaluator(v_emb_proj.unsqueeze(0) + u_t_proj.unsqueeze(1)).squeeze(-1)

        efe_mean = efe_field.mean(dim=-1, keepdim=True)
        efe_std = efe_field.std(dim=-1, keepdim=True).clamp_min(1e-5)
        efe_field_norm = (efe_field - efe_mean) / efe_std

        gamma_volition = torch.clamp(
            0.15 * curiosity + 0.10 * na_level + 0.05 * (1.0 - energy),
            min=0.01, max=0.50
        )
        volitional_logits = raw_logits - gamma_volition * efe_field_norm
        return volitional_logits

    brain_prop.volitional_head.compute_volitional_logits = patched_compute_volitional_logits

    # Verify zero-delta identity at step 0
    with torch.no_grad():
        test_h = torch.randn(10, brain_prop.hidden_dim, device=hw.device)
        test_u = torch.tensor([[0.5, 0.8, 0.9, 1.0, 0.2, 0.3]], device=hw.device).expand(10, 6)
        test_w = brain_prop.pos_embeddings.byte_embed.weight

        l_orig = orig_compute_logits(test_h, test_u, test_w)
        l_patch = patched_compute_volitional_logits(test_h, test_u, test_w)
        zero_delta = float(torch.abs(l_orig - l_patch).max())
        logger.info(f"Zero-Delta Verification at Birth t_0: Max Abs Diff = {zero_delta:.8f}")
        assert zero_delta < 1e-6, f"Net2Net Zero-Delta Identity Violated! Diff: {zero_delta}"

    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    # Telemetry Analysis & KEP Verdict
    loss_delta = b_results['final_loss'] - p_results['final_loss']
    fe_delta = b_results['final_fe'] - p_results['final_fe']
    verdict = "POSITIVE" if loss_delta >= 0.08 else "NEUTRAL"

    logger.info("=" * 80)
    logger.info("📊 === EXP-221 TELEMETRY REPORT ===")
    logger.info(f"  - Baseline Final Loss : {b_results['final_loss']:.4f} nats | Throughput: {b_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Proposed Final Loss : {p_results['final_loss']:.4f} nats | Throughput: {p_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Speech Loss Delta   : {loss_delta:+.4f} nats (Target >= +0.08)")
    logger.info(f"  - Free Energy Delta   : {fe_delta:+.4f} nats")
    logger.info(f"  - KEP Rule #2 Verdict : 🟢 {verdict}" if verdict == "POSITIVE" else f"  - KEP Rule #2 Verdict : ⚪ {verdict}")
    logger.info("=" * 80)

    results = {
        "exp_id": "EXP-221",
        "verdict": verdict,
        "base_initial_loss": b_results['initial_loss'],
        "base_final_loss": b_results['final_loss'],
        "prop_initial_loss": p_results['initial_loss'],
        "prop_final_loss": p_results['final_loss'],
        "loss_delta": loss_delta,
        "fe_delta": fe_delta,
        "tok_per_sec": p_results['tok_per_sec']
    }

    with open("exp_221_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_benchmark()
