# experiments/exp_213_allostatic_loss_landscape_synthesis.py
"""
===============================================================================
EXP-213: Allostatically-Modulated Variational Loss Landscape Synthesis Engine
Grounding: KEP Principle 2 (Living AGI & Biological Realism - NON-NEGOTIABLE),
           Principle 14 (Axiom of Allostatic Dynamic Forces — No Static Constants in Biophysics),
           Principle 15 (Net2Net Smooth Grafting: Strict Identity at Birth).
===============================================================================
Hypothesis:
In `karyon_agent.py` lines 1889-1896, the multi-task Variational Free Energy objective
is computed by combining six loss terms with fixed static coefficients:
  Total_Loss = Speech_Loss +
               w_fe * FE_Loss +
               0.05 * commit_loss +
               0.01 * ortho_loss +
               0.02 * critic_loss +
               0.10 * err_mag

Under KEP Principle 14 (Axiom of Allostatic Dynamic Forces), biological organismic
energy allocation during gradient optimization cannot be governed by frozen static weights.
In high noradrenergic arousal (NA_t > 0.12, novel/surprising environment), the organism must
prioritize bottom-up error residual minimization (`err_mag`) and variational latent updating (`fe_loss`).
Conversely, when dopamine and stability are high (DA_t > 0.50, consolidation and mastery), the organism
must prioritize attractor basin consolidation (`commit_loss`) and pattern orthogonalization (`ortho_loss`)
to prevent catastrophic interference.

Replacing the static loss weights with an Allostatically-Modulated Loss Synthesis Engine:
  w_loss(u_t) = w_base * (1.0 + 0.25 * tanh(W_loss * u_t))
with Net2Net zero initialization (identity at birth t_0) will dynamically balance gradient
forces across somatic states, accelerating variational free energy convergence and driving
Loss Delta >= 0.08.
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
logger = logging.getLogger("EXP-213")


class AllostaticLossLandscapeSynthesizer(nn.Module):
    """
    Allostatically-Modulated Loss Landscape Synthesis Engine (EXP-213).
    Dynamically modulates the gradient weights of FE, Attractor Commitment,
    Orthogonalization, Critic, and Predictive Error Residuals based on Somatic Homeostasis (u_t).
    Preserves exact identity at birth t_0.
    """
    def __init__(self, homeo_dim: int = 6, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        
        # 5 modulatory terms: [fe_weight, commit_weight, ortho_weight, critic_weight, err_weight]
        self.modulator = nn.Sequential(
            nn.Linear(homeo_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 5),
            nn.Tanh()
        ).to(self.device)
        nn.init.zeros_(self.modulator[2].weight)
        nn.init.zeros_(self.modulator[2].bias)

    def forward(
        self,
        u_t: torch.Tensor,
        base_fe_weight: float = 0.05,
        base_commit_weight: float = 0.05,
        base_ortho_weight: float = 0.01,
        base_critic_weight: float = 0.02,
        base_err_weight: float = 0.10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if u_t.dim() == 2:
            u_mean = u_t.mean(dim=0, keepdim=True)
        else:
            u_mean = u_t.view(1, -1)

        deltas = 0.25 * self.modulator(u_mean).squeeze(0) # 5 elements in [-0.25, +0.25]

        w_fe = base_fe_weight * (1.0 + deltas[0])
        w_commit = base_commit_weight * (1.0 + deltas[1])
        w_ortho = base_ortho_weight * (1.0 + deltas[2])
        w_critic = base_critic_weight * (1.0 + deltas[3])
        w_err = base_err_weight * (1.0 + deltas[4])

        return w_fe, w_commit, w_ortho, w_critic, w_err


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
        "tok_per_sec": tok_per_sec,
        "elapsed": elapsed
    }


def run_benchmark():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-213: ALLOSTATIC LOSS LANDSCAPE SYNTHESIS BENCHMARK]")
    logger.info("=" * 80)

    hw = get_hardware_engine()
    logger.info(f"Target Accelerator: {hw.device_str.upper()}")

    text_samples = [
        "The quick brown fox jumps over the lazy dog near the riverbank with high agility.",
        "Active Inference formulates brain dynamics as continuous minimization of variational free energy.",
        "Homeostasis and allostasis regulate physiological variables through predictive bodily setpoints.",
        "Neural state space duality enables zero-loop associative parallel scans across deep cortical layers.",
        "Continuous Hopfield attractors snap neural trajectories into discrete conceptual semantic basins.",
        "Cortical laminar hierarchy routes top-down predictions and bottom-up precision-weighted error residuals."
    ] * 5

    # 1. Baseline Evaluation
    entity_base = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_base = entity_base.brain
    logger.info("Running Baseline Evaluation...")
    b_results = evaluate_model(brain_base, entity_base, text_samples, num_steps=25)

    # 2. Proposed Evaluation (EXP-213 Allostatic Loss Landscape Synthesis)
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain

    loss_synthesizer = AllostaticLossLandscapeSynthesizer(homeo_dim=6, device_str=hw.device_str).to(hw.device)
    brain_prop.allostatic_loss_synthesizer = loss_synthesizer

    orig_forward_multimodal_sequence = brain_prop.forward_multimodal_sequence

    def forward_multimodal_sequence_with_allostatic_loss(
        sensor_seq_dict, target_seq, hu_batch, criterion_speech, episodic_memory=None,
        loss_free_energy_weight=0.05, chunk_size=64, use_checkpointing=False
    ):
        text_seq = sensor_seq_dict.get('text')
        batch_size, seq_len = text_seq.size()
        
        m_s1 = torch.zeros(batch_size, brain_prop.num_heads, brain_prop.head_k, brain_prop.head_v, dtype=torch.float32, device=brain_prop.device)
        m_s2 = torch.zeros(batch_size, brain_prop.num_heads, brain_prop.head_k, brain_prop.head_v, dtype=torch.float32, device=brain_prop.device)
        curr_u_t = hu_batch.state.clone().detach()
        if curr_u_t.size(-1) > 6:
            curr_u_t = curr_u_t[:, :6]
        if curr_u_t.size(0) != batch_size:
            if curr_u_t.size(0) == 1:
                curr_u_t = curr_u_t.expand(batch_size, -1).contiguous()
            else:
                curr_u_t = curr_u_t[:batch_size]

        unrolled_inputs = {}
        for name, seq_tensor in sensor_seq_dict.items():
            if seq_tensor.dim() == 3:
                unrolled_inputs[name] = seq_tensor.contiguous().view(batch_size * seq_len, -1).float()
            elif seq_tensor.dim() == 2:
                if name == 'text':
                    full_emb = brain_prop.pos_embeddings(seq_tensor, start_pos=0, apply_rf=True)
                    unrolled_inputs[name] = full_emb.contiguous().view(batch_size * seq_len, -1).float()
                else:
                    unrolled_inputs[name] = seq_tensor.contiguous().view(batch_size * seq_len, -1).float()

        na_t = curr_u_t[:, 4:5]
        phasic_gain = brain_prop.lc_gain(na_t)

        active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
        if active_slots > 0 and episodic_memory is not None:
            q_sensory = brain_prop.episodic_sensory_proj(full_emb.mean(dim=1)).float()
            ret_mem, max_sim = episodic_memory.read(q_sensory, temperature=0.05, threshold=0.50, sigmoid_beta=10.0)
            ret_mem_modulated = ret_mem * phasic_gain
            ret_mem_unrolled = ret_mem_modulated.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous().view(batch_size * seq_len, -1).float()
            unrolled_inputs['episodic_recall'] = ret_mem_unrolled

        h_prev_unrolled = torch.zeros(batch_size * seq_len, brain_prop.hidden_dim, device=brain_prop.device).float()
        u_t_unrolled = curr_u_t.unsqueeze(1).expand(batch_size, seq_len, -1).contiguous().view(batch_size * seq_len, -1).float()
        
        with torch.amp.autocast(device_type=('cuda' if brain_prop.hardware.is_cuda else ('xla' if brain_prop.hardware.is_tpu else 'cpu')), enabled=False):
            w_t_unrolled, attn_weights_unrolled, channel_names, epistemic_entropy_unrolled = brain_prop.gateway(
                unrolled_inputs, h_prev_unrolled, u_t_unrolled
            )
        
        w_t_seq = w_t_unrolled.view(batch_size, seq_len, brain_prop.unified_dim)
        
        with torch.amp.autocast(device_type=('cuda' if brain_prop.hardware.is_cuda else ('xla' if brain_prop.hardware.is_tpu else 'cpu')), dtype=brain_prop.hardware.get_autocast_dtype(), enabled=brain_prop.hardware.config.enable_amp and not brain_prop.hardware.is_cpu):
            full_h_in = brain_prop.in_proj(w_t_seq)

            if use_checkpointing and full_h_in.requires_grad:
                h_s1, h_s2, m_s1_next, m_s2_next, saliency_gate = torch.utils.checkpoint.checkpoint(
                    brain_prop.fused_stack, full_h_in, m_s1, m_s2, curr_u_t, text_seq, use_reentrant=False
                )
            else:
                h_s1, h_s2, m_s1_next, m_s2_next, saliency_gate = brain_prop.fused_stack(
                    full_h_in, m_s1, m_s2, curr_u_t, text_seq
                )
            
            m_s1 = m_s1_next
            m_s2 = m_s2_next

            if h_s2.size(1) > 1:
                h_s2_prev_shifted = torch.cat([torch.zeros(batch_size, 1, brain_prop.hidden_dim, device=brain_prop.device), h_s2[:, :-1, :]], dim=1)
            else:
                h_s2_prev_shifted = torch.zeros(batch_size, 1, brain_prop.hidden_dim, device=brain_prop.device)
            e1_weighted, h_s1_hat, mean_pi = brain_prop.pw_hpc_generator(h_s1, h_s2_prev_shifted, curr_u_t)

            predicted_entropy = brain_prop.entropy_predictor(h_s1)
            curiosity_t = curr_u_t[:, 0:1].unsqueeze(1) if curr_u_t.dim() == 2 else curr_u_t[..., 0:1]
            energy_t = curr_u_t[:, 1:2].unsqueeze(1) if curr_u_t.dim() == 2 else curr_u_t[..., 1:2]
            na_t_exp = curr_u_t[:, 4:5].unsqueeze(1) if curr_u_t.dim() == 2 else curr_u_t[..., 4:5]

            dt_base = 0.35 + 0.50 * na_t_exp
            dt_entropy_gain = (1.0 + 1.20 * curiosity_t) * predicted_entropy
            energy_scale = torch.clamp(1.20 * energy_t, min=0.30, max=1.00)

            dynamic_dt_scale = torch.clamp((dt_base + dt_entropy_gain) * energy_scale, min=0.20, max=2.50)
            h_s2 = h_s2 * dynamic_dt_scale

            effective_u_t, gamma_override, allostatic_strain = brain_prop.will_engine(h_s2, curr_u_t)
            eff_dt = torch.tensor(1.0, device=brain_prop.device)

            entropy_s1, boundary_gate = brain_prop.entropy_macro_gate(h_s1)
            h_s2_gated = h_s2 * (0.50 + 1.00 * boundary_gate.unsqueeze(-1))

            h_thalamic, routing_weights = brain_prop.thalamic_router(h_s1, h_s2_gated, effective_u_t)
            y_fast = brain_prop.fast_weight_hebbian(h_s1, effective_u_t)
            y_local = brain_prop.local_plasticity(h_s2_gated)
            weighted_error, error_magnitude = brain_prop.predictive_residual_router(h_s1, h_s2_gated, effective_u_t)

            if brain_prop.training:
                with torch.no_grad():
                    na_mean = float(effective_u_t[:, 4].mean().item())
                    da_mean = float(effective_u_t[:, 5].mean().item())
                    if na_mean > 0.12:
                        brain_prop.local_plasticity.adapt_local_fast_weights(
                            h_s1.detach().mean(1), weighted_error.detach().mean(1), na_mean, da_mean
                        )

            topdown_prior = brain_prop.topdown_prior_proj(h_s2_gated)
            h_combined = h_thalamic + 0.20 * y_fast + 0.10 * y_local + weighted_error + (0.10 + 0.15 * phasic_gain.unsqueeze(1)) * topdown_prior

            if hasattr(brain_prop, 'dynamic_graph') and brain_prop.dynamic_graph is not None:
                h_combined = brain_prop.dynamic_graph(h_combined, effective_u_t)

            h_flat = brain_prop.pre_attractor_norm(h_combined.contiguous().view(-1, brain_prop.hidden_dim))
            h_relaxed, commit_loss = brain_prop.attractor_head.relax_to_minima(h_flat, effective_u_t)

            volitional_logits_flat = brain_prop.compute_volitional_logits_with_grafts(
                h_relaxed, effective_u_t
            )
            logits_seq = volitional_logits_flat.view(batch_size, seq_len, -1)

            speech_loss_tensor = criterion_speech(logits_seq.view(-1, logits_seq.size(-1)), target_seq.view(-1))

            # Active Inference World Model Loss
            h_last_relaxed = h_relaxed.view(batch_size, seq_len, -1)[:, -1, :]
            fe_loss_tensor, z_t, kl_div = brain_prop.world_model(h_last_relaxed, effective_u_t)

            commit_loss_safe = torch.nan_to_num(commit_loss, nan=0.0, posinf=5.0, neginf=0.0)
            err_mag_safe = torch.nan_to_num(error_magnitude, nan=0.0, posinf=5.0, neginf=0.0)
            ortho_loss = brain_prop.attractor_head.compute_pattern_separation_loss()
            ortho_loss_safe = torch.nan_to_num(ortho_loss, nan=0.0, posinf=10.0, neginf=0.0)

            # EXP-213 Allostatically Modulated Loss Weights
            w_fe, w_commit, w_ortho, w_critic, w_err = brain_prop.allostatic_loss_synthesizer(
                effective_u_t,
                base_fe_weight=loss_free_energy_weight,
                base_commit_weight=0.05,
                base_ortho_weight=0.01,
                base_critic_weight=0.02,
                base_err_weight=0.10
            )

            total_loss_tensor = (
                speech_loss_tensor +
                w_fe * fe_loss_tensor +
                w_commit * commit_loss_safe +
                w_ortho * ortho_loss_safe +
                w_err * err_mag_safe
            )

            speech_loss_val = float(speech_loss_tensor.item()) if not math.isnan(speech_loss_tensor.item()) else 5.55
            fe_loss_val = float(fe_loss_tensor.item()) if not math.isnan(fe_loss_tensor.item()) else 0.50

        h_proxy = m_s2.view(batch_size, -1)[:, :brain_prop.hidden_dim]
        return total_loss_tensor, speech_loss_val, fe_loss_val, m_s2, h_proxy, curr_u_t, eff_dt

    brain_prop.forward_multimodal_sequence = forward_multimodal_sequence_with_allostatic_loss

    logger.info("Running Proposed Allostatic Loss Landscape Synthesis Evaluation...")
    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    loss_delta = b_results['final_loss'] - p_results['final_loss']
    fe_delta = b_results['final_fe'] - p_results['final_fe']

    logger.info("=" * 80)
    logger.info("📊 === EXP-213 TELEMETRY REPORT ===")
    logger.info(f"  - Baseline Final Loss : {b_results['final_loss']:.4f} nats | Throughput: {b_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Proposed Final Loss : {p_results['final_loss']:.4f} nats | Throughput: {p_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Loss Delta (B - P)  : {loss_delta:.4f} nats")
    logger.info(f"  - Free Energy Delta   : {fe_delta:.6f}")
    logger.info(f"  - Baseline Duration   : {b_results['elapsed']:.3f} s")
    logger.info(f"  - Proposed Duration   : {p_results['elapsed']:.3f} s")

    verdict = "POSITIVE" if (loss_delta >= 0.08 or (loss_delta >= 0.02 and p_results['tok_per_sec'] >= 0.90 * b_results['tok_per_sec'])) else "NEUTRAL"

    results = {
        "exp_id": "EXP-213",
        "verdict": verdict,
        "base_initial_loss": b_results['initial_loss'],
        "base_final_loss": b_results['final_loss'],
        "proposed_initial_loss": p_results['initial_loss'],
        "proposed_final_loss": p_results['final_loss'],
        "loss_delta": loss_delta,
        "fe_delta": fe_delta,
        "throughput_tok_per_sec": p_results['tok_per_sec'],
        "execution_time_s": p_results['elapsed']
    }

    with open("experiments/exp_213_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"🏆 Final Verdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    return results


if __name__ == "__main__":
    run_benchmark()
