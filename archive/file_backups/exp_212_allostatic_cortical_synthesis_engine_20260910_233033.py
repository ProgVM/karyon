# experiments/exp_212_allostatic_cortical_synthesis_engine.py
"""
===============================================================================
EXP-212: Unified Allostatically-Modulated Neo-Cortical Synthesis Engine
Grounding: KEP Principle 1 (C++20 as Engine), Principle 2 (Biological Realism),
           Principle 8 (Compositional Depth), Principle 14 (Axiom of Allostatic Dynamic Forces),
           Principle 15 (Net2Net Smooth Grafting: Strict Identity at Birth).
===============================================================================
Hypothesis:
In the Grand Synthesis of Neo-Cortical vectors (`forward_sequence` line 1833, `forward` line 1212,
`generate_thought_and_speech` line 2085), five distinct information streams are integrated
into the pre-attractor cortical representation `h_combined`:
  1. `h_thalamic`: Thalamocortical routed sensory-associative trunk (weight: 1.00)
  2. `y_fast`: Fast-weight Hebbian association vector (weight: 0.20)
  3. `y_local`: Local 3-factor neuromodulated plasticity vector (weight: 0.10)
  4. `weighted_error`: Precision-weighted bottom-up predictive error residual (weight: 1.00)
  5. `topdown_prior`: Top-down hierarchical expectation prior (weight: 0.10 + 0.15 * phasic_gain)

Currently, the synthesis coefficients (1.00, 0.20, 0.10, 1.00, 0.10) are frozen static constants
except for the partial phasic gain on the prior.
Under KEP Principle 14 (Axiom of Allostatic Dynamic Forces), the biological neocortex does not sum
divergent pathways using static ratios. Under high Noradrenaline (arousal/surprise), sensory prediction
errors (`weighted_error`) and local synaptic adjustments (`y_local`, `y_fast`) must dominate, while
under high Stability and Dopamine (consolidation/confidence), top-down expectations (`topdown_prior`)
and thalamic associative highways (`h_thalamic`) must take precedence.

Replacing static summation with an Allostatically-Modulated Neo-Cortical Synthesis Engine:
  w_vec(u_t) = [1.0, 0.20, 0.10, 1.0, 0.10 + 0.15 * phasic_gain] * (1.0 + 0.20 * tanh(W_syn * u_t))
coupled with a zero-initialized Net2Net non-linear cross-stream interaction refinement (KEP Principle 15,
strictly preserving zero-delta function identity at birth t_0) will dynamically balance sensory reality,
plastic memory, and predictive expectations according to somatic homeostatic demands, accelerating
variational convergence and driving Loss Delta >= 0.08.
"""

import sys
import os
import time
import math
import json
import logging
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_entity import KaryonEntity
from karyon_hardware import get_hardware_engine

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-212")


class UnifiedAllostaticCorticalSynthesizer(nn.Module):
    """
    Unified Allostatically-Modulated Neo-Cortical Synthesis Engine (EXP-212).
    Dynamically balances Thalamocortical trunk, Fast Hebbian weights, Local Plasticity,
    Predictive Residual Errors, and Top-down Priors based on Somatic Homeostasis (u_t).
    Preserves exact mathematical identity at birth via zero-initialized Net2Net grafting.
    """
    def __init__(self, hidden_dim: int, homeo_dim: int = 6, device_str: str = 'cpu'):
        super().__init__()
        self.device = torch.device('cuda' if 'cuda' in device_str else 'cpu')
        self.hidden_dim = hidden_dim

        # Dynamic allostatic modulation network: outputs 5 stream scaling deltas in [-0.20, +0.20]
        self.allostatic_gating = nn.Sequential(
            nn.Linear(homeo_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 5),
            nn.Tanh()
        ).to(self.device)
        nn.init.zeros_(self.allostatic_gating[2].weight)
        nn.init.zeros_(self.allostatic_gating[2].bias)

        # Non-linear cross-stream refinement graft (KEP Principle 15 Net2Net Smooth Grafting)
        self.refine_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        ).to(self.device)
        nn.init.zeros_(self.refine_proj[2].weight)
        nn.init.zeros_(self.refine_proj[2].bias)

    def forward(
        self,
        h_thalamic: torch.Tensor,
        y_fast: torch.Tensor,
        y_local: torch.Tensor,
        weighted_error: torch.Tensor,
        topdown_prior: torch.Tensor,
        phasic_gain: torch.Tensor,
        u_t: torch.Tensor
    ) -> torch.Tensor:
        B = h_thalamic.size(0)
        is_3d = (h_thalamic.dim() == 3)

        if u_t.dim() == 2:
            u_flat = u_t if u_t.size(0) == B else u_t[0:1].expand(B, -1)
        else:
            u_flat = u_t.view(1, -1).expand(B, -1)

        deltas = 0.20 * self.allostatic_gating(u_flat) # [B, 5]

        scale_thalamic = 1.00 * (1.0 + deltas[:, 0:1])
        scale_fast = 0.20 * (1.0 + deltas[:, 1:2])
        scale_local = 0.10 * (1.0 + deltas[:, 2:3])
        scale_error = 1.00 * (1.0 + deltas[:, 3:4])

        if is_3d:
            scale_thalamic = scale_thalamic.unsqueeze(1)
            scale_fast = scale_fast.unsqueeze(1)
            scale_local = scale_local.unsqueeze(1)
            scale_error = scale_error.unsqueeze(1)
            delta_prior = deltas[:, 4:5].unsqueeze(1)
            pg_exp = phasic_gain.unsqueeze(1) if phasic_gain.dim() == 2 else phasic_gain
        else:
            delta_prior = deltas[:, 4:5]
            pg_exp = phasic_gain

        base_prior_weight = 0.10 + 0.15 * pg_exp
        scale_prior = base_prior_weight * (1.0 + delta_prior)

        # Primary weighted combination
        h_synthesis = (
            scale_thalamic * h_thalamic
            + scale_fast * y_fast
            + scale_local * y_local
            + scale_error * weighted_error
            + scale_prior * topdown_prior
        )

        # Zero-initialized Net2Net refinement (identity at birth)
        refined = self.refine_proj(h_synthesis)
        return h_synthesis + refined


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
    logger.info("🔬 [STARTING EXP-212: UNIFIED ALLOSTATIC CORTICAL SYNTHESIS BENCHMARK]")
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

    # 2. Proposed Evaluation (EXP-212 Unified Allostatic Cortical Synthesis)
    entity_prop = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain_prop = entity_prop.brain

    synthesizer = UnifiedAllostaticCorticalSynthesizer(
        hidden_dim=brain_prop.hidden_dim,
        homeo_dim=6,
        device_str=hw.device_str
    ).to(hw.device)
    brain_prop.allostatic_cortical_synthesizer = synthesizer

    orig_forward_multimodal_sequence = brain_prop.forward_multimodal_sequence

    def forward_multimodal_sequence_with_synthesizer(
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
        h_prev_fast = torch.zeros(batch_size, brain_prop.hidden_dim, device=brain_prop.device)
        h1_prev_last = torch.zeros(batch_size, 1, brain_prop.hidden_dim, device=brain_prop.device)
        
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

        # Continuous Locus Coeruleus Phasic Gain Modulation
        na_t = curr_u_t[:, 4:5]
        phasic_gain = brain_prop.lc_gain(na_t)

        active_slots = getattr(episodic_memory, 'max_active_cpu', 0) if episodic_memory is not None else 0
        if active_slots > 0 and episodic_memory is not None:
            q_sensory = unrolled_inputs.get('text', list(unrolled_inputs.values())[0]).view(batch_size, seq_len, -1).mean(dim=1)
            ret_mem, max_sim = episodic_memory.read(q_sensory, temperature=0.05, threshold=0.50, sigmoid_beta=12.0)
            u_mem = curr_u_t
            g_recall = brain_prop.memory_gate(u_mem)
            for name in unrolled_inputs:
                unrolled_inputs[name] = (unrolled_inputs[name].view(batch_size, seq_len, -1) + (g_recall.unsqueeze(1) * ret_mem.unsqueeze(1))).view(batch_size * seq_len, -1)

        fused_sensory_flat, modal_att = brain_prop.sensory_gateway(unrolled_inputs, curr_u_t)
        full_h_in = fused_sensory_flat.view(batch_size, seq_len, -1)

        if use_checkpointing and brain_prop.training:
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

        # EXP-212 Unified Allostatic Neo-Cortical Synthesis Engine
        h_combined = brain_prop.allostatic_cortical_synthesizer(
            h_thalamic, y_fast, y_local, weighted_error, topdown_prior, phasic_gain, effective_u_t
        )

        if hasattr(brain_prop, 'dynamic_graph') and brain_prop.dynamic_graph is not None:
            h_combined = brain_prop.dynamic_graph(h_combined, effective_u_t)

        h_flat = brain_prop.pre_attractor_norm(h_combined.contiguous().view(-1, brain_prop.hidden_dim))
        h_relaxed, commit_loss = brain_prop.attractor_head.relax_to_minima(h_flat, effective_u_t)

        volitional_logits_flat = brain_prop.compute_volitional_logits_with_grafts(
            h_relaxed, effective_u_t
        )
        logits_seq = volitional_logits_flat.view(batch_size, seq_len, -1)

        speech_loss = criterion_speech(logits_seq.view(-1, logits_seq.size(-1)), target_seq.view(-1))

        h_last = h_relaxed.view(batch_size, seq_len, -1)[:, -1, :]
        q_mu, q_logvar = brain_prop.world_model.encode(h_last)
        p_mu, p_logvar = brain_prop.world_model.prior(h_last)
        z = brain_prop.world_model.reparameterize(q_mu, q_logvar)

        d_kl = 0.5 * torch.sum(
            p_logvar - q_logvar + (q_logvar.exp() + (q_mu - p_mu).pow(2)) / (p_logvar.exp() + 1e-6) - 1.0,
            dim=-1
        ).mean()

        fe_loss = speech_loss + 0.10 * d_kl + 0.05 * commit_loss + 0.05 * error_magnitude + 0.02 * allostatic_strain
        total_loss = fe_loss

        return total_loss, float(speech_loss.item()), float(fe_loss.item()), h_relaxed, effective_u_t, None, None

    brain_prop.forward_multimodal_sequence = forward_multimodal_sequence_with_synthesizer

    logger.info("Running Proposed Unified Allostatic Cortical Synthesis Evaluation...")
    p_results = evaluate_model(brain_prop, entity_prop, text_samples, num_steps=25)

    loss_delta = b_results['final_loss'] - p_results['final_loss']
    fe_delta = b_results['final_fe'] - p_results['final_fe']

    logger.info("=" * 80)
    logger.info("📊 === EXP-212 TELEMETRY REPORT ===")
    logger.info(f"  - Baseline Final Loss : {b_results['final_loss']:.4f} nats | Throughput: {b_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Proposed Final Loss : {p_results['final_loss']:.4f} nats | Throughput: {p_results['tok_per_sec']:.1f} tok/s")
    logger.info(f"  - Loss Delta (B - P)  : {loss_delta:.4f} nats")
    logger.info(f"  - Free Energy Delta   : {fe_delta:.6f}")
    logger.info(f"  - Baseline Duration   : {b_results['elapsed']:.3f} s")
    logger.info(f"  - Proposed Duration   : {p_results['elapsed']:.3f} s")

    verdict = "POSITIVE" if (loss_delta >= 0.08 or (loss_delta >= 0.02 and p_results['tok_per_sec'] >= 0.90 * b_results['tok_per_sec'])) else "NEUTRAL"

    results = {
        "exp_id": "EXP-212",
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

    with open("experiments/exp_212_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"🏆 Final Verdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    return results


if __name__ == "__main__":
    run_benchmark()
