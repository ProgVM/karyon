# experiments/exp_153_tsodyks_markram_attractor_fatigue.py
"""
EXP-153: Tsodyks-Markram Short-Term Synaptic Depression & Attractor Fatigue in Hopfield State Dynamics

Hypothesis:
Incorporate biophysical Tsodyks-Markram Short-Term Depression (STD) and Spike-Frequency Adaptation
(Calcium-activated Potassium AHP currents) directly into the Hopfield Attractor Basin energy landscape
and motor refractory trace.
This will:
1. Elevate energy boundaries around recently visited concept basins (preventing semantic attractor trapping).
2. Eliminate repetitive word loops ("an analogy and analyze and an analogy...") in closed-loop generation.
3. Increase lexical diversity and semantic entropy without degrading phonotactic correctness.

Architecture Delta:
1. `TsodyksMarkramHopfieldAttractor`:
   - Active transmitter fraction: R_{t+1} = R_t + (1 - R_t)/tau_rec - U_0 * R_t * E_t
   - Facilitated calcium state: u_{t+1} = u_t + (U_0 - u_t)/tau_fac + U_0 * (1 - u_t) * E_t
   - Effective attractor gain: G_{eff, i} = R_i * u_i * (1.0 + 1.5 * DA)
   - Dynamic Attractor Energy Landscape:
     E(h_t, b_i) = -G_{eff, i} * <h_t, b_i> + gamma_fatigue * AHP_trace_i
2. `BiophysicalGABAActionSelector` with Tsodyks-Markram Motor Refractory Adaptation:
   - Dynamic byte-level habituation decay + AHP hyperpolarization
"""

import sys
import os
import time
import math
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_entity import KaryonEntity
from karyon_agent import CoREAgent
from karyon_config import CoREConfig
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-153")


class TsodyksMarkramHopfieldAttractor(nn.Module):
    """
    Tsodyks-Markram Synaptic Depression & Spike-Frequency Adaptation Hopfield Head.
    """
    def __init__(self, hidden_dim: int = 768, num_attractors: int = 256, tau_rec: float = 12.0, tau_fac: float = 8.0, u0: float = 0.20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_attractors = num_attractors
        self.tau_rec = tau_rec
        self.tau_fac = tau_fac
        self.u0 = u0
        self.scale = 1.0 / math.sqrt(hidden_dim)

        self.attractor_basins = nn.Parameter(torch.randn(num_attractors, hidden_dim) * 0.05)
        self.register_buffer("R_state", torch.ones(num_attractors)) # Vesicle availability R in [0, 1]
        self.register_buffer("u_state", torch.full((num_attractors,), u0)) # Facilitation u
        self.register_buffer("ahp_trace", torch.zeros(num_attractors)) # Afterhyperpolarization AHP
        self.norm = nn.LayerNorm(hidden_dim)

    def reset_states(self):
        self.R_state.fill_(1.0)
        self.u_state.fill_(self.u0)
        self.ahp_trace.zero_()

    def relax_to_minima(self, h_state: torch.Tensor, hu_st: torch.Tensor) -> torch.Tensor:
        """
        h_state: [1, H]
        hu_st: [1, 6]
        """
        device = h_state.device
        da_val = float(hu_st[0, 5].detach()) if hu_st is not None else 0.20
        na_val = float(hu_st[0, 4].detach()) if hu_st is not None else 0.10

        # 1. Base similarity to all 256 Hopfield Attractor Basins
        sim = torch.matmul(h_state, self.attractor_basins.t()) * self.scale # [1, 256]

        # 2. Tsodyks-Markram Effective Synaptic Strength
        # G_eff = R * u * (1 + 1.5 * DA)
        g_eff = (self.R_state * self.u_state).unsqueeze(0) * (1.0 + 1.5 * da_val)

        # 3. Calcium-activated Potassium AHP Fatigue Penalty
        ahp_penalty = 1.80 * self.ahp_trace.unsqueeze(0)

        # 4. Modulated Energy Landscape
        fatigued_sim = sim * g_eff - ahp_penalty

        attn_weights = F.softmax(fatigued_sim * (1.0 + 1.2 * na_val), dim=-1) # [1, 256]

        # 5. Tsodyks-Markram State Transitions (No Grad for buffer updates)
        with torch.no_grad():
            E_t = attn_weights.squeeze(0) # [256]
            
            # dR/dt = (1 - R)/tau_rec - u * R * E
            dR = (1.0 - self.R_state) / self.tau_rec - self.u_state * self.R_state * E_t
            self.R_state.copy_(torch.clamp(self.R_state + dR, 0.05, 1.0))

            # du/dt = (U0 - u)/tau_fac + U0 * (1 - u) * E
            du = (self.u0 - self.u_state) / self.tau_fac + self.u0 * (1.0 - self.u_state) * E_t
            self.u_state.copy_(torch.clamp(self.u_state + du, self.u0, 1.0))

            # dAHP/dt = -AHP / tau_ahp + 1.2 * E
            dAHP = -self.ahp_trace / 15.0 + 1.2 * E_t
            self.ahp_trace.copy_(torch.clamp(self.ahp_trace + dAHP, 0.0, 5.0))

        # 6. Attractor Shift
        attractor_shift = torch.matmul(attn_weights, self.attractor_basins)
        h_relaxed = self.norm(h_state + 0.25 * attractor_shift)
        return h_relaxed


def calculate_type_token_ratio(text: str) -> float:
    """Calculates Type-Token Ratio (TTR) as a measure of lexical diversity."""
    words = [w.lower().strip(".,!?\"'") for w in text.split() if w.strip()]
    if not words:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)


def calculate_repetition_rate(text: str) -> float:
    """Calculates 3-gram repetition rate."""
    words = [w.lower().strip(".,!?\"'") for w in text.split() if w.strip()]
    if len(words) < 3:
        return 0.0
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    repeated = len(trigrams) - len(set(trigrams))
    return repeated / len(trigrams)


def main():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-153: TSODYKS-MARKRAM SYNAPTIC DEPRESSION & ATTRACTOR FATIGUE BENCHMARK]")
    logger.info("=" * 80)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"⚡ Hardware Backend: {device_str.upper()}")

    # Load active entity
    kcore_path = "karyon_soul.kcore"
    entity = KaryonEntity.load(filepath=kcore_path, device=device_str)

    prompts = [
        "User: What is the primary source of energy for Earth?\nKaryon:",
        "User: Tell me a short story about a brave knight.\nKaryon:",
        "Question: What is photosynthesis?\nAnswer:",
        "Problem: Write a Python function that returns the square of a number.\nSolution:\n"
    ]

    logger.info("\n>>> Phase 1: Baseline Closed-Loop Dialogue Generation (Without TM Depression) <<<")
    baseline_samples = []
    for p in prompts:
        gen_events = entity.interact(user_input=p, max_tokens=100)
        chars = [ev.get("text", "") for ev in gen_events if ev.get("status") == "token"]
        text = "".join(chars).strip()
        baseline_samples.append(text)
        logger.info(f"PROMPT: {p.strip().replace(chr(10), ' ')}")
        logger.info(f"BASELINE OUTPUT:\n   -> \"{text}\"\n")

    # Instantiate Tsodyks-Markram Hopfield Head
    tm_hopfield = TsodyksMarkramHopfieldAttractor(hidden_dim=entity.brain.hidden_dim, num_attractors=256).to(device_str)
    # Initialize basins from brain's attractor head
    with torch.no_grad():
        tm_hopfield.attractor_basins.copy_(entity.brain.attractor_head.attractor_basins.detach())

    logger.info("\n>>> Phase 2: Proposed Generation with Tsodyks-Markram Attractor Fatigue <<<")
    proposed_samples = []

    for p in prompts:
        tm_hopfield.reset_states()
        prompt_ids = [ord(c) for c in p]
        rolling_ids = list(prompt_ids)
        hu_st = entity.hu.state.clone()
        refractory_trace = torch.zeros(1, entity.brain.text_gen_dim, device=device_str)
        
        t_start = time.perf_counter()
        
        for step in range(100):
            ctx_t = torch.tensor([rolling_ids[-512:]], dtype=torch.long, device=device_str)
            ctx_len = ctx_t.size(1)
            embs = entity.brain.pos_embeddings(ctx_t, start_pos=0, apply_rf=True)
            unrolled = {'text': embs.view(ctx_len, -1).float()}
            h_prev = torch.zeros(ctx_len, entity.brain.hidden_dim, device=device_str)
            u_t = hu_st.unsqueeze(1).expand(1, ctx_len, -1).contiguous().view(ctx_len, -1).float()
            
            w_t, _, _, _ = entity.brain.gateway(unrolled, h_prev, u_t)
            w_seq = w_t.view(1, ctx_len, entity.brain.unified_dim)
            h_in = entity.brain.in_proj(w_seq)
            
            m1 = torch.zeros(1, entity.brain.num_heads, entity.brain.head_k, entity.brain.head_v, device=device_str)
            m2 = torch.zeros(1, entity.brain.num_heads, entity.brain.head_k, entity.brain.head_v, device=device_str)
            h_s1, h_s2, _, _, _ = entity.brain.fused_stack(h_in, m1, m2, hu_st, ctx_t)
            
            h_s1_last = h_s1[:, -1:, :]
            h_s2_last = h_s2[:, -1:, :]
            
            h_thal, _ = entity.brain.thalamic_router(h_s1_last, h_s2_last, hu_st)
            y_fast = entity.brain.fast_weight_hebbian(h_s1, hu_st)[:, -1:, :]
            w_err, _ = entity.brain.predictive_residual_router(h_s1_last, h_s2_last, hu_st)
            topdown = entity.brain.topdown_prior_proj(h_s2_last)
            h_comb = h_thal + 0.20 * y_fast + w_err + 0.10 * topdown
            
            # Apply Tsodyks-Markram Hopfield Relaxation
            h_relaxed = tm_hopfield.relax_to_minima(h_comb.view(1, entity.brain.hidden_dim), hu_st)
            raw_logits = entity.brain.volitional_head.compute_volitional_logits(h_relaxed, hu_st, entity.brain.pos_embeddings.byte_embed.weight)
            
            # Somatic byte penalty
            somatic_byte_penalty = torch.zeros(1, entity.brain.text_gen_dim, device=device_str)
            somatic_byte_penalty[0, 256] = 12.0
            somatic_byte_penalty[0, :9] = 10.0
            somatic_byte_penalty[0, 11:13] = 10.0
            somatic_byte_penalty[0, 14:32] = 10.0
            somatic_byte_penalty[0, 127] = 8.0
            
            logits = raw_logits - somatic_byte_penalty - 0.35 * refractory_trace
            p_dist = F.softmax(logits, dim=-1)
            entropy = -(p_dist * torch.log(p_dist + 1e-9)).sum(dim=-1)
            entropy_val = float(entropy.mean().item())
            
            # GABA Shunting Action Selection
            curiosity_val = float(hu_st[0, 0].detach())
            stability_val = float(hu_st[0, 2].detach())
            na_val = float(hu_st[0, 4].detach())
            da_val = float(hu_st[0, 5].detach())

            delta_gaba = 3.20 * (1.0 + 0.40 * curiosity_val) / (1.0 + 1.60 * da_val + 1.20 * na_val)
            z_max = torch.max(logits, dim=-1, keepdim=True).values
            shunting_threshold = z_max - delta_gaba

            suprathreshold_mask = (logits >= shunting_threshold)
            beta_eff = 2.80 * (1.0 + 1.80 * na_val + 1.20 * da_val)
            scaled_logits = logits * beta_eff

            shunted_logits = scaled_logits.masked_fill(~suprathreshold_mask, -1e9)
            probs = F.softmax(shunted_logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            prob_sum = probs.sum(dim=-1, keepdim=True)
            
            if (prob_sum <= 0).any():
                nxt = int(torch.argmax(logits, dim=-1))
            else:
                probs = probs / prob_sum
                nxt = int(torch.multinomial(probs, 1).squeeze(0))
                
            rolling_ids.append(nxt)
            refractory_trace = 0.72 * refractory_trace
            refractory_trace[0, nxt] += 1.0
            
            if nxt == 257 or (nxt == 10 and len(rolling_ids) > len(prompt_ids) + 5 and rolling_ids[-2:] == [10, 10]):
                break

        gen_tokens = rolling_ids[len(prompt_ids):]
        text = bytes(gen_tokens).decode('utf-8', errors='replace')
        proposed_samples.append(text)
        logger.info(f"PROMPT: {p.strip().replace(chr(10), ' ')}")
        logger.info(f"TM PROPOSED OUTPUT:\n   -> \"{text}\"\n")

    # Evaluate Comparative Diversity & Repetition Metrics
    base_ttr = sum(calculate_type_token_ratio(s) for s in baseline_samples) / len(baseline_samples)
    prop_ttr = sum(calculate_type_token_ratio(s) for s in proposed_samples) / len(proposed_samples)

    base_rep = sum(calculate_repetition_rate(s) for s in baseline_samples) / len(baseline_samples)
    prop_rep = sum(calculate_repetition_rate(s) for s in proposed_samples) / len(proposed_samples)

    ttr_gain_pct = (prop_ttr - base_ttr) / max(base_ttr, 1e-5) * 100.0
    rep_reduction_pct = (base_rep - prop_rep) / max(base_rep, 1e-5) * 100.0

    verdict = "POSITIVE" if (prop_ttr > base_ttr and prop_rep < base_rep) else "NEUTRAL"

    logger.info("=" * 80)
    logger.info("📊 === EXP-153 EMPIRICAL TELEMETRY SUMMARY ===")
    logger.info(f"🏆 Verdict                         : 🟢 {verdict}")
    logger.info(f"📚 Baseline Type-Token Ratio (TTR)  : {base_ttr:.4f}")
    logger.info(f"🌿 Proposed TM TTR                 : {prop_ttr:.4f} ({ttr_gain_pct:+.2f}% Lexical Diversity)")
    logger.info(f"🔄 Baseline 3-gram Repetition Rate : {base_rep:.4f}")
    logger.info(f"🛡️ Proposed TM Repetition Rate     : {prop_rep:.4f} ({rep_reduction_pct:+.2f}% Repetition Reduction)")
    logger.info("=" * 80)

    summary = {
        "exp_id": "EXP-153",
        "verdict": verdict,
        "metrics": {
            "baseline_ttr": base_ttr,
            "proposed_tm_ttr": prop_ttr,
            "ttr_gain_pct": ttr_gain_pct,
            "baseline_3gram_rep_rate": base_rep,
            "proposed_3gram_rep_rate": prop_rep,
            "rep_reduction_pct": rep_reduction_pct
        },
        "baseline_samples": baseline_samples,
        "proposed_samples": proposed_samples
    }
    with open("experiments/exp_153_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("✅ EXP-153 execution complete. Results saved to experiments/exp_153_results.json.")


if __name__ == "__main__":
    main()
