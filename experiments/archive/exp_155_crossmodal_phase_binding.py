# experiments/exp_155_crossmodal_phase_binding.py
"""
EXP-155: Cross-Modal Co-Activation Resonance & Phase Alignment in DynamicSensoryGateway
         vs. Static Query-Key Attention Baseline

Hypothesis:
Integrating a biophysical Cross-Modal Co-Activation Resonance (Phase Alignment) mechanism
into DynamicSensoryGateway will allow semantically aligned multi-modal inputs (e.g. Text + Vision of the same concept)
to mutually amplify each other's saliency via coincidence detection. This will significantly reduce
Sensory Free Energy (variational surprise) and Epistemic Entropy (attention uncertainty) on aligned inputs,
while maintaining high robustness (suppressing noise) on misaligned inputs.

Architecture Delta:
1. `CrossModalResonanceGateway`:
   - Computes pairwise cosine similarity (phase alignment) between all active projected channels:
     R_{ij} = <z_i, z_j> / (||z_i|| * ||z_j||)
   - Computes channel activity magnitudes: A_i = ||z_i||_2
   - Mutual resonance boost: Resonance_i = sum_{j != i} R_{ij} * A_i * A_j
   - Resonance-amplified similarity: sim_i = (Query * z_i) / sqrt(D) + alpha_resonance * Resonance_i
   - Softmax attention over channels to produce unified representation w_t.
2. Direct comparative audit of Aligned vs. Misaligned multi-modal inputs across Epistemic Entropy and Attention Confidence.
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

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-155")


class CrossModalResonanceGateway(nn.Module):
    """
    DynamicSensoryGateway with Cross-Modal Co-Activation Resonance (Phase Alignment).
    """
    def __init__(self, baseline_gateway, alpha_resonance: float = 0.35):
        super().__init__()
        self.baseline = baseline_gateway
        self.alpha_resonance = alpha_resonance

    def forward(self, sensor_inputs: dict, h_prev: torch.Tensor, u_t: torch.Tensor):
        batch_size = h_prev.size(0)
        if h_prev.dim() > 2 or h_prev.size(-1) != self.baseline.hidden_dim:
            if h_prev.numel() >= batch_size * self.baseline.hidden_dim:
                h_prev = h_prev.view(batch_size, -1)[:, :self.baseline.hidden_dim]
            else:
                h_prev = torch.zeros(batch_size, self.baseline.hidden_dim, device=self.baseline.device)

        projected_channels = []
        channel_names = []

        for name, x_in in sensor_inputs.items():
            if name in self.baseline.projections:
                proj_x = self.baseline.projections[name](x_in.float() if x_in.dtype != torch.float32 else x_in)
                projected_channels.append(proj_x)
                channel_names.append(name)

        projected_channels.append(self.baseline.homeo_proj(u_t.float() if u_t.dtype != torch.float32 else u_t))
        channel_names.append('body')

        projected_channels.append(self.baseline.mind_proj(h_prev.float() if h_prev.dtype != torch.float32 else h_prev))
        channel_names.append('mind')

        stacked_channels = torch.stack(projected_channels, dim=1) # [B, N, D]
        norm_stacked = self.baseline.channel_norm(stacked_channels)

        # Pairwise Cosine Similarity (Phase Alignment) between active sensory channels
        norm_for_sim = F.normalize(norm_stacked, p=2, dim=-1) # [B, N, D]
        similarity_matrix = torch.matmul(norm_for_sim, norm_for_sim.transpose(1, 2)) # [B, N, N]

        # Activity magnitudes
        activity = norm_stacked.norm(p=2, dim=-1, keepdim=True) # [B, N, 1]
        activity_matrix = torch.matmul(activity, activity.transpose(1, 2)) # [B, N, N]

        # Mutual Resonance Matrix
        resonance_matrix = similarity_matrix * activity_matrix # [B, N, N]
        
        # Zero out self-resonance diagonal
        eye = torch.eye(resonance_matrix.size(1), device=h_prev.device).unsqueeze(0)
        resonance_matrix = resonance_matrix * (1.0 - eye)

        # Total resonance boost per channel
        resonance_boost = resonance_matrix.sum(dim=-1) # [B, N]

        # Baseline Volition Query Attention
        volition_query = self.baseline.attention_query_layer(h_prev.float() if h_prev.dtype != torch.float32 else h_prev).unsqueeze(1)
        norm_query = self.baseline.query_norm(volition_query)
        base_sim = (norm_query * norm_stacked).sum(dim=-1) / math.sqrt(self.baseline.unified_dim) # [B, N]

        # Apply Resonance Boost to similarity score
        final_sim = base_sim + self.alpha_resonance * resonance_boost

        attention_weights = F.softmax(final_sim, dim=-1)

        eps = 1e-9
        epistemic_entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-1, keepdim=True)

        w_t = (attention_weights.unsqueeze(-1) * stacked_channels).sum(dim=1)
        return w_t, attention_weights, channel_names, epistemic_entropy


def main():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-155: CROSS-MODAL CO-ACTIVATION RESONANCE BENCHMARK]")
    logger.info("=" * 80)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"⚡ Hardware Backend: {device_str.upper()}")

    # Load active entity
    kcore_path = "karyon_soul.kcore"
    entity = KaryonEntity.load(filepath=kcore_path, device=device_str)
    baseline_gateway = entity.brain.gateway

    proposed_gateway = CrossModalResonanceGateway(baseline_gateway, alpha_resonance=0.35).to(device_str)

    batch_size = 1
    hidden_dim = entity.brain.hidden_dim

    h_prev = torch.randn(batch_size, hidden_dim, device=device_str)
    u_t = entity.hu.state.clone()

    # Aligned representations
    text_aligned = torch.randn(batch_size, 256, device=device_str)
    # Vision is aligned with text (highly correlated, cosine similarity ~ 0.95)
    vision_aligned = text_aligned.clone() + torch.randn(batch_size, 256, device=device_str) * 0.05
    motor_aligned = torch.zeros(batch_size, 3, device=device_str)

    aligned_inputs = {
        'text': text_aligned,
        'vision': vision_aligned,
        'motor': motor_aligned
    }

    # Misaligned representations
    text_misaligned = torch.randn(batch_size, 256, device=device_str)
    # Vision is misaligned (random orthogonal noise)
    vision_misaligned = torch.randn(batch_size, 256, device=device_str)
    motor_misaligned = torch.zeros(batch_size, 3, device=device_str)

    misaligned_inputs = {
        'text': text_misaligned,
        'vision': vision_misaligned,
        'motor': motor_misaligned
    }

    with torch.no_grad():
        logger.info("\n>>> 1. EVALUATING BASELINE GATEWAY (Static Query-Key Attention) <<<")
        
        # Aligned
        w_t_base_al, att_base_al, _, ent_base_al = baseline_gateway(aligned_inputs, h_prev, u_t)
        # Misaligned
        w_t_base_mis, att_base_mis, _, ent_base_mis = baseline_gateway(misaligned_inputs, h_prev, u_t)

        logger.info(f"Aligned Baseline Epistemic Entropy   : {ent_base_al.mean().item():.4f}")
        logger.info(f"Misaligned Baseline Epistemic Entropy : {ent_base_mis.mean().item():.4f}")

        logger.info("\n>>> 2. EVALUATING PROPOSED GATEWAY (Cross-Modal Co-Activation Resonance) <<<")
        
        # Aligned
        w_t_prop_al, att_prop_al, _, ent_prop_al = proposed_gateway(aligned_inputs, h_prev, u_t)
        # Misaligned
        w_t_prop_mis, att_prop_mis, _, ent_prop_mis = proposed_gateway(misaligned_inputs, h_prev, u_t)

        logger.info(f"Aligned Proposed Epistemic Entropy   : {ent_prop_al.mean().item():.4f}")
        logger.info(f"Misaligned Proposed Epistemic Entropy : {ent_prop_mis.mean().item():.4f}")

    # Calculate metrics
    entropy_reduction_aligned = (ent_base_al.mean().item() - ent_prop_al.mean().item())
    entropy_change_misaligned = (ent_prop_mis.mean().item() - ent_base_mis.mean().item())

    # Verdict criteria:
    # 🟢 POSITIVE if aligned entropy is significantly reduced (higher attention confidence/binding)
    verdict = "POSITIVE" if (entropy_reduction_aligned > 0.10) else "NEUTRAL"

    logger.info("=" * 80)
    logger.info("📊 === EXP-155 EMPIRICAL TELEMETRY SUMMARY ===")
    logger.info(f"🏆 Verdict                         : 🟢 {verdict}")
    logger.info(f"📉 Aligned Epistemic Entropy Drop   : {entropy_reduction_aligned:+.4f} (Attention Confidence Gain)")
    logger.info(f"🛡️ Misaligned Entropy Change       : {entropy_change_misaligned:+.4f} (Noise Rejection)")
    logger.info(f"🎯 Aligned Attention Weights (Prop) : Text: {att_prop_al[0, 0].item():.3f} | Vision: {att_prop_al[0, 1].item():.3f} | Mind: {att_prop_al[0, 4].item():.3f}")
    logger.info("=" * 80)

    summary = {
        "exp_id": "EXP-155",
        "verdict": verdict,
        "metrics": {
            "baseline_aligned_entropy": ent_base_al.mean().item(),
            "proposed_aligned_entropy": ent_prop_al.mean().item(),
            "entropy_reduction_aligned": entropy_reduction_aligned,
            "baseline_misaligned_entropy": ent_base_mis.mean().item(),
            "proposed_misaligned_entropy": ent_prop_mis.mean().item(),
            "entropy_change_misaligned": entropy_change_misaligned
        }
    }
    with open("experiments/exp_155_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("✅ EXP-155 execution complete. Results saved to experiments/exp_155_results.json.")


if __name__ == "__main__":
    main()
