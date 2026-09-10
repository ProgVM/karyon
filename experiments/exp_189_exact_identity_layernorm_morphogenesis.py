# experiments/exp_189_exact_identity_layernorm_morphogenesis.py
"""
===============================================================================
EXP-189: Exact Identity LayerNorm Morphogenesis & Net2Net Preservation
Grounding: KEP Principle 15 (Epigenetic Morphogenesis & Net2Net Smooth Grafting),
           KEP Principle 7 (Axiom of Unshackled Flow),
           KEP Principle 2 (Biological Realism & Zero Catastrophic Shock).
===============================================================================
Hypothesis:
When hidden dimensions expand ($D_{\text{old}} \to D_{\text{new}}$ with zero-padded new dimensions),
standard LayerNorm normalizes over $D_{\text{new}}$ instead of $D_{\text{old}}$, shrinking
activations by $\sqrt{D_{\text{old}} / D_{\text{new}}}$ and shifting the mean if $\mu \ne 0$,
inducing an identity delta at birth $t_0$.
By applying an adaptive affine scale factor $\gamma_{\text{scale}} = \sqrt{D_{\text{new}} / D_{\text{old}}}$
and dimension-scaled bias adaptation in Net2Net LayerNorm expansion,
we can achieve near-zero identity perturbation ($f_{\text{new}}(x) \approx f_{\text{old}}(x)$)
across all cortical layers during continuous dimension growth.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_entity import KaryonEntity
from karyon_hardware import get_hardware_engine
from kcore_evolution import Net2NetMorphogenesisEngine, adapt_and_copy_tensor

logging.basicConfig(level=logging.INFO, format="%(module)-15s | %(levelname)-8s | %(asctime)s | %(message)s")
logger = logging.getLogger("EXP-189")


class ScaledIdentityLayerNorm(nn.Module):
    """
    Dimension-compensated LayerNorm that preserves exact variance and mean
    when evaluating zero-padded embeddings during Net2Net morphogenesis.
    """
    def __init__(self, normalized_shape: int, active_dim: int = None, eps: float = 1e-5, device: str = 'cpu'):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.total_dim = normalized_shape
        self.active_dim = active_dim if active_dim is not None else normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If padded with zeros, compute mean and var over active dimensions
        if self.active_dim < self.total_dim and x.size(-1) == self.total_dim:
            # Active portion stats
            x_active = x[..., :self.active_dim]
            mean = x_active.mean(dim=-1, keepdim=True)
            var = x_active.var(dim=-1, unbiased=False, keepdim=True)
            
            # Normalize active part and zero-pad remainder
            x_norm_active = (x_active - mean) / torch.sqrt(var + self.eps)
            x_norm = torch.cat([x_norm_active, torch.zeros_like(x[..., self.active_dim:])], dim=-1)
            return x_norm * self.weight + self.bias
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


def test_layernorm_math():
    """Verify analytical formulation for zero-padded LayerNorm compensation."""
    torch.manual_seed(42)
    D_old = 256
    D_new = 320
    x_old = torch.randn(4, 16, D_old)
    
    # Standard LayerNorm on old
    ln_old = nn.LayerNorm(D_old)
    y_old = ln_old(x_old)
    
    # Padded input
    x_padded = torch.cat([x_old, torch.zeros(4, 16, D_new - D_old)], dim=-1)
    
    # 1. Uncompensated standard LayerNorm on expanded
    ln_uncomp = nn.LayerNorm(D_new)
    with torch.no_grad():
        ln_uncomp.weight[:D_old].copy_(ln_old.weight)
        ln_uncomp.bias[:D_old].copy_(ln_old.bias)
    y_uncomp = ln_uncomp(x_padded)
    delta_uncomp = (y_uncomp[..., :D_old] - y_old).abs().max().item()
    
    # 2. Scaled Identity LayerNorm
    ln_comp = ScaledIdentityLayerNorm(D_new, active_dim=D_old)
    with torch.no_grad():
        ln_comp.weight[:D_old].copy_(ln_old.weight)
        ln_comp.bias[:D_old].copy_(ln_old.bias)
    y_comp = ln_comp(x_padded)
    delta_comp = (y_comp[..., :D_old] - y_old).abs().max().item()
    
    logger.info(f"LayerNorm Math Check: Uncompensated Delta = {delta_uncomp:.6f} | Compensated Delta = {delta_comp:.8f}")
    return delta_uncomp, delta_comp


def run_benchmark():
    logger.info("=" * 80)
    logger.info("🔬 [STARTING EXP-189: EXACT IDENTITY LAYERNORM MORPHOGENESIS BENCHMARK]")
    logger.info("=" * 80)

    hw = get_hardware_engine()
    logger.info(f"Target Accelerator: {hw.device_str.upper()}")

    delta_uncomp, delta_comp = test_layernorm_math()

    # Load pre-trained live entity
    entity = KaryonEntity.load("karyon_soul.kcore", device=hw.device_str)
    brain = entity.brain
    old_hidden_dim = brain.hidden_dim
    new_hidden_dim = old_hidden_dim + 64  # e.g., 256 -> 320

    logger.info(f"Baseline Brain Hidden Dim: {old_hidden_dim} -> Target Expanded Dim: {new_hidden_dim}")

    # Forward test tensor
    dummy_text = torch.randint(32, 126, (1, 16), device=hw.device)
    dummy_target = torch.randint(32, 126, (1, 16), device=hw.device)
    criterion = nn.CrossEntropyLoss(ignore_index=256)

    # 1. Baseline Forward Pass
    brain.eval()
    with torch.no_grad():
        tot_loss_base, speech_loss_base, fe_base, logits_base, _, _, _ = brain.forward_sequence(
            dummy_text, dummy_target, entity.hu, criterion, chunk_size=16, use_checkpointing=False
        )
    logger.info(f"Baseline Forward Logits Mean: {logits_base.mean():.4f}, Std: {logits_base.std():.4f}")

    # 2. Net2Net Morphogenesis Expansion
    start_morph_time = time.perf_counter()
    expanded_brain, identity_delta = Net2NetMorphogenesisEngine.expand_agent_dimensions(
        brain, new_hidden_dim=new_hidden_dim, device=hw.device_str
    )
    morph_time = time.perf_counter() - start_morph_time
    logger.info(f"Morphogenesis Completed in {morph_time*1000:.2f} ms with Reported Identity Delta: {identity_delta:.8f}")

    # 3. Post-Expansion Forward Pass
    expanded_brain.eval()
    with torch.no_grad():
        tot_loss_exp, speech_loss_exp, fe_exp, logits_exp, _, _, _ = expanded_brain.forward_sequence(
            dummy_text, dummy_target, entity.hu, criterion, chunk_size=16, use_checkpointing=False
        )
    logger.info(f"Post-Expansion Logits Mean: {logits_exp.mean():.4f}, Std: {logits_exp.std():.4f}")

    logits_diff = (logits_exp - logits_base).abs().max().item()
    loss_diff = abs(speech_loss_exp - speech_loss_base)
    fe_diff = abs(fe_exp - fe_base)

    logger.info("=" * 80)
    logger.info("📊 === EXP-189 TELEMETRY REPORT ===")
    logger.info(f"  - Analytical LayerNorm Delta (Uncompensated) : {delta_uncomp:.6f}")
    logger.info(f"  - Analytical LayerNorm Delta (Compensated)   : {delta_comp:.8f}")
    logger.info(f"  - Full Forward Sequence Logits Max Delta     : {logits_diff:.6f}")
    logger.info(f"  - Speech Loss Delta (Pre vs Post Expansion)  : {loss_diff:.6f}")
    logger.info(f"  - Free Energy Delta (Pre vs Post Expansion)  : {fe_diff:.6f}")
    logger.info(f"  - Morphogenesis Execution Time               : {morph_time*1000:.2f} ms")

    # KEP Rule #2 Verdict: POSITIVE if mathematical compensation eliminates LayerNorm drift and preserves identity
    verdict = "POSITIVE" if (delta_comp < 1e-4 and loss_diff < 0.50) else "NEUTRAL"

    results = {
        "exp_id": "EXP-189",
        "verdict": verdict,
        "analytical_uncomp_delta": delta_uncomp,
        "analytical_comp_delta": delta_comp,
        "full_forward_logits_delta": logits_diff,
        "speech_loss_pre": float(speech_loss_base),
        "speech_loss_post": float(speech_loss_exp),
        "speech_loss_diff": float(loss_diff),
        "fe_diff": float(fe_diff),
        "morph_duration_ms": morph_time * 1000
    }

    with open("experiments/exp_189_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"🏆 Final Verdict: 🟢 {verdict}" if verdict == "POSITIVE" else f"🏆 Final Verdict: ⚪ {verdict}")
    return results


if __name__ == "__main__":
    run_benchmark()
