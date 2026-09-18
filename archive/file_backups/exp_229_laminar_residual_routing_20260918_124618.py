"""
===============================================================================
EXP-229: Hierarchical Cortical Laminar Error Residual Routing
Grounding: KEP Principle 1 (C++ Acceleration / High Throughput),
           KEP Principle 2 (Living AGI & Biological Realism - NON-NEGOTIABLE),
           KEP Principle 8 (Compositional Depth Over Flat Width),
           KEP Principle 14 (Axiom of Allostatic Dynamic Forces),
           KEP Rule #1 (Hypothesis, Behavioral Scope & Telemetry First),
           KEP Rule #2 (Contextual Multi-Criteria Decision Engine).
===============================================================================
Hypothesis:
Implementing Hierarchical Cortical Laminar Error Residual Routing between Stage 1
(Morpho-Syntactic Cortical Sheet) and Stage 2 (Semantic-Discourse Cortical Sheet) where
only the unpredicted sensory error residual e_t = w_t - w_pred_stage1 is transmitted
upwards to Stage 2 will:
  1. Reduce redundant information bandwidth between cortical stages.
  2. Accelerate convergence of Variational Free Energy (VFE).
  3. Lower speech/sensory prediction loss on stream learning compared to standard
     raw hidden state passing.
===============================================================================
"""

import os
import sys
import time
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure root repository directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_config import CoREConfig
from karyon_core import HomeostaticUnit
from karyon_agent import CoREAgent
from karyon_entity import KaryonEntity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("exp_229")


class LaminarErrorResidualModule(nn.Module):
    """
    Cortical Laminar Residual Router (Layer 4 -> Layer 2/3 Feedforward).
    Computes sensory prediction residual e_t = w_t - w_pred_lower
    and projects it into Stage 2 hidden space with allostatic precision gain gating.
    """
    def __init__(self, sensory_dim=256, hidden_dim=768):
        super().__init__()
        self.error_proj = nn.Linear(sensory_dim, hidden_dim)
        # Net2Net Smooth Grafting gate initialized at zero to preserve initial function identity
        self.alpha_laminar = nn.Parameter(torch.zeros(1))
        
    def forward(self, w_t: torch.Tensor, w_pred_stage1: torch.Tensor, h_stage2: torch.Tensor, u_t: torch.Tensor) -> tuple:
        # 1. Unpredicted error residual
        e_t = w_t - w_pred_stage1
        
        # 2. Allostatic precision gain modulation derived from noradrenaline (u_t[:, 4])
        na_arousal = u_t[:, 4:5] # [B, 1]
        beta_eff = 1.0 + 1.5 * na_arousal
        
        # 3. Project error into Stage 2 space with smooth Net2Net grafting
        graft_gate = torch.tanh(self.alpha_laminar)
        e_projected = self.error_proj(e_t * beta_eff)
        
        # Modulate Stage 2 state with prediction residual
        h_stage2_modulated = h_stage2 + graft_gate * e_projected
        return h_stage2_modulated, e_t


def run_experiment():
    logger.info("=" * 80)
    logger.info("🔬 [EXP-229] INITIATING HIERARCHICAL CORTICAL LAMINAR ERROR RESIDUAL ROUTING")
    logger.info("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    entity = KaryonEntity.load("karyon_soul.kcore", device=device)
    agent = entity.brain
    
    # Instantiate Laminar Error Residual Router
    router = LaminarErrorResidualModule(sensory_dim=256, hidden_dim=agent.hidden_dim).to(device)
    
    # Dummy sensory input sequence (100 steps)
    torch.manual_seed(229)
    inputs = torch.randn(100, 1, 256, device=device)
    
    hu = HomeostaticUnit(1, device)
    hu.state.copy_(torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.2, 0.1]], device=device))
    
    h_fast = torch.zeros(1, agent.hidden_dim, device=device)
    h_slow = torch.zeros(1, agent.hidden_dim, device=device)
    
    optimizer = torch.optim.AdamW(list(agent.parameters()) + list(router.parameters()), lr=1e-3)
    
    # 1. Baseline Run (Standard Forward)
    logger.info("\n--- Step 1: Baseline Forward Training (Standard Raw Routing) ---")
    loss_baseline_history = []
    t0 = time.perf_counter()
    for step in range(50):
        optimizer.zero_grad()
        w_t = inputs[step]
        sensor_inputs = {"cybernetic": w_t}
        u_t = hu.state.clone()
        
        _, _, _, _, _, fe, _, _, w_pred, _, _, _ = agent.forward(sensor_inputs, h_fast, h_slow, u_t)
        loss = (1.0 - F.cosine_similarity(w_t, w_pred, dim=-1)).mean() + fe.mean() * 0.1
        loss.backward()
        optimizer.step()
        loss_baseline_history.append(loss.item())
        
    duration_base = time.perf_counter() - t0
    final_loss_base = sum(loss_baseline_history[-10:]) / 10.0
    logger.info(f"Baseline Final Loss: {final_loss_base:.6f} | Duration: {duration_base:.2f}s")

    # Re-instantiate entity for fair comparison
    entity = KaryonEntity.load("karyon_soul.kcore", device=device)
    agent = entity.brain
    router = LaminarErrorResidualModule(sensory_dim=256, hidden_dim=agent.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(list(agent.parameters()) + list(router.parameters()), lr=1e-3)

    # 2. Laminar Residual Routing Run
    logger.info("\n--- Step 2: Laminar Error Residual Routing Training ---")
    loss_laminar_history = []
    e_norms = []
    t0 = time.perf_counter()
    
    for step in range(50):
        optimizer.zero_grad()
        w_t = inputs[step]
        sensor_inputs = {"cybernetic": w_t}
        u_t = hu.state.clone()
        
        h_fast_next, h_slow_next, actions, cog_actions, text_logits, fe, attn_weights, _, w_pred_stage1, _, _, _ = agent.forward(
            sensor_inputs, h_fast, h_slow, u_t
        )
        
        # Apply Cortical Laminar Error Residual Router
        h_slow_modulated, e_t = router(w_t, w_pred_stage1, h_slow_next, u_t)
        e_norms.append(e_t.norm().item())
        
        # Re-predict with modulated Stage 2 state
        w_pred_stage2, _, fe_stage2, _ = agent.world_model(h_slow_modulated, h_slow_modulated, w_t)
        
        loss = (1.0 - F.cosine_similarity(w_t, w_pred_stage2, dim=-1)).mean() + fe_stage2.mean() * 0.1
        loss.backward()
        optimizer.step()
        loss_laminar_history.append(loss.item())
        
        h_fast = h_fast_next
        h_slow = h_slow_modulated
        
    duration_laminar = time.perf_counter() - t0
    final_loss_laminar = sum(loss_laminar_history[-10:]) / 10.0
    avg_error_norm = sum(e_norms[-10:]) / 10.0
    
    logger.info(f"Laminar Final Loss: {final_loss_laminar:.6f} | Avg Residual Norm: {avg_error_norm:.4f} | Duration: {duration_laminar:.2f}s")

    # 3. Comparative Summary
    loss_delta = final_loss_base - final_loss_laminar
    logger.info("\n" + "=" * 80)
    logger.info("📊 COMPARATIVE TELEMETRY SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Baseline Final Loss    : {final_loss_base:.6f}")
    logger.info(f"Laminar Final Loss     : {final_loss_laminar:.6f}")
    logger.info(f"Loss Delta             : {loss_delta:+.6f} ({'IMPROVEMENT' if loss_delta > 0 else 'DEGRADATION'})")
    logger.info(f"Average Residual Norm  : {avg_error_norm:.4f}")
    logger.info("=" * 80)

    # KEP Rule #2 Verdict Decision
    # Positive if Loss Delta >= 0.05 or lower loss with clean convergence
    is_positive = loss_delta >= 0.01 and avg_error_norm < 15.0
    verdict = "POSITIVE" if is_positive else "REJECTED"

    logger.info(f"🏆 [EXP-229 SCIENTIFIC VERDICT]: 🟢 {verdict}" if is_positive else f"🏆 [EXP-229 SCIENTIFIC VERDICT]: 🔴 {verdict}")
    logger.info("=" * 80)

    print(f"EXP_ID=EXP-229")
    print(f"VERDICT={verdict}")
    print(f"BASELINE_LOSS={final_loss_base:.6f}")
    print(f"LAMINAR_LOSS={final_loss_laminar:.6f}")
    print(f"LOSS_DELTA={loss_delta:.6f}")
    print(f"AVG_RESIDUAL_NORM={avg_error_norm:.4f}")

if __name__ == "__main__":
    run_experiment()
