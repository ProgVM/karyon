"""
===============================================================================
EXP-227: Dynamic GRN Morphogens & SOS-Reaction under High-VFE Shock
Grounding: KEP Principle 2 (Living AGI & Biological Realism - NON-NEGOTIABLE),
           KEP Principle 14 (Axiom of Allostatic Dynamic Forces),
           KEP Principle 15 (Epigenetic Morphogenesis & Net2Net Smooth Grafting),
           KEP Principle 16 (Dynamic Neural Graph Assembly),
           KEP Rule #1 (Hypothesis, Behavioral Scope & Telemetry First),
           KEP Rule #2 (Contextual Multi-Criteria Decision Engine).
===============================================================================
Hypothesis:
Under high-shock conditions (sudden spikes in Variational Free Energy, VFE,
representing unexpected hazards or prediction errors), biological organisms trigger
epigenetic Gene Regulatory Network (GRN) hypermutation and sprouting (SOS-reaction).
Implementing an automated shock-responsive morphogenetic trigger inside Karyon's
ContinuousDynamicNeuralGraph that sprouts a new mathematical operator brick
(e.g., DelayOp, GateOp, or NonLinearOp) when VFE exceeds a dynamic homeostatic
threshold will:
  1. Demonstrate autonomous structural adaptation to high-shock environments.
  2. Maintain strict zero-shock function identity at birth using zero-weight
     epigenetic grafting (alpha_epi = 0.0).
  3. Validate AST, linter, and unit compliance of dynamic self-neurogenesis.
===============================================================================
"""

import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure root repository directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_config import CoREConfig
from karyon_core import HomeostaticUnit
from karyon_agent import CoREAgent, ContinuousDynamicNeuralGraph
from karyon_entity import KaryonEntity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("exp_227")

class ShockSurvivalEnvironment:
    """
    Environmental stream characterized by sudden high-VFE shock events.
    Simulates sensory stream transitions where some steps introduce severe
    prediction errors (unexpected high-dimensional transitions).
    """
    def __init__(self, seed=42):
        import random
        random.seed(seed)
        self.step_idx = 0
        
    def get_sensory_input(self, device='cpu') -> tuple:
        # Standard input has periodic patterns, shock events introduce pure noise
        is_shock = (self.step_idx > 0 and self.step_idx % 8 == 0)
        if is_shock:
            # High-VFE shock event
            w_t = torch.randn(1, 256, device=device) * 15.0
        else:
            # Regular predictable patterns
            t = float(self.step_idx)
            w_t = torch.sin(torch.arange(256, device=device).float() * 0.05 + t).unsqueeze(0)
            
        self.step_idx += 1
        return w_t, is_shock


def run_experiment():
    logger.info("=" * 80)
    logger.info("🔬 [EXP-227] INITIATING DYNAMIC GRN MORPHOGENS & SOS-REACTION BENCHMARK")
    logger.info("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    entity = KaryonEntity.load("karyon_soul.kcore", device=device)
    agent = entity.brain
    
    if "cybernetic" not in agent.gateway.projections:
        agent.register_sensory_channel("cybernetic", 256)
        
    env = ShockSurvivalEnvironment(seed=227)
    hu = HomeostaticUnit(1, device)
    hu.state.copy_(torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], device=device))
    
    h_fast = torch.zeros(1, agent.hidden_dim, device=device)
    h_slow = torch.zeros(1, agent.hidden_dim, device=device)
    
    initial_brick_count = len(agent.dynamic_graph.bricks)
    logger.info(f"🧱 Initial dynamic graph brick count: {initial_brick_count}")
    
    # Track metrics
    shocks_encountered = 0
    sprouts_triggered = 0
    vfe_history = []
    
    # Dynamic VFE shock threshold based on moving average of past non-shock steps
    vfe_threshold = 1.5
    
    for step in range(20):
        w_t, is_shock = env.get_sensory_input(device=device)
        sensor_inputs = {"cybernetic": w_t}
        u_t = hu.state.clone()
        
        # Forward pass through the active architecture
        h_fast_next, h_slow_next, actions, cog_actions, text_logits, fe, attn_weights, _, _, _, _, _ = agent.forward(
            sensor_inputs, h_fast, h_slow, u_t
        )
        
        h_fast = h_fast_next
        h_slow = h_slow_next
        vfe_val = fe.mean().item()
        vfe_history.append(vfe_val)
        
        if is_shock:
            shocks_encountered += 1
            logger.info(f"💥 [Step {step}] Shock Event! Measured VFE: {vfe_val:.4f} (Threshold: {vfe_threshold:.2f})")
            
            # KEP Principle 15 & 16: Shock-responsive epigenesis (SOS Sprouting)
            if vfe_val > vfe_threshold:
                # Alternate brick sprouting types to diversify graph topology
                brick_type = "DelayOp" if sprouts_triggered % 2 == 0 else "GateOp"
                success = agent.dynamic_graph.sprout_brick(brick_type=brick_type)
                if success:
                    sprouts_triggered += 1
                    # Verify zero-shock identity: newly sprouted brick has alpha_epi = 0.0
                    alpha_epi_new = agent.dynamic_graph.alpha_epi[-1].item()
                    assert alpha_epi_new == 0.0, f"Error: Sprouted brick must have alpha_epi = 0.0, got {alpha_epi_new}"
                    logger.info(f"🧬 Verified zero-shock Net2Net identity at birth for sprouted brick #{len(agent.dynamic_graph.bricks)-1}.")
        else:
            # Update running threshold with baseline VFE
            vfe_threshold = 0.8 * vfe_threshold + 0.2 * (vfe_val * 1.5)
            
        # Update somatic homeostasis
        action_cost = torch.tensor([[0.01]], device=device)
        pred_error = torch.tensor([[vfe_val / 10.0]], device=device)
        entropy_t = torch.tensor([[0.1]], device=device)
        cog_act = torch.tensor([[0]], device=device)
        hu.update(action_cost, pred_error, entropy_t, cog_act)
        
    final_brick_count = len(agent.dynamic_graph.bricks)
    logger.info(f"🧱 Final dynamic graph brick count: {final_brick_count}")
    
    # Save mutated soul back to the container
    entity.save("karyon_soul.kcore")
    logger.info("💾 Successfully saved updated KaryonEntity with mutated dynamic graph to 'karyon_soul.kcore'")
    
    # Evaluate success
    is_positive = (
        shocks_encountered > 0 and
        sprouts_triggered > 0 and
        final_brick_count == initial_brick_count + sprouts_triggered
    )
    
    verdict = "POSITIVE" if is_positive else "REJECTED"
    logger.info(f"🏆 [EXP-227 SCIENTIFIC VERDICT]: 🟢 {verdict}" if is_positive else f"🏆 [EXP-227 SCIENTIFIC VERDICT]: 🔴 {verdict}")
    logger.info("=" * 80)
    
    print(f"EXP_ID=EXP-227")
    print(f"VERDICT={verdict}")
    print(f"INITIAL_BRICKS={initial_brick_count}")
    print(f"FINAL_BRICKS={final_brick_count}")
    print(f"SPROUTS_TRIGGERED={sprouts_triggered}")
    print(f"SHOCKS_ENCOUNTERED={shocks_encountered}")

if __name__ == "__main__":
    run_experiment()
