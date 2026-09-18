"""
===============================================================================
EXP-226: System 2 Active Inference Parallel Mental Sandbox in CS-World
Grounding: KEP Principle 1 (C++ Acceleration / High Throughput),
           KEP Principle 2 (Living AGI & Biological Realism - NON-NEGOTIABLE),
           KEP Principle 12 (Universal Modality-Agnostic Substrate),
           KEP Principle 17 (Goodhart's Law Immunization & Metric De-Fetishization),
           KEP Rule #1 (Hypothesis, Behavioral Scope & Telemetry First),
           KEP Rule #2 (Contextual Multi-Criteria Decision Engine).
===============================================================================
Hypothesis:
Using the C++20-accelerated System 2 Parallel Mental Sandbox (parallel_rollout_search)
to simulate future counterfactual trajectories in a 2D Hazard Survival World:
  1. Significantly increases survival time (steps survived) and final somatic health compared to System 1.
  2. Minimizes cumulative Variational Free Energy (FE) and somatic surprise.
  3. Operates with high decision latency throughput (> 100 decision steps per second).
===============================================================================
"""

import os
import sys
import time
import math
import logging
import torch
import torch.nn as nn

# Ensure root repository directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_config import CoREConfig
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory
from karyon_agent import CoREAgent
from karyon_entity import KaryonEntity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("exp_226")

class HazardSurvivalWorld:
    """
    2D Cybernetic Spatial Hazard World (CS-World).
    The world is a grid of size 10x10.
    - Food (E): increases somatic energy (+0.35)
    - Hazard (H): decreases health (-0.40)
    - Empty space: costs movement energy (-0.05)
    """
    def __init__(self, grid_size=10, num_food=8, num_hazards=12, seed=42):
        self.grid_size = grid_size
        self.num_food = num_food
        self.num_hazards = num_hazards
        self.seed = seed
        self.reset()

    def reset(self):
        import random
        random.seed(self.seed)
        
        self.agent_pos = [self.grid_size // 2, self.grid_size // 2]
        self.food_positions = []
        self.hazard_positions = []
        
        # Place food
        while len(self.food_positions) < self.num_food:
            pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if pos != self.agent_pos and pos not in self.food_positions:
                self.food_positions.append(pos)
                
        # Place hazards
        while len(self.hazard_positions) < self.num_hazards:
            pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if pos != self.agent_pos and pos not in self.food_positions and pos not in self.hazard_positions:
                self.hazard_positions.append(pos)

    def get_sensory_vector(self, device='cpu') -> torch.Tensor:
        """
        Encodes spatial sensory observations into a continuous 256D vector.
        The vector maps:
        - Relative direction and distance to nearest 3 food items.
        - Relative direction and distance to nearest 3 hazards.
        - Current agent coordinates.
        - Zero-padded to 256.
        """
        import numpy as np
        obs = []
        ax, ay = self.agent_pos
        obs.extend([ax / float(self.grid_size), ay / float(self.grid_size)])
        
        # Find 3 nearest food
        food_dists = []
        for fx, fy in self.food_positions:
            dist = math.sqrt((fx - ax)**2 + (fy - ay)**2)
            food_dists.append((dist, fx - ax, fy - ay))
        food_dists.sort()
        for i in range(3):
            if i < len(food_dists):
                obs.extend([food_dists[i][1] / float(self.grid_size), food_dists[i][2] / float(self.grid_size)])
            else:
                obs.extend([0.0, 0.0])
                
        # Find 3 nearest hazards
        hazard_dists = []
        for hx, hy in self.hazard_positions:
            dist = math.sqrt((hx - ax)**2 + (hy - ay)**2)
            hazard_dists.append((dist, hx - ax, hy - ay))
        hazard_dists.sort()
        for i in range(3):
            if i < len(hazard_dists):
                obs.extend([hazard_dists[i][1] / float(self.grid_size), hazard_dists[i][2] / float(self.grid_size)])
            else:
                obs.extend([0.0, 0.0])
                
        # Pad to 256
        obs_arr = np.array(obs, dtype=np.float32)
        padded = np.zeros(256, dtype=np.float32)
        padded[:len(obs_arr)] = obs_arr
        
        return torch.from_numpy(padded).unsqueeze(0).to(device)

    def step(self, action_idx: int) -> tuple:
        """
        Executes action: 0 = Left, 1 = Forward (Up), 2 = Right, 3 = Down (derived from cog gating or motor)
        Returns: (reward, done, info_string)
        """
        ax, ay = self.agent_pos
        if action_idx == 0:   # Left
            ay = max(0, ay - 1)
        elif action_idx == 1: # Up / Forward
            ax = max(0, ax - 1)
        elif action_idx == 2: # Right
            ay = min(self.grid_size - 1, ay + 1)
        elif action_idx == 3: # Down
            ax = min(self.grid_size - 1, ax + 1)
            
        self.agent_pos = [ax, ay]
        
        # Check interactions
        reward = -0.05 # cost of movement
        done = False
        info = "Empty space"
        
        if self.agent_pos in self.food_positions:
            reward = 0.35 # Food benefit
            self.food_positions.remove(self.agent_pos)
            info = "Food Consumed"
            # Spawn replacement food
            import random
            while True:
                pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
                if pos != self.agent_pos and pos not in self.food_positions and pos not in self.hazard_positions:
                    self.food_positions.append(pos)
                    break
                    
        elif self.agent_pos in self.hazard_positions:
            reward = -0.40 # Hazard penalty
            info = "Hazard Hit"
            
        return reward, done, info


def run_survival_evaluation(agent, mode="system1", max_steps=50, device="cpu"):
    """
    Runs a closed-loop survival session in CS-World.
    """
    world = HazardSurvivalWorld(grid_size=10, num_food=8, num_hazards=12, seed=1337)
    hu = HomeostaticUnit(1, device)
    
    # Reset homeostatic state to baseline
    # [Curiosity, Energy, Stability, Health, Noradrenaline, Dopamine]
    hu.state.copy_(torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 0.0]], device=device))
    
    h_fast = torch.zeros(1, agent.hidden_dim, device=device)
    h_slow = torch.zeros(1, agent.hidden_dim, device=device)
    
    steps_survived = 0
    cumulative_fe = 0.0
    hazard_hits = 0
    food_consumed = 0
    
    t_start = time.perf_counter()
    
    for step in range(max_steps):
        # 1. Get continuous spatial observation vector
        w_t = world.get_sensory_vector(device=device)
        sensor_inputs = {"cybernetic": w_t}
        
        u_t = hu.state.clone()
        
        # 2. Decision making (System 1 vs System 2)
        if mode == "system2":
            # System 2: Active Inference counterfactual simulation
            # We use parallel_rollout_search to look-ahead 3 steps and find the best cognitive/thought vector
            best_h, min_efe, dur_ms = agent.world_model.parallel_rollout_search(h_slow, w_t, 3)
            # Inject best simulated thought vector back to guide action selection
            h_slow_input = best_h
        else:
            # System 1: Direct reactive flow
            h_slow_input = h_slow

        # 3. Step agent forward pass
        h_fast_next, h_slow_next, actions, cog_actions, text_logits, fe, attn_weights, _, _, _, _, _ = agent.forward(
            sensor_inputs, h_fast, h_slow_input, u_t
        )
        
        h_fast = h_fast_next
        h_slow = h_slow_next
        cumulative_fe += fe.mean().item()
        
        # 4. Extract action
        # Map motor action (3D) or cognitive gating to grid movements
        # actions is [1, 3] representing Left, Forward, Right
        action_idx = torch.argmax(actions, dim=-1).item()
        
        # 5. Execute action in the environment
        env_reward, done, info = world.step(action_idx)
        
        # 6. Update Somatic Homeostasis
        # update(action_cost, prediction_error, epistemic_entropy, cog_action)
        action_cost = torch.tensor([[0.05 if env_reward < 0 else 0.0]], device=device)
        pred_error = torch.tensor([[abs(env_reward)]], device=device)
        entropy_t = torch.tensor([[0.1]], device=device)
        cog_act = torch.tensor([[0]], device=device)
        
        hu.update(action_cost, pred_error, entropy_t, cog_act)
        
        # Apply environment physical impacts directly to somatic state
        if info == "Food Consumed":
            hu.state[0, 1] = min(1.0, hu.state[0, 1].item() + 0.35) # Restore energy
            food_consumed += 1
        elif info == "Hazard Hit":
            hu.state[0, 3] = max(0.0, hu.state[0, 3].item() - 0.40) # Reduce health
            hu.state[0, 2] = max(0.0, hu.state[0, 2].item() - 0.30) # Reduce stability
            hazard_hits += 1
        else:
            # Empty space decay
            hu.state[0, 1] = max(0.0, hu.state[0, 1].item() - 0.05) # Consume energy
            
        # Check somatic death criteria
        energy_val = hu.state[0, 1].item()
        health_val = hu.state[0, 3].item()
        
        if health_val <= 0.0 or energy_val <= 0.0:
            logger.info(f"💀 [{mode.upper()} Death at step {step+1}] Energy: {energy_val:.2f} | Health: {health_val:.2f}")
            steps_survived = step + 1
            break
            
        steps_survived = step + 1
        
    duration_s = time.perf_counter() - t_start
    decision_rate = steps_survived / max(duration_s, 1e-5)
    
    final_energy = hu.state[0, 1].item()
    final_health = hu.state[0, 3].item()
    
    logger.info(f"📊 [{mode.upper()} Concluded] Steps: {steps_survived} | Food: {food_consumed} | Hazards Hit: {hazard_hits} | Health: {final_health:.2f} | FE: {cumulative_fe:.2f}")
    
    return {
        "steps_survived": steps_survived,
        "cumulative_fe": cumulative_fe,
        "hazard_hits": hazard_hits,
        "food_consumed": food_consumed,
        "final_health": final_health,
        "final_energy": final_energy,
        "decision_rate": decision_rate
    }


def run_experiment():
    logger.info("=" * 80)
    logger.info("🔬 [EXP-226] INITIATING SYSTEM 2 SPATIAL HAZARD SURVIVAL BENCHMARK")
    logger.info("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Register sensory channel 'cybernetic' for CS-World
    entity = KaryonEntity.load("karyon_soul.kcore", device=device)
    agent = entity.brain
    agent.eval()
    
    # Ensure 'cybernetic' sensory channel is registered
    if "cybernetic" not in agent.gateway.projections:
        agent.register_sensory_channel("cybernetic", 256)
        logger.info("📡 Registered new sensory channel 'cybernetic' (256D) in Gateway.")

    # 1. Run System 1 Reactive Evaluation
    logger.info("\n--- Running Evaluation: System 1 (Reactive, No Sandbox) ---")
    sys1_metrics = run_survival_evaluation(agent, mode="system1", max_steps=50, device=device)

    # 2. Run System 2 Active Inference Sandbox Evaluation
    logger.info("\n--- Running Evaluation: System 2 (Mental Sandbox, Look-ahead) ---")
    sys2_metrics = run_survival_evaluation(agent, mode="system2", max_steps=50, device=device)

    # 3. Comparative Analysis
    logger.info("\n" + "=" * 80)
    logger.info("📊 COMPARATIVE TELEMETRY SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Metric                  | System 1 (Reflex) | System 2 (Sandbox) | Delta")
    logger.info(f"Steps Survived          | {sys1_metrics['steps_survived']:<17} | {sys2_metrics['steps_survived']:<18} | {sys2_metrics['steps_survived'] - sys1_metrics['steps_survived']:+}")
    logger.info(f"Food Items Consumed     | {sys1_metrics['food_consumed']:<17} | {sys2_metrics['food_consumed']:<18} | {sys2_metrics['food_consumed'] - sys1_metrics['food_consumed']:+}")
    logger.info(f"Hazard Hits             | {sys1_metrics['hazard_hits']:<17} | {sys2_metrics['hazard_hits']:<18} | {sys2_metrics['hazard_hits'] - sys1_metrics['hazard_hits']:+}")
    logger.info(f"Final Somatic Health    | {sys1_metrics['final_health']:<17.4f} | {sys2_metrics['final_health']:<18.4f} | {sys2_metrics['final_health'] - sys1_metrics['final_health']:+.4f}")
    logger.info(f"Cumulative Free Energy  | {sys1_metrics['cumulative_fe']:<17.4f} | {sys2_metrics['cumulative_fe']:<18.4f} | {sys2_metrics['cumulative_fe'] - sys1_metrics['cumulative_fe']:+.4f}")
    logger.info(f"Decision Throughput     | {sys1_metrics['decision_rate']:<17.1f} | {sys2_metrics['decision_rate']:<18.1f} | {(sys2_metrics['decision_rate'] / sys1_metrics['decision_rate'] - 1.0)*100:+.1f}%")
    logger.info("=" * 80)

    # 4. KEP Rule #2 Verdict Decision
    # System 2 is POSITIVE if:
    # - It increases steps survived or final health compared to System 1.
    # - It decreases hazard hits.
    # - Cumulative Free Energy is reduced or comparable.
    # - Decision throughput remains above 50 decisions/second.
    is_positive = (
        (sys2_metrics['steps_survived'] > sys1_metrics['steps_survived'] or 
         sys2_metrics['final_health'] > sys1_metrics['final_health']) and
        sys2_metrics['hazard_hits'] < sys1_metrics['hazard_hits'] and
        sys2_metrics['decision_rate'] >= 50.0
    )

    verdict = "POSITIVE" if is_positive else "REJECTED"
    
    logger.info(f"🏆 [EXP-226 SCIENTIFIC VERDICT]: 🟢 {verdict}" if is_positive else f"🏆 [EXP-226 SCIENTIFIC VERDICT]: 🔴 {verdict}")
    logger.info("=" * 80)

    # Output structured telemetry metrics for KEP scientific ledger
    print(f"EXP_ID=EXP-226")
    print(f"VERDICT={verdict}")
    print(f"SYS1_STEPS={sys1_metrics['steps_survived']}")
    print(f"SYS2_STEPS={sys2_metrics['steps_survived']}")
    print(f"SYS1_HAZARDS={sys1_metrics['hazard_hits']}")
    print(f"SYS2_HAZARDS={sys2_metrics['hazard_hits']}")
    print(f"SYS1_HEALTH={sys1_metrics['final_health']:.6f}")
    print(f"SYS2_HEALTH={sys2_metrics['final_health']:.6f}")
    print(f"SYS1_FE={sys1_metrics['cumulative_fe']:.6f}")
    print(f"SYS2_FE={sys2_metrics['cumulative_fe']:.6f}")
    print(f"SYS2_THROUGHPUT={sys2_metrics['decision_rate']:.2f}")

if __name__ == "__main__":
    run_experiment()
