"""
===============================================================================
EXP-232: Comprehensive Tri-Vector System 2 Active Imagination Benchmarks
Grounding: KEP Principle 1 (C++ Parallelism & Acceleration),
           KEP Principle 2 (Living AGI & Biological Realism - NON-NEGOTIABLE),
           KEP Principle 12 (Universal Modality-Agnostic Substrate),
           KEP Principle 17 (Goodhart's Law Immunization & Metric De-Fetishization),
           KEP Rule #1 (Hypothesis, Behavioral Scope & Telemetry First),
           KEP Rule #2 (Contextual Multi-Criteria Decision Engine).
===============================================================================
Hypothesis:
Formulating and evaluating Active Imagination across 3 distinct vectors:
  Vector A (System 2 PAC Generative Sandbox):
    Integrating parallel EFE-guided candidate token evaluation on high-entropy (H > 0.70)
    word boundaries reduces semantic perplexity and eliminates pseudo-morphemic drift.
  Vector B (Spontaneous Latent Daydreaming & Active Synthesis):
    Autonomous idle-time mental rollouts without external sensory input drive internal state
    evolution, lowering cumulative Free Energy (FE) and enhancing curiosity/dopamine balance.
  Vector C (Episodic Counterfactual Simulation in Spatial Hazard World):
    Coupling episodic memory recall with 3-step counterfactual look-ahead steers action selection,
    maximizing survival steps and somatic health in hazardous environments.
  Vector ABC (Unified Active Imagination Engine):
    Combined integration produces synergistic improvements in coherence, survival efficiency,
    and thermodynamic stability compared to reactive System 1 execution.
===============================================================================
"""

import os
import sys
import time
import math
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from karyon_config import CoREConfig
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory
from karyon_agent import CoREAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("exp_232")


class HazardSurvivalWorld:
    """
    2D Cybernetic Spatial Hazard World (CS-World) for Vector C testing.
    Grid size: 10x10.
    - Food (E): increases energy (+0.35)
    - Hazard (H): decreases health (-0.40)
    - Empty space: movement energy cost (-0.05)
    """
    def __init__(self, grid_size=10, num_food=8, num_hazards=12, seed=42):
        self.grid_size = grid_size
        self.num_food = num_food
        self.num_hazards = num_hazards
        self.seed = seed
        self.reset()

    def reset(self):
        random.seed(self.seed)
        self.agent_pos = [self.grid_size // 2, self.grid_size // 2]
        self.food_positions = []
        self.hazard_positions = []

        while len(self.food_positions) < self.num_food:
            pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if pos != self.agent_pos and pos not in self.food_positions:
                self.food_positions.append(pos)

        while len(self.hazard_positions) < self.num_hazards:
            pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if pos != self.agent_pos and pos not in self.food_positions and pos not in self.hazard_positions:
                self.hazard_positions.append(pos)

    def get_sensory_vector(self, device='cpu') -> torch.Tensor:
        obs = []
        ax, ay = self.agent_pos
        obs.extend([ax / float(self.grid_size), ay / float(self.grid_size)])

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

        # Pad to 256 dimensions for unified sensory embedding space
        sensory_tensor = torch.zeros(1, 256, device=device)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        sensory_tensor[0, :obs_tensor.size(0)] = obs_tensor
        return sensory_tensor

    def step(self, action_idx: int):
        # 0: Up, 1: Down, 2: Left, 3: Right
        moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        move = moves[action_idx % 4]
        self.agent_pos[0] = max(0, min(self.grid_size - 1, self.agent_pos[0] + move[0]))
        self.agent_pos[1] = max(0, min(self.grid_size - 1, self.agent_pos[1] + move[1]))

        reward = -0.05 # Movement cost
        health_delta = 0.0
        hit_food = False
        hit_hazard = False

        if self.agent_pos in self.food_positions:
            reward = 0.35
            hit_food = True
            self.food_positions.remove(self.agent_pos)
            # Respawn food
            while True:
                pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
                if pos != self.agent_pos and pos not in self.food_positions and pos not in self.hazard_positions:
                    self.food_positions.append(pos)
                    break

        if self.agent_pos in self.hazard_positions:
            health_delta = -0.40
            hit_hazard = True

        return reward, health_delta, hit_food, hit_hazard


def test_vector_a(agent: CoREAgent, config: CoREConfig, device: torch.device):
    """
    Vector A: System 2 PAC Generative Sandbox on High Entropy Word Boundaries
    """
    logger.info("=== Testing Vector A: System 2 Generative Sandbox ===")
    prompt = "The fundamental law of active inference states that"
    
    hu = HomeostaticUnit(batch_size=1, device=str(device))
    episodic_memory = BatchedEpisodicMemory(batch_size=1, memory_dim=256, max_capacity=500, device=str(device))

    # Run with Sandbox enabled (standard in generate_thought_and_speech)
    gen_tokens_sandbox = []
    t0 = time.perf_counter()
    for item in agent.generate_thought_and_speech(
        prompt, None, None, hu, episodic_memory, config, max_generated_tokens=60, temperature=0.45
    ):
        if item.get("status") == "token" and item.get("token_id") != 257:
            gen_tokens_sandbox.append(item["text"])
    time_sandbox = time.perf_counter() - t0
    text_sandbox = "".join(gen_tokens_sandbox)

    logger.info(f"Vector A Generated Speech: '{text_sandbox.strip()}' (Time: {time_sandbox:.3f}s)")
    return {
        "text_sample": text_sandbox.strip(),
        "time_sec": time_sandbox,
        "token_count": len(gen_tokens_sandbox)
    }


def test_vector_b(agent: CoREAgent, device: torch.device):
    """
    Vector B: Spontaneous Latent Daydreaming & Active Synthesis in Quiet Idle State
    """
    logger.info("=== Testing Vector B: Spontaneous Latent Daydreaming ===")
    hu = HomeostaticUnit(batch_size=1, device=str(device))

    # Initialize a latent state
    h_curr = torch.randn(1, agent.hidden_dim, device=device)

    # Execute 10 spontaneous idle rollouts without external sensory input
    initial_fe = 0.0
    final_fe = 0.0
    fe_history = []

    with torch.no_grad():
        for step in range(10):
            # Generate candidate thought delta via world model
            w_dummy = torch.zeros(1, agent.unified_dim, device=device)
            w_pred, kl_div, fe, _ = agent.world_model(h_curr, h_curr, w_dummy)
            
            fe_val = fe.mean().item()
            fe_history.append(fe_val)
            if step == 0:
                initial_fe = fe_val
            final_fe = fe_val

            # Apply spontaneous curiosity boost
            hu.state[0, 0] = torch.clamp(hu.state[0, 0] + 0.02, 0.0, 1.0) # Curiosity
            hu.state[0, 5] = torch.clamp(hu.state[0, 5] + 0.01, 0.0, 1.0) # Dopamine

            # Shift latent state
            h_curr = h_curr + 0.10 * agent.in_proj(w_pred)

    fe_delta = final_fe - initial_fe
    logger.info(f"Vector B Spontaneous Daydreaming: FE Start={initial_fe:.4f}, FE End={final_fe:.4f}, Delta={fe_delta:.4f}")
    return {
        "fe_start": initial_fe,
        "fe_end": final_fe,
        "fe_delta": fe_delta,
        "fe_history": fe_history
    }


def run_survival_simulation(agent: CoREAgent, mode: str, max_steps: int = 40, seed: int = 42, device: str = 'cpu'):
    """
    Simulates survival in HazardSurvivalWorld under 'system1' (reactive) or 'system2' (counterfactual imagination).
    """
    world = HazardSurvivalWorld(grid_size=10, num_food=8, num_hazards=12, seed=seed)
    hu = HomeostaticUnit(batch_size=1, device=device)
    episodic_memory = BatchedEpisodicMemory(batch_size=1, memory_dim=256, max_capacity=500, device=device)

    m_s1 = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=agent.device)
    m_s2 = torch.zeros(1, agent.num_heads, agent.head_k, agent.head_v, device=agent.device)

    steps_survived = 0
    food_collected = 0
    hazard_hits = 0
    total_fe = 0.0

    for step in range(max_steps):
        s_vec = world.get_sensory_vector(device=agent.device)
        u_t = hu.state

        # Check if agent is dead
        if hu.state[0, 3].item() <= 0.05 or hu.state[0, 1].item() <= 0.05:
            break

        steps_survived += 1

        with torch.no_grad():
            sensor_dict = {'spatial_grid': s_vec}
            outs, fe, commit_loss, _, _, m_s1_next, m_s2_next, z_t = agent.forward_multimodal_step(sensor_dict, m_s1, m_s2, u_t)
            m_s1, m_s2 = m_s1_next, m_s2_next
            total_fe += fe.mean().item()

            h_s2 = m_s2.view(1, -1)[:, :agent.hidden_dim]

            # Motor decision
            if mode == 'system1':
                # System 1: Direct reactive mapping from h_s2 to motor logits
                motor_logits = outs.get('spatial_grid', torch.randn(1, 256, device=agent.device))[:, :4]
                action_idx = int(torch.argmax(motor_logits, dim=-1).item())
            elif mode in ['system2', 'unified']:
                # System 2: Active Imagination Counterfactual Simulation
                # Evaluate 4 candidate direction vectors using world_model
                candidate_actions = [
                    torch.tensor([[0.2, 0.0, 0.0, 0.0]], device=agent.device), # Up
                    torch.tensor([[-0.2, 0.0, 0.0, 0.0]], device=agent.device),# Down
                    torch.tensor([[0.0, -0.2, 0.0, 0.0]], device=agent.device),# Left
                    torch.tensor([[0.0, 0.2, 0.0, 0.0]], device=agent.device)  # Right
                ]
                
                best_action = 0
                lowest_efe = 1e9

                for cand_idx, cand_act in enumerate(candidate_actions):
                    # Base sensory projection candidate
                    w_cand = s_vec.clone()
                    w_cand[:, :4] = cand_act

                    # 3-step counterfactual rollout
                    _, efe = agent.evaluate_mental_sandbox(h_s2, w_cand, num_steps=3)

                    # In unified mode, consult episodic memory for prior hazard associations
                    if mode == 'unified':
                        q_k = agent.episodic_sensory_proj(w_cand)
                        ret_mem, max_sim = episodic_memory.read(q_k, temperature=0.05, threshold=0.10)
                        # If past memory indicates hazard / high surprise, penalize candidate EFE
                        if max_sim > 0.30:
                            efe += 0.50 * max_sim

                    if efe < lowest_efe:
                        lowest_efe = efe
                        best_action = cand_idx

                action_idx = best_action

        # Step world
        reward, health_delta, hit_food, hit_hazard = world.step(action_idx)

        if hit_food:
            food_collected += 1
            # Replenish energy (+0.35)
            hu.state[0, 1] = torch.clamp(hu.state[0, 1] + 0.35, 0.0, 1.0)
            hu.state[0, 5] = torch.clamp(hu.state[0, 5] + 0.20, 0.0, 1.0) # Dopamine
        else:
            # Energy decay
            hu.state[0, 1] = torch.clamp(hu.state[0, 1] - 0.04, 0.0, 1.0)

        if hit_hazard:
            hazard_hits += 1
            # Health damage (-0.40)
            hu.state[0, 3] = torch.clamp(hu.state[0, 3] - 0.40, 0.0, 1.0)
            hu.state[0, 4] = torch.clamp(hu.state[0, 4] + 0.30, 0.0, 1.0) # Noradrenaline

        # Record high-surprise events into episodic memory
        if (hit_food or hit_hazard or fe.mean().item() > 0.15) and episodic_memory is not None:
            q_val = agent.episodic_sensory_proj(s_vec)
            episodic_memory.write(q_val, q_val)

    avg_fe = total_fe / max(1, steps_survived)
    final_health = hu.state[0, 3].item()
    final_energy = hu.state[0, 1].item()

    return {
        "mode": mode,
        "steps_survived": steps_survived,
        "food_collected": food_collected,
        "hazard_hits": hazard_hits,
        "avg_fe": avg_fe,
        "final_health": final_health,
        "final_energy": final_energy
    }


def main():
    logger.info("Starting EXP-232: Comprehensive Active Imagination Benchmarks")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Target Compute Device: {device}")

    # Initialize CoREAgent
    config = CoREConfig()
    agent = CoREAgent(config=config, device=device_str)
    agent.to(device)
    agent.eval()

    # 1. Run Vector A
    res_a = test_vector_a(agent, config, device)

    # 2. Run Vector B
    res_b = test_vector_b(agent, device)

    # 3. Run Vector C (System 1 vs System 2 Counterfactual Simulation)
    logger.info("=== Testing Vector C: Episodic Counterfactual Simulation ===")
    res_c_sys1 = run_survival_simulation(agent, mode='system1', max_steps=40, seed=101, device=device_str)
    res_c_sys2 = run_survival_simulation(agent, mode='system2', max_steps=40, seed=101, device=device_str)
    logger.info(f"Vector C (System 1): Steps={res_c_sys1['steps_survived']}, Food={res_c_sys1['food_collected']}, Hazards={res_c_sys1['hazard_hits']}, AvgFE={res_c_sys1['avg_fe']:.4f}")
    logger.info(f"Vector C (System 2): Steps={res_c_sys2['steps_survived']}, Food={res_c_sys2['food_collected']}, Hazards={res_c_sys2['hazard_hits']}, AvgFE={res_c_sys2['avg_fe']:.4f}")

    # 4. Run Vector ABC (Unified Active Imagination Engine)
    logger.info("=== Testing Vector ABC: Unified Active Imagination Engine ===")
    res_abc = run_survival_simulation(agent, mode='unified', max_steps=40, seed=101, device=device_str)
    logger.info(f"Vector ABC (Unified): Steps={res_abc['steps_survived']}, Food={res_abc['food_collected']}, Hazards={res_abc['hazard_hits']}, AvgFE={res_abc['avg_fe']:.4f}")

    # Telemetry Summary
    logger.info("=========================================================")
    logger.info("=== EXP-232 FINAL EMPIRICAL TELEMETRY REPORT ===")
    logger.info("=========================================================")
    logger.info(f"Vector A (Speech Gen): Tokens={res_a['token_count']}, Latency={res_a['time_sec']:.3f}s")
    logger.info(f"Vector B (Daydreaming): FE Delta={res_b['fe_delta']:.4f} (Start: {res_b['fe_start']:.4f} -> End: {res_b['fe_end']:.4f})")
    logger.info(f"Vector C (System 1 Reactive): Steps={res_c_sys1['steps_survived']}/40, Food={res_c_sys1['food_collected']}, Hazards={res_c_sys1['hazard_hits']}, Health={res_c_sys1['final_health']:.2f}")
    logger.info(f"Vector C (System 2 Sandbox): Steps={res_c_sys2['steps_survived']}/40, Food={res_c_sys2['food_collected']}, Hazards={res_c_sys2['hazard_hits']}, Health={res_c_sys2['final_health']:.2f}")
    logger.info(f"Vector ABC (Unified Engine): Steps={res_abc['steps_survived']}/40, Food={res_abc['food_collected']}, Hazards={res_abc['hazard_hits']}, Health={res_abc['final_health']:.2f}")
    logger.info("=========================================================")

    # Verification Assertions for POSITIVE Verdict
    assert res_c_sys2['hazard_hits'] <= res_c_sys1['hazard_hits'], "System 2 imagination failed to reduce hazard hits!"
    assert res_abc['steps_survived'] >= res_c_sys1['steps_survived'], "Unified imagination failed to maintain or improve survival steps!"
    
    print("\n--- EXP-232 VERIFIED: ALL TELEMETRY ASSERTIONS PASSED SUCESSFULLY ---\n")


if __name__ == "__main__":
    main()
