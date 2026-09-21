"""
[EXP-283] Multi-Mechanism Autonomous Evolution Arena v2:
Empirical Comparative Evaluation of Paradigms E, F, G, H:
1. Paradigm E: Allostatic GRN & Ashby Neurotransmitter Coupling (Arousal Plasticity & Dopaminergic Locks)
2. Paradigm F: Continuous Hopfield Attractor Basin Sprouting (Energy Landscape Volumetric Growth)
3. Paradigm G: Theta-Gamma PAC Phase-Synchronized Morphogenesis (Cross-Frequency Coherence)
4. Paradigm H: Recurrent Latent Thinking Depth (System 2 Recirculation Loops)

Real-time telemetry tracking: Loss, Free Energy, Tok/s, Catastrophic Forgetting Delta, Domain Shift Recovery, and Paradigm-Specific Metrics.
"""

import sys
import os
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath('.'))

import karyon_core as kcore
from karyon_agent import CoREAgent
from kcore_evolution import rebind_optimizer_moments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Datasets with Rich Morpho-Semantic & Algorithmic Structures ---
DOMAIN_1 = [
    "Cognitive architectures model intentionality and dynamic predictive representations.",
    "Variational free energy minimization drives continuous allostatic homeostasis.",
    "Active inference couples somatic states with sensory-motor representations.",
    "Synaptic plasticity and neural morphogenesis establish stable attractors."
]

DOMAIN_2 = [
    "def solve_bellman_optimality(states, actions, transitions, gamma=0.99): return value_iteration(states)",
    "class EpigeneticOperator(nn.Module): def forward(self, x): return torch.tanh(self.alpha) * self.op(x)",
    "async def coordinate_parallel_tensor_dispatch(streams): return await asyncio.gather(*streams)",
    "for epoch in range(iterations): optimizer.zero_grad(); loss.backward(); optimizer.step()"
]


# =====================================================================
# 1. PARADIGM IMPLEMENTATIONS (E, F, G, H)
# =====================================================================

class BaseEvolutionaryMechanism:
    def __init__(self, name: str, agent: CoREAgent):
        self.name = name
        self.agent = agent

    def step_awake(self, step: int, loss: float, f_val: float, h_states: torch.Tensor):
        pass

    def step_sleep(self, step: int) -> Dict[str, Any]:
        raise NotImplementedError

    def get_local_metrics(self) -> Dict[str, float]:
        raise NotImplementedError


# --- PARADIGM E: Allostatic GRN & Ashby Neurotransmitter Coupling ---
class AllostaticGRNEvolution(BaseEvolutionaryMechanism):
    def __init__(self, agent: CoREAgent):
        super().__init__("Allostatic_GRN", agent)
        self.methylation_locks = {}
        self.noradrenaline = 0.5 # Arousal / Stress
        self.dopamine = 0.5       # Reward / Predictive success
        self.manifest_synced = False
        self._sync_locks()

    def _sync_locks(self):
        manifest = json.loads(self.agent.get_topology_manifest())
        for node in manifest["nodes"]:
            name = node["name"]
            if name not in self.methylation_locks:
                self.methylation_locks[name] = 0.8 if node.get("is_core", False) else 0.2

    def step_awake(self, step: int, loss: float, f_val: float, h_states: torch.Tensor):
        self._sync_locks()
        # Ashby Neurotransmitter update
        target_na = min(1.0, f_val / 3.0)
        self.noradrenaline = 0.9 * self.noradrenaline + 0.1 * target_na
        target_da = max(0.0, 1.0 - (loss / 4.0))
        self.dopamine = 0.9 * self.dopamine + 0.1 * target_da

        # NA induces demethylation (hyperplasticity); DA induces crystallization (locks)
        for name in list(self.methylation_locks.keys()):
            delta_lock = 0.03 * self.dopamine - 0.02 * self.noradrenaline
            self.methylation_locks[name] = max(0.05, min(0.99, self.methylation_locks[name] + delta_lock))

    def step_sleep(self, step: int) -> Dict[str, Any]:
        sprouted = 0
        pruned = 0
        # High arousal (NA) triggers neurogenesis
        if self.noradrenaline > 0.45 and self.agent.graph.k_nodes < 8:
            new_name = f"allo_sprout_{self.agent.graph.k_nodes}_{int(time.time()*1000)%1000}"
            self.agent.add_node(new_name, "StateSpaceMemory", is_core=False, initial_alpha=0.0)
            self.methylation_locks[new_name] = 0.1
            sprouted += 1

        # Unlocked & inactive nodes get pruned
        pruned = self.agent.prune_inactive_nodes(threshold=0.04)
        manifest = json.loads(self.agent.get_topology_manifest())
        active_names = {n["name"] for n in manifest["nodes"]}
        self.methylation_locks = {k: v for k, v in self.methylation_locks.items() if k in active_names}

        return {"sprouted": sprouted, "pruned": pruned, "noradrenaline": self.noradrenaline, "dopamine": self.dopamine}

    def get_local_metrics(self) -> Dict[str, float]:
        mean_lock = sum(self.methylation_locks.values()) / max(1, len(self.methylation_locks))
        plasticity_idx = self.noradrenaline / (mean_lock + 1e-6)
        return {
            "mean_lock": mean_lock,
            "noradrenaline": self.noradrenaline,
            "dopamine": self.dopamine,
            "plasticity_index": plasticity_idx
        }


# --- PARADIGM F: Continuous Hopfield Attractor Basin Sprouting ---
class HopfieldBasinSproutingEvolution(BaseEvolutionaryMechanism):
    def __init__(self, agent: CoREAgent):
        super().__init__("Hopfield_Basin_Sprouting", agent)
        self.energy_depth = 0.5
        self.unfamiliar_triggers = 0
        self.basin_orthogonality = 0.8

    def step_awake(self, step: int, loss: float, f_val: float, h_states: torch.Tensor):
        # Measure attractor energy
        h_norm = F.normalize(h_states.view(-1, h_states.size(-1)), p=2, dim=-1)
        sim_matrix = torch.matmul(h_norm, h_norm.t())
        off_diag = sim_matrix - torch.eye(sim_matrix.size(0), device=sim_matrix.device)
        self.basin_orthogonality = float(1.0 - torch.clamp(off_diag.abs().mean(), 0.0, 1.0).item())
        
        # Surprise above threshold indicates unfamiliar attractor basin requirement
        if f_val > 3.2:
            self.unfamiliar_triggers += 1
        self.energy_depth = max(0.1, 0.95 * self.energy_depth + 0.05 * (1.0 / max(0.5, f_val)))

    def step_sleep(self, step: int) -> Dict[str, Any]:
        sprouted = 0
        pruned = 0
        if self.unfamiliar_triggers >= 3 and self.agent.graph.k_nodes < 8:
            new_name = f"hopfield_basin_{self.agent.graph.k_nodes}_{int(time.time()*1000)%1000}"
            self.agent.add_node(new_name, "ContinuousHopfield", is_core=False, initial_alpha=0.0)
            sprouted += 1
            self.unfamiliar_triggers = 0

        pruned = self.agent.prune_inactive_nodes(threshold=0.04)
        return {"sprouted": sprouted, "pruned": pruned, "orthogonality": self.basin_orthogonality}

    def get_local_metrics(self) -> Dict[str, float]:
        return {
            "basin_orthogonality": self.basin_orthogonality,
            "energy_depth": self.energy_depth,
            "unfamiliar_triggers": float(self.unfamiliar_triggers)
        }


# --- PARADIGM G: Theta-Gamma PAC Phase-Synchronized Morphogenesis ---
class ThetaGammaPACEvolution(BaseEvolutionaryMechanism):
    def __init__(self, agent: CoREAgent):
        super().__init__("Theta_Gamma_PAC", agent)
        self.phase = 0.0
        self.plv = 0.7 # Phase locking value
        self.gamma_bursts = 0.0

    def step_awake(self, step: int, loss: float, f_val: float, h_states: torch.Tensor):
        # Progress theta cycle (period ~ 6 steps)
        self.phase = (self.phase + (2.0 * math.pi / 6.0)) % (2.0 * math.pi)
        # Gamma burst activity correlates with high surprise
        self.gamma_bursts = float(torch.clamp(torch.tensor(f_val / 2.0), 0.0, 5.0).item())
        # Phase locking coherence
        self.plv = max(0.1, min(0.99, 0.95 * self.plv + 0.05 * math.cos(self.phase)))

    def step_sleep(self, step: int) -> Dict[str, Any]:
        sprouted = 0
        pruned = 0
        # Sprout at high gamma burst during receptive phase window
        if self.gamma_bursts > 1.2 and self.agent.graph.k_nodes < 8:
            new_name = f"pac_gamma_{self.agent.graph.k_nodes}_{int(time.time()*1000)%1000}"
            self.agent.add_node(new_name, "SaturatedAttractor", is_core=False, initial_alpha=0.0)
            sprouted += 1

        pruned = self.agent.prune_inactive_nodes(threshold=0.04)
        return {"sprouted": sprouted, "pruned": pruned, "plv": self.plv}

    def get_local_metrics(self) -> Dict[str, float]:
        return {
            "theta_phase": self.phase,
            "plv_coherence": self.plv,
            "gamma_burst_density": self.gamma_bursts
        }


# --- PARADIGM H: Recurrent Latent Thinking Depth (System 2 Recirculation) ---
class RecurrentLatentDepthEvolution(BaseEvolutionaryMechanism):
    def __init__(self, agent: CoREAgent):
        super().__init__("System2_Recurrent_Depth", agent)
        self.thinking_steps = 2
        self.equilibrium_delta = 0.5
        self.recirculation_efficiency = 1.0

    def step_awake(self, step: int, loss: float, f_val: float, h_states: torch.Tensor):
        # Check if internal surprise demands additional recirculation steps
        if f_val > 3.5 and self.thinking_steps < 4:
            self.thinking_steps += 1
        elif f_val < 2.2 and self.thinking_steps > 1:
            self.thinking_steps -= 1

        self.equilibrium_delta = 0.9 * self.equilibrium_delta + 0.1 * (loss / max(1, self.thinking_steps))
        self.recirculation_efficiency = 1.0 / max(0.1, self.equilibrium_delta)

    def step_sleep(self, step: int) -> Dict[str, Any]:
        sprouted = 0
        pruned = 0
        # High recirculation demand builds an explicit state-space loop operator
        if self.thinking_steps >= 3 and self.agent.graph.k_nodes < 8:
            new_name = f"loop_op_{self.agent.graph.k_nodes}_{int(time.time()*1000)%1000}"
            self.agent.add_node(new_name, "LinearAccumulator", is_core=False, initial_alpha=0.0)
            sprouted += 1

        pruned = self.agent.prune_inactive_nodes(threshold=0.04)
        return {"sprouted": sprouted, "pruned": pruned, "thinking_steps": self.thinking_steps}

    def get_local_metrics(self) -> Dict[str, float]:
        return {
            "current_thinking_steps": float(self.thinking_steps),
            "equilibrium_delta": self.equilibrium_delta,
            "recirculation_efficiency": self.recirculation_efficiency
        }


# =====================================================================
# 2. ARENA BENCHMARK EVALUATOR
# =====================================================================

def evaluate_mechanism(mech_class, name: str) -> Dict[str, Any]:
    print(f"\n=======================================================================")
    print(f"🥊 ARENA EVALUATION v2: {name}")
    print(f"=======================================================================")

    torch.manual_seed(42)
    k_dim = 128
    vocab_size = 258
    agent = CoREAgent(vocab_size=vocab_size, embed_dim=k_dim, use_graph=True, device=str(device)).to(device)
    
    agent.add_node("path_0_baseline", "LinearAccumulator", is_core=False, initial_alpha=0.5)
    
    mechanism: BaseEvolutionaryMechanism = mech_class(agent)
    
    params = list(agent.get_complete_state_dict().values())
    optimizer = torch.optim.AdamW(params, lr=0.003, weight_decay=1e-4)

    start_time = time.time()
    total_tokens_processed = 0

    # Phase 1: Ingestion on Domain 1 (30 steps)
    print("  ▶ Phase 1: Domain 1 Stream Ingestion & Wakefulness...")
    for step in range(30):
        t_start = time.perf_counter()
        optimizer.zero_grad()
        text = DOMAIN_1[step % len(DOMAIN_1)]
        tokens = torch.tensor(list(text.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
        
        # Adaptive thinking steps for System 2 mechanism if applicable
        t_steps = getattr(mechanism, "thinking_steps", 2)
        logits = agent(tokens, thinking_steps=t_steps)
        shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
        shift_labels = tokens[:, 1:].reshape(-1)
        
        loss = F.cross_entropy(shift_logits, shift_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        t_elapsed = time.perf_counter() - t_start
        tokens_count = tokens.size(1)
        total_tokens_processed += tokens_count
        tok_per_sec = tokens_count / max(1e-6, t_elapsed)
        f_val = loss.item() + 0.1 * float(torch.var(logits).item())

        with torch.no_grad():
            h_states = logits.detach()
            mechanism.step_awake(step, loss.item(), f_val, h_states)

        if (step + 1) % 10 == 0 or step == 0:
            local_m = mechanism.get_local_metrics()
            print(f"    [Step {step+1:02d}/30] Loss: {loss.item():.4f} | F: {f_val:.4f} | Tok/s: {tok_per_sec:7.1f} | Nodes: {agent.graph.k_nodes} | Local: {local_m}")

    phase1_loss = loss.item()

    # Phase 2: Sleep & Morphogenesis
    print("\n  ▶ Phase 2: Sleep & Structural Morphogenesis...")
    sleep_res = mechanism.step_sleep(30)
    new_params = list(agent.get_complete_state_dict().values())
    optimizer = rebind_optimizer_moments(optimizer, new_params, lr=0.003, weight_decay=1e-4)
    print(f"    • Sleep Result: {sleep_res} | Active Nodes: {agent.graph.k_nodes}")

    # Phase 3: Domain Shift (Domain 2 - Code/Assembly) (30 steps)
    print("\n  ▶ Phase 3: Sudden Domain Shift (Natural Language -> Code Stream)...")
    domain_shift_losses = []
    for step in range(30):
        t_start = time.perf_counter()
        optimizer.zero_grad()
        text = DOMAIN_2[step % len(DOMAIN_2)]
        tokens = torch.tensor(list(text.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
        
        t_steps = getattr(mechanism, "thinking_steps", 2)
        logits = agent(tokens, thinking_steps=t_steps)
        shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
        shift_labels = tokens[:, 1:].reshape(-1)
        
        loss = F.cross_entropy(shift_logits, shift_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        t_elapsed = time.perf_counter() - t_start
        tokens_count = tokens.size(1)
        total_tokens_processed += tokens_count
        tok_per_sec = tokens_count / max(1e-6, t_elapsed)
        f_val = loss.item() + 0.1 * float(torch.var(logits).item())
        domain_shift_losses.append(loss.item())

        with torch.no_grad():
            h_states = logits.detach()
            mechanism.step_awake(30 + step, loss.item(), f_val, h_states)

        if (step + 1) % 10 == 0:
            local_m = mechanism.get_local_metrics()
            print(f"    [Shift Step {step+1:02d}/30] Loss: {loss.item():.4f} | F: {f_val:.4f} | Tok/s: {tok_per_sec:7.1f} | Nodes: {agent.graph.k_nodes} | Local: {local_m}")

    phase3_loss = domain_shift_losses[-1]

    # Phase 4: Retention & Catastrophic Forgetting Audit (Evaluate Domain 1 without updates)
    print("\n  ▶ Phase 4: Retention Audit on Domain 1 (Zero-Shot Replay)...")
    agent.eval()
    retention_losses = []
    with torch.no_grad():
        for text in DOMAIN_1:
            tokens = torch.tensor(list(text.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
            logits = agent(tokens, thinking_steps=2)
            shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
            shift_labels = tokens[:, 1:].reshape(-1)
            l = F.cross_entropy(shift_logits, shift_labels)
            retention_losses.append(l.item())
    
    retention_loss = sum(retention_losses) / len(retention_losses)
    forgetting_delta = retention_loss - phase1_loss
    total_elapsed = time.time() - start_time
    avg_tok_per_sec = total_tokens_processed / max(1e-6, total_elapsed)
    active_param_count = sum(p.numel() for p in agent.parameters() if p.requires_grad)

    results = {
        "mechanism": name,
        "phase1_loss": phase1_loss,
        "phase3_shift_loss": phase3_loss,
        "domain_adaptation_gain": domain_shift_losses[0] - phase3_loss,
        "retention_loss": retention_loss,
        "forgetting_delta": forgetting_delta,
        "avg_throughput_tok_s": avg_tok_per_sec,
        "active_param_count": active_param_count,
        "final_nodes": agent.graph.k_nodes,
        "final_local_metrics": mechanism.get_local_metrics()
    }

    print(f"\n📊 {name} SUMMARY RESULTS:")
    for k, v in results.items():
        print(f"   • {k}: {v}")
    
    return results


def main():
    print("=" * 80)
    print("🏆 KARYON-CORE AUTONOMOUS EVOLUTION ARENA: BATTLE OF PARADIGMS (E, F, G, H)")
    print("=" * 80)

    mechanisms = [
        (AllostaticGRNEvolution, "Paradigm_E_Allostatic_GRN"),
        (HopfieldBasinSproutingEvolution, "Paradigm_F_Hopfield_Basin_Sprouting"),
        (ThetaGammaPACEvolution, "Paradigm_G_Theta_Gamma_PAC"),
        (RecurrentLatentDepthEvolution, "Paradigm_H_System2_Recurrent_Depth")
    ]

    leaderboard = []

    for mech_class, name in mechanisms:
        res = evaluate_mechanism(mech_class, name)
        leaderboard.append(res)

    print("\n" + "=" * 80)
    print("🏁 FINAL ARENA v2 LEADERBOARD & COMPARATIVE RANKING")
    print("=" * 80)

    # Multi-dimensional Composite Score:
    # 2.0 * Adapt_Gain - 1.5 * Forget_Delta - Shift_Loss + Throughput_Boost
    for entry in leaderboard:
        composite = (
            entry["domain_adaptation_gain"] * 2.0
            - entry["forgetting_delta"] * 1.5
            - entry["phase3_shift_loss"]
            + (entry["avg_throughput_tok_s"] / 5000.0)
        )
        entry["composite_score"] = composite

    leaderboard.sort(key=lambda x: x["composite_score"], reverse=True)

    header = f"{'Rank':<4} | {'Paradigm Name':<35} | {'Shift Loss':<10} | {'Forget Δ':<9} | {'Adapt Gain':<10} | {'Tok/s':<8} | {'Nodes':<5} | {'Composite Score'}"
    print(header)
    print("-" * len(header))
    for idx, r in enumerate(leaderboard, 1):
        print(f"{idx:<4} | {r['mechanism']:<35} | {r['phase3_shift_loss']:<10.4f} | {r['forgetting_delta']:<9.4f} | {r['domain_adaptation_gain']:<10.4f} | {r['avg_throughput_tok_s']:<8.0f} | {r['final_nodes']:<5} | {r['composite_score']:<15.4f}")

    winner = leaderboard[0]
    print(f"\n🥇 CHAMPION EVOLUTION MECHANISM: {winner['mechanism']} (Composite Score: {winner['composite_score']:.4f})")
    
    with open("arena_v2_results.json", "w") as f:
        json.dump(leaderboard, f, indent=2)

    print(f"\n[ARENA_SUMMARY] Final Winner: {winner['mechanism']} | Shift Loss: {winner['phase3_shift_loss']:.4f} | Retention Loss: {winner['retention_loss']:.4f} | Throughput: {winner['avg_throughput_tok_s']:.1f} tok/s")


if __name__ == "__main__":
    main()
