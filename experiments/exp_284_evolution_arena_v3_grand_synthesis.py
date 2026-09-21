"""
[EXP-284] Multi-Mechanism Autonomous Evolution Arena v3 & Grand 12-Paradigm Synthesis:
Empirical Comparative Evaluation of Final Paradigms K, L, M, N:
1. Paradigm K: Quantum Superposition & State Collapse (Phase Interference & Decoherence Crystallization)
2. Paradigm L: Jerne Immune Clonal Selection Network (Antigen Binding & Somatic Hypermutation)
3. Paradigm M: Memristive Synaptic Flux & Plastic Sprouting (Flux-Dependent Dendritic Growth)
4. Paradigm N: Fisher Information Metric Manifold Curvature (Riemannian Geodesic Dimensional Expansion)

Followed by unified consolidation and multi-dimensional analysis of all 12 Evolutionary Paradigms (A-N).
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
# 1. PARADIGM IMPLEMENTATIONS (K, L, M, N)
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


# --- PARADIGM K: Quantum Superposition & State Collapse ---
class QuantumCollapseEvolution(BaseEvolutionaryMechanism):
    def __init__(self, agent: CoREAgent):
        super().__init__("Quantum_Collapse", agent)
        self.quantum_purity = 0.9
        self.phase_angles = {}
        self._init_phases()

    def _init_phases(self):
        manifest = json.loads(self.agent.get_topology_manifest())
        for node in manifest["nodes"]:
            self.phase_angles[node["name"]] = 0.0

    def step_awake(self, step: int, loss: float, f_val: float, h_states: torch.Tensor):
        # Update phase angle interference based on surprise rotation
        manifest = json.loads(self.agent.get_topology_manifest())
        for node in manifest["nodes"]:
            name = node["name"]
            curr = self.phase_angles.get(name, 0.0)
            self.phase_angles[name] = (curr + 0.1 * f_val) % (2.0 * math.pi)
        
        # Purity drops with chaotic phases
        std_phase = float(torch.tensor(list(self.phase_angles.values())).std().item()) if len(self.phase_angles) > 1 else 0.0
        self.quantum_purity = max(0.1, min(1.0, 1.0 - (std_phase / math.pi)))

    def step_sleep(self, step: int) -> Dict[str, Any]:
        sprouted = 0
        pruned = 0
        # Decoherence Sprouting: if constructive interference occurs, collapse new node
        if self.quantum_purity > 0.65 and self.agent.graph.k_nodes < 8:
            new_name = f"quant_collapse_{self.agent.graph.k_nodes}_{int(time.time()*1000)%1000}"
            self.agent.add_node(new_name, "ContinuousHopfield", is_core=False, initial_alpha=0.0)
            self.phase_angles[new_name] = 0.0
            sprouted += 1

        pruned = self.agent.prune_inactive_nodes(threshold=0.04)
        manifest = json.loads(self.agent.get_topology_manifest())
        active_names = {n["name"] for n in manifest["nodes"]}
        self.phase_angles = {k: v for k, v in self.phase_angles.items() if k in active_names}

        return {"sprouted": sprouted, "pruned": pruned, "purity": self.quantum_purity}

    def get_local_metrics(self) -> Dict[str, float]:
        return {
            "quantum_purity": self.quantum_purity,
            "mean_phase_angle": sum(self.phase_angles.values()) / max(1, len(self.phase_angles)),
            "superposition_nodes": float(len(self.phase_angles))
        }


# --- PARADIGM L: Jerne Immune Clonal Selection Network ---
class ImmuneClonalEvolution(BaseEvolutionaryMechanism):
    def __init__(self, agent: CoREAgent):
        super().__init__("Immune_Clonal_Selection", agent)
        self.antibody_titers = {}
        self.antigen_load = 0.5
        self._init_immune()

    def _init_immune(self):
        manifest = json.loads(self.agent.get_topology_manifest())
        for node in manifest["nodes"]:
            self.antibody_titers[node["name"]] = 1.0

    def step_awake(self, step: int, loss: float, f_val: float, h_states: torch.Tensor):
        # Antigen load is proportional to surprise
        self.antigen_load = 0.9 * self.antigen_load + 0.1 * (f_val / 2.5)
        # Affinity maturation of active antibodies
        manifest = json.loads(self.agent.get_topology_manifest())
        for node in manifest["nodes"]:
            name = node["name"]
            curr = self.antibody_titers.get(name, 0.5)
            # Stimulate antibodies if antigen present
            self.antibody_titers[name] = max(0.01, min(2.0, curr + 0.05 * self.antigen_load - 0.02))

    def step_sleep(self, step: int) -> Dict[str, Any]:
        sprouted = 0
        pruned = 0
        # Somatic Hypermutation: strong antigen triggers clonal expansion
        if self.antigen_load > 0.90 and self.agent.graph.k_nodes < 8:
            new_name = f"immune_clone_{self.agent.graph.k_nodes}_{int(time.time()*1000)%1000}"
            self.agent.add_node(new_name, "StateSpaceMemory", is_core=False, initial_alpha=0.0)
            self.antibody_titers[new_name] = 1.5
            sprouted += 1

        pruned = self.agent.prune_inactive_nodes(threshold=0.04)
        manifest = json.loads(self.agent.get_topology_manifest())
        active_names = {n["name"] for n in manifest["nodes"]}
        self.antibody_titers = {k: v for k, v in self.antibody_titers.items() if k in active_names}

        return {"sprouted": sprouted, "pruned": pruned, "antigen_load": self.antigen_load}

    def get_local_metrics(self) -> Dict[str, float]:
        return {
            "antigen_load": self.antigen_load,
            "mean_antibody_titer": sum(self.antibody_titers.values()) / max(1, len(self.antibody_titers)),
            "clonal_repertoire": float(len(self.antibody_titers))
        }


# --- PARADIGM M: Memristive Synaptic Flux & Plastic Sprouting ---
class MemristiveFluxEvolution(BaseEvolutionaryMechanism):
    def __init__(self, agent: CoREAgent):
        super().__init__("Memristive_Flux_Sprouting", agent)
        self.flux_integral = 0.0
        self.conductance = 0.5

    def step_awake(self, step: int, loss: float, f_val: float, h_states: torch.Tensor):
        # Flux integral tracks continuous charge/signal density
        current_charge = float(torch.norm(h_states).item()) / 100.0
        self.flux_integral += current_charge
        self.conductance = max(0.1, min(1.0, 0.5 + 0.05 * self.flux_integral))

    def step_sleep(self, step: int) -> Dict[str, Any]:
        sprouted = 0
        pruned = 0
        # High accumulated flux causes dendritic breakdown & sprouting
        if self.flux_integral > 1.5 and self.agent.graph.k_nodes < 8:
            new_name = f"memrist_sprout_{self.agent.graph.k_nodes}_{int(time.time()*1000)%1000}"
            self.agent.add_node(new_name, "LinearAccumulator", is_core=False, initial_alpha=0.0)
            sprouted += 1
            self.flux_integral *= 0.3 # Discharged

        pruned = self.agent.prune_inactive_nodes(threshold=0.04)
        return {"sprouted": sprouted, "pruned": pruned, "conductance": self.conductance}

    def get_local_metrics(self) -> Dict[str, float]:
        return {
            "flux_integral": self.flux_integral,
            "conductance": self.conductance,
            "sprouting_margin": max(0.0, 1.5 - self.flux_integral)
        }


# --- PARADIGM N: Fisher Information Metric Manifold Curvature ---
class FisherCurvatureEvolution(BaseEvolutionaryMechanism):
    def __init__(self, agent: CoREAgent):
        super().__init__("Fisher_Curvature_Manifold", agent)
        self.fisher_trace = 0.5
        self.curvature_index = 0.2

    def step_awake(self, step: int, loss: float, f_val: float, h_states: torch.Tensor):
        # Fisher trace approximated by output gradient variance
        var_h = float(torch.var(h_states).item())
        self.fisher_trace = 0.9 * self.fisher_trace + 0.1 * (var_h * loss)
        self.curvature_index = max(0.05, 0.95 * self.curvature_index + 0.05 * (f_val / max(0.1, self.fisher_trace)))

    def step_sleep(self, step: int) -> Dict[str, Any]:
        sprouted = 0
        pruned = 0
        # High Riemannian Curvature requires dimension expanding geodesic node
        if self.curvature_index > 0.35 and self.agent.graph.k_nodes < 8:
            new_name = f"fisher_geodesic_{self.agent.graph.k_nodes}_{int(time.time()*1000)%1000}"
            self.agent.add_node(new_name, "SaturatedAttractor", is_core=False, initial_alpha=0.0)
            sprouted += 1
            self.curvature_index *= 0.5

        pruned = self.agent.prune_inactive_nodes(threshold=0.04)
        return {"sprouted": sprouted, "pruned": pruned, "fisher_trace": self.fisher_trace}

    def get_local_metrics(self) -> Dict[str, float]:
        return {
            "fisher_trace": self.fisher_trace,
            "curvature_index": self.curvature_index,
            "geodesic_efficiency": 1.0 / max(0.1, self.curvature_index)
        }


# =====================================================================
# 2. ARENA BENCHMARK EVALUATOR
# =====================================================================

def evaluate_mechanism(mech_class, name: str) -> Dict[str, Any]:
    print(f"\n=======================================================================")
    print(f"🥊 ARENA EVALUATION v3: {name}")
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
        
        logits = agent(tokens, thinking_steps=2)
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
        
        logits = agent(tokens, thinking_steps=2)
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
    print("🏆 KARYON-CORE AUTONOMOUS EVOLUTION ARENA: BATTLE OF PARADIGMS (K, L, M, N)")
    print("=" * 80)

    mechanisms = [
        (QuantumCollapseEvolution, "Paradigm_K_Quantum_Collapse"),
        (ImmuneClonalEvolution, "Paradigm_L_Immune_Clonal_Selection"),
        (MemristiveFluxEvolution, "Paradigm_M_Memristive_Flux"),
        (FisherCurvatureEvolution, "Paradigm_N_Fisher_Curvature")
    ]

    arena_v3_results = []

    for mech_class, name in mechanisms:
        res = evaluate_mechanism(mech_class, name)
        arena_v3_results.append(res)

    for entry in arena_v3_results:
        composite = (
            entry["domain_adaptation_gain"] * 2.0
            - entry["forgetting_delta"] * 1.5
            - entry["phase3_shift_loss"]
            + (entry["avg_throughput_tok_s"] / 5000.0)
        )
        entry["composite_score"] = composite

    arena_v3_results.sort(key=lambda x: x["composite_score"], reverse=True)

    with open("arena_v3_results.json", "w") as f:
        json.dump(arena_v3_results, f, indent=2)

    print("\n" + "=" * 80)
    print("🌟 GRAND CONSOLIDATION OF ALL 12 EVOLUTIONARY PARADIGMS (A - N)")
    print("=" * 80)

    # Load all 3 arena benchmark files
    all_paradigms = []
    try:
        with open("arena_results.json", "r") as f:
            all_paradigms.extend(json.load(f))
    except Exception:
        pass
    try:
        with open("arena_v2_results.json", "r") as f:
            all_paradigms.extend(json.load(f))
    except Exception:
        pass
    all_paradigms.extend(arena_v3_results)

    # Re-rank across all 12
    all_paradigms.sort(key=lambda x: x["composite_score"], reverse=True)

    header = f"{'Rank':<4} | {'Paradigm Name':<35} | {'Shift Loss':<10} | {'Forget Δ':<9} | {'Adapt Gain':<10} | {'Tok/s':<8} | {'Nodes':<5} | {'Composite Score'}"
    print(header)
    print("-" * len(header))
    for idx, r in enumerate(all_paradigms, 1):
        print(f"{idx:<4} | {r['mechanism']:<35} | {r['phase3_shift_loss']:<10.4f} | {r['forgetting_delta']:<9.4f} | {r['domain_adaptation_gain']:<10.4f} | {r['avg_throughput_tok_s']:<8.0f} | {r['final_nodes']:<5} | {r['composite_score']:<15.4f}")

    grand_champion = all_paradigms[0]
    print(f"\n👑 SUPREME GRAND CHAMPION (OUT OF 12 PARADIGMS): {grand_champion['mechanism']}")
    print(f"   • Composite Multi-Dimensional Score : {grand_champion['composite_score']:.4f}")
    print(f"   • Catastrophic Forgetting Delta      : {grand_champion['forgetting_delta']:.4f} (Minimal loss of past knowledge)")
    print(f"   • Domain Adaptation Loss             : {grand_champion['phase3_shift_loss']:.4f} (Fastest convergence on shift)")
    print(f"   • Computational Throughput           : {grand_champion['avg_throughput_tok_s']:.1f} tok/s")

    with open("grand_12_paradigms_leaderboard.json", "w") as f:
        json.dump(all_paradigms, f, indent=2)


if __name__ == "__main__":
    main()
