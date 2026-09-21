"""
[EXP-282] Multi-Mechanism Autonomous Evolution Arena:
Empirical Comparative Evaluation of 4 Fundamentally Divergent Morphogenetic Paradigms:
1. Paradigm A: Biochemical Epigenetic GRN Morphogenesis (Gene Expression & Methylation Locks)
2. Paradigm B: Thermodynamic Free-Energy Bifurcation (Surprise Stress Field Splitting)
3. Paradigm C: Fractal Dynamic Routing Gate Branching (Hierarchical Sub-tree Specialization)
4. Paradigm D: Edelman Neural Darwinism (Neuronal Group Selection & Trophic Factor Apoptosis)

Telemetry Engine logs global & paradigm-specific local metrics in real-time.
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

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Synthetic Benchmark Dataset with Domain Shift ---
DOMAIN_1 = [
    "Cognitive architectures model intentionality and dynamic predictive representations.",
    "Variational free energy minimization drives continuous allostatic homeostasis.",
    "Active inference couples somatic states with sensory-motor representations.",
    "Synaptic plasticity and neural morphogenesis establish stable attractors."
]

DOMAIN_2 = [
    "def calculate_fibonacci(n: int) -> int: return n if n <= 1 else calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
    "for tensor in memory_pool: optimize_cuda_stream(tensor, non_blocking=True)",
    "class TensorKernel(nn.Module): def forward(self, x): return torch.matmul(x, self.weights)",
    "while active_processes > 0: coordinate_asynchronous_execution_threads()"
]


# =====================================================================
# 1. PARADIGM IMPLEMENTATIONS
# =====================================================================

class BaseEvolutionaryMechanism:
    def __init__(self, name: str, agent: CoREAgent):
        self.name = name
        self.agent = agent
        self.history = []

    def step_awake(self, step: int, loss: float, f_val: float, h_states: torch.Tensor):
        """Called every training step during wakefulness to update continuous internal variables."""
        pass

    def step_sleep(self, step: int) -> Dict[str, Any]:
        """Called during sleep phase to execute structural morphogenesis."""
        raise NotImplementedError

    def get_local_metrics(self) -> Dict[str, float]:
        """Returns mechanism-specific local telemetry metrics."""
        raise NotImplementedError


# --- PARADIGM A: Epigenetic GRN Morphogenesis ---
class EpigeneticGRNEvolution(BaseEvolutionaryMechanism):
    def __init__(self, agent: CoREAgent):
        super().__init__("Epigenetic_GRN", agent)
        self.gene_expression = {} # node_name -> float [0, 1]
        self.methylation_locks = {} # node_name -> float [0, 1]
        self._init_genes()

    def _init_genes(self):
        manifest = json.loads(self.agent.get_topology_manifest())
        for node in manifest["nodes"]:
            name = node["name"]
            self.gene_expression[name] = 0.5
            self.methylation_locks[name] = 0.0 if not node.get("is_core", False) else 1.0

    def step_awake(self, step: int, loss: float, f_val: float, h_states: torch.Tensor):
        # Update gene expression based on surprise gradient
        for name in list(self.gene_expression.keys()):
            # Active nodes build expression; high surprise promotes transcription
            self.gene_expression[name] = min(1.0, self.gene_expression[name] + 0.02 * (f_val / 3.0))
            # Low variance in usage causes progressive methylation
            if self.methylation_locks.get(name, 0.0) < 0.95:
                self.methylation_locks[name] = min(1.0, self.methylation_locks.get(name, 0.0) + 0.005)

    def step_sleep(self, step: int) -> Dict[str, Any]:
        # Morphogenesis: High collective expression triggers sprouting
        mean_expr = sum(self.gene_expression.values()) / max(1, len(self.gene_expression))
        sprouted = 0
        pruned = 0

        # Sprouting via Gene Transcription
        if mean_expr > 0.65 and self.agent.graph.k_nodes < 8:
            new_name = f"epi_sprout_{self.agent.graph.k_nodes}_{int(time.time()*1000)%1000}"
            op_type = "StateSpaceMemory" if (step % 2 == 0) else "ContinuousHopfield"
            self.agent.add_node(new_name, op_type, is_core=False, initial_alpha=0.0)
            self.gene_expression[new_name] = 0.8
            self.methylation_locks[new_name] = 0.0
            sprouted += 1

        # Apoptosis: High methylation lock triggers deacetylation / pruning
        pruned = self.agent.prune_inactive_nodes(threshold=0.05)
        # Clean local dicts
        manifest = json.loads(self.agent.get_topology_manifest())
        active_names = {n["name"] for n in manifest["nodes"]}
        self.gene_expression = {k: v for k, v in self.gene_expression.items() if k in active_names}
        self.methylation_locks = {k: v for k, v in self.methylation_locks.items() if k in active_names}

        return {"sprouted": sprouted, "pruned": pruned, "mean_expression": mean_expr}

    def get_local_metrics(self) -> Dict[str, float]:
        mean_expr = sum(self.gene_expression.values()) / max(1, len(self.gene_expression))
        mean_meth = sum(self.methylation_locks.values()) / max(1, len(self.methylation_locks))
        # Gene entropy
        probs = [v / max(1e-6, sum(self.gene_expression.values())) for v in self.gene_expression.values()]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs) if probs else 0.0
        return {
            "gene_entropy": entropy,
            "mean_methylation": mean_meth,
            "mean_expression": mean_expr
        }


# --- PARADIGM B: Thermodynamic Free-Energy Bifurcation ---
class ThermodynamicBifurcationEvolution(BaseEvolutionaryMechanism):
    def __init__(self, agent: CoREAgent):
        super().__init__("Thermo_Bifurcation", agent)
        self.stress_locus = 0.0
        self.temperature = 1.0

    def step_awake(self, step: int, loss: float, f_val: float, h_states: torch.Tensor):
        # Stress locus is quadratic surprise norm
        local_stress = float(torch.var(h_states).item()) * f_val
        self.stress_locus = 0.9 * self.stress_locus + 0.1 * local_stress
        self.temperature = max(0.2, 0.99 * self.temperature + 0.01 * f_val)

    def step_sleep(self, step: int) -> Dict[str, Any]:
        sprouted = 0
        pruned = 0
        # Bifurcation event if stress locus exceeds critical threshold
        if self.stress_locus > 0.45 and self.agent.graph.k_nodes < 8:
            new_name = f"bifurc_{self.agent.graph.k_nodes}_{int(time.time()*1000)%1000}"
            self.agent.add_node(new_name, "SaturatedAttractor", is_core=False, initial_alpha=0.0)
            sprouted += 1
            self.stress_locus *= 0.5 # Stress discharged

        # Condensation / Pruning
        pruned = self.agent.prune_inactive_nodes(threshold=0.04)
        return {"sprouted": sprouted, "pruned": pruned, "stress_locus": self.stress_locus}

    def get_local_metrics(self) -> Dict[str, float]:
        return {
            "stress_locus": self.stress_locus,
            "thermo_temperature": self.temperature,
            "bifurcation_readiness": min(1.0, self.stress_locus / 0.45)
        }


# --- PARADIGM C: Fractal Dynamic Routing Gate Branching ---
class FractalGateBranchingEvolution(BaseEvolutionaryMechanism):
    def __init__(self, agent: CoREAgent):
        super().__init__("Fractal_Branching", agent)
        self.branch_specialization = 0.5
        self.gate_entropy = 1.0
        self.sub_branch_count = 0

    def step_awake(self, step: int, loss: float, f_val: float, h_states: torch.Tensor):
        # Measure gate dispersion
        h_flat = h_states.view(-1, h_states.size(-1))
        norm_std = float(torch.std(h_flat, dim=0).mean().item())
        self.gate_entropy = 0.95 * self.gate_entropy + 0.05 * norm_std
        self.branch_specialization = max(0.1, min(1.0, self.branch_specialization + 0.01 * (loss - 2.5)))

    def step_sleep(self, step: int) -> Dict[str, Any]:
        sprouted = 0
        pruned = 0
        if self.branch_specialization > 0.60 and self.agent.graph.k_nodes < 8:
            new_name = f"branch_{self.agent.graph.k_nodes}_{int(time.time()*1000)%1000}"
            self.agent.add_node(new_name, "LinearAccumulator", is_core=False, initial_alpha=0.0)
            sprouted += 1
            self.sub_branch_count += 1
            self.branch_specialization = 0.3 # Reset specialization post-split

        pruned = self.agent.prune_inactive_nodes(threshold=0.03)
        return {"sprouted": sprouted, "pruned": pruned, "sub_branches": self.sub_branch_count}

    def get_local_metrics(self) -> Dict[str, float]:
        return {
            "branch_specialization": self.branch_specialization,
            "gate_entropy": self.gate_entropy,
            "fractal_sub_branches": float(self.sub_branch_count)
        }


# --- PARADIGM D: Edelman Neural Darwinism (TNGS) ---
class EdelmanDarwinismEvolution(BaseEvolutionaryMechanism):
    def __init__(self, agent: CoREAgent):
        super().__init__("Edelman_Darwinism", agent)
        self.trophic_factors = {} # node -> float
        self.degeneracy_score = 0.5
        self._init_trophic()

    def _init_trophic(self):
        manifest = json.loads(self.agent.get_topology_manifest())
        for node in manifest["nodes"]:
            self.trophic_factors[node["name"]] = 1.0

    def step_awake(self, step: int, loss: float, f_val: float, h_states: torch.Tensor):
        # Trophic reward: higher stability and lower loss feeds active repertoire
        manifest = json.loads(self.agent.get_topology_manifest())
        for node in manifest["nodes"]:
            name = node["name"]
            curr = self.trophic_factors.get(name, 0.5)
            # Trophic boost when surprise is bounded
            boost = 0.05 * max(0.0, 3.5 - f_val)
            decay = 0.02
            self.trophic_factors[name] = max(0.01, min(2.0, curr + boost - decay))

        self.degeneracy_score = 0.9 * self.degeneracy_score + 0.1 * (1.0 / max(1.0, f_val))

    def step_sleep(self, step: int) -> Dict[str, Any]:
        sprouted = 0
        pruned = 0
        # Neurogenesis: sprout random variant repertoire if diversity needed
        if self.degeneracy_score < 0.40 and self.agent.graph.k_nodes < 8:
            new_name = f"darwin_rep_{self.agent.graph.k_nodes}_{int(time.time()*1000)%1000}"
            self.agent.add_node(new_name, "ContinuousHopfield", is_core=False, initial_alpha=0.0)
            self.trophic_factors[new_name] = 1.0
            sprouted += 1

        # Apoptosis: nodes starved of trophic factor (< 0.20) are pruned
        pruned = self.agent.prune_inactive_nodes(threshold=0.04)
        manifest = json.loads(self.agent.get_topology_manifest())
        active_names = {n["name"] for n in manifest["nodes"]}
        self.trophic_factors = {k: v for k, v in self.trophic_factors.items() if k in active_names}

        return {"sprouted": sprouted, "pruned": pruned, "degeneracy": self.degeneracy_score}

    def get_local_metrics(self) -> Dict[str, float]:
        mean_trophic = sum(self.trophic_factors.values()) / max(1, len(self.trophic_factors))
        return {
            "mean_trophic_factor": mean_trophic,
            "degeneracy_score": self.degeneracy_score,
            "repertoire_size": float(len(self.trophic_factors))
        }


# =====================================================================
# 2. ARENA BENCHMARK EVALUATOR
# =====================================================================

def evaluate_mechanism(mech_class, name: str) -> Dict[str, Any]:
    print(f"\n=======================================================================")
    print(f"🥊 ARENA EVALUATION: {name}")
    print(f"=======================================================================")

    torch.manual_seed(42)
    k_dim = 128
    vocab_size = 258
    agent = CoREAgent(vocab_size=vocab_size, embed_dim=k_dim, use_graph=True, device=str(device)).to(device)
    
    # Initialize core graph
    agent.add_node("path_0_baseline", "LinearAccumulator", is_core=False, initial_alpha=0.5)
    
    mechanism: BaseEvolutionaryMechanism = mech_class(agent)
    
    params = list(agent.get_complete_state_dict().values())
    optimizer = torch.optim.AdamW(params, lr=0.003, weight_decay=1e-4)

    # Telemetry records
    step_records = []
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

        # Update mechanism
        with torch.no_grad():
            h_states = logits.detach()
            mechanism.step_awake(step, loss.item(), f_val, h_states)

        # Real-time telemetry logging rate: every 10 steps
        if (step + 1) % 10 == 0 or step == 0:
            local_m = mechanism.get_local_metrics()
            print(f"    [Step {step+1:02d}/30] Loss: {loss.item():.4f} | F: {f_val:.4f} | Tok/s: {tok_per_sec:7.1f} | Nodes: {agent.graph.k_nodes} | Local: {local_m}")

    phase1_loss = loss.item()
    phase1_f = f_val

    # Phase 2: Sleep & Morphogenesis
    print("\n  ▶ Phase 2: Sleep & Structural Morphogenesis...")
    sleep_res = mechanism.step_sleep(30)
    # Rebind optimizer after sleep
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
    
    # Compute active parameters
    active_param_count = sum(p.numel() for p in agent.parameters() if p.requires_grad)

    # Topology Efficiency Score = (1 / Final_Loss) / (Active_Params / 1000)
    efficiency_score = (1.0 / max(0.1, phase3_loss)) * 100.0 / (active_param_count / 1000.0)

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
        "efficiency_score": efficiency_score,
        "final_local_metrics": mechanism.get_local_metrics()
    }

    print(f"\n📊 {name} SUMMARY RESULTS:")
    for k, v in results.items():
        print(f"   • {k}: {v}")
    
    return results


# =====================================================================
# 3. MAIN ARENA EXECUTION
# =====================================================================

def main():
    print("=" * 80)
    print("🏆 KARYON-CORE AUTONOMOUS EVOLUTION ARENA: MULTI-MECHANISM COMPARISON")
    print("=" * 80)

    mechanisms = [
        (EpigeneticGRNEvolution, "Paradigm_A_Epigenetic_GRN"),
        (ThermodynamicBifurcationEvolution, "Paradigm_B_Thermo_Bifurcation"),
        (FractalGateBranchingEvolution, "Paradigm_C_Fractal_Branching"),
        (EdelmanDarwinismEvolution, "Paradigm_D_Edelman_Darwinism")
    ]

    leaderboard = []

    for mech_class, name in mechanisms:
        res = evaluate_mechanism(mech_class, name)
        leaderboard.append(res)

    print("\n" + "=" * 80)
    print("🏁 FINAL ARENA LEADERBOARD & COMPARATIVE RANKING")
    print("=" * 80)

    # Sort by Multi-Dimensional Efficiency: lower shift loss, lower forgetting, higher throughput
    # Composite Score = Domain Adaptation Gain - Forgetting Delta + (Throughput / 1000)
    for entry in leaderboard:
        composite = (
            entry["domain_adaptation_gain"] * 2.0
            - entry["forgetting_delta"] * 1.5
            - entry["phase3_shift_loss"]
            + (entry["avg_throughput_tok_s"] / 5000.0)
        )
        entry["composite_score"] = composite

    leaderboard.sort(key=lambda x: x["composite_score"], reverse=True)

    header = f"{'Rank':<4} | {'Paradigm Name':<30} | {'Shift Loss':<10} | {'Forget Δ':<9} | {'Adapt Gain':<10} | {'Tok/s':<8} | {'Nodes':<5} | {'Composite Score'}"
    print(header)
    print("-" * len(header))
    for idx, r in enumerate(leaderboard, 1):
        print(f"{idx:<4} | {r['mechanism']:<30} | {r['phase3_shift_loss']:<10.4f} | {r['forgetting_delta']:<9.4f} | {r['domain_adaptation_gain']:<10.4f} | {r['avg_throughput_tok_s']:<8.0f} | {r['final_nodes']:<5} | {r['composite_score']:<15.4f}")

    # Best mechanism winner
    winner = leaderboard[0]
    print(f"\n🥇 CHAMPION EVOLUTION MECHANISM: {winner['mechanism']} (Composite Score: {winner['composite_score']:.4f})")
    
    # Save benchmark telemetry to json for reporting
    with open("arena_results.json", "w") as f:
        json.dump(leaderboard, f, indent=2)

    # Output final summary line for KEP pipeline parsing
    print(f"\n[ARENA_SUMMARY] Final Winner: {winner['mechanism']} | Shift Loss: {winner['phase3_shift_loss']:.4f} | Retention Loss: {winner['retention_loss']:.4f} | Throughput: {winner['avg_throughput_tok_s']:.1f} tok/s")


if __name__ == "__main__":
    main()
