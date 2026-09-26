#!/usr/bin/env python3
"""
EXP-299: SOVEREIGN PARALLEL MORPHIC CRUCIBLE BENCHMARK
================================================================================
A paradigm shift: Zero Epochs, Zero Hand-Crafted Frankensteins, Zero Static AdamW.
Implements the 4 Fundamental Pillars:
  1. Continuity (Sequential Single-Pass Streaming Time: t -> t+1)
  2. Discreteness (Topological Graphs composed of Discrete Primitive Operators)
  3. Spatial Parallelism (Populations of P=64 candidate morphic graphs evaluated concurrently on GPU)
  4. Natural Selection & Edelman Apoptosis (Thermodynamic Free Energy Minimization F_t + Sprouting/Pruning)
================================================================================
"""

import sys, os
sys.path.insert(0, '.')
import time, math, random, json
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import karyon_core
from multi_domain_benchmark import generate_multi_domain_suite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 1. PARALLEL MORPHIC POPULATION CRUCIBLE (GPU-VECTORIZED)
# =====================================================================

class ParallelMorphicCrucible:
    """
    Simulates a population of P sovereign morphic organisms simultaneously on Tensor Cores.
    Each organism possesses:
      - Discrete Graph Topology (Adjacency Matrix W_route [K x K])
      - Epigenetic Methylation Vector alpha_epi [K]
      - State-Space & Hopfield Attractor Memory banks
      - Free Energy / Metabolic Energy Ledger
    """
    def __init__(
        self,
        pop_size: int = 64,
        dim: int = 64,
        max_nodes: int = 8,
        vocab_size: int = 258,
        device: torch.device = device
    ):
        self.P = pop_size
        self.D = dim
        self.K = max_nodes
        self.V = vocab_size
        self.device = device

        # 1. Population DNA Genome
        # Embeddings & Readout shared or individual
        self.w_emb = nn.Parameter(torch.randn(vocab_size, dim, device=device) * 0.1)
        
        # Routing matrices for each organism in population: [P, K, K]
        self.w_route = nn.Parameter(torch.randn(pop_size, max_nodes, max_nodes, device=device) * 0.1)
        
        # Epigenetic Expression & Methylation: [P, K] (tanh(alpha) determines node activity)
        self.alpha_epi = nn.Parameter(torch.zeros(pop_size, max_nodes, device=device))
        # Initial core nodes active: node 0 (Accumulator), node 1 (Multiplicative), node 2 (Attractor)
        self.alpha_epi.data[:, 0] = 3.0 # Core Linear Accumulator
        self.alpha_epi.data[:, 1] = 3.0 # Core Multiplicative
        self.alpha_epi.data[:, 2] = 3.0 # Core Continuous Attractor

        # Node internal parameters across population [P, K, D, D]
        self.w_node_in = nn.Parameter(torch.randn(pop_size, max_nodes, dim, dim, device=device) * 0.1)
        self.w_node_out = nn.Parameter(torch.randn(pop_size, max_nodes, dim, dim, device=device) * 0.1)
        
        # Sensory in & Motor out projections: [P, dim, dim]
        self.w_sensory = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        self.w_motor = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        
        # Focus Query & Readout per organism
        self.w_focus_q = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        self.w_content_k = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        self.w_salience = nn.Parameter(torch.randn(pop_size, dim, 1, device=device) * 0.1)
        self.w_copy_gate = nn.Parameter(torch.randn(pop_size, dim, 1, device=device) * 0.1)

        # 2. Thermodynamic Vitality / Free Energy Ledgers
        self.vitality = torch.ones(pop_size, device=device) * 1.0 # Metabolic health
        self.free_energy_history = torch.zeros(pop_size, device=device)

    def forward_stream_step(
        self,
        x_tokens: torch.Tensor,       # [P, B]
        node_states: torch.Tensor,    # [P, K, B, D]
        thinking_cycles: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes one continuous causal step across all P organisms in parallel.
        Returns:
          - logits: [P, B, V]
          - next_node_states: [P, K, B, D]
        """
        P, B = x_tokens.shape
        D = self.D
        K = self.K

        # 1. Sensory Embedding: [P, B, D]
        # x_tokens: [P, B] -> index into w_emb [V, D]
        x_emb = F.embedding(x_tokens, self.w_emb) # [P, B, D]
        
        # Sensory input per organism: [P, B, D]
        x_sens = torch.bmm(x_emb, self.w_sensory) # [P, B, D]
        
        # Feed sensory into root node 0
        current_states = node_states.clone()
        current_states[:, 0] = current_states[:, 0] + x_sens

        # 2. Parallel Recurrent Deliberation (Thinking Cycles across Graph Topology)
        gated_route = torch.tanh(self.w_route) # [P, K, K]
        # Epigenetic gating: [P, K, 1, 1]
        graft_gate = torch.tanh(self.alpha_epi).unsqueeze(-1).unsqueeze(-1)

        for _ in range(thinking_cycles):
            # Message Passing: einsum 'pij,pibd->pjbd'
            agg_inputs = torch.einsum('pij,pibd->pjbd', gated_route, current_states)
            agg_inputs[:, 0] = agg_inputs[:, 0] + x_sens
            
            # Node Nonlinear Transforms: einsum 'pkbd,pkde->pkbe'
            # Node In projection
            projected_in = torch.einsum('pkbd,pkde->pkbe', agg_inputs, self.w_node_in)
            
            # Heterogeneous Primitive Activation:
            # Node 0 & 3: Linear Leaky Accumulator (tanh)
            # Node 1 & 4: Multiplicative / Bilinear Gating (SiLU)
            # Node 2 & 5: Attractor / Decision Snapping (Hopfield/Sign-like steep sigmoid)
            act = torch.empty_like(projected_in)
            act[:, 0::3] = torch.tanh(projected_in[:, 0::3])
            act[:, 1::3] = F.silu(projected_in[:, 1::3])
            act[:, 2::3] = torch.tanh(projected_in[:, 2::3] * 3.0)
            
            # Node Out projection
            projected_out = torch.einsum('pkbe,pked->pkbd', act, self.w_node_out)
            
            # Apply Epigenetic Grafting:
            current_states = graft_gate * projected_out + (1.0 - graft_gate) * current_states

        # 3. Motor Readout: Sum active nodes weighted by alpha_epi
        effective_alpha = torch.softmax(self.alpha_epi, dim=-1).unsqueeze(-1).unsqueeze(-1) # [P, K, 1, 1]
        integrated_field = (current_states * effective_alpha).sum(dim=1) # [P, B, D]
        
        h_motor = torch.bmm(integrated_field, self.w_motor) # [P, B, D]
        
        # Project to vocabulary logits: [P, B, V]
        # w_emb: [V, D]
        logits = torch.matmul(h_motor, self.w_emb.t()) # [P, B, V]
        
        return logits, current_states

    def evaluate_and_select(
        self,
        stream_batch: List[Tuple[str, str]], # List of (prompt_str, target_str)
        thinking_cycles: int = 3,
        mutation_rate: float = 0.05
    ) -> Dict[str, float]:
        """
        Single-Pass Evaluation over continuous stream, computing Free Energy F_t,
        applying Thermodynamic Darwinian Selection, Apoptosis, and Epigenetic Sprouting.
        """
        P = self.P
        B = len(stream_batch)
        D = self.D
        K = self.K

        # Convert strings to byte tensors (stream_batch items: (p, t, full))
        prompts = [list(item[0].encode('utf-8')) for item in stream_batch]
        targets = [list(item[1].encode('utf-8')) + [10] for item in stream_batch]
        
        # Max lengths
        p_max = max(len(p) for p in prompts)
        t_max = max(len(t) for t in targets)
        
        p_tensor = torch.zeros(B, p_max, dtype=torch.long, device=self.device)
        t_tensor = torch.zeros(B, t_max, dtype=torch.long, device=self.device)
        
        for b in range(B):
            p_tensor[b, -len(prompts[b]):] = torch.tensor(prompts[b], device=self.device)
            t_tensor[b, :len(targets[b])] = torch.tensor(targets[b], device=self.device)

        # Expand for all P organisms: [P, B, L]
        p_pop = p_tensor.unsqueeze(0).expand(P, -1, -1)
        t_pop = t_tensor.unsqueeze(0).expand(P, -1, -1)

        # Initial zero states for population: [P, K, B, D]
        node_states = torch.zeros(P, K, B, D, device=self.device)
        
        # Phase 1: Stream prompt sequentially (Continuous Time: t -> t+1)
        for t_step in range(p_max):
            token_in = p_pop[:, :, t_step] # [P, B]
            _, node_states = self.forward_stream_step(token_in, node_states, thinking_cycles=1)

        # Phase 2: Autoregressive Target Stream Generation & Free Energy Measurement
        total_free_energy = torch.zeros(P, device=self.device)
        exact_matches = torch.zeros(P, B, device=self.device)
        
        curr_token = p_pop[:, :, -1] # [P, B]
        
        with torch.no_grad():
            for t_step in range(t_max):
                logits, node_states = self.forward_stream_step(curr_token, node_states, thinking_cycles=thinking_cycles)
                target_tok = t_pop[:, :, t_step] # [P, B]
                
                # Cross-Entropy / Surprise Free Energy: F_t = -log P(target)
                log_probs = F.log_softmax(logits, dim=-1) # [P, B, V]
                step_nll = F.nll_loss(
                    log_probs.reshape(P * B, self.V),
                    target_tok.reshape(P * B),
                    reduction='none'
                ).reshape(P, B)
                
                total_free_energy += step_nll.mean(dim=1)
                
                pred_tok = logits.argmax(dim=-1) # [P, B]
                # Check match if before newline
                is_correct = (pred_tok == target_tok).float()
                exact_matches[:, :] += is_correct
                
                curr_token = target_tok # Teacher forcing during energy evaluation

        # Normalize Free Energy per token
        avg_free_energy = total_free_energy / t_max # [P]
        
        # 3. Thermodynamic Darwinian Selection (Edelman Neural Darwinism)
        # Lower Free Energy = Higher Fitness
        best_energy, best_idx = avg_free_energy.min(dim=0)
        median_energy = avg_free_energy.median()
        
        # Sort organisms by Free Energy
        sorted_indices = torch.argsort(avg_free_energy)
        survivor_count = P // 4 # Top 25% survive
        survivors = sorted_indices[:survivor_count]
        eliminated = sorted_indices[survivor_count:]

        # 4. Epigenetic Morphogenesis & Sprouting (Reproduction + Mutation into Eliminated slots)
        with torch.no_grad():
            for slot_idx in eliminated:
                # Select a random parent from survivors
                parent_idx = survivors[random.randint(0, survivor_count - 1)]
                
                # Clone parent DNA
                self.w_route.data[slot_idx] = self.w_route.data[parent_idx].clone()
                self.alpha_epi.data[slot_idx] = self.alpha_epi.data[parent_idx].clone()
                self.w_node_in.data[slot_idx] = self.w_node_in.data[parent_idx].clone()
                self.w_node_out.data[slot_idx] = self.w_node_out.data[parent_idx].clone()
                self.w_sensory.data[slot_idx] = self.w_sensory.data[parent_idx].clone()
                self.w_motor.data[slot_idx] = self.w_motor.data[parent_idx].clone()
                
                # Mutate Routing Topology (Sprouting new synapses)
                route_mutation = torch.randn_like(self.w_route.data[slot_idx]) * mutation_rate
                mask = (torch.rand_like(route_mutation) < 0.2).float()
                self.w_route.data[slot_idx] += route_mutation * mask
                
                # Mutate Epigenetic Expression (Neurogenesis / Sprouting Nodes)
                epi_mutation = torch.randn_like(self.alpha_epi.data[slot_idx]) * (mutation_rate * 2.0)
                self.alpha_epi.data[slot_idx] += epi_mutation
                
                # Mutate Synaptic Weights
                syn_mutation = torch.randn_like(self.w_node_in.data[slot_idx]) * mutation_rate
                self.w_node_in.data[slot_idx] += syn_mutation

        return {
            "min_free_energy": best_energy.item(),
            "median_free_energy": median_energy.item(),
            "best_organism_id": best_idx.item(),
            "active_nodes_best": (torch.tanh(self.alpha_epi[best_idx]) > 0.1).sum().item()
        }

# =====================================================================
# 2. RUN EXPERIMENT EXP-299
# =====================================================================

def main():
    print("=" * 80)
    print("🧬 EXP-299: SOVEREIGN PARALLEL MORPHIC CRUCIBLE BENCHMARK")
    print("   [Zero Epochs | Stream Time | Discrete Space | GPU Parallelism | F_t Darwinism]")
    print("=" * 80)

    suite = generate_multi_domain_suite(seed=42)
    stream_data = suite['reversal'] + suite['pointer'] + suite['dyck'] + suite['parity']
    random.shuffle(stream_data)

    print(f"Total Stream Length: {len(stream_data)} continuous events.")
    print(f"Population Size: P=64 organisms in parallel on {device}.")
    print(f"Graph Capacity: K=8 primitive nodes per organism.")

    crucible = ParallelMorphicCrucible(
        pop_size=64,
        dim=64,
        max_nodes=8,
        vocab_size=258,
        device=device
    )

    batch_size = 16
    total_steps = len(stream_data) // batch_size
    
    start_time = time.time()
    print("\n--- Initiating Continuous Stream Evolution (Single-Pass) ---")
    
    for step in range(total_steps):
        batch = stream_data[step * batch_size : (step + 1) * batch_size]
        # Dynamic mutation rate: cooled as system adapts
        mut_rate = max(0.01, 0.08 * (1.0 - step / total_steps))
        
        metrics = crucible.evaluate_and_select(
            batch,
            thinking_cycles=3,
            mutation_rate=mut_rate
        )
        
        if step % 10 == 0 or step == total_steps - 1:
            print(
                f"Stream Step {step:3d}/{total_steps} | "
                f"Best F_t: {metrics['min_free_energy']:.4f} nats | "
                f"Median F_t: {metrics['median_free_energy']:.4f} nats | "
                f"Active Nodes: {metrics['active_nodes_best']}/{crucible.K} | "
                f"MutRate: {mut_rate:.4f}"
            )

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"✅ Continuous Stream Evolution Completed in {elapsed:.2f}s!")
    print(f"Final Sovereign Best Free Energy: {metrics['min_free_energy']:.4f} nats.")
    print("=" * 80)

if __name__ == "__main__":
    main()
