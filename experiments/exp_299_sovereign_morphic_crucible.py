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
      - Content/Focus resonance & copy gating
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
        self.w_emb = nn.Parameter(torch.randn(vocab_size, dim, device=device) * 0.1)
        
        # Routing matrices for each organism in population: [P, K, K]
        self.w_route = nn.Parameter(torch.randn(pop_size, max_nodes, max_nodes, device=device) * 0.1)
        
        # Epigenetic Expression & Methylation: [P, K] (tanh(alpha) determines node activity)
        self.alpha_epi = nn.Parameter(torch.zeros(pop_size, max_nodes, device=device))
        self.alpha_epi.data[:, 0] = 3.0 # Core Linear Accumulator
        self.alpha_epi.data[:, 1] = 3.0 # Core Multiplicative
        self.alpha_epi.data[:, 2] = 3.0 # Core Continuous Attractor

        # Node internal parameters across population [P, K, D, D]
        self.w_node_in = nn.Parameter(torch.randn(pop_size, max_nodes, dim, dim, device=device) * 0.1)
        self.w_node_out = nn.Parameter(torch.randn(pop_size, max_nodes, dim, dim, device=device) * 0.1)
        
        # Sensory in & Motor out projections: [P, dim, dim]
        self.w_sensory = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        self.w_motor = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        
        # Focus Query, Content Key, & Copy gating per organism:
        self.w_focus_q = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        self.w_content_k = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        self.w_copy_gate = nn.Parameter(torch.randn(pop_size, dim, 1, device=device) * 0.1)

    def forward_stream_step(
        self,
        x_tokens: torch.Tensor,       # [P, B]
        node_states: torch.Tensor,    # [P, K, B, D]
        thinking_cycles: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes one continuous causal step across all P organisms in parallel.
        Returns:
          - h_integrated: [P, B, D]
          - logits: [P, B, V]
          - next_node_states: [P, K, B, D]
        """
        P, B = x_tokens.shape
        D = self.D
        K = self.K

        # 1. Sensory Embedding: [P, B, D]
        x_emb = F.embedding(x_tokens, self.w_emb) # [P, B, D]
        x_sens = torch.bmm(x_emb, self.w_sensory)  # [P, B, D]
        
        # Feed sensory into root node 0
        current_states = node_states.clone()
        current_states[:, 0] = current_states[:, 0] + x_sens

        # 2. Parallel Recurrent Deliberation (Thinking Cycles across Graph Topology)
        gated_route = torch.tanh(self.w_route) # [P, K, K]
        graft_gate = torch.tanh(self.alpha_epi).unsqueeze(-1).unsqueeze(-1)

        for _ in range(thinking_cycles):
            agg_inputs = torch.einsum('pij,pibd->pjbd', gated_route, current_states)
            agg_inputs[:, 0] = agg_inputs[:, 0] + x_sens
            
            projected_in = torch.einsum('pkbd,pkde->pkbe', agg_inputs, self.w_node_in)
            
            act = torch.empty_like(projected_in)
            act[:, 0::3] = torch.tanh(projected_in[:, 0::3])
            act[:, 1::3] = F.silu(projected_in[:, 1::3])
            act[:, 2::3] = torch.tanh(projected_in[:, 2::3] * 3.0)
            
            projected_out = torch.einsum('pkbe,pked->pkbd', act, self.w_node_out)
            current_states = graft_gate * projected_out + (1.0 - graft_gate) * current_states

        # 3. Motor Readout
        effective_alpha = torch.softmax(self.alpha_epi, dim=-1).unsqueeze(-1).unsqueeze(-1)
        integrated_field = (current_states * effective_alpha).sum(dim=1) # [P, B, D]
        h_motor = torch.bmm(integrated_field, self.w_motor)              # [P, B, D]
        
        logits = torch.matmul(h_motor, self.w_emb.t()) # [P, B, V]
        return integrated_field, logits, current_states

    def evaluate_and_select(
        self,
        stream_batch: List[Tuple[str, str, str]],
        thinking_cycles: int = 3,
        mutation_rate: float = 0.05
    ) -> Dict[str, float]:
        """
        Continuous Single-Pass Stream evaluation and thermodynamic natural selection.
        """
        P = self.P
        B = len(stream_batch)
        D = self.D
        K = self.K

        # Convert strings to byte tensors
        prompts = [list(item[0].encode('utf-8')) for item in stream_batch]
        targets = [list(item[1].encode('utf-8')) + [10] for item in stream_batch]
        
        p_max = max(len(p) for p in prompts)
        t_max = max(len(t) for t in targets)
        
        p_tensor = torch.zeros(B, p_max, dtype=torch.long, device=self.device)
        t_tensor = torch.zeros(B, t_max, dtype=torch.long, device=self.device)
        
        for b in range(B):
            p_tensor[b, -len(prompts[b]):] = torch.tensor(prompts[b], device=self.device)
            t_tensor[b, :len(targets[b])] = torch.tensor(targets[b], device=self.device)

        p_pop = p_tensor.unsqueeze(0).expand(P, -1, -1)
        t_pop = t_tensor.unsqueeze(0).expand(P, -1, -1)

        node_states = torch.zeros(P, K, B, D, device=self.device)
        prompt_field_history = []
        
        # Phase 1: Stream prompt sequentially (Continuous Time: t -> t+1)
        for t_step in range(p_max):
            token_in = p_pop[:, :, t_step] # [P, B]
            h_int, _, node_states = self.forward_stream_step(token_in, node_states, thinking_cycles=1)
            prompt_field_history.append(h_int)

        # prompt_field: [P, B, p_max, D]
        p_field = torch.stack(prompt_field_history, dim=2)
        
        # Phase 2: Autoregressive Target Stream Generation with Content Resonance & Gaze
        total_free_energy = torch.zeros(P, device=self.device)
        correct_token_count = torch.zeros(P, device=self.device)
        total_tokens = B * t_max
        
        curr_token = p_pop[:, :, -1] # [P, B]
        
        # Compute Content Keys once: [P, B, p_max, D]
        k_field = torch.einsum('pblm,pme->pble', p_field, self.w_content_k)
        
        # Salience deviation (information contrast):
        mean_field = p_field.mean(dim=2, keepdim=True)
        contrast = torch.norm(p_field - mean_field, dim=-1) # [P, B, p_max]
        
        with torch.no_grad():
            for t_step in range(t_max):
                h_int, gen_logits, node_states = self.forward_stream_step(
                    curr_token, node_states, thinking_cycles=thinking_cycles
                )
                target_tok = t_pop[:, :, t_step] # [P, B]
                
                # Dynamic Content Query & Gaze:
                q = torch.einsum('pbd,pde->pbe', h_int, self.w_focus_q).unsqueeze(2) # [P, B, 1, D]
                scores = torch.einsum('pbxd,pbyd->pbxy', q, k_field).squeeze(2) / (D ** 0.5) # [P, B, p_max]
                gaze_bump = F.softmax((scores + contrast) * 10.0, dim=-1) # [P, B, p_max]
                
                # Copy distribution scatter
                p_copy = torch.sigmoid(torch.einsum('pbd,pdm->pbm', h_int, self.w_copy_gate)).squeeze(-1) # [P, B]
                copy_dist = torch.zeros(P, B, self.V, device=self.device)
                copy_dist.scatter_add_(2, p_pop, gaze_bump)
                
                # Dual-Gated Probabilities:
                gen_probs = F.softmax(gen_logits, dim=-1)
                p_copy_exp = p_copy.unsqueeze(-1)
                fused_probs = (1.0 - p_copy_exp) * gen_probs + p_copy_exp * copy_dist + 1e-9
                
                log_probs = torch.log(fused_probs)
                step_nll = F.nll_loss(
                    log_probs.reshape(P * B, self.V),
                    target_tok.reshape(P * B),
                    reduction='none'
                ).reshape(P, B)
                
                total_free_energy += step_nll.mean(dim=1)
                
                pred_tok = fused_probs.argmax(dim=-1) # [P, B]
                correct_token_count += (pred_tok == target_tok).float().sum(dim=1)
                
                curr_token = target_tok # Stream causality

        avg_free_energy = total_free_energy / t_max # [P]
        accuracy_percent = (correct_token_count / total_tokens) * 100.0 # [P]
        
        # 3. Thermodynamic Natural Selection
        best_energy, best_idx = avg_free_energy.min(dim=0)
        best_acc = accuracy_percent[best_idx].item()
        median_energy = avg_free_energy.median().item()
        
        sorted_indices = torch.argsort(avg_free_energy)
        survivor_count = max(4, P // 4)
        survivors = sorted_indices[:survivor_count]
        eliminated = sorted_indices[survivor_count:]

        # 4. Epigenetic Morphogenesis, Sprouting & Recombination
        with torch.no_grad():
            for slot_idx in eliminated:
                parent_idx = survivors[random.randint(0, survivor_count - 1)]
                
                # Clone parent DNA
                self.w_route.data[slot_idx] = self.w_route.data[parent_idx].clone()
                self.alpha_epi.data[slot_idx] = self.alpha_epi.data[parent_idx].clone()
                self.w_node_in.data[slot_idx] = self.w_node_in.data[parent_idx].clone()
                self.w_node_out.data[slot_idx] = self.w_node_out.data[parent_idx].clone()
                self.w_sensory.data[slot_idx] = self.w_sensory.data[parent_idx].clone()
                self.w_motor.data[slot_idx] = self.w_motor.data[parent_idx].clone()
                self.w_focus_q.data[slot_idx] = self.w_focus_q.data[parent_idx].clone()
                self.w_content_k.data[slot_idx] = self.w_content_k.data[parent_idx].clone()
                self.w_copy_gate.data[slot_idx] = self.w_copy_gate.data[parent_idx].clone()
                
                # Mutate Routing Topology
                route_mut = torch.randn_like(self.w_route.data[slot_idx]) * mutation_rate
                mask = (torch.rand_like(route_mut) < 0.25).float()
                self.w_route.data[slot_idx] += route_mut * mask
                
                # Mutate Epigenetic Expression (Sprouting / Gating)
                epi_mut = torch.randn_like(self.alpha_epi.data[slot_idx]) * (mutation_rate * 2.0)
                self.alpha_epi.data[slot_idx] += epi_mut
                
                # Mutate Focus & Gaze Resonances
                self.w_focus_q.data[slot_idx] += torch.randn_like(self.w_focus_q.data[slot_idx]) * mutation_rate
                self.w_content_k.data[slot_idx] += torch.randn_like(self.w_content_k.data[slot_idx]) * mutation_rate
                self.w_copy_gate.data[slot_idx] += torch.randn_like(self.w_copy_gate.data[slot_idx]) * mutation_rate
                self.w_node_in.data[slot_idx] += torch.randn_like(self.w_node_in.data[slot_idx]) * mutation_rate

        return {
            "min_free_energy": best_energy.item(),
            "best_acc": best_acc,
            "median_free_energy": median_energy,
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
                f"Best Acc: {metrics['best_acc']:.1f}% | "
                f"Median F_t: {metrics['median_free_energy']:.4f} nats | "
                f"Active Nodes: {metrics['active_nodes_best']}/{crucible.K} | "
                f"MutRate: {mut_rate:.4f}"
            )

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"✅ Continuous Stream Evolution Completed in {elapsed:.2f}s!")
    print(f"Final Sovereign Best Free Energy: {metrics['min_free_energy']:.4f} nats.")
    print(f"Final Sovereign Best Token Accuracy: {metrics['best_acc']:.2f}%.")
    print("=" * 80)

if __name__ == "__main__":
    main()
