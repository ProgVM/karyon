#!/usr/bin/env python3
"""
================================================================================
EXP-300: CAMBRIAN MORPHIC CRUCIBLE BENCHMARK (THE CENTENNIAL MILESTONE)
================================================================================
Grand Centennial Synthesis of KEP Cybernetics:
  1. Strict Single-Pass Stream Reality (t -> t+1, N=1, Zero Epochs, Zero AdamW)
  2. Spatiotemporal Parallelism: Population of P=128 Sovereign Organisms on Tensor Cores
  3. Genetic Crossover & Sexual Recombination (Dual-Parent DNA Merging)
  4. Speciation & Ecological Niche Protection (NEAT Fitness Sharing & Speciation)
  5. Local Fast-Weights Synaptic Plasticity (Online Hebbian/STDP Trace without backprop)
  6. Multi-Domain Challenge Suite (Reversal, Pointer, Dyck-1, Parity, Addition)
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
# 1. EXP-300: CAMBRIAN MORPHIC CRUCIBLE ENGINE
# =====================================================================

class CambrianMorphicCrucible:
    """
    Sovereign Cambrian Population Crucible (P=128 parallel organisms).
    Evaluates, breeds, speciates, and mutates dynamic discrete graph topologies
    in continuous streaming time.
    """
    def __init__(
        self,
        pop_size: int = 128,
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

        # 1. Genome DNA Pool across P=128 organisms
        self.w_emb = nn.Parameter(torch.randn(vocab_size, dim, device=device) * 0.1)
        
        # Routing Topology: [P, K, K]
        self.w_route = nn.Parameter(torch.randn(pop_size, max_nodes, max_nodes, device=device) * 0.1)
        
        # Epigenetic Expression & Methylation: [P, K]
        self.alpha_epi = nn.Parameter(torch.zeros(pop_size, max_nodes, device=device))
        self.alpha_epi.data[:, 0] = 3.0 # Core Linear Accumulator
        self.alpha_epi.data[:, 1] = 3.0 # Core Multiplicative
        self.alpha_epi.data[:, 2] = 3.0 # Core Continuous Attractor

        # Node internal parameters: [P, K, D, D]
        self.w_node_in = nn.Parameter(torch.randn(pop_size, max_nodes, dim, dim, device=device) * 0.1)
        self.w_node_out = nn.Parameter(torch.randn(pop_size, max_nodes, dim, dim, device=device) * 0.1)
        
        # Sensory in & Motor out projections: [P, dim, dim]
        self.w_sensory = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        self.w_motor = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        
        # Saccadic Focus Query, Content Key, & Dynamic Copy gating:
        self.w_focus_q = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        self.w_content_k = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        self.w_copy_gate = nn.Parameter(torch.randn(pop_size, dim, 1, device=device) * 0.1)

        # Local Online STDP / Fast-Weights Hyperparameters:
        # Fast plasticity trace factor per organism [P, 1, 1, 1]
        self.fast_weight_decay = nn.Parameter(torch.sigmoid(torch.randn(pop_size, 1, 1, 1, device=device) + 2.0)) # ~0.88
        self.fast_weight_lr = nn.Parameter(torch.sigmoid(torch.randn(pop_size, 1, 1, 1, device=device) - 2.0) * 0.05) # ~0.005

        # Species IDs for Ecological Niche Protection
        self.species_ids = torch.zeros(pop_size, dtype=torch.long, device=device)
        self.species_count = 1

    def forward_stream_step(
        self,
        x_tokens: torch.Tensor,       # [P, B]
        node_states: torch.Tensor,    # [P, K, B, D]
        fast_weights: torch.Tensor,   # [P, K, D, D] - Local dynamic synaptic trace
        thinking_cycles: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes one continuous causal step across all P organisms with local STDP plasticity.
        Returns:
          - h_integrated: [P, B, D]
          - logits: [P, B, V]
          - next_node_states: [P, K, B, D]
          - next_fast_weights: [P, K, D, D]
        """
        P, B = x_tokens.shape
        D = self.D
        K = self.K

        # 1. Sensory Embedding: [P, B, D]
        x_emb = F.embedding(x_tokens, self.w_emb) # [P, B, D]
        x_sens = torch.bmm(x_emb, self.w_sensory)  # [P, B, D]
        
        current_states = node_states.clone()
        current_states[:, 0] = current_states[:, 0] + x_sens

        # 2. Parallel Recurrent Deliberation + Fast Synaptic Modulation
        gated_route = torch.tanh(self.w_route) # [P, K, K]
        graft_gate = torch.tanh(self.alpha_epi).unsqueeze(-1).unsqueeze(-1) # [P, K, 1, 1]

        # Effective node weights = Slow DNA weights + Local Fast-Weight Trace
        eff_node_in = self.w_node_in + fast_weights # [P, K, D, D]

        for _ in range(thinking_cycles):
            agg_inputs = torch.einsum('pij,pibd->pjbd', gated_route, current_states)
            agg_inputs[:, 0] = agg_inputs[:, 0] + x_sens
            
            projected_in = torch.einsum('pkbd,pkde->pkbe', agg_inputs, eff_node_in)
            
            # Atomic Primitive Operators:
            # 0,3: Linear Leaky Accumulator (tanh)
            # 1,4: Multiplicative / Bilinear Gating (SiLU)
            # 2,5: Attractor / Decision Snapping (Hopfield steep sigmoid)
            act = torch.empty_like(projected_in)
            act[:, 0::3] = torch.tanh(projected_in[:, 0::3])
            act[:, 1::3] = F.silu(projected_in[:, 1::3])
            act[:, 2::3] = torch.tanh(projected_in[:, 2::3] * 3.0)
            
            projected_out = torch.einsum('pkbe,pked->pkbd', act, self.w_node_out)
            current_states = graft_gate * projected_out + (1.0 - graft_gate) * current_states

        # 3. Online Local STDP Plasticity Update on Fast Weights (No Backprop)
        # \Delta W = x_in^T \cdot act (Hebbian association)
        with torch.no_grad():
            stdp_delta = torch.einsum('pkbd,pkbe->pkde', agg_inputs, act) / float(B)
            next_fast_weights = self.fast_weight_decay * fast_weights + self.fast_weight_lr * stdp_delta
            next_fast_weights = torch.clamp(next_fast_weights, -0.5, 0.5)

        # 4. Motor Readout
        effective_alpha = torch.softmax(self.alpha_epi, dim=-1).unsqueeze(-1).unsqueeze(-1)
        integrated_field = (current_states * effective_alpha).sum(dim=1) # [P, B, D]
        h_motor = torch.bmm(integrated_field, self.w_motor)              # [P, B, D]
        
        logits = torch.matmul(h_motor, self.w_emb.t()) # [P, B, V]
        return integrated_field, logits, current_states, next_fast_weights

    def compute_speciation_and_fitness_sharing(
        self,
        raw_free_energies: torch.Tensor, # [P]
        compatibility_threshold: float = 0.35
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Implements NEAT / Edelman Ecological Speciation.
        Groups organisms into species by DNA distance, normalizing fitness by species size
        to protect novel evolutionary innovations.
        """
        P = self.P
        # Compute genomic pairwise distance (Routing matrix + Epigenetic expression)
        route_flat = self.w_route.view(P, -1)
        epi_flat = self.alpha_epi.view(P, -1)
        genome_flat = torch.cat([route_flat, epi_flat], dim=-1) # [P, GeneLen]
        
        # Normalize vectors for cosine/euclidean genomic distance
        dist_matrix = torch.cdist(genome_flat, genome_flat, p=2) # [P, P]
        
        # Cluster into species
        species_assignment = torch.full((P,), -1, dtype=torch.long, device=self.device)
        representatives = []
        
        for i in range(P):
            assigned = False
            for s_idx, rep_idx in enumerate(representatives):
                if dist_matrix[i, rep_idx] < compatibility_threshold:
                    species_assignment[i] = s_idx
                    assigned = True
                    break
            if not assigned:
                new_s_idx = len(representatives)
                representatives.append(i)
                species_assignment[i] = new_s_idx

        # Calculate Species Sizes
        species_counts = torch.bincount(species_assignment, minlength=len(representatives)).float()
        
        # Fitness Sharing: Shared Free Energy = Raw F_t * (1 + 0.1 * species_size)
        # (Minimizing Free Energy -> penalties for oversized over-dominant species)
        species_sizes_per_org = species_counts[species_assignment]
        shared_free_energies = raw_free_energies * (1.0 + 0.05 * torch.log(species_sizes_per_org + 1.0))
        
        self.species_ids = species_assignment
        self.species_count = len(representatives)

        return shared_free_energies, {
            "num_species": len(representatives),
            "max_species_size": int(species_counts.max().item()),
            "min_species_size": int(species_counts.min().item())
        }

    def evaluate_and_evolve_stream(
        self,
        stream_batch: List[Tuple[str, str, str]],
        thinking_cycles: int = 3,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.70
    ) -> Dict[str, float]:
        """
        Single-Pass Stream Processing with Genetic Crossover & Ecological Speciation.
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
        fast_weights = torch.zeros(P, K, D, D, device=self.device) # Fresh fast synaptic traces per stream event
        prompt_field_history = []
        
        # Phase 1: Stream prompt sequentially (Continuous Time: t -> t+1)
        for t_step in range(p_max):
            token_in = p_pop[:, :, t_step] # [P, B]
            h_int, _, node_states, fast_weights = self.forward_stream_step(
                token_in, node_states, fast_weights, thinking_cycles=1
            )
            prompt_field_history.append(h_int)

        # prompt_field: [P, B, p_max, D]
        p_field = torch.stack(prompt_field_history, dim=2)
        
        # Phase 2: Autoregressive Generation with Content-Resonant Saccades
        total_free_energy = torch.zeros(P, device=self.device)
        correct_token_count = torch.zeros(P, device=self.device)
        total_tokens = B * t_max
        
        curr_token = p_pop[:, :, -1] # [P, B]
        k_field = torch.einsum('pblm,pme->pble', p_field, self.w_content_k)
        
        mean_field = p_field.mean(dim=2, keepdim=True)
        contrast = torch.norm(p_field - mean_field, dim=-1) # [P, B, p_max]
        
        with torch.no_grad():
            for t_step in range(t_max):
                h_int, gen_logits, node_states, fast_weights = self.forward_stream_step(
                    curr_token, node_states, fast_weights, thinking_cycles=thinking_cycles
                )
                target_tok = t_pop[:, :, t_step] # [P, B]
                
                # Dynamic Content Query & Gaze:
                q = torch.einsum('pbd,pde->pbe', h_int, self.w_focus_q).unsqueeze(2) # [P, B, 1, D]
                scores = torch.einsum('pbxd,pbyd->pbxy', q, k_field).squeeze(2) / (D ** 0.5)
                gaze_bump = F.softmax((scores + contrast) * 10.0, dim=-1)
                
                # Copy distribution scatter
                p_copy = torch.sigmoid(torch.einsum('pbd,pdm->pbm', h_int, self.w_copy_gate)).squeeze(-1)
                copy_dist = torch.zeros(P, B, self.V, device=self.device)
                copy_dist.scatter_add_(2, p_pop, gaze_bump)
                
                # Fused Output Distribution
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
                
                pred_tok = fused_probs.argmax(dim=-1)
                correct_token_count += (pred_tok == target_tok).float().sum(dim=1)
                
                curr_token = target_tok

        raw_avg_free_energy = total_free_energy / t_max # [P]
        accuracy_percent = (correct_token_count / total_tokens) * 100.0 # [P]
        
        # 3. Speciation & Fitness Sharing Calculation
        shared_free_energy, spec_stats = self.compute_speciation_and_fitness_sharing(raw_avg_free_energy)
        
        best_energy, best_idx = raw_avg_free_energy.min(dim=0)
        best_acc = accuracy_percent[best_idx].item()
        median_energy = raw_avg_free_energy.median().item()
        
        # 4. Selection based on Shared Fitness (Preserves niche diversity)
        sorted_indices = torch.argsort(shared_free_energy)
        survivor_count = max(8, P // 4) # Top 25% survivors
        survivors = sorted_indices[:survivor_count]
        eliminated = sorted_indices[survivor_count:]

        # 5. Sexual Recombination (Genetic Crossover) + Epigenetic Mutation
        with torch.no_grad():
            for slot_idx in eliminated:
                # Select two parents from survivors
                p1_idx = survivors[random.randint(0, survivor_count - 1)]
                p2_idx = survivors[random.randint(0, survivor_count - 1)]
                
                # Crossover Mask for DNA attributes (50% from Parent 1, 50% from Parent 2)
                if random.random() < crossover_rate and p1_idx != p2_idx:
                    # Route Crossover
                    r_mask = (torch.rand_like(self.w_route.data[slot_idx]) < 0.5).float()
                    self.w_route.data[slot_idx] = r_mask * self.w_route.data[p1_idx] + (1.0 - r_mask) * self.w_route.data[p2_idx]
                    
                    # Epigenetic Crossover
                    e_mask = (torch.rand_like(self.alpha_epi.data[slot_idx]) < 0.5).float()
                    self.alpha_epi.data[slot_idx] = e_mask * self.alpha_epi.data[p1_idx] + (1.0 - e_mask) * self.alpha_epi.data[p2_idx]
                    
                    # Node In/Out Crossover
                    n_mask = (torch.rand_like(self.w_node_in.data[slot_idx]) < 0.5).float()
                    self.w_node_in.data[slot_idx] = n_mask * self.w_node_in.data[p1_idx] + (1.0 - n_mask) * self.w_node_in.data[p2_idx]
                    self.w_node_out.data[slot_idx] = n_mask * self.w_node_out.data[p1_idx] + (1.0 - n_mask) * self.w_node_out.data[p2_idx]
                    
                    # Sensory/Motor Crossover
                    self.w_sensory.data[slot_idx] = 0.5 * (self.w_sensory.data[p1_idx] + self.w_sensory.data[p2_idx])
                    self.w_motor.data[slot_idx] = 0.5 * (self.w_motor.data[p1_idx] + self.w_motor.data[p2_idx])
                    
                    # Focus & Gaze Crossover
                    self.w_focus_q.data[slot_idx] = 0.5 * (self.w_focus_q.data[p1_idx] + self.w_focus_q.data[p2_idx])
                    self.w_content_k.data[slot_idx] = 0.5 * (self.w_content_k.data[p1_idx] + self.w_content_k.data[p2_idx])
                    self.w_copy_gate.data[slot_idx] = 0.5 * (self.w_copy_gate.data[p1_idx] + self.w_copy_gate.data[p2_idx])
                else:
                    # Asexual Clone from Parent 1
                    self.w_route.data[slot_idx] = self.w_route.data[p1_idx].clone()
                    self.alpha_epi.data[slot_idx] = self.alpha_epi.data[p1_idx].clone()
                    self.w_node_in.data[slot_idx] = self.w_node_in.data[p1_idx].clone()
                    self.w_node_out.data[slot_idx] = self.w_node_out.data[p1_idx].clone()
                    self.w_sensory.data[slot_idx] = self.w_sensory.data[p1_idx].clone()
                    self.w_motor.data[slot_idx] = self.w_motor.data[p1_idx].clone()
                    self.w_focus_q.data[slot_idx] = self.w_focus_q.data[p1_idx].clone()
                    self.w_content_k.data[slot_idx] = self.w_content_k.data[p1_idx].clone()
                    self.w_copy_gate.data[slot_idx] = self.w_copy_gate.data[p1_idx].clone()

                # Mutations:
                route_mut = torch.randn_like(self.w_route.data[slot_idx]) * mutation_rate
                mask = (torch.rand_like(route_mut) < 0.20).float()
                self.w_route.data[slot_idx] += route_mut * mask
                
                epi_mut = torch.randn_like(self.alpha_epi.data[slot_idx]) * (mutation_rate * 2.0)
                self.alpha_epi.data[slot_idx] += epi_mut
                
                self.w_focus_q.data[slot_idx] += torch.randn_like(self.w_focus_q.data[slot_idx]) * mutation_rate
                self.w_content_k.data[slot_idx] += torch.randn_like(self.w_content_k.data[slot_idx]) * mutation_rate
                self.w_copy_gate.data[slot_idx] += torch.randn_like(self.w_copy_gate.data[slot_idx]) * mutation_rate
                self.w_node_in.data[slot_idx] += torch.randn_like(self.w_node_in.data[slot_idx]) * mutation_rate

        return {
            "min_free_energy": best_energy.item(),
            "best_acc": best_acc,
            "median_free_energy": median_energy,
            "best_organism_id": best_idx.item(),
            "active_nodes_best": (torch.tanh(self.alpha_epi[best_idx]) > 0.1).sum().item(),
            "num_species": spec_stats["num_species"],
            "max_species_size": spec_stats["max_species_size"]
        }

# =====================================================================
# 2. RUN CENTENNIAL EXPERIMENT EXP-300
# =====================================================================

def main():
    print("=" * 80)
    print("🌟 EXP-300: CAMBRIAN MORPHIC CRUCIBLE BENCHMARK (CENTENNIAL MILESTONE)")
    print("   [P=128 Organisms | Genetic Crossover | STDP Fast-Weights | Speciation]")
    print("=" * 80)

    suite = generate_multi_domain_suite(seed=42)
    stream_data = (
        suite['reversal'] + 
        suite['pointer'] + 
        suite['dyck'] + 
        suite['parity'] + 
        suite['addition']
    )
    random.shuffle(stream_data)

    print(f"Total Stream Length: {len(stream_data)} continuous events across 5 domains.")
    print(f"Population Size: P=128 organisms in parallel on {device}.")
    print(f"Graph Capacity: K=8 primitive nodes per organism.")

    crucible = CambrianMorphicCrucible(
        pop_size=128,
        dim=64,
        max_nodes=8,
        vocab_size=258,
        device=device
    )

    batch_size = 16
    total_steps = len(stream_data) // batch_size
    
    start_time = time.time()
    print("\n--- Initiating Centennial Cambrian Evolution (Single-Pass Stream) ---")
    
    for step in range(total_steps):
        batch = stream_data[step * batch_size : (step + 1) * batch_size]
        mut_rate = max(0.01, 0.08 * (1.0 - step / total_steps))
        
        metrics = crucible.evaluate_and_evolve_stream(
            batch,
            thinking_cycles=3,
            mutation_rate=mut_rate,
            crossover_rate=0.70
        )
        
        if step % 10 == 0 or step == total_steps - 1:
            print(
                f"Stream Step {step:3d}/{total_steps} | "
                f"Best F_t: {metrics['min_free_energy']:.4f} nats | "
                f"Best Acc: {metrics['best_acc']:.1f}% | "
                f"Median F_t: {metrics['median_free_energy']:.4f} nats | "
                f"Species: {metrics['num_species']} | "
                f"Active Nodes: {metrics['active_nodes_best']}/{crucible.K} | "
                f"MutRate: {mut_rate:.4f}"
            )

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"✅ Centennial Cambrian Evolution Completed in {elapsed:.2f}s!")
    print(f"Final Sovereign Best Free Energy: {metrics['min_free_energy']:.4f} nats.")
    print(f"Final Sovereign Best Token Accuracy: {metrics['best_acc']:.2f}%.")
    print(f"Final Ecological Species Count: {metrics['num_species']}.")
    print("=" * 80)

if __name__ == "__main__":
    main()
