#!/usr/bin/env python3
"""
================================================================================
EXP-302: DUAL-PHASE LATENT SETTLING WITH CONTINUOUS C-SSD SOLITON FOCUS
================================================================================
Strict KEP-compliant implementation:
  1. Phase 1 (Perception): Sequential stream processing of prompt bytes (t -> t+1).
  2. Phase 1.5 (Latent Settling / Deliberative Thinking - KEP Principle 21):
     Decouples problem time from thinking time. Recirculates prompt field through
     K_settle = 4 closed-loop recurrent cycles to condense phase-space attractors.
  3. Phase 2 (Continuous Soliton Motor Generation):
     Zero discrete index integer pointers! Gaze focus is driven strictly by
     differentiable continuous C-SSD gaussian soliton resonant field.
  4. Two-Tier Baldwinian Engine:
     - Macro: Topology & Epigenetic evolution across P=32 organisms.
     - Micro: Online local predictive coding gradient step with Synaptic Scaling.
  5. Multi-Domain Audit: Reversal, Pointer, Dyck-1, Parity, Addition.
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
# 1. CONTINUOUS C-SSD SOLITON FOCUS & SETTLING ENGINE
# =====================================================================

class DualPhaseContinuousSolitonEngine:
    """
    Sovereign Cognitive Engine with Deliberative Latent Settling (KEP Principle 21)
    and Continuous C-SSD Soliton Attractor Gaze.
    """
    def __init__(
        self,
        pop_size: int = 32,
        dim: int = 64,
        max_nodes: int = 8,
        vocab_size: int = 258,
        settle_cycles: int = 4,
        local_lr: float = 0.08,
        device: torch.device = device
    ):
        self.P = pop_size
        self.D = dim
        self.K = max_nodes
        self.V = vocab_size
        self.settle_cycles = settle_cycles
        self.local_lr = local_lr
        self.device = device

        # Macro-Genome (Evolutionary Topologies & Epigenetic Locks)
        self.w_route = nn.Parameter(torch.randn(pop_size, max_nodes, max_nodes, device=device) * 0.1)
        self.alpha_epi = nn.Parameter(torch.zeros(pop_size, max_nodes, device=device))
        self.alpha_epi.data[:, 0] = 3.0 # Accumulator
        self.alpha_epi.data[:, 1] = 3.0 # Multiplicative
        self.alpha_epi.data[:, 2] = 3.0 # Continuous Attractor

        # Micro-Synaptic Continuous Plasticity Substrate
        self.w_emb = nn.Parameter(torch.randn(vocab_size, dim, device=device) * 0.1)
        self.w_node_in = nn.Parameter(torch.randn(pop_size, max_nodes, dim, dim, device=device) * 0.1)
        self.w_node_out = nn.Parameter(torch.randn(pop_size, max_nodes, dim, dim, device=device) * 0.1)
        self.w_sensory = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        self.w_motor = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        
        # Continuous C-SSD Soliton Resonance Matrices (100% Differentiable Field)
        self.w_focus_q = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        self.w_content_k = nn.Parameter(torch.randn(pop_size, dim, dim, device=device) * 0.1)
        self.w_copy_gate = nn.Parameter(torch.randn(pop_size, dim, 1, device=device) * 0.1)
        
        # Soliton wave width / sharpen factor (learned continuously per organism)
        self.soliton_sharpness = nn.Parameter(torch.ones(pop_size, 1, 1, device=device) * 5.0)

    def forward_graph_cycle(
        self,
        current_states: torch.Tensor, # [P, K, B, D]
        sensory_bias: torch.Tensor,   # [P, B, D]
        cycles: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Executes internal recurrent deliberation cycles across the morphic graph.
        """
        P, K, B, D = current_states.shape
        gated_route = torch.tanh(self.w_route) # [P, K, K]
        graft_gate = torch.tanh(self.alpha_epi).unsqueeze(-1).unsqueeze(-1) # [P, K, 1, 1]

        recorded_inputs = []
        recorded_acts = []

        states = current_states.clone()
        for _ in range(cycles):
            agg_inputs = torch.einsum('pij,pibd->pjbd', gated_route, states)
            agg_inputs[:, 0] = agg_inputs[:, 0] + sensory_bias
            recorded_inputs.append(agg_inputs)

            projected_in = torch.einsum('pkbd,pkde->pkbe', agg_inputs, self.w_node_in)

            # Atomic Primitive Wave Equations
            act = torch.empty_like(projected_in)
            act[:, 0::3] = torch.tanh(projected_in[:, 0::3])
            act[:, 1::3] = F.silu(projected_in[:, 1::3])
            act[:, 2::3] = torch.tanh(projected_in[:, 2::3] * 3.0)
            recorded_acts.append(act)

            projected_out = torch.einsum('pkbe,pked->pkbd', act, self.w_node_out)
            states = graft_gate * projected_out + (1.0 - graft_gate) * states

        effective_alpha = torch.softmax(self.alpha_epi, dim=-1).unsqueeze(-1).unsqueeze(-1)
        integrated_field = (states * effective_alpha).sum(dim=1) # [P, B, D]

        return integrated_field, states, recorded_inputs, recorded_acts

    def execute_latent_settling_phase(
        self,
        node_states: torch.Tensor,   # [P, K, B, D]
        prompt_field: torch.Tensor,  # [P, B, L, D]
        cycles: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 1.5: Decoupled Deliberative Thinking (KEP Principle 21).
        Recirculates the integrated prompt field through the recurrent morphic graph
        without emitting tokens, allowing phase space attractors to settle.
        """
        # Global context vector from prompt field: [P, B, D]
        prompt_summary = prompt_field.mean(dim=2)
        
        states = node_states.clone()
        for _ in range(cycles):
            h_int, states, _, _ = self.forward_graph_cycle(states, prompt_summary, cycles=1)
            
        return h_int, states

    def compute_continuous_soliton_gaze(
        self,
        h_deliberation: torch.Tensor, # [P, B, D]
        prompt_field: torch.Tensor   # [P, B, L, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes 100% continuous, differentiable C-SSD Soliton resonance gaze over prompt field.
        Returns:
          - fused_copy_dist: [P, B, V]
          - p_copy: [P, B]
        """
        P, B, L, D = prompt_field.shape
        
        # 1. Project Continuous Query & Continuous Keys
        q = torch.einsum('pbd,pde->pbe', h_deliberation, self.w_focus_q).unsqueeze(2) # [P, B, 1, D]
        k = torch.einsum('pblm,pme->pble', prompt_field, self.w_content_k)            # [P, B, L, D]
        
        # 2. Continuous Resonant Energy Landscape (Inner Product + Contrast Field)
        mean_field = prompt_field.mean(dim=2, keepdim=True)
        contrast = torch.norm(prompt_field - mean_field, dim=-1) # [P, B, L]
        
        raw_energy = torch.einsum('pbxd,pbyd->pbxy', q, k).squeeze(2) / (D ** 0.5) # [P, B, L]
        
        # 3. Continuous Soliton Wave Sharpening
        soliton_logits = (raw_energy + contrast) * self.soliton_sharpness # [P, B, L]
        continuous_gaze = F.softmax(soliton_logits, dim=-1)               # [P, B, L]
        
        # 4. Continuous Copy Gating
        p_copy = torch.sigmoid(torch.einsum('pbd,pdm->pbm', h_deliberation, self.w_copy_gate)).squeeze(-1) # [P, B]
        
        return continuous_gaze, p_copy

    def apply_predictive_coding_step(
        self,
        error_gradient: torch.Tensor, # [P, B, D]
        recorded_inputs: List[torch.Tensor],
        recorded_acts: List[torch.Tensor],
        integrated_field: torch.Tensor,
        lr: float
    ):
        """
        Applies local predictive coding update with homeostatic synaptic scaling.
        """
        P, B, D = error_gradient.shape
        with torch.no_grad():
            grad_w_motor = torch.einsum('pbd,pbe->pde', integrated_field, error_gradient) / float(B)
            self.w_motor.data -= lr * grad_w_motor
            motor_norm = torch.norm(self.w_motor.data, dim=(-2, -1), keepdim=True) + 1e-6
            self.w_motor.data = self.w_motor.data / motor_norm * (D ** 0.5 * 0.1)

            delta_int = torch.bmm(error_gradient, self.w_motor.transpose(1, 2))
            
            if recorded_inputs and recorded_acts:
                last_in = recorded_inputs[-1]
                last_act = recorded_acts[-1]
                
                delta_int_exp = delta_int.unsqueeze(1).expand(-1, self.K, -1, -1)
                grad_node_out = torch.einsum('pkbe,pkbd->pked', last_act, delta_int_exp) / float(B)
                self.w_node_out.data -= lr * grad_node_out
                
                delta_act = torch.einsum('pkbd,pked->pkbe', delta_int_exp, self.w_node_out.data)
                grad_node_in = torch.einsum('pkbd,pkbe->pkde', last_in, delta_act) / float(B)
                self.w_node_in.data -= lr * grad_node_in
                
                node_in_norm = torch.norm(self.w_node_in.data, dim=(-2, -1), keepdim=True) + 1e-6
                self.w_node_in.data = self.w_node_in.data / node_in_norm * (D ** 0.5 * 0.1)
                
                node_out_norm = torch.norm(self.w_node_out.data, dim=(-2, -1), keepdim=True) + 1e-6
                self.w_node_out.data = self.w_node_out.data / node_out_norm * (D ** 0.5 * 0.1)

    def process_and_evolve_stream(
        self,
        stream_batch: List[Tuple[str, str, str]],
        thinking_cycles: int = 3,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.70
    ) -> Dict[str, float]:
        """
        Executes Dual-Phase Stream Processing:
          - Phase 1: Sequential Prompt Ingestion
          - Phase 1.5: Latent Settling Deliberation (Thinking Time)
          - Phase 2: Autoregressive Motor Generation with Continuous Soliton Gaze
        """
        P = self.P
        B = len(stream_batch)
        D = self.D
        K = self.K

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
        
        # --- PHASE 1: SEQUENTIAL PROMPT INGESTION ---
        for t_step in range(p_max):
            token_in = p_pop[:, :, t_step]
            x_emb = F.embedding(token_in, self.w_emb)
            x_sens = torch.bmm(x_emb, self.w_sensory)
            node_states[:, 0] = node_states[:, 0] + x_sens
            
            h_int, node_states, _, _ = self.forward_graph_cycle(
                node_states, x_sens, cycles=1
            )
            prompt_field_history.append(h_int)

        p_field = torch.stack(prompt_field_history, dim=2) # [P, B, p_max, D]
        
        # --- PHASE 1.5: LATENT SETTLING DELIBERATION (THINKING TIME) ---
        h_settled, node_states = self.execute_latent_settling_phase(
            node_states, p_field, cycles=self.settle_cycles
        )

        # --- PHASE 2: CONTINUOUS SOLITON MOTOR GENERATION ---
        total_free_energy = torch.zeros(P, device=self.device)
        correct_token_count = torch.zeros(P, device=self.device)
        total_tokens = B * t_max
        
        curr_token = p_pop[:, :, -1]
        
        for t_step in range(t_max):
            target_tok = t_pop[:, :, t_step]
            
            x_emb = F.embedding(curr_token, self.w_emb)
            x_sens = torch.bmm(x_emb, self.w_sensory)
            node_states[:, 0] = node_states[:, 0] + x_sens
            
            h_int, node_states, rec_in, rec_act = self.forward_graph_cycle(
                node_states, x_sens, cycles=thinking_cycles
            )
            
            # Continuous Soliton Gaze over settled prompt field:
            soliton_gaze, p_copy = self.compute_continuous_soliton_gaze(h_int, p_field)
            
            copy_dist = torch.zeros(P, B, self.V, device=self.device)
            copy_dist.scatter_add_(2, p_pop, soliton_gaze)
            
            h_motor = torch.bmm(h_int, self.w_motor)
            gen_logits = torch.matmul(h_motor, self.w_emb.t())
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
            
            # Local Predictive Coding Update
            with torch.no_grad():
                target_one_hot = F.one_hot(target_tok, num_classes=self.V).float()
                prob_error = (fused_probs - target_one_hot)
                motor_error = torch.matmul(prob_error, self.w_emb)
                self.apply_predictive_coding_step(
                    error_gradient=motor_error,
                    recorded_inputs=rec_in,
                    recorded_acts=rec_act,
                    integrated_field=h_int,
                    lr=self.local_lr
                )
            
            curr_token = target_tok

        avg_free_energy = total_free_energy / t_max
        accuracy_percent = (correct_token_count / total_tokens) * 100.0
        
        # Macro Evolution on Topologies
        best_energy, best_idx = avg_free_energy.min(dim=0)
        best_acc = accuracy_percent[best_idx].item()
        median_energy = avg_free_energy.median().item()
        
        sorted_indices = torch.argsort(avg_free_energy)
        survivor_count = max(4, P // 4)
        survivors = sorted_indices[:survivor_count]
        eliminated = sorted_indices[survivor_count:]
        
        with torch.no_grad():
            for slot_idx in eliminated:
                p1_idx = survivors[random.randint(0, survivor_count - 1)]
                p2_idx = survivors[random.randint(0, survivor_count - 1)]
                
                # Topological Recombination
                if random.random() < crossover_rate and p1_idx != p2_idx:
                    r_mask = (torch.rand_like(self.w_route.data[slot_idx]) < 0.5).float()
                    self.w_route.data[slot_idx] = r_mask * self.w_route.data[p1_idx] + (1.0 - r_mask) * self.w_route.data[p2_idx]
                    
                    e_mask = (torch.rand_like(self.alpha_epi.data[slot_idx]) < 0.5).float()
                    self.alpha_epi.data[slot_idx] = e_mask * self.alpha_epi.data[p1_idx] + (1.0 - e_mask) * self.alpha_epi.data[p2_idx]
                else:
                    self.w_route.data[slot_idx] = self.w_route.data[p1_idx].clone()
                    self.alpha_epi.data[slot_idx] = self.alpha_epi.data[p1_idx].clone()

                # Inherit plastic weights from winning lineage
                self.w_node_in.data[slot_idx] = self.w_node_in.data[p1_idx].clone()
                self.w_node_out.data[slot_idx] = self.w_node_out.data[p1_idx].clone()
                self.w_sensory.data[slot_idx] = self.w_sensory.data[p1_idx].clone()
                self.w_motor.data[slot_idx] = self.w_motor.data[p1_idx].clone()
                self.w_focus_q.data[slot_idx] = self.w_focus_q.data[p1_idx].clone()
                self.w_content_k.data[slot_idx] = self.w_content_k.data[p1_idx].clone()
                self.w_copy_gate.data[slot_idx] = self.w_copy_gate.data[p1_idx].clone()
                self.soliton_sharpness.data[slot_idx] = self.soliton_sharpness.data[p1_idx].clone()

                # Mutations
                route_mut = torch.randn_like(self.w_route.data[slot_idx]) * mutation_rate
                self.w_route.data[slot_idx] += route_mut * (torch.rand_like(route_mut) < 0.20).float()
                
                epi_mut = torch.randn_like(self.alpha_epi.data[slot_idx]) * (mutation_rate * 2.0)
                self.alpha_epi.data[slot_idx] += epi_mut
                
                self.soliton_sharpness.data[slot_idx] += torch.randn_like(self.soliton_sharpness.data[slot_idx]) * 0.1

        return {
            "min_free_energy": best_energy.item(),
            "best_acc": best_acc,
            "median_free_energy": median_energy,
            "best_organism_id": best_idx.item(),
            "active_nodes": (torch.tanh(self.alpha_epi[best_idx]) > 0.1).sum().item(),
            "sharpness": self.soliton_sharpness[best_idx].item()
        }

# =====================================================================
# 2. RUN EXPERIMENT EXP-302 & ISOLATED DOMAIN AUDIT
# =====================================================================

def evaluate_exact_matches_by_domain(engine: DualPhaseContinuousSolitonEngine, suite: Dict[str, List]) -> Dict[str, Dict]:
    results = {}
    P = engine.P
    D = engine.D
    K = engine.K
    
    with torch.no_grad():
        for domain_name, data in suite.items():
            B = len(data)
            prompts = [list(item[0].encode('utf-8')) for item in data]
            targets = [list(item[1].encode('utf-8')) + [10] for item in data]
            raw_targets = [item[1] for item in data]
            
            p_max = max(len(p) for p in prompts)
            t_max = max(len(t) for t in targets)
            
            p_tensor = torch.zeros(B, p_max, dtype=torch.long, device=device)
            t_tensor = torch.zeros(B, t_max, dtype=torch.long, device=device)
            
            for b in range(B):
                p_tensor[b, -len(prompts[b]):] = torch.tensor(prompts[b], device=device)
                t_tensor[b, :len(targets[b])] = torch.tensor(targets[b], device=device)
                
            p_pop = p_tensor.unsqueeze(0).expand(P, -1, -1)
            t_pop = t_tensor.unsqueeze(0).expand(P, -1, -1)
            
            node_states = torch.zeros(P, K, B, D, device=device)
            prompt_field_history = []
            
            # Perception
            for t_step in range(p_max):
                token_in = p_pop[:, :, t_step]
                x_emb = F.embedding(token_in, engine.w_emb)
                x_sens = torch.bmm(x_emb, engine.w_sensory)
                node_states[:, 0] = node_states[:, 0] + x_sens
                h_int, node_states, _, _ = engine.forward_graph_cycle(node_states, x_sens, cycles=1)
                prompt_field_history.append(h_int)
                
            p_field = torch.stack(prompt_field_history, dim=2)
            
            # Latent Settling (Thinking)
            _, node_states = engine.execute_latent_settling_phase(node_states, p_field, cycles=engine.settle_cycles)
            
            # Generation
            curr_token = p_pop[:, :, -1]
            generated_tokens = torch.zeros(P, B, t_max, dtype=torch.long, device=device)
            total_nll = torch.zeros(P, device=device)
            
            for t_step in range(t_max):
                target_tok = t_pop[:, :, t_step]
                
                x_emb = F.embedding(curr_token, engine.w_emb)
                x_sens = torch.bmm(x_emb, engine.w_sensory)
                node_states[:, 0] = node_states[:, 0] + x_sens
                
                h_int, node_states, _, _ = engine.forward_graph_cycle(node_states, x_sens, cycles=3)
                
                soliton_gaze, p_copy = engine.compute_continuous_soliton_gaze(h_int, p_field)
                copy_dist = torch.zeros(P, B, engine.V, device=device)
                copy_dist.scatter_add_(2, p_pop, soliton_gaze)
                
                h_motor = torch.bmm(h_int, engine.w_motor)
                gen_logits = torch.matmul(h_motor, engine.w_emb.t())
                gen_probs = F.softmax(gen_logits, dim=-1)
                
                p_copy_exp = p_copy.unsqueeze(-1)
                fused_probs = (1.0 - p_copy_exp) * gen_probs + p_copy_exp * copy_dist + 1e-9
                
                step_nll = F.nll_loss(
                    torch.log(fused_probs).reshape(P * B, engine.V),
                    target_tok.reshape(P * B),
                    reduction='none'
                ).reshape(P, B)
                total_nll += step_nll.mean(dim=1)
                
                pred_tok = fused_probs.argmax(dim=-1)
                generated_tokens[:, :, t_step] = pred_tok
                curr_token = pred_tok
                
            best_org = total_nll.argmin().item()
            
            exact_matches = 0
            for b in range(B):
                pred_str = bytes(generated_tokens[best_org, b].tolist()).split(b'\n')[0].decode('utf-8', errors='replace')
                if pred_str == raw_targets[b]:
                    exact_matches += 1
                    
            results[domain_name] = {
                "exact_match": (exact_matches / B) * 100.0,
                "loss": (total_nll[best_org] / t_max).item(),
                "sample_target": raw_targets[0],
                "sample_pred": bytes(generated_tokens[best_org, 0].tolist()).split(b'\n')[0].decode('utf-8', errors='replace')
            }
    return results

def main():
    print("=" * 80)
    print("🌟 EXP-302: DUAL-PHASE LATENT SETTLING WITH CONTINUOUS C-SSD SOLITON FOCUS")
    print("   [Thinking Time: K_settle=4 | Gaze: 100% Continuous Soliton Wave]")
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

    print(f"Total Stream Length: {len(stream_data)} continuous events.")
    print(f"Population: P=32 on {device} | Settling Cycles: K=4.")

    engine = DualPhaseContinuousSolitonEngine(
        pop_size=32,
        dim=64,
        max_nodes=8,
        vocab_size=258,
        settle_cycles=4,
        local_lr=0.08,
        device=device
    )

    batch_size = 16
    total_steps = len(stream_data) // batch_size
    
    start_time = time.time()
    print("\n--- Initiating Dual-Phase Settling Stream (N=1 Single Pass) ---")
    
    for step in range(total_steps):
        batch = stream_data[step * batch_size : (step + 1) * batch_size]
        mut_rate = max(0.01, 0.08 * (1.0 - step / total_steps))
        
        metrics = engine.process_and_evolve_stream(
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
                f"Active Nodes: {metrics['active_nodes']}/{engine.K} | "
                f"Sharpness: {metrics['sharpness']:.2f}"
            )

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"✅ Dual-Phase Settling Stream Completed in {elapsed:.2f}s!")
    
    print("\n--- 🔍 RUTHLESS DOMAIN BREAKDOWN AUDIT (EXACT TASK COMPLETION) ---")
    domain_eval = evaluate_exact_matches_by_domain(engine, suite)
    for d, r in domain_eval.items():
        print(f"Domain: {d:10s} | Exact Match: {r['exact_match']:5.1f}% | Loss: {r['loss']:.4f} nats")
        print(f"   Target: '{r['sample_target']}' -> Generated: '{r['sample_pred']}'")
    print("=" * 80)

if __name__ == "__main__":
    main()
