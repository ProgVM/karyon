#!/usr/bin/env python3
"""
EXP-300 DEEP MECHANISTIC AUDIT & DOMAIN DISSECTION
================================================================================
Audits the exact task generation accuracy per domain (Pointer, Reversal, Dyck-1,
Parity, Addition) and diagnoses the exact mechanisms behind Step 0 accuracy and
end-of-stream degradation.
================================================================================
"""

import sys, os
sys.path.insert(0, '.')
import time, math, random, json
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.exp_300_cambrian_morphic_crucible import CambrianMorphicCrucible
from multi_domain_benchmark import generate_multi_domain_suite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_isolated_domain_eval(crucible: CambrianMorphicCrucible, test_suite: Dict[str, List]):
    """
    Evaluates exact task-level generation accuracy (Exact Match on target string)
    and per-byte target accuracy across each of the 5 domains separately.
    """
    domain_results = {}
    P = crucible.P
    D = crucible.D
    K = crucible.K
    
    # Evaluate the current best organism or population on each domain
    with torch.no_grad():
        for domain_name, domain_data in test_suite.items():
            B = len(domain_data)
            prompts = [list(item[0].encode('utf-8')) for item in domain_data]
            targets = [list(item[1].encode('utf-8')) + [10] for item in domain_data]
            raw_target_strs = [item[1] for item in domain_data]
            
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
            fast_weights = torch.zeros(P, K, D, D, device=device)
            prompt_field_history = []
            
            for t_step in range(p_max):
                token_in = p_pop[:, :, t_step]
                h_int, _, node_states, fast_weights = crucible.forward_stream_step(
                    token_in, node_states, fast_weights, thinking_cycles=1
                )
                prompt_field_history.append(h_int)
                
            p_field = torch.stack(prompt_field_history, dim=2)
            k_field = torch.einsum('pblm,pme->pble', p_field, crucible.w_content_k)
            mean_field = p_field.mean(dim=2, keepdim=True)
            contrast = torch.norm(p_field - mean_field, dim=-1)
            
            # Autoregressive generation without teacher forcing for Exact Match audit
            curr_token = p_pop[:, :, -1]
            generated_tokens = torch.zeros(P, B, t_max, dtype=torch.long, device=device)
            total_nll = torch.zeros(P, device=device)
            
            for t_step in range(t_max):
                h_int, gen_logits, node_states, fast_weights = crucible.forward_stream_step(
                    curr_token, node_states, fast_weights, thinking_cycles=3
                )
                target_tok = t_pop[:, :, t_step]
                
                q = torch.einsum('pbd,pde->pbe', h_int, crucible.w_focus_q).unsqueeze(2)
                scores = torch.einsum('pbxd,pbyd->pbxy', q, k_field).squeeze(2) / (D ** 0.5)
                gaze_bump = F.softmax((scores + contrast) * 10.0, dim=-1)
                
                p_copy = torch.sigmoid(torch.einsum('pbd,pdm->pbm', h_int, crucible.w_copy_gate)).squeeze(-1)
                copy_dist = torch.zeros(P, B, crucible.V, device=device)
                copy_dist.scatter_add_(2, p_pop, gaze_bump)
                
                gen_probs = F.softmax(gen_logits, dim=-1)
                p_copy_exp = p_copy.unsqueeze(-1)
                fused_probs = (1.0 - p_copy_exp) * gen_probs + p_copy_exp * copy_dist + 1e-9
                
                log_probs = torch.log(fused_probs)
                step_nll = F.nll_loss(
                    log_probs.reshape(P * B, crucible.V),
                    target_tok.reshape(P * B),
                    reduction='none'
                ).reshape(P, B)
                total_nll += step_nll.mean(dim=1)
                
                pred_tok = fused_probs.argmax(dim=-1) # [P, B]
                generated_tokens[:, :, t_step] = pred_tok
                curr_token = pred_tok # True autoregressive generation
                
            avg_nll = total_nll / t_max # [P]
            best_org = avg_nll.argmin().item()
            
            # Exact sequence match for best organism
            exact_matches = 0
            char_matches = 0
            total_chars = 0
            
            for b in range(B):
                pred_bytes = bytes(generated_tokens[best_org, b].tolist()).split(b'\n')[0]
                target_bytes = raw_target_strs[b].encode('utf-8')
                if pred_bytes == target_bytes:
                    exact_matches += 1
                
                # byte-level matches on target
                t_b = t_tensor[b].tolist()
                p_b = generated_tokens[best_org, b].tolist()
                for c1, c2 in zip(p_b, t_b):
                    if c1 == c2:
                        char_matches += 1
                total_chars += len(t_b)
                
            domain_results[domain_name] = {
                "exact_match_acc": (exact_matches / B) * 100.0,
                "target_byte_acc": (char_matches / total_chars) * 100.0,
                "domain_loss_nats": avg_nll[best_org].item(),
                "sample_target": raw_target_strs[0],
                "sample_pred": bytes(generated_tokens[best_org, 0].tolist()).split(b'\n')[0].decode('utf-8', errors='replace')
            }
            
    return domain_results

def main():
    print("=" * 80)
    print("🔍 EXP-300 RUTHLESS ENDOSCOPIC AUDIT & DOMAIN DISSECTION")
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
    
    crucible = CambrianMorphicCrucible(
        pop_size=128,
        dim=64,
        max_nodes=8,
        vocab_size=258,
        device=device
    )
    
    # 1. Audit Step 0 (Untrained Population)
    print("\n--- [AUDIT 1] Step 0 (Untrained Population Breakdown) ---")
    step0_results = run_isolated_domain_eval(crucible, suite)
    for d, r in step0_results.items():
        print(f"Domain: {d:10s} | Exact Match: {r['exact_match_acc']:5.1f}% | Target Byte Acc: {r['target_byte_acc']:5.1f}% | Loss: {r['domain_loss_nats']:.4f} nats")
        print(f"   Target: '{r['sample_target']}' -> Pred: '{r['sample_pred']}'")
        
    # 2. Run Stream Evolution
    batch_size = 16
    total_steps = len(stream_data) // batch_size
    
    step_history = []
    
    for step in range(total_steps):
        batch = stream_data[step * batch_size : (step + 1) * batch_size]
        mut_rate = max(0.01, 0.08 * (1.0 - step / total_steps))
        
        metrics = crucible.evaluate_and_evolve_stream(
            batch,
            thinking_cycles=3,
            mutation_rate=mut_rate,
            crossover_rate=0.70
        )
        step_history.append((step, metrics))
        
    # 3. Audit Step 125 (Final Evolved Population Breakdown)
    print("\n--- [AUDIT 2] Step 125 (Final Evolved Population Breakdown) ---")
    final_results = run_isolated_domain_eval(crucible, suite)
    for d, r in final_results.items():
        print(f"Domain: {d:10s} | Exact Match: {r['exact_match_acc']:5.1f}% | Target Byte Acc: {r['target_byte_acc']:5.1f}% | Loss: {r['domain_loss_nats']:.4f} nats")
        print(f"   Target: '{r['sample_target']}' -> Pred: '{r['sample_pred']}'")

if __name__ == "__main__":
    main()
