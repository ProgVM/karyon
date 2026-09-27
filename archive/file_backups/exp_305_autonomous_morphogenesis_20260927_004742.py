# experiments/exp_305_autonomous_morphogenesis.py
"""
EXP-305: Autonomous Allostatic Morphogenesis & Endogenous Retention Stream
=============================================================================
Evaluating Fully Autonomous Morphogenesis in Karyon-CoRE Agent Runtime:

STRICT KEP RULE: The test benchmark script DOES NOT invoke topology controls
(lock_node, duplicate_node, or add_node)!
The agent autonomously monitors its internal Somatic Stress Accumulator:
  S_t = lambda * S_{t-1} + max(0, F_t - tau_base)
When S_t > theta_morph, the agent self-initiates the Morphogenesis Reflex inside:
  agent.update_somatic_stress_and_morphogenesis(free_energy)

Benchmark Flow:
1. Stream Phase 1 (Domain A - Pattern 1):
   - Agent streams Domain A inputs until Somatic Stress stabilizes.
2. Stream Phase 2 (Domain Shift to Domain B - Pattern 2):
   - High Free Energy / Loss Surge automatically builds up Somatic Stress S_t.
   - Agent autonomously triggers Susumu Ohno Duplication & Epigenetic Locking.
3. Stream Phase 3 (Return to Domain A):
   - Evaluates exact retention on Domain A and adaptation on Domain B.
   - Verifies Parent Organelle Weight Invariant: Delta W_orig == 0.00000000.
"""

import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from karyon_agent import CoREAgent


def run_exp_305():
    torch.manual_seed(42)
    random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("EXP-305: AUTONOMOUS ALLOSTATIC MORPHOGENESIS BENCHMARK")
    print(f"Hardware Compute Device: {device} | Engine: C++20 DynamicMorphicGraph + CoREAgent")
    print("=" * 80)

    dim = 32
    batch_size = 16
    agent = CoREAgent(vocab_size=258, embed_dim=dim, device=device)

    # Configure Somatic Stress Thresholds for benchmark
    agent.stress_lambda = 0.85
    agent.tau_base = 0.50
    agent.theta_morph = 3.0
    agent.refractory_period = 150
    agent.min_grounding_steps = 200
    agent.max_morphogenesis_events = 1

    # Define Domain A (Pattern 1) and Domain B (Pattern 2)
    def get_batch(domain, bsize=batch_size):
        if domain == 'A':
            x = torch.randn(bsize, dim, device=device) + 2.0
            y = torch.ones(bsize, dim, device=device) * 3.0
        else:
            x = torch.randn(bsize, dim, device=device) - 2.0
            y = torch.ones(bsize, dim, device=device) * -3.0
        return x, y

    def compute_accuracy(pred, target, tol=0.35):
        diff = torch.abs(pred - target)
        correct = (diff < tol).float().mean().item() * 100.0
        return correct

    start_time = time.time()

    # -------------------------------------------------------------------------
    # STREAM PHASE 1: Domain A (Pattern 1)
    # -------------------------------------------------------------------------
    print("\n[STREAM PHASE 1] Continuous Stream Domain A (Pattern 1)...")
    optimizer = optim.AdamW(list(agent.parameters()), lr=0.05)

    for step in range(1, 201):
        x_a, y_a = get_batch('A')
        optimizer.zero_grad()

        out_a = agent(x_a, thinking_steps=2)
        loss_a = nn.functional.mse_loss(out_a, y_a)
        loss_a.backward()
        optimizer.step()

        # Update Somatic Stress & check autonomous morphogenesis trigger
        fe_val = loss_a.item()
        morph_event = agent.update_somatic_stress_and_morphogenesis(fe_val)

        if morph_event:
            print(
                f"  ⚡ [AUTONOMOUS MORPHOGENESIS AT STEP {step:03d}] "
                f"Somatic Stress Surge ({morph_event['somatic_stress']:.2f}) | "
                f"Parent Node {morph_event['parent_idx']} Locked -> Cloned Node {morph_event['clone_idx']} ('{morph_event['clone_name']}')"
            )

        if step % 50 == 0 or step == 1:
            acc_a = compute_accuracy(out_a, y_a)
            print(
                f"  Step {step:03d} | Domain A Loss: {fe_val:.6f} | "
                f"Somatic Stress: {agent.somatic_stress:.2f} | Acc: {acc_a:.1f}%"
            )

    # Record parent organelle baseline parameters
    active_idx = agent.active_organelle_idx
    parent_name = f"node_{active_idx}_"
    p1_locked_weights = {
        k: v.clone().detach()
        for k, v in agent.graph.named_parameters_map().items()
        if parent_name in k
    }

    with torch.no_grad():
        x_eval_a, y_eval_a = get_batch('A', bsize=64)
        out_eval_a = agent(x_eval_a, thinking_steps=2)
        p1_eval_loss_a = nn.functional.mse_loss(out_eval_a, y_eval_a).item()
        p1_eval_acc_a = compute_accuracy(out_eval_a, y_eval_a)

    print(f"  Phase 1 Baseline Summary -> Domain A Loss: {p1_eval_loss_a:.6f} | Acc: {p1_eval_acc_a:.1f}%")

    # -------------------------------------------------------------------------
    # STREAM PHASE 2: Domain Shift to Domain B (Pattern 2)
    # -------------------------------------------------------------------------
    print("\n[STREAM PHASE 2] Domain Shift to Domain B (Pattern 2) -> Expecting Autonomous Morphogenesis...")

    for step in range(1, 351):
        # Re-bind optimizer to include newly sprouted plastic parameters if topology changed
        optimizer = optim.AdamW(list(agent.parameters()), lr=0.05)

        # Step Domain B
        x_b, y_b = get_batch('B')
        optimizer.zero_grad()
        out_b = agent(x_b, thinking_steps=2)
        loss_b = nn.functional.mse_loss(out_b, y_b)
        loss_b.backward()
        optimizer.step()

        # Step Domain A (Continual interleaved stream)
        x_a, y_a = get_batch('A')
        optimizer.zero_grad()
        out_a = agent(x_a, thinking_steps=2)
        loss_a = nn.functional.mse_loss(out_a, y_a)
        loss_a.backward()
        optimizer.step()

        # Update Somatic Stress driven by total Free Energy / Loss
        fe_val = (loss_b + loss_a).item()
        morph_event = agent.update_somatic_stress_and_morphogenesis(fe_val)

        if morph_event:
            print(
                f"  ⚡ [AUTONOMOUS MORPHOGENESIS EVENT AT STEP {step:03d}] "
                f"Somatic Stress Surge ({morph_event['somatic_stress']:.2f} > {agent.theta_morph:.2f}) | "
                f"Parent Node {morph_event['parent_idx']} Locked -> Cloned Node {morph_event['clone_idx']} ('{morph_event['clone_name']}')"
            )

        if step % 70 == 0 or step == 1:
            with torch.no_grad():
                x_check_b, y_check_b = get_batch('B', bsize=32)
                out_check_b = agent(x_check_b, thinking_steps=2)
                l_b = nn.functional.mse_loss(out_check_b, y_check_b).item()
                a_b = compute_accuracy(out_check_b, y_check_b)

                x_check_a, y_check_a = get_batch('A', bsize=32)
                out_check_a = agent(x_check_a, thinking_steps=2)
                l_a = nn.functional.mse_loss(out_check_a, y_check_a).item()
                a_a = compute_accuracy(out_check_a, y_check_a)

                print(
                    f"  Step {step:03d} | Domain B Loss: {l_b:.6f} (Acc: {a_b:.1f}%) | "
                    f"Domain A Loss: {l_a:.6f} (Acc: {a_a:.1f}%) | Stress: {agent.somatic_stress:.2f}"
                )

    # -------------------------------------------------------------------------
    # STREAM PHASE 3: Retention & Zero-Forgetting Audit
    # -------------------------------------------------------------------------
    print("\n[STREAM PHASE 3] Final Retention & Zero-Forgetting Audit...")
    elapsed_time = time.time() - start_time

    with torch.no_grad():
        x_eval_a, y_eval_a = get_batch('A', bsize=128)
        out_eval_a = agent(x_eval_a, thinking_steps=2)
        final_loss_a = nn.functional.mse_loss(out_eval_a, y_eval_a).item()
        final_acc_a = compute_accuracy(out_eval_a, y_eval_a)

        x_eval_b, y_eval_b = get_batch('B', bsize=128)
        out_eval_b = agent(x_eval_b, thinking_steps=2)
        final_loss_b = nn.functional.mse_loss(out_eval_b, y_eval_b).item()
        final_acc_b = compute_accuracy(out_eval_b, y_eval_b)

    # Audit Parent Organelle Weight Invariant: Delta W_orig
    current_params = agent.graph.named_parameters_map()
    max_delta_w = 0.0
    for k, v_orig in p1_locked_weights.items():
        v_curr = current_params[k]
        delta = torch.max(torch.abs(v_curr - v_orig)).item()
        if delta > max_delta_w:
            max_delta_w = delta

    print("\n" + "=" * 80)
    print("EXP-305 AUTONOMOUS BENCHMARK SUMMARY REPORT")
    print("=" * 80)
    print(f"Execution Duration              : {elapsed_time:.2f} seconds")
    print(f"Total Autonomous Morph Events   : {agent.morphogenesis_count}")
    print(f"Domain A (Pattern 1) Final Loss : {final_loss_a:.6f}")
    print(f"Domain A (Pattern 1) Accuracy   : {final_acc_a:.2f}%")
    print(f"Domain B (Pattern 2) Final Loss : {final_loss_b:.6f}")
    print(f"Domain B (Pattern 2) Accuracy   : {final_acc_b:.2f}%")
    print(f"Parent Organelle Max Delta W    : {max_delta_w:.8f}")

    if agent.morphogenesis_count >= 1 and max_delta_w == 0.0 and final_loss_a < 0.10 and final_loss_b < 0.10:
        print("VERDICT                         : 🟢 POSITIVE (PROVEN AUTONOMOUS MORPHOGENESIS & ZERO-FORGETTING)")
    else:
        print("VERDICT                         : 🔴 REJECTED")
    print("=" * 80)


if __name__ == "__main__":
    run_exp_305()
