"""
EXP-306: LAMINAR COMPOSITION & ORGANELLE CHAINING BENCHMARK
================================================================================
Core Objective:
- Prove the "Lego-principle" of functional organelle composition in Karyon-CoRE.
- Solves complex composite tasks h(x) = g(f(x)) = -x + 2.0 WITHOUT modifying internal
  organelle weights (Delta W_A == 0.00000000, Delta W_B == 0.00000000).
- Pure continuous functional composition via learned inter-organelle commutation routing
  R(h_t) and orchestrator motor readout.
================================================================================
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from karyon_agent import CoREAgent

def run_exp_306_laminar_composition_benchmark():
    print("=" * 80)
    print("EXP-306: LAMINAR COMPOSITION & ORGANELLE CHAINING BENCHMARK")
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Hardware Compute Device: {device} | Engine: C++20 DynamicMorphicGraph + CoREAgent")
    print("=" * 80)

    # Set seeds
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    dim = 16
    batch_size = 32
    thinking_steps = 2

    # Instantiate Agent
    agent = CoREAgent(vocab_size=258, embed_dim=dim, device=device_str)

    # 1. SETUP ORGANELLES & METHYLATION LOCKS
    # Duplicate Node 0 (Organelle A: LinearAccumulator) to create Node 2 (Organelle B: LinearAccumulator)
    agent.duplicate_node(0, 'organelle_b', initial_alpha=1.0)

    # Construct explicit linear transformation matrices for Organelle A and Organelle B:
    # Organelle A: f(x) = -x (Inversion)
    W_A = -torch.eye(dim, device=device)
    W_A[dim - 1, dim - 1] = 1.0  # Affine anchor

    # Organelle B: g(x) = x + 2.0 (Translation)
    W_B = torch.eye(dim, device=device)
    W_B[:dim - 1, dim - 1] = 2.0  # Bias +2.0 in affine dimension

    params = agent.graph.named_parameters_map()
    with torch.no_grad():
        # Copy exact organelle functional weights
        params['node_0_core_acc_w'].copy_(W_A)
        params['node_2_organelle_b_w'].copy_(W_B)

        # Set epigenetic gates alpha ~ 3.0 (tanh(3.0) ~ 0.995)
        params['alpha_core_acc'].copy_(torch.tensor(3.0, device=device))
        params['alpha_organelle_b'].copy_(torch.tensor(3.0, device=device))

        # Lock both Organelle A (Node 0) and Organelle B (Node 2) with methylation mu = 1.0
        agent.lock_node(0, 1.0)
        agent.lock_node(2, 1.0)

        # Lock Node 1 (core_sat) out of activation
        agent.lock_node(1, 1.0)
        params['alpha_core_sat'].copy_(torch.tensor(0.0, device=device))

    # Baseline snapshots for strict weight invariance verification
    p_A_orig = W_A.clone().detach()
    p_B_orig = W_B.clone().detach()

    # 2. COLLECT PLASTIC PARAMETERS (COMMUTATION ORCHESTRATOR ONLY)
    plastic_params = [p for p in agent.parameters() if p.requires_grad]

    print("\n[ORGANELLE INITIALIZATION & LOCK STATUS]")
    print(f"  • Organelle A (Node 0: 'core_acc')      : f(x) = -x | Methylation Lock: mu = 1.0 (FROZEN)")
    print(f"  • Organelle B (Node 2: 'organelle_b')   : g(x) = x + 2.0 | Methylation Lock: mu = 1.0 (FROZEN)")
    print(f"  • Composite Objective h(x) = g(f(x))    : h(x) = -x + 2.0")
    print(f"  • Commutation Plastic Parameters        : {len(plastic_params)} tensors (W_route, W_route_ctx, W_motor, W_sensory)")

    optimizer = optim.AdamW(plastic_params, lr=0.05)

    def generate_composite_batch(bsize=batch_size):
        x_raw = torch.randn(bsize, dim - 1, device=device)
        ones = torch.ones(bsize, 1, device=device)
        x = torch.cat([x_raw, ones], dim=-1)

        # Target composite function h(x) = g(f(x)) = -x + 2.0
        y_target_raw = -x_raw + 2.0
        y_target = torch.cat([y_target_raw, ones], dim=-1)
        return x, y_target

    def compute_accuracy(pred, target, tol=0.10):
        # Accuracy evaluated on functional output dimensions (excluding affine anchor)
        diff = torch.abs(pred[:, :dim - 1] - target[:, :dim - 1])
        correct = (diff < tol).float().mean().item() * 100.0
        return correct

    # 3. TRAINING COMMUTATION ROUTING
    print("\n[TRAINING LAMINAR COMMUTATION ROUTING]")
    start_time = time.time()

    for step in range(1, 301):
        agent.graph.reset_state()
        x, y_target = generate_composite_batch(batch_size)

        optimizer.zero_grad()
        out = agent(x, thinking_steps=thinking_steps)
        loss = nn.functional.mse_loss(out, y_target)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            acc = compute_accuracy(out, y_target)
            print(f"  Step {step:03d} | Composite MSE Loss: {loss.item():.6f} | Functional Accuracy: {acc:.1f}%")

    elapsed_time = time.time() - start_time

    # 4. FINAL VERIFICATION & WEIGHT INVARIANCE AUDIT
    print("\n[EVALUATION & INVARIANT AUDIT]")
    with torch.no_grad():
        agent.graph.reset_state()
        x_eval, y_eval = generate_composite_batch(128)
        out_eval = agent(x_eval, thinking_steps=thinking_steps)
        final_loss = nn.functional.mse_loss(out_eval, y_eval).item()
        final_acc = compute_accuracy(out_eval, y_eval, tol=0.05)

        # Inspect routing profile for Organelle A -> Organelle B chaining
        delta_route = torch.matmul(x_eval[:1], params['w_route_ctx'].t()).view(1, 64, 64)
        active_w = params['w_route'][:3, :3]
        active_d = delta_route[:, :3, :3]
        logits = active_w.unsqueeze(0) + active_d
        logits[:, :, 1] = -1e4  # Mask node 1
        R = torch.softmax(logits, dim=-1)[0]
        a_to_b_prob = R[0, 2].item() * 100.0

    # Weight Invariance Check
    curr_params = agent.graph.named_parameters_map()
    delta_w_a = torch.max(torch.abs(curr_params['node_0_core_acc_w'] - p_A_orig)).item()
    delta_w_b = torch.max(torch.abs(curr_params['node_2_organelle_b_w'] - p_B_orig)).item()

    print("\n" + "=" * 80)
    print("EXP-306 LAMINAR COMPOSITION BENCHMARK REPORT")
    print("=" * 80)
    print(f"Execution Duration              : {elapsed_time:.2f} seconds")
    print(f"Composite Functional MSE Loss   : {final_loss:.6f}")
    print(f"Composite Functional Accuracy   : {final_acc:.2f}% (Target: >= 90.0%)")
    print(f"Organelle A (f(x) = -x) Delta W : {delta_w_a:.8f}")
    print(f"Organelle B (g(x)=x+2) Delta W  : {delta_w_b:.8f}")
    print(f"Inter-Organelle Chaining Prob   : A -> B = {a_to_b_prob:.1f}%")

    is_positive = (
        final_acc >= 90.0 and
        delta_w_a == 0.0 and
        delta_w_b == 0.0
    )

    if is_positive:
        print("VERDICT                         : 🟢 POSITIVE (PROVEN LAMINAR ORGANELLE COMPOSITION)")
    else:
        print("VERDICT                         : 🔴 REJECTED")
    print("=" * 80)

if __name__ == "__main__":
    run_exp_306_laminar_composition_benchmark()
