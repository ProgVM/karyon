import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from karyon_agent import CoREAgent  # noqa: E402


def run_exp_307_sleep_apoptosis_benchmark():
    print("=" * 80)
    print("EXP-307: NEURODARWINIAN SLEEP PRUNING & ORGANELLE APOPTOSIS BENCHMARK")
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
    lambda_util = 0.90

    # Instantiate Agent
    agent = CoREAgent(vocab_size=258, embed_dim=dim, device=device_str)

    # 1. WAKING PHASE: SETUP VITAL ORGANELLES & SPROUT PARASITIC DUMMY ORGANELLE
    # Organelle A (Node 0: 'core_acc') -> f(x) = -x
    # Organelle B (Node 2: 'organelle_b') -> g(x) = x + 2.0 (Vital plastic clone)
    agent.duplicate_node(0, 'organelle_b', initial_alpha=1.0)

    # Organelle C (Node 3: 'parasite_c') -> Parasitic/Idle unutilised organelle (initial_alpha = 0.0)
    agent.duplicate_node(0, 'parasite_c', initial_alpha=0.0)

    # Functional Matrices
    W_A = -torch.eye(dim, device=device)
    W_A[dim - 1, dim - 1] = 1.0  # Affine anchor

    W_B = torch.eye(dim, device=device)
    W_B[:dim - 1, dim - 1] = 2.0  # Bias +2.0

    params = agent.graph.named_parameters_map()
    with torch.no_grad():
        params['node_0_core_acc_w'].copy_(W_A)
        params['node_2_organelle_b_w'].copy_(W_B)

        # Set epigenetic gates
        params['alpha_core_acc'].copy_(torch.tensor(3.0, device=device))
        params['alpha_organelle_b'].copy_(torch.tensor(3.0, device=device))
        # Parasite C has alpha_epi = 0.0

        # Lock Vital Organelles
        agent.lock_node(0, 1.0)
        agent.lock_node(2, 1.0)

        # Lock Node 1 out of activation
        agent.lock_node(1, 1.0)
        params['alpha_core_sat'].copy_(torch.tensor(0.0, device=device))

    # Baseline snapshots for strict weight invariance verification
    p_A_orig = W_A.clone().detach()
    p_B_orig = W_B.clone().detach()

    # Functional Utility Tracker U_k for all K nodes
    k_nodes = agent.graph.k_nodes
    organelle_utility = torch.zeros(k_nodes, device=device)

    print("\n[WAKING PHASE INITIALIZATION]")
    print("  • Vital Organelle A (Node 0: 'core_acc')    : f(x) = -x | Lock: mu = 1.0 (FROZEN)")
    print("  • Vital Organelle B (Node 2: 'organelle_b') : g(x) = x + 2.0 | Lock: mu = 1.0 (FROZEN)")
    print("  • Idle Parasite C   (Node 3: 'parasite_c')  : Unused clone | Lock: mu = 0.0 (PLASTIC, alpha=0.0)")
    print(f"  • Total Active Nodes in Substrate          : {k_nodes} nodes")

    # Plastic parameters (Commutation Routing Orchestrator)
    plastic_params = [p for p in agent.parameters() if p.requires_grad]
    optimizer = optim.AdamW(plastic_params, lr=0.05)

    def generate_batch(bsize=batch_size):
        x_raw = torch.randn(bsize, dim - 1, device=device)
        ones = torch.ones(bsize, 1, device=device)
        x = torch.cat([x_raw, ones], dim=-1)
        y_target = torch.cat([-x_raw + 2.0, ones], dim=-1)
        return x, y_target

    def compute_accuracy(pred, target, tol=0.10):
        diff = torch.abs(pred[:, :dim - 1] - target[:, :dim - 1])
        return (diff < tol).float().mean().item() * 100.0

    # 2. WAKING PHASE: TRAIN COMPOSITE ROUTING & TRACK UTILITY
    print("\n[WAKING PHASE: TRAINING & FUNCTIONAL UTILITY TRACKING]")
    start_time = time.time()

    for step in range(1, 201):
        agent.graph.reset_state()
        x, y_target = generate_batch(batch_size)

        optimizer.zero_grad()
        out = agent(x, thinking_steps=thinking_steps)
        loss = nn.functional.mse_loss(out, y_target)
        loss.backward()
        optimizer.step()

        # Update Functional Utility Tracking U_k:
        # U_k(t) = lambda_util * U_k(t-1) + (1 - lambda_util) * Sum_j R_{j->k}
        with torch.no_grad():
            delta_route = torch.matmul(x, params['w_route_ctx'].t()).view(batch_size, 64, 64)
            active_w = params['w_route'][:k_nodes, :k_nodes]
            active_d = delta_route[:, :k_nodes, :k_nodes]
            logits = active_w.unsqueeze(0) + active_d
            logits[:, :, 1] = -1e4  # Mask node 1
            R = torch.softmax(logits, dim=-1)  # [B, src, tgt]
            step_flow = R.sum(dim=1).mean(dim=0)  # [tgt]
            organelle_utility[:k_nodes] = (
                lambda_util * organelle_utility[:k_nodes] +
                (1.0 - lambda_util) * step_flow
            )

        if step % 50 == 0 or step == 1:
            acc = compute_accuracy(out, y_target)
            print(f"  Step {step:03d} | Loss: {loss.item():.6f} | Functional Acc: {acc:.1f}%")

    # Evaluate Pre-Sleep Performance
    with torch.no_grad():
        agent.graph.reset_state()
        x_eval, y_eval = generate_batch(128)
        out_pre = agent(x_eval, thinking_steps=thinking_steps)
        pre_acc = compute_accuracy(out_pre, y_eval, tol=0.05)
        pre_loss = nn.functional.mse_loss(out_pre, y_eval).item()

    print("\n[PRE-SLEEP FUNCTIONAL UTILITY AUDIT]")
    print(f"  • Node 0 (Vital Organelle A) Utility U_0  : {organelle_utility[0].item():.4f}")
    print(f"  • Node 1 (Disabled Core Sat) Utility U_1  : {organelle_utility[1].item():.4f}")
    print(f"  • Node 2 (Vital Organelle B) Utility U_2  : {organelle_utility[2].item():.4f}")
    print(f"  • Node 3 (Idle Parasite C)   Utility U_3  : {organelle_utility[3].item():.4f}")
    print(f"  • Pre-Sleep Composite Accuracy            : {pre_acc:.2f}% (Loss: {pre_loss:.6f})")

    # 3. SLEEP PHASE: APOPTOSIS & TONONI SHY SLEEP CYCLE
    print("\n[SLEEP PHASE: TONONI SHY APOPTOSIS CYCLE]")
    pruned_nodes_count = agent.prune_inactive_nodes(threshold=0.02)

    print(f"  • Inactive Parasite Organelles Apoptosed  : {pruned_nodes_count} node(s) dissolved")
    print("  • C++20 NodePool Reclamation             : Free slots restored to pool")

    # 4. POST-SLEEP VERIFICATION & WEIGHT INVARIANCE AUDIT
    print("\n[POST-SLEEP FUNCTIONAL RECOVERY & INVARIANT AUDIT]")
    with torch.no_grad():
        agent.graph.reset_state()
        out_post = agent(x_eval, thinking_steps=thinking_steps)
        post_acc = compute_accuracy(out_post, y_eval, tol=0.05)
        post_loss = nn.functional.mse_loss(out_post, y_eval).item()

    # Weight Invariance Check on Vital Organelles
    curr_params = agent.graph.named_parameters_map()
    delta_w_a = torch.max(torch.abs(curr_params['node_0_core_acc_w'] - p_A_orig)).item()
    delta_w_b = torch.max(torch.abs(curr_params['node_2_organelle_b_w'] - p_B_orig)).item()
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("EXP-307 NEURODARWINIAN SLEEP APOPTOSIS BENCHMARK REPORT")
    print("=" * 80)
    print(f"Execution Duration                      : {elapsed_time:.2f} seconds")
    print(f"Apoptosed Parasitic Nodes Count         : {pruned_nodes_count} (Target: >= 1)")
    print(f"Pre-Sleep Composite Functional Loss     : {pre_loss:.6f}")
    print(f"Post-Sleep Composite Functional Loss    : {post_loss:.6f}")
    print(f"Pre-Sleep Composite Accuracy            : {pre_acc:.2f}%")
    print(f"Post-Sleep Composite Accuracy           : {post_acc:.2f}% (Target: 100.0%)")
    print(f"Vital Organelle A (f(x)=-x) Delta W     : {delta_w_a:.8f}")
    print(f"Vital Organelle B (g(x)=x+2) Delta W    : {delta_w_b:.8f}")

    is_positive = (
        pruned_nodes_count >= 1 and
        post_acc >= 95.0 and
        delta_w_a == 0.0 and
        delta_w_b == 0.0
    )

    if is_positive:
        print("VERDICT                                 : 🟢 POSITIVE (PROVEN SLEEP APOPTOSIS & VITALITY PRESERVATION)")
    else:
        print("VERDICT                                 : 🔴 REJECTED")
    print("=" * 80)


if __name__ == "__main__":
    run_exp_307_sleep_apoptosis_benchmark()
