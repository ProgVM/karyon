"""
=====================================================================================
EXP-330: STEP-WISE INSTRUCTION POINTER & CHAINED DEDUCTION BENCHMARK
=====================================================================================
Hypothesis:
  In EXP-329, providing the entire instruction chain as a static flat bag-of-words
  (context_chain) caused attention bleed: the routing query at step 1 was polluted
  with embeddings for step 0 and step 2, preventing sharp selective inter-organelle
  transitions.

  By introducing a Step-Wise Instruction Pointer where context_chain is shaped
  [B, S_ops, dim] and dynamically sliced at each thinking step:
    Q_step = (h_active_sum + context_chain[:, step, :]) * W_query_step
    routing_matrix = Softmax(W_route + Q_step * K_signatures^T / sqrt(d))
    q_readout = context_chain[:, -1, :] * W_readout_proj

  the morphic graph will execute clean, unpolluted step-by-step instruction dispatch,
  reaching >= 75% step-to-step routing fidelity across all steps (step 0, 1, 2) and
  achieving >= 75% in-distribution and >= 60% out-of-distribution deduction accuracy.
=====================================================================================
"""
import time
import math
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from karyon_agent import CoREAgent


class StepWiseChainedDeductionEnvironment:
    def __init__(self, dim: int = 16, num_ops: int = 4, device: str = "cpu"):
        self.dim = dim
        self.num_ops = num_ops
        self.device = torch.device(device)

    def apply_op(self, op_id: int, x: torch.Tensor) -> torch.Tensor:
        if op_id == 0:
            # Cyclic roll
            return torch.roll(x, shifts=2, dims=-1) * 1.05 + 0.05
        elif op_id == 1:
            # Negative reflection & sine
            return torch.flip(x, dims=[-1]) * -0.9 + torch.sin(x * 1.5) * 0.1
        elif op_id == 2:
            # Multiplicative split & swap
            h_dim = self.dim // 2
            x1, x2 = x[..., :h_dim], x[..., h_dim:]
            return torch.cat([x2 * 1.1 - 0.2 * torch.cos(x1), x1 * 0.9 + 0.2 * torch.sin(x2)], dim=-1)
        elif op_id == 3:
            # Continuous sign-modulated scale
            return torch.sign(x + 1e-5) * torch.log1p(torch.abs(x)) * 1.2
        return x

    def sample_batch(self, batch_size: int = 32, chain_length: int = 2) -> Dict[str, torch.Tensor]:
        x_0 = (torch.rand(batch_size, self.dim, device=self.device) - 0.5) * 3.0
        chain = [random.randint(0, self.num_ops - 1) for _ in range(chain_length)]

        y_curr = x_0.clone()
        for op_id in chain:
            y_curr = self.apply_op(op_id, y_curr)

        # Build 3D step-wise instruction tensor: [B, chain_length, dim]
        step_tensors = []
        for op_id in chain:
            step_oh = torch.zeros(batch_size, self.dim, device=self.device)
            step_oh[:, op_id] = 1.0
            step_tensors.append(step_oh)

        context_sequence = torch.stack(step_tensors, dim=1) # [B, chain_length, dim]

        # First op embedding vector [B, dim]
        op_first_oh = torch.zeros(batch_size, self.dim, device=self.device)
        op_first_oh[:, chain[0]] = 1.0

        return {
            "sensory": x_0,
            "target": y_curr,
            "chain": chain,
            "context_sequence": context_sequence,
            "op_first_embed": op_first_oh
        }


def run_exp_330():
    print("=" * 85)
    print("EXP-330: STEP-WISE INSTRUCTION POINTER & CHAINED DEDUCTION BENCHMARK")
    print("=" * 85)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Substrate Compute Device: {device_str.upper()}")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    embed_dim = 16
    env = StepWiseChainedDeductionEnvironment(dim=embed_dim, device=device_str)

    # 1. INITIALIZE CO-RE AGENT
    print("\n[STEP 1: INITIALIZING CO-RE AGENT SUBSTRATE]")
    agent = CoREAgent(
        vocab_size=258,
        embed_dim=embed_dim,
        device=device_str
    )
    agent.to(device)

    # Verify Identity Sensorimotor Trakt
    param_map = agent.graph.named_parameters_map()
    assert torch.equal(param_map["w_sensory_in"], torch.eye(embed_dim, device=device)), "w_sensory_in must be Identity!"
    assert torch.equal(param_map["w_motor_out"], torch.eye(embed_dim, device=device)), "w_motor_out must be Identity!"
    print("  ✓ Identity Sensorimotor Trakt Verified: w_sensory_in = Eye, w_motor_out = Eye")

    # Freeze w_sensory_in and w_motor_out to preserve identity coordinate alignment
    param_map["w_sensory_in"].requires_grad_(False)
    param_map["w_motor_out"].requires_grad_(False)
    print("  🔒 Coordinate Highway Frozen (requires_grad = False for sensory/motor identity)")

    # Add Slot Memory Organelle (Node 2)
    slot_mem_idx = agent.add_node("slot_mem_0", "SlotMemory", is_core=True, initial_alpha=1.0)

    # Add 4 Specialized NonLinearTransform Organelle Nodes (Nodes 3, 4, 5, 6)
    organelle_indices = []
    op_names = ["roll", "flip", "split_swap", "sign_log"]
    for i, name in enumerate(op_names):
        idx = agent.add_node(f"organelle_op_{i}_{name}", "NonLinearTransform", is_core=True, initial_alpha=1.0)
        organelle_indices.append(idx)

    print(f"  • Total Substrate Organelles Created  : {agent.graph.k_nodes} nodes")
    print(f"  • Slot Memory Register Organelle     : Node {slot_mem_idx} ('SlotMemory')")
    for i, idx in enumerate(organelle_indices):
        print(f"  • Specialized Organelle {i}          : Node {idx} ('NonLinearTransform' for {op_names[i]})")

    # =========================================================================
    # PHASE 1: ONTOGENETIC SPECIALIZATION & FUNCTIONAL PASSPORT INITIALIZATION
    # =========================================================================
    print("\n" + "=" * 85)
    print("PHASE 1: ONTOGENETIC SPECIALIZATION & PASSPORT INITIALIZATION")
    print("=" * 85)

    phase1_start = time.time()
    op_convergence_metrics = {}

    for op_id in range(4):
        node_idx = organelle_indices[op_id]
        op_name = op_names[op_id]
        print(f"\n  [Specializing Organelle Node {node_idx}: 'organelle_op_{op_id}_{op_name}']")

        param_map = agent.graph.named_parameters_map()
        node_params = [
            param for name, param in param_map.items()
            if f"node_{node_idx}_" in name
        ]

        optimizer_op = optim.Adam(node_params, lr=0.01)

        for step in range(1, 401):
            x_batch = (torch.rand(64, embed_dim, device=device) - 0.5) * 3.0
            y_batch = env.apply_op(op_id, x_batch)

            w_up = param_map[f"node_{node_idx}_organelle_op_{op_id}_{op_name}_w_up"]
            b_up = param_map[f"node_{node_idx}_organelle_op_{op_id}_{op_name}_b_up"]
            w_down = param_map[f"node_{node_idx}_organelle_op_{op_id}_{op_name}_w_down"]
            b_down = param_map[f"node_{node_idx}_organelle_op_{op_id}_{op_name}_b_down"]

            h = torch.matmul(x_batch, w_up.t()) + b_up
            h_act = nn.functional.gelu(h)
            pred = torch.matmul(h_act, w_down.t()) + b_down

            loss = nn.functional.mse_loss(pred, y_batch)

            optimizer_op.zero_grad()
            loss.backward()
            optimizer_op.step()

            if step % 100 == 0 or step == 400:
                with torch.no_grad():
                    diff = torch.abs(pred - y_batch)
                    acc = (diff < 0.20).float().mean().item() * 100.0
                print(f"    Step {step:03d} | Loss: {loss.item():.6f} | Primitive Accuracy: {acc:.2f}%")

        with torch.no_grad():
            x_val = (torch.rand(1000, embed_dim, device=device) - 0.5) * 3.0
            y_val = env.apply_op(op_id, x_val)
            h_val = torch.matmul(x_val, w_up.t()) + b_up
            pred_val = torch.matmul(nn.functional.gelu(h_val), w_down.t()) + b_down
            final_loss = nn.functional.mse_loss(pred_val, y_val).item()
            final_acc = (torch.abs(pred_val - y_val) < 0.20).float().mean().item() * 100.0
            op_convergence_metrics[op_id] = {"loss": final_loss, "acc": final_acc}

        print(f"  --> Organelle {op_id} ('{op_name}') Converged: Loss = {final_loss:.6f}, Acc = {final_acc:.2f}%")

        # Apply Susumu Ohno Methylation Lock (mu = 1.0, grad = False, gate = 1.0 lossless)
        agent.lock_node(node_idx, 1.0)
        print(f"  🔒 METHYLATION LOCK APPLIED to Organelle Node {node_idx} (mu = 1.0, gate = 1.0 lossless)")

    phase1_duration = time.time() - phase1_start
    print(f"\n[PHASE 1 COMPLETE] Duration: {phase1_duration:.2f}s | All 4 primitive organelles specialized & locked.")

    # =========================================================================
    # PHASE 2: TRAIN INSTRUCTION POINTER & PASSPORT ADDRESSING
    # =========================================================================
    print("\n" + "=" * 85)
    print("PHASE 2: END-TO-END CHAINED DEDUCTION WITH STEP-WISE INSTRUCTION POINTER")
    print("=" * 85)

    trainable_params = []
    param_map = agent.graph.named_parameters_map()
    for name, param in param_map.items():
        if param.requires_grad:
            trainable_params.append(param)
            print(f"  • Trainable Parameter : {name:25s} | shape {list(param.shape)}")

    base_lr = 0.02
    optimizer = optim.AdamW(trainable_params, lr=base_lr, weight_decay=1e-4)

    total_steps = 800
    batch_size = 64
    running_fe_mean = 0.50
    running_fe_var = 0.10
    step_losses = []

    print("\n  [Starting Phase 2 Optimization with Step-Wise Instruction Pointer]")
    phase2_start = time.time()

    for step in range(1, total_steps + 1):
        chain_len = random.choice([2, 3])
        batch = env.sample_batch(batch_size=batch_size, chain_length=chain_len)
        x_sensory = batch["sensory"]
        y_target = batch["target"]
        ctx_seq = batch["context_sequence"] # [B, chain_len, dim]
        op_first = batch["op_first_embed"]

        optimizer.zero_grad()

        agent.graph.reset_state()
        thinking_steps = chain_len
        out = agent.graph.forward(x_sensory, ctx_seq, op_first, thinking_steps)

        loss = nn.functional.mse_loss(out, y_target)
        loss.backward()

        fe_val = float(loss.item())
        step_losses.append(fe_val)

        # Dynamic Allostatic Routing LR Modulation
        fe_diff = fe_val - running_fe_mean
        running_fe_mean = 0.95 * running_fe_mean + 0.05 * fe_val
        running_fe_var = 0.95 * running_fe_var + 0.05 * (fe_diff ** 2)
        std_fe = math.sqrt(max(1e-6, running_fe_var))
        eta_scale = 1.0 + abs(fe_val - running_fe_mean) / (std_fe + 1e-5)

        with torch.no_grad():
            for p_name in ["w_route", "organelle_signatures", "w_query_step", "w_readout_ctx_proj", "w_init_route"]:
                if p_name in param_map and param_map[p_name].grad is not None:
                    param_map[p_name].grad.mul_(min(3.0, eta_scale))

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                diff = torch.abs(out - y_target)
                acc_metric = (diff < 0.20).float().mean().item() * 100.0
            print(
                f"  Step {step:03d} | Loss: {fe_val:.6f} | Acc: {acc_metric:.1f}% | "
                f"Thinking Steps: {thinking_steps} | Eta Scale: {eta_scale:.2f}"
            )

    phase2_duration = time.time() - phase2_start
    total_tokens = total_steps * batch_size
    throughput = total_tokens / max(phase2_duration, 1e-5)
    print(f"\n[PHASE 2 COMPLETE] Duration: {phase2_duration:.2f}s | Throughput: {throughput:.1f} tok/s")

    # =========================================================================
    # STEP 3: STEP-BY-STEP ADAPTIVE ROUTING FIDELITY & ADDRESSING AUDIT
    # =========================================================================
    print("\n" + "=" * 85)
    print("STEP 3: STEP-BY-STEP ADAPTIVE ROUTING FIDELITY & INSTRUCTION POINTER AUDIT")
    print("=" * 85)

    step_routing_accs = {0: [], 1: [], 2: []}
    terminal_prob_list = []

    with torch.no_grad():
        for eval_len in [2, 3]:
            sample_batch = env.sample_batch(batch_size=30, chain_length=eval_len)
            chain_ops = sample_batch["chain"]

            agent.graph.reset_state()
            _, rw, iw, routings = agent.graph.forward_with_diagnostics(
                sample_batch["sensory"],
                sample_batch["context_sequence"],
                sample_batch["op_first_embed"],
                eval_len
            )

            # iw: [B, K] -> Step 0 injection
            target_0 = organelle_indices[chain_ops[0]]
            p_step_0 = float(iw.mean(dim=0)[target_0].item())
            step_routing_accs[0].append(p_step_0)

            # Step 1 routing: routings is [B, steps, K_src, K_tgt]
            target_1 = organelle_indices[chain_ops[1]]
            p_step_1 = float(routings[:, 1, target_0, target_1].mean().item())
            step_routing_accs[1].append(p_step_1)

            if eval_len == 3:
                target_2 = organelle_indices[chain_ops[2]]
                p_step_2 = float(routings[:, 2, target_1, target_2].mean().item())
                step_routing_accs[2].append(p_step_2)

            target_last = organelle_indices[chain_ops[-1]]
            p_term = float(rw.mean(dim=0)[target_last].item())
            terminal_prob_list.append(p_term)

            print(
                f"  • Chain: {[op_names[c] for c in chain_ops]} (Len {eval_len})\n"
                f"    - Step 0 (Sensory -> Node {target_0} [{op_names[chain_ops[0]]}]): P = {p_step_0 * 100.0:.2f}%\n"
                f"    - Step 1 (Node {target_0} -> Node {target_1} [{op_names[chain_ops[1]]}]): P = {p_step_1 * 100.0:.2f}%\n"
                f"    - Terminal Readout (-> Node {target_last} [{op_names[chain_ops[-1]]}]): P = {p_term * 100.0:.2f}%"
            )

    mean_p_step0 = float(np.mean(step_routing_accs[0])) * 100.0
    mean_p_step1 = float(np.mean(step_routing_accs[1])) * 100.0
    mean_p_step2 = float(np.mean(step_routing_accs[2])) * 100.0 if len(step_routing_accs[2]) > 0 else 0.0
    mean_p_term = float(np.mean(terminal_prob_list)) * 100.0

    print(f"\n  Average Step 0 Injection Fidelity  : {mean_p_step0:.2f}% (Target >= 75.0%)")
    print(f"  Average Step 1 Inter-Organelle Route: {mean_p_step1:.2f}% (Target >= 75.0%)")
    if mean_p_step2 > 0:
        print(f"  Average Step 2 Inter-Organelle Route: {mean_p_step2:.2f}% (Target >= 75.0%)")
    print(f"  Average Terminal Readout Fidelity  : {mean_p_term:.2f}% (Target >= 75.0%)")

    # =========================================================================
    # STEP 4: SYSTEMIC MULTI-STEP CHAINED DEDUCTION EVALUATION
    # =========================================================================
    print("\n" + "=" * 85)
    print("STEP 4: SYSTEMIC MULTI-STEP DEDUCTION REASONING BENCHMARK")
    print("=" * 85)

    eval_lengths = [2, 3, 4, 5]
    eval_results = {}

    for length in eval_lengths:
        accuracies = []
        losses = []
        eval_batches = 40
        for _ in range(eval_batches):
            with torch.no_grad():
                batch = env.sample_batch(batch_size=batch_size, chain_length=length)
                agent.graph.reset_state()
                thinking_steps = length
                out = agent.graph.forward(
                    batch["sensory"],
                    batch["context_sequence"],
                    batch["op_first_embed"],
                    thinking_steps
                )

                eval_loss = nn.functional.mse_loss(out, batch["target"]).item()
                diff = torch.abs(out - batch["target"])
                acc = (diff < 0.20).float().mean().item() * 100.0
                accuracies.append(acc)
                losses.append(eval_loss)

        mean_acc = float(np.mean(accuracies))
        mean_loss = float(np.mean(losses))
        eval_results[length] = {"acc": mean_acc, "loss": mean_loss}
        regime = "IN-DIST" if length <= 3 else "OUT-OF-DIST OOD"
        print(f"  • Chain Length {length} ({regime:15s}): Deduction Acc = {mean_acc:5.1f}% | Loss = {mean_loss:.4f}")

    in_dist_acc = (eval_results[2]["acc"] + eval_results[3]["acc"]) / 2.0
    ood_acc = (eval_results[4]["acc"] + eval_results[5]["acc"]) / 2.0

    # =========================================================================
    # STEP 5: FINAL SYNTHESIS & KEP RULE #2 VERDICT
    # =========================================================================
    print("\n" + "=" * 85)
    print("STEP 5: FINAL SYNTHESIS & KEP VERDICT AUDIT")
    print("=" * 85)

    final_loss = float(np.mean(step_losses[-20:]))
    is_positive = (in_dist_acc >= 75.0 and ood_acc >= 60.0)
    verdict_str = "🟢 POSITIVE" if is_positive else "🔴 REJECTED"

    print("=" * 85)
    print(f"EXP-330 VERDICT: {verdict_str}")
    print(f"  • Final Stream Loss                   : {final_loss:.6f}")
    print(f"  • In-Distribution Deduction Acc (2-3) : {in_dist_acc:.2f}% (Target >= 75.0%)")
    print(f"  • Out-of-Distribution Deduction Acc (4-5): {ood_acc:.2f}% (Target >= 60.0%)")
    print("  • Linear Baseline Threshold           : 16.47%")
    print(f"  • Acceleration Gain over Baseline    : {in_dist_acc / 16.47:.2f}x Accuracy Boost")
    print(f"  • Step 0 Address Fidelity             : {mean_p_step0:.2f}%")
    print(f"  • Step 1 Inter-Organelle Fidelity     : {mean_p_step1:.2f}%")
    print(f"  • Tensor Core Throughput              : {throughput:.1f} tok/s")
    print("=" * 85)


if __name__ == "__main__":
    run_exp_330()
