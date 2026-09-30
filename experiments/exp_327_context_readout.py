"""
=====================================================================================
EXP-327: CONTEXT-DEPENDENT DYNAMIC READOUT & TARGETED STEP-0 ROUTING BENCHMARK
=====================================================================================
Hypothesis:
  The 16.47% linear bottleneck in previous experiments was caused by two structural bugs:
  1. Static Readout Query (w_readout_query):
     A fixed parameter [dim, 1] cannot distinguish which node holds the terminal answer
     for varying chain lengths, causing readout softmax to collapse into uniform noise (1/7).
     Replacing it with a dynamic context projection:
       q_readout = context_chain @ w_readout_ctx_proj  # [B, 1, dim]
       readout_logits = (node_states @ q_readout.T) / sqrt(dim)
       readout_weights = softmax(readout_logits, dim=1)
     enables dynamic, chain-specific addressing of the terminal organelle.
  2. Degenerate Step-0 Key-Query Routing:
     At step 0, all organelles (except node 0) have state = 0, causing Key-Query routing
     to broadcast the sensory input uniformly across all organelles.
     Routing the input at step 0 via the first instruction embedding:
       routing_step_0 = softmax(op_first_embed @ w_init_route, dim=-1)
     directs sensory data strictly to the intended first operator organelle.

  Together, these mechanisms ensure end-to-end causal computation without signal dissipation,
  breaking the 16.47% barrier to achieve >= 75% in-distribution and >= 60% out-of-distribution accuracy.
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


class DynamicChainedDeductionEnvironment:
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

        chain_one_hots = []
        for op_id in chain:
            oh = torch.zeros(batch_size, self.num_ops, device=self.device)
            oh[:, op_id] = 1.0
            chain_one_hots.append(oh)

        chain_ctx = torch.cat(chain_one_hots, dim=-1)
        if chain_ctx.size(-1) < self.dim:
            pad = torch.zeros(batch_size, self.dim - chain_ctx.size(-1), device=self.device)
            chain_ctx = torch.cat([chain_ctx, pad], dim=-1)
        else:
            chain_ctx = chain_ctx[:, :self.dim]

        # First op embedding vector [B, dim]
        op_first_oh = torch.zeros(batch_size, self.dim, device=self.device)
        op_first_oh[:, chain[0]] = 1.0

        sensory_input = x_0

        return {
            "sensory": sensory_input,
            "target": y_curr,
            "chain": chain,
            "context_chain": chain_ctx,
            "op_first_embed": op_first_oh
        }


def run_exp_327():
    print("=" * 85)
    print("EXP-327: DYNAMIC CONTEXTUAL READOUT & TARGETED STEP-0 ROUTING BENCHMARK")
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
    env = DynamicChainedDeductionEnvironment(dim=embed_dim, device=device_str)

    # 1. INITIALIZE CO-RE AGENT
    print("\n[STEP 1: INITIALIZING CO-RE AGENT SUBSTRATE]")
    agent = CoREAgent(
        vocab_size=258,
        embed_dim=embed_dim,
        device=device_str
    )
    agent.to(device)

    # Add Slot Memory Organelle (Node 0)
    slot_mem_idx = agent.add_node("slot_mem_0", "SlotMemory", is_core=True, initial_alpha=1.0)

    # Add 4 Specialized NonLinearTransform Organelle Nodes (Nodes 1, 2, 3, 4)
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
    # PHASE 1: ONTOGENETIC SPECIALIZATION & SUSUMU OHNO METHYLATION LOCKING
    # =========================================================================
    print("\n" + "=" * 85)
    print("PHASE 1: ONTOGENETIC SPECIALIZATION OF PRIMITIVE ORGANELLES")
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

        # Apply Susumu Ohno Methylation Lock (mu = 1.0, grad = False)
        agent.lock_node(node_idx, 1.0)
        print(f"  🔒 METHYLATION LOCK APPLIED to Organelle Node {node_idx} (mu = 1.0, grad = False)")

    phase1_duration = time.time() - phase1_start
    print(f"\n[PHASE 1 COMPLETE] Duration: {phase1_duration:.2f}s | All 4 primitive organelles specialized & locked.")

    # =========================================================================
    # PHASE 2: CHAINED DEDUCTION VIA DYNAMIC READOUT & INIT-STEP ROUTER
    # =========================================================================
    print("\n" + "=" * 85)
    print("PHASE 2: END-TO-END CHAINED DEDUCTION (DYNAMIC READOUT & TARGETED STEP-0)")
    print("=" * 85)

    readout = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, embed_dim)
    ).to(device)

    trainable_params = []
    param_map = agent.graph.named_parameters_map()
    for name, param in param_map.items():
        if param.requires_grad:
            trainable_params.append(param)
            print(f"  • Trainable Router/Readout Parameter : {name:25s} | shape {list(param.shape)}")

    trainable_params.extend(list(readout.parameters()))

    base_lr = 0.015
    optimizer = optim.AdamW(trainable_params, lr=base_lr, weight_decay=1e-4)

    total_steps = 700
    batch_size = 64
    running_fe_mean = 0.50
    running_fe_var = 0.10
    step_losses = []

    print("\n  [Starting Phase 2 Optimization with Dynamic Contextual Readout & Step-0 Routing]")
    phase2_start = time.time()

    for step in range(1, total_steps + 1):
        chain_len = random.choice([2, 3])
        batch = env.sample_batch(batch_size=batch_size, chain_length=chain_len)
        x_sensory = batch["sensory"]
        y_target = batch["target"]
        ctx_chain = batch["context_chain"]
        op_first = batch["op_first_embed"]

        optimizer.zero_grad()

        agent.graph.reset_state()
        thinking_steps = chain_len
        h_graph = agent.graph.forward(x_sensory, ctx_chain, op_first, thinking_steps)
        out = readout(h_graph)

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
            for p_name in ["w_route", "w_query", "w_key", "w_readout_ctx_proj", "w_init_route"]:
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
    # STEP 3: READOUT & INITIAL STEP SELECTIVITY & ADDRESS PURITY AUDIT
    # =========================================================================
    print("\n" + "=" * 85)
    print("STEP 3: CONTEXTUAL READOUT & STEP-0 ADDRESS PURITY AUDIT")
    print("=" * 85)

    terminal_prob_list = []
    initial_prob_list = []

    with torch.no_grad():
        for eval_len in [2, 3]:
            sample_batch = env.sample_batch(batch_size=10, chain_length=eval_len)
            chain_ops = sample_batch["chain"]
            first_op = chain_ops[0]
            last_op = chain_ops[-1]
            target_first_node = organelle_indices[first_op]
            target_last_node = organelle_indices[last_op]

            agent.graph.reset_state()
            _, rw, iw = agent.graph.forward_with_diagnostics(
                sample_batch["sensory"],
                sample_batch["context_chain"],
                sample_batch["op_first_embed"],
                eval_len
            )

            mean_rw = rw.mean(dim=0).cpu().numpy()
            mean_iw = iw.mean(dim=0).cpu().numpy()

            p_init_correct = float(mean_iw[target_first_node])
            p_term_correct = float(mean_rw[target_last_node])
            initial_prob_list.append(p_init_correct)
            terminal_prob_list.append(p_term_correct)

            print(
                f"  • Chain: {[op_names[c] for c in chain_ops]} (Len {eval_len})\n"
                f"    - Target Init Node ({op_names[first_op]} -> Node {target_first_node}): "
                f"P(init) = {p_init_correct * 100.0:.2f}%\n"
                f"    - Target Term Node ({op_names[last_op]} -> Node {target_last_node}): "
                f"P(term) = {p_term_correct * 100.0:.2f}%\n"
                f"    - Full Readout Weight Distribution : {np.round(mean_rw, 3).tolist()}"
            )

    mean_p_term = float(np.mean(terminal_prob_list)) * 100.0
    mean_p_init = float(np.mean(initial_prob_list)) * 100.0
    print(f"\n  Average Target Init Node Probability    : {mean_p_init:.2f}% (Noise baseline = 14.3%)")
    print(f"  Average Target Readout Node Probability : {mean_p_term:.2f}% (Noise baseline = 14.3%)")

    # =========================================================================
    # STEP 4: SYSTEMIC MULTI-STEP CHAINED DEDUCTION EVALUATION
    # =========================================================================
    print("\n" + "=" * 85)
    print("STEP 4: SYSTEMIC MULTI-STEP DEDUCTION REASONING BENCHMARK")
    print("=" * 85)

    eval_lengths = [2, 3, 4, 5]
    eval_results = {}

    readout.eval()
    for length in eval_lengths:
        accuracies = []
        losses = []
        eval_batches = 40
        for _ in range(eval_batches):
            with torch.no_grad():
                batch = env.sample_batch(batch_size=batch_size, chain_length=length)
                agent.graph.reset_state()
                thinking_steps = length
                h_graph = agent.graph.forward(
                    batch["sensory"],
                    batch["context_chain"],
                    batch["op_first_embed"],
                    thinking_steps
                )
                out = readout(h_graph)

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
    print(f"EXP-327 VERDICT: {verdict_str}")
    print(f"  • Final Stream Loss                   : {final_loss:.6f}")
    print(f"  • In-Distribution Deduction Acc (2-3) : {in_dist_acc:.2f}% (Target >= 75.0%)")
    print(f"  • Out-of-Distribution Deduction Acc (4-5): {ood_acc:.2f}% (Target >= 60.0%)")
    print("  • Linear Baseline Threshold           : 16.47%")
    print(f"  • Acceleration Gain over Baseline    : {in_dist_acc / 16.47:.2f}x Accuracy Boost")
    print(f"  • Tensor Core Throughput              : {throughput:.1f} tok/s")
    print("=" * 85)


if __name__ == "__main__":
    run_exp_327()
