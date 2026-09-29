"""
=====================================================================================
EXP-322: BADDELEY MULTI-SLOT WORKING MEMORY & SCRATCHPAD REASONING BENCHMARK
=====================================================================================
Hypothesis:
Equipping DynamicMorphicGraph with Baddeley Multi-Slot Working Memory (SlotMemoryOp)
provides isolated register scratchpads R^{S x D} (S=4 slots) for intermediate composite
states. This eliminates the Single-Vector Bottleneck, enabling sequential step-by-step
deduction without destructive state interference and elevating reasoning accuracy >= 65%.

Scientific Directives Evaluated:
1. Multi-Slot Register Isolation: S=4 working memory registers in R^{B x 4 x D}.
2. Epistemic Routing & Slot Selection: Dynamic Commutation matrix routes intermediate
   transformations to/from scratchpad slots.
3. Chained Rule Deduction Task: Multi-step symbolic transformations (Len 2-3 in-dist, Len 4-5 OOD).
4. Slot Occupancy & Activation Tracking across sequential thinking steps.
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
    """
    Chained Deductive Reasoning Environment (EXP-321/322 Specification):
    Encodes composite symbolic transformations over vector manifolds.
    Length 2-3: In-distribution train & eval.
    Length 4-5: Out-Of-Distribution (OOD) deep composition generalization.
    """
    def __init__(self, dim: int = 16, device: str = "cpu"):
        self.dim = dim
        self.device = torch.device(device)
        self.num_ops = 4

    def apply_op(self, op_id: int, x: torch.Tensor) -> torch.Tensor:
        """Applies a deterministic orthogonal/non-linear transformation."""
        if op_id == 0:
            # Cyclic right shift by 1 + tanh
            return torch.roll(x, shifts=1, dims=-1) * 0.9 + 0.1 * torch.tanh(x)
        elif op_id == 1:
            # Reflection across midpoint + non-linear saturation
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
        # Sample base representation x_0 in [-1.5, 1.5]
        x_0 = (torch.rand(batch_size, self.dim, device=self.device) - 0.5) * 3.0

        chain = [random.randint(0, self.num_ops - 1) for _ in range(chain_length)]

        y_curr = x_0.clone()
        for op_id in chain:
            y_curr = self.apply_op(op_id, y_curr)

        # Context representation: one-hot or normalized embedding of the chain IDs
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

        # Sensory input: [x_0, chain_context]
        sensory_input = x_0 + 0.5 * chain_ctx
        return {
            "sensory": sensory_input,
            "target": y_curr,
            "chain": chain
        }


def run_exp_322():
    print("=" * 85)
    print("EXP-322: BADDELEY MULTI-SLOT WORKING MEMORY & SCRATCHPAD REASONING BENCHMARK")
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

    # 1. INITIALIZE AGENT WITH DYNAMIC MORPHIC GRAPH & SLOT WORKING MEMORY
    print("\n[STEP 1: INITIAL STATE & BADDELEY WORKING MEMORY SEEDING]")
    agent = CoREAgent(
        vocab_size=258,
        embed_dim=embed_dim,
        device=device_str
    )
    agent.to(device)

    # Seed core primitives including Baddeley Multi-Slot Working Memory (SlotMemoryOp)
    agent.add_node("slot_mem_0", "SlotMemory", is_core=True, initial_alpha=1.0)
    agent.add_node("bilinear_0", "BilinearMultiplicative", is_core=True, initial_alpha=1.0)
    agent.add_node("hopfield_0", "ContinuousHopfield", is_core=True, initial_alpha=1.0)
    agent.add_node("delay_0", "ProgrammableDelay", is_core=True, initial_alpha=1.0)
    agent.add_node("tsodyks_0", "TsodyksMarkram", is_core=True, initial_alpha=1.0)

    # Freeze core primitive nodes with Susumu Ohno methylation locks (mu=1.0)
    for i in range(agent.graph.k_nodes):
        agent.lock_node(i, 1.0)

    print(f"  • Initial Dynamic Graph Organelles   : {agent.graph.k_nodes} nodes")
    print("  • Baddeley Slot Working Memory       : Active (S=4 isolated slots)")
    print(f"  • Max Graph Node Capacity            : {agent.max_nodes} slots")

    readout = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, embed_dim)
    ).to(device)

    # 2. CONTINUOUS STREAM LEARNING WITH MOMENTUM PRESERVATION
    base_lr = 0.02
    all_params = list(agent.graph.parameters()) + list(readout.parameters())
    optimizer = optim.AdamW(all_params, lr=base_lr, weight_decay=1e-4)

    total_steps = 400
    batch_size = 64
    running_fe_mean = 0.50
    running_fe_var = 0.10
    morph_events_count = 0

    print("\n[STEP 2: CONTINUOUS DEDUCTIVE STREAM TRAINING WITH SCRATCHPAD MEMORY]")
    start_time = time.time()
    step_losses = []

    for step in range(1, total_steps + 1):
        # Sample curriculum chain lengths (2 to 3 operations)
        chain_len = random.choice([2, 3])
        batch = env.sample_batch(batch_size=batch_size, chain_length=chain_len)
        x_sensory = batch["sensory"]
        y_target = batch["target"]

        optimizer.zero_grad()

        # Dynamic graph thinking forward pass with episodic state reset
        agent.reset_state()
        thinking_steps = 3 + chain_len
        h_graph = agent.graph.forward(x_sensory, thinking_steps)
        out = readout(h_graph)

        loss = nn.functional.mse_loss(out, y_target)
        loss.backward()

        fe_val = float(loss.item())
        step_losses.append(fe_val)

        # Dynamic allostatic routing learning rate scaling
        fe_diff = fe_val - running_fe_mean
        running_fe_mean = 0.95 * running_fe_mean + 0.05 * fe_val
        running_fe_var = 0.95 * running_fe_var + 0.05 * (fe_diff ** 2)
        std_fe = math.sqrt(max(1e-6, running_fe_var))
        eta_scale = 1.0 + abs(fe_val - running_fe_mean) / (std_fe + 1e-5)

        param_map = agent.graph.named_parameters_map()
        with torch.no_grad():
            if "w_route" in param_map and param_map["w_route"].grad is not None:
                param_map["w_route"].grad.mul_(min(3.0, eta_scale))
            if "w_route_ctx" in param_map and param_map["w_route_ctx"].grad is not None:
                param_map["w_route_ctx"].grad.mul_(min(3.0, eta_scale))

        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()

        # Step Somatic Homeostasis & Morphogenesis Trigger
        morph_event = agent.update_somatic_stress_and_morphogenesis(fe_val)
        if morph_event is not None:
            new_idx = agent.graph.k_nodes - 1
            new_param_list = [
                p for name, p in agent.graph.named_parameters_map().items()
                if name.startswith(f"node_{new_idx}_")
            ]
            if new_param_list:
                optimizer.add_param_group({"params": new_param_list, "lr": base_lr})
                all_params.extend(new_param_list)
            morph_events_count += 1

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                diff = torch.abs(out - y_target)
                acc_metric = (diff < 0.20).float().mean().item() * 100.0
                mat_str = f"{agent.compute_organelle_maturity(0):.3f}"
            print(
                f"  Step {step:03d} | Loss: {fe_val:.6f} | Acc: {acc_metric:.1f}% | "
                f"Active Organelles: {agent.graph.k_nodes:02d} | M_k(t): {mat_str} | Morph Events: {morph_events_count}"
            )

    train_duration = time.time() - start_time
    throughput = (total_steps * batch_size) / max(train_duration, 1e-5)
    print(f"\n[STREAM DEDUCTION FINISHED] Duration: {train_duration:.2f}s | Throughput: {throughput:.1f} tok/s")

    # 3. SCRATCHPAD SLOT OCCUPANCY & DYNAMICS AUDIT
    print("\n[STEP 3: BADDELEY WORKING MEMORY SLOT OCCUPANCY & ACCESS AUDIT]")
    with torch.no_grad():
        test_batch = env.sample_batch(batch_size=16, chain_length=3)
        agent.reset_state()
        h_test = agent.graph.forward(test_batch["sensory"], thinking_steps=5)
        _ = readout(h_test)

        # Audit Slot Memory Parameters from parameter map
        param_map = agent.graph.named_parameters_map()
        slot_params = [k for k in param_map.keys() if "slot_mem" in k]
        print(f"  • Baddeley Scratchpad Node Registered Parameters: {len(slot_params)} tensors")
        for p_name in slot_params:
            print(f"    - Parameter: {p_name} | Shape: {list(param_map[p_name].shape)}")

    # 4. SYSTEMIC EVALUATION ON MULTI-STEP DEDUCTION
    print("\n[STEP 4: SYSTEMIC MULTI-STEP DEDUCTION REASONING BENCHMARK]")
    eval_lengths = [2, 3, 4, 5]
    eval_results = {}

    readout.eval()
    for length in eval_lengths:
        accuracies = []
        losses = []
        eval_batches = 30
        for _ in range(eval_batches):
            with torch.no_grad():
                batch = env.sample_batch(batch_size=batch_size, chain_length=length)
                agent.reset_state()
                thinking_steps = 3 + length
                h_graph = agent.graph.forward(batch["sensory"], thinking_steps)
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

    # 5. TONONI SLEEP & RELATIVE NEURAL DARWINISM CONSOLIDATION
    print("\n[STEP 5: 3-PHASE SLEEP REPLAY & RELATIVE NEURAL DARWINISM CONSOLIDATION]")
    pruned_count = agent.prune_relative_darwinism(relative_threshold_factor=0.15)
    print(f"  • Pruned Redundant Organelles        : {pruned_count} nodes")
    print(f"  • Post-Sleep Consolidated Organelles : {agent.graph.k_nodes} nodes")

    # 6. ROUTING MATRIX AUDIT & THROUGHPUT BENCHMARK
    print("\n[STEP 6: COMMUTATION MATRIX R(h_t) & GPU PERFORMANCE AUDIT]")
    with torch.no_grad():
        params = agent.graph.named_parameters_map()
        w_r = params["w_route"][:agent.graph.k_nodes, :agent.graph.k_nodes]
        routing_density = (w_r.abs() > 1e-3).float().mean().item() * 100.0
        active_locks = agent.graph.get_methylation_locks()
        locked_nodes = sum(1 for lock_val in active_locks if lock_val > 0.5)

    print(f"  • Commutation Graph Active Nodes      : {agent.graph.k_nodes} / {agent.max_nodes}")
    print(f"  • Methylation Frozen Nodes (mu=1.0)   : {locked_nodes} nodes")
    print(f"  • Routing Connection Density          : {routing_density:.1f}%")

    final_loss = float(np.mean(step_losses[-20:]))
    print("\n" + "=" * 85)
    print("EXP-322 FINAL SYNTHESIS:")
    print(f"  • Final Stream Loss                   : {final_loss:.6f}")
    print(f"  • In-Distribution Deduction Acc       : {in_dist_acc:.2f}%")
    print(f"  • Out-of-Distribution Deduction Acc   : {ood_acc:.2f}%")
    print(f"  • Tensor Core Throughput              : {throughput:.1f} tok/s")
    print("=" * 85)


if __name__ == "__main__":
    run_exp_322()
