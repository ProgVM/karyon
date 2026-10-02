"""
=====================================================================================
EXP-332: DEEP ENDOSCOPIC INFORMATION FLOW & MECHANISTIC PROBE
=====================================================================================
We halt abstract training and run a dedicated endoscopic micro-probe to extract:
  1) Exact step-by-step routing matrix R_step.
  2) Input/Output norms and write gates for each organelle.
  3) CosSim trajectory of each node state relative to mathematical ground truth.
  4) Exact gradient norms across all routing parameters.
  5) Multi-panel visualization experiments/exp_332_endoscopic_trace.png.
=====================================================================================
"""
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from karyon_agent import CoREAgent
from experiments.exp_331_selective_write import StepWiseChainedDeductionEnvironment


def run_endoscopic_probe():
    print("=" * 85)
    print("EXP-332: ENDOSCOPIC INFORMATION FLOW & MECHANISTIC PROBE")
    print("=" * 85)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Diagnostics Substrate Device: {device_str.upper()}")

    # Force strict seed for reproducibility
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    embed_dim = 16
    env = StepWiseChainedDeductionEnvironment(dim=embed_dim, device=device_str)

    # 1. INITIALIZE CO-RE AGENT SUBSTRATE
    agent = CoREAgent(vocab_size=258, embed_dim=embed_dim, device=device_str)
    agent.to(device)

    # Add Slot Memory Organelle (Node 2)
    agent.add_node("slot_mem_0", "SlotMemory", is_core=True, initial_alpha=1.0)

    # Add 4 Specialized NonLinearTransform Organelle Nodes (Nodes 3, 4, 5, 6)
    organelle_indices = []
    op_names = ["roll", "flip", "split_swap", "sign_log"]
    for i, name in enumerate(op_names):
        idx = agent.add_node(f"organelle_op_{i}_{name}", "NonLinearTransform", is_core=True, initial_alpha=1.0)
        organelle_indices.append(idx)

    # ONTOGENETIC SPECIALIZATION (Phase 1)
    print("\n[ONTOGENETIC SPECIALIZATION FOR DIAGNOSTICS]")
    param_map = agent.graph.named_parameters_map()
    for op_id in range(4):
        node_idx = organelle_indices[op_id]
        op_name = op_names[op_id]
        node_params = [p for n, p in param_map.items() if f"node_{node_idx}_" in n]
        optimizer_op = optim.Adam(node_params, lr=0.01)

        for step in range(300):
            x_b = (torch.rand(64, embed_dim, device=device) - 0.5) * 3.0
            y_b = env.apply_op(op_id, x_b)

            w_up = param_map[f"node_{node_idx}_organelle_op_{op_id}_{op_name}_w_up"]
            b_up = param_map[f"node_{node_idx}_organelle_op_{op_id}_{op_name}_b_up"]
            w_down = param_map[f"node_{node_idx}_organelle_op_{op_id}_{op_name}_w_down"]
            b_down = param_map[f"node_{node_idx}_organelle_op_{op_id}_{op_name}_b_down"]

            h = torch.matmul(x_b, w_up.t()) + b_up
            pred = torch.matmul(nn.functional.gelu(h), w_down.t()) + b_down
            loss = nn.functional.mse_loss(pred, y_b)

            optimizer_op.zero_grad()
            loss.backward()
            optimizer_op.step()

        agent.lock_node(node_idx, 1.0)
        print(f"  ✓ Locked Node {node_idx} ('{op_name}')")

    # Train routing parameters slightly to have meaningful gradient traces
    trainable_params = [p for p in param_map.values() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=0.02)

    # Sample a clean, deterministic batch with a specific chain: [roll (0), flip (1), sign_log (3)]
    # Target node sequence: Node 3 -> Node 4 -> Node 6
    chain = [0, 1, 3]
    batch_size = 8
    x_0 = (torch.rand(batch_size, embed_dim, device=device) - 0.5) * 3.0

    y_1_true = env.apply_op(0, x_0)
    y_2_true = env.apply_op(1, y_1_true)
    y_3_true = env.apply_op(3, y_2_true)

    step_tensors = []
    for op_id in chain:
        step_oh = torch.zeros(batch_size, embed_dim, device=device)
        step_oh[:, op_id] = 1.0
        step_tensors.append(step_oh)
    ctx_seq = torch.stack(step_tensors, dim=1)

    op_first = torch.zeros(batch_size, embed_dim, device=device)
    op_first[:, 0] = 1.0

    # 2. RUN DIAGNOSTIC PROBE
    agent.graph.reset_state()
    out, rw, iw, routings, states = agent.graph.forward_with_diagnostics(
        x_0, ctx_seq, op_first, 3
    )

    # 3. COMPUTE GRADIENTS
    loss = nn.functional.mse_loss(out, y_3_true)
    optimizer.zero_grad()
    loss.backward()

    # 4. EXTRACT METRICS FOR ANALYSIS
    print("\n" + "=" * 85)
    print("ANATOMICAL DIAGNOSTIC SNAPSHOT (ONE DETERMINISTIC SAMPLE)")
    print("=" * 85)
    print(f"Chain Ops Sequence: {chain} (roll -> flip -> sign_log)")
    print(f"Target Nodes      : Node 3 -> Node 4 -> Node 6")
    print(f"Output Loss       : {loss.item():.6f}")

    # Extract routing matrices. shape is [B, steps, K_src, K_tgt]
    # mean over batch
    routings_mean = routings.mean(dim=0).cpu().numpy()  # [steps, K_src, K_tgt]
    iw_mean = iw.mean(dim=0).cpu().numpy()  # [K]
    rw_mean = rw.mean(dim=0).cpu().numpy()  # [K]

    # Node states shape [K, steps, B, dim]
    states_cpu = states.cpu()

    # Step 0: Initial Injection
    print("\n--- STEP 0: INITIAL SENSORY INJECTION ---")
    print(f"  • iw (Initial weights per node):")
    for j in range(agent.graph.k_nodes):
        print(f"    Node {j:2d} ({agent.graph.get_topology_manifest()[j].split(':')[0]:18s}): {iw_mean[j]*100.0:6.2f}%")

    # Step 1, 2, 3 details
    node_norms = []
    node_cossims = []
    write_gates_all = []

    for step in range(3):
        print(f"\n--- STEP {step+1} METABOLISM ---")
        # Compute norms and cossims for this step
        # states_cpu is [K, steps, B, dim]
        step_states = states_cpu[:, step]  # [K, B, dim]
        
        # Calculate write gate for this step
        if step == 0:
            w_gate = iw_mean
        else:
            # step_routing_matrix sum over src
            w_gate = routings_mean[step].sum(axis=0)

        write_gates_all.append(w_gate)

        norms = []
        cossims = []
        
        # True state for correlation
        if step == 0:
            y_true = y_1_true
        elif step == 1:
            y_true = y_2_true
        else:
            y_true = y_3_true

        for j in range(agent.graph.k_nodes):
            h_j = step_states[j]  # [B, dim]
            norm = float(h_j.norm(dim=-1).mean().item())
            norms.append(norm)

            # CosSim with true state
            cossim = float(nn.functional.cosine_similarity(h_j, y_true.cpu(), dim=-1).mean().item())
            cossims.append(cossim)

            name = agent.graph.get_topology_manifest()[j].split(":")[0]
            print(
                f"  Node {j:2d} ({name:18s}) | "
                f"||h_j|| = {norm:6.4f} | "
                f"WriteGate = {w_gate[j]:6.4f} | "
                f"CosSim with y_{step+1}* = {cossim:7.4f}"
            )
        
        node_norms.append(norms)
        node_cossims.append(cossims)

        print(f"\n  • R_step {step+1} Routing Matrix (from Row_src to Col_tgt):")
        R = routings_mean[step]
        header = "      " + "".join([f"Node {j:<4d}" for j in range(agent.graph.k_nodes)])
        print(header)
        for src in range(agent.graph.k_nodes):
            row_str = f"Node {src:2d}: "
            for tgt in range(agent.graph.k_nodes):
                row_str += f"{R[src, tgt]*100.0:5.1f}% "
            print(row_str)

    print("\n--- TERMINAL READOUT ATTENTION ---")
    print(f"  • rw (Readout weights per node):")
    for j in range(agent.graph.k_nodes):
        print(f"    Node {j:2d} ({agent.graph.get_topology_manifest()[j].split(':')[0]:18s}): {rw_mean[j]*100.0:6.2f}%")

    # 5. GRADIENT ANALYSIS
    print("\n" + "=" * 85)
    print("GRADIENT FLOW AUDIT")
    print("=" * 85)
    
    grad_norms = {}
    for name, param in param_map.items():
        if param.grad is not None:
            norm_val = float(param.grad.norm().item())
            grad_norms[name] = norm_val
            print(f"  • ||grad({name:20s})|| = {norm_val:.6e}")
        else:
            if param.requires_grad:
                print(f"  • ||grad({name:20s})|| = NONE (requires_grad=True, but no grad!)")

    # 6. PLOT EXPERIMENT TRACE
    print("\n[GENERATING MULTI-PANEL DIAGNOSTIC PLOT]")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EXP-332: Endoscopic Information Flow & Mechanistic Audit", fontsize=16, color="white", weight="bold")
    fig.patch.set_facecolor("#121212")

    for ax in axs.flat:
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")

    # Panel 1: Step 1 Routing Matrix Heatmap
    im1 = axs[0, 0].imshow(routings_mean[1], cmap="hot", aspect="auto")
    axs[0, 0].set_title("Step 1 Routing Matrix (R_1)", color="white", weight="bold")
    axs[0, 0].set_xlabel("Target Node", color="white")
    axs[0, 0].set_ylabel("Source Node", color="white")
    fig.colorbar(im1, ax=axs[0, 0])

    # Panel 2: Node State Norms trajectory
    node_norms_arr = np.array(node_norms)  # [steps, K]
    for j in range(agent.graph.k_nodes):
        name = agent.graph.get_topology_manifest()[j].split(":")[0]
        axs[0, 1].plot([1, 2, 3], node_norms_arr[:, j], marker="o", label=f"Node {j} ({name})")
    axs[0, 1].set_title("Node State Norms (||h_j||) across Steps", color="white", weight="bold")
    axs[0, 1].set_xlabel("Thinking Step", color="white")
    axs[0, 1].set_ylabel("Norm", color="white")
    axs[0, 1].set_xticks([1, 2, 3])
    axs[0, 1].legend(loc="upper right", facecolor="#1e1e1e", edgecolor="white", labelcolor="white", fontsize="small")

    # Panel 3: Correlation with Ground Truth
    node_cossims_arr = np.array(node_cossims)  # [steps, K]
    for j in range(agent.graph.k_nodes):
        name = agent.graph.get_topology_manifest()[j].split(":")[0]
        axs[1, 0].plot([1, 2, 3], node_cossims_arr[:, j], marker="s", linestyle="--", label=f"Node {j} ({name})")
    axs[1, 0].set_title("CosSim with Step Ground Truth", color="white", weight="bold")
    axs[1, 0].set_xlabel("Thinking Step", color="white")
    axs[1, 0].set_ylabel("Cosine Similarity", color="white")
    axs[1, 0].set_xticks([1, 2, 3])
    axs[1, 0].legend(loc="lower left", facecolor="#1e1e1e", edgecolor="white", labelcolor="white", fontsize="small")

    # Panel 4: Gradient Norms Bar Chart
    param_names = list(grad_norms.keys())
    param_vals = list(grad_norms.values())
    axs[1, 1].barh(param_names, param_vals, color="skyblue")
    axs[1, 1].set_xscale("log")
    axs[1, 1].set_title("Gradient Norms (Log Scale)", color="white", weight="bold")
    axs[1, 1].set_xlabel("||grad||", color="white")

    plt.tight_layout()
    plot_path = "experiments/exp_332_endoscopic_trace.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"\n  ✓ Multi-panel diagnostic plot saved to: {plot_path}")


if __name__ == "__main__":
    run_endoscopic_probe()
