# experiments/exp_274_continuous_morphogenetic_graph_stream.py
"""
===============================================================================
EXP-274: Continuous Online Stream Learning with Dynamic Morphogenetic Graph Assembly
===============================================================================
Hypothesis:
  Empowering the native C++20 DynamicMorphicGraph with autonomous online sleep
  morphogenesis (Net2Net zero-shock operator sprouting + Tononi SHY downscaling)
  will allow the network to dynamically grow its computational capacity during
  stream learning without experiencing catastrophic forgetting or training instability.
===============================================================================
"""
import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from karyon_agent import CoREAgent


def run_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 [EXP-274] Starting Continuous Stream Learning on {device}...")

    torch.manual_seed(42)
    dim = 128
    agent = CoREAgent(vocab_size=258, embed_dim=dim, device=device, use_graph=True).to(device)

    # Core Sensory & Motor Nodes
    agent.add_node(name="node_sensory", op_type="LinearAccumulator", is_core=True, initial_alpha=1.0)
    agent.add_node(name="node_motor", op_type="LinearAccumulator", is_core=True, initial_alpha=1.0)

    # Synthetic non-linear transformation stream
    stream_length = 50
    batch_size = 16
    criterion = nn.MSELoss()

    # Target continuous task: chaotic Mackey-Glass or non-linear recurrent mapping
    target_weights = torch.randn(dim, dim, device=device) * 0.5

    loss_history = []
    initial_loss = None

    t0 = time.perf_counter()

    for step in range(stream_length):
        # Synthetic sensory inputs
        x_sensory = torch.randn(batch_size, dim, device=device)
        y_target = torch.tanh(torch.matmul(x_sensory, target_weights) + torch.sin(x_sensory))

        # Dynamic graph forward (recurrent latent thinking)
        optimizer = torch.optim.AdamW(list(agent.graph.named_parameters_map().values()), lr=0.01)

        optimizer.zero_grad()
        y_pred = agent(x_sensory, thinking_steps=4)
        loss = criterion(y_pred, y_target)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        if step == 0:
            initial_loss = current_loss

        # Sleep & Morphogenesis Trigger every 15 steps
        if (step + 1) % 15 == 0:
            nodes_before = agent.graph.k_nodes
            sleep_info = agent.execute_deep_allostatic_sleep(
                downscaling_factor=0.005,
                sprout_probability=1.0
            )
            nodes_after = agent.graph.k_nodes
            print(f"🌙 [Step {step+1:02d}] Sleep Triggered | Nodes: {nodes_before} -> {nodes_after} | Scaled Params: {int(sleep_info['scaled_params'])}")

    final_loss = loss_history[-1]
    elapsed = time.perf_counter() - t0
    delta_loss = initial_loss - final_loss

    print("\n📊 [EXP-274 Telemetry]")
    print(f"  • Initial Loss : {initial_loss:.4f}")
    print(f"  • Final Loss   : {final_loss:.4f}")
    print(f"  • Loss Delta   : {delta_loss:.4f}")
    print(f"  • Final Nodes  : {agent.graph.k_nodes}")
    print(f"  • Duration     : {elapsed:.2f}s")
    print(f"  • Manifest     : {agent.get_topology_manifest()}")

    # Print KEP-compliant summary metrics for automated parsing
    print(f"METRICS_SUMMARY: final_loss={final_loss:.5f}, delta_loss={delta_loss:.5f}, final_nodes={agent.graph.k_nodes}, elapsed_sec={elapsed:.2f}")


if __name__ == "__main__":
    run_benchmark()
