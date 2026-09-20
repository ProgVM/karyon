"""
Verification Unit Test for DynamicMorphicGraph Checkpoint Serialization/Deserialization
Validates:
1. Instantiation of CoREAgent with DynamicMorphicGraph (C++20).
2. Sprouting heterogeneous nodes (LinearAccumulator, BilinearMultiplicative, SaturatedAttractor).
3. Forward pass consistency.
4. Full .kcore v5.0 container save with SHA-256 integrity and topology manifest.
5. Restoration into a completely fresh CoREAgent instance.
6. Numerical equality of weights and forward activations between original and restored agents.
"""
import os
import torch
import torch.nn as nn
from karyon_agent import CoREAgent
from karyon_checkpoint import save_karyon, load_karyon

class MockHomeostasis:
    def __init__(self, dim=6):
        self.state = torch.tensor([0.8, 0.9, 0.7, 0.95, 0.3, 0.5])

class MockMemory:
    def __init__(self, batch_size=1, dim=128, max_cap=50):
        self.batch_size = batch_size
        self.max_capacity = max_cap
        self.keys = torch.zeros(batch_size, max_cap, dim)
        self.values = torch.zeros(batch_size, max_cap, dim)
        self.pointer = torch.zeros(batch_size, dtype=torch.long)
        self.size = torch.zeros(batch_size, dtype=torch.long)

def test_checkpoint_roundtrip():
    print("=== Step 1: Initialize Original Agent with Evolved Graph ===")
    embed_dim = 128
    agent_orig = CoREAgent(vocab_size=258, embed_dim=embed_dim, use_graph=True, device='cpu')
    
    # Sprout nodes
    agent_orig.add_node("node_core_linear", "LinearAccumulator", is_core=True, initial_alpha=1.0)
    agent_orig.add_node("node_bilinear", "BilinearMultiplicative", is_core=False, initial_alpha=0.25)
    agent_orig.add_node("node_attractor", "SaturatedAttractor", is_core=False, initial_alpha=0.75)

    print(f"Original Graph Nodes: {agent_orig.graph.k_nodes}")
    manifest_orig = agent_orig.get_topology_manifest()
    print(f"Original Topology Manifest: {manifest_orig}")

    # Dummy input
    x_test = torch.randn(2, embed_dim)
    with torch.no_grad():
        out_orig = agent_orig(x_test, thinking_steps=4)
    print(f"Original Forward Output Shape: {out_orig.shape}")

    # Mock components for save_karyon
    hu = MockHomeostasis()
    mem = MockMemory(batch_size=1, dim=embed_dim)
    h_fast = torch.randn(1, embed_dim)
    h_slow = torch.randn(1, embed_dim)

    checkpoint_file = "test_graph_checkpoint.kcore"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print("\n=== Step 2: Save Container to test_graph_checkpoint.kcore ===")
    save_karyon(
        agent=agent_orig,
        memory=mem,
        hu=hu,
        h_fast=h_fast,
        h_slow=h_slow,
        epoch=1,
        story_idx=100,
        filepath=checkpoint_file
    )
    assert os.path.exists(checkpoint_file), "Checkpoint file was not created!"
    file_size_kb = os.path.getsize(checkpoint_file) / 1024
    print(f"Saved .kcore container size: {file_size_kb:.2f} KB")

    print("\n=== Step 3: Instantiate Completely Fresh Agent & Restore ===")
    agent_restored = CoREAgent(vocab_size=258, embed_dim=embed_dim, use_graph=True, device='cpu')
    mem_restored = MockMemory(batch_size=1, dim=embed_dim)
    hu_restored = MockHomeostasis()

    print(f"Restored Agent Nodes BEFORE load: {agent_restored.graph.k_nodes}")
    assert agent_restored.graph.k_nodes == 0, "Fresh agent should have 0 nodes before loading!"

    load_karyon(
        agent=agent_restored,
        memory=mem_restored,
        hu=hu_restored,
        filepath=checkpoint_file,
        device='cpu'
    )

    print(f"Restored Agent Nodes AFTER load: {agent_restored.graph.k_nodes}")
    assert agent_restored.graph.k_nodes == agent_orig.graph.k_nodes, (
        f"Node count mismatch! Expected {agent_orig.graph.k_nodes}, got {agent_restored.graph.k_nodes}"
    )

    manifest_restored = agent_restored.get_topology_manifest()
    print(f"Restored Topology Manifest: {manifest_restored}")
    assert manifest_restored == manifest_orig, "Topology manifest mismatch between original and restored!"

    print("\n=== Step 4: Verify Weight Precision & Numerical Equivalence ===")
    state_orig = agent_orig.get_complete_state_dict()
    state_restored = agent_restored.get_complete_state_dict()

    for k in state_orig.keys():
        assert k in state_restored, f"Parameter '{k}' missing from restored state dict!"
        max_diff = torch.max(torch.abs(state_orig[k] - state_restored[k])).item()
        assert max_diff < 1e-6, f"Weight divergence in '{k}': max diff = {max_diff}"
    print("All weights matched with zero delta (< 1e-6)!")

    with torch.no_grad():
        out_restored = agent_restored(x_test, thinking_steps=4)

    forward_diff = torch.max(torch.abs(out_orig - out_restored)).item()
    print(f"Forward Output Max Absolute Difference: {forward_diff:.8e}")
    assert forward_diff < 1e-6, f"Forward output divergence! Max diff = {forward_diff}"

    # Cleanup
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    print("\n✅ TEST PASSED: Dynamic Morphogenetic Graph roundtrip serialization verified perfectly!")

if __name__ == "__main__":
    test_checkpoint_roundtrip()
