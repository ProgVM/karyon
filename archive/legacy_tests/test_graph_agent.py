
import torch
from karyon_agent import CoREAgent

agent = CoREAgent(vocab_size=258, embed_dim=128, use_graph=True, device="cpu")
agent.add_node("node_0", "LinearAccumulator", is_core=True, initial_alpha=1.0)
agent.add_node("node_1", "BilinearMultiplicative", is_core=False, initial_alpha=0.0)
agent.add_node("node_2", "SaturatedAttractor", is_core=False, initial_alpha=0.5)

print("Nodes in C++ graph:", agent.graph.k_nodes)
print("Topology manifest:", agent.get_topology_manifest())

# Run a forward pass
x = torch.randn(4, 128)
out = agent(x)
print("Forward output shape:", out.shape)
assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"
print("All assertions passed!")
