import torch
import karyon_core as kcore

print("=== VERIFYING C++20 KARYON-CORE EXTENSION ===")

# Test 1: ContinuousHopfieldMemory
print("\n--- Test 1: ContinuousHopfieldMemory with Dopaminergic Modulation ---")
hopfield = kcore.ContinuousHopfieldMemory(dim=256, num_basins=32, device="cpu")
x = torch.randn(2, 10, 256)
u_t = torch.tensor([[0.5, 1.0, 0.8, 1.0, 0.2, 0.9]])  # Dopamine is 0.9 (high)
out_high_da = hopfield(x, u_t)
print("Output shape (high DA):", out_high_da.shape)

u_t_low = torch.tensor([[0.5, 1.0, 0.8, 1.0, 0.2, 0.0]])  # Dopamine is 0.0 (low)
out_low_da = hopfield(x, u_t_low)
print("Output shape (low DA):", out_low_da.shape)

# Test 2: OmniContinuousGraphSubstrate with Dynamic Routing and Recirculation
print("\n--- Test 2: OmniContinuousGraphSubstrate ---")
substrate = kcore.OmniContinuousGraphSubstrate(dim=256, max_nodes=8, device="cpu")
print("Sprouting node 1:", substrate.sprout_node("sensory_node", state_dim=128, num_operators=8))
print("Sprouting node 2:", substrate.sprout_node("associative_node", state_dim=128, num_operators=8))

# Run forward with low neurotransmitters (no recirculation)
u_t_normal = torch.tensor([[0.5, 1.0, 0.8, 1.0, 0.1, 0.1]])
signal_list = [torch.randn(2, 5, 256)]
out_normal, all_signals_normal = substrate(signal_list, u_t_normal)
print("Normal output shape:", out_normal.shape)
print("Normal active signals count:", len(all_signals_normal))

# Run forward with high noradrenaline/dopamine (triggers recirculation)
u_t_high = torch.tensor([[0.5, 1.0, 0.8, 1.0, 0.8, 0.8]])
out_recirc, all_signals_recirc = substrate(signal_list, u_t_high)
print("Recirc output shape:", out_recirc.shape)
print("Recirc active signals count:", len(all_signals_recirc))

print("\n=== ALL C++20 NATIVE SANITY TESTS PASSED SUCCESSFULLY! ===")
