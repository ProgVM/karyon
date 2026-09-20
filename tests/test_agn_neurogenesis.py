import unittest
import torch
from karyon_agent import CoREAgent


class TestAGNNativeNeurogenesis(unittest.TestCase):
    def setUp(self):
        self.dim = 128
        self.batch_size = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = CoREAgent(embed_dim=self.dim, device=self.device, use_graph=True).to(self.device)

    def test_native_zero_identity_at_birth(self):
        """Verify Net2Net zero-delta identity on native C++20 DynamicMorphicGraph at birth."""
        # Setup core nodes
        self.agent.add_node(name="core_in", op_type="LinearAccumulator", is_core=True, initial_alpha=1.0)
        self.agent.add_node(name="core_out", op_type="LinearAccumulator", is_core=True, initial_alpha=1.0)

        x_sensory = torch.randn(self.batch_size, self.dim, device=self.device)
        y_before = self.agent(x_sensory, thinking_steps=3).clone()

        # Sprout new candidate operators with alpha_epi = 0.0
        self.agent.add_node(name="sprout_bilinear", op_type="BilinearMultiplicative", is_core=False, initial_alpha=0.0)
        self.agent.add_node(name="sprout_attractor", op_type="SaturatedAttractor", is_core=False, initial_alpha=0.0)

        y_after = self.agent(x_sensory, thinking_steps=3)

        delta = torch.max(torch.abs(y_after - y_before)).item()
        self.assertAlmostEqual(delta, 0.0, places=5, msg=f"Net2Net zero-shock violated! Delta = {delta}")
        print(f"✅ [Test Native Net2Net] Sprouted 2 nodes into C++20 graph with zero-delta output difference: {delta:.8f}")


if __name__ == "__main__":
    unittest.main()
