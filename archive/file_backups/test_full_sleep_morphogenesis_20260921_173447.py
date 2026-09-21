import unittest
import torch
from karyon_agent import CoREAgent


class TestFullSleepMorphogenesisIntegration(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dim = 128
        self.agent = CoREAgent(embed_dim=self.dim, device=self.device, use_graph=True).to(self.device)
        self.agent.add_node(name="input_core", op_type="LinearAccumulator", is_core=True, initial_alpha=1.0)
        self.agent.add_node(name="output_core", op_type="LinearAccumulator", is_core=True, initial_alpha=1.0)

    def test_e2e_sleep_cycle(self):
        """Verify that deep allostatic sleep runs end-to-end with neurogenesis and synaptic scaling."""
        x_sensory = torch.randn(4, self.dim, device=self.device)

        # Run Sleep Cycle with mandatory node sprouting (sprout_probability=1.0)
        sleep_metrics = self.agent.execute_deep_allostatic_sleep(
            downscaling_factor=0.01,
            sprout_probability=1.0
        )

        self.assertGreater(sleep_metrics["scaled_params"], 0)
        self.assertEqual(sleep_metrics["sprouted"], 1.0)
        self.assertEqual(self.agent.graph.k_nodes, 3)

        # Output check: since alpha_epi was initialized to 0.0, output should be preserved
        out_after = self.agent(x_sensory, thinking_steps=3)
        self.assertFalse(torch.isnan(out_after).any().item(), "NaN detected post-sleep!")
        print(f"✅ [Test Sleep E2E] Sleep completed with dynamic neurogenesis. k_nodes={self.agent.graph.k_nodes}, scaled_params={sleep_metrics['scaled_params']}")


if __name__ == "__main__":
    unittest.main()
