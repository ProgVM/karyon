import unittest
import torch
import torch.nn as nn
from karyon_config import CoREConfig
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory
from karyon_agent import CoREAgent

class TestFullSleepMorphogenesisIntegration(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = CoREConfig()
        self.agent = CoREAgent(self.config, device=self.device).to(self.device)
        self.hu = HomeostaticUnit(1, self.device)
        self.memory = BatchedEpisodicMemory(1, 64, 256, self.device)

    def test_e2e_sleep_cycle(self):
        """Verify that deep allostatic sleep runs end-to-end without device mismatches, exceptions, or NaN values."""
        eval_inputs = torch.randint(0, 256, (2, 32), device=self.device)
        eval_targets = torch.randint(0, 256, (2, 32), device=self.device)
        criterion = nn.CrossEntropyLoss(ignore_index=256)

        pruned, evolved_agent, is_struct = self.agent.execute_deep_allostatic_sleep(
            episodic_memory=self.memory,
            hu=self.hu,
            num_replay_cycles=2,
            downscaling_factor=0.01,
            pruning_percentile=0.01,
            eval_inputs=eval_inputs,
            eval_targets=eval_targets,
            criterion_speech=criterion
        )

        self.assertGreaterEqual(pruned, 0)
        self.assertIsNotNone(evolved_agent)
        
        # Test agent vitality post-sleep
        out = evolved_agent.forward_sequence(eval_inputs, eval_targets, self.hu, criterion)
        self.assertIsNotNone(out)
        self.assertFalse(torch.isnan(out[0]).any().item(), "NaN loss detected post-sleep!")
        print(f"✅ [Test Sleep E2E] Full Allostatic Sleep completed cleanly. Post-sleep loss = {out[0].item():.4f}")

if __name__ == "__main__":
    unittest.main()
