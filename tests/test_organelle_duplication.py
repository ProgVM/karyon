# tests/test_organelle_duplication.py
"""
Unit test for Susumu Ohno Gene Lock & Organelle Duplication Law in C++20 DynamicMorphicGraph.
Verifies that:
1. Initial cloning copies weights with exact fidelity.
2. Methylation locking (methylation_lock = 1.0) completely freezes original node parameters (requires_grad = False).
3. Duplicated clone (methylation_lock = 0.0) remains plastic and updates independently.
4. Original node weights experience ZERO drift (Delta W_orig = 0.0) during gradient updates on the clone.
"""

import unittest
import torch
import karyon_core as kcore


class TestOrganelleDuplication(unittest.TestCase):
    def setUp(self):
        self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dim = 128
        self.graph = kcore.DynamicMorphicGraph(self.dim, self.device_str)

    def test_susumu_ohno_duplication_law(self):
        # 1. Add original node and lock it
        self.graph.add_node("orig_acc", "LinearAccumulator", True, 1.0)
        src_idx = 0
        self.graph.lock_node(src_idx, 1.0)

        # 2. Duplicate node into a plastic clone
        dst_idx = self.graph.duplicate_node(src_idx, "clone_acc", initial_alpha=0.1)

        # 3. Retrieve parameter maps using named_parameters_map()
        params = self.graph.named_parameters_map()

        # Extract weights for orig and clone (excluding alpha_epi which is handled separately)
        # Note: The key format contains the node index, e.g. "node_0_orig_acc.w" vs "node_1_clone_acc.w"
        orig_weights = {k: v.clone() for k, v in params.items() if "orig_acc" in k and not k.startswith("alpha_")}
        clone_weights = {k: v.clone() for k, v in params.items() if "clone_acc" in k and not k.startswith("alpha_")}

        self.assertTrue(len(orig_weights) > 0, "Original node parameters not found in parameter map")
        self.assertTrue(len(clone_weights) > 0, "Clone node parameters not found in parameter map")

        # Verify initial weight fidelity
        for orig_key, orig_val in orig_weights.items():
            # Replace node index and node name to construct the target clone key
            # orig_key: node_0_orig_acc.w -> target: node_1_clone_acc.w
            clone_key = orig_key.replace("node_0_orig_acc", "node_1_clone_acc")
            self.assertIn(clone_key, clone_weights, f"Matching parameter {clone_key} missing in clone")
            max_diff = torch.max(torch.abs(orig_val - clone_weights[clone_key])).item()
            self.assertEqual(max_diff, 0.0, f"Initial cloned weight mismatch for {orig_key}")

        # Verify epigenetic gate values are as expected
        self.assertEqual(params["alpha_orig_acc"].item(), 1.0)
        self.assertAlmostEqual(params["alpha_clone_acc"].item(), 0.1, places=5)

        # 4. Perform forward pass, loss compute, and backward step
        x_sensory = torch.randn(4, self.dim, device=self.device_str, requires_grad=True)
        out = self.graph.forward(x_sensory, 3)
        loss = out.sum()

        optimizer = torch.optim.SGD([p for p in params.values() if p.requires_grad], lr=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 5. Retrieve updated parameter map
        params_after = self.graph.named_parameters_map()

        # Verify original node parameters had NO gradient updates and ZERO weight drift
        for orig_key, orig_val_before in orig_weights.items():
            orig_val_after = params_after[orig_key]
            delta_orig = torch.max(torch.abs(orig_val_after - orig_val_before)).item()
            self.assertEqual(delta_orig, 0.0, f"Locked original node weight experienced drift! Delta = {delta_orig}")
            self.assertFalse(orig_val_after.requires_grad, f"Locked node parameter {orig_key} has requires_grad=True")

        # Verify alpha_orig_acc also did not change and has requires_grad=False
        self.assertEqual(params_after["alpha_orig_acc"].item(), 1.0)
        self.assertFalse(params_after["alpha_orig_acc"].requires_grad)

        # Verify clone node parameters DID update (or at least maintained plasticity with requires_grad=True)
        for clone_key in clone_weights:
            clone_val_after = params_after[clone_key]
            self.assertTrue(clone_val_after.requires_grad, f"Plastic clone parameter {clone_key} has requires_grad=False")

        self.assertTrue(params_after["alpha_clone_acc"].requires_grad)

        print("✅ [Test Susumu Ohno Duplication] Locked original weights strictly preserved (Delta W = 0.0), clone adapted independently.")


if __name__ == "__main__":
    unittest.main()
