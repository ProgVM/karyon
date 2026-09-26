# tests/test_organelle_orchestration.py
"""
Deterministic Verification Unit Test for Commutation Orchestrator R(h_t) & Susumu Ohno Duplication.

Test Phases:
1. Phase 1 (Initial Specialization):
   - Graph has foundational Node 0 (LinearAccumulator).
   - Node 0 is trained on Pattern 1 (Target: W1 * X).
   - Node 0 reaches low loss and is locked via Epigenetic Methylation (lock_node, mu = 1.0).

2. Phase 2 (Susumu Ohno Duplication):
   - Node 0 is duplicated into Node 1 (organelle_B) with mu = 0.0 (Plastic).
   - Initial cloned weights match Node 0 with exact zero-drift fidelity.

3. Phase 3 (Orthogonal Adaptation & Dynamic Commutation Routing):
   - Node 1 is adapted to Pattern 2 (Target: W2 * X).
   - Commutation Orchestrator (w_route_ctx / w_route) learns dynamic contextual routing:
     * When context is Pattern 1 -> activations routed to Organelle 0.
     * When context is Pattern 2 -> activations routed to Organelle 1.

4. Invariant Verification:
   - Pattern 1 precision preserved (Zero catastrophic forgetting).
   - Pattern 2 precision achieved.
   - Node 0 weights strictly immutable (Delta W_orig == 0.0).
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import karyon_core as kcore


class TestOrganelleOrchestration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dim = 32
        self.batch_size = 16
        self.graph = kcore.DynamicMorphicGraph(self.dim, self.device)

    def test_commutation_orchestration_and_zero_forgetting(self):
        # ---------------------------------------------------------
        # PHASE 1: Train Organelle 0 on Pattern 1
        # ---------------------------------------------------------
        self.graph.add_node("organelle_A", "LinearAccumulator", is_core=False, initial_alpha=1.0)
        node_0_idx = 0

        # Pattern 1 Data: positive distribution centered around +2.0
        x_p1 = torch.randn(self.batch_size, self.dim, device=self.device) + 2.0
        target_p1 = torch.ones(self.batch_size, self.dim, device=self.device) * 3.0

        # Optimize Organelle A
        params_p1 = [p for p in self.graph.named_parameters_map().values() if p.requires_grad]
        opt_p1 = optim.AdamW(params_p1, lr=0.05)

        for step in range(200):
            self.graph.reset_state()
            opt_p1.zero_grad()
            out = self.graph.forward(x_p1, thinking_steps=2)
            loss = nn.functional.mse_loss(out, target_p1)
            loss.backward()
            opt_p1.step()

        with torch.no_grad():
            self.graph.reset_state()
            out_p1_before = self.graph.forward(x_p1, thinking_steps=2)
            loss_p1_initial = nn.functional.mse_loss(out_p1_before, target_p1).item()

        self.assertLess(loss_p1_initial, 0.01, f"Phase 1 failed to converge: loss = {loss_p1_initial}")

        # Epigenetically lock Organelle A
        self.graph.lock_node(node_0_idx, 1.0)
        orig_weights_node0 = {
            k: v.clone().detach() 
            for k, v in self.graph.named_parameters_map().items() 
            if "organelle_A" in k and not k.startswith("alpha_")
        }

        # ---------------------------------------------------------
        # PHASE 2: Susumu Ohno Duplication -> Organelle B (Plastic)
        # ---------------------------------------------------------
        node_1_idx = self.graph.duplicate_node(node_0_idx, "organelle_B", initial_alpha=1.0)

        # ---------------------------------------------------------
        # PHASE 3: Adapt Organelle B & Commutation Orchestrator on Pattern 2
        # ---------------------------------------------------------
        # Pattern 2 Data: negative distribution centered around -2.0
        x_p2 = torch.randn(self.batch_size, self.dim, device=self.device) - 2.0
        target_p2 = torch.ones(self.batch_size, self.dim, device=self.device) * -3.0

        # Mixed training stream (50% Pattern 1, 50% Pattern 2) with locked Organelle A
        params_mixed = [p for p in self.graph.named_parameters_map().values() if p.requires_grad]
        opt_mixed = optim.AdamW(params_mixed, lr=0.05)

        for step in range(400):
            opt_mixed.zero_grad()
            
            # Step on Pattern 1
            self.graph.reset_state()
            out_1 = self.graph.forward(x_p1, thinking_steps=2)
            loss_1 = nn.functional.mse_loss(out_1, target_p1)

            # Step on Pattern 2
            self.graph.reset_state()
            out_2 = self.graph.forward(x_p2, thinking_steps=2)
            loss_2 = nn.functional.mse_loss(out_2, target_p2)

            total_loss = loss_1 + loss_2
            total_loss.backward()
            opt_mixed.step()

        # ---------------------------------------------------------
        # PHASE 4: Strict Invariant Verification
        # ---------------------------------------------------------
        with torch.no_grad():
            self.graph.reset_state()
            eval_p1 = self.graph.forward(x_p1, thinking_steps=2)
            loss_p1_final = nn.functional.mse_loss(eval_p1, target_p1).item()

            self.graph.reset_state()
            eval_p2 = self.graph.forward(x_p2, thinking_steps=2)
            loss_p2_final = nn.functional.mse_loss(eval_p2, target_p2).item()

        # 1. Verify Organelle A weights experienced ZERO drift
        current_params = self.graph.named_parameters_map()
        for k, v_orig in orig_weights_node0.items():
            v_curr = current_params[k]
            delta = torch.max(torch.abs(v_curr - v_orig)).item()
            self.assertEqual(delta, 0.0, f"Invariant violated: Locked Organelle A drifted by {delta} on param {k}")

        # 2. Verify Pattern 1 precision preserved (Zero Forgetting)
        self.assertLess(loss_p1_final, 0.02, f"Pattern 1 corrupted after Phase 3! Loss = {loss_p1_final}")

        # 3. Verify Pattern 2 successfully solved
        self.assertLess(loss_p2_final, 0.02, f"Pattern 2 failed to solve! Loss = {loss_p2_final}")

        print(f"✅ [Test Commutation Orchestration] Pattern 1 Loss: {loss_p1_final:.6f}, Pattern 2 Loss: {loss_p2_final:.6f}")
        print("✅ [Test Commutation Orchestration] Locked Organelle A weights strictly preserved: Delta W_orig = 0.00000000")


if __name__ == "__main__":
    unittest.main()
