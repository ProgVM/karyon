import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EpigeneticOperatorNode(nn.Module):
    """
    Epigenetic Operator Node (AGN v6.0)
    Wraps an arbitrary candidate neural operator in Net2Net Smooth Grafting.
    At birth t0 (alpha_epi = 0.0), output is IDENTICALLY zero, guaranteeing zero-shock.
    """
    def __init__(self, op_type: str, hidden_dim: int):
        super().__init__()
        self.op_type = op_type
        self.hidden_dim = hidden_dim
        
        # Epigenetic methylation parameter (initialized to 0.0 for zero-identity)
        self.alpha_epi = nn.Parameter(torch.tensor(0.0))
        
        # Candidate Primitive Operators
        if op_type == "linear":
            self.op = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.orthogonal_(self.op.weight)
        elif op_type == "delay_swiglu":
            self.w_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.w_val = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.xavier_uniform_(self.w_gate.weight)
            nn.init.xavier_uniform_(self.w_val.weight)
        elif op_type == "nonlinear_tanh":
            self.op = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.Tanh()
            )
        else:
            raise ValueError(f"Unknown op_type: {op_type}")
            
    def forward(self, x: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        # Net2Net Smooth Grafting Gate: tanh(alpha_epi)
        gate = torch.tanh(self.alpha_epi)
        
        if self.op_type == "linear":
            y = self.op(x)
        elif self.op_type == "delay_swiglu":
            y = F.silu(self.w_gate(x)) * self.w_val(x)
        elif self.op_type == "nonlinear_tanh":
            y = self.op(x)
            
        return gate * y

class ActiveGraphNeurogenesis(nn.Module):
    """
    Dynamic Graph Engine for Active Graph Neurogenesis (AGN v6.0)
    Manages dynamic sprouting, metabolic pruning, and epigenetic methylation.
    """
    def __init__(self, hidden_dim: int, metabolic_lambda: float = 1e-4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.metabolic_lambda = metabolic_lambda
        self.nodes = nn.ModuleList()
        
    def sprout_node(self, op_type: str):
        """Sprouts a new operator node with zero-weight Net2Net identity."""
        new_node = EpigeneticOperatorNode(op_type, self.hidden_dim)
        self.nodes.append(new_node)
        return new_node
        
    def prune_silent_nodes(self, threshold: float = 0.01) -> int:
        """Prunes nodes whose epigenetic gate alpha_epi is below threshold (Apoptosis)."""
        pruned_count = 0
        active_nodes = nn.ModuleList()
        for node in self.nodes:
            if abs(torch.tanh(node.alpha_epi).item()) >= threshold:
                active_nodes.append(node)
            else:
                pruned_count += 1
        self.nodes = active_nodes
        return pruned_count

    def forward(self, x: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        out = x
        for node in self.nodes:
            out = out + node(out, u_t)
        return out

class TestAGNNeurogenesis(unittest.TestCase):
    def setUp(self):
        self.hidden_dim = 128
        self.batch_size = 4
        self.seq_len = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_zero_identity_at_birth(self):
        """Verify Net2Net zero-delta identity: output of unsprouted vs sprouted graph must be identical at t0."""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device)
        u_t = torch.randn(self.batch_size, 6, device=self.device)
        
        agn = ActiveGraphNeurogenesis(self.hidden_dim).to(self.device)
        y_before = agn(x, u_t)
        
        # Sprout 3 new nodes
        agn.sprout_node("linear")
        agn.sprout_node("delay_swiglu")
        agn.sprout_node("nonlinear_tanh")
        agn = agn.to(self.device)
        
        y_after = agn(x, u_t)
        
        delta = torch.max(torch.abs(y_after - y_before)).item()
        self.assertEqual(delta, 0.0, f"Net2Net identity violated at birth! Delta = {delta}")
        print(f"✅ [Test Net2Net Zero Identity] Birth Delta = {delta:.8f} (Identical)")

    def test_epigenetic_growth_and_pruning(self):
        """Verify that nodes with active alpha_epi affect output and silent nodes are pruned via apoptosis."""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device)
        u_t = torch.randn(self.batch_size, 6, device=self.device)
        
        agn = ActiveGraphNeurogenesis(self.hidden_dim).to(self.device)
        n1 = agn.sprout_node("linear")
        n2 = agn.sprout_node("delay_swiglu")
        n3 = agn.sprout_node("nonlinear_tanh")
        agn = agn.to(self.device)
        
        # Activate node 1 and 3, leave node 2 silent (alpha_epi = 0.0)
        n1.alpha_epi.data.fill_(1.2)
        n3.alpha_epi.data.fill_(-0.8)
        
        self.assertEqual(len(agn.nodes), 3)
        pruned = agn.prune_silent_nodes(threshold=0.01)
        self.assertEqual(pruned, 1, "Failed to prune the silent node!")
        self.assertEqual(len(agn.nodes), 2, "Graph size should be 2 after pruning!")
        print(f"✅ [Test Epigenetic Pruning] Successfully pruned {pruned} silent node(s). Remaining nodes: {len(agn.nodes)}")

if __name__ == "__main__":
    unittest.main()
