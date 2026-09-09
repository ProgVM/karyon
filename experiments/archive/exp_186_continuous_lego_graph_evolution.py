import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import time
import gc
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("exp_186")

# 1. PRIMITIVE OPERATOR BRICKS SPECIFICATION (AGN v6.0)
class LinearOp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / math.sqrt(in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        # Epigenetic viability parameter
        self.gamma = nn.Parameter(torch.ones(out_dim, in_dim) * 2.0) # initialized to high viability (sigma(2.0) ~ 0.88)

    def forward(self, x):
        # Apply differentiable soft-masking
        mask = torch.sigmoid(self.gamma)
        effective_weight = self.weight * mask
        return F.linear(x, effective_weight, self.bias)

class NonLinearOp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim * 2, bias=False)
        self.down_proj = nn.Linear(dim, dim, bias=False)
        self.gamma = nn.Parameter(torch.ones(dim) * 2.0)

    def forward(self, x):
        g = self.gate_proj(x)
        x1, x2 = g.chunk(2, dim=-1) # x1: [..., dim], x2: [..., dim]
        hidden = x1 * F.silu(x2)    # hidden: [..., dim]
        out = self.down_proj(hidden) # down_proj from dim to dim
        # Apply soft-masking to output channels
        return out * torch.sigmoid(self.gamma)

class DelayOp(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Allostatic decay and gate parameters
        self.decay_param = nn.Parameter(torch.ones(num_heads, 1, 1) * 1.5) # alpha
        self.gain_param = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5)  # beta

    def forward(self, x, u_t):
        # u_t: [B, 6] (Curiosity, Energy, Stability, Health, Noradrenaline, Dopamine)
        B, S, D = x.size()
        
        # Derive dynamic decay and gain from allostatic state
        na = u_t[:, 4:5].view(B, 1, 1, 1) # Noradrenaline
        da = u_t[:, 5:6].view(B, 1, 1, 1) # Dopamine
        
        # Dynamic decay alpha and gain beta
        alpha = torch.sigmoid(self.decay_param + 0.5 * na) # [B, num_heads, 1, 1]
        beta = torch.sigmoid(self.gain_param + 0.5 * da)   # [B, num_heads, 1, 1]
        
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # [B, h, S, d]
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Zero-Loop Parallel State-Space Duality (SSD) Scan
        # Compute parallel scan matrix
        decay_matrix = alpha.expand(-1, -1, S, S) # [B, h, S, S]
        # Construct lower-triangular decay mask
        indices = torch.arange(S, device=x.device)
        mask = (indices.view(-1, 1) >= indices.view(1, -1)).float() # [S, S]
        
        # Distance matrix for exponential decay
        dist = indices.view(-1, 1) - indices.view(1, -1)
        dist = torch.clamp(dist, min=0)
        
        decay_factors = torch.pow(decay_matrix, dist) * mask.unsqueeze(0).unsqueeze(0) # [B, h, S, S]
        
        # Parallel scan: Y = (Q @ K.T) * decay_factors @ V
        attn = torch.matmul(q, k.transpose(-1, -2)) * beta # [B, h, S, S]
        attn = attn * decay_factors
        
        y = torch.matmul(attn, v) # [B, h, S, d]
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(y)

class GateOp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim)
        
    def forward(self, x, u_t):
        da = u_t[:, 5:6].unsqueeze(1) # [B, 1, 1] Dopamine
        gate = torch.sigmoid(self.gate_proj(x) * (1.0 + 1.5 * da))
        return x * gate

class ResidualOp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x, residual):
        return self.ln(x) + residual

# 2. CONTINUOUS EPIGENETIC EVOLUTIONARY LEGO-GRAPH AGENT
class ContinuousLegoGraphAgent(nn.Module):
    def __init__(self, dim=256, max_bricks=10):
        super().__init__()
        self.dim = dim
        self.max_bricks = max_bricks
        
        # Input/Output projections
        self.embed = nn.Embedding(258, dim)
        self.head = nn.Linear(dim, 258, bias=False)
        
        # Initial core computational pathway (The Seed Graph)
        self.bricks = nn.ModuleList([
            DelayOp(dim),
            NonLinearOp(dim),
            DelayOp(dim),
            NonLinearOp(dim)
        ])
        
        # Epigenetic smooth grafting gates for each brick
        # Initial mature bricks have alpha_epi = 5.0 (fully active, tanh(5.0) ~ 0.9999)
        # Sprouted bricks will start with alpha_epi = 0.0
        self.alpha_epi = nn.ParameterList([
            nn.Parameter(torch.tensor(5.0)) for _ in range(4)
        ])
        
        # GRN Morphogen state
        self.e_sprout = 0.1
        self.e_prune = 0.1
        
    def forward(self, x, u_t):
        return self.forward_sequence(x, u_t)
        
    def forward_sequence(self, x, u_t):
        # x: [B, S]
        # u_t: [B, 6]
        B, S = x.size()
        h = self.embed(x) # [B, S, D]
        
        # Traverse the dynamic LEGO-Graph
        for idx, brick in enumerate(self.bricks):
            gate = torch.tanh(self.alpha_epi[idx])
            
            # Forward pass through the brick depending on its signature
            if isinstance(brick, (DelayOp, GateOp)):
                out = brick(h, u_t)
            else:
                out = brick(h)
                
            # Apply Net2Net Smooth Grafting Gate
            h = h + gate * out
            
        logits = self.head(h)
        return logits

    def sprout_new_brick(self, brick_type="NonLinearOp", device="cpu"):
        """Sprouts a new mathematical operator brick with zero-weight epigenetic gating."""
        if len(self.bricks) >= self.max_bricks:
            logger.warning("Max bricks reached. Skipping sprouting.")
            return False
            
        if brick_type == "NonLinearOp":
            new_brick = NonLinearOp(self.dim)
        elif brick_type == "DelayOp":
            new_brick = DelayOp(self.dim)
        elif brick_type == "GateOp":
            new_brick = GateOp(self.dim)
        else:
            new_brick = NonLinearOp(self.dim)
            
        new_brick = new_brick.to(device)
        # Append to ModuleList and register parameter
        self.bricks.append(new_brick)
        # Strict zero-shock initialization: alpha_epi = 0.0
        self.alpha_epi.append(nn.Parameter(torch.tensor(0.0, device=device)))
        logger.info(f"🌱 [CEE Sprouting] Sprouted new '{brick_type}' brick at index {len(self.bricks)-1} with alpha_epi=0.0")
        return True

    def prune_dead_bricks(self):
        """Prunes inactive or redundant bricks whose epigenetic gate alpha_epi or viability is below threshold."""
        pruned_indices = []
        # Keep at least 2 seed bricks
        for i in range(len(self.bricks) - 1, 1, -1):
            gate_val = torch.tanh(self.alpha_epi[i]).item()
            if abs(gate_val) < 1e-3:
                pruned_indices.append(i)
                
        for idx in pruned_indices:
            logger.info(f"🪓 [CEE Pruning] Pruning inactive brick '{self.bricks[idx].__class__.__name__}' at index {idx} (gate={torch.tanh(self.alpha_epi[idx]).item():.6f})")
            del self.bricks[idx]
            del self.alpha_epi[idx]
            
        return len(pruned_indices) > 0

# 3. EMPIRICAL BENCHMARK PIPELINE
def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running benchmark on device: {device}")
    
    # Initialize agent
    agent = ContinuousLegoGraphAgent(dim=128, max_bricks=8).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Generate synthetic stream data
    B, S = 4, 64
    x_data = torch.randint(0, 256, (B, S), device=device)
    targets = torch.randint(0, 256, (B, S), device=device)
    u_t = torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.2, 0.1]] * B, dtype=torch.float32, device=device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Baseline forward pass
    logits = agent(x_data, u_t)
    loss = criterion(logits.view(-1, 258), targets.view(-1))
    logger.info(f"Baseline Loss: {loss.item():.6f}")
    
    # Step 1: Sprout a new brick during active stream
    agent.sprout_new_brick("NonLinearOp", device=device)
    
    # Verify strict zero-shock function identity
    logits_post_sprout = agent(x_data, u_t)
    loss_post_sprout = criterion(logits_post_sprout.view(-1, 258), targets.view(-1))
    logger.info(f"Post-Sprout Loss: {loss_post_sprout.item():.6f}")
    
    # Mathematical assertion of zero-shock identity
    diff = torch.abs(logits - logits_post_sprout).max().item()
    logger.info(f"Max absolute output difference (Identity Delta): {diff:.12e}")
    assert diff < 1e-5, "Net2Net Smooth Grafting Identity Invariant Violated!"
    logger.info("🟢 SUCCESS: Net2Net Smooth Grafting Identity Invariant Verified!")
    
    # Step 2: Train the sprouted brick
    logger.info("Training the sprouted graph for 5 steps...")
    for step in range(5):
        optimizer.zero_grad()
        logits_step = agent(x_data, u_t)
        loss_step = criterion(logits_step.view(-1, 258), targets.view(-1))
        
        # Add metabolic constraint penalty to force pruning of redundant paths
        metabolic_penalty = 0.0
        for p in agent.alpha_epi:
            metabolic_penalty += 1e-4 * torch.abs(torch.tanh(p))
            
        total_loss = loss_step + metabolic_penalty
        total_loss.backward()
        optimizer.step()
        logger.info(f"  Step {step+1}: Loss = {loss_step.item():.6f} | Alpha_epi (sprouted) = {agent.alpha_epi[-1].item():.4f}")
        
    # Step 3: Prune dead bricks
    # Manually decay the sprouted gate to test pruning trigger
    with torch.no_grad():
        agent.alpha_epi[-1].copy_(torch.tensor(0.0001))
    
    pruned = agent.prune_dead_bricks()
    assert pruned, "Pruning trigger failed to remove inactive brick!"
    logger.info("🟢 SUCCESS: Continuous Epigenetic Evolution Pruning Verified!")

if __name__ == "__main__":
    run_benchmark()
