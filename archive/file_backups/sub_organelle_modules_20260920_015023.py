import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SubOrganelleVectorA(nn.Module):
    """
    Vector A: Micro-Operator Nodes
    Primitive atomic operators:
    1. Leaky Temporal Integrator: s_t = alpha * s_{t-1} + (1-alpha) * x_t
    2. Bilinear Gating: a * b
    3. Spatial Projection: W * x
    4. Non-linear Activation: SiLU(x)
    5. Adaptive Normalization: RMSNorm(x)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.log_alpha = nn.Parameter(torch.randn(dim) * 0.1 - 2.0)
        self.w_proj = nn.Linear(dim, dim, bias=False)
        self.gate_w = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        
        # Epigenetic zero-shock gate
        self.alpha_epi = nn.Parameter(torch.zeros(1))
        
        # Identity / safe initialization
        nn.init.orthogonal_(self.w_proj.weight, gain=0.2)
        nn.init.orthogonal_(self.gate_w.weight, gain=0.2)

    def forward(self, x):
        # x: [B, S, D]
        x_proj = F.silu(self.w_proj(x))
        alpha = torch.sigmoid(self.log_alpha).view(1, 1, -1)
        integrated = x_proj * (1.0 - alpha)
        gating = torch.sigmoid(self.gate_w(x))
        x_gated = integrated * gating
        x_norm = self.norm(x_gated)
        
        # Principle 15: Zero-shock grafting
        return x + torch.tanh(self.alpha_epi) * x_norm


class SubOrganelleVectorB(nn.Module):
    """
    Vector B: Micro-Channel / Synaptic Fiber Sprouting
    Splits latent space into N specialized sub-channels with independent time scales.
    """
    def __init__(self, dim, num_channels=4):
        super().__init__()
        self.dim = dim
        self.num_channels = num_channels
        self.sub_dim = dim // num_channels
        
        self.sub_alphas = nn.Parameter(torch.linspace(-4.0, 0.0, num_channels).repeat_interleave(self.sub_dim))
        self.sub_projs = nn.ModuleList([
            nn.Linear(self.sub_dim, self.sub_dim, bias=False) for _ in range(num_channels)
        ])
        self.norm = nn.LayerNorm(dim)
        self.alpha_epi = nn.Parameter(torch.zeros(1))
        
        for p in self.sub_projs:
            nn.init.orthogonal_(p.weight, gain=0.2)

    def forward(self, x):
        B, S, D = x.shape
        x_split = x.view(B, S, self.num_channels, self.sub_dim)
        
        out_channels = []
        alphas = torch.sigmoid(self.sub_alphas).view(1, 1, self.num_channels, self.sub_dim)
        
        for i in range(self.num_channels):
            ch_x = x_split[:, :, i, :]
            ch_proj = self.sub_projs[i](ch_x)
            ch_alpha = alphas[:, :, i, :]
            ch_out = ch_proj * (1.0 - ch_alpha)
            out_channels.append(ch_out)
            
        x_sprouted = self.norm(torch.cat(out_channels, dim=-1))
        return x + torch.tanh(self.alpha_epi) * x_sprouted


class SubOrganelleVectorC(nn.Module):
    """
    Vector C: Functional Gene Assembly
    Differentiable program encoding:
    Gene = <OpSelect, LinearTransform, Gating, Activation, Identity>
    """
    def __init__(self, dim, num_genes=4):
        super().__init__()
        self.dim = dim
        self.num_genes = num_genes
        
        self.gene_opcodes = nn.Parameter(torch.randn(num_genes, 4) * 0.1)
        self.gene_linears = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_genes)])
        self.norm = nn.LayerNorm(dim)
        self.alpha_epi = nn.Parameter(torch.zeros(num_genes))

        for lin in self.gene_linears:
            nn.init.orthogonal_(lin.weight, gain=0.2)

    def forward(self, x):
        out = x
        for i in range(self.num_genes):
            op_probs = F.softmax(self.gene_opcodes[i], dim=-1)
            lin_x = self.gene_linears[i](x)
            
            y0 = lin_x
            y1 = x * torch.sigmoid(lin_x)
            y2 = F.silu(lin_x)
            y3 = x
            
            gene_out = op_probs[0] * y0 + op_probs[1] * y1 + op_probs[2] * y2 + op_probs[3] * y3
            out = out + torch.tanh(self.alpha_epi[i]) * self.norm(gene_out)
            
        return out


class SubOrganelleVectorD(nn.Module):
    """
    Vector D: Autonomous Circuit & Formula Builder with Universal Signal Transporters
    Micro-units assemble dynamic computational formulas, route signal buses,
    and build non-linear equation pipelines.
    """
    def __init__(self, dim, num_units=6):
        super().__init__()
        self.dim = dim
        self.num_units = num_units
        
        # Signal Routing Matrix
        self.routing_matrix = nn.Parameter(torch.randn(num_units, num_units) * 0.05)
        self.unit_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_units)])
        self.formula_gate = nn.Parameter(torch.randn(num_units, dim) * 0.01)
        self.unit_alphas = nn.Parameter(torch.zeros(num_units))
        self.norm = nn.LayerNorm(dim)

        for p in self.unit_projs:
            nn.init.orthogonal_(p.weight, gain=0.2)

    def forward(self, x):
        # x: [B, S, D]
        bus = torch.stack([x for _ in range(self.num_units)], dim=0) # [U, B, S, D]
        route_weights = F.softmax(self.routing_matrix, dim=-1) # [U, U]
        routed_bus = torch.einsum('uv, vbsd -> ubsd', route_weights, bus)
        
        unit_outputs = []
        for u in range(self.num_units):
            signal = routed_bus[u]
            lin_term = self.unit_projs[u](signal)
            gate_term = torch.sigmoid(signal * self.formula_gate[u])
            formula_out = F.silu(lin_term * gate_term)
            
            grafted = signal + torch.tanh(self.unit_alphas[u]) * self.norm(formula_out)
            unit_outputs.append(grafted)
            
        final_signal = torch.stack(unit_outputs, dim=0).mean(dim=0)
        return final_signal


class OmniSubOrganellarEvolutionCore(nn.Module):
    """
    Full Combinatorial Sub-Organellar Architecture (EXP-259)
    """
    def __init__(self, vocab_size=258, dim=256, active_vectors=['A', 'B', 'C', 'D']):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.active_vectors = set(active_vectors)
        
        self.emb = nn.Embedding(vocab_size, dim)
        self.norm = nn.LayerNorm(dim)
        
        if 'A' in self.active_vectors:
            self.vec_A = SubOrganelleVectorA(dim)
        if 'B' in self.active_vectors:
            self.vec_B = SubOrganelleVectorB(dim, num_channels=4)
        if 'C' in self.active_vectors:
            self.vec_C = SubOrganelleVectorC(dim, num_genes=4)
        if 'D' in self.active_vectors:
            self.vec_D = SubOrganelleVectorD(dim, num_units=6)
            
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Proper scaled init
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, tokens):
        x = self.emb(tokens) # [B, S, D]
        
        if 'A' in self.active_vectors:
            x = self.vec_A(x)
        if 'B' in self.active_vectors:
            x = self.vec_B(x)
        if 'C' in self.active_vectors:
            x = self.vec_C(x)
        if 'D' in self.active_vectors:
            x = self.vec_D(x)
            
        x = self.norm(x)
        logits = self.head(x)
        return logits
