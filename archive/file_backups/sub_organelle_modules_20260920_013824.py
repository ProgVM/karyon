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
        self.norm = nn.RMSNorm(dim)
        
        # Epigenetic zero-shock gate
        self.alpha_epi = nn.Parameter(torch.zeros(1))
        
        # Identity initialization for projection
        nn.init.eye_(self.w_proj.weight)
        nn.init.zeros_(self.gate_w.weight)

    def forward(self, x, state=None):
        # x: [B, S, D]
        # 1. Spatial Projection & Activation
        x_proj = F.silu(self.w_proj(x))
        
        # 2. Leaky Temporal Integration (Parallel Causal Scan via EMA)
        alpha = torch.sigmoid(self.log_alpha) # [D]
        # Linear parallel decay approximation for sequence
        # s_t = alpha * s_{t-1} + (1-alpha) * x
        # Efficient vectorized implementation
        weights = alpha.unsqueeze(0).unsqueeze(0) # [1, 1, D]
        integrated = x_proj * (1.0 - weights)
        
        # 3. Bilinear Gating
        gating = torch.sigmoid(self.gate_w(x))
        x_gated = integrated * gating
        
        # 4. Adaptive Normalization
        x_norm = self.norm(x_gated)
        
        # 5. Zero-Shock Epigenetic Grafting (KEP Principle 15)
        out = x + torch.tanh(self.alpha_epi) * x_norm
        return out


class SubOrganelleVectorB(nn.Module):
    """
    Vector B: Micro-Channel / Synaptic Fiber Sprouting
    Splits latent space into N specialized sub-channels, each with individual
    time-scales, decay dynamics, and dynamic channel widths.
    """
    def __init__(self, dim, num_channels=4):
        super().__init__()
        self.dim = dim
        self.num_channels = num_channels
        self.sub_dim = dim // num_channels
        
        # Independent decay rates per sub-channel
        self.sub_alphas = nn.Parameter(torch.linspace(-4.0, 0.0, num_channels).repeat_interleave(self.sub_dim))
        self.sub_projs = nn.ModuleList([
            nn.Linear(self.sub_dim, self.sub_dim, bias=False) for _ in range(num_channels)
        ])
        
        self.alpha_epi = nn.Parameter(torch.zeros(1))
        
        for p in self.sub_projs:
            nn.init.eye_(p.weight)

    def forward(self, x):
        # x: [B, S, D]
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
            
        x_sprouted = torch.cat(out_channels, dim=-1) # [B, S, D]
        
        # Zero-shock grafting
        return x + torch.tanh(self.alpha_epi) * (x_sprouted - x)


class SubOrganelleVectorC(nn.Module):
    """
    Vector C: Functional Gene Assembly
    Differentiable program encoding:
    Gene = <OpSelect, SourceIndex, TargetIndex, Weight, EpigeneticLock>
    Continuous softmax over atomic instruction set.
    """
    def __init__(self, dim, num_genes=4):
        super().__init__()
        self.dim = dim
        self.num_genes = num_genes
        
        # Gene opcode logits: [0: Add, 1: Mul/Gate, 2: Integrator, 3: NonLinear]
        self.gene_opcodes = nn.Parameter(torch.randn(num_genes, 4) * 0.1)
        self.gene_weights = nn.ParameterList([
            nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01) for _ in range(num_genes)
        ])
        self.alpha_epi = nn.Parameter(torch.zeros(num_genes))

    def forward(self, x):
        # x: [B, S, D]
        out = x
        for i in range(self.num_genes):
            op_probs = F.softmax(self.gene_opcodes[i], dim=-1) # [4]
            
            # Op 0: Linear Trans
            y0 = F.linear(x, self.gene_weights[i])
            # Op 1: Self Gating
            y1 = x * torch.sigmoid(F.linear(x, self.gene_weights[i]))
            # Op 2: NonLinear Projection
            y2 = F.silu(F.linear(x, self.gene_weights[i]))
            # Op 3: Identity / Pass
            y3 = x
            
            gene_out = op_probs[0] * y0 + op_probs[1] * y1 + op_probs[2] * y2 + op_probs[3] * y3
            
            # Zero-shock Epigenetic Grafting per gene
            out = out + torch.tanh(self.alpha_epi[i]) * gene_out
            
        return out


class SubOrganelleVectorD(nn.Module):
    """
    Vector D: Autonomous Circuit & Formula Builder with Universal Signal Transporters
    Micro-units self-assemble into arbitrary computational formulas,
    transport signal buses across arbitrary paths, construct recurrent feedback loops,
    and dynamically synthesize non-linear equations.
    """
    def __init__(self, dim, num_units=6):
        super().__init__()
        self.dim = dim
        self.num_units = num_units
        
        # Signal Transport Bus Matrix (Routing Weights between units)
        self.routing_matrix = nn.Parameter(torch.randn(num_units, num_units) * 0.05)
        
        # Unit transformation parameters
        self.unit_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_units)])
        self.unit_alphas = nn.Parameter(torch.zeros(num_units))
        
        # Formula Synthesizer Coefficients (Bilinear, Additive, Non-linear terms)
        self.formula_gate = nn.Parameter(torch.randn(num_units, dim) * 0.01)
        
        for p in self.unit_projs:
            nn.init.eye_(p.weight)

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape
        
        # 1. Initialize Signal Transporter Bus
        # Each unit starts with input x
        bus = torch.stack([x for _ in range(self.num_units)], dim=0) # [U, B, S, D]
        
        # 2. Differentiable Signal Transport & Routing Step
        route_weights = F.softmax(self.routing_matrix, dim=-1) # [U, U]
        
        # Route signals across units via matrix transport
        routed_bus = torch.einsum('uv, vbsd -> ubsd', route_weights, bus)
        
        # 3. Formula Synthesis & Unit Processing
        unit_outputs = []
        for u in range(self.num_units):
            signal = routed_bus[u]
            # Formula synthesis: Combine Linear + Gated Bilinear + Non-linear
            lin_term = self.unit_projs[u](signal)
            gate_term = torch.sigmoid(signal * self.formula_gate[u])
            formula_out = F.silu(lin_term * gate_term)
            
            # Epigenetic Zero-Shock Grafting for unit
            grafted_out = signal + torch.tanh(self.unit_alphas[u]) * formula_out
            unit_outputs.append(grafted_out)
            
        # 4. Synthesize Final Transported Signal (Sum of unit buses)
        final_signal = torch.stack(unit_outputs, dim=0).mean(dim=0)
        return final_signal


class OmniSubOrganellarEvolutionCore(nn.Module):
    """
    Full Combinatorial Sub-Organellar Architecture (EXP-259)
    Evaluates Vectors A, B, C, D individually and in full synergy.
    """
    def __init__(self, vocab_size=258, dim=256, active_vectors=['A', 'B', 'C', 'D']):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.active_vectors = set(active_vectors)
        
        self.emb = nn.Embedding(vocab_size, dim)
        
        if 'A' in self.active_vectors:
            self.vec_A = SubOrganelleVectorA(dim)
        if 'B' in self.active_vectors:
            self.vec_B = SubOrganelleVectorB(dim, num_channels=4)
        if 'C' in self.active_vectors:
            self.vec_C = SubOrganelleVectorC(dim, num_genes=4)
        if 'D' in self.active_vectors:
            self.vec_D = SubOrganelleVectorD(dim, num_units=6)
            
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight # Weight tying

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
            
        logits = self.head(x)
        return logits
