class OmniMorphicNode(nn.Module):
    """
    Omni-Morphic Dynamic Cognitive Node (AGN v9.0 / True Open-Ended Morphogenesis).
    An unconstrained, self-governing computational organelle.
    It does NOT have fixed static ports or predetermined roles:
      1. Generates its own dynamic Affinity Query to attend over an arbitrary continuous signal manifold.
      2. Synthesizes its own internal state space & elementary operator bank dynamically.
      3. Broadcasts its output back into the global signal manifold via learnable Net2Net gating.
    """
    def __init__(self, dim: int, state_dim: int = 128, num_operators: int = 8, device: str = 'cpu'):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.device = torch.device(device)
        self.num_operators = num_operators

        # Dynamic internal memory/state buffer
        self.register_buffer("internal_state", torch.zeros(1, state_dim, device=self.device))

        # Learnable Affinity Query vector: allows the node to autonomously select what signals to read
        self.affinity_query = nn.Parameter(torch.randn(dim, device=self.device) * (1.0 / (dim ** 0.5)))

        # Morphic Hyper-Controller: synthesizes internal weights and operator blend
        self.hyper_controller = nn.Sequential(
            nn.Linear(dim, dim // 2, device=self.device),
            nn.SiLU(),
            nn.Linear(dim // 2, (dim * state_dim) + (state_dim * dim) + num_operators + 4, device=self.device)
        )

        # Baseline projections for guaranteed gradient highway
        self.base_in = nn.Linear(dim, state_dim, bias=False, device=self.device)
        self.base_out = nn.Linear(state_dim, dim, bias=False, device=self.device)
        self.state_norm = nn.LayerNorm(state_dim, device=self.device)

    def forward(self, signal_manifold: torch.Tensor, u_t: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        signal_manifold: [batch, num_signals, seq_len, dim] or [batch, num_signals, dim]
        Returns:
            broadcast_signal: [batch, seq_len, dim] (or [batch, dim])
            attention_weights: [batch, num_signals]
        """
        is_4d = (signal_manifold.dim() == 4)
        if is_4d:
            batch, num_signals, seq_len, dim = signal_manifold.size()
        else:
            batch, num_signals, dim = signal_manifold.size()
            seq_len = 1
            signal_manifold = signal_manifold.unsqueeze(2)  # [batch, num_signals, 1, dim]

        # 1. Compute autonomous affinity attention over the signal manifold
        # Key vector: average representation of each signal stream
        keys = signal_manifold.mean(dim=2)  # [batch, num_signals, dim]
        q = self.affinity_query.view(1, 1, dim).expand(batch, 1, dim)  # [batch, 1, dim]
        
        # Dot-product attention scores
        attn_logits = torch.matmul(q, keys.transpose(1, 2)).squeeze(1) * (1.0 / (dim ** 0.5))  # [batch, num_signals]
        attn_weights = F.softmax(attn_logits, dim=-1)  # [batch, num_signals]

        # 2. Synthesize attended dynamic input stream
        # Weighted sum across signals: [batch, 1, num_signals] x [batch, num_signals, seq_len, dim]
        weights_expanded = attn_weights.unsqueeze(-1).unsqueeze(-1)  # [batch, num_signals, 1, 1]
        x_attended = torch.sum(signal_manifold * weights_expanded, dim=1)  # [batch, seq_len, dim]

        # Context vector for parameter generation
        context = x_attended.mean(dim=1)  # [batch, dim]
        if u_t is not None:
            u_flat = u_t.view(batch, -1) if u_t.dim() > 2 else u_t
            if u_flat.size(0) == batch and u_flat.size(-1) <= dim:
                context = context + F.pad(u_flat, (0, dim - u_flat.size(-1)))

        # 3. Dynamic weight and operator synthesis
        morphic_params = self.hyper_controller(context)
        scale_in = (1.0 / (dim ** 0.5))
        scale_out = (1.0 / (self.state_dim ** 0.5))

        ptr = 0
        w_in = morphic_params[:, ptr:ptr + (dim * self.state_dim)].view(batch, dim, self.state_dim) * scale_in
        ptr += dim * self.state_dim
        w_out = morphic_params[:, ptr:ptr + (self.state_dim * dim)].view(batch, self.state_dim, dim) * scale_out
        ptr += self.state_dim * dim

        operator_coeffs = F.softmax(morphic_params[:, ptr:ptr + self.num_operators], dim=-1).view(batch, 1, 1, self.num_operators)
        ptr += self.num_operators

        factors = torch.sigmoid(morphic_params[:, ptr:ptr + 4])
        decay, gain = factors[:, 0:1], factors[:, 1:2]

        # 4. Project into dynamic state space
        projected = self.base_in(x_attended) + torch.matmul(x_attended, w_in)
        projected = self.state_norm(projected)

        # 5. Continuous internal state integration
        state_update = projected.mean(dim=1)
        current_state = self.internal_state.expand(batch, -1)
        new_state = torch.tanh((1.0 - decay) * current_state + gain * state_update)
        if not self.training:
            self.internal_state.copy_(new_state.mean(dim=0, keepdim=True).detach())

        # 6. Apply elementary non-linear operator primitives
        h_state = new_state.unsqueeze(1).expand(-1, seq_len, -1)
        op_0 = h_state                                              # Identity
        op_1 = F.gelu(h_state)                                      # GELU
        op_2 = F.silu(h_state)                                      # SiLU
        op_3 = torch.tanh(h_state)                                  # Tanh
        op_4 = torch.sin(h_state * 3.14159265)                      # Sinusoidal phase oscillation
        op_5 = torch.abs(h_state)                                   # Threshold
        op_6 = h_state * torch.sigmoid(h_state)                      # Self-gating
        op_7 = torch.cos(h_state) * torch.sin(h_state)              # Harmonic resonance

        all_ops = torch.stack([op_0, op_1, op_2, op_3, op_4, op_5, op_6, op_7], dim=-1)
        blended = torch.sum(all_ops * operator_coeffs, dim=-1)

        # 7. Project back to global manifold space
        broadcast_signal = self.base_out(blended) + torch.matmul(blended, w_out)
        if not is_4d:
            broadcast_signal = broadcast_signal.squeeze(1)

        return broadcast_signal, attn_weights


class OmniContinuousGraphSubstrate(nn.Module):
    """
    Omni-Continuous Dynamic Graph Substrate (AGN v9.0).
    Eradicates fixed ports, hardcoded DAG topologies, and rigid layer hierarchies.
    Maintains a continuous open-ended pool of OmniMorphicNodes operating on a
    unified dynamic signal manifold with 100% Zero-Shock Net2Net epigenetic gating.
    """
    def __init__(self, dim: int, max_nodes: int = 16, device: str = 'cpu'):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes
        self.device = torch.device(device)

        self.nodes = nn.ModuleList()
        self.alpha_nodes = nn.ParameterList()
        self.node_names = []

    def sprout_node(self, name: str, state_dim: int = 128, num_operators: int = 8) -> bool:
        """Sprouts an unconstrained omni-morphic node with exact Zero-Shock Net2Net birth identity."""
        if len(self.nodes) >= self.max_nodes or name in self.node_names:
            return False
        node = OmniMorphicNode(dim=self.dim, state_dim=state_dim, num_operators=num_operators, device=str(self.device))
        self.nodes.append(node)
        # Strict Zero-Shock Net2Net identity: alpha = 0.0 -> tanh(0.0) = 0.0
        self.alpha_nodes.append(nn.Parameter(torch.tensor(0.0, device=self.device)))
        self.node_names.append(name)
        logger.info(f"🌌 [Omni-Substrate AGN v9.0] Sprouted node '{name}' (state_dim={state_dim}) with alpha_epi=0.0.")
        return True

    def forward(self, signal_list: List[torch.Tensor], u_t: torch.Tensor = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        signal_list: List of tensors of shape [batch, seq_len, dim] or [batch, dim]
        Returns:
            synthesized_output: [batch, seq_len, dim]
            updated_signal_list: list containing original + newly generated node signals
        """
        if not signal_list:
            raise ValueError("signal_list cannot be empty")

        # Determine reference shape (2D or 3D)
        sample = signal_list[0]
        is_3d = (sample.dim() == 3)
        batch = sample.size(0)
        seq_len = sample.size(1) if is_3d else 1

        # Standardize all signals to [batch, seq_len, dim]
        norm_signals = []
        for s in signal_list:
            if s.dim() == 2:
                s = s.unsqueeze(1).expand(-1, seq_len, -1)
            norm_signals.append(s)

        current_signals = list(norm_signals)

        # Sequentially execute each sprouted node, allowing subsequent nodes to observe earlier nodes
        for idx, (node, alpha) in enumerate(zip(self.nodes, self.alpha_nodes)):
            # Stack all available signals: [batch, num_signals, seq_len, dim]
            manifold_stack = torch.stack(current_signals, dim=1)
            node_out, _ = node(manifold_stack, u_t)
            
            # Continuous tensor gating (Zero PCIe sync stall, differentiable)
            gate = torch.tanh(alpha)
            gated_signal = gate * node_out
            current_signals.append(gated_signal)

        # Synthesize final output: base signal + contribution of all active nodes
        final_signal = current_signals[0]
        for s in current_signals[len(norm_signals):]:
            final_signal = final_signal + s

        if not is_3d:
            final_signal = final_signal.squeeze(1)

        return final_signal, current_signals
