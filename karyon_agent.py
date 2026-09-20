# karyon_agent.py
"""
===============================================================================
KARYON CORE AGENT MASTER WRAPPER v30.0
Python Orchestrator Wrapper for C++20 UniversalMorphicSpace Engine
===============================================================================
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

import karyon_core as kcore
from karyon_config import CoREConfig

class ConfigMock:
    class NetMock:
        text_dim = 256
        text_gen_dim = 256
        vocab_size = 258
    net = NetMock()

class CoREAgent(nn.Module):
    """
    Master Python wrapper over C++20 UniversalMorphicSpace engine.
    Ensures 100% compliance with karyon_checkpoint.py (.kcore v5.0 serialization).
    """
    def __init__(self, vocab_size=258, embed_dim=256, num_cells=2, num_operators=4, device='cpu'):
        super().__init__()
        self.config = ConfigMock()
        self.device = device
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.unified_dim = embed_dim
        self.hidden_dim = embed_dim
        self.latent_dim = 64
        self.action_dim = vocab_size

        # C++20 Engine core instantiation
        self.space = kcore.UniversalMorphicSpace(
            vocab_size, embed_dim, num_cells, num_operators, device
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Direct forward pass returning logits [B, S, V]."""
        return self.space(input_ids)

    def forward_latent(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning latent manifold states [B, S, D]."""
        return self.space.forward_latent(input_ids)

    def get_complete_state_dict(self) -> Dict[str, torch.Tensor]:
        """Exposes C++20 module parameters for .kcore v5.0 container serialization."""
        return self.space.named_parameters_map()

    def load_complete_state_dict(self, state_dict: Dict[str, torch.Tensor], device: str = 'cpu'):
        """Restores parameters into C++20 core engine."""
        param_map = self.space.named_parameters_map()
        with torch.no_grad():
            for name, param in param_map.items():
                if name in state_dict:
                    param.copy_(state_dict[name].to(device))
