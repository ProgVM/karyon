# karyon_entity.py
"""
KaryonEntity: The Unified, Self-Contained Cognitive and Biophysical Entity.
Encapsulates CoREAgent, HomeostaticNexus, ContinuousHopfieldMemory, and persistent states
into a single, self-executing, modality-agnostic cognitive substrate.
"""

import os
import time
import torch
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, Tuple, Generator

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticNexus, ContinuousHopfieldMemory
from karyon_checkpoint import load_karyon, save_karyon

logger = logging.getLogger("KaryonEntity")

class KaryonEntity:
    def __init__(self, config: CoREConfig = None, device: str = "cpu"):
        self.config = config or CoREConfig()
        self.device_str = device
        self.device = torch.device(device)
        
        # 1. Instantiate Core Components
        self.brain = CoREAgent(embed_dim=self.config.net.unified_dim, device=device).to(self.device)
        self.hu = HomeostaticNexus(device=device) if HomeostaticNexus else None
        
        # 2. Episodic / Working Memory Buffer
        self.memory = ContinuousHopfieldMemory(
            dim=self.config.net.unified_dim, 
            num_basins=32, 
            device=device
        ) if ContinuousHopfieldMemory else None
        
        # 3. Persistent Recurrent States
        self.h_fast = torch.zeros(1, self.brain.hidden_dim, device=self.device)
        self.h_slow = torch.zeros(1, self.brain.hidden_dim, device=self.device)
        
        # 4. Metadata & History
        self.epoch = 0
        self.story_idx = 0
        self.dialogue_history = ""
        self.prev_karyon_representation = None

    @classmethod
    def load(cls, filepath: str = "karyon_soul.kcore", device: str = "cpu") -> "KaryonEntity":
        """Loads a complete Karyon Soul (.kcore) container and returns a fully initialized entity."""
        config = CoREConfig()
        entity = cls(config=config, device=device)
        
        if os.path.exists(filepath):
            try:
                h_fast_loaded, h_slow_loaded, epoch, story_idx = load_karyon(
                    entity.brain, entity.memory, entity.hu, filepath=filepath, device=device
                )
                entity.h_fast = h_fast_loaded
                entity.h_slow = h_slow_loaded
                entity.epoch = epoch
                entity.story_idx = story_idx
                logger.info(f"Successfully loaded self-contained KaryonEntity from '{filepath}'")
            except Exception as e:
                logger.warning(f"Could not load state dict from '{filepath}': {e}. Using live brain state.")
        
        return entity

    def save(self, filepath: str = "karyon_soul.kcore"):
        """Persists the complete entity state, DNA, and logic into a single .kcore container."""
        # Wrap states for serializer
        class MockHU:
            def __init__(self, hu_nexus, device):
                if hu_nexus is not None:
                    self.state = hu_nexus.get_states().unsqueeze(0)
                else:
                    self.state = torch.tensor([[0.8, 1.0, 0.9, 1.0, 0.2, 0.1]], device=device)

        class MockMem:
            def __init__(self, hopfield, device):
                self.keys = torch.zeros(1, 32, 256, device=device)
                self.values = torch.zeros(1, 32, 256, device=device)
                self.pointer = torch.zeros(1, dtype=torch.long, device=device)
                self.size = torch.tensor([32], dtype=torch.long, device=device)

        save_karyon(
            agent=self.brain,
            memory=MockMem(self.memory, self.device),
            hu=MockHU(self.hu, self.device),
            h_fast=self.h_fast,
            h_slow=self.h_slow,
            epoch=self.epoch,
            story_idx=self.story_idx,
            filepath=filepath,
            root_dir="."
        )

    def step(self, input_text: str, thinking_steps: int = 4, max_new_tokens: int = 48) -> str:
        """Processes user input through spatiotemporal dualism and generates continuous reply."""
        self.brain.eval()
        prompt = f"Human: {input_text}\nKaryon:"
        prompt_bytes = torch.tensor(list(prompt.encode('utf-8')), dtype=torch.long, device=self.device).unsqueeze(0)
        curr = prompt_bytes
        generated_bytes = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.brain(curr, thinking_steps=thinking_steps)
                last_logits = logits[:, -1, :] / 0.70
                probs = F.softmax(last_logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                val = nxt.item()
                if val == 10 or val == 257: # newline or EOS
                    break
                generated_bytes.append(val)
                curr = torch.cat([curr, nxt], dim=1)
                
        reply = bytes(generated_bytes).decode('utf-8', errors='ignore').strip()
        return reply
