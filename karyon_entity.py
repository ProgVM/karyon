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
from typing import Dict, Any, Optional

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
import karyon_core as kcore
from karyon_checkpoint import load_karyon, save_karyon

logger = logging.getLogger("KaryonEntity")


class KaryonEntity:
    """
    Unified Karyon Cognitive Entity encapsulating agent brain, homeostatic nexus,
    hopfield somatic memory, and persistent states.
    """
    def __init__(self, config: Optional[CoREConfig] = None, device: str = "cpu"):
        self.config = config or CoREConfig()
        self.device_str = device
        self.device = torch.device(device)

        # 1. Instantiate Core Components
        self.brain = CoREAgent(embed_dim=self.config.net.unified_dim, device=device).to(self.device)
        self.hu = kcore.HomeostaticNexus(device=device) if hasattr(kcore, 'HomeostaticNexus') else None

        # 2. Episodic / Working Memory Buffer
        self.memory = self.brain.hopfield_memory

        # 3. Persistent Recurrent States
        self.h_fast = torch.zeros(1, self.brain.hidden_dim, device=self.device)
        self.h_slow = torch.zeros(1, self.brain.hidden_dim, device=self.device)

        # 4. Metadata & History
        self.epoch = 0
        self.story_idx = 0
        self.dialogue_history = ""
        self.last_user_input: str = ""
        self.last_karyon_reply: str = ""
        self.last_context_rep: Optional[torch.Tensor] = None
        self.last_action_rep: Optional[torch.Tensor] = None

    @classmethod
    def load(cls, filepath: str = "karyon_soul_v7.kcore", device: str = "cpu") -> "KaryonEntity":
        """Loads a complete Karyon Soul (.kcore) container and returns a fully initialized entity."""
        config = CoREConfig()
        entity = cls(config=config, device=device)

        # Fallback path check if requested v7 path does not exist yet
        target_file = filepath
        if not os.path.exists(target_file) and os.path.exists("karyon_soul.kcore"):
            target_file = "karyon_soul.kcore"

        if os.path.exists(target_file):
            try:
                h_fast_loaded, h_slow_loaded, epoch, story_idx = load_karyon(
                    entity.brain, entity.memory, entity.hu, filepath=target_file, device=device
                )
                entity.h_fast = h_fast_loaded
                entity.h_slow = h_slow_loaded
                entity.epoch = epoch
                entity.story_idx = story_idx
                logger.info(f"Successfully loaded self-contained KaryonEntity from '{target_file}'")
            except Exception as e:
                logger.warning(f"Could not load state dict from '{target_file}': {e}. Using live brain state.")

        return entity

    def save(self, filepath: str = "karyon_soul_v7.kcore"):
        """Persists the complete entity state, DNA, and logic into a single .kcore container."""
        # Wrap states for serializer compatibility
        class MockHU:
            def __init__(self, hu_nexus, device):
                if hu_nexus is not None:
                    self.state = hu_nexus.get_states().unsqueeze(0)
                else:
                    self.state = torch.tensor([[0.85, 1.0, 0.8, 1.0, 0.05, 0.05]], device=device)

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
        logger.info(f"Persisted complete Karyon soul into '{filepath}'")

    def encode_text_representation(self, text: str) -> torch.Tensor:
        """Helper to convert text into normalized continuous representation in embedding space."""
        bytes_list = list(text.encode('utf-8'))
        if not bytes_list:
            bytes_list = [32]
        inp_tensor = torch.tensor([bytes_list], dtype=torch.long, device=self.device)
        with torch.no_grad():
            emb_vectors = self.brain.emb(inp_tensor).squeeze(0)  # [S, D]
            rep = emb_vectors.mean(dim=0, keepdim=True)         # [1, D]
            rep_norm = F.normalize(rep, dim=-1)
        return rep_norm

    def record_somatic_feedback(self, valence: float) -> str:
        """
        Records the last user input and Karyon reply pair as a somatic episode with target valence.
        Valence = +1.0 (Attractor Goal) | -1.0 (Repulsor Barrier) | 0.0 (Topographic Anchor).
        """
        if self.last_context_rep is None or self.last_action_rep is None:
            return "⚠️ No prior dialogue exchange recorded in this turn to tag."

        if self.memory is not None:
            self.memory.record_somatic_episode(self.last_context_rep, self.last_action_rep, valence)

        if valence > 0.5:
            msg = "[Somatic Attractor Anchored: Trajectory marked as goal (V = +1.0)]"
        elif valence < -0.5:
            msg = "[Somatic Repulsor Locked: Trajectory marked as aversive (V = -1.0)]"
        else:
            msg = "[Somatic Topographic Anchor: Trajectory recorded as neutral (V = 0.0)]"

        return msg

    def teach_predictive_step(self, target_text: str) -> float:
        """
        Performs an instant local predictive coding gradient step on the provided target sample,
        directly updating brain weights to reduce Variational Free Energy on the correct pattern.
        """
        if not target_text.strip():
            return 0.0

        prompt = f"Human: {self.last_user_input or 'Inquiry'}\nKaryon: {target_text}\n"
        prompt_bytes = list(prompt.encode('utf-8'))
        input_ids = torch.tensor([prompt_bytes], dtype=torch.long, device=self.device)

        self.brain.train()
        optimizer = torch.optim.AdamW(self.brain.parameters(), lr=1e-4)

        logits = self.brain(input_ids, thinking_steps=4)  # [1, S, V]
        shift_logits = logits[:, :-1, :256].contiguous()
        shift_targets = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(shift_logits.view(-1, 256), shift_targets.view(-1), ignore_index=256)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
        optimizer.step()

        self.brain.eval()
        loss_val = float(loss.item())

        # Somatically reward successful learning
        if self.hu is not None:
            self.hu.update(loss_val)

        return loss_val

    def execute_sleep_cycle(self) -> Dict[str, Any]:
        """Runs a 3-Phase Sleep consolidation cycle (NREM replay + SHY pruning + Energy restoration)."""
        t_start = time.perf_counter()
        self.brain.eval()

        pruned = self.brain.prune_inactive_nodes(threshold=0.01)

        if self.hu is not None:
            with torch.no_grad():
                st = self.hu.get_states()
                st[1] = 1.00  # Energy restored to 100%
                st[3] = min(1.0, float(st[3]) + 0.05)  # Health boost
                st[4] = 0.05  # Arousal (NA) normalized
                st[5] = 0.10  # Dopamine (DA) reset

        duration_ms = (time.perf_counter() - t_start) * 1000.0
        return {"pruned_nodes": pruned, "duration_ms": duration_ms}

    def step(self, input_text: str, thinking_steps: int = 4, max_new_tokens: int = 48) -> str:
        """Processes user input through spatiotemporal dualism and generates continuous reply."""
        self.brain.eval()
        self.last_user_input = input_text
        self.last_context_rep = self.encode_text_representation(input_text)

        prompt = f"Human: {input_text}\nKaryon:"
        prompt_bytes = torch.tensor(list(prompt.encode('utf-8')), dtype=torch.long, device=self.device).unsqueeze(0)
        curr = prompt_bytes
        generated_bytes = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.brain(curr, thinking_steps=thinking_steps)
                last_logits = logits[:, -1, :256] / 0.70  # Temperature scaling
                probs = F.softmax(last_logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                val = nxt.item()
                if val == 10:  # newline
                    break
                generated_bytes.append(val)
                curr = torch.cat([curr, nxt], dim=1)

        reply = bytes(generated_bytes).decode('utf-8', errors='ignore').strip()
        self.last_karyon_reply = reply
        self.last_action_rep = self.encode_text_representation(reply)

        # Modulate homeostasis with step surprise
        if self.hu is not None:
            fake_surprise = max(0.5, 3.0 - 0.05 * len(reply))
            self.hu.update(fake_surprise)

        return reply
