# karyon_entity.py
"""
KaryonEntity: The Unified, Self-Contained Cognitive and Biophysical Entity.
Encapsulates CoREAgent, HomeostaticUnit, BatchedEpisodicMemory, and persistent states
into a single, self-executing, modality-agnostic cognitive substrate.
"""

import os
import time
import torch
import logging
from typing import Dict, Any, Optional, Tuple, Generator

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory, ByteTokenizer
from karyon_checkpoint import load_karyon, save_karyon

logger = logging.getLogger("KaryonEntity")

class KaryonEntity:
    def __init__(self, config: CoREConfig, device: str = "cpu"):
        self.config = config
        self.device_str = device
        self.device = torch.device(device)
        
        # 1. Instantiate Core Components
        self.brain = CoREAgent(config=config, device=device).to(self.device)
        self.hu = HomeostaticUnit(batch_size=1, device=device)
        
        max_capacity = getattr(config.net, 'max_capacity', 500)
        self.memory = BatchedEpisodicMemory(
            batch_size=1, 
            memory_dim=config.net.unified_dim, 
            max_capacity=max_capacity, 
            device=device
        )
        
        # 2. Persistent Recurrent States
        self.h_fast = torch.zeros(1, self.brain.hidden_dim, device=self.device)
        self.h_slow = torch.zeros(1, self.brain.hidden_dim, device=self.device)
        
        # 3. Metadata & History
        self.epoch = 0
        self.story_idx = 0
        self.dialogue_history = ""
        self.prev_karyon_representation = None

    @classmethod
    def load(cls, filepath: str = "karyon_soul.kcore", device: str = "cpu") -> "KaryonEntity":
        """Loads a complete Karyon Soul (.kcore) container and returns a fully initialized entity."""
        config = CoREConfig()
        
        # Read genome dimensions from container manifest first
        import struct, json, zlib
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    magic = f.read(8)
                    if magic == b"KCORE\x02\x00\x00" or magic.startswith(b"KCORE"):
                        header_raw = f.read(24)
                        header_size, num_sections, total_file_size, flags = struct.unpack('<IIQQ', header_raw)
                        sections = []
                        for _ in range(num_sections):
                            sec_raw = f.read(64)
                            s_type, s_flags, offset, size, align = struct.unpack('<IIQQQ', sec_raw[:32])
                            s_name = sec_raw[32:].rstrip(b'\x00').decode('utf-8', errors='replace')
                            sections.append({'type': s_type, 'flags': s_flags, 'offset': offset, 'size': size, 'name': s_name})
                        
                        sec_manifest = next(s for s in sections if s['type'] == 1)
                        f.seek(sec_manifest['offset'])
                        m_raw = f.read(sec_manifest['size'])
                        if sec_manifest['flags'] & 1:
                            m_raw = zlib.decompress(m_raw)
                        manifest = json.loads(m_raw.decode('utf-8'))
                        genome = manifest.get("genome", {})
                        
                        for k in ["text_dim", "text_gen_dim", "unified_dim", "hidden_dim", "latent_dim", "expand_dim"]:
                            if k in genome:
                                setattr(config.net, k, genome[k])
            except Exception as e:
                logger.warning(f"Failed to pre-parse genome from container manifest: {e}. Using default config.")

        entity = cls(config=config, device=device)
        
        # Restore parameters and states
        h_fast_loaded, h_slow_loaded, epoch, story_idx = load_karyon(
            entity.brain, entity.memory, entity.hu, filepath=filepath, device=device
        )
        
        entity.h_fast = h_fast_loaded
        entity.h_slow = h_slow_loaded
        entity.epoch = epoch
        entity.story_idx = story_idx
        
        logger.info(f"Successfully loaded self-contained KaryonEntity from '{filepath}'")
        return entity

    def save(self, filepath: str = "karyon_soul.kcore"):
        """Persists the complete entity state, DNA, and logic into a single .kcore container."""
        save_karyon(
            self.brain, self.memory, self.hu, self.h_fast, self.h_slow,
            epoch=self.epoch, story_idx=self.story_idx, filepath=filepath
        )
        logger.info(f"Persisted KaryonEntity state into '{filepath}'")

    def interact(self, user_input: str, max_tokens: int = 120, temperature: float = 0.45, top_p: float = 0.90) -> Generator[Dict[str, Any], None, None]:
        """
        Performs a complete closed-loop social active inference turn.
        Yields thought/speech events in real-time.
        """
        is_spontaneous = not bool(user_input.strip())
        
        # 1. Perception & Rest Phase
        if is_spontaneous:
            full_prompt = (self.dialogue_history + " Karyon (Spontaneous Thought):").strip() if self.dialogue_history else "Karyon (Spontaneous Thought):"
        else:
            # Listening actively restores somatic energy (Magistretti 2015)
            with torch.no_grad():
                rest_boost = getattr(self.config.homeo, 'perceptive_rest_recovery', 0.0040) * float(len(user_input))
                self.hu.state[0, 1] = torch.clamp(self.hu.state[0, 1] + rest_boost, 0.0, 1.0)
            
            turn_str = f"User: {user_input.strip()}\nKaryon:"
            if len(self.dialogue_history) + len(turn_str) > 1800:
                self.dialogue_history = self.dialogue_history[-1000:]
            full_prompt = (self.dialogue_history + " " + turn_str).strip() if self.dialogue_history else turn_str

        # 2. Compute Human Surprise (F_t) & Write to Episodic Memory
        avg_human_surprise = 0.0
        if not is_spontaneous:
            with torch.no_grad():
                user_tokens = self.brain.encode_text(user_input)
                reaction_fe_list = []
                h_f_tmp = self.h_fast.clone()
                h_s_tmp = self.h_slow.clone()
                
                for idx, token_id in enumerate(user_tokens):
                    t_emb = self.brain.pos_embeddings(token_id.unsqueeze(0).unsqueeze(0), start_pos=idx, apply_rf=False)
                    s_in = {
                        'text': t_emb.squeeze(1),
                        'vision': torch.zeros(1, self.config.net.vision_dim, device=self.device),
                        'motor_efference': torch.zeros(1, self.config.net.action_dim, device=self.device)
                    }
                    h_f_tmp, h_s_tmp, _, _, _, fe_reaction, _, w_human, _, _, _, _ = self.brain(
                        s_in, h_f_tmp, h_s_tmp, self.hu.state
                    )
                    reaction_fe_list.append(fe_reaction.mean().item())
                
                avg_human_surprise = sum(reaction_fe_list) / max(len(reaction_fe_list), 1)
                
                if self.prev_karyon_representation is not None:
                    self.memory.write(self.prev_karyon_representation.detach().float(), w_human.detach().float(), 3)

        # 3. Generate Speech Output via Thought Generator
        thought_generator = self.brain.generate_thought_and_speech(
            full_prompt,
            m_state=torch.zeros(1, self.brain.num_heads, self.brain.head_k, self.brain.head_v, device=self.device),
            h_state=self.h_fast,
            hu=self.hu,
            episodic_memory=self.memory,
            config=self.config,
            max_generated_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        generated_tokens = []
        generated_chars = []
        
        for event in thought_generator:
            if event["status"] == "token":
                generated_tokens.append(event["token_id"])
                generated_chars.append(event["text"])
                yield event
            elif event["status"] in ("speech_start", "thought_start", "thought_end"):
                yield event
            elif event["status"] in ("exhausted", "speech_end"):
                h_st = event.get("h_state", self.h_fast)
                self.h_fast = h_st.squeeze(1) if (h_st is not None and h_st.dim() == 3) else h_st
                if "m_state" in event:
                    self.h_slow = event["m_state"].view(1, -1)[:, :self.brain.hidden_dim]
                yield event

        response_text = "".join(generated_chars).strip()
        
        # Update internal dialogue history
        if is_spontaneous:
            self.dialogue_history = (self.dialogue_history + f" Karyon (Spontaneous Thought): {response_text}").strip()
        else:
            self.dialogue_history = (self.dialogue_history + f" User: {user_input.strip()}\nKaryon: {response_text}").strip()

        # Update previous representation for next turn's episodic mapping
        if len(generated_tokens) > 0:
            with torch.no_grad():
                last_token_t = torch.tensor([[generated_tokens[-1]]], device=self.device)
                last_emb = self.brain.pos_embeddings(last_token_t, start_pos=len(generated_tokens), apply_rf=False)
                s_in_last = {
                    'text': last_emb.squeeze(1),
                    'vision': torch.zeros(1, self.config.net.vision_dim, device=self.device),
                    'motor_efference': torch.zeros(1, self.config.net.action_dim, device=self.device)
                }
                _, _, _, _, _, _, _, self.prev_karyon_representation, _, _, _, _ = self.brain(
                    s_in_last, self.h_fast, self.h_slow, self.hu.state
                )

        # 4. Awake SWR Micro-Replay during inter-turn pause
        self.brain.execute_wake_swr_micro_replay(self.memory, num_samples=4)

    def self_learn(self, num_sequences: int = 3, seq_len: int = 64) -> Dict[str, Any]:
        """Executes a spontaneous thought and self-learning cycle to update weights and memories."""
        optimizer = torch.optim.AdamW(self.brain.get_all_parameters(), lr=1e-4, weight_decay=0.01)
        criterion_speech = torch.nn.CrossEntropyLoss(ignore_index=256)
        
        with torch.enable_grad():
            results = self.brain.execute_autonomous_self_learning_cycle(
                self.hu, self.memory, optimizer, criterion_speech, 
                num_self_sequences=num_sequences, seq_len=seq_len
            )
        return results

    def sleep(self, num_replay_cycles: int = 5, downscaling_factor: float = 0.03, pruning_percentile: float = 0.05) -> int:
        """Enters deep evolutionary sleep, replaying memories and pruning quiescent synapses."""
        pruned_weights = self.brain.execute_deep_allostatic_sleep(
            episodic_memory=self.memory,
            hu=self.hu,
            num_replay_cycles=num_replay_cycles,
            downscaling_factor=downscaling_factor,
            pruning_percentile=pruning_percentile
        )
        return pruned_weights

    def introspect(self) -> Dict[str, Any]:
        """
        Executes a self-reflective introspective scan using the Predictive Interoceptive Self-Model (PISM).
        Returns predicted self-state, actual somatic state, self-prediction error, and interoceptive fidelity.
        """
        with torch.no_grad():
            u_pred, self_err = self.brain.predictive_self_model(self.h_fast, self.hu.state)
            u_act = self.hu.state
            fidelity = float(1.0 - self_err.mean().item())
            
            names = ["curiosity", "energy", "stability", "health", "noradrenaline", "dopamine"]
            report = {
                "predicted_state": {names[i]: float(u_pred[0, i].item()) for i in range(6)},
                "actual_state": {names[i]: float(u_act[0, i].item()) for i in range(6)},
                "self_prediction_error": float(self_err.mean().item()),
                "interoceptive_fidelity": max(0.0, min(1.0, fidelity))
            }
            return report
