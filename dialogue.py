# dialogue.py
"""
===============================================================================
KARYON CLOSED-LOOP INTERACTIVE DIALOGUE RUNTIME (v16.5 MASTER)
Real-time Social Active Inference Session with Somatic Feedback,
Multi-Timescale State-Space Duality, Universal CPU/CUDA Execution,
and Volitional Episodic Fact Recall.
===============================================================================
"""

import sys
import types
import os
import struct
import json
import importlib
import torch
import torch.nn.functional as F

# Dynamo Hotfix for Python 3.12 environments
class DummyDynamoModule(types.ModuleType):
    def __getattr__(self, name):
        if name == "decorators":
            return decorators_mod
        if name == "disable":
            return _disable
        if name == "is_compiling":
            return lambda *args, **kwargs: False
        return lambda *args, **kwargs: None

def _disable(fn=None, *args, **kwargs):
    if fn is None or not callable(fn):
        return lambda *a, **kw: None
    return fn

decorators_mod = types.ModuleType("torch._dynamo.decorators")
class _DimRange:
    pass
decorators_mod._DimRange = _DimRange

dynamo_mod = DummyDynamoModule("torch._dynamo")
dynamo_mod.decorators = decorators_mod
dynamo_mod.disable = _disable

sys.modules["torch._dynamo"] = dynamo_mod
sys.modules["torch._dynamo.decorators"] = decorators_mod
torch._dynamo = dynamo_mod

import karyon_config, karyon_core, karyon_agent, karyon_checkpoint, karyon_logger
importlib.reload(karyon_agent)

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory
from karyon_checkpoint import load_karyon, save_karyon
from karyon_logger import get_logger
from init_priors import initialize_priors

logger = get_logger()

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

kcore_path = "karyon_soul.kcore"

if not os.path.exists(kcore_path):
    logger.warning(f"Container '{kcore_path}' not found! Automatically initializing base model priors...")
    initialize_priors(recreate=True, filepath=kcore_path, device=device_str)

config = CoREConfig()
config.train.batch_size = 1
max_capacity = 1000

if os.path.exists(kcore_path):
    with open(kcore_path, 'rb') as f:
        f.seek(8)
        header_raw = f.read(24)
        _, num_sections, _, _ = struct.unpack('<IIQQ', header_raw)
        sections = []
        for _ in range(num_sections):
            sec_raw = f.read(64)
            s_type, _, offset, size, _ = struct.unpack('<IIQQQ', sec_raw[:32])
            sections.append({"type": s_type, "offset": offset, "size": size})
        sec_manifest = next((s for s in sections if s["type"] == 1), None)
        if sec_manifest:
            f.seek(sec_manifest["offset"])
            manifest = json.loads(f.read(sec_manifest["size"]).decode('utf-8'))
            genome = manifest.get("genome", {})
            if "text_dim" in genome: config.net.text_dim = genome["text_dim"]
            if "text_gen_dim" in genome: config.net.text_gen_dim = genome["text_gen_dim"]
            if "unified_dim" in genome: config.net.unified_dim = genome["unified_dim"]
            if "hidden_dim" in genome: config.net.hidden_dim = genome["hidden_dim"]
            if "latent_dim" in genome: config.net.latent_dim = genome["latent_dim"]
            if "max_capacity" in genome: max_capacity = genome["max_capacity"]

tokenizer = ByteTokenizer(vocab_size=config.net.text_gen_dim)

agent_brain = CoREAgent(config=config, device=device_str).to(device)
hu = HomeostaticUnit(batch_size=1, device=device_str)
episodic_mem = BatchedEpisodicMemory(batch_size=1, memory_dim=config.net.unified_dim, max_capacity=max_capacity, device=device_str)

h_fast, h_slow, epoch, story_idx = load_karyon(agent_brain, episodic_mem, hu, filepath=kcore_path, device=device_str)

logger.info(f"Loaded Karyon Soul (.kcore) | Device: {device_str.upper()} | Genome DNA -> hidden_dim: {agent_brain.hidden_dim}, unified_dim: {agent_brain.unified_dim}")
logger.info("Welcome to Closed-Loop Social Active Inference Session with Karyon-CoRE v16.5!")
logger.info("Type 'exit' to save state and close.")

prev_karyon_representation = None

while True:
    try:
        user_input = input("\nYou (Human Reaction): ")
    except (KeyboardInterrupt, EOFError):
        break

    if user_input.lower() == 'exit':
        save_karyon(agent_brain, episodic_mem, hu, h_fast, h_slow, epoch=epoch, story_idx=story_idx, filepath=kcore_path)
        logger.info(f"Session closed. State persisted into '{kcore_path}'.")
        break
        
    if not user_input.strip():
        continue

    formatted_turn = f"User: {user_input.strip()}\nKaryon:"

    with torch.no_grad():
        user_tokens = agent_brain.encode_text(user_input)
        reaction_fe_list = []
        h_f_tmp = h_fast.clone()
        h_s_tmp = h_slow.clone()
        
        for idx, token_id in enumerate(user_tokens):
            t_emb = agent_brain.pos_embeddings(token_id.unsqueeze(0).unsqueeze(0), start_pos=idx, apply_rf=False)
            s_in = {
                'text': t_emb.squeeze(1), 
                'vision': torch.zeros(1, config.net.vision_dim, device=device), 
                'motor_efference': torch.zeros(1, config.net.action_dim, device=device)
            }
            h_f_tmp, h_s_tmp, _, _, _, fe_reaction, _, w_human, _, _, epistemic_ent, _ = agent_brain(s_in, h_f_tmp, h_s_tmp, hu.state)
            reaction_fe_list.append(fe_reaction.mean().item())
            
        avg_human_surprise = sum(reaction_fe_list) / max(len(reaction_fe_list), 1)
        
        if prev_karyon_representation is not None:
            episodic_mem.write(prev_karyon_representation.detach(), w_human.detach(), 3)

    thought_generator = agent_brain.generate_thought_and_speech(
        formatted_turn,
        m_state=torch.zeros(1, agent_brain.num_heads, agent_brain.head_k, agent_brain.head_v, device=device),
        h_state=h_fast,
        hu=hu,
        episodic_memory=episodic_mem,
        config=config,
        max_generated_tokens=120,
        temperature=0.7,
        top_p=0.90
    )
    
    generated_tokens = []
    
    for event in thought_generator:
        if event["status"] == "speech_start":
            print("Karyon: ", end="", flush=True)
            
        elif event["status"] == "token":
            print(event["text"], end="", flush=True)
            generated_tokens.append(event["token_id"])
            
        elif event["status"] == "exhausted":
            print(event["text"], end="", flush=True)
            h_fast = event.get("h_state", h_fast)
            
        elif event["status"] == "speech_end":
            print()
            h_fast = event.get("h_state", h_fast)
            curiosity, energy, stability, health, na, da = hu.state[0].tolist()
            logger.info(f"Somatic State | Energy: {energy:.3f} | Health: {health:.3f} | Arousal (NA): {na:.3f} | Reward (DA): {da:.3f} | Human Surprise (F_t): {avg_human_surprise:.4f}")

    if len(generated_tokens) > 0:
        with torch.no_grad():
            last_token_t = torch.tensor([[generated_tokens[-1]]], device=device)
            last_emb = agent_brain.pos_embeddings(last_token_t, start_pos=len(generated_tokens), apply_rf=False)
            s_in_last = {'text': last_emb.squeeze(1), 'vision': torch.zeros(1, config.net.vision_dim, device=device), 'motor_efference': torch.zeros(1, config.net.action_dim, device=device)}
            _, _, _, _, _, _, _, prev_karyon_representation, _, _, _, _ = agent_brain(s_in_last, h_fast, h_slow, hu.state)

    episodic_mem.consolidate_and_prune(config.memory.pruning_similarity_threshold, 3)
    print("-" * 80 + "\n")
