# run_evolution.py
import torch
import torch.nn as nn
import os
import struct
import json

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import ByteTokenizer, HomeostaticUnit, BatchedEpisodicMemory
from karyon_checkpoint import load_karyon, save_karyon
from kcore_evolution import KaryonEvolver
from init_priors import initialize_priors

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"[Evolution Driver] Execution context: {device.upper()}")

kcore_path = "karyon_soul.kcore"

# 1. Fault-tolerant container check
if not os.path.exists(kcore_path):
    print(f"[Evolution Driver] Container '{kcore_path}' not found. Initializing base model...")
    initialize_priors(recreate=True, filepath=kcore_path, device=device)

config = CoREConfig()
config.net.text_dim = 128
config.net.unified_dim = 128
config.net.text_gen_dim = 258

# 2. Extract DNA Genome dimensions
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
            if "hidden_dim" in genome: config.net.hidden_dim = genome["hidden_dim"]
            if "latent_dim" in genome: config.net.latent_dim = genome["latent_dim"]

agent_brain = CoREAgent(config=config, device=device).to(device)
hu = HomeostaticUnit(batch_size=1, device=device)
episodic_mem = BatchedEpisodicMemory(batch_size=1, memory_dim=config.net.unified_dim, max_capacity=200, device=device)

# 3. Preserve epoch & story_idx
h_fast, h_slow, epoch, story_idx = load_karyon(agent_brain, episodic_mem, hu, filepath=kcore_path, device=device)

print(f"[Evolution Driver] Current Architecture -> hidden_dim: {agent_brain.hidden_dim}, latent_dim: {agent_brain.latent_dim} | Current Epoch: {epoch}")

# 4. Morphogenesis Trial
evolver = KaryonEvolver(kcore_file_path=kcore_path, device=device)

old_genome = {"hidden_dim": agent_brain.hidden_dim, "latent_dim": agent_brain.latent_dim, "sde_gamma": 0.1}

print("\n--- [STEP 1: DNA MUTATION] ---")
manifest_stub = {"genome": old_genome}
updated_manifest = evolver.mutate_genome(manifest_stub)
new_genome = updated_manifest["genome"]

print("\n--- [STEP 2: NET2NET WEIGHT MORPHING] ---")
old_state_dict = agent_brain.state_dict()
morphed_state_dict = evolver.morph_agent_weights(old_state_dict, old_genome, new_genome)

config.net.hidden_dim = new_genome["hidden_dim"]
config.net.latent_dim = new_genome["latent_dim"]

morphed_agent = CoREAgent(config=config, device=device).to(device)
morphed_agent.load_state_dict(morphed_state_dict, strict=False)

print(f"[Evolution Driver] Morphogenesis Successful! New brain dimension -> hidden_dim: {morphed_agent.hidden_dim}, latent_dim: {morphed_agent.latent_dim}")

# 5. Persist mutated entity while keeping current epoch progress!
h_fast_expanded = torch.zeros(1, morphed_agent.hidden_dim, device=device)
h_slow_expanded = torch.zeros(1, morphed_agent.hidden_dim, device=device)

save_karyon(morphed_agent, episodic_mem, hu, h_fast_expanded, h_slow_expanded, epoch=epoch, story_idx=story_idx, filepath=kcore_path)

print(f"[Evolution Driver] Evolved entity DNA successfully saved into '{kcore_path}' (Preserved Epoch {epoch}).")
