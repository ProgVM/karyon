# karyon_checkpoint.py
"""
===============================================================================
KARYON CHECKPOINT & BINARY CONTAINER v4.0
Zero-Copy Serializer and Loader for .kcore Containers.
===============================================================================
"""

import os
import struct
import json
import torch
import numpy as np

def adapt_and_copy_batch_buffer(target_tensor, source_tensor):
    """Safely copies tensor data between target and source, handling dimension differences."""
    src = source_tensor.to(target_tensor.device)
    
    if target_tensor.shape == src.shape:
        target_tensor.copy_(src)
        return

    if target_tensor.dim() == 1 and src.dim() == 1:
        copy_b = min(target_tensor.size(0), src.size(0))
        target_tensor[:copy_b].copy_(src[:copy_b])
        return

    if target_tensor.dim() == 2 and src.dim() == 2:
        copy_b = min(target_tensor.size(0), src.size(0))
        copy_d = min(target_tensor.size(1), src.size(1))
        target_tensor[:copy_b, :copy_d].copy_(src[:copy_b, :copy_d])
        return

    if target_tensor.dim() == 3 and src.dim() == 3:
        copy_b = min(target_tensor.size(0), src.size(0))
        copy_c = min(target_tensor.size(1), src.size(1))
        copy_d = min(target_tensor.size(2), src.size(2))
        target_tensor[:copy_b, :copy_c, :copy_d].copy_(src[:copy_b, :copy_c, :copy_d])
        return

    slices = tuple(slice(0, min(t_d, s_d)) for t_d, s_d in zip(target_tensor.shape, src.shape))
    target_tensor[slices].copy_(src[slices])

def save_karyon(agent, memory, hu, h_fast, h_slow, epoch=0, story_idx=0, filepath="karyon_soul.kcore", cpp_source_path="karyon_core.cpp"):
    """Saves agent parameters, memory buffers, homeostasis, and DNA into a .kcore container."""
    if hasattr(agent, 'get_complete_state_dict'):
        state_dict = agent.get_complete_state_dict()
    else:
        state_dict = agent.state_dict()
    
    cpp_code = b""
    if os.path.exists(cpp_source_path):
        with open(cpp_source_path, 'rb') as f:
            cpp_code = f.read()

    weights_buffer = bytearray()
    tensor_index = {}
    current_offset = 0
    
    for name, tensor in state_dict.items():
        padding = (64 - (current_offset % 64)) % 64
        weights_buffer.extend(b'\x00' * padding)
        current_offset += padding
        
        t_data = tensor.detach().cpu().contiguous().numpy().tobytes()
        tensor_index[name] = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "offset": current_offset,
            "size": len(t_data)
        }
        weights_buffer.extend(t_data)
        current_offset += len(t_data)

    state_buffer = bytearray()
    state_index = {}
    state_offset = 0
    
    states_dict = {
        "thought_fast_state": h_fast.detach().cpu(),
        "thought_slow_state": h_slow.detach().cpu(),
        "homeostasis_state": hu.state.detach().cpu(),
        "memory_keys": memory.keys.detach().cpu(),
        "memory_values": memory.values.detach().cpu(),
        "memory_pointer": memory.pointer.detach().cpu(),
        "memory_size": memory.size.detach().cpu()
    }
    
    for name, tensor in states_dict.items():
        padding = (64 - (state_offset % 64)) % 64
        state_buffer.extend(b'\x00' * padding)
        state_offset += padding
        
        s_data = tensor.contiguous().numpy().tobytes()
        state_index[name] = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "offset": state_offset,
            "size": len(s_data)
        }
        state_buffer.extend(s_data)
        state_offset += len(s_data)

    genome_dna = {
        "text_dim": agent.config.net.text_dim,
        "text_gen_dim": agent.config.net.text_gen_dim,
        "unified_dim": agent.unified_dim,
        "hidden_dim": agent.hidden_dim,
        "latent_dim": agent.latent_dim,
        "action_dim": agent.action_dim,
        "max_capacity": memory.max_capacity
    }

    manifest = {
        "version": "1.0.0",
        "arch": "Karyon-CoRE v15.2 Master SSD-SwiGLU",
        "epoch": epoch,
        "story_idx": story_idx,
        "genome": genome_dna,
        "tensors": tensor_index,
        "states": state_index
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode('utf-8')

    header_size = 32
    sec_header_size = 64
    num_sections = 4
    payload_start = header_size + (sec_header_size * num_sections)
    
    offset_manifest = payload_start
    size_manifest = len(manifest_bytes)
    
    offset_logic = offset_manifest + size_manifest
    padding_logic = (64 - (offset_logic % 64)) % 64
    offset_logic += padding_logic
    size_logic = len(cpp_code)
    
    offset_weights = offset_logic + size_logic
    padding_weights = (64 - (offset_weights % 64)) % 64
    offset_weights += padding_weights
    size_weights = len(weights_buffer)
    
    offset_state = offset_weights + size_weights
    padding_state = (64 - (offset_state % 64)) % 64
    offset_state += padding_state
    size_state = len(state_buffer)
    
    total_file_size = offset_state + size_state

    with open(filepath, 'wb') as f:
        f.write(bytes([75, 67, 79, 82, 69, 1, 0, 0]))
        f.write(struct.pack('<IIQQ', header_size, num_sections, total_file_size, 0))
        
        def write_sec_hdr(s_type, flags, offset, size, name):
            name_bytes = name.encode('utf-8')[:31].ljust(32, b'\x00')
            f.write(struct.pack('<IIQQQ', s_type, flags, offset, size, 64))
            f.write(name_bytes)

        write_sec_hdr(1, 0, offset_manifest, size_manifest, "manifest")
        write_sec_hdr(2, 0, offset_logic, size_logic, "logic_cpp")
        write_sec_hdr(3, 0, offset_weights, size_weights, "weights")
        write_sec_hdr(4, 0, offset_state, size_state, "persistent_state")
        
        f.write(manifest_bytes)
        f.write(b'\x00' * padding_logic)
        f.write(cpp_code)
        f.write(b'\x00' * padding_weights)
        f.write(weights_buffer)
        f.write(b'\x00' * padding_state)
        f.write(state_buffer)

    print(f"[KCORE Checkpoint] Complete State & DNA Genome persisted into container: '{filepath}' ({total_file_size / (1024*1024):.2f} MB)")

def load_karyon(agent, memory, hu, filepath="karyon_soul.kcore", device='cpu'):
    """Loads agent weights, memory, homeostasis, and states from .kcore container."""
    if not os.path.exists(filepath):
        print(f"[KCORE Checkpoint] Container file '{filepath}' not found. Initializing base state.")
        h_fast = torch.zeros(1, agent.hidden_dim, device=device)
        h_slow = torch.zeros(1, agent.hidden_dim, device=device)
        return h_fast, h_slow, 0, 0

    with open(filepath, 'rb') as f:
        magic = f.read(8)
        if magic[:5] != b'KCORE':
            print(f"[KCORE Checkpoint] File '{filepath}' is not a valid .kcore container.")
            return torch.zeros(1, agent.hidden_dim, device=device), torch.zeros(1, agent.hidden_dim, device=device), 0, 0

        header_raw = f.read(24)
        header_size, num_sections, total_file_size, flags = struct.unpack('<IIQQ', header_raw)

        sections = []
        for _ in range(num_sections):
            sec_raw = f.read(64)
            s_type, s_flags, offset, size, align = struct.unpack('<IIQQQ', sec_raw[:32])
            s_name = sec_raw[32:].rstrip(b'\x00').decode('utf-8')
            sections.append({"type": s_type, "offset": offset, "size": size, "name": s_name})

        sec_manifest = next(s for s in sections if s["type"] == 1)
        f.seek(sec_manifest["offset"])
        manifest = json.loads(f.read(sec_manifest["size"]).decode('utf-8'))

        sec_weights = next(s for s in sections if s["type"] == 3)
        f.seek(sec_weights["offset"])
        weights_data = f.read(sec_weights["size"])

        sec_state = next(s for s in sections if s["type"] == 4)
        f.seek(sec_state["offset"])
        state_data = f.read(sec_state["size"])

    dtype_map = {
        "torch.float32": np.float32,
        "torch.int64": np.int64,
        "torch.float16": np.float16
    }

    agent_state_dict = {}
    for name, meta in manifest["tensors"].items():
        np_dtype = dtype_map.get(meta["dtype"], np.float32)
        raw_bytes = weights_data[meta["offset"]:meta["offset"] + meta["size"]]
        array = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(meta["shape"])
        agent_state_dict[name] = torch.from_numpy(array.copy()).to(device)

    if hasattr(agent, 'load_complete_state_dict'):
        agent.load_complete_state_dict(agent_state_dict, device=device)
    else:
        agent.load_state_dict(agent_state_dict, strict=False)

    states_dict = {}
    for name, meta in manifest["states"].items():
        np_dtype = dtype_map.get(meta["dtype"], np.float32)
        raw_bytes = state_data[meta["offset"]:meta["offset"] + meta["size"]]
        array = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(meta["shape"])
        states_dict[name] = torch.from_numpy(array.copy()).to(device)

    if "memory_keys" in states_dict:
        adapt_and_copy_batch_buffer(memory.keys, states_dict["memory_keys"])
        adapt_and_copy_batch_buffer(memory.values, states_dict["memory_values"])
        adapt_and_copy_batch_buffer(memory.pointer, states_dict["memory_pointer"])
        adapt_and_copy_batch_buffer(memory.size, states_dict["memory_size"])

    if "homeostasis_state" in states_dict:
        adapt_and_copy_batch_buffer(hu.state, states_dict["homeostasis_state"])

    h_fast_saved = states_dict.get("thought_fast_state", torch.zeros(1, agent.hidden_dim, device=device))
    h_slow_saved = states_dict.get("thought_slow_state", torch.zeros(1, agent.hidden_dim, device=device))

    h_fast = torch.zeros(memory.batch_size, agent.hidden_dim, device=device)
    h_slow = torch.zeros(memory.batch_size, agent.hidden_dim, device=device)

    adapt_and_copy_batch_buffer(h_fast, h_fast_saved)
    adapt_and_copy_batch_buffer(h_slow, h_slow_saved)

    epoch = manifest.get("epoch", 0)
    story_idx = manifest.get("story_idx", 0)

    print(f"[KCORE Checkpoint] Successfully restored 100% of entity state & DNA from container '{filepath}'")
    return h_fast, h_slow, epoch, story_idx
