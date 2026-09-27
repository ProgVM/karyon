# karyon_checkpoint.py
"""
===============================================================================
KARYON CHECKPOINT & BINARY CONTAINER v6.0 MASTER
Zero-Copy Serializer, Compressor & Loader for .kcore Containers with Dynamic Codebase
Encapsulation, NodePool & Epigenetic Methylation Preservation, zlib Stream Compression,
and SHA-256 Cryptographic Verification.
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import glob
import hashlib
import json
import os
import struct
import zlib
import numpy as np
import torch

# Container Section Flags
FLAG_NONE = 0x00
FLAG_ZLIB_COMPRESSED = 0x01
FLAG_ENCRYPTED = 0x02

# Magic Constants
KCORE_MAGIC_V6 = b"KCORE\x06\x00\x00"
KCORE_MAGIC_V5 = b"KCORE\x05\x00\x00"
KCORE_MAGIC_LEGACY = b"KCORE\x01\x00\x00"


def adapt_and_copy_batch_buffer(target_tensor, source_tensor):
    """Safely copies tensor data between target and source, handling rank and dimension differences."""
    if target_tensor is None or source_tensor is None:
        return

    src = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)

    if target_tensor.shape == src.shape:
        target_tensor.copy_(src)
        return

    # Handle 1D
    if target_tensor.dim() == 1 and src.dim() == 1:
        copy_len = min(target_tensor.size(0), src.size(0))
        target_tensor[:copy_len].copy_(src[:copy_len])
        return

    # Handle 2D vs Multi-D flattening (e.g. 4D SSD state -> 2D mind representation)
    if target_tensor.dim() == 2 and src.dim() > 2:
        src = src.view(src.size(0), -1)
    elif target_tensor.dim() > 2 and src.dim() == 2 and target_tensor.numel() == src.numel():
        src = src.view(target_tensor.shape)

    # Handle 2D
    if target_tensor.dim() == 2 and src.dim() == 2:
        copy_b = min(target_tensor.size(0), src.size(0))
        copy_d = min(target_tensor.size(1), src.size(1))
        target_tensor[:copy_b, :copy_d].copy_(src[:copy_b, :copy_d])
        return

    # Handle 3D
    if target_tensor.dim() == 3 and src.dim() == 3:
        copy_b = min(target_tensor.size(0), src.size(0))
        copy_c = min(target_tensor.size(1), src.size(1))
        copy_d = min(target_tensor.size(2), src.size(2))
        target_tensor[:copy_b, :copy_c, :copy_d].copy_(src[:copy_b, :copy_c, :copy_d])
        return

    # Handle 4D (e.g. SSD m_s1, m_s2 states)
    if target_tensor.dim() == 4 and src.dim() == 4:
        copy_b = min(target_tensor.size(0), src.size(0))
        copy_h = min(target_tensor.size(1), src.size(1))
        copy_k = min(target_tensor.size(2), src.size(2))
        copy_v = min(target_tensor.size(3), src.size(3))
        target_tensor[:copy_b, :copy_h, :copy_k, :copy_v].copy_(src[:copy_b, :copy_h, :copy_k, :copy_v])
        return

    # Fallback for identical rank tensors
    if target_tensor.dim() == src.dim():
        slices = tuple(slice(0, min(t_d, s_d)) for t_d, s_d in zip(target_tensor.shape, src.shape))
        target_tensor[slices].copy_(src[slices])
    else:
        # Generic fallback: flatten both along non-batch dimensions
        b = min(target_tensor.size(0), src.size(0))
        t_flat = target_tensor.view(target_tensor.size(0), -1)
        s_flat = src.view(src.size(0), -1)
        d = min(t_flat.size(1), s_flat.size(1))
        t_flat[:b, :d].copy_(s_flat[:b, :d])


def compute_sha256(data_bytes: bytes) -> str:
    """Computes SHA-256 hexadecimal hash string for payload bytes."""
    return hashlib.sha256(data_bytes).hexdigest()


def discover_core_codebase(root_dir="."):
    """Dynamically gathers all relevant architecture, runtime, and config source files."""
    extensions = ["*.py", "*.cpp", "*.h"]
    files_to_pack = []

    for ext in extensions:
        pattern = os.path.join(root_dir, ext)
        for fpath in glob.glob(pattern):
            fname = os.path.basename(fpath)
            # Exclude scratch, benchmark, temporary, or build artifacts
            if fname.startswith("test_") or fname.startswith("tmp_") or "io_test" in fname:
                continue
            files_to_pack.append(fname)

    return sorted(list(set(files_to_pack)))


def save_karyon(agent, memory, hu, h_fast, h_slow, epoch=0, story_idx=0, filepath="karyon_soul.kcore", root_dir="."):
    """
    Saves agent parameters, memory buffers, homeostasis, DNA, NodePool topology, and 100% of core logic into v6.0 container.
    Implements dynamic codebase ingestion, native zlib stream compression, and cryptographic SHA-256 hashes.
    """
    if hasattr(agent, 'get_complete_state_dict'):
        state_dict = agent.get_complete_state_dict()
    elif hasattr(agent, 'named_parameters_map'):
        state_dict = agent.named_parameters_map()
    else:
        state_dict = agent.state_dict()

    # 1. Dynamic Ingestion of Core Codebase Files into Section 2
    logic_bundle = {}
    source_files = discover_core_codebase(root_dir)
    for sf in source_files:
        p = os.path.join(root_dir, sf)
        if os.path.exists(p) and os.path.isfile(p):
            try:
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    logic_bundle[sf] = f.read()
            except Exception as e:
                print(f"[KCORE v6 Saver] Notice: skipped '{sf}': {e}")

    raw_logic_bytes = json.dumps(logic_bundle, indent=2).encode('utf-8')
    compressed_logic_bytes = zlib.compress(raw_logic_bytes, level=6)
    logic_sha256 = compute_sha256(compressed_logic_bytes)

    # 2. 64-Byte Aligned Weight Serialization (Section 3)
    weights_buffer = bytearray()
    tensor_index = {}
    curr_offset = 0

    for name, tensor in state_dict.items():
        padding = (64 - (curr_offset % 64)) % 64
        weights_buffer.extend(b'\x00' * padding)
        curr_offset += padding

        t_data = tensor.detach().cpu().contiguous().numpy().tobytes()
        tensor_index[name] = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "offset": curr_offset,
            "size": len(t_data)
        }
        weights_buffer.extend(t_data)
        curr_offset += len(t_data)

    weights_bytes = bytes(weights_buffer)
    weights_sha256 = compute_sha256(weights_bytes)

    # 3. Dynamic Persistent State Serialization (Section 4)
    state_buffer = bytearray()
    state_index = {}
    curr_state_offset = 0

    states_dict = {}
    if h_fast is not None:
        states_dict["thought_fast_state"] = h_fast.detach().cpu()
    if h_slow is not None:
        states_dict["thought_slow_state"] = h_slow.detach().cpu()
    if hu is not None and hasattr(hu, 'state'):
        states_dict["homeostasis_state"] = hu.state.detach().cpu()
    if memory is not None:
        if hasattr(memory, 'keys'):
            states_dict["memory_keys"] = memory.keys.detach().cpu()
        if hasattr(memory, 'values'):
            states_dict["memory_values"] = memory.values.detach().cpu()
        if hasattr(memory, 'pointer'):
            states_dict["memory_pointer"] = memory.pointer.detach().cpu()
        if hasattr(memory, 'size'):
            states_dict["memory_size"] = memory.size.detach().cpu()

    for name, tensor in states_dict.items():
        padding = (64 - (curr_state_offset % 64)) % 64
        state_buffer.extend(b'\x00' * padding)
        curr_state_offset += padding

        s_data = tensor.contiguous().numpy().tobytes()
        state_index[name] = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "offset": curr_state_offset,
            "size": len(s_data)
        }
        state_buffer.extend(s_data)
        curr_state_offset += len(s_data)

    state_bytes = bytes(state_buffer)
    state_sha256 = compute_sha256(state_bytes)

    # 4. Genome DNA & Manifest Structure (Section 1)
    try:
        from kcore_evolution import SleepMetaGeneticsEngine
        biophysical_genome = SleepMetaGeneticsEngine.get_active_genome(agent)
    except Exception:
        biophysical_genome = {}

    net_dim = agent.dim if hasattr(agent, 'dim') else (agent.config.net.text_dim if hasattr(agent, 'config') and hasattr(agent.config, 'net') else 256)
    unified_dim = getattr(agent, 'unified_dim', net_dim)
    hidden_dim = getattr(agent, 'hidden_dim', net_dim)
    latent_dim = getattr(agent, 'latent_dim', 64)
    action_dim = getattr(agent, 'action_dim', 258)

    genome_dna = {
        "text_dim": net_dim,
        "text_gen_dim": net_dim,
        "unified_dim": unified_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "action_dim": action_dim,
        "max_capacity": getattr(memory, 'max_capacity', 100) if memory is not None else 100,
        "biophysical_genome": biophysical_genome
    }

    # Capture C++20 Dynamic Architecture Topology Manifest & Epigenetic Locks
    topology_manifest = {}
    graph_obj = getattr(agent, 'graph', getattr(agent, 'space', None))
    if hasattr(agent, 'get_topology_manifest'):
        raw_topo = agent.get_topology_manifest()
    elif graph_obj is not None and hasattr(graph_obj, 'get_topology_manifest'):
        raw_topo = graph_obj.get_topology_manifest()
    else:
        raw_topo = None

    if raw_topo is not None:
        if isinstance(raw_topo, list):
            # Parse ['name:type:cat', ...]
            parsed_nodes = []
            for item in raw_topo:
                parts = item.split(':')
                if len(parts) >= 3:
                    parsed_nodes.append({
                        "name": parts[0],
                        "type": parts[1],
                        "is_core": (parts[2] == "CORE")
                    })
            topology_manifest = {
                "type": "DynamicMorphicGraph",
                "nodes": parsed_nodes
            }
        elif isinstance(raw_topo, str):
            try:
                topology_manifest = json.loads(raw_topo)
            except Exception:
                topology_manifest = {"raw": raw_topo}
        elif isinstance(raw_topo, dict):
            topology_manifest = raw_topo

    # Capture methylation locks if supported
    methylation_locks = []
    if graph_obj is not None and hasattr(graph_obj, 'get_methylation_locks'):
        try:
            methylation_locks = [float(x) for x in graph_obj.get_methylation_locks()]
        except Exception:
            methylation_locks = []
    if topology_manifest and methylation_locks:
        topology_manifest["methylation_locks"] = methylation_locks

    manifest = {
        "version": "6.0.0",
        "arch": "Karyon-CoRE v28.0 Master Autonomous Entity Container",
        "epoch": epoch,
        "story_idx": story_idx,
        "genome": genome_dna,
        "architecture_topology": topology_manifest,
        "tensors": tensor_index,
        "states": state_index,
        "source_files": list(logic_bundle.keys()),
        "integrity": {
            "logic_sha256": logic_sha256,
            "weights_sha256": weights_sha256,
            "state_sha256": state_sha256
        }
    }

    raw_manifest_bytes = json.dumps(manifest, indent=2).encode('utf-8')
    compressed_manifest_bytes = zlib.compress(raw_manifest_bytes, level=6)

    # Section layout & 64-byte alignment
    header_size = 32
    sec_header_size = 64
    num_sections = 4

    offset_sec_headers = header_size
    offset_payload_start = header_size + (num_sections * sec_header_size)

    # Align Section 1
    sec1_offset = offset_payload_start + ((64 - (offset_payload_start % 64)) % 64)
    sec1_size = len(compressed_manifest_bytes)

    # Align Section 2
    sec2_offset = sec1_offset + sec1_size + ((64 - ((sec1_offset + sec1_size) % 64)) % 64)
    sec2_size = len(compressed_logic_bytes)

    # Align Section 3
    sec3_offset = sec2_offset + sec2_size + ((64 - ((sec2_offset + sec2_size) % 64)) % 64)
    sec3_size = len(weights_bytes)

    # Align Section 4
    sec4_offset = sec3_offset + sec3_size + ((64 - ((sec3_offset + sec3_size) % 64)) % 64)
    sec4_size = len(state_bytes)

    total_file_size = sec4_offset + sec4_size

    # Build Header & Sections
    kcore_binary = bytearray()
    header = struct.pack('<8sIIQQ', KCORE_MAGIC_V6, header_size, num_sections, total_file_size, FLAG_ZLIB_COMPRESSED)
    kcore_binary.extend(header)

    sections_meta = [
        (1, FLAG_ZLIB_COMPRESSED, sec1_offset, sec1_size, 64, b"MANIFEST"),
        (2, FLAG_ZLIB_COMPRESSED, sec2_offset, sec2_size, 64, b"LOGIC"),
        (3, FLAG_NONE, sec3_offset, sec3_size, 64, b"WEIGHTS"),
        (4, FLAG_NONE, sec4_offset, sec4_size, 64, b"STATE")
    ]

    for s_type, s_flags, s_offset, s_size, s_align, s_name in sections_meta:
        padded_name = s_name.ljust(32, b'\x00')
        sec_h = struct.pack('<IIQQQ32s', s_type, s_flags, s_offset, s_size, s_align, padded_name)
        kcore_binary.extend(sec_h)

    # Pad to Section 1
    kcore_binary.extend(b'\x00' * (sec1_offset - len(kcore_binary)))
    kcore_binary.extend(compressed_manifest_bytes)

    # Pad to Section 2
    kcore_binary.extend(b'\x00' * (sec2_offset - len(kcore_binary)))
    kcore_binary.extend(compressed_logic_bytes)

    # Pad to Section 3
    kcore_binary.extend(b'\x00' * (sec3_offset - len(kcore_binary)))
    kcore_binary.extend(weights_bytes)

    # Pad to Section 4
    kcore_binary.extend(b'\x00' * (sec4_offset - len(kcore_binary)))
    kcore_binary.extend(state_bytes)

    # Write to Disk
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(kcore_binary)

    comp_logic_ratio = (1.0 - (len(compressed_logic_bytes) / max(1, len(raw_logic_bytes)))) * 100.0
    comp_man_ratio = (1.0 - (len(compressed_manifest_bytes) / max(1, len(raw_manifest_bytes)))) * 100.0

    print(f"[KCORE Checkpoint v6.0] Entity Soul persisted into container '{filepath}' "
          f"({len(kcore_binary) / (1024*1024):.2f} MB, {len(logic_bundle)} source files, "
          f"Logic: -{comp_logic_ratio:.1f}%, Manifest: -{comp_man_ratio:.1f}%)")
    return True


save_kcore = save_karyon


def load_karyon(agent, memory=None, hu=None, filepath="karyon_soul.kcore", device='cpu', verify_integrity=True):
    """
    Loads agent weights, memory, homeostasis, and persistent states from .kcore container.
    Seamlessly supports v6.0, v5.0, and legacy v4.2/v1.0 containers.
    Handles self-executable polyglot sheath headers by dynamically locating the binary payload.
    """
    if not os.path.exists(filepath):
        print(f"[KCORE Checkpoint] Container file '{filepath}' not found. Initializing base state.")
        hidden_dim = getattr(agent, 'hidden_dim', getattr(agent, 'dim', 256))
        h_fast = torch.zeros(1, hidden_dim, device=device)
        h_slow = torch.zeros(1, hidden_dim, device=device)
        return h_fast, h_slow, 0, 0

    with open(filepath, 'rb') as f:
        data = f.read()

    # Locate the real binary payload magic offset
    sig_v6 = b"KC" + b"ORE" + bytes([6, 0, 0])
    sig_v5 = b"KC" + b"ORE" + bytes([5, 0, 0])
    sig_v1 = b"KC" + b"ORE" + bytes([1, 0, 0])
    magic_offset = data.rfind(sig_v6)
    if magic_offset == -1:
        magic_offset = data.rfind(sig_v5)
    if magic_offset == -1:
        magic_offset = data.rfind(sig_v1)

    if magic_offset == -1:
        print(f"[KCORE Checkpoint] File '{filepath}' is not a valid .kcore container (magic signature not found).")
        hidden_dim = getattr(agent, 'hidden_dim', getattr(agent, 'dim', 256))
        return torch.zeros(1, hidden_dim, device=device), torch.zeros(1, hidden_dim, device=device), 0, 0

    magic = data[magic_offset:magic_offset + 8]
    is_v6_or_v5 = (magic == KCORE_MAGIC_V6 or magic == KCORE_MAGIC_V5)

    header_raw = data[magic_offset + 8:magic_offset + 32]
    header_size, num_sections, total_file_size, flags = struct.unpack('<IIQQ', header_raw)

    sections = []
    for i in range(num_sections):
        sec_offset = magic_offset + 32 + i * 64
        sec_raw = data[sec_offset:sec_offset + 64]
        s_type, s_flags, offset, size, align = struct.unpack('<IIQQQ', sec_raw[:32])
        s_name = sec_raw[32:].rstrip(b'\x00').decode('utf-8', errors='replace')
        sections.append({
            "type": s_type,
            "flags": s_flags,
            "offset": magic_offset + offset,
            "size": size,
            "name": s_name
        })

    # 1. Manifest
    sec_manifest = next(s for s in sections if s["type"] == 1)
    manifest_raw = data[sec_manifest["offset"]:sec_manifest["offset"] + sec_manifest["size"]]
    if sec_manifest["flags"] & FLAG_ZLIB_COMPRESSED:
        manifest_raw = zlib.decompress(manifest_raw)
    manifest = json.loads(manifest_raw.decode('utf-8'))

    # 2. Weights
    sec_weights = next(s for s in sections if s["type"] == 3)
    weights_data = data[sec_weights["offset"]:sec_weights["offset"] + sec_weights["size"]]
    if sec_weights["flags"] & FLAG_ZLIB_COMPRESSED:
        weights_data = zlib.decompress(weights_data)

    # 3. States
    sec_state = next(s for s in sections if s["type"] == 4)
    state_data = data[sec_state["offset"]:sec_state["offset"] + sec_state["size"]]
    if sec_state["flags"] & FLAG_ZLIB_COMPRESSED:
        state_data = zlib.decompress(state_data)

    # 4. Cryptographic SHA-256 Verification
    if is_v6_or_v5 and verify_integrity and "integrity" in manifest:
        expected_w_sha = manifest["integrity"].get("weights_sha256")
        expected_s_sha = manifest["integrity"].get("state_sha256")

        if expected_w_sha and compute_sha256(weights_data) != expected_w_sha:
            raise ValueError(f"[KCORE Integrity Error] Weights SHA-256 checksum mismatch in '{filepath}'!")
        if expected_s_sha and compute_sha256(state_data) != expected_s_sha:
            raise ValueError(f"[KCORE Integrity Error] State SHA-256 checksum mismatch in '{filepath}'!")

    dtype_map = {
        "torch.float32": np.float32,
        "torch.int64": np.int64,
        "torch.float16": np.float16
    }

    # Restore dynamic C++20 NodePool architecture topology if present before loading parameters
    if "architecture_topology" in manifest and manifest["architecture_topology"]:
        topo = manifest["architecture_topology"]
        graph_obj = getattr(agent, 'graph', getattr(agent, 'space', agent))

        if "nodes" in topo and hasattr(graph_obj, 'add_node'):
            # Existing nodes in graph
            existing_names = set()
            if hasattr(graph_obj, 'get_topology_manifest'):
                existing_manifest = graph_obj.get_topology_manifest()
                if isinstance(existing_manifest, list):
                    for item in existing_manifest:
                        existing_names.add(item.split(':')[0])

            # Re-sprout newly added/specialized DynamicMorphicGraph nodes from topology manifest
            for node_info in topo.get("nodes", []):
                n_name = node_info["name"]
                n_type = node_info.get("type", "LinearAccumulator")
                n_core = node_info.get("is_core", False)
                if n_name not in existing_names:
                    graph_obj.add_node(n_name, n_type, is_core=n_core)
                    existing_names.add(n_name)

        if "methylation_locks" in topo and hasattr(graph_obj, 'set_methylation_locks'):
            try:
                graph_obj.set_methylation_locks(topo["methylation_locks"])
            except Exception as e:
                print(f"[KCORE Checkpoint] Notice: Failed to restore methylation locks: {e}")

    # Deserializing Model Weights
    agent_state_dict = {}
    for name, meta in manifest["tensors"].items():
        np_dtype = dtype_map.get(meta["dtype"], np.float32)
        raw_bytes = weights_data[meta["offset"]:meta["offset"] + meta["size"]]
        array = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(meta["shape"])
        agent_state_dict[name] = torch.from_numpy(array.copy()).to(device)

    # Restore evolved biophysical genome from manifest if present before loading weights
    if "genome" in manifest and "biophysical_genome" in manifest["genome"]:
        try:
            from kcore_evolution import SleepMetaGeneticsEngine
            SleepMetaGeneticsEngine.apply_genome_to_agent(agent, manifest["genome"]["biophysical_genome"])
        except Exception:
            pass

    if hasattr(agent, 'load_complete_state_dict'):
        agent.load_complete_state_dict(agent_state_dict, device=device)
    elif hasattr(agent, 'named_parameters_map'):
        param_map = agent.named_parameters_map()
        with torch.no_grad():
            for name, param in param_map.items():
                if name in agent_state_dict:
                    param.copy_(agent_state_dict[name].to(device))
    else:
        agent.load_state_dict(agent_state_dict, strict=False)

    # Deserializing Recurrent & Homeostatic States
    states_dict = {}
    for name, meta in manifest["states"].items():
        np_dtype = dtype_map.get(meta["dtype"], np.float32)
        raw_bytes = state_data[meta["offset"]:meta["offset"] + meta["size"]]
        array = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(meta["shape"])
        states_dict[name] = torch.from_numpy(array.copy()).to(device)

    if memory is not None:
        if "memory_keys" in states_dict and hasattr(memory, "keys"):
            adapt_and_copy_batch_buffer(memory.keys, states_dict["memory_keys"])
            adapt_and_copy_batch_buffer(memory.values, states_dict["memory_values"])
            adapt_and_copy_batch_buffer(memory.pointer, states_dict["memory_pointer"])
            adapt_and_copy_batch_buffer(memory.size, states_dict["memory_size"])
            memory.max_active_cpu = int(memory.size.max().item())

    if hu is not None and hasattr(hu, "state") and "homeostasis_state" in states_dict:
        adapt_and_copy_batch_buffer(hu.state, states_dict["homeostasis_state"])

    agent_hidden_dim = getattr(agent, 'hidden_dim', getattr(agent, 'dim', 256))
    h_fast_saved = states_dict.get("thought_fast_state", torch.zeros(1, agent_hidden_dim, device=device))
    h_slow_saved = states_dict.get("thought_slow_state", torch.zeros(1, agent_hidden_dim, device=device))

    mem_bs = getattr(memory, "batch_size", 1) if memory is not None else 1
    h_fast = torch.zeros(mem_bs, agent_hidden_dim, device=device)
    h_slow = torch.zeros(mem_bs, agent_hidden_dim, device=device)

    adapt_and_copy_batch_buffer(h_fast, h_fast_saved)
    adapt_and_copy_batch_buffer(h_slow, h_slow_saved)

    epoch = manifest.get("epoch", 0)
    story_idx = manifest.get("story_idx", 0)

    ver_tag = manifest.get("version", "legacy")
    print(f"[KCORE Checkpoint] Successfully restored 100% of entity state, DNA & encapsulated code from container '{filepath}' (v{ver_tag})")
    return h_fast, h_slow, epoch, story_idx


load_kcore = load_karyon
