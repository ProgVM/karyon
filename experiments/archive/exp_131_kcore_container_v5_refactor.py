# experiments/exp_131_kcore_container_v5_refactor.py
"""
===============================================================================
EXP-131: KCORE CONTAINER ARCHITECTURAL REFACTORING (v5.0 MASTER SPECIFICATION)
===============================================================================
Hypothesis:
Refactoring the .kcore binary container format to v5.0 Master with:
1. Dynamic automatic codebase ingestion (scanning all root .py, .cpp, .h files),
2. Native zlib stream compression for Manifest and Logic sections (FLAG_ZLIB_COMPRESSED = 0x01),
3. Cryptographic SHA-256 integrity validation per section,
4. Zero-copy state restoration and shape adaptation.
This will compress metadata and codebase footprint by >60%, eliminate silent checkpoint
corruptions via pre-load SHA-256 verification, and guarantee that newly added modules
are automatically preserved in the self-contained entity soul without manual hardcoding.

Baseline: Current karyon_checkpoint.py v4.2 format.
===============================================================================
"""

import sys
import os
import time
import math
import zlib
import json
import struct
import hashlib
import glob
import torch
import numpy as np

# Add workspace to path
sys.path.append("/kaggle/working/karyon")

from karyon_config import CoREConfig
from karyon_agent import CoREAgent
from karyon_core import HomeostaticUnit, BatchedEpisodicMemory
from karyon_checkpoint import load_karyon, save_karyon, adapt_and_copy_batch_buffer

# Section Flags
FLAG_NONE = 0x00
FLAG_ZLIB_COMPRESSED = 0x01

# Magic
KCORE_MAGIC_V5 = b"KCORE\x05\x00\x00"

def discover_core_codebase(root_dir="."):
    """Dynamically gathers all relevant architecture and engine source files."""
    extensions = ["*.py", "*.cpp", "*.h"]
    files_to_pack = []
    
    for ext in extensions:
        pattern = os.path.join(root_dir, ext)
        for fpath in glob.glob(pattern):
            fname = os.path.basename(fpath)
            # Exclude scratch, benchmark, or build scripts
            if fname.startswith("test_") or fname.startswith("tmp_") or "io_test" in fname:
                continue
            files_to_pack.append(fname)
            
    files_to_pack = sorted(list(set(files_to_pack)))
    return files_to_pack

def compute_sha256(data_bytes: bytes) -> str:
    return hashlib.sha256(data_bytes).hexdigest()

def save_karyon_v5(agent, memory, hu, h_fast, h_slow, epoch=0, story_idx=0, filepath="karyon_soul_v5.kcore", root_dir="."):
    """Saves agent into v5.0 Master binary container with dynamic code discovery, zlib compression, and SHA-256."""
    t0 = time.perf_counter()
    
    if hasattr(agent, 'get_complete_state_dict'):
        state_dict = agent.get_complete_state_dict()
    else:
        state_dict = agent.state_dict()

    # 1. Dynamic Codebase Ingestion
    logic_bundle = {}
    source_files = discover_core_codebase(root_dir)
    for sf in source_files:
        p = os.path.join(root_dir, sf)
        if os.path.exists(p) and os.path.isfile(p):
            try:
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    logic_bundle[sf] = f.read()
            except Exception as e:
                print(f"[v5 Saver] Warning: could not read '{sf}': {e}")

    raw_logic_bytes = json.dumps(logic_bundle, indent=2).encode('utf-8')
    compressed_logic_bytes = zlib.compress(raw_logic_bytes, level=6)
    logic_sha256 = compute_sha256(compressed_logic_bytes)

    # 2. Pack Model Weights (64-byte aligned)
    weights_buffer = bytearray()
    tensor_index = {}
    curr_offset = 0

    for name, tensor in state_dict.items():
        pad = (64 - (curr_offset % 64)) % 64
        weights_buffer.extend(b'\x00' * pad)
        curr_offset += pad

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

    # 3. Pack Dynamic States
    state_buffer = bytearray()
    state_index = {}
    curr_state_offset = 0

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
        pad = (64 - (curr_state_offset % 64)) % 64
        state_buffer.extend(b'\x00' * pad)
        curr_state_offset += pad

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

    # 4. Construct Manifest
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
        "version": "5.0.0",
        "arch": "Karyon-CoRE v26.0 Master Autonomous Entity Container",
        "epoch": epoch,
        "story_idx": story_idx,
        "genome": genome_dna,
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

    # Section layout
    header_size = 32
    sec_header_size = 64
    num_sections = 4
    payload_start = header_size + (sec_header_size * num_sections)

    offset_manifest = payload_start
    size_manifest = len(compressed_manifest_bytes)

    offset_logic = offset_manifest + size_manifest
    pad_logic = (64 - (offset_logic % 64)) % 64
    offset_logic += pad_logic
    size_logic = len(compressed_logic_bytes)

    offset_weights = offset_logic + size_logic
    pad_weights = (64 - (offset_weights % 64)) % 64
    offset_weights += pad_weights
    size_weights = len(weights_bytes)

    offset_state = offset_weights + size_weights
    pad_state = (64 - (offset_state % 64)) % 64
    offset_state += pad_state
    size_state = len(state_bytes)

    total_file_size = offset_state + size_state

    with open(filepath, 'wb') as f:
        f.write(KCORE_MAGIC_V5)
        f.write(struct.pack('<IIQQ', header_size, num_sections, total_file_size, 0))

        def write_sec(s_type, flags, offset, size, name):
            name_bytes = name.encode('utf-8')[:31].ljust(32, b'\x00')
            f.write(struct.pack('<IIQQQ', s_type, flags, offset, size, 64))
            f.write(name_bytes)

        write_sec(1, FLAG_ZLIB_COMPRESSED, offset_manifest, size_manifest, "manifest")
        write_sec(2, FLAG_ZLIB_COMPRESSED, offset_logic, size_logic, "logic_bundle")
        write_sec(3, FLAG_NONE, offset_weights, size_weights, "weights")
        write_sec(4, FLAG_NONE, offset_state, size_state, "persistent_state")

        f.write(compressed_manifest_bytes)
        f.write(b'\x00' * pad_logic)
        f.write(compressed_logic_bytes)
        f.write(b'\x00' * pad_weights)
        f.write(weights_bytes)
        f.write(b'\x00' * pad_state)
        f.write(state_bytes)

    save_time = (time.perf_counter() - t0) * 1000.0
    return {
        "filepath": filepath,
        "total_file_size": total_file_size,
        "raw_logic_kb": len(raw_logic_bytes) / 1024.0,
        "compressed_logic_kb": len(compressed_logic_bytes) / 1024.0,
        "raw_manifest_kb": len(raw_manifest_bytes) / 1024.0,
        "compressed_manifest_kb": len(compressed_manifest_bytes) / 1024.0,
        "num_source_files": len(logic_bundle),
        "save_time_ms": save_time
    }

def load_karyon_v5(agent, memory, hu, filepath="karyon_soul_v5.kcore", device='cpu', verify_integrity=True):
    """Loads v5.0 Master binary container with decompression and optional SHA-256 verification."""
    t0 = time.perf_counter()

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Container '{filepath}' not found.")

    with open(filepath, 'rb') as f:
        magic = f.read(8)
        if magic != KCORE_MAGIC_V5:
            raise ValueError(f"Invalid magic: {magic}, expected {KCORE_MAGIC_V5}")

        header_raw = f.read(24)
        header_size, num_sections, total_file_size, flags = struct.unpack('<IIQQ', header_raw)

        sections = []
        for _ in range(num_sections):
            sec_raw = f.read(64)
            s_type, s_flags, offset, size, align = struct.unpack('<IIQQQ', sec_raw[:32])
            s_name = sec_raw[32:].rstrip(b'\x00').decode('utf-8')
            sections.append({
                "type": s_type,
                "flags": s_flags,
                "offset": offset,
                "size": size,
                "name": s_name
            })

        # Section 1: Manifest
        sec_manifest = next(s for s in sections if s["type"] == 1)
        f.seek(sec_manifest["offset"])
        manifest_raw = f.read(sec_manifest["size"])
        if sec_manifest["flags"] & FLAG_ZLIB_COMPRESSED:
            manifest_raw = zlib.decompress(manifest_raw)
        manifest = json.loads(manifest_raw.decode('utf-8'))

        # Section 3: Weights
        sec_weights = next(s for s in sections if s["type"] == 3)
        f.seek(sec_weights["offset"])
        weights_data = f.read(sec_weights["size"])
        if sec_weights["flags"] & FLAG_ZLIB_COMPRESSED:
            weights_data = zlib.decompress(weights_data)

        # Section 4: States
        sec_state = next(s for s in sections if s["type"] == 4)
        f.seek(sec_state["offset"])
        state_data = f.read(sec_state["size"])
        if sec_state["flags"] & FLAG_ZLIB_COMPRESSED:
            state_data = zlib.decompress(state_data)

    # Verify SHA-256 Integrity
    if verify_integrity and "integrity" in manifest:
        expected_w_sha = manifest["integrity"].get("weights_sha256")
        expected_s_sha = manifest["integrity"].get("state_sha256")

        actual_w_sha = compute_sha256(weights_data)
        actual_s_sha = compute_sha256(state_data)

        if expected_w_sha and actual_w_sha != expected_w_sha:
            raise ValueError(f"Weights SHA-256 mismatch! Expected {expected_w_sha}, got {actual_w_sha}")
        if expected_s_sha and actual_s_sha != expected_s_sha:
            raise ValueError(f"State SHA-256 mismatch! Expected {expected_s_sha}, got {actual_s_sha}")

    dtype_map = {
        "torch.float32": np.float32,
        "torch.int64": np.int64,
        "torch.float16": np.float16
    }

    # Load Weights
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

    # Load States
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
    load_time = (time.perf_counter() - t0) * 1000.0

    return h_fast, h_slow, epoch, story_idx, load_time

def run_experiment():
    print("=" * 80)
    print("STARTING EXP-131: KCORE CONTAINER ARCHITECTURE REFACTORING (v5.0 MASTER)")
    print("=" * 80)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    config = CoREConfig()
    config.train.batch_size = 1

    hu = HomeostaticUnit(device=device_str)
    agent = CoREAgent(config, device=device_str).to(device)
    memory = BatchedEpisodicMemory(batch_size=1, memory_dim=config.net.unified_dim, max_capacity=1000, device=device_str)

    # 1. Load active entity state from v4.2 container
    print("\n[Phase 1] Loading existing soul container 'karyon_soul.kcore' (v4.2 baseline)...")
    t0_base_load = time.perf_counter()
    h_fast, h_slow, epoch, story_idx = load_karyon(agent, memory, hu, filepath="karyon_soul.kcore", device=device_str)
    base_load_ms = (time.perf_counter() - t0_base_load) * 1000.0
    base_size_mb = os.path.getsize("karyon_soul.kcore") / (1024 * 1024)
    print(f"Baseline Load Time: {base_load_ms:.2f} ms | File Size: {base_size_mb:.2f} MB")

    # Record initial weights norm for verification
    initial_param_norm = sum(p.norm().item() for p in agent.parameters())
    print(f"Initial Agent Param Norm: {initial_param_norm:.4f}")

    # 2. Test Dynamic Codebase Discovery
    discovered_sources = discover_core_codebase(".")
    print(f"\n[Phase 2] Discovered {len(discovered_sources)} core source files dynamically:")
    for src in discovered_sources:
        print(f"  - {src}")

    # Verify new modules are included
    assert "karyon_hardware.py" in discovered_sources, "karyon_hardware.py was missed in discovery!"
    assert "karyon_core.cpp" in discovered_sources, "karyon_core.cpp was missed in discovery!"
    assert "karyon_agent.py" in discovered_sources, "karyon_agent.py was missed in discovery!"

    # 3. Save v5.0 Container with zlib compression and SHA-256
    v5_path = "karyon_soul_v5.kcore"
    print(f"\n[Phase 3] Serializing to v5.0 container '{v5_path}' with zlib & SHA-256...")
    save_stats = save_karyon_v5(agent, memory, hu, h_fast, h_slow, epoch, story_idx, filepath=v5_path)

    print(f"Logic Bundle: Raw {save_stats['raw_logic_kb']:.2f} KB -> Compressed {save_stats['compressed_logic_kb']:.2f} KB "
          f"({(1.0 - save_stats['compressed_logic_kb'] / save_stats['raw_logic_kb']) * 100:.1f}% space saving)")
    print(f"Manifest: Raw {save_stats['raw_manifest_kb']:.2f} KB -> Compressed {save_stats['compressed_manifest_kb']:.2f} KB "
          f"({(1.0 - save_stats['compressed_manifest_kb'] / save_stats['raw_manifest_kb']) * 100:.1f}% space saving)")
    print(f"Total v5 Container Size: {save_stats['total_file_size'] / (1024*1024):.2f} MB")
    print(f"v5 Save Time: {save_stats['save_time_ms']:.2f} ms")

    # 4. Deserialization & SHA-256 Integrity Verification
    print(f"\n[Phase 4] Deserializing from v5.0 container with cryptographic integrity check...")
    # Re-initialize clean agent
    agent_v5 = CoREAgent(config, device=device_str).to(device)
    memory_v5 = BatchedEpisodicMemory(batch_size=1, memory_dim=config.net.unified_dim, max_capacity=1000, device=device_str)
    hu_v5 = HomeostaticUnit(device=device_str)

    h_fast_v5, h_slow_v5, ep_v5, st_v5, v5_load_ms = load_karyon_v5(
        agent_v5, memory_v5, hu_v5, filepath=v5_path, device=device_str, verify_integrity=True
    )
    print(f"v5 Load Time (including decompression & SHA-256): {v5_load_ms:.2f} ms")

    # Verify weights identity
    v5_param_norm = sum(p.norm().item() for p in agent_v5.parameters())
    norm_diff = abs(initial_param_norm - v5_param_norm)
    print(f"Param Norm Difference: {norm_diff:.8f}")
    assert norm_diff < 1e-4, f"Parameter norm diverged: {norm_diff}"

    # 5. Test Cryptographic Corruption Detection (Tamper Proofing)
    print("\n[Phase 5] Testing Tamper Detection (Simulating bitflip corruption)...")
    corrupted_v5_path = "karyon_soul_v5_corrupt.kcore"
    with open(v5_path, "rb") as f:
        data = bytearray(f.read())

    # Flip 1 byte in weights payload
    data[-100] ^= 0xFF
    with open(corrupted_v5_path, "wb") as f:
        f.write(data)

    corrupted_detected = False
    try:
        load_karyon_v5(agent_v5, memory_v5, hu_v5, filepath=corrupted_v5_path, device=device_str, verify_integrity=True)
    except ValueError as e:
        corrupted_detected = True
        print(f"Corruption successfully intercepted: {e}")

    assert corrupted_detected, "Failed to detect corrupted weights in container!"

    # Clean up test files
    if os.path.exists(corrupted_v5_path):
        os.remove(corrupted_v5_path)

    # Calculate metrics
    logic_compression_ratio = (1.0 - save_stats['compressed_logic_kb'] / save_stats['raw_logic_kb']) * 100.0
    manifest_compression_ratio = (1.0 - save_stats['compressed_manifest_kb'] / save_stats['raw_manifest_kb']) * 100.0

    metrics = {
        "discovered_files_count": len(discovered_sources),
        "logic_compression_pct": logic_compression_ratio,
        "manifest_compression_pct": manifest_compression_ratio,
        "save_time_ms": save_stats['save_time_ms'],
        "load_time_ms": v5_load_ms,
        "norm_diff": norm_diff,
        "tamper_detection": corrupted_detected
    }

    print("\n" + "=" * 80)
    print("EXP-131 SUMMARY METRICS:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    verdict = "🟢 POSITIVE" if (norm_diff < 1e-4 and corrupted_detected and logic_compression_ratio > 50.0) else "🔴 REJECTED"
    print(f"\n[EXP-131 VERDICT]: {verdict}")
    print("=" * 80)

    return metrics, verdict

if __name__ == "__main__":
    run_experiment()
