# kcore_builder.py
import os
import struct
import json
import subprocess
import torch

from init_priors import initialize_priors

def compile_to_llvm_bitcode(cpp_source_path: str, output_bc_path: str = "karyon_core.bc") -> tuple:
    """Compiles C++ source into optimized LLVM Bitcode (.bc) using Clang frontend."""
    print(f"[LLVM Builder] Compiling '{cpp_source_path}' to LLVM IR Bitcode...")
    
    try:
        import torch.utils.cpp_extension
        includes = [f"-I{p}" for p in torch.utils.cpp_extension.include_paths()]
        
        cmd = [
            "clang++", "-O3", "-std=c++20", "-emit-llvm", "-c",
            cpp_source_path, "-o", output_bc_path, "-fPIC"
        ] + includes

        subprocess.run(cmd, check=True)
        
        with open(output_bc_path, "rb") as f:
            bitcode_data = f.read()
            
        print(f"[LLVM Builder] SUCCESS: LLVM Bitcode compiled ({len(bitcode_data) / 1024:.2f} KB)")
        return bitcode_data, 5 # LOGIC_LLVM_BITCODE (0x05)
        
    except Exception as e:
        print(f"[LLVM Builder] Clang bitcode emission notice ({e}). Storing raw C++ source text fallback.")
        with open(cpp_source_path, "rb") as f:
            return f.read(), 2 # LOGIC_CPP_SOURCE fallback (0x02)

def pack_kcore(checkpoint_path: str = "karyon_soul.pt", cpp_source_path: str = "karyon_core.cpp", output_kcore_path: str = "karyon_soul.kcore"):
    print(f"[KCORE Builder v4.0] Packing autonomous LLVM entity into container: '{output_kcore_path}'...")
    
    if not os.path.exists(cpp_source_path):
        raise FileNotFoundError(f"C++ source '{cpp_source_path}' not found.")

    # Fallback to creating a new baseline .kcore container if .pt file is absent
    if not os.path.exists(checkpoint_path):
        print(f"[KCORE Builder] Legacy checkpoint '{checkpoint_path}' not found. Generating base container via init_priors...")
        initialize_priors(recreate=True, filepath=output_kcore_path)
        return

    chk = torch.load(checkpoint_path, map_location='cpu')
    logic_buffer, logic_section_type = compile_to_llvm_bitcode(cpp_source_path)

    weights_buffer = bytearray()
    tensor_index = {}
    current_offset = 0
    state_dict = chk['agent_state_dict']
    
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
        "thought_fast_state": chk['thought_fast_state'],
        "thought_slow_state": chk['thought_slow_state'],
        "homeostasis_state": chk['homeostasis_state'],
        "memory_keys": chk['memory_keys'],
        "memory_values": chk['memory_values'],
        "memory_pointer": chk['memory_pointer'],
        "memory_size": chk['memory_size']
    }
    
    for name, tensor in states_dict.items():
        padding = (64 - (state_offset % 64)) % 64
        state_buffer.extend(b'\x00' * padding)
        state_offset += padding
        
        s_data = tensor.detach().cpu().contiguous().numpy().tobytes()
        state_index[name] = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "offset": state_offset,
            "size": len(s_data)
        }
        state_buffer.extend(s_data)
        state_offset += len(s_data)

    manifest = {
        "version": "2.0.0",
        "arch": "Karyon-CoRE LLVM Bitcode Architecture",
        "genome": {
            "hidden_dim": 256,
            "latent_dim": 64,
            "sde_gamma": 0.1
        },
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
    size_logic = len(logic_buffer)
    
    offset_weights = offset_logic + size_logic
    padding_weights = (64 - (offset_weights % 64)) % 64
    offset_weights += padding_weights
    size_weights = len(weights_buffer)
    
    offset_state = offset_weights + size_weights
    padding_state = (64 - (offset_state % 64)) % 64
    offset_state += padding_state
    size_state = len(state_buffer)
    
    total_file_size = offset_state + size_state

    with open(output_kcore_path, 'wb') as f:
        f.write(bytes([75, 67, 79, 82, 69, 2, 0, 0])) # Magic KCORE\x02\x00\x00
        f.write(struct.pack('<IIQQ', header_size, num_sections, total_file_size, 0))
        
        def write_sec_hdr(s_type, flags, offset, size, name):
            name_bytes = name.encode('utf-8')[:31].ljust(32, b'\x00')
            f.write(struct.pack('<IIQQQ', s_type, flags, offset, size, 64))
            f.write(name_bytes)

        write_sec_hdr(1, 0, offset_manifest, size_manifest, "manifest")
        write_sec_hdr(logic_section_type, 0, offset_logic, size_logic, "logic_llvm_bitcode")
        write_sec_hdr(3, 0, offset_weights, size_weights, "weights")
        write_sec_hdr(4, 0, offset_state, size_state, "persistent_state")
        
        f.write(manifest_bytes)
        f.write(b'\x00' * padding_logic)
        f.write(logic_buffer)
        f.write(b'\x00' * padding_weights)
        f.write(weights_buffer)
        f.write(b'\x00' * padding_state)
        f.write(state_buffer)

    print(f"[KCORE Builder] SUCCESS! LLVM Container '{output_kcore_path}' built successfully ({total_file_size / (1024*1024):.2f} MB).")

if __name__ == "__main__":
    pack_kcore()
