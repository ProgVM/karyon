# kcore_builder.py
"""
===============================================================================
KARYON KCORE CONTAINER BUILDER v4.1
Compiles C++20 source into LLVM Bitcode and packs weights, multi-layer cortical states,
and DNA genome into single-file portable binary containers (.kcore).
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import os
import struct
import json
import subprocess
import torch

from init_priors import initialize_priors

def compile_to_llvm_bitcode(cpp_source_path: str = "karyon_core.cpp", output_bc_path: str = "karyon_core.bc") -> tuple:
    """Compiles C++20 source into optimized LLVM Bitcode (.bc) using Clang frontend."""
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
        print(f"[LLVM Builder] Clang bitcode notice ({e}). Storing raw C++ source text fallback.")
        with open(cpp_source_path, "rb") as f:
            return f.read(), 2 # LOGIC_CPP_SOURCE fallback (0x02)

def pack_kcore(output_kcore_path: str = "karyon_soul.kcore", cpp_source_path: str = "karyon_core.cpp"):
    """Packs or initializes the complete autonomous entity into a .kcore container."""
    print(f"[KCORE Builder v4.1] Initializing and packing autonomous container: '{output_kcore_path}'...")
    
    if not os.path.exists(cpp_source_path):
        raise FileNotFoundError(f"C++ source '{cpp_source_path}' not found.")

    initialize_priors(recreate=True, filepath=output_kcore_path)
    print(f"[KCORE Builder v4.1] SUCCESS! Container '{output_kcore_path}' generated successfully.")

if __name__ == "__main__":
    pack_kcore()
