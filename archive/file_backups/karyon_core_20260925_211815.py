# karyon_core.py
"""
===============================================================================
KARYON CORE C++20 JIT LOADER & RUNTIME DISPATCHER (v34.5)
===============================================================================
Compiles and loads karyon_core.cpp LibTorch extension dynamically on GPU/CPU.
Injects compiled classes into the Python karyon_core namespace.
===============================================================================
"""
import os
import sys
import torch
from torch.utils.cpp_extension import load

workspace_dir = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(workspace_dir, "karyon_core.cpp")
build_dir = os.path.join(workspace_dir, "build", "karyon_core_jit")
os.makedirs(build_dir, exist_ok=True)

extra_cflags = ["-O3", "-std=c++20"]

# Check if a karyon_core_ext module is already loaded in sys.modules to prevent PyBind11 duplicate registration
karyon_cpp = None
for mod_name, mod in list(sys.modules.items()):
    if mod_name.startswith("karyon_core_ext"):
        karyon_cpp = mod
        break

if karyon_cpp is None:
    module_name = "karyon_core_ext"
    try:
        karyon_cpp = load(
            name=module_name,
            sources=[source_path],
            extra_cflags=extra_cflags,
            build_directory=build_dir,
            verbose=False
        )
    except Exception as e:
        sys.stderr.write(f"❌ Karyon C++ JIT Compilation Failed: {str(e)}\n")
        raise e

if karyon_cpp is not None:
    globals()["UniversalManifold"] = getattr(karyon_cpp, "UniversalManifold", None)
    globals()["CausalParallelSSD"] = getattr(karyon_cpp, "CausalParallelSSD", None)
    globals()["ParallelOperatorBank"] = getattr(karyon_cpp, "ParallelOperatorBank", None)
    globals()["ContinuousHopfieldMemory"] = getattr(karyon_cpp, "ContinuousHopfieldMemory", None)
    globals()["LatentPredictor"] = getattr(karyon_cpp, "LatentPredictor", None)
    globals()["HomeostaticNexus"] = getattr(karyon_cpp, "HomeostaticNexus", None)
    globals()["LinearAccumulatorOp"] = getattr(karyon_cpp, "LinearAccumulatorOp", None)
    globals()["BilinearMultiplicativeOp"] = getattr(karyon_cpp, "BilinearMultiplicativeOp", None)
    globals()["SaturatedAttractorOp"] = getattr(karyon_cpp, "SaturatedAttractorOp", None)
    globals()["ContinuousHopfieldOp"] = getattr(karyon_cpp, "ContinuousHopfieldOp", None)
    globals()["StateSpaceMemoryOp"] = getattr(karyon_cpp, "StateSpaceMemoryOp", None)
    globals()["DynamicMorphicGraph"] = getattr(karyon_cpp, "DynamicMorphicGraph", None)
else:
    raise ImportError("Failed to load or compile Karyon C++ extension module.")
