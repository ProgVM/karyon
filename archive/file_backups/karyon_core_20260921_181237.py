# karyon_core.py
"""
===============================================================================
KARYON CORE C++20 LIBTORCH COMPILATION & PYTHON BRIDGE v34.0 MASTER
Python as Client, C++20 as Engine (KEP Principle 1)
Clean Slate AGN Parallel Evolution Core
===============================================================================
"""
import os
import sys
import importlib
import torch
from torch.utils.cpp_extension import load

def _is_valid_karyon_cpp_module(mod):
    if mod is None:
        return False
    required_attrs = [
        "UniversalManifold", "ParallelOperatorBank", "OmniMorphicNode",
        "OmniContinuousGraphSubstrate", "CognitiveEvolvableAgent",
        "UniversalMorphicCell", "UniversalMorphicSpace", "DynamicMorphicGraph"
    ]
    return all(hasattr(mod, attr) and isinstance(getattr(mod, attr), type) for attr in required_attrs)

# Step 1: Check sys.modules for any already loaded C++ extension binary
karyon_cpp = None
for mod_name, mod in list(sys.modules.items()):
    if "karyon_cpp_ext" in mod_name or "karyon_core_ext" in mod_name:
        if _is_valid_karyon_cpp_module(mod):
            karyon_cpp = mod
            break

# Step 2: Attempt dynamic JIT compilation if not loaded
if karyon_cpp is None:
    # Use workspace directory as root
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(workspace_dir, "karyon_core.cpp")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Karyon-CoRE C++ source file not found at: {source_path}")

    # Build directory setup
    build_dir = os.path.join(workspace_dir, "build", "karyon_core_jit")
    os.makedirs(build_dir, exist_ok=True)

    # Detect active compiler flags
    extra_cflags = ["-O3", "-std=c++20", "-ffast-math", "-march=native", "-w"]
    if torch.cuda.is_available():
        extra_cflags.append("-D__CUDA_INTERNAL__")

    # Generate unique candidate module name to prevent registration collisions
    import uuid
    candidate_name = f"karyon_core_ext_{uuid.uuid4().hex[:8]}"

    try:
        # Load C++ extension dynamically
        karyon_cpp = load(
            name=candidate_name,
            sources=[source_path],
            extra_cflags=extra_cflags,
            build_directory=build_dir,
            verbose=False
        )
    except Exception as e:
        sys.stderr.write(f"❌ Karyon C++ JIT Compilation Failed: {str(e)}\n")
        raise e

# Step 3: Inject C++ classes into python namespace
if karyon_cpp is not None:
    globals()["UniversalManifold"] = getattr(karyon_cpp, "UniversalManifold")
    globals()["CausalParallelSSD"] = getattr(karyon_cpp, "CausalParallelSSD", None)
    globals()["ParallelOperatorBank"] = getattr(karyon_cpp, "ParallelOperatorBank")
    globals()["ContinuousHopfieldMemory"] = getattr(karyon_cpp, "ContinuousHopfieldMemory", None)
    globals()["LatentPredictor"] = getattr(karyon_cpp, "LatentPredictor", None)
    globals()["HomeostaticNexus"] = getattr(karyon_cpp, "HomeostaticNexus", None)
    globals()["OmniMorphicNode"] = getattr(karyon_cpp, "OmniMorphicNode")
    globals()["OmniContinuousGraphSubstrate"] = getattr(karyon_cpp, "OmniContinuousGraphSubstrate")
    globals()["CognitiveEvolvableAgent"] = getattr(karyon_cpp, "CognitiveEvolvableAgent")
    globals()["UniversalMorphicCell"] = getattr(karyon_cpp, "UniversalMorphicCell", None)
    globals()["UniversalMorphicSpace"] = getattr(karyon_cpp, "UniversalMorphicSpace", None)
    globals()["LinearAccumulatorOp"] = getattr(karyon_cpp, "LinearAccumulatorOp", None)
    globals()["BilinearMultiplicativeOp"] = getattr(karyon_cpp, "BilinearMultiplicativeOp", None)
    globals()["SaturatedAttractorOp"] = getattr(karyon_cpp, "SaturatedAttractorOp", None)
    globals()["DynamicMorphicGraph"] = getattr(karyon_cpp, "DynamicMorphicGraph", None)
else:
    raise ImportError("Failed to load or compile Karyon C++ extension module.")
