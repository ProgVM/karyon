# karyon_core.py
"""
===============================================================================
KARYON CORE C++20 JIT LOADER & RUNTIME DISPATCHER (v35.0)
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


def _is_valid_karyon_cpp(mod):
    return mod is not None and hasattr(mod, "CausalParallelSSD") and getattr(mod, "CausalParallelSSD") is not None


# Step 1: Check sys.modules for any already valid loaded C++ extension binary
karyon_cpp = None
for mod_name, mod in list(sys.modules.items()):
    if (mod_name.startswith("karyon_core_ext") or mod_name.startswith("karyon_cpp_ext")) and _is_valid_karyon_cpp(mod):
        karyon_cpp = mod
        break

# Step 2: Load or compile if not already valid in memory
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

if _is_valid_karyon_cpp(karyon_cpp):
    globals()["UniversalManifold"] = getattr(karyon_cpp, "UniversalManifold", None)
    globals()["CausalParallelSSD"] = getattr(karyon_cpp, "CausalParallelSSD", None)
    globals()["EndogenousThetaGammaPAC"] = getattr(karyon_cpp, "EndogenousThetaGammaPAC", None)
    globals()["TriScaleHierarchicalPAC"] = getattr(karyon_cpp, "TriScaleHierarchicalPAC", None)
    globals()["ParallelOperatorBank"] = getattr(karyon_cpp, "ParallelOperatorBank", None)
    globals()["ContinuousHopfieldMemory"] = getattr(karyon_cpp, "ContinuousHopfieldMemory", None)
    globals()["ContinuousSaccadicDrift"] = getattr(karyon_cpp, "ContinuousSaccadicDrift", None)
    globals()["LatentPredictor"] = getattr(karyon_cpp, "LatentPredictor", None)
    globals()["HomeostaticNexus"] = getattr(karyon_cpp, "HomeostaticNexus", None)
    globals()["LinearAccumulatorOp"] = getattr(karyon_cpp, "LinearAccumulatorOp", None)
    globals()["BilinearMultiplicativeOp"] = getattr(karyon_cpp, "BilinearMultiplicativeOp", None)
    globals()["SaturatedAttractorOp"] = getattr(karyon_cpp, "SaturatedAttractorOp", None)
    globals()["LandauDoubleWellOp"] = getattr(karyon_cpp, "LandauDoubleWellOp", None)
    globals()["ContinuousHopfieldOp"] = getattr(karyon_cpp, "ContinuousHopfieldOp", None)
    globals()["StateSpaceMemoryOp"] = getattr(karyon_cpp, "StateSpaceMemoryOp", None)
    globals()["StochasticLangevinOp"] = getattr(karyon_cpp, "StochasticLangevinOp", None)
    globals()["ProgrammableDelayOp"] = getattr(karyon_cpp, "ProgrammableDelayOp", None)
    globals()["TsodyksMarkramSynapticDepressionOp"] = getattr(karyon_cpp, "TsodyksMarkramSynapticDepressionOp", None)
    globals()["SlotMemoryOp"] = getattr(karyon_cpp, "SlotMemoryOp", None)
    globals()["NonLinearTransformOp"] = getattr(karyon_cpp, "NonLinearTransformOp", None)
    globals()["DynamicMorphicGraph"] = getattr(karyon_cpp, "DynamicMorphicGraph", None)
    globals()["kcore"] = karyon_cpp
else:
    raise ImportError("Failed to load or compile valid Karyon C++ extension module.")
