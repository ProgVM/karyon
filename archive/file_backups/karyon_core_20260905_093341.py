# karyon_core.py
"""
===============================================================================
KARYON CORE C++20 LIBTORCH COMPILATION & PYTHON BRIDGE v26.0 MASTER
Python as Client, C++20 as Engine (KEP Principle 1)
Universal Multimodal & Cross-Modal State-Space Cognitive Engine (EXP-130 Validated)
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
    required_attrs = ["ByteTokenizer", "SensoryGateway", "ParallelLogDecaySSDLayer", "FusedCascadedLaminarStack"]
    return all(hasattr(mod, attr) and isinstance(getattr(mod, attr), type) for attr in required_attrs)

# Step 1: Check sys.modules for any already loaded C++ extension binary (excluding Python wrappers & __main__)
karyon_cpp = None
for mod_name, mod in list(sys.modules.items()):
    if mod_name not in ["__main__", "karyon_core", "karyon_agent", "dialogue"] and (mod_name.startswith("karyon_cpp_ext") or mod_name.startswith("karyon_core_ext")):
        if _is_valid_karyon_cpp_module(mod):
            print(f"[C++ JIT] Reusing already loaded C++ module: '{mod_name}'")
            karyon_cpp = mod
            break

# Step 2: Try importing candidate names directly from disk / cache (prefer newest v26)
if karyon_cpp is None:
    for path in [
        "/root/.cache/torch_extensions/py312_cu128/karyon_cpp_ext_v26",
        "/root/.cache/torch_extensions/py312_cu128/karyon_cpp_ext_v25",
        "/root/.cache/torch_extensions/py312_cu128/karyon_cpp_ext_v24",
        "/kaggle/working/karyon/build/karyon_core_jit"
    ]:
        if os.path.exists(path) and path not in sys.path:
            sys.path.append(path)
            
    for candidate_name in ["karyon_cpp_ext_v26", "karyon_cpp_ext_v25", "karyon_cpp_ext_v24"]:
        try:
            mod = importlib.import_module(candidate_name)
            if _is_valid_karyon_cpp_module(mod):
                print(f"[C++ JIT] Successfully imported existing compiled module: '{candidate_name}'")
                karyon_cpp = mod
                break
        except Exception:
            pass

# Step 3: Compile and load if not found
if karyon_cpp is None:
    print("[C++ JIT] Compiling and linking native Karyon C++20 architecture (v26.0 Master Habituation)...")
    try:
        karyon_cpp = load(
            name="karyon_cpp_ext_v26",
            sources=["karyon_core.cpp"],
            extra_cflags=["-O3", "-std=c++20"],
            verbose=False
        )
        sys.modules["karyon_cpp_ext_v26"] = karyon_cpp
        print("[C++ JIT] Native C++20 v26.0 Master Habituation architecture successfully compiled and initialized!")
    except Exception as e:
        if "already registered" in str(e):
            print("[C++ JIT] PyBind11 type registration conflict detected. Attempting fallback import...")
            for candidate_name in ["karyon_cpp_ext_v26", "karyon_cpp_ext_v25", "karyon_cpp_ext_v24"]:
                try:
                    mod = importlib.import_module(candidate_name)
                    if _is_valid_karyon_cpp_module(mod):
                        print(f"[C++ JIT] Fallback successful! Reusing '{candidate_name}'")
                        karyon_cpp = mod
                        break
                except Exception:
                    pass
            if karyon_cpp is None:
                raise e
        else:
            raise e

# Export all native C++ classes to Python interface
ByteTokenizer = karyon_cpp.ByteTokenizer
HomeostaticUnit = karyon_cpp.HomeostaticUnit
SensoryGateway = karyon_cpp.SensoryGateway
MotorGateway = karyon_cpp.MotorGateway
CausalByteReceptiveField = karyon_cpp.CausalByteReceptiveField
MultiScaleBytePyramidReceptiveField = karyon_cpp.MultiScaleBytePyramidReceptiveField
ParallelLogDecaySSDLayer = karyon_cpp.ParallelLogDecaySSDLayer
CalibratedParallelSSDCore = karyon_cpp.ParallelLogDecaySSDLayer # Alias for backward compatibility
CausalConvSwiGLUBlock = karyon_cpp.CausalConvSwiGLUBlock
ParallelSwiGLUBlock = karyon_cpp.CausalConvSwiGLUBlock # Alias for backward compatibility
EntropyAdaptiveBoundaryDetector = karyon_cpp.EntropyAdaptiveBoundaryDetector
CorticalStage = karyon_cpp.CorticalStage
PrecisionWeightedLPER = karyon_cpp.PrecisionWeightedLPER
FusedCascadedLaminarStack = karyon_cpp.FusedCascadedLaminarStack
DesaturatedHopfieldAttractorHead = karyon_cpp.DesaturatedHopfieldAttractorHead
LatentPredictor = karyon_cpp.LatentPredictor
TDFreeEnergyCritic = karyon_cpp.TDFreeEnergyCritic
BatchedEpisodicMemory = karyon_cpp.BatchedEpisodicMemory
VolitionalActionEvaluator = karyon_cpp.VolitionalActionEvaluator
LocalNeuromodulatedPlasticity = karyon_cpp.LocalNeuromodulatedPlasticity
