# karyon_core.py
"""
===============================================================================
KARYON CORE C++20 LIBTORCH COMPILATION & PYTHON BRIDGE v24.0 MASTER
Python as Client, C++20 as Engine (KEP Principle 1)
Universal Multimodal & Cross-Modal State-Space Cognitive Engine (EXP-90 Validated)
===============================================================================
"""
import os
import sys
import importlib
import torch
from torch.utils.cpp_extension import load

# Step 1: Check if any real C++ module with ByteTokenizer is already loaded in sys.modules
karyon_cpp = None
for mod_name, mod in list(sys.modules.items()):
    # Skip torch.ops/classes/dynamo to avoid dummy attributes
    if mod is not None and not any(x in mod_name for x in ["torch.ops", "torch.classes", "torch._dynamo"]):
        try:
            val = getattr(mod, "ByteTokenizer", None)
            if val is not None and isinstance(val, type):
                print(f"[C++ JIT] Reusing already loaded C++ module: '{mod_name}'")
                karyon_cpp = mod
                break
        except Exception:
            pass

# Step 2: Try importing candidate names directly
if karyon_cpp is None:
    # Add common build paths to sys.path
    for path in ["/kaggle/working/karyon/build/karyon_core_jit", "/root/.cache/torch_extensions/py312_cu128/karyon_cpp_ext_v24"]:
        if os.path.exists(path) and path not in sys.path:
            sys.path.append(path)
            
    for candidate_name in ["karyon_cpp_ext_v24", "karyon_core_ext", "karyon_core_ext_v1"]:
        try:
            mod = importlib.import_module(candidate_name)
            val = getattr(mod, "ByteTokenizer", None)
            if val is not None and isinstance(val, type):
                print(f"[C++ JIT] Successfully imported existing compiled module: '{candidate_name}'")
                karyon_cpp = mod
                break
        except Exception:
            pass

# Step 3: Compile and load if not found
if karyon_cpp is None:
    print("[C++ JIT] Compiling and linking native Karyon C++20 architecture (v24.0 Master Universal Multimodal)...")
    try:
        karyon_cpp = load(
            name="karyon_cpp_ext_v24",
            sources=["karyon_core.cpp"],
            extra_cflags=["-O3", "-std=c++20"],
            verbose=False
        )
        sys.modules["karyon_cpp_ext_v24"] = karyon_cpp
        print("[C++ JIT] Native C++20 v24.0 Master Universal Multimodal architecture successfully compiled and initialized!")
    except Exception as e:
        # If compilation failed because of "already registered", try to find any loaded .so or fallback to karyon_core_ext_v1
        if "already registered" in str(e):
            print("[C++ JIT] PyBind11 type registration conflict detected. Attempting fallback import...")
            for candidate_name in ["karyon_core_ext_v1", "karyon_core_ext"]:
                try:
                    mod = importlib.import_module(candidate_name)
                    val = getattr(mod, "ByteTokenizer", None)
                    if val is not None and isinstance(val, type):
                        print(f"[C++ JIT] Fallback successful! Reusing '{candidate_name}'")
                        karyon_cpp = mod
                        break
                except Exception:
                    pass
            if karyon_cpp is None:
                raise e
        else:
            raise e

# Export all 16 native C++ classes to Python interface
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
