# karyon_core.py
"""
===============================================================================
KARYON CORE C++20 LIBTORCH COMPILATION & PYTHON BRIDGE v23.0 MASTER
Python as Client, C++20 as Engine (KEP Principle 1)
Universal Multimodal & Cross-Modal State-Space Cognitive Engine (EXP-87 Validated)
===============================================================================
"""
import os
import torch
from torch.utils.cpp_extension import load

print("[C++ JIT] Compiling and linking native Karyon C++20 architecture (v23.0 Master Universal Multimodal)...")

karyon_cpp = load(
    name="karyon_cpp_ext_v23",
    sources=["karyon_core.cpp"],
    extra_cflags=["-O3", "-std=c++20"],
    verbose=False
)

# Export all 15 native C++ classes to Python interface
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
DesaturatedHopfieldAttractorHead = karyon_cpp.DesaturatedHopfieldAttractorHead
LatentPredictor = karyon_cpp.LatentPredictor
BatchedEpisodicMemory = karyon_cpp.BatchedEpisodicMemory

print("[C++ JIT] Native C++20 v23.0 Master Universal Multimodal architecture successfully compiled and initialized!")
