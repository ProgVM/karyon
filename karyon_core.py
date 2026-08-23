# karyon_core.py
import os
import torch
from torch.utils.cpp_extension import load

print("[C++ JIT] Compiling and linking native Karyon architecture (v16 Fractal 3-Tier Master)...")

karyon_cpp = load(
    name="karyon_cpp_ext_v16",
    sources=["karyon_core.cpp"],
    extra_cflags=["-O3", "-std=c++20"],
    verbose=False
)

# Export all 9 native C++ classes to Python interface
ByteTokenizer = karyon_cpp.ByteTokenizer
HomeostaticUnit = karyon_cpp.HomeostaticUnit
SensoryGateway = karyon_cpp.SensoryGateway
MotorGateway = karyon_cpp.MotorGateway
CausalByteReceptiveField = karyon_cpp.CausalByteReceptiveField
ThalamocorticalGatedFractalSSDCore = karyon_cpp.ThalamocorticalGatedFractalSSDCore
DesaturatedHopfieldAttractorHead = karyon_cpp.DesaturatedHopfieldAttractorHead
LatentPredictor = karyon_cpp.LatentPredictor
BatchedEpisodicMemory = karyon_cpp.BatchedEpisodicMemory

print("[C++ JIT] Native C++ v16 Fractal 3-Tier Master successfully initialized!")
