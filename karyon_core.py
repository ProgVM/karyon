# karyon_core.py
import os
import torch
from torch.utils.cpp_extension import load

print("[C++ JIT] Compiling and linking native Karyon architecture (v6)...")

karyon_cpp = load(
    name="karyon_cpp_ext_v6",
    sources=["karyon_core.cpp"],
    extra_cflags=["-O3", "-std=c++20"],
    verbose=False
)

# Export native C++ classes to Python interface
ByteTokenizer = karyon_cpp.ByteTokenizer
HomeostaticUnit = karyon_cpp.HomeostaticUnit
SensoryGateway = karyon_cpp.SensoryGateway
MotorGateway = karyon_cpp.MotorGateway
DynamicRecurrentCore = karyon_cpp.DynamicRecurrentCore
LatentPredictor = karyon_cpp.LatentPredictor
BatchedEpisodicMemory = karyon_cpp.BatchedEpisodicMemory

print("[C++ JIT] Native C++ core architecture successfully initialized!")
