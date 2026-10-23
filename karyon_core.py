"""
Karyon-CoRE v45.0 Sovereign Python Orchestrator
Thin orchestration wrapper over libkaryon_core.so native C++20 engine.
Standard: KEP v30.0 (Principles 1, 2, 3, 4, 5, 6, 8, 27, 28, 29, 30)
"""

import ctypes
import os
from typing import Dict, Any

_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libkaryon_core.so")
if not os.path.exists(_lib_path):
    raise FileNotFoundError(f"Native library not found at {_lib_path}. Run `g++ -O3 -std=c++20 -mavx2 -mfma -shared -fPIC karyon_core.cpp -o libkaryon_core.so` first.")

_lib = ctypes.CDLL(_lib_path)

# PurePhysicsTelemetry struct mapping
class PurePhysicsTelemetry(ctypes.Structure):
    _fields_ = [
        ("active_modes", ctypes.c_size_t),
        ("phase_dim", ctypes.c_size_t),
        ("free_energy", ctypes.c_float),
        ("dt_eff", ctypes.c_float),
        ("refractory_strength", ctypes.c_float),
        ("refractory_recovery", ctypes.c_float),
        ("avg_stiffness", ctypes.c_float),
        ("avg_damping", ctypes.c_float),
        ("coupling_density", ctypes.c_float),
        ("mean_coupling_weight", ctypes.c_float),
        ("hippocampal_gain", ctypes.c_float),
        ("endogenous_beta", ctypes.c_float),
        ("stored_phase_traces", ctypes.c_size_t),
        ("phase_retrieval_resonance", ctypes.c_float),
        ("rollout_depth", ctypes.c_size_t),
        ("autopoietic_stress", ctypes.c_float),
        ("shannon_entropy", ctypes.c_float),
    ]

# C-ABI function signatures
_lib.karyon_create.restype = ctypes.c_void_p
_lib.karyon_create.argtypes = []

_lib.karyon_destroy.restype = None
_lib.karyon_destroy.argtypes = [ctypes.c_void_p]

_lib.karyon_step_instance.restype = ctypes.c_uint8
_lib.karyon_step_instance.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_float]

_lib.karyon_assimilate_instance.restype = None
_lib.karyon_assimilate_instance.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float]

_lib.karyon_get_free_energy_instance.restype = ctypes.c_float
_lib.karyon_get_free_energy_instance.argtypes = [ctypes.c_void_p]

_lib.karyon_get_active_modes_instance.restype = ctypes.c_size_t
_lib.karyon_get_active_modes_instance.argtypes = [ctypes.c_void_p]

_lib.karyon_reset_instance.restype = None
_lib.karyon_reset_instance.argtypes = [ctypes.c_void_p]

_lib.karyon_get_telemetry_instance.restype = None
_lib.karyon_get_telemetry_instance.argtypes = [ctypes.c_void_p, ctypes.POINTER(PurePhysicsTelemetry)]


class SovereignCoreAgent:
    """
    Karyon-CoRE Sovereign Agent Python Wrapper.
    """
    def __init__(self):
        self._handle = _lib.karyon_create()
        if not self._handle:
            raise RuntimeError("Failed to instantiate Karyon C++ Sovereign Organism")

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            _lib.karyon_destroy(self._handle)
            self._handle = None

    def perceive_and_decide(self, input_channel: int, dt: float = 0.01) -> int:
        """
        Step perception and wavefield decision loop.
        """
        return int(_lib.karyon_step_instance(self._handle, ctypes.c_uint8(input_channel), ctypes.c_float(dt)))

    def assimilate_experience(self, input_channel: int, valence: float, magnitude: float, target_hint: int = -1, lr: float = 0.05):
        """
        Assimilate feedback and adapt continuous physical substrate.
        """
        _lib.karyon_assimilate_instance(self._handle, ctypes.c_uint8(input_channel), ctypes.c_float(valence), ctypes.c_float(magnitude), ctypes.c_int(target_hint), ctypes.c_float(lr))

    def reset(self):
        _lib.karyon_reset_instance(self._handle)

    def get_telemetry(self) -> Dict[str, Any]:
        t = PurePhysicsTelemetry()
        _lib.karyon_get_telemetry_instance(self._handle, ctypes.byref(t))
        return {
            "active_modes": t.active_modes,
            "phase_dim": t.phase_dim,
            "free_energy": t.free_energy,
            "dt_eff": t.dt_eff,
            "refractory_strength": t.refractory_strength,
            "refractory_recovery": t.refractory_recovery,
            "avg_stiffness": t.avg_stiffness,
            "avg_damping": t.avg_damping,
            "coupling_density": t.coupling_density,
            "mean_coupling_weight": t.mean_coupling_weight,
            "hippocampal_gain": t.hippocampal_gain,
            "endogenous_beta": t.endogenous_beta,
            "stored_phase_traces": t.stored_phase_traces,
            "phase_retrieval_resonance": t.phase_retrieval_resonance,
            "rollout_depth": t.rollout_depth,
            "autopoietic_stress": t.autopoietic_stress,
            "shannon_entropy": t.shannon_entropy,
        }
