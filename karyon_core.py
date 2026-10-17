"""
Karyon-CoRE v29.0 Sovereign Python Orchestrator
Thin orchestration wrapper over libkaryon_core.so native engine.
"""

import ctypes
import os
from typing import Tuple

_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libkaryon_core.so")
if not os.path.exists(_lib_path):
    raise FileNotFoundError(f"Native library not found at {_lib_path}. Run `make` first.")

_lib = ctypes.CDLL(_lib_path)

# C-ABI function signatures
_lib.karyon_create.restype = ctypes.c_void_p
_lib.karyon_create.argtypes = []

_lib.karyon_destroy.restype = None
_lib.karyon_destroy.argtypes = [ctypes.c_void_p]

_lib.karyon_step_instance.restype = ctypes.c_float
_lib.karyon_step_instance.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]

_lib.karyon_get_free_energy_instance.restype = ctypes.c_float
_lib.karyon_get_free_energy_instance.argtypes = [ctypes.c_void_p]

_lib.karyon_get_expected_free_energy_instance.restype = ctypes.c_float
_lib.karyon_get_expected_free_energy_instance.argtypes = [ctypes.c_void_p]

_lib.karyon_get_active_modes_instance.restype = ctypes.c_size_t
_lib.karyon_get_active_modes_instance.argtypes = [ctypes.c_void_p]

_lib.karyon_reset_instance.restype = None
_lib.karyon_reset_instance.argtypes = [ctypes.c_void_p]


class SovereignCoreAgent:
    """
    Karyon-CoRE Sovereign Agent.
    Operates on a continuous physical tensor wavefield with autopoietic morphogenesis,
    evaluating both Current Free Energy F_t and Expected Free Energy G_{t+1}.
    """
    def __init__(self):
        self._handle = _lib.karyon_create()
        if not self._handle:
            raise RuntimeError("Failed to allocate native karyon core instance.")

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            _lib.karyon_destroy(self._handle)
            self._handle = None

    def step(self, error: float, error_rate: float, dt: float = 0.01) -> float:
        """Step the non-blind autopoietic continuum and return synthesized motor traction."""
        return _lib.karyon_step_instance(self._handle, ctypes.c_float(error), ctypes.c_float(error_rate), ctypes.c_float(dt))

    @property
    def free_energy(self) -> float:
        """Get current Variational Free Energy F_t."""
        return _lib.karyon_get_free_energy_instance(self._handle)

    @property
    def expected_free_energy(self) -> float:
        """Get expected lookahead Free Energy G_{t+1}."""
        return _lib.karyon_get_expected_free_energy_instance(self._handle)

    @property
    def active_modes(self) -> int:
        """Get current number of active evolved physical modes."""
        return _lib.karyon_get_active_modes_instance(self._handle)

    def reset(self):
        """Reset core to primordial state."""
        _lib.karyon_reset_instance(self._handle)
