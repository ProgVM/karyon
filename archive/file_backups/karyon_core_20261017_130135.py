"""
Karyon-CoRE v28.0 Sovereign Architecture (Master Python Gateway)
Thin, zero-overhead Python orchestration binding for libkaryon_core.so
"""

import ctypes
import os
import pathlib
from typing import Optional

_CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
_SO_PATH = _CURRENT_DIR / "libkaryon_core.so"

if not _SO_PATH.exists():
    raise FileNotFoundError(f"Native core library not found at {_SO_PATH}. Please compile with g++.")

_lib = ctypes.CDLL(str(_SO_PATH))

# Instance API
_lib.karyon_create.argtypes = []
_lib.karyon_create.restype = ctypes.c_void_p

_lib.karyon_destroy.argtypes = [ctypes.c_void_p]
_lib.karyon_destroy.restype = None

_lib.karyon_step_instance.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]
_lib.karyon_step_instance.restype = ctypes.c_float

_lib.karyon_get_free_energy_instance.argtypes = [ctypes.c_void_p]
_lib.karyon_get_free_energy_instance.restype = ctypes.c_float

_lib.karyon_reset_instance.argtypes = [ctypes.c_void_p]
_lib.karyon_reset_instance.restype = None


class SovereignKaryonCore:
    def __init__(self):
        self._handle = _lib.karyon_create()
        if not self._handle:
            raise RuntimeError("Failed to allocate native SovereignContinuumCore instance.")

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            _lib.karyon_destroy(self._handle)
            self._handle = None

    def reset(self) -> None:
        if self._handle:
            _lib.karyon_reset_instance(self._handle)

    def step(self, error: float, error_rate: float, dt: float = 0.02) -> float:
        """
        Executes single exact Lie-flow perception-action step.
        Returns generated restorative boundary torque/action.
        """
        if not self._handle:
            raise RuntimeError("Core instance was deallocated.")
        return float(_lib.karyon_step_instance(self._handle, ctypes.c_float(error), ctypes.c_float(error_rate), ctypes.c_float(dt)))

    @property
    def free_energy(self) -> float:
        if not self._handle:
            return 0.0
        return float(_lib.karyon_get_free_energy_instance(self._handle))
