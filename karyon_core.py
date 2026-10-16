"""
Karyon-CoRE v28.0 Sovereign Architecture (Master Python Gateway)
Thin, zero-overhead Python orchestration binding for libkaryon_core.so
"""

import ctypes
import os
import pathlib
from typing import Tuple

_CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
_SO_PATH = _CURRENT_DIR / "libkaryon_core.so"

if not _SO_PATH.exists():
    raise FileNotFoundError(f"Native core library not found at {_SO_PATH}. Please compile with g++.")

_lib = ctypes.CDLL(str(_SO_PATH))

_lib.karyon_init.argtypes = []
_lib.karyon_init.restype = None

_lib.karyon_step.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
_lib.karyon_step.restype = ctypes.c_float

_lib.karyon_get_free_energy.argtypes = []
_lib.karyon_get_free_energy.restype = ctypes.c_float

_lib.karyon_reset.argtypes = []
_lib.karyon_reset.restype = None

class SovereignKaryonCore:
    def __init__(self):
        _lib.karyon_init()

    def reset(self) -> None:
        _lib.karyon_reset()

    def step(self, error: float, error_rate: float, dt: float = 0.02) -> float:
        """
        Executes single exact Lie-flow perception-action step.
        Returns generated restorative boundary torque/action.
        """
        return _lib.karyon_step(ctypes.c_float(error), ctypes.c_float(error_rate), ctypes.c_float(dt))

    @property
    def free_energy(self) -> float:
        return _lib.karyon_get_free_energy()
