# karyon_agent.py — High-Performance Sovereign Python Gateway for Karyon Native Core v7.7
# Standard: KEP v20.0 Sovereign Master (Strict Principles 2 to 27 Compliant)
# Author: Bazilevs (ProgVM) & Autonomous Lead Cyberneticist

import os
import ctypes
import numpy as np
from typing import Tuple, Dict, Optional

class KaryonNativeWrapper:
    """Zero-overhead ctypes wrapper around libkaryon_native.so (C++20 Core v7.7)."""
    def __init__(self, so_path: str = "./libkaryon_native.so",
                 sensory_dim: int = 256, latent_dim: int = 32, action_dim: int = 4,
                 min_horizon_k: int = 1, max_horizon_k: int = 5,
                 plasticity_rate: float = 0.04, aversive_weight: float = 6.0,
                 peril_threshold: float = 0.55):
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"Native core library not found at: {so_path}")

        self.sensory_dim = sensory_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.lib = ctypes.CDLL(so_path)

        # C-ABI signatures
        self.lib.karyon_cabi_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        self.lib.karyon_cabi_create.restype = ctypes.c_void_p

        self.lib.karyon_cabi_create_custom.argtypes = [
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]
        self.lib.karyon_cabi_create_custom.restype = ctypes.c_void_p

        self.lib.karyon_cabi_destroy.argtypes = [ctypes.c_void_p]
        self.lib.karyon_cabi_destroy.restype = None

        self.lib.karyon_cabi_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_size_t)
        ]
        self.lib.karyon_cabi_step.restype = None

        self.lib.karyon_cabi_step_with_valence.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_size_t)
        ]
        self.lib.karyon_cabi_step_with_valence.restype = None

        self.handle = self.lib.karyon_cabi_create_custom(
            sensory_dim, latent_dim, action_dim,
            min_horizon_k, max_horizon_k,
            plasticity_rate, aversive_weight, peril_threshold
        )

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            self.lib.karyon_cabi_destroy(self.handle)
            self.handle = None

    def step(self,
             sensory_curr: np.ndarray,
             sensory_target: np.ndarray,
             sensory_next: np.ndarray,
             external_valence: Optional[float] = None,
             action_init: np.ndarray = None) -> Tuple[np.ndarray, float, float, int]:
        """
        Executes a closed-loop step through C++20 Core v7.7:
        Returns: (action_taken, free_energy, noradrenaline, active_primitives_count)
        """
        s_curr = np.ascontiguousarray(sensory_curr, dtype=np.float32)
        s_tgt = np.ascontiguousarray(sensory_target, dtype=np.float32)
        s_nxt = np.ascontiguousarray(sensory_next, dtype=np.float32)

        if action_init is None:
            act = np.zeros(self.action_dim, dtype=np.float32)
        else:
            act = np.ascontiguousarray(action_init, dtype=np.float32)

        fe = ctypes.c_float(0.0)
        na = ctypes.c_float(0.0)
        prims = ctypes.c_size_t(0)

        if external_valence is not None:
            self.lib.karyon_cabi_step_with_valence(
                self.handle,
                s_curr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s_tgt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s_nxt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_float(external_valence),
                act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(fe),
                ctypes.byref(na),
                ctypes.byref(prims)
            )
        else:
            self.lib.karyon_cabi_step(
                self.handle,
                s_curr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s_tgt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s_nxt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(fe),
                ctypes.byref(na),
                ctypes.byref(prims)
            )

        return act, float(fe.value), float(na.value), int(prims.value)


class CoREAgent:
    """
    Sovereign Karyon-CoRE Agent v7.7.
    Universal Continuous Multimodal Substrate with Zero PyTorch overhead.
    """
    def __init__(self, so_path: str = "./libkaryon_native.so",
                 sensory_dim: int = 256, latent_dim: int = 32, action_dim: int = 4,
                 min_horizon_k: int = 1, max_horizon_k: int = 5,
                 plasticity_rate: float = 0.04, aversive_weight: float = 6.0,
                 peril_threshold: float = 0.55):
        self.sensory_dim = sensory_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.native_core = KaryonNativeWrapper(
            so_path, sensory_dim, latent_dim, action_dim,
            min_horizon_k, max_horizon_k,
            plasticity_rate, aversive_weight, peril_threshold
        )

    def step_embodied(self,
                      sensory_curr: np.ndarray,
                      sensory_target: np.ndarray,
                      sensory_next: np.ndarray,
                      external_valence: Optional[float] = None,
                      action_init: np.ndarray = None) -> Dict:
        """Single closed-loop embodied active inference step with predictive peril avoidance."""
        act, fe, na, prims = self.native_core.step(
            sensory_curr, sensory_target, sensory_next, external_valence, action_init
        )
        return {
            "action": act,
            "free_energy": fe,
            "noradrenaline": na,
            "active_primitives": prims
        }
