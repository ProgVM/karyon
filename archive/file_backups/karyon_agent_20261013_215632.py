"""
karyon_agent.py — Pure Zero-Torch Sovereign Autonomous Cognitive Agent Runtime v7.4.
Created for Bazilevs (ProgVM) | Standard: KEP v20.0 Sovereign Master (Principles 2 to 27 Compliant).

Operates with ZERO dependencies on torch or libtorch.
Communicates directly with the compiled native C++20 engine (`libkaryon_native.so`)
via zero-overhead C-ABI ctypes bindings.
Supports:
- 256D Optical Pixel Vision
- 32D Laminar Compressed Substrate
- Autonomous in-flight Morphogenesis Sprouting
- 100% Vectorized AVX2-FMA Plasticity Engine
"""

import os
import ctypes
import numpy as np
from typing import Dict, List, Tuple


class KaryonNativeWrapper:
    """Zero-overhead ctypes wrapper around libkaryon_native.so (C++20 Core v7.4)."""
    def __init__(self, so_path: str = "./libkaryon_native.so",
                 sensory_dim: int = 256, latent_dim: int = 32, action_dim: int = 4):
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"Native core library not found at: {so_path}")

        self.sensory_dim = sensory_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.lib = ctypes.CDLL(so_path)

        # C-ABI signatures
        self.lib.karyon_cabi_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        self.lib.karyon_cabi_create.restype = ctypes.c_void_p

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

        self.handle = self.lib.karyon_cabi_create(sensory_dim, latent_dim, action_dim)

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            self.lib.karyon_cabi_destroy(self.handle)
            self.handle = None

    def step(self,
             sensory_curr: np.ndarray,
             sensory_target: np.ndarray,
             sensory_next: np.ndarray,
             action_init: np.ndarray = None) -> Tuple[np.ndarray, float, float, int]:
        """
        Executes a closed-loop step through C++20 Core v7.4:
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
    Sovereign Karyon-CoRE Agent v7.4.
    Universal Continuous Multimodal Substrate with Zero PyTorch overhead.
    """
    def __init__(self, so_path: str = "./libkaryon_native.so",
                 sensory_dim: int = 256, latent_dim: int = 32, action_dim: int = 4):
        self.sensory_dim = sensory_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.native_core = KaryonNativeWrapper(so_path, sensory_dim, latent_dim, action_dim)

    def step_embodied(self,
                      sensory_curr: np.ndarray,
                      sensory_target: np.ndarray,
                      sensory_next: np.ndarray,
                      action_init: np.ndarray = None) -> Dict:
        """Single closed-loop embodied active inference step."""
        act, fe, na, prims = self.native_core.step(
            sensory_curr, sensory_target, sensory_next, action_init
        )
        return {
            "action": act,
            "free_energy": fe,
            "noradrenaline": na,
            "active_primitives": prims
        }


if __name__ == "__main__":
    import time
    print("=== Testing Sovereign Zero-Torch CoREAgent v7.4 ===")
    agent = CoREAgent(sensory_dim=256, latent_dim=32, action_dim=4)

    s_c = np.random.randn(256).astype(np.float32)
    s_t = np.random.randn(256).astype(np.float32)
    s_n = np.random.randn(256).astype(np.float32)

    t0 = time.time()
    steps = 10000
    for _ in range(steps):
        res = agent.step_embodied(s_c, s_t, s_n)
    t1 = time.time()

    elapsed = t1 - t0
    rate = steps / elapsed
    print(f"Executed {steps} steps in {elapsed:.3f}s -> {rate:.1f} steps/s!")
    print(f"Telemetry: FE={res['free_energy']:.5f}, NA={res['noradrenaline']:.3f}, Primitives={res['active_primitives']}")
