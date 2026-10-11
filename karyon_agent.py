"""
karyon_agent.py — Pure Zero-Torch Sovereign Autonomous Cognitive Agent Runtime.
Created for Bazilevs (ProgVM) | Standard: KEP v16.0 Sovereign Master.

Operates with ZERO dependencies on torch or libtorch.
Communicates directly with the compiled native C++20 engine (`libkaryon_native.so`)
via zero-overhead C-ABI ctypes bindings, or falls back to pure vector arithmetic.
"""

import os
import ctypes
import numpy as np
from typing import Dict, Tuple


class KaryonNativeWrapper:
    """Zero-overhead ctypes wrapper around libkaryon_native.so (C++20)."""
    def __init__(self, so_path: str = "./libkaryon_native.so",
                 dim: int = 258, num_heads: int = 8, dim_k: int = 64, dim_v: int = 64):
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"Native core library not found at: {so_path}")

        self.lib = ctypes.CDLL(so_path)

        # Declare signatures
        self.lib.karyon_native_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        self.lib.karyon_native_create.restype = ctypes.c_void_p

        self.lib.karyon_native_destroy.argtypes = [ctypes.c_void_p]
        self.lib.karyon_native_destroy.restype = None

        self.lib.karyon_native_reset.argtypes = [ctypes.c_void_p]
        self.lib.karyon_native_reset.restype = None

        self.lib.karyon_native_step.argtypes = [
            ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.karyon_native_step.restype = None

        self.lib.karyon_native_sample.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_float]
        self.lib.karyon_native_sample.restype = ctypes.c_uint8

        self.lib.karyon_native_save_kcore.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.karyon_native_save_kcore.restype = ctypes.c_size_t

        self.lib.karyon_native_load_kcore.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.karyon_native_load_kcore.restype = ctypes.c_bool

        self.handle = self.lib.karyon_native_create(dim, num_heads, dim_k, dim_v)

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            self.lib.karyon_native_destroy(self.handle)
            self.handle = None

    def reset(self):
        self.lib.karyon_native_reset(self.handle)

    def step(self, byte_in: int, byte_target: int) -> Tuple[float, int, int, float]:
        loss = ctypes.c_float(0.0)
        pred = ctypes.c_uint8(0)
        turns = ctypes.c_size_t(0)
        entropy = ctypes.c_float(0.0)

        self.lib.karyon_native_step(
            self.handle,
            ctypes.c_uint8(byte_in),
            ctypes.c_uint8(byte_target),
            ctypes.byref(loss),
            ctypes.byref(pred),
            ctypes.byref(turns),
            ctypes.byref(entropy)
        )
        return float(loss.value), int(pred.value), int(turns.value), float(entropy.value)

    def save_kcore(self, filepath: str):
        # Query buffer size
        needed = self.lib.karyon_native_save_kcore(self.handle, None, 0)
        buf = ctypes.create_string_buffer(needed)
        self.lib.karyon_native_save_kcore(self.handle, buf, needed)
        with open(filepath, "wb") as f:
            f.write(buf.raw)

    def load_kcore(self, filepath: str) -> bool:
        with open(filepath, "rb") as f:
            data = f.read()
        buf = ctypes.create_string_buffer(data, len(data))
        return bool(self.lib.karyon_native_load_kcore(self.handle, buf, len(data)))


class CoREAgent:
    """
    Sovereign Karyon-CoRE Agent.
    Operates at raw byte level V=258 with zero PyTorch dependencies.
    """
    def __init__(self, so_path: str = "./libkaryon_native.so",
                 dim: int = 258, num_heads: int = 8, dim_k: int = 64, dim_v: int = 64):
        self.dim = dim
        self.num_heads = num_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.native_core = KaryonNativeWrapper(so_path, dim, num_heads, dim_k, dim_v)

    def reset_state(self):
        self.native_core.reset()

    def process_byte(self, byte_in: int, byte_target: int) -> Dict:
        loss, pred, turns, entropy = self.native_core.step(byte_in, byte_target)
        return {
            "loss": loss,
            "pred": pred,
            "is_correct": (pred == byte_target),
            "turns_taken": turns,
            "entropy": entropy
        }

    def perceive_stream(self, stream_bytes: bytes) -> Dict:
        """Processes an online streaming sequence of bytes (Single-Pass N=1)."""
        losses = []
        correct = 0
        total_turns = 0

        for t in range(len(stream_bytes) - 1):
            res = self.process_byte(stream_bytes[t], stream_bytes[t + 1])
            losses.append(res["loss"])
            if res["is_correct"]:
                correct += 1
            total_turns += res["turns_taken"]

        return {
            "mean_loss": float(np.mean(losses)),
            "accuracy": (correct / max(1, len(stream_bytes) - 1)) * 100.0,
            "avg_turns": total_turns / max(1, len(stream_bytes) - 1),
            "total_bytes": len(stream_bytes)
        }

    def save_soul(self, filepath: str = "karyon_soul.kcore"):
        self.native_core.save_kcore(filepath)

    def load_soul(self, filepath: str = "karyon_soul.kcore") -> bool:
        return self.native_core.load_kcore(filepath)


if __name__ == "__main__":
    print("=== Testing Sovereign Zero-Torch CoREAgent ===")
    agent = CoREAgent()
    sample_text = b"Hello world! Sovereign autopoiesis without PyTorch."
    stats = agent.perceive_stream(sample_text)
    print(f"Stats: Loss={stats['mean_loss']:.4f}, Accuracy={stats['accuracy']:.2f}%, Turns={stats['avg_turns']:.2f}")
    agent.save_soul("test_soul.kcore")
    print(f"Saved test_soul.kcore successfully ({os.path.getsize('test_soul.kcore')} bytes).")
