"""
Karyon Multiverse Wavefield Python Module (karyon_multiverse.py)
Master Specification: KEP v30.0 Substrate Binding
Author: Bazilevs (ProgVM) & Lead Cyberneticist

Exposes the native CUDA C++20 Multiverse Wavefield Engine directly to Python.
Eliminates code duplication in experiments.
"""

import ctypes
import os
import struct
import numpy as np

class Obstacle(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("oid", ctypes.c_int32)
    ]

class MultiverseSimulationResult(ctypes.Structure):
    _fields_ = [
        ("active_universes", ctypes.c_int32),
        ("gpu_execution_time_ms", ctypes.c_double),
        ("mean_navigated_distance_x", ctypes.c_float),
        ("hazard_evasion_rate_pct", ctypes.c_float),
        ("mean_discovered_symbols", ctypes.c_float),
        ("closed_loop_free_energy_F", ctypes.c_float),
        ("intrinsic_order_integrity_O", ctypes.c_float)
    ]

class KaryonMultiverseEngine:
    def __init__(self, so_path=None):
        if so_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            so_path = os.path.join(base_dir, "libkaryon_multiverse_core.so")
            if not os.path.exists(so_path):
                so_path = "libkaryon_multiverse_core.so"
        
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"Shared library {so_path} not found. Please compile karyon_multiverse_core.cu first.")
            
        self.lib = ctypes.CDLL(so_path)
        self.lib.run_multiverse_simulation.argtypes = [
            ctypes.POINTER(Obstacle),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        self.lib.run_multiverse_simulation.restype = MultiverseSimulationResult

    @staticmethod
    def load_stream_data(bin_path="data/bloodbath_stream.bin"):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Obstacle stream file {bin_path} not found.")
            
        obstacles = []
        with open(bin_path, "rb") as f:
            while True:
                chunk = f.read(12)
                if len(chunk) < 12:
                    break
                x, y, oid = struct.unpack("ffi", chunk)
                obs = Obstacle()
                obs.x = float(x)
                obs.y = float(y)
                obs.oid = int(oid)
                obstacles.append(obs)
                
        arr_type = Obstacle * len(obstacles)
        return arr_type(*obstacles), len(obstacles)

    def simulate(self, obstacles_array, total_obstacles, num_universes=128, ticks=8192):
        return self.lib.run_multiverse_simulation(
            obstacles_array,
            ctypes.c_int(total_obstacles),
            ctypes.c_int(num_universes),
            ctypes.c_int(ticks)
        )
