import ctypes
import numpy as np

class Obstacle(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("oid", ctypes.c_float)]

lib = ctypes.CDLL("./libexp_711_sovereign_metric_chaos.so")
lib.run_sovereign_metric_chaos_benchmark.argtypes = [
    ctypes.POINTER(Obstacle),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_double)
]
lib.run_sovereign_metric_chaos_benchmark.restype = None

# Load Bloodbath obstacle stream
with open("data/bloodbath_stream.bin", "rb") as f:
    raw_data = f.read()

num_obstacles = len(raw_data) // 12
obstacles_array = (Obstacle * num_obstacles)()

for i in range(num_obstacles):
    x, y, oid = np.frombuffer(raw_data[i*12:(i+1)*12], dtype=np.float32)
    obstacles_array[i] = Obstacle(x, y, oid)

print("=" * 80)
print("🌀 EXP-711: Sovereign Metric Expansion-Contraction & Stochastic Chaos Resonance")
print(f"Loaded {num_obstacles:,} physical hazards into GPU constant/stream memory")
print("=" * 80)

out_fe = ctypes.c_float()
out_order = ctypes.c_float()
out_evasion = ctypes.c_float()
out_symbols = ctypes.c_float()
out_scale = ctypes.c_float()
out_chaos = ctypes.c_float()
out_nav_x = ctypes.c_float()
out_time = ctypes.c_double()

lib.run_sovereign_metric_chaos_benchmark(
    obstacles_array,
    ctypes.c_int(num_obstacles),
    ctypes.c_int(128),
    ctypes.c_int(8192),
    ctypes.byref(out_fe),
    ctypes.byref(out_order),
    ctypes.byref(out_evasion),
    ctypes.byref(out_symbols),
    ctypes.byref(out_scale),
    ctypes.byref(out_chaos),
    ctypes.byref(out_nav_x),
    ctypes.byref(out_time)
)

print(f"GPU Execution Time     : {out_time.value:.2f} ms ({out_time.value/1000.0:.2f} s)")
print(f"Navigated Distance (X) : {out_nav_x.value:.2f} level units")
print(f"Hazard Evasion Rate %  : {out_evasion.value:.2f} %")
print(f"Discovered Concept Type: {out_symbols.value:.2f} / 4")
print(f"Closed-Loop Free Energy: {out_fe.value:.6f} nats")
print(f"Order Integrity (O_u)  : {out_order.value:.6f}")
print(f"Avg Spatial Metric G  : {out_scale.value:.6f} (Dynamic Scale Factor)")
print(f"Avg Chaos Energy (\u03be)  : {out_chaos.value:.6f} (Stochastic Resonance)")
print("=" * 80)
