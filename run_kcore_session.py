# run_kcore_session.py
import ctypes
import os
import torch

kcore_path = "karyon_soul.kcore"

# 1. Fault-tolerant container check (Auto-initialize via init_priors if missing)
if not os.path.exists(kcore_path):
    print(f"[Host] Container '{kcore_path}' not found! Automatically building base model via init_priors...")
    from init_priors import initialize_priors
    initialize_priors(recreate=True, filepath=kcore_path)

# 2. Preload PyTorch shared libraries globally to resolve dynamic symbols
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
for lib_name in ["libtorch.so", "libtorch_cpu.so", "libtorch_python.so"]:
    lib_path = os.path.join(torch_lib_dir, lib_name)
    if os.path.exists(lib_path):
        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)

# 3. Load C-ABI Dynamic Library
so_path = "./libkaryon_runtime.so"
if not os.path.exists(so_path):
    raise FileNotFoundError(f"Shared library {so_path} not found. Run g++ compilation first.")

lib = ctypes.CDLL(so_path)

# 4. Configure C-ABI Signatures
lib.karyon_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.karyon_load.restype = ctypes.c_void_p

lib.karyon_perceive_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float]
lib.karyon_perceive_text.restype = None

lib.karyon_step.argtypes = [ctypes.c_void_p]
lib.karyon_step.restype = None

lib.karyon_express_text.argtypes = [ctypes.c_void_p]
lib.karyon_express_text.restype = ctypes.c_char_p

lib.karyon_get_somatic_state.argtypes = [
    ctypes.c_void_p, 
    ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float)
]
lib.karyon_get_somatic_state.restype = None

lib.karyon_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.karyon_save.restype = None

lib.karyon_free.argtypes = [ctypes.c_void_p]
lib.karyon_free.restype = None

# 5. Initialize Karyon Entity from Container
kcore_bytes = kcore_path.encode('utf-8')
print(f"[Host] Initializing autonomous entity from container '{kcore_path}'...")
entity = lib.karyon_load(kcore_bytes, b"cuda")

if not entity:
    print("[Host] CUDA context fallback -> loading on CPU...")
    entity = lib.karyon_load(kcore_bytes, b"cpu")

if not entity:
    raise RuntimeError("Failed to initialize Karyon Entity from container!")

# 6. Perception, Execution & Expression
user_message = b"Hello Karyon, execute step within your continuous state."
lib.karyon_perceive_text(entity, user_message, 1.0)
lib.karyon_step(entity)

e, h, a = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
lib.karyon_get_somatic_state(entity, ctypes.byref(e), ctypes.byref(h), ctypes.byref(a))
print(f"[Somatic State] Energy = {e.value:.4f} | Health = {h.value:.4f} | Arousal (NA) = {a.value:.4f}")

response = lib.karyon_express_text(entity)
print(f"Karyon -> {response.decode('utf-8')}")

# 7. Persist Updated State Back to Container
lib.karyon_save(entity, kcore_bytes)
lib.karyon_free(entity)
print("[Host] Entity successfully saved and detached.")
