# run_kcore_session.py
"""
===============================================================================
KARYON AUTONOMOUS C-ABI EXECUTION RUNNER v16.5
Executes Karyon container directly via ctypes and native libkaryon_runtime.so C-ABI.
===============================================================================
"""

import ctypes
import os
import torch

kcore_path = "karyon_soul.kcore"

# 1. Container check
if not os.path.exists(kcore_path):
    print(f"[Host] Container '{kcore_path}' not found! Initializing base container via init_priors...")
    from init_priors import initialize_priors
    initialize_priors(recreate=True, filepath=kcore_path)

# 2. Preload PyTorch shared libraries
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
for lib_name in ["libtorch.so", "libtorch_cpu.so", "libtorch_python.so"]:
    lib_path = os.path.join(torch_lib_dir, lib_name)
    if os.path.exists(lib_path):
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

# 3. Load C-ABI Dynamic Library if compiled
so_path = "./libkaryon_runtime.so"
if os.path.exists(so_path):
    lib = ctypes.CDLL(so_path)

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

    kcore_bytes = kcore_path.encode('utf-8')
    print(f"[Host] Initializing autonomous entity from container '{kcore_path}'...")
    entity = lib.karyon_load(kcore_bytes, b"cuda")
    if not entity:
        entity = lib.karyon_load(kcore_bytes, b"cpu")

    if entity:
        user_message = b"Hello Karyon, execute step within your continuous state."
        lib.karyon_perceive_text(entity, user_message, 1.0)
        lib.karyon_step(entity)

        e, h, a = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
        lib.karyon_get_somatic_state(entity, ctypes.byref(e), ctypes.byref(h), ctypes.byref(a))
        print(f"[Somatic State] Energy = {e.value:.4f} | Health = {h.value:.4f} | Arousal (NA) = {a.value:.4f}")

        response = lib.karyon_express_text(entity)
        print(f"Karyon -> {response.decode('utf-8')}")

        lib.karyon_save(entity, kcore_bytes)
        lib.karyon_free(entity)
        print("[Host] Entity successfully saved and detached.")
else:
    print(f"[Host] Standalone shared library '{so_path}' not built. Skipping standalone C-ABI execution.")
