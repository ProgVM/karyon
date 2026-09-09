"""
Karyon-CoRE Hardware Engine & Universal Device Abstraction Layer (v25.0 Master)
Provides dynamic, hardware-agnostic execution across CPU, NVIDIA CUDA GPUs, and Google Cloud / Kaggle TPUs (PyTorch-XLA).
Fully KEP Rule #10 Compliant.
"""

import os
import gc
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch

logger = logging.getLogger("karyon_hardware")

# Optional PyTorch-XLA (TPU) imports
TPU_AVAILABLE = False
xm = None
try:
    import torch_xla
    import torch_xla.core.xla_model as xm_module
    xm = xm_module
    TPU_AVAILABLE = True
except Exception:
    TPU_AVAILABLE = False


@dataclass
class HardwareConfig:
    """Dynamic Hardware Accelerator Configuration (KEP Rule #10)."""
    preferred_device: str = "auto"  # 'auto', 'cuda', 'tpu', 'cpu'
    cuda_device_index: int = 0
    enable_amp: bool = True
    amp_dtype: str = "float16"  # 'float16' or 'bfloat16'
    allow_tpu_pjrt: bool = True
    expandable_segments: bool = True


class HardwareEngine:
    """
    Universal Hardware Accelerator Abstraction Engine.
    Handles device resolution, barrier synchronization, memory cache flushing,
    and adaptive step execution across CPU, CUDA, and TPU backends.
    """

    def __init__(self, config: Optional[HardwareConfig] = None):
        self.config = config or HardwareConfig()
        if self.config.expandable_segments and torch.cuda.is_available():
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.device, self.device_type = self._resolve_device()
        logger.info(f"⚡ [Karyon HardwareEngine] Active Hardware Backend: {self.device_type.upper()} ({self.device})")

    def _resolve_device(self) -> Tuple[torch.device, str]:
        pref = self.config.preferred_device.lower()
        
        # 1. Explicit TPU request or auto-detection
        if pref in ("auto", "tpu") and TPU_AVAILABLE:
            try:
                device = xm.xla_device()
                return device, "tpu"
            except Exception as e:
                logger.warning(f"⚠️ TPU requested or auto-detected, but initialization failed: {e}")

        # 2. CUDA GPU Detection
        if pref in ("auto", "cuda") and torch.cuda.is_available():
            idx = min(self.config.cuda_device_index, torch.cuda.device_count() - 1)
            device = torch.device(f"cuda:{idx}")
            return device, "cuda"

        # 3. CPU Fallback
        return torch.device("cpu"), "cpu"

    @property
    def device_str(self) -> str:
        return "xla" if self.is_tpu else self.device_type

    @property
    def is_tpu(self) -> bool:
        return self.device_type == "tpu"

    @property
    def is_cuda(self) -> bool:
        return self.device_type == "cuda"

    @property
    def is_cpu(self) -> bool:
        return self.device_type == "cpu"

    def mark_step(self):
        """Triggers TPU execution graph evaluation or CUDA stream execution."""
        if self.is_tpu and xm is not None:
            xm.mark_step()

    def synchronize(self):
        """Forces hardware barrier synchronization across active compute units."""
        if self.is_tpu and xm is not None:
            xm.mark_step()
        elif self.is_cuda:
            torch.cuda.synchronize(self.device)

    def empty_cache(self):
        """Flushes memory allocators and triggers Python garbage collection."""
        gc.collect()
        if self.is_cuda:
            torch.cuda.empty_cache()
        elif self.is_tpu and xm is not None:
            # TPU garbage collection / graph flush
            pass

    def get_autocast_dtype(self) -> torch.dtype:
        """Resolves target precision dtype for Automatic Mixed Precision."""
        if self.config.amp_dtype == "bfloat16" or self.is_tpu:
            return torch.bfloat16
        return torch.float16

    def optimizer_step(self, optimizer: torch.optim.Optimizer, scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """
        Safely executes optimizer step across hardware backends:
        - TPU: xm.optimizer_step(optimizer)
        - CUDA with AMP: scaler.step(optimizer) + scaler.update()
        - CPU / Standard: optimizer.step()
        """
        if self.is_tpu and xm is not None:
            xm.optimizer_step(optimizer)
            xm.mark_step()
        elif self.is_cuda and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns hardware utilization diagnostics."""
        telemetry = {
            "device_type": self.device_type,
            "device_str": str(self.device),
            "amp_enabled": self.config.enable_amp,
        }
        if self.is_cuda:
            telemetry["gpu_name"] = torch.cuda.get_device_name(self.device)
            telemetry["allocated_mb"] = round(torch.cuda.memory_allocated(self.device) / (1024 ** 2), 2)
            telemetry["reserved_mb"] = round(torch.cuda.memory_reserved(self.device) / (1024 ** 2), 2)
            telemetry["max_allocated_mb"] = round(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2), 2)
        elif self.is_tpu:
            telemetry["tpu_backend"] = "PyTorch-XLA"
        else:
            telemetry["cpu_threads"] = torch.get_num_threads()
        return telemetry


# Global default engine singleton helper
_GLOBAL_HARDWARE_ENGINE: Optional[HardwareEngine] = None

def get_hardware_engine(config: Optional[HardwareConfig] = None) -> HardwareEngine:
    global _GLOBAL_HARDWARE_ENGINE
    if _GLOBAL_HARDWARE_ENGINE is None or config is not None:
        _GLOBAL_HARDWARE_ENGINE = HardwareEngine(config)
    return _GLOBAL_HARDWARE_ENGINE
