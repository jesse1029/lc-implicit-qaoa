from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Backend:
    xp: Any
    name: str
    gpu: bool

    def sync(self) -> None:
        if self.gpu:
            self.xp.cuda.Stream.null.synchronize()

    def memory_pool_bytes(self) -> int:
        if not self.gpu:
            return 0
        return int(self.xp.get_default_memory_pool().total_bytes())

    def allocated_memory_bytes(self) -> int:
        if not self.gpu:
            return 0
        return int(self.xp.get_default_memory_pool().used_bytes())

    def free_memory_pool(self) -> None:
        if self.gpu:
            self.xp.get_default_memory_pool().free_all_blocks()


def get_backend(prefer_gpu: bool = True) -> Backend:
    if prefer_gpu:
        try:
            import cupy as cp

            _ = cp.cuda.runtime.getDeviceCount()
            return Backend(cp, "cupy", True)
        except Exception:
            pass

    import numpy as np

    return Backend(np, "numpy", False)
