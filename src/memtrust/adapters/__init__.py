"""Backend adapters. Each module maps one vendor's real API/SDK shape onto
the shared MemoryBackendAdapter interface defined in base.py.
"""

from memtrust.adapters.base import (
    BackendAPIError,
    BackendNotConfiguredError,
    ConflictSignal,
    MemoryBackendAdapter,
    MemoryRecord,
    QueryResult,
    StoreResult,
    UpdateResult,
)
from memtrust.adapters.mem0_adapter import Mem0Adapter
from memtrust.adapters.mempalace_adapter import MemPalaceAdapter
from memtrust.adapters.openviking_adapter import OpenVikingAdapter
from memtrust.adapters.zep_graphiti_adapter import ZepGraphitiAdapter

#: Registry the CLI resolves --backends names against. Keys are the
#: user-facing backend names used on the command line.
ADAPTER_REGISTRY: dict[str, type[MemoryBackendAdapter]] = {
    "mempalace": MemPalaceAdapter,
    "mem0": Mem0Adapter,
    "zep": ZepGraphitiAdapter,
    "graphiti": ZepGraphitiAdapter,
    "openviking": OpenVikingAdapter,
}

__all__ = [
    "ADAPTER_REGISTRY",
    "BackendAPIError",
    "BackendNotConfiguredError",
    "ConflictSignal",
    "MemoryBackendAdapter",
    "MemoryRecord",
    "QueryResult",
    "StoreResult",
    "UpdateResult",
    "Mem0Adapter",
    "MemPalaceAdapter",
    "OpenVikingAdapter",
    "ZepGraphitiAdapter",
]
