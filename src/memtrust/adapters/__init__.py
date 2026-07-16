"""Backend adapters. Each module maps one vendor's real API/SDK shape onto
the shared MemoryBackendAdapter interface defined in base.py.
"""

from memtrust.adapters.base import (
    BackendAPIError,
    BackendNotConfiguredError,
    ConflictSignal,
    DeleteResult,
    MemoryBackendAdapter,
    MemoryRecord,
    QueryResult,
    StoreResult,
    UpdateResult,
)
from memtrust.adapters.mem0_adapter import Mem0Adapter, Mem0SelfHostedAdapter
from memtrust.adapters.mem0_direct_adapter import Mem0DirectAdapter
from memtrust.adapters.mempalace_adapter import MemPalaceAdapter
from memtrust.adapters.openviking_adapter import OpenVikingAdapter
from memtrust.adapters.zep_graphiti_adapter import ZepGraphitiAdapter

#: Registry the CLI resolves --backends names against. Keys are the
#: user-facing backend names used on the command line.
#:
#: "mem0_selfhosted" and "mem0_direct" are intentionally not part of
#: cli.ALL_BACKENDS (the set "all" expands to) -- neither targets a single
#: hosted vendor API the way the other four do ("mem0_selfhosted" is a
#: self-run local server, "mem0_direct" is a self-assembled in-process
#: embedder/vector-store stack), so both are opt-in only, never
#: auto-included. See docs/methodology.md for their confidence levels.
ADAPTER_REGISTRY: dict[str, type[MemoryBackendAdapter]] = {
    "mempalace": MemPalaceAdapter,
    "mem0": Mem0Adapter,
    "mem0_selfhosted": Mem0SelfHostedAdapter,
    "mem0_direct": Mem0DirectAdapter,
    "zep": ZepGraphitiAdapter,
    "graphiti": ZepGraphitiAdapter,
    "openviking": OpenVikingAdapter,
}

__all__ = [
    "ADAPTER_REGISTRY",
    "BackendAPIError",
    "BackendNotConfiguredError",
    "ConflictSignal",
    "DeleteResult",
    "MemoryBackendAdapter",
    "MemoryRecord",
    "QueryResult",
    "StoreResult",
    "UpdateResult",
    "Mem0Adapter",
    "Mem0SelfHostedAdapter",
    "Mem0DirectAdapter",
    "MemPalaceAdapter",
    "OpenVikingAdapter",
    "ZepGraphitiAdapter",
]
