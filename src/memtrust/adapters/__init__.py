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
from memtrust.adapters.mempalace_adapter import MemPalaceAdapter
from memtrust.adapters.openviking_adapter import OpenVikingAdapter
from memtrust.adapters.zep_graphiti_adapter import ZepGraphitiAdapter
from memtrust.adapters.zep_graphiti_selfhosted_adapter import ZepGraphitiSelfHostedAdapter

#: Registry the CLI resolves --backends names against. Keys are the
#: user-facing backend names used on the command line.
#:
#: "mem0_selfhosted" and "graphiti_selfhosted" are intentionally not part
#: of cli.ALL_BACKENDS (the set "all" expands to) -- both target a
#: self-run local deployment rather than a hosted vendor API, so both are
#: opt-in only, never auto-included. See docs/methodology.md for their
#: confidence levels.
ADAPTER_REGISTRY: dict[str, type[MemoryBackendAdapter]] = {
    "mempalace": MemPalaceAdapter,
    "mem0": Mem0Adapter,
    "mem0_selfhosted": Mem0SelfHostedAdapter,
    "zep": ZepGraphitiAdapter,
    "graphiti": ZepGraphitiAdapter,
    "graphiti_selfhosted": ZepGraphitiSelfHostedAdapter,
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
    "MemPalaceAdapter",
    "OpenVikingAdapter",
    "ZepGraphitiAdapter",
    "ZepGraphitiSelfHostedAdapter",
]
