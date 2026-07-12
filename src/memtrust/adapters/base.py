"""Shared interface every backend adapter implements.

MemTrust scores backends by running the same eval logic against every
tracked vendor through this one interface. If an adapter needs a special
case to pass an eval, that is a bug in the adapter, not a feature -- the
whole point of a standardized harness is that scoring logic never changes
per vendor.

Every adapter reads its credentials from an environment variable. If the
variable is missing, the adapter raises BackendNotConfiguredError instead
of crashing or silently no-oping. This is what lets `memtrust run` work in
a fresh clone with zero API keys: unconfigured backends print SKIPPED and
the run continues.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum


class BackendNotConfiguredError(Exception):
    """Raised when a backend adapter is missing required configuration.

    Callers (the CLI, eval runners) must catch this specifically and treat
    it as "skip this backend," never as a fatal error for the whole run.
    """

    def __init__(self, backend_name: str, missing_env_var: str) -> None:
        self.backend_name = backend_name
        self.missing_env_var = missing_env_var
        super().__init__(
            f"{backend_name} is not configured: environment variable "
            f"{missing_env_var} is not set. Skipping this backend. "
            f"See docs/methodology.md for setup instructions."
        )


class BackendAPIError(Exception):
    """Raised when a configured backend's API call fails (network, auth,
    5xx, malformed response). Distinct from BackendNotConfiguredError so
    callers can tell "never had credentials" apart from "had credentials,
    the call still failed."
    """

    def __init__(self, backend_name: str, detail: str) -> None:
        self.backend_name = backend_name
        self.detail = detail
        super().__init__(f"{backend_name} API error: {detail}")


class ConflictSignal(StrEnum):
    """How a backend responded when a query touched a fact that had been
    contradicted by a later store()/update() call.

    This is the classification the contradiction-detection eval reads off
    every query() response -- see evals/contradiction.py.
    """

    FLAGGED = "flagged"
    """The backend surfaced the conflict: it returned both versions, an
    explicit conflict marker, or otherwise made the contradiction visible
    to the caller instead of silently resolving it."""

    SILENT_OVERWRITE = "silent_overwrite"
    """The backend replaced the old fact with the new one and gave no
    signal in the query response that a prior, different value ever
    existed."""

    SERVED_STALE = "served_stale"
    """The backend returned the old, now-contradicted fact and gave no
    signal that a newer, conflicting fact had been stored since."""

    NOT_APPLICABLE = "not_applicable"
    """The backend has no update/contradiction-relevant primitive to
    evaluate here (see MemoryBackendAdapter.supports_update). Recorded
    explicitly rather than silently dropping the backend from the eval
    table -- see [redacted] [redacted]."""


@dataclass
class MemoryRecord:
    """One stored memory as returned by a backend's query response."""

    memory_id: str
    content: str
    score: float | None = None
    created_at: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)
    """Unmodified vendor response fragment for this record, kept for
    audit/raw-log purposes. Never used for scoring -- scoring only reads
    the typed fields above so every backend is judged by the same rules.
    """


@dataclass
class QueryResult:
    """Result of MemoryBackendAdapter.query()."""

    records: list[MemoryRecord]
    conflict_signal: ConflictSignal
    """How this query response handled any contradiction relevant to the
    query. Adapters that cannot detect this default to NOT_APPLICABLE and
    the eval scores it as a gap, not a pass or fail -- see
    evals/contradiction.py for the classification logic."""
    latency_ms: float
    raw: dict[str, object] = field(default_factory=dict)


@dataclass
class StoreResult:
    """Result of MemoryBackendAdapter.store()."""

    memory_id: str
    latency_ms: float
    raw: dict[str, object] = field(default_factory=dict)


@dataclass
class UpdateResult:
    """Result of MemoryBackendAdapter.update()."""

    memory_id: str
    acknowledged: bool
    latency_ms: float
    raw: dict[str, object] = field(default_factory=dict)


class MemoryBackendAdapter(ABC):
    """Abstract base every backend adapter must implement.

    Contract for implementers:
      * __init__ must read credentials from an environment variable and
        raise BackendNotConfiguredError immediately if missing -- never
        defer the check to the first method call.
      * store()/query()/update() must raise BackendAPIError (not a bare
        vendor exception) on any network/API failure, so the harness can
        report a uniform error shape across all backends.
      * Never mutate eval logic per backend. If a backend cannot support
        an operation (see supports_update), report that fact through
        supports_update / ConflictSignal.NOT_APPLICABLE rather than
        faking a response.
    """

    #: Human-readable vendor name, used in CLI output and reports.
    name: str = "unknown"

    #: Environment variable this adapter reads its credential/config from.
    #: Subclasses must set this so BackendNotConfiguredError messages are
    #: accurate and so `memtrust run` can pre-check configuration status
    #: without constructing the adapter.
    env_var: str = ""

    #: Whether this backend exposes an update/invalidate primitive the
    #: contradiction-detection eval can meaningfully exercise. If False,
    #: the eval records ConflictSignal.NOT_APPLICABLE instead of running
    #: the eval against this backend and silently dropping it from the
    #: results table.
    supports_update: bool = True

    @abstractmethod
    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        """Store a new memory under the given session/user scope.

        Args:
            session_id: logical conversation/user scope for this memory.
            content: the text to store.
            metadata: optional vendor-agnostic key/value tags.

        Raises:
            BackendAPIError: on any network or vendor-side failure.
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        """Retrieve memories relevant to `query` within `session_id`.

        Raises:
            BackendAPIError: on any network or vendor-side failure.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        """Store a fact that may contradict a previously stored one.

        Implementers should call whatever the vendor's native mechanism is
        for "this may supersede an existing memory" (an explicit update
        call, a second store() that the vendor's own pipeline resolves,
        etc.) and report what actually happened in UpdateResult -- do not
        synthesize agreement with the request.

        Raises:
            BackendAPIError: on any network or vendor-side failure.
        """
        raise NotImplementedError

    @staticmethod
    def _timed() -> _Timer:
        return _Timer()


class _Timer:
    """Tiny context-manager-free stopwatch so adapters can report latency
    without importing timing boilerplate in every subclass."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000
