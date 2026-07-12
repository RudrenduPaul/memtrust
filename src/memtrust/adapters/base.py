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
    table."""

    EMPTY_OR_LOST = "empty_or_lost"
    """The backend DOES have an update/contradiction-relevant primitive
    (MemoryBackendAdapter.supports_update is True), the store()/update()/
    query() calls all completed without raising BackendAPIError, but the
    query response came back with zero records -- no exception, no error,
    just nothing. This is distinct from NOT_APPLICABLE, which means the
    backend structurally cannot be evaluated here at all: EMPTY_OR_LOST
    means the backend *should* have had something to say and silently
    didn't. This is the "call succeeded but produced nothing" failure mode
    the benchmark exists to catch -- see evals/contradiction.py's
    classify_case for exactly when this is assigned, never a vendor's own
    self-report."""


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
    verified: bool | None = None
    """Whether a post-write read-back confirmed the content is actually
    retrievable. `None` means the adapter did not attempt verification
    (the default -- see MemoryBackendAdapter.verify_store) -- this is
    deliberately distinct from `False`. `None` is "we don't know," and
    must never be treated as "verified passed" by scoring or reporting
    code. `True`/`False` mean an adapter actually called verify_store()
    (or equivalent) and got a definitive answer.

    Why this field exists at all: store() raising no exception has never
    been proof that a write was durable -- a vendor can return 200 OK (or
    a fake in-process success) while silently dropping or corrupting the
    write server-side. Two independently root-caused MemPalace bug
    classes did exactly this (checkpoint corruption via NUL bytes;
    stale/self-deadlocked locks silently no-oping a write). Without this
    field, that failure mode is indistinguishable from "the model just
    didn't recall the fact," which misattributes a backend durability bug
    to model quality. See docs/methodology.md for why verification is
    opt-in rather than automatic.
    """


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

        Note on durability: returning without raising is *not* proof the
        write is durable or even retrievable -- a vendor can silently
        drop or corrupt a write server-side and still return a normal
        response (this is exactly what happened in two independently
        root-caused MemPalace bugs: NUL-byte checkpoint corruption and
        stale/self-deadlocked locks). Implementers that want to guard
        against this should accept an opt-in `verify: bool = False`
        keyword-only parameter and, when True, call `self.verify_store()`
        after the write succeeds and set `StoreResult.verified` from its
        return value. This is intentionally NOT part of every adapter's
        required signature (existing callers that never pass `verify`
        must keep working unchanged against any adapter), and it is
        intentionally NOT on by default anywhere -- see verify_store()
        and docs/methodology.md for why (it doubles API calls per store()
        when enabled).
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

    def verify_store(self, store_result: StoreResult, session_id: str, content: str) -> bool:
        """Opt-in read-after-write check: query() immediately after a
        store() call and confirm the just-written content is actually
        retrievable, instead of trusting "store() didn't raise" as proof
        of a durable write.

        This is a helper, not something store() calls automatically --
        no adapter's default `store()` behavior changes by this method
        existing. An adapter opts in by accepting a `verify: bool = False`
        keyword-only parameter on its own store() and calling this
        explicitly when `verify=True` (see mempalace_adapter.py for the
        reference implementation). Left off by default because it costs
        one extra vendor API call (a query()) for every store() call made
        with verify=True -- turning that on unconditionally for every eval
        run would silently double memtrust's own API/latency cost against
        every backend under test. See docs/methodology.md.

        Args:
            store_result: the StoreResult just returned by this adapter's
                own store() call, used for its memory_id.
            session_id: the same session/scope the content was stored
                under.
            content: the exact text that was just stored.

        Returns:
            True only if a query() call for `session_id` returns a record
            whose *content* actually contains `content` -- either the
            record matching `store_result.memory_id` by id (checked
            first, and its content must still match: a record that comes
            back under the right id but with corrupted content is a
            failed verification, not a pass) or, if no record shares that
            id (some backends don't echo a stable id on the query path),
            any returned record whose content contains `content`. False
            covers both "no matching record at all" (dropped write) and
            "a record came back but the content doesn't match" (corrupted
            write) -- both are reported as a normal `False` return, never
            raised as an error.

        Raises:
            BackendAPIError: if the verification query() call itself
                fails (a real network/vendor error is a different failure
                mode than "the write was silently dropped," and should
                still surface as an error rather than being swallowed
                into `False`).
        """
        result = self.query(session_id, content)
        for record in result.records:
            if record.memory_id and record.memory_id == store_result.memory_id:
                return bool(content) and content in record.content
        return any(content and content in record.content for record in result.records)


class _Timer:
    """Tiny context-manager-free stopwatch so adapters can report latency
    without importing timing boilerplate in every subclass."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000
