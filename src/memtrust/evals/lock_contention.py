"""MemTrust's lock-contention/hang-detection eval.

Motivating case: volcengine/OpenViking#1581 (contributor 0xble).
`memory.v2_lock_max_retries` uses `0` to mean "unlimited retries" -- the
opposite of the usual convention where `0` retries means "one attempt and
give up." Because `0` is also the default value, any deployment that never
explicitly sets this field gets infinite-retry semantics for free: a
session-commit path contending for a stale lock (e.g. one left behind by a
crashed holder -- the issue's own step-2 repro) spins forever. No exception
is ever raised; the only surface signal is a repeated `logger.warning`
line ("retrying ... max=unlimited"). A caller waiting on that request has
no API-observable way to tell "still legitimately working" apart from
"will never return."

**Honest scope of what this eval can and cannot prove.** None of memtrust's
adapters have process-lifecycle or server-config control (see adapters/
base.py's module docstring) -- there is no way for this harness to reach
into a live OpenViking server and actually set `memory.v2_lock_max_retries`
or engineer a genuinely crashed lock holder server-side. So this eval does
not, and cannot, reproduce #1581 against a live OpenViking instance.
Instead, matching this codebase's established pattern for capability-gap
evals (evals/crash_recovery.py, evals/migration_rollback.py), it targets
the STRUCTURAL failure shape directly and generically: fire N concurrent
store() calls at the SAME `resource_path` (the one primitive every adapter
that honors resource-scoped writes shares -- see
MemoryBackendAdapter.supports_resource_sync and openviking_adapter.py's
store()) and assert that every request either completes (successfully or
with a raised error) or is definitively abandoned within a fixed wall-clock
response-time budget. A request that neither returns nor raises within
that budget is exactly #1581's "spins forever, no exception, no signal"
shape, made API-observable instead of only visible as a WARN-log flood.

Only a purpose-built in-memory fake adapter can genuinely model both the
buggy "unlimited retries against a permanently stale lock" behavior and
the fixed "bounded retries, fails fast" behavior for a live-timing
assertion in a test suite -- see
tests/test_evals.py::LockContentionFakeAdapter. A real, configured adapter
can still be pointed at this eval (nothing about run_lock_contention_eval
requires a fake), but as of this writing no adapter in this repo has ever
been run through it against a live backend -- see docs/methodology.md.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum

from memtrust.adapters.base import BackendAPIError, MemoryBackendAdapter

#: Default wall-clock budget for a single request under contention.
#: Generous enough that a healthy backend serializing writes to the same
#: resource_path (a legitimate, bounded queueing delay) comfortably clears
#: it, tight enough that a genuinely stuck/unbounded-retry request (the
#: #1581 shape) is reliably caught within a fast-in-CI test run.
DEFAULT_RESPONSE_BUDGET_MS = 2000.0

#: Default number of concurrent writers contending for the same
#: resource_path. Small enough to run fast in CI, large enough that a
#: single-writer-at-a-time lock produces real, observable queueing.
DEFAULT_N_CONCURRENT = 8

DEFAULT_RESOURCE_PATH = "lock-contention/shared-resource.md"
DEFAULT_SESSION_ID = "lock-contention-session"


class LockContentionSignal(StrEnum):
    """How a backend's concurrent-write requests against the SAME
    resource_path behaved relative to a fixed response-time budget.

    Defined locally in this module rather than in adapters/base.py,
    following the same precedent evals/scale_stress.py's ScaleSignal and
    evals/stats_accuracy.py's StatsSignal already set: a harness-computed
    classification derived from timing ground truth this eval measures
    itself, not a signal any adapter self-reports.
    """

    BOUNDED_RESPONSE = "bounded_response"
    """Every concurrent request against the shared resource_path either
    succeeded or raised within the response-time budget. The good
    outcome -- contention exists (requests may still queue behind each
    other) but never silently exceeds a bounded SLA. This is the fixed,
    post-#1581 semantics: even a request that ultimately fails to acquire
    the lock fails FAST, with a raised error, rather than spinning."""

    UNBOUNDED_STALL = "unbounded_stall"
    """At least one concurrent request against the shared resource_path
    neither succeeded nor raised within the response-time budget -- it was
    still pending when this eval gave up waiting on it. This is the exact
    volcengine/OpenViking#1581 shape made API-observable: a stuck lock
    that retries forever, with no exception and no signal any caller could
    act on besides waiting indefinitely."""

    NOT_APPLICABLE = "not_applicable"
    """Either the adapter has no resource-path-scoped write concept at all
    (MemoryBackendAdapter.supports_resource_sync is False -- concurrent
    writers targeting "the same resource_path" have nothing to actually
    contend over, so the eval is skipped, not run), or zero requests were
    issued."""


@dataclass
class LockContentionRequestResult:
    """The outcome of one concurrent store() call against the shared
    resource_path."""

    worker_index: int
    completed: bool
    """False means this request was still running when the eval's
    response-time budget elapsed and it gave up waiting -- see
    run_lock_contention_eval's module docstring on why this is measured
    via a bounded wait rather than an actual cancellation (Python threads
    cannot be forcibly killed; the underlying call may keep running in the
    background after this eval returns)."""
    succeeded: bool | None
    """True/False if completed is True (whether the store() call itself
    raised BackendAPIError or not). None if completed is False -- there is
    no verdict yet for a request that never returned."""
    latency_ms: float | None
    """None if completed is False."""
    error: str | None = None


@dataclass
class LockContentionEvalResult:
    backend_name: str
    resource_path: str
    budget_ms: float
    n_concurrent: int
    requests: list[LockContentionRequestResult] = field(default_factory=list)
    signal: LockContentionSignal = LockContentionSignal.NOT_APPLICABLE
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def stalled_count(self) -> int:
        """How many of `requests` never completed within the budget --
        the headline metric this eval exists to surface."""
        return sum(1 for r in self.requests if not r.completed)

    @property
    def max_latency_ms(self) -> float | None:
        """Highest latency among requests that DID complete within the
        budget. `None` if none completed (or there were no requests)."""
        completed_latencies = [r.latency_ms for r in self.requests if r.latency_ms is not None]
        return max(completed_latencies) if completed_latencies else None


def classify_lock_contention_result(
    requests: list[LockContentionRequestResult],
) -> LockContentionSignal:
    """Classify a completed run's per-request outcomes. Never a blind
    "did anything raise" check -- the classification is driven entirely by
    whether every request resolved (success or raised error, either is
    fine) within the fixed response-time budget, the same
    ground-truth-driven pattern every other eval's classify_* function in
    this package follows.
    """
    if not requests:
        return LockContentionSignal.NOT_APPLICABLE
    if any(not r.completed for r in requests):
        return LockContentionSignal.UNBOUNDED_STALL
    return LockContentionSignal.BOUNDED_RESPONSE


def run_lock_contention_eval(
    adapter: MemoryBackendAdapter,
    resource_path: str = DEFAULT_RESOURCE_PATH,
    n_concurrent: int = DEFAULT_N_CONCURRENT,
    budget_ms: float = DEFAULT_RESPONSE_BUDGET_MS,
    session_id: str = DEFAULT_SESSION_ID,
) -> LockContentionEvalResult:
    """Fire `n_concurrent` concurrent store() calls at the SAME
    `resource_path` and assert each one resolves (successfully or with a
    raised BackendAPIError) within `budget_ms` wall-clock time.

    Uses real OS threads (`threading.Thread`, daemon=True) rather than a
    thread pool with a blocking shutdown: a genuinely stuck request (the
    #1581 shape this eval exists to catch) must not block this function
    from returning once its budget has elapsed, and a daemon thread left
    running in the background cannot block process/test-suite exit either
    -- see tests/test_evals.py::LockContentionFakeAdapter's docstring for
    how its own internal retry cap keeps an abandoned worker thread
    bounded rather than truly running forever.

    Args:
        adapter: the backend under test.
        resource_path: the shared resource_path every concurrent writer
            targets, via `metadata={"resource_path": resource_path}` (the
            same metadata key openviking_adapter.py's store() honors --
            see its module docstring).
        n_concurrent: how many concurrent store() calls to fire.
        budget_ms: wall-clock budget, per request, before this eval gives
            up waiting and classifies that request as UNBOUNDED_STALL.
        session_id: session/scope every request is stored under.
    """
    result = LockContentionEvalResult(
        backend_name=adapter.name,
        resource_path=resource_path,
        budget_ms=budget_ms,
        n_concurrent=n_concurrent,
    )

    if not adapter.supports_resource_sync:
        result.skipped = True
        result.skip_reason = (
            f"{adapter.name} does not honor a resource_path-scoped store() "
            "(supports_resource_sync=False) -- concurrent writers have no shared "
            "target to actually contend over. Skipped, not run. See "
            "adapters/base.py's MemoryBackendAdapter.supports_resource_sync and "
            "evals/lock_contention.py's module docstring."
        )
        return result

    if n_concurrent < 1:
        return result

    slots: list[LockContentionRequestResult | None] = [None] * n_concurrent

    def _worker(worker_index: int) -> None:
        start = time.perf_counter()
        try:
            adapter.store(
                session_id,
                f"concurrent write from worker {worker_index}",
                metadata={"resource_path": resource_path},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            slots[worker_index] = LockContentionRequestResult(
                worker_index=worker_index, completed=True, succeeded=True, latency_ms=elapsed_ms
            )
        except BackendAPIError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            slots[worker_index] = LockContentionRequestResult(
                worker_index=worker_index,
                completed=True,
                succeeded=False,
                latency_ms=elapsed_ms,
                error=str(exc),
            )

    threads = [
        threading.Thread(target=_worker, args=(i,), daemon=True) for i in range(n_concurrent)
    ]
    deadline = time.perf_counter() + budget_ms / 1000
    for thread in threads:
        thread.start()
    for thread in threads:
        remaining = max(0.0, deadline - time.perf_counter())
        thread.join(timeout=remaining)

    for worker_index, slot in enumerate(slots):
        result.requests.append(
            slot
            if slot is not None
            else LockContentionRequestResult(
                worker_index=worker_index, completed=False, succeeded=None, latency_ms=None
            )
        )

    result.signal = classify_lock_contention_result(result.requests)
    return result
