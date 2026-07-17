"""Tests for the lock-contention/hang-detection eval
(src/memtrust/evals/lock_contention.py). All in-memory fake adapters --
no real backend or network calls, matching the pattern in
tests/test_evals.py and tests/test_scale_stress.py.

LockContentionFakeAdapter reproduces volcengine/OpenViking#1581's exact
bug shape: the FIRST store() call for a given resource_path acquires a
simulated lock and never releases it (modeling the issue's own "a stale
lock file from a crashed holder" repro step), so every subsequent
concurrent store() call for the SAME resource_path must retry-poll to
acquire a lock that will never become free. Constructed with
`max_retries=0`, it reproduces the pre-#1581 buggy default ("0 means
unlimited retries"). Constructed with `max_retries>0`, it reproduces the
fixed semantics (bounded retries, fails fast).
"""

from __future__ import annotations

import threading
import time

from memtrust.adapters.base import (
    BackendAPIError,
    ConflictSignal,
    DeleteResult,
    MemoryBackendAdapter,
    QueryResult,
    StoreResult,
    UpdateResult,
)
from memtrust.evals.lock_contention import (
    LockContentionRequestResult,
    LockContentionSignal,
    classify_lock_contention_result,
    run_lock_contention_eval,
)


class LockContentionFakeAdapter(MemoryBackendAdapter):
    """See module docstring above. Each `resource_path` gets an
    independent simulated lock; the first caller for a given path
    acquires it and never releases it.

    `max_retries=0` reproduces the pre-#1581 buggy "unlimited retries"
    default: retries are capped at `_UNLIMITED_RETRY_TEST_CAP`, not
    truly infinite -- an actually-infinite loop would hang this test
    suite (and, via a daemon thread abandoned by run_lock_contention_eval
    once its budget elapses, would otherwise spin in the background for
    the rest of the process's life). The cap is chosen far larger than
    any budget_ms this test suite uses, so from the eval's perspective
    under a realistic budget it is indistinguishable from "never
    returns" -- the same honest, bounded-standin-for-unbounded pattern
    tests/test_evals.py::CrashRecoveryFakeAdapter and friends use
    throughout this repo. `max_retries>0` reproduces the fixed,
    post-#1581 semantics: the caller fails fast with a raised
    BackendAPIError after exactly `max_retries` attempts.
    """

    name = "fake-lock-contention"
    env_var = "FAKE_API_KEY"
    supports_resource_sync = True

    _UNLIMITED_RETRY_TEST_CAP = 400
    """Stands in for #1581's real "unlimited" semantics. At the default
    poll_interval_s=0.01, this bounds an abandoned background thread's
    real run time to ~4s -- long enough that it is never mistaken for
    "bounded" by any budget_ms this suite uses (all well under 1s), short
    enough that it self-terminates quickly instead of truly running
    forever."""

    def __init__(self, max_retries: int = 0, poll_interval_s: float = 0.01) -> None:
        self.max_retries = max_retries
        self.poll_interval_s = poll_interval_s
        self._locked: dict[str, bool] = {}
        self._guard = threading.Lock()

    def _try_acquire(self, resource_path: str) -> bool:
        with self._guard:
            if not self._locked.get(resource_path, False):
                self._locked[resource_path] = True
                return True
            return False

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        del session_id, content, mode
        metadata = metadata or {}
        resource_path = metadata.get("resource_path", "default")
        timer = self._timed()

        if self._try_acquire(resource_path):
            return StoreResult(memory_id=f"{resource_path}:holder", latency_ms=timer.elapsed_ms())

        retry_ceiling = (
            self._UNLIMITED_RETRY_TEST_CAP if self.max_retries == 0 else self.max_retries
        )
        attempts = 0
        while attempts < retry_ceiling:
            time.sleep(self.poll_interval_s)
            attempts += 1
            if self._try_acquire(resource_path):
                return StoreResult(memory_id=f"{resource_path}:late", latency_ms=timer.elapsed_ms())
        raise BackendAPIError(
            self.name, f"failed to acquire lock for {resource_path} after {attempts} attempts"
        )

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        del session_id, query, top_k, mode
        return QueryResult(
            records=[], conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        del session_id, content
        return UpdateResult(memory_id=memory_id, acknowledged=True, latency_ms=0.1)

    def delete(self, memory_id: str) -> DeleteResult:
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=0.1)


class NoResourceSyncFakeAdapter(MemoryBackendAdapter):
    """Negative control: a backend with no resource_path-scoped write
    concept at all (supports_resource_sync=False, the base default) --
    the eval must report NOT_APPLICABLE/skipped rather than crashing or
    guessing at contention that has no shared target to contend over."""

    name = "fake-no-resource-sync"
    env_var = "FAKE_API_KEY"

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        del session_id, content, metadata, mode
        return StoreResult(memory_id="m1", latency_ms=0.1)

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        del session_id, query, top_k, mode
        return QueryResult(
            records=[], conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        del session_id, content
        return UpdateResult(memory_id=memory_id, acknowledged=True, latency_ms=0.1)

    def delete(self, memory_id: str) -> DeleteResult:
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=0.1)


# ---------------------------------------------------------------------------
# run_lock_contention_eval -- end-to-end against fake adapters
# ---------------------------------------------------------------------------


def test_unlimited_retries_shape_stalls_past_budget() -> None:
    """The exact #1581 bug: max_retries=0 (the buggy default) means a
    request contending for a permanently-held lock is still pending well
    past a realistic response-time budget."""
    adapter = LockContentionFakeAdapter(max_retries=0, poll_interval_s=0.02)
    result = run_lock_contention_eval(
        adapter, n_concurrent=5, budget_ms=150.0, resource_path="shared/doc.md"
    )
    assert result.signal == LockContentionSignal.UNBOUNDED_STALL
    assert result.stalled_count >= 1
    # The very first acquirer always completes immediately -- confirms this
    # isn't a blanket "everything failed" result, only the contended ones.
    assert any(r.completed and r.succeeded for r in result.requests)


def test_bounded_retries_shape_completes_within_budget() -> None:
    """The fixed, post-#1581 semantics: a bounded max_retries fails fast
    (raises) instead of spinning -- every request resolves well within
    budget."""
    adapter = LockContentionFakeAdapter(max_retries=3, poll_interval_s=0.01)
    result = run_lock_contention_eval(
        adapter, n_concurrent=5, budget_ms=2000.0, resource_path="shared/doc.md"
    )
    assert result.signal == LockContentionSignal.BOUNDED_RESPONSE
    assert result.stalled_count == 0
    assert all(r.completed for r in result.requests)
    # Contending workers should have failed fast (raised), not silently
    # succeeded -- the lock genuinely never frees in this fake.
    non_holders = [r for r in result.requests if r.succeeded is False]
    assert len(non_holders) == 4


def test_skips_adapter_without_resource_sync() -> None:
    adapter = NoResourceSyncFakeAdapter()
    result = run_lock_contention_eval(adapter)
    assert result.skipped is True
    assert result.signal == LockContentionSignal.NOT_APPLICABLE
    assert result.skip_reason is not None
    assert "supports_resource_sync" in result.skip_reason


def test_single_uncontended_writer_completes_bounded() -> None:
    """No contention at all (n_concurrent=1) should trivially be
    BOUNDED_RESPONSE -- the eval shouldn't manufacture a stall out of
    nothing."""
    adapter = LockContentionFakeAdapter(max_retries=0, poll_interval_s=0.01)
    result = run_lock_contention_eval(adapter, n_concurrent=1, budget_ms=500.0)
    assert result.signal == LockContentionSignal.BOUNDED_RESPONSE
    assert result.requests[0].completed is True
    assert result.requests[0].succeeded is True


def test_result_reports_backend_name_and_config() -> None:
    adapter = LockContentionFakeAdapter()
    result = run_lock_contention_eval(
        adapter, n_concurrent=3, budget_ms=100.0, resource_path="custom/path.md"
    )
    assert result.backend_name == "fake-lock-contention"
    assert result.resource_path == "custom/path.md"
    assert result.budget_ms == 100.0
    assert result.n_concurrent == 3
    assert len(result.requests) == 3


# ---------------------------------------------------------------------------
# classify_lock_contention_result -- pure classification logic
# ---------------------------------------------------------------------------


def test_classify_not_applicable_on_empty_requests() -> None:
    assert classify_lock_contention_result([]) == LockContentionSignal.NOT_APPLICABLE


def test_classify_bounded_when_all_completed() -> None:
    requests = [
        LockContentionRequestResult(worker_index=0, completed=True, succeeded=True, latency_ms=1.0),
        LockContentionRequestResult(
            worker_index=1, completed=True, succeeded=False, latency_ms=2.0, error="boom"
        ),
    ]
    assert classify_lock_contention_result(requests) == LockContentionSignal.BOUNDED_RESPONSE


def test_classify_unbounded_stall_when_any_incomplete() -> None:
    requests = [
        LockContentionRequestResult(worker_index=0, completed=True, succeeded=True, latency_ms=1.0),
        LockContentionRequestResult(
            worker_index=1, completed=False, succeeded=None, latency_ms=None
        ),
    ]
    assert classify_lock_contention_result(requests) == LockContentionSignal.UNBOUNDED_STALL


def test_max_latency_ms_ignores_incomplete_requests() -> None:
    from memtrust.evals.lock_contention import LockContentionEvalResult

    result = LockContentionEvalResult(
        backend_name="x", resource_path="p", budget_ms=100.0, n_concurrent=2
    )
    result.requests = [
        LockContentionRequestResult(worker_index=0, completed=True, succeeded=True, latency_ms=5.0),
        LockContentionRequestResult(
            worker_index=1, completed=False, succeeded=None, latency_ms=None
        ),
    ]
    assert result.max_latency_ms == 5.0
    assert result.stalled_count == 1
