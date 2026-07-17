"""Tests for the stats/dashboard-accuracy eval
(src/memtrust/evals/stats_accuracy.py). All in-memory fake adapters -- no
real backend or network calls, matching the pattern in tests/test_evals.py
and tests/test_scale_stress.py.

StatsUndercountingFakeAdapter reproduces volcengine/OpenViking#1255's exact
bug shape (contributor SeeYangZhi): store()/query() both work correctly
(memories are genuinely retrievable, exactly like the issue's own
`POST /v1/search/find` evidence), but get_stats() reads from a separate,
never-wired-up counter that always reports zero -- the exact "stats
undercounted vs. verified filesystem/search state" shape this eval exists
to catch.
"""

from __future__ import annotations

from memtrust.adapters.base import (
    ConflictSignal,
    DeleteResult,
    MemoryBackendAdapter,
    MemoryRecord,
    QueryResult,
    StatsResult,
    StoreResult,
    UpdateResult,
)
from memtrust.evals.stats_accuracy import (
    StatsAccuracyEvalResult,
    StatsSignal,
    classify_stats_accuracy,
    run_stats_accuracy_eval,
)


class _StatsFakeAdapterBase(MemoryBackendAdapter):
    """Shared store()/query() plumbing: a plain in-memory dict, matched by
    substring on query() -- the same minimal-but-real pattern
    tests/test_evals.py::RecallAllFakeAdapter establishes."""

    env_var = "FAKE_API_KEY"
    supports_stats = True

    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}
        self._counter = 0

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        del metadata, mode
        self._counter += 1
        self._store.setdefault(session_id, []).append(content)
        return StoreResult(memory_id=f"m{self._counter}", latency_ms=0.1)

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        del mode
        matches = [
            MemoryRecord(memory_id=f"m{i}", content=c)
            for i, c in enumerate(self._store.get(session_id, []))
            if query.lower() in c.lower()
        ][:top_k]
        return QueryResult(
            records=matches, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        del session_id, content
        return UpdateResult(memory_id=memory_id, acknowledged=True, latency_ms=0.1)

    def delete(self, memory_id: str) -> DeleteResult:
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=0.1)


class StatsAccurateFakeAdapter(_StatsFakeAdapterBase):
    """Negative control: get_stats() genuinely counts what was stored."""

    name = "fake-stats-accurate"

    def get_stats(self, session_id: str | None = None) -> StatsResult:
        total = sum(len(records) for records in self._store.values())
        if session_id is not None:
            total = len(self._store.get(session_id, []))
        return StatsResult(total_memories=total, latency_ms=0.1)


class StatsUndercountingFakeAdapter(_StatsFakeAdapterBase):
    """The exact volcengine/OpenViking#1255 shape: store() and query() both
    work correctly (memories are genuinely retrievable), but get_stats()
    reads from a separate, never-wired-up counter that always reports
    zero, regardless of how much is actually stored."""

    name = "fake-stats-undercounting"

    def get_stats(self, session_id: str | None = None) -> StatsResult:
        del session_id
        return StatsResult(total_memories=0, latency_ms=0.1)


class StatsUnsupportedFakeAdapter(_StatsFakeAdapterBase):
    """Negative control: a backend with no stats endpoint at all
    (supports_stats=False, the base default) -- the eval must report
    NOT_APPLICABLE/skipped rather than crashing on the base class's
    NotImplementedError."""

    name = "fake-stats-unsupported"
    supports_stats = False


class StatsFailingFakeAdapter(_StatsFakeAdapterBase):
    """A stats-capable adapter whose store() calls themselves always fail
    -- the eval must report NOT_APPLICABLE (nothing to verify a count
    against) rather than crashing."""

    name = "fake-stats-failing-store"

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        del session_id, content, metadata, mode
        from memtrust.adapters.base import BackendAPIError

        raise BackendAPIError(self.name, "simulated store failure")

    def get_stats(self, session_id: str | None = None) -> StatsResult:
        del session_id
        return StatsResult(total_memories=0, latency_ms=0.1)


# ---------------------------------------------------------------------------
# run_stats_accuracy_eval -- end-to-end against fake adapters
# ---------------------------------------------------------------------------


def test_undercounting_backend_is_flagged() -> None:
    adapter = StatsUndercountingFakeAdapter()
    result = run_stats_accuracy_eval(adapter, n_records=5)
    assert result.signal == StatsSignal.STATS_UNDERCOUNTED
    assert result.verified_count == 5
    assert result.reported_count == 0
    assert result.undercount_gap == 5


def test_accurate_backend_matches() -> None:
    adapter = StatsAccurateFakeAdapter()
    result = run_stats_accuracy_eval(adapter, n_records=5)
    assert result.signal == StatsSignal.STATS_MATCH
    assert result.verified_count == 5
    assert result.reported_count == 5
    assert result.undercount_gap == 0


def test_skips_adapter_without_stats_support() -> None:
    adapter = StatsUnsupportedFakeAdapter()
    result = run_stats_accuracy_eval(adapter)
    assert result.skipped is True
    assert result.signal == StatsSignal.NOT_APPLICABLE
    assert result.skip_reason is not None
    assert "supports_stats" in result.skip_reason


def test_not_applicable_when_nothing_stored() -> None:
    adapter = StatsFailingFakeAdapter()
    result = run_stats_accuracy_eval(adapter, n_records=3)
    assert result.signal == StatsSignal.NOT_APPLICABLE
    assert result.records_stored == 0
    assert result.verified_count is None


def test_result_reports_backend_name_and_records_requested() -> None:
    adapter = StatsAccurateFakeAdapter()
    result = run_stats_accuracy_eval(adapter, n_records=3)
    assert result.backend_name == "fake-stats-accurate"
    assert result.n_records_requested == 3
    assert result.records_stored == 3


# ---------------------------------------------------------------------------
# classify_stats_accuracy -- pure classification logic
# ---------------------------------------------------------------------------


def test_classify_not_applicable_when_verified_count_missing() -> None:
    assert classify_stats_accuracy(None, 5) == StatsSignal.NOT_APPLICABLE


def test_classify_not_applicable_when_reported_count_missing() -> None:
    assert classify_stats_accuracy(5, None) == StatsSignal.NOT_APPLICABLE


def test_classify_undercounted_when_reported_less_than_verified() -> None:
    assert classify_stats_accuracy(10, 0) == StatsSignal.STATS_UNDERCOUNTED
    assert classify_stats_accuracy(10, 3) == StatsSignal.STATS_UNDERCOUNTED


def test_classify_match_when_reported_equals_verified() -> None:
    assert classify_stats_accuracy(5, 5) == StatsSignal.STATS_MATCH


def test_classify_match_when_reported_exceeds_verified() -> None:
    # Not flagged as a separate "overcounted" signal -- see
    # StatsSignal.STATS_MATCH's docstring for why.
    assert classify_stats_accuracy(5, 9) == StatsSignal.STATS_MATCH


def test_undercount_gap_none_when_either_count_missing() -> None:
    result = StatsAccuracyEvalResult(backend_name="x", n_records_requested=5)
    assert result.undercount_gap is None
    result.verified_count = 5
    assert result.undercount_gap is None
