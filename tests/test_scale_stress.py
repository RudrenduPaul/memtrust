"""Tests for the scale/volume stress-testing eval
(src/memtrust/evals/scale_stress.py) and its fixture generator
(src/memtrust/evals/scale_fixtures.py). All in-memory fake adapters -- no
real backend or network calls, matching the pattern in tests/test_evals.py
and tests/test_compression.py.

The two "breaks at scale" fake adapters below are engineered to reproduce
the two real, documented bug shapes this eval exists to catch:

  * ScaleEmptyAtVolumeFakeAdapter models volcengine/OpenViking#2850 (BM25
    search silently returning empty results once a corpus grows large):
    query() unconditionally returns zero records once the session has more
    than EMPTY_AT_VOLUME_THRESHOLD stored records, regardless of which
    marker is being searched for.
  * ScaleEvictsOldFakeAdapter models getzep/graphiti#1275 (O(n)
    entity-resolution context growth silently dropping old episodes):
    query() only ever searches the most recently stored
    EVICTION_WINDOW_SIZE records, so older content becomes permanently
    unrecoverable once enough new records have been stored, with no error
    raised anywhere.

ScaleCleanFakeAdapter is the negative control: a backend whose recall is
genuinely scale-invariant (a plain linear substring scan over everything
ever stored), which must be classified WORKED_AT_SCALE at every N this
suite exercises it at -- proving the classifier does not cry wolf on a
backend that is actually fine.
"""

from __future__ import annotations

import pytest

from memtrust.adapters.base import (
    BackendAPIError,
    ConflictSignal,
    DeleteResult,
    MemoryBackendAdapter,
    MemoryRecord,
    QueryResult,
    StoreResult,
    UpdateResult,
)
from memtrust.evals.scale_fixtures import generate_scale_corpus
from memtrust.evals.scale_stress import (
    DEFAULT_N_RECORDS,
    ScaleCheckpointResult,
    ScaleSignal,
    _default_checkpoints,
    _percentile,
    classify_scale_result,
    run_scale_stress_eval,
)


class ScaleCleanFakeAdapter(MemoryBackendAdapter):
    """Negative control: recall is genuinely scale-invariant -- a linear
    substring scan over every record ever stored in the session, however
    many there are. Should classify WORKED_AT_SCALE at every N."""

    name = "fake-scale-clean"
    env_var = "FAKE_API_KEY"
    supports_update = True

    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        records = self._store.setdefault(session_id, [])
        records.append(content)
        return StoreResult(memory_id=f"{session_id}-{len(records) - 1}", latency_ms=0.5)

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        matches = [c for c in self._store.get(session_id, []) if query.lower() in c.lower()]
        records = [
            MemoryRecord(memory_id=f"m{i}", content=c) for i, c in enumerate(matches[:top_k])
        ]
        return QueryResult(
            records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.5
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        result = self.store(session_id, content)
        return UpdateResult(memory_id=result.memory_id, acknowledged=True, latency_ms=0.5)

    def delete(self, memory_id: str) -> DeleteResult:
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=0.5)


EMPTY_AT_VOLUME_THRESHOLD = 200


class ScaleEmptyAtVolumeFakeAdapter(ScaleCleanFakeAdapter):
    """Positive control modeling volcengine/OpenViking#2850: search
    silently returns zero records for every query once the corpus has
    grown past EMPTY_AT_VOLUME_THRESHOLD records -- no error raised, a
    completely ordinary-looking empty success."""

    name = "fake-scale-empty-at-volume"

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        if len(self._store.get(session_id, [])) > EMPTY_AT_VOLUME_THRESHOLD:
            return QueryResult(
                records=[], conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.5
            )
        return super().query(session_id, query, top_k=top_k)


EVICTION_WINDOW_SIZE = 100


class ScaleEvictsOldFakeAdapter(ScaleCleanFakeAdapter):
    """Positive control modeling getzep/graphiti#1275: only the most
    recently stored EVICTION_WINDOW_SIZE records are ever searchable --
    older content silently becomes permanently unrecoverable as more is
    ingested, with no error raised anywhere."""

    name = "fake-scale-evicts-old"

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        window = self._store.get(session_id, [])[-EVICTION_WINDOW_SIZE:]
        matches = [c for c in window if query.lower() in c.lower()]
        records = [
            MemoryRecord(memory_id=f"m{i}", content=c) for i, c in enumerate(matches[:top_k])
        ]
        return QueryResult(
            records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.5
        )


class AllStoresFailFakeAdapter(MemoryBackendAdapter):
    name = "fake-scale-all-stores-fail"
    env_var = "FAKE_API_KEY"
    supports_update = True

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        raise BackendAPIError(self.name, "simulated network failure")

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        raise BackendAPIError(self.name, "simulated network failure")

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        raise BackendAPIError(self.name, "simulated network failure")

    def delete(self, memory_id: str) -> DeleteResult:
        raise BackendAPIError(self.name, "simulated network failure")


# ---------------------------------------------------------------------------
# scale_fixtures.generate_scale_corpus
# ---------------------------------------------------------------------------


def test_generate_scale_corpus_produces_requested_count() -> None:
    records = generate_scale_corpus(50, seed=1)
    assert len(records) == 50
    assert [r.index for r in records] == list(range(50))


def test_generate_scale_corpus_is_deterministic_for_same_seed() -> None:
    a = generate_scale_corpus(30, seed=7)
    b = generate_scale_corpus(30, seed=7)
    assert [r.content for r in a] == [r.content for r in b]
    assert [r.marker for r in a] == [r.marker for r in b]


def test_generate_scale_corpus_differs_across_seeds() -> None:
    a = generate_scale_corpus(30, seed=1)
    b = generate_scale_corpus(30, seed=2)
    assert [r.content for r in a] != [r.content for r in b]


def test_generate_scale_corpus_markers_are_unique() -> None:
    records = generate_scale_corpus(1000, seed=3)
    markers = {r.marker for r in records}
    assert len(markers) == 1000


def test_generate_scale_corpus_marker_embedded_in_content() -> None:
    records = generate_scale_corpus(10, seed=4)
    for record in records:
        assert record.marker in record.content


def test_generate_scale_corpus_rejects_n_below_one() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        generate_scale_corpus(0)


# ---------------------------------------------------------------------------
# _default_checkpoints / _percentile -- small pure helpers
# ---------------------------------------------------------------------------


def test_default_checkpoints_ascending_and_bounded() -> None:
    checkpoints = _default_checkpoints(500)
    assert checkpoints == sorted(checkpoints)
    assert checkpoints[0] >= 1
    assert checkpoints[-1] == 500
    assert len(checkpoints) >= 2


def test_default_checkpoints_small_n_still_has_at_least_two() -> None:
    checkpoints = _default_checkpoints(5)
    assert len(checkpoints) >= 2
    assert checkpoints[-1] == 5


def test_percentile_empty_is_none() -> None:
    assert _percentile([], 99) is None


def test_percentile_single_value() -> None:
    assert _percentile([42.0], 99) == 42.0


def test_percentile_p99_of_uniform_range() -> None:
    values = [float(i) for i in range(1, 101)]
    p99 = _percentile(values, 99)
    assert p99 is not None
    assert p99 > 95.0


# ---------------------------------------------------------------------------
# run_scale_stress_eval -- the eval itself
# ---------------------------------------------------------------------------


def test_clean_adapter_classified_worked_at_scale() -> None:
    adapter = ScaleCleanFakeAdapter()
    result = run_scale_stress_eval(adapter, n_records=500, seed=42)

    assert result.signal == ScaleSignal.WORKED_AT_SCALE
    assert result.records_stored == 500
    assert result.records_store_errors == 0
    assert result.anchor_lost_at_n is None
    assert result.records_recoverable == result.records_checked
    assert result.recall_degradation_pct is not None
    assert result.recall_degradation_pct == 0.0
    assert result.latency_p99_ms is not None


def test_empty_at_volume_adapter_classified_silently_degraded() -> None:
    """Reproduces the volcengine/OpenViking#2850 shape: recall is perfect
    at the small checkpoints and collapses to zero once the corpus passes
    EMPTY_AT_VOLUME_THRESHOLD -- exactly the "works at N=5, breaks at
    N=1000+" pattern this eval exists to catch."""
    adapter = ScaleEmptyAtVolumeFakeAdapter()
    result = run_scale_stress_eval(adapter, n_records=500, seed=42)

    assert result.signal == ScaleSignal.SILENTLY_DEGRADED_AT_SCALE
    # The smallest checkpoints (well under the threshold) should have had
    # perfect recall -- this backend genuinely "worked at N=5".
    small_checkpoints = [c for c in result.checkpoints if c.checkpoint_n <= 50]
    assert small_checkpoints
    assert all(c.recall_rate == 1.0 for c in small_checkpoints)
    # The largest checkpoint (past the threshold) should have collapsed.
    largest = result.checkpoints[-1]
    assert largest.checkpoint_n == 500
    assert largest.recall_rate == 0.0
    assert result.recall_degradation_pct is not None
    assert result.recall_degradation_pct > 50.0


def test_evicts_old_adapter_loses_anchor_and_is_classified_degraded() -> None:
    """Reproduces the getzep/graphiti#1275 shape: the very first record
    ingested becomes silently unrecoverable once enough later records have
    pushed it out of the backend's effective search window, even though
    recently-added content is still fine."""
    adapter = ScaleEvictsOldFakeAdapter()
    result = run_scale_stress_eval(adapter, n_records=500, seed=42)

    assert result.signal == ScaleSignal.SILENTLY_DEGRADED_AT_SCALE
    assert result.anchor_lost_at_n is not None
    # The anchor was recoverable at the small checkpoints (5, 50 <
    # EVICTION_WINDOW_SIZE) and lost by the time the corpus grew past the
    # window size.
    assert result.anchor_lost_at_n > EVICTION_WINDOW_SIZE - 1 or result.anchor_lost_at_n >= 250
    first_checkpoint = result.checkpoints[0]
    assert first_checkpoint.anchor_recall is True


def test_all_stores_fail_classified_error_not_not_applicable() -> None:
    adapter = AllStoresFailFakeAdapter()
    result = run_scale_stress_eval(adapter, n_records=50, seed=1)

    assert result.signal == ScaleSignal.ERROR
    assert result.records_stored == 0
    assert result.records_store_errors == 50
    assert result.error is not None


def test_n_records_below_one_raises() -> None:
    adapter = ScaleCleanFakeAdapter()
    with pytest.raises(ValueError, match="n_records must be >= 1"):
        run_scale_stress_eval(adapter, n_records=0)


def test_default_n_records_is_fast_enough_for_ci() -> None:
    """Documents the deliberate CI-speed default -- see
    scale_stress.py's DEFAULT_N_RECORDS docstring. Not a timing
    assertion (too flaky across CI runners); just confirms the constant
    itself stays well below the 10K+ regime the real vendor bugs need."""
    assert 0 < DEFAULT_N_RECORDS <= 2000


def test_explicit_checkpoints_are_honored() -> None:
    adapter = ScaleCleanFakeAdapter()
    result = run_scale_stress_eval(adapter, n_records=20, seed=1, checkpoints=[3, 10, 20])

    assert [c.checkpoint_n for c in result.checkpoints] == [3, 10, 20]


def test_checkpoints_outside_range_are_filtered() -> None:
    adapter = ScaleCleanFakeAdapter()
    result = run_scale_stress_eval(adapter, n_records=10, seed=1, checkpoints=[0, 5, 999])

    assert [c.checkpoint_n for c in result.checkpoints] == [5]


# ---------------------------------------------------------------------------
# classify_scale_result -- pure classification logic
# ---------------------------------------------------------------------------


def test_classify_not_applicable_with_fewer_than_two_scoreable_checkpoints() -> None:
    checkpoints = [ScaleCheckpointResult(checkpoint_n=5, records_stored_so_far=5, recall_rate=1.0)]
    signal = classify_scale_result(checkpoints, recall_degradation_pct=None, anchor_lost_at_n=None)
    assert signal == ScaleSignal.NOT_APPLICABLE


def test_classify_worked_at_scale_when_stable_high_recall() -> None:
    checkpoints = [
        ScaleCheckpointResult(checkpoint_n=5, records_stored_so_far=5, recall_rate=1.0),
        ScaleCheckpointResult(checkpoint_n=500, records_stored_so_far=500, recall_rate=1.0),
    ]
    signal = classify_scale_result(checkpoints, recall_degradation_pct=0.0, anchor_lost_at_n=None)
    assert signal == ScaleSignal.WORKED_AT_SCALE


def test_classify_partial_degradation_when_consistently_imperfect() -> None:
    checkpoints = [
        ScaleCheckpointResult(checkpoint_n=5, records_stored_so_far=5, recall_rate=0.8),
        ScaleCheckpointResult(checkpoint_n=500, records_stored_so_far=500, recall_rate=0.8),
    ]
    signal = classify_scale_result(checkpoints, recall_degradation_pct=0.0, anchor_lost_at_n=None)
    assert signal == ScaleSignal.PARTIAL_DEGRADATION


def test_classify_silently_degraded_on_anchor_loss_alone() -> None:
    checkpoints = [
        ScaleCheckpointResult(checkpoint_n=5, records_stored_so_far=5, recall_rate=1.0),
        ScaleCheckpointResult(checkpoint_n=500, records_stored_so_far=500, recall_rate=0.95),
    ]
    signal = classify_scale_result(checkpoints, recall_degradation_pct=5.0, anchor_lost_at_n=250)
    assert signal == ScaleSignal.SILENTLY_DEGRADED_AT_SCALE
