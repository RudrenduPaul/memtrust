"""Tests for the MemPalace MCP metadata-tool scale eval
(src/memtrust/evals/mempalace_metadata_scale.py).

Classification-logic tests below use fake, in-memory MemPalaceAdapter
subclasses -- no chromadb or real `mempalace` package required, matching
the pattern in tests/test_scale_stress.py. The two "breaks at scale" fake
adapters are engineered to reproduce the two real failure shapes this
eval exists to catch:

  * MetadataScaleQuadraticFakeAdapter models MemPalace/mempalace#1871
    (alionar): metadata-overview latency grows quadratically with the
    seeded record count, exactly the repeated-full-collection-scan shape
    the real bug report describes.
  * MetadataScaleWrongCountsFakeAdapter models a backend that responds
    quickly but reports incorrect wing/room counts -- a correctness
    failure this eval must catch independently of (and before) any
    latency judgment.

MetadataScaleCleanFakeAdapter is the negative control: correct counts,
flat latency regardless of scale, must classify WORKED_AT_SCALE.

The real, live seeder (build_chroma_metadata_seeder(), backed by the real
installed `mempalace` package) is exercised separately, in
test_real_chroma_seeder_reports_correct_counts_at_small_scale at the
bottom of this file, which calls `pytest.importorskip("mempalace")`
itself so only that one test skips (not this whole module) when the
optional `mempalace-direct` dependency group isn't installed -- see
mempalace_metadata_scale.py's module docstring for what that live test
does and does not prove.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from memtrust.adapters.base import (
    BackendAPIError,
    DeleteResult,
    MemoryBackendAdapter,
    MetadataCategoryCountsResult,
    MetadataOverviewResult,
    QueryResult,
    StoreResult,
    UpdateResult,
)
from memtrust.evals.mempalace_metadata_scale import (
    DEFAULT_N_RECORDS,
    MetadataScaleCheckpointResult,
    MetadataScaleSignal,
    _default_checkpoints,
    classify_metadata_scale_result,
    run_mempalace_metadata_scale_eval,
)


def _wing(i: int, n_wings: int = 4) -> str:
    return f"wing_{i % n_wings}"


def _room(i: int, n_rooms: int = 10) -> str:
    return f"room_{i % n_rooms}"


def _ground_truth(total_n: int) -> tuple[dict[str, int], dict[str, int]]:
    categories: dict[str, int] = {}
    subcategories: dict[str, int] = {}
    for i in range(total_n):
        w, r = _wing(i), _room(i)
        categories[w] = categories.get(w, 0) + 1
        subcategories[r] = subcategories.get(r, 0) + 1
    return categories, subcategories


class MetadataScaleCleanFakeAdapter(MemoryBackendAdapter):
    """Negative control: always reports correct ground-truth counts, with
    flat latency regardless of scale. Must classify WORKED_AT_SCALE."""

    name = "fake-metadata-scale-clean"
    env_var = "FAKE_API_KEY"
    supports_metadata_overview = True

    def __init__(self) -> None:
        self._current_n = 0

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        raise NotImplementedError

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        raise NotImplementedError

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        raise NotImplementedError

    def delete(self, memory_id: str) -> DeleteResult:
        raise NotImplementedError

    def note_seeded(self, total_n: int) -> None:
        self._current_n = total_n

    def metadata_overview(self) -> MetadataOverviewResult:
        categories, subcategories = _ground_truth(self._current_n)
        return MetadataOverviewResult(
            total_records=self._current_n,
            categories=categories,
            subcategories=subcategories,
            latency_ms=2.0,
        )

    def list_metadata_categories(self) -> MetadataCategoryCountsResult:
        categories, _ = _ground_truth(self._current_n)
        return MetadataCategoryCountsResult(counts=categories, scope=None, latency_ms=1.0)

    def list_metadata_subcategories(
        self, category: str | None = None
    ) -> MetadataCategoryCountsResult:
        _, subcategories = _ground_truth(self._current_n)
        return MetadataCategoryCountsResult(counts=subcategories, scope=category, latency_ms=1.0)


class MetadataScaleQuadraticFakeAdapter(MetadataScaleCleanFakeAdapter):
    """Positive control modeling MemPalace/mempalace#1871: counts are
    always correct, but metadata_overview()'s reported latency grows with
    the square of the current record count -- the repeated-full-
    collection-scan shape alionar's bug report describes."""

    name = "fake-metadata-scale-quadratic"

    def metadata_overview(self) -> MetadataOverviewResult:
        result = super().metadata_overview()
        result.latency_ms = 0.01 * (self._current_n**2) + 1.0
        return result


class MetadataScaleWrongCountsFakeAdapter(MetadataScaleCleanFakeAdapter):
    """Positive control: fast, but always reports empty wing/room counts
    regardless of how much was actually seeded -- a correctness failure
    independent of any latency shape."""

    name = "fake-metadata-scale-wrong-counts"

    def metadata_overview(self) -> MetadataOverviewResult:
        return MetadataOverviewResult(
            total_records=self._current_n, categories={}, subcategories={}, latency_ms=1.0
        )


class MetadataScaleErroringSeederAdapter(MetadataScaleCleanFakeAdapter):
    """Adapter side is fine; used with a seeder that fails outright, to
    exercise the ERROR classification path."""

    name = "fake-metadata-scale-erroring-seeder"


def _seeder_that_fails(total_n: int) -> tuple[dict[str, int], dict[str, int]]:
    raise BackendAPIError("fake-backend", "seed collection failed: disk full")


def _tracking_seeder(
    adapter: MetadataScaleCleanFakeAdapter,
) -> Callable[[int], tuple[dict[str, int], dict[str, int]]]:
    def seed(total_n: int) -> tuple[dict[str, int], dict[str, int]]:
        adapter.note_seeded(total_n)
        return _ground_truth(total_n)

    return seed


# ---------------------------------------------------------------------------
# _default_checkpoints -- small pure helper
# ---------------------------------------------------------------------------


def test_default_checkpoints_ascending_and_bounded() -> None:
    checkpoints = _default_checkpoints(2000)
    assert checkpoints == sorted(checkpoints)
    assert checkpoints[0] == 5
    assert checkpoints[-1] == 2000
    assert all(1 <= c <= 2000 for c in checkpoints)


def test_default_checkpoints_small_n_deduplicates() -> None:
    checkpoints = _default_checkpoints(3)
    assert checkpoints == [1, 3]


def test_default_n_records_is_positive() -> None:
    assert DEFAULT_N_RECORDS > 0


# ---------------------------------------------------------------------------
# classify_metadata_scale_result -- direct classification-logic tests
# ---------------------------------------------------------------------------


def _checkpoint(
    n: int, latency_ms: float, correct: bool = True, error: str | None = None
) -> MetadataScaleCheckpointResult:
    categories, subcategories = _ground_truth(n)
    return MetadataScaleCheckpointResult(
        checkpoint_n=n,
        overview_latency_ms=latency_ms,
        categories_latency_ms=latency_ms,
        subcategories_latency_ms=latency_ms,
        total_records_reported=n,
        categories_reported=categories if correct else {},
        categories_expected=categories,
        subcategories_reported=subcategories if correct else {},
        subcategories_expected=subcategories,
        counts_correct=correct,
        error=error,
    )


def test_classify_worked_at_scale_on_flat_latency() -> None:
    checkpoints = [_checkpoint(5, 2.0), _checkpoint(2000, 2.5)]
    signal, latency_ratio, record_ratio = classify_metadata_scale_result(checkpoints)
    assert signal == MetadataScaleSignal.WORKED_AT_SCALE
    assert latency_ratio is not None and latency_ratio < 2.0
    assert record_ratio == 400.0


def test_classify_superlinear_on_quadratic_latency() -> None:
    checkpoints = [_checkpoint(5, 1.25), _checkpoint(2000, 40001.0)]
    signal, latency_ratio, record_ratio = classify_metadata_scale_result(checkpoints)
    assert signal == MetadataScaleSignal.SUPERLINEAR_LATENCY_GROWTH
    assert latency_ratio is not None
    assert record_ratio is not None


def test_classify_incorrect_counts_takes_priority_over_latency() -> None:
    """A backend that is both slow AND wrong must be reported as wrong,
    not just slow -- correctness is checked first."""
    checkpoints = [_checkpoint(5, 1.0, correct=True), _checkpoint(2000, 99999.0, correct=False)]
    signal, latency_ratio, record_ratio = classify_metadata_scale_result(checkpoints)
    assert signal == MetadataScaleSignal.INCORRECT_COUNTS_AT_SCALE
    assert latency_ratio is None
    assert record_ratio is None


def test_classify_error_when_any_checkpoint_errored() -> None:
    checkpoints = [_checkpoint(5, 1.0), _checkpoint(2000, 1.0, error="seeding failed")]
    signal, _latency_ratio, _record_ratio = classify_metadata_scale_result(checkpoints)
    assert signal == MetadataScaleSignal.ERROR


def test_classify_not_applicable_with_fewer_than_two_scoreable_checkpoints() -> None:
    checkpoints = [_checkpoint(5, 1.0)]
    signal, latency_ratio, record_ratio = classify_metadata_scale_result(checkpoints)
    assert signal == MetadataScaleSignal.NOT_APPLICABLE
    assert latency_ratio is None
    assert record_ratio is None


# ---------------------------------------------------------------------------
# run_mempalace_metadata_scale_eval -- end-to-end against fake adapters
# ---------------------------------------------------------------------------


class NoSupportFakeAdapter(MemoryBackendAdapter):
    """Ordinary adapter with no metadata-overview capability at all --
    the default supports_metadata_overview=False path."""

    name = "fake-no-metadata-support"
    env_var = "FAKE_API_KEY"

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        raise NotImplementedError

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        raise NotImplementedError

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        raise NotImplementedError

    def delete(self, memory_id: str) -> DeleteResult:
        raise NotImplementedError


def test_run_eval_skips_adapter_without_metadata_overview_support() -> None:
    adapter = NoSupportFakeAdapter()

    result = run_mempalace_metadata_scale_eval(adapter, _seeder_that_fails, n_records=100)

    assert result.skipped is True
    assert result.skip_reason is not None
    assert "supports_metadata_overview" in result.skip_reason
    assert result.checkpoints == []


def test_run_eval_worked_at_scale_against_clean_adapter() -> None:
    adapter = MetadataScaleCleanFakeAdapter()

    result = run_mempalace_metadata_scale_eval(adapter, _tracking_seeder(adapter), n_records=2000)

    assert result.skipped is False
    assert result.signal == MetadataScaleSignal.WORKED_AT_SCALE
    assert len(result.checkpoints) == len(_default_checkpoints(2000))
    assert all(c.counts_correct for c in result.checkpoints)


def test_run_eval_flags_superlinear_growth_against_quadratic_adapter() -> None:
    adapter = MetadataScaleQuadraticFakeAdapter()

    result = run_mempalace_metadata_scale_eval(adapter, _tracking_seeder(adapter), n_records=2000)

    assert result.signal == MetadataScaleSignal.SUPERLINEAR_LATENCY_GROWTH
    assert result.latency_ratio is not None
    assert result.record_ratio is not None
    assert result.latency_ratio > result.record_ratio


def test_run_eval_flags_incorrect_counts_against_wrong_counts_adapter() -> None:
    adapter = MetadataScaleWrongCountsFakeAdapter()

    result = run_mempalace_metadata_scale_eval(adapter, _tracking_seeder(adapter), n_records=500)

    assert result.signal == MetadataScaleSignal.INCORRECT_COUNTS_AT_SCALE
    assert any(not c.counts_correct for c in result.checkpoints)


def test_run_eval_error_when_seeder_fails() -> None:
    adapter = MetadataScaleErroringSeederAdapter()

    result = run_mempalace_metadata_scale_eval(adapter, _seeder_that_fails, n_records=100)

    assert result.signal == MetadataScaleSignal.ERROR
    assert result.error is not None
    assert "seed collection failed" in result.error


def test_run_eval_rejects_n_records_below_one() -> None:
    adapter = MetadataScaleCleanFakeAdapter()
    with pytest.raises(ValueError, match="n_records must be >= 1"):
        run_mempalace_metadata_scale_eval(adapter, _tracking_seeder(adapter), n_records=0)


def test_run_eval_respects_explicit_checkpoints() -> None:
    adapter = MetadataScaleCleanFakeAdapter()

    result = run_mempalace_metadata_scale_eval(
        adapter, _tracking_seeder(adapter), n_records=1000, checkpoints=[10, 1000]
    )

    assert [c.checkpoint_n for c in result.checkpoints] == [10, 1000]


# ---------------------------------------------------------------------------
# Real, live seeder integration test -- requires the `mempalace-direct`
# optional dependency group (`pip install -e ".[dev,mempalace-direct]"`).
# Skips cleanly (not a collection error) when it isn't installed, matching
# test_mem0_direct_adapter.py's convention -- but the skip is scoped to
# this one test, not the whole module, so every test above still runs
# fully offline with zero optional dependencies installed.
# ---------------------------------------------------------------------------


def test_real_chroma_seeder_reports_correct_counts_at_small_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercises the real, confirmed mempalace.mcp_server functions
    against a real, locally seeded chromadb-backed palace -- not a fake.
    Kept small (checkpoints capped well below this build's own
    investigation, which confirmed correctness and sub-100ms latency at
    N=20,000) so this test stays fast in CI; see this module's docstring
    and mempalace_metadata_scale.py's module docstring for the larger,
    manually-run investigation this small in-CI check is a regression
    guard for.
    """
    pytest.importorskip(
        "mempalace",
        reason=(
            "requires the optional `mempalace-direct` extra: "
            "pip install -e '.[dev,mempalace-direct]'. See "
            "mempalace_metadata_scale.py's module docstring."
        ),
    )
    from memtrust.adapters.mempalace_adapter import MemPalaceAdapter
    from memtrust.evals.mempalace_metadata_scale import build_chroma_metadata_seeder

    storage_path = os.path.join(str(tmp_path), "palace")
    os.makedirs(storage_path, exist_ok=True)
    monkeypatch.setenv("MEMPALACE_STORAGE_PATH", storage_path)
    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)

    seeder = build_chroma_metadata_seeder(storage_path, n_categories=3, n_subcategories=6)
    adapter = MemPalaceAdapter()

    result = run_mempalace_metadata_scale_eval(
        adapter, seeder, n_records=300, checkpoints=[5, 150, 300]
    )

    assert result.skipped is False
    assert result.error is None
    assert all(c.counts_correct for c in result.checkpoints), result.checkpoints
    assert result.signal in (
        MetadataScaleSignal.WORKED_AT_SCALE,
        MetadataScaleSignal.SUPERLINEAR_LATENCY_GROWTH,
    )
