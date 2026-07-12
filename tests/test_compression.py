"""Tests for the compression/round-trip fidelity eval
(src/memtrust/evals/compression.py). All in-memory fake adapters -- no
real backend or network calls, matching the pattern in tests/test_evals.py.
"""

from __future__ import annotations

from memtrust.adapters.base import (
    BackendAPIError,
    ConflictSignal,
    MemoryBackendAdapter,
    MemoryRecord,
    QueryResult,
    StoreResult,
    UpdateResult,
)
from memtrust.evals.compression import (
    DEFAULT_MODE_LABEL,
    CompressionCase,
    fidelity_ratio,
    load_dataset,
    run_compression_eval,
)


class NoModeFakeAdapter(MemoryBackendAdapter):
    """A backend with no mode variants at all -- exercises the
    `supported_modes == ()` -> single "default" mode path. Returns the
    exact content it was given, verbatim, so its round trip is perfectly
    lossless."""

    name = "fake-no-mode"
    env_var = "FAKE_API_KEY"
    supports_update = True

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self.store_calls: list[str | None] = []
        self.query_calls: list[str | None] = []

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        self.store_calls.append(mode)
        self._store[session_id] = content
        return StoreResult(memory_id=f"{session_id}-m0", latency_ms=0.1)

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        self.query_calls.append(mode)
        content = self._store.get(session_id, "")
        records = [MemoryRecord(memory_id=f"{session_id}-m0", content=content)] if content else []
        return QueryResult(
            records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        raise NotImplementedError


class LossyModeFakeAdapter(MemoryBackendAdapter):
    """Two declared modes: "raw" returns content verbatim, "lossy"
    truncates it to simulate a vendor's lossy "compressed" mode -- this is
    what proves the fidelity metric actually distinguishes a lossless
    round trip from a lossy one, per case, per mode."""

    name = "fake-lossy-mode"
    env_var = "FAKE_API_KEY"
    supports_update = True
    supported_modes = ("raw", "lossy")

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        self._store[session_id] = content
        return StoreResult(memory_id=f"{session_id}-m0", latency_ms=0.1)

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        content = self._store.get(session_id, "")
        if mode == "lossy":
            # Simulate a lossy "compressed" mode: only the first third of
            # the content survives the round trip.
            content = content[: max(1, len(content) // 3)]
        records = [MemoryRecord(memory_id=f"{session_id}-m0", content=content)] if content else []
        return QueryResult(
            records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        raise NotImplementedError


class FailingFakeAdapter(MemoryBackendAdapter):
    name = "fake-failing-compression"
    env_var = "FAKE_API_KEY"
    supports_update = True

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        raise BackendAPIError(self.name, "simulated network failure")

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        raise BackendAPIError(self.name, "simulated network failure")

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        raise BackendAPIError(self.name, "simulated network failure")


# ---------------------------------------------------------------------------
# fidelity_ratio -- the core metric
# ---------------------------------------------------------------------------


def test_fidelity_ratio_perfect_round_trip_scores_near_one() -> None:
    original = "The meeting is at 2pm on Friday."
    assert fidelity_ratio(original, original) == 1.0


def test_fidelity_ratio_truncated_round_trip_scores_measurably_lower() -> None:
    original = (
        "Baxter is a three-year-old golden retriever who loves chasing tennis balls at the "
        "park every morning before breakfast."
    )
    truncated = original[: len(original) // 3]
    score = fidelity_ratio(original, truncated)
    assert score < 1.0
    # Not just "less than 1.0" -- measurably, meaningfully lower, so this
    # metric could actually distinguish a lossy vendor mode from a
    # lossless one rather than returning a near-constant value.
    assert score < 0.6


def test_fidelity_ratio_empty_retrieval_against_nonempty_original_is_zero() -> None:
    assert fidelity_ratio("some stored content", "") == 0.0


def test_fidelity_ratio_both_empty_is_one() -> None:
    assert fidelity_ratio("", "") == 1.0


def test_fidelity_ratio_completely_different_content_scores_low() -> None:
    score = fidelity_ratio("The quick brown fox jumps over the lazy dog.", "xyz 123 !!!")
    assert score < 0.3


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def test_compression_dataset_loads() -> None:
    cases = load_dataset()
    assert len(cases) >= 4
    assert all(isinstance(c, CompressionCase) for c in cases)
    # Fixture set was written to cover short, long, and special-character
    # content -- confirm at least one case of meaningfully different
    # length exists so the eval isn't only exercising one content shape.
    lengths = sorted(len(c.content) for c in cases)
    assert lengths[0] < lengths[-1] / 2


# ---------------------------------------------------------------------------
# run_compression_eval
# ---------------------------------------------------------------------------


def test_no_mode_adapter_runs_once_under_default_and_does_not_crash() -> None:
    adapter = NoModeFakeAdapter()
    result = run_compression_eval(adapter)

    assert result.modes == [DEFAULT_MODE_LABEL]
    assert set(result.mode_results.keys()) == {DEFAULT_MODE_LABEL}
    mode_result = result.mode_results[DEFAULT_MODE_LABEL]
    assert len(mode_result.case_results) == len(load_dataset())
    assert mode_result.mean_fidelity is not None
    assert mode_result.mean_fidelity > 0.99  # verbatim echo -> near-perfect fidelity

    # `mode=None` was actually threaded through, not silently swallowed --
    # the fake records every call it received.
    assert adapter.store_calls == [None] * len(load_dataset())
    assert adapter.query_calls == [None] * len(load_dataset())


def test_lossy_mode_adapter_reports_lower_fidelity_for_lossy_mode() -> None:
    adapter = LossyModeFakeAdapter()
    result = run_compression_eval(adapter)

    assert result.modes == ["raw", "lossy"]
    raw_fidelity = result.mode_results["raw"].mean_fidelity
    lossy_fidelity = result.mode_results["lossy"].mean_fidelity
    assert raw_fidelity is not None
    assert lossy_fidelity is not None
    assert raw_fidelity > 0.99
    assert lossy_fidelity < raw_fidelity
    assert result.fidelity_drop_pp is not None
    assert result.fidelity_drop_pp > 0


def test_failing_adapter_records_error_without_crashing() -> None:
    adapter = FailingFakeAdapter()
    result = run_compression_eval(adapter)

    mode_result = result.mode_results[DEFAULT_MODE_LABEL]
    assert len(mode_result.case_results) == len(load_dataset())
    assert all(c.error is not None for c in mode_result.case_results)
    assert mode_result.scored_cases == []
    assert mode_result.mean_fidelity is None


def test_content_length_and_retrieved_length_are_reported() -> None:
    adapter = NoModeFakeAdapter()
    result = run_compression_eval(adapter)
    for case_result in result.mode_results[DEFAULT_MODE_LABEL].case_results:
        assert case_result.content_length > 0
        assert case_result.retrieved_length == case_result.content_length


def test_fidelity_drop_pp_is_none_for_single_mode_adapter() -> None:
    adapter = NoModeFakeAdapter()
    result = run_compression_eval(adapter)
    assert result.fidelity_drop_pp is None
