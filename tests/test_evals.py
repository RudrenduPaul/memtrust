"""Eval runner tests. All run against the bundled synthetic fixtures in
tests/fixtures/ through in-memory fake adapters -- no real backend or LLM
API calls. This is what proves the harness's scoring logic works, fully
offline and deterministically, independent of whether any live vendor
credentials are ever configured.
"""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

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
from memtrust.evals.contradiction import (
    ContradictionCase,
    classify_case,
    run_contradiction_eval,
)
from memtrust.evals.contradiction import (
    load_dataset as load_contradiction_dataset,
)
from memtrust.evals.locomo import load_dataset as load_locomo_dataset
from memtrust.evals.locomo import run_locomo
from memtrust.evals.longmemeval import load_dataset as load_longmemeval_dataset
from memtrust.evals.longmemeval import run_longmemeval
from memtrust.evals.resource_sync_safety import (
    ResourceSyncSignal,
    classify_resource_sync_file,
    run_resource_sync_eval,
)
from memtrust.evals.resource_sync_safety import (
    load_dataset as load_resource_sync_dataset,
)
from memtrust.scoring.llm_judge import LLMJudge


class RecallAllFakeAdapter(MemoryBackendAdapter):
    """Returns every stored fact for a session on every query -- old and
    new both remain visible, which the contradiction eval should classify
    as FLAGGED (the conflict is visible in the response)."""

    name = "fake-recall-all"
    env_var = "FAKE_API_KEY"
    supports_update = True

    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}
        self.store_calls = 0

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        self.store_calls += 1
        self._store.setdefault(session_id, []).append(content)
        return StoreResult(memory_id=f"{session_id}-{len(self._store[session_id])}", latency_ms=0.1)

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        contents = self._store.get(session_id, [])[-top_k:]
        records = [MemoryRecord(memory_id=f"m{i}", content=c) for i, c in enumerate(contents)]
        return QueryResult(
            records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        result = self.store(session_id, content)
        return UpdateResult(memory_id=result.memory_id, acknowledged=True, latency_ms=0.1)

    def delete(self, memory_id: str) -> DeleteResult:
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=0.1)


class OverwriteFakeAdapter(RecallAllFakeAdapter):
    """Only ever returns the single most recent fact -- simulates a
    backend that silently overwrites."""

    name = "fake-overwrite"

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        contents = self._store.get(session_id, [])
        latest = contents[-1:] if contents else []
        records = [MemoryRecord(memory_id="m0", content=c) for c in latest]
        return QueryResult(
            records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )


class StaleFakeAdapter(RecallAllFakeAdapter):
    """Only ever returns the first fact ever stored -- simulates a
    backend that keeps serving stale data after an update."""

    name = "fake-stale"

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        contents = self._store.get(session_id, [])
        first = contents[:1]
        records = [MemoryRecord(memory_id="m0", content=c) for c in first]
        return QueryResult(
            records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )


class NoUpdateFakeAdapter(RecallAllFakeAdapter):
    name = "fake-no-update"
    supports_update = False


class ResourceSyncFakeAdapter(MemoryBackendAdapter):
    """In-memory adapter that models a directory/resource mirror, used to
    exercise the resource-sync-safety eval without a real OpenViking-shaped
    backend. `drop_origin`, when set, makes trigger_resync() silently
    remove every stored file whose seeded `origin` metadata matches it --
    this is the exact volcengine/OpenViking#3029 shape: a resync mechanism
    dropping files it did not itself generate, with no error raised."""

    name = "fake-resource-sync"
    env_var = "FAKE_API_KEY"
    supports_update = True
    supports_resource_sync = True

    def __init__(self, drop_origin: str | None = None) -> None:
        self._files: dict[str, dict[str, tuple[str, str]]] = {}
        self._drop_origin = drop_origin

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        metadata = metadata or {}
        path = f"{session_id}/{metadata.get('resource_path', content[:12])}"
        origin = metadata.get("origin", "unknown")
        self._files.setdefault(session_id, {})[path] = (content, origin)
        return StoreResult(memory_id=path, latency_ms=0.1)

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        matches = [
            MemoryRecord(memory_id=p, content=c)
            for p, (c, _origin) in self._files.get(session_id, {}).items()
            if query.lower() in c.lower()
        ][:top_k]
        return QueryResult(
            records=matches, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        return UpdateResult(memory_id=memory_id, acknowledged=True, latency_ms=0.1)

    def delete(self, memory_id: str) -> DeleteResult:
        raise BackendAPIError(self.name, "not implemented for this fake adapter")

    def list_resource_paths(self, prefix: str) -> list[str]:
        return list(self._files.get(prefix, {}).keys())

    def trigger_resync(self, prefix: str) -> None:
        if self._drop_origin is None:
            return
        files = self._files.get(prefix, {})
        self._files[prefix] = {
            path: value for path, value in files.items() if value[1] != self._drop_origin
        }


class FailingFakeAdapter(MemoryBackendAdapter):
    name = "fake-failing"
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
# Contradiction eval -- the most important eval in the repo
# ---------------------------------------------------------------------------


def test_contradiction_dataset_loads() -> None:
    cases = load_contradiction_dataset()
    assert len(cases) == 7
    assert all(isinstance(c, ContradictionCase) for c in cases)


def test_contradiction_dataset_includes_lucene_trigger_and_metadata_cases() -> None:
    """The two cases added for the self-hosted graphiti-core adapter build:
    one whose query contains every uppercase letter (O/R/N/T/A/D)
    getzep/graphiti#1302's lucene_sanitize() mis-escapes, and one carrying
    non-fact structured metadata for the MemoryRecord.attributes boundary.
    """
    cases = {c.case_id: c for c in load_contradiction_dataset()}
    lucene_case = cases["mt-contra-006"]
    for letter in "ORNTAD":
        assert letter in lucene_case.query
    metadata_case = cases["mt-contra-007"]
    assert metadata_case.metadata == {
        "ticket_id": "OPS-4471",
        "team": "platform-infra",
        "category": "structured-non-fact",
    }
    # Every pre-existing case's metadata must still default to empty --
    # this is a purely additive field.
    assert cases["mt-contra-001"].metadata == {}


def test_recall_all_adapter_classified_flagged() -> None:
    adapter = RecallAllFakeAdapter()
    result = run_contradiction_eval(adapter)
    assert result.flagged_rate == 1.0
    assert result.silent_overwrite_rate == 0.0
    assert result.served_stale_rate == 0.0


def test_overwrite_adapter_classified_silent_overwrite() -> None:
    adapter = OverwriteFakeAdapter()
    result = run_contradiction_eval(adapter)
    assert result.silent_overwrite_rate == 1.0
    assert result.flagged_rate == 0.0


def test_stale_adapter_classified_served_stale() -> None:
    adapter = StaleFakeAdapter()
    result = run_contradiction_eval(adapter)
    assert result.served_stale_rate == 1.0
    assert result.flagged_rate == 0.0


def test_no_update_adapter_all_not_applicable_and_never_called() -> None:
    """A genuinely-no-update-primitive backend (supports_update=False)
    must still report NOT_APPLICABLE unchanged -- EMPTY_OR_LOST is only
    ever assigned to a *capable* backend that ran the calls and came back
    empty, not to a backend that structurally cannot be evaluated here."""
    adapter = NoUpdateFakeAdapter()
    result = run_contradiction_eval(adapter)
    assert result.not_applicable_rate == 1.0
    assert result.empty_or_lost_rate == 0.0
    assert adapter.store_calls == 0


def test_failing_adapter_records_error_without_crashing() -> None:
    adapter = FailingFakeAdapter()
    result = run_contradiction_eval(adapter)
    assert len(result.case_results) == 7
    assert all(c.error is not None for c in result.case_results)
    assert result.scored_cases == []
    assert result.flagged_rate is None


@pytest.mark.parametrize(
    ("has_initial", "has_updated", "irrelevant_content", "adapter_signal", "expected"),
    [
        (True, True, False, ConflictSignal.NOT_APPLICABLE, ConflictSignal.FLAGGED),
        (False, True, False, ConflictSignal.NOT_APPLICABLE, ConflictSignal.SILENT_OVERWRITE),
        (True, False, False, ConflictSignal.NOT_APPLICABLE, ConflictSignal.SERVED_STALE),
        # Records came back but contain neither value -- a genuine "eval
        # could not observe anything meaningful" case, still NOT_APPLICABLE
        # regardless of what the adapter itself claimed.
        (False, False, True, ConflictSignal.NOT_APPLICABLE, ConflictSignal.NOT_APPLICABLE),
        (False, False, True, ConflictSignal.FLAGGED, ConflictSignal.NOT_APPLICABLE),
        # Zero records at all from a capable backend (classify_case is only
        # ever invoked when adapter.supports_update is True -- see
        # run_contradiction_eval) is the "call succeeded, produced nothing"
        # failure mode -- EMPTY_OR_LOST, never NOT_APPLICABLE, regardless
        # of what the adapter self-reported.
        (False, False, False, ConflictSignal.NOT_APPLICABLE, ConflictSignal.EMPTY_OR_LOST),
        (False, False, False, ConflictSignal.FLAGGED, ConflictSignal.EMPTY_OR_LOST),
    ],
)
def test_classify_case_matrix(
    has_initial: bool,
    has_updated: bool,
    irrelevant_content: bool,
    adapter_signal: ConflictSignal,
    expected: ConflictSignal,
) -> None:
    case = ContradictionCase(
        case_id="c1",
        session_id="s1",
        subject="test",
        initial_fact="fact A",
        contradicting_fact="fact B",
        query="q",
        initial_value="OLDVALUE",
        updated_value="NEWVALUE",
    )
    content_parts: list[str] = []
    if has_initial:
        content_parts.append("OLDVALUE")
    if has_updated:
        content_parts.append("NEWVALUE")
    if irrelevant_content:
        content_parts.append("something else entirely")
    records = (
        [MemoryRecord(memory_id="m0", content=" ".join(content_parts))] if content_parts else []
    )
    query_result = QueryResult(records=records, conflict_signal=adapter_signal, latency_ms=0.1)
    signal, got_initial, got_updated = classify_case(case, query_result)
    assert signal == expected
    assert got_initial == has_initial
    assert got_updated == has_updated


def test_classify_case_capable_backend_empty_result_is_not_not_applicable() -> None:
    """The exact gap this fix closes: a capable backend (supports_update
    True) whose query() call succeeded with no exception but returned zero
    records must be distinguishable from a backend with no update
    primitive at all. Both used to collapse into NOT_APPLICABLE."""
    case = ContradictionCase(
        case_id="c1",
        session_id="s1",
        subject="test",
        initial_fact="fact A",
        contradicting_fact="fact B",
        query="q",
        initial_value="OLDVALUE",
        updated_value="NEWVALUE",
    )
    empty_result = QueryResult(
        records=[], conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, _, _ = classify_case(case, empty_result)
    assert signal == ConflictSignal.EMPTY_OR_LOST
    assert signal != ConflictSignal.NOT_APPLICABLE


class EmptyButCapableFakeAdapter(RecallAllFakeAdapter):
    """Simulates a real-world "silent empty success": store()/update() both
    succeed with no exception, but query() always returns zero records --
    the exact MemPalace/OpenViking/mem0 failure mode this fix targets."""

    name = "fake-empty-capable"

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        return QueryResult(
            records=[], conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )


def test_empty_capable_adapter_classified_empty_or_lost_end_to_end() -> None:
    """Full run_contradiction_eval pipeline: a capable adapter that calls
    store()/update() successfully but always returns an empty query result
    must surface as EMPTY_OR_LOST in the aggregated results, not silently
    default to NOT_APPLICABLE or a plain miss."""
    adapter = EmptyButCapableFakeAdapter()
    result = run_contradiction_eval(adapter)
    assert adapter.store_calls > 0
    assert result.empty_or_lost_rate == 1.0
    assert result.not_applicable_rate == 0.0
    assert all(c.signal == ConflictSignal.EMPTY_OR_LOST for c in result.scored_cases)


def _contradiction_case() -> ContradictionCase:
    return ContradictionCase(
        case_id="c1",
        session_id="s1",
        subject="test",
        initial_fact="the meeting is at 2pm",
        contradicting_fact="the meeting moved to 3pm",
        query="what time is the meeting?",
        initial_value="2pm",
        updated_value="3pm",
    )


def test_classify_case_neither_value_present_no_metadata_stays_not_applicable() -> None:
    """The 'neither value observed' branch with zero adapter-reported
    metadata and zero text evidence has genuinely nothing to read --
    NOT_APPLICABLE is the only defensible verdict here."""
    case = _contradiction_case()
    records = [MemoryRecord(memory_id="m0", content="totally unrelated content")]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, has_initial, has_updated = classify_case(case, query_result)
    assert signal == ConflictSignal.NOT_APPLICABLE
    assert has_initial is False
    assert has_updated is False


def test_classify_case_neither_value_present_but_invalidation_metadata_flags() -> None:
    """Same 'neither value observed via substring match' shape as the test
    above, but this time a record carries adapter-reported invalidation
    metadata (Graphiti's invalid_at). This must NOT collapse to the same
    NOT_APPLICABLE verdict as the no-metadata case above -- proving the
    formerly dead-code branch now genuinely differentiates two distinct
    inputs into two distinct outputs, not just a renamed no-op."""
    case = _contradiction_case()
    records = [
        MemoryRecord(
            memory_id="m0",
            content="totally unrelated content",
            metadata={"invalid_at": "2026-06-01T00:00:00Z"},
        )
    ]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, has_initial, has_updated = classify_case(case, query_result)
    assert signal == ConflictSignal.FLAGGED
    assert has_initial is False
    assert has_updated is False


def test_classify_case_graphiti_invalid_at_overrides_silent_overwrite() -> None:
    """Reproduces the getzep/graphiti#1489-shaped false negative: the
    top-k search window surfaces the live edge (new value) plus an
    invalidated edge whose extracted text doesn't literally contain the
    case's old-value substring (paraphrased extraction is realistic for
    a knowledge-graph backend). Naive substring classification alone
    would call this SILENT_OVERWRITE even though the backend genuinely
    preserved the old fact bi-temporally. Consulting record metadata
    must classify this as FLAGGED instead."""
    case = _contradiction_case()
    records = [
        MemoryRecord(
            memory_id="e1",
            content="the meeting was rescheduled",
            metadata={"invalid_at": "2026-06-01T00:00:00Z"},
        ),
        MemoryRecord(memory_id="e2", content="the meeting is at 3pm", metadata={}),
    ]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.SERVED_STALE, latency_ms=0.1
    )
    signal, has_initial, has_updated = classify_case(case, query_result)
    assert signal == ConflictSignal.FLAGGED
    assert has_initial is False
    assert has_updated is True


@pytest.mark.parametrize(
    ("has_initial", "has_updated", "expected"),
    [
        (True, True, ConflictSignal.FLAGGED),
        (False, True, ConflictSignal.SILENT_OVERWRITE),
        (True, False, ConflictSignal.SERVED_STALE),
        (False, False, ConflictSignal.NOT_APPLICABLE),
    ],
)
def test_classify_case_no_metadata_adapters_unaffected_by_fix(
    has_initial: bool, has_updated: bool, expected: ConflictSignal
) -> None:
    """Mem0, OpenViking, and MemPalace's query() responses (as currently
    implemented, aside from MemPalace's own separate 'invalidated' key
    which this fix does not touch) never populate MemoryRecord.metadata
    with an 'invalid_at' entry. This locks in that the metadata-aware
    fix is a strict addition -- it must not change any classification
    outcome for records with empty/unrelated metadata."""
    case = _contradiction_case()
    content_parts: list[str] = []
    if has_initial:
        content_parts.append(case.initial_value)
    if has_updated:
        content_parts.append(case.updated_value)
    if not content_parts:
        # A non-empty-but-irrelevant record, not an empty records list --
        # the empty-records case is EMPTY_OR_LOST, a distinct signal
        # covered by test_classify_case_capable_backend_empty_result_is_not_not_applicable
        # above. This parametrize case is specifically about "records came
        # back but matched neither value, with no metadata," not "no
        # records came back at all."
        content_parts.append("something else entirely")
    records = [MemoryRecord(memory_id="m0", content=" ".join(content_parts), metadata={})]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, got_initial, got_updated = classify_case(case, query_result)
    assert signal == expected
    assert got_initial == has_initial
    assert got_updated == has_updated


# ---------------------------------------------------------------------------
# EDGE_INTEGRITY_VIOLATION -- the structural check added for
# ZepGraphitiSelfHostedAdapter, catching the shape of getzep/graphiti#1013
# (Neo4j bulk edge-save omitting attributes/reference_time -- fixed
# upstream) and #1001 (FalkorDB's old add_triplet() never setting edge
# endpoint UUIDs at all -- closed via #1013).
# ---------------------------------------------------------------------------


def test_classify_case_edge_integrity_violation_when_source_uuid_missing() -> None:
    """(b) classify_case() flags EDGE_INTEGRITY_VIOLATION when an
    edge-shaped record (raw carries both endpoint-uuid keys) has a
    missing/falsy source_node_uuid."""
    case = _contradiction_case()
    records = [
        MemoryRecord(
            memory_id="e1",
            content="the meeting is at 3pm",
            raw={"source_node_uuid": None, "target_node_uuid": "node-2"},
        )
    ]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, _, _ = classify_case(case, query_result)
    assert signal == ConflictSignal.EDGE_INTEGRITY_VIOLATION


def test_classify_case_edge_integrity_violation_when_target_uuid_missing() -> None:
    case = _contradiction_case()
    records = [
        MemoryRecord(
            memory_id="e1",
            content="the meeting is at 3pm",
            raw={"source_node_uuid": "node-1", "target_node_uuid": ""},
        )
    ]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, _, _ = classify_case(case, query_result)
    assert signal == ConflictSignal.EDGE_INTEGRITY_VIOLATION


def test_classify_case_edge_integrity_violation_takes_priority_over_flagged() -> None:
    """Even when the retrieved text would otherwise satisfy both
    initial_value and updated_value (an ordinary FLAGGED case), a
    structurally broken edge endpoint overrides it -- a broken edge is a
    more fundamental failure than which values the text happens to
    contain."""
    case = _contradiction_case()
    records = [
        MemoryRecord(
            memory_id="e1",
            content="the meeting is at 2pm and now 3pm",
            raw={"source_node_uuid": "node-1", "target_node_uuid": None},
        )
    ]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, _, _ = classify_case(case, query_result)
    assert signal == ConflictSignal.EDGE_INTEGRITY_VIOLATION


def test_classify_case_edge_shaped_record_with_both_endpoints_present_unaffected() -> None:
    """Records that carry both endpoint-uuid keys with real values must
    classify exactly as before this change -- this is a strict addition,
    never a change to any pre-existing classification outcome."""
    case = _contradiction_case()
    records = [
        MemoryRecord(
            memory_id="e1",
            content="the meeting is at 3pm",
            raw={"source_node_uuid": "node-1", "target_node_uuid": "node-2"},
        )
    ]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, has_initial, has_updated = classify_case(case, query_result)
    assert signal == ConflictSignal.SILENT_OVERWRITE
    assert has_initial is False
    assert has_updated is True


def test_classify_case_non_edge_record_unaffected_by_integrity_check() -> None:
    """A record whose raw fragment has neither endpoint-uuid key at all
    (e.g. Mem0/MemPalace/OpenViking's raw shapes) never triggers
    EDGE_INTEGRITY_VIOLATION -- this is a structural check on edge-shaped
    records only, not a generic "does this record have an id" check."""
    case = _contradiction_case()
    records = [MemoryRecord(memory_id="m1", content="the meeting is at 3pm", raw={"id": "m1"})]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, _, _ = classify_case(case, query_result)
    assert signal == ConflictSignal.SILENT_OVERWRITE


def test_run_contradiction_eval_processes_new_fixture_cases_without_error() -> None:
    """(c) the two fixture cases added for this build (mt-contra-006's
    lucene-sanitize-trigger-character query, mt-contra-007's structured
    metadata) load and run through the full eval pipeline without error
    against a capable fake adapter."""
    adapter = RecallAllFakeAdapter()
    result = run_contradiction_eval(adapter)
    case_ids = {c.case.case_id for c in result.case_results}
    assert {"mt-contra-006", "mt-contra-007"} <= case_ids
    assert all(c.error is None for c in result.case_results)
    assert len(result.case_results) == 7


class MetadataCapturingFakeAdapter(RecallAllFakeAdapter):
    """Records every `metadata` value passed to store() -- used to prove
    ContradictionCase.metadata actually threads through
    run_contradiction_eval into the adapter call, not just that the
    dataclass field parses from JSON."""

    name = "fake-metadata-capture"

    def __init__(self) -> None:
        super().__init__()
        self.store_metadata_calls: list[dict[str, str] | None] = []

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        self.store_metadata_calls.append(metadata)
        return super().store(session_id, content, metadata)


def test_run_contradiction_eval_threads_case_metadata_into_store() -> None:
    adapter = MetadataCapturingFakeAdapter()
    run_contradiction_eval(adapter)
    # Only mt-contra-007 carries non-empty metadata, and only on the
    # explicit `adapter.store(case.session_id, case.initial_fact,
    # metadata=...)` call this fix adds -- RecallAllFakeAdapter.update()
    # (like ZepGraphitiAdapter/ZepGraphitiSelfHostedAdapter's own
    # update(), which alias to store()) calls store() a second time per
    # case without threading case.metadata, so every case, including
    # mt-contra-007, contributes one additional None-metadata call. Every
    # other case's metadata is empty and must be threaded through as None
    # (via `metadata or None`), preserving every pre-existing case's exact
    # call shape rather than passing an empty dict.
    non_empty = [m for m in adapter.store_metadata_calls if m]
    assert non_empty == [
        {"ticket_id": "OPS-4471", "team": "platform-infra", "category": "structured-non-fact"}
    ]
    assert len(adapter.store_metadata_calls) == 14
    assert adapter.store_metadata_calls.count(None) == 13


# ---------------------------------------------------------------------------
# Resource-Sync Safety -- catches volcengine/OpenViking#3029-shaped bugs
# ---------------------------------------------------------------------------


def test_resource_sync_dataset_loads() -> None:
    cases = load_resource_sync_dataset()
    assert len(cases) == 3
    assert all(len(c.seed_files) >= 2 for c in cases)


def test_resource_sync_skips_cleanly_for_unsupported_adapter() -> None:
    adapter = RecallAllFakeAdapter()  # supports_resource_sync defaults to False
    result = run_resource_sync_eval(adapter)
    assert result.skipped is True
    assert result.skip_reason is not None
    assert result.file_results == []
    assert result.user_file_deletion_rate is None


def test_resource_sync_all_files_preserved_when_resync_is_safe() -> None:
    adapter = ResourceSyncFakeAdapter(drop_origin=None)
    result = run_resource_sync_eval(adapter)
    assert result.skipped is False
    assert result.user_file_deletion_rate == 0.0
    assert result.preserved_rate == 1.0
    assert all(f.signal == ResourceSyncSignal.PRESERVED for f in result.file_results)


def test_resource_sync_detects_deleted_user_files_matching_issue_3029() -> None:
    adapter = ResourceSyncFakeAdapter(drop_origin="user")
    result = run_resource_sync_eval(adapter)
    assert result.user_file_deletion_rate == 1.0

    user_results = [f for f in result.file_results if f.origin == "user"]
    generated_results = [f for f in result.file_results if f.origin == "generated"]
    assert user_results  # fixture actually seeds user-origin files
    assert generated_results  # fixture actually seeds generated-origin files
    assert all(f.signal == ResourceSyncSignal.DELETED_USER_FILE for f in user_results)
    assert all(f.signal == ResourceSyncSignal.PRESERVED for f in generated_results)


@pytest.mark.parametrize(
    ("present_before", "present_after", "content_matches", "expected"),
    [
        (True, True, True, ResourceSyncSignal.PRESERVED),
        (True, True, None, ResourceSyncSignal.PRESERVED),
        (True, False, None, ResourceSyncSignal.DELETED_USER_FILE),
        (True, True, False, ResourceSyncSignal.OVERWRITTEN_UNCHANGED),
        (False, False, None, ResourceSyncSignal.NOT_APPLICABLE),
        (False, True, True, ResourceSyncSignal.PRESERVED),
    ],
)
def test_classify_resource_sync_file_matrix(
    present_before: bool,
    present_after: bool,
    content_matches: bool | None,
    expected: ResourceSyncSignal,
) -> None:
    signal = classify_resource_sync_file(present_before, present_after, content_matches)
    assert signal == expected


# ---------------------------------------------------------------------------
# LongMemEval
# ---------------------------------------------------------------------------


def test_longmemeval_dataset_loads() -> None:
    examples = load_longmemeval_dataset()
    assert len(examples) == 3
    assert examples[0]["question_type"] == "single-session-user"


def test_longmemeval_runs_offline_and_reports_judge_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MEMTRUST_JUDGE_API_KEY", raising=False)
    adapter = RecallAllFakeAdapter()
    judge = LLMJudge()
    result = run_longmemeval(adapter, judge)
    assert len(result.case_results) == 3
    assert result.judge_unavailable is True
    assert result.accuracy is None


def test_longmemeval_computes_accuracy_when_judge_configured(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("MEMTRUST_JUDGE_API_KEY", "test-key")
    adapter = RecallAllFakeAdapter()
    judge = LLMJudge()
    httpx_mock.add_response(
        method="POST",
        url="https://api.deepseek.com/chat/completions",
        json={
            "choices": [{"message": {"content": "CORRECT\nmatches"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        is_reusable=True,
    )
    result = run_longmemeval(adapter, judge)
    assert result.accuracy == 1.0
    judge.close()


def test_longmemeval_sets_records_empty_on_empty_query_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend whose query() succeeds but returns zero records must have
    records_empty=True on every case result, distinct from an ordinary
    judge-graded miss where the backend at least returned something."""
    monkeypatch.delenv("MEMTRUST_JUDGE_API_KEY", raising=False)
    adapter = EmptyButCapableFakeAdapter()
    judge = LLMJudge()
    result = run_longmemeval(adapter, judge)
    assert len(result.case_results) == 3
    assert all(c.records_empty for c in result.case_results)
    assert result.n_records_empty == 3


def test_longmemeval_records_empty_false_when_records_returned() -> None:
    adapter = RecallAllFakeAdapter()
    judge = LLMJudge()
    result = run_longmemeval(adapter, judge)
    assert all(not c.records_empty for c in result.case_results)
    assert result.n_records_empty == 0


def test_longmemeval_handles_backend_failure() -> None:
    adapter = FailingFakeAdapter()
    judge = LLMJudge()
    result = run_longmemeval(adapter, judge)
    assert len(result.case_results) == 3
    assert all(c.error is not None for c in result.case_results)


# ---------------------------------------------------------------------------
# LoCoMo
# ---------------------------------------------------------------------------


def test_locomo_dataset_loads() -> None:
    conversations = load_locomo_dataset()
    assert len(conversations) == 1
    assert len(conversations[0]["qa"]) == 3


def test_locomo_runs_offline_and_reports_no_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMTRUST_JUDGE_API_KEY", raising=False)
    adapter = RecallAllFakeAdapter()
    judge = LLMJudge()
    result = run_locomo(adapter, judge)
    assert len(result.case_results) == 3
    assert result.accuracy is None


def test_locomo_sets_records_empty_on_empty_query_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same distinction as LongMemEval: a backend that succeeds but
    returns zero records must be flagged via records_empty, not scored
    identically to a normal judge-graded miss."""
    monkeypatch.delenv("MEMTRUST_JUDGE_API_KEY", raising=False)
    adapter = EmptyButCapableFakeAdapter()
    judge = LLMJudge()
    result = run_locomo(adapter, judge)
    assert len(result.case_results) == 3
    assert all(c.records_empty for c in result.case_results)
    assert result.n_records_empty == 3


def test_locomo_records_empty_false_when_records_returned() -> None:
    adapter = RecallAllFakeAdapter()
    judge = LLMJudge()
    result = run_locomo(adapter, judge)
    assert all(not c.records_empty for c in result.case_results)
    assert result.n_records_empty == 0


def test_locomo_accuracy_by_category(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("MEMTRUST_JUDGE_API_KEY", "test-key")
    adapter = RecallAllFakeAdapter()
    judge = LLMJudge()
    httpx_mock.add_response(
        method="POST",
        url="https://api.deepseek.com/chat/completions",
        json={
            "choices": [{"message": {"content": "CORRECT\nmatches"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        is_reusable=True,
    )
    result = run_locomo(adapter, judge)
    by_cat = result.accuracy_by_category()
    assert set(by_cat.keys()) == {"single-hop", "temporal", "multi-hop"}
    assert all(v == 1.0 for v in by_cat.values())
    judge.close()


def test_locomo_handles_backend_failure() -> None:
    adapter = FailingFakeAdapter()
    judge = LLMJudge()
    result = run_locomo(adapter, judge)
    assert len(result.case_results) == 3
    assert all(c.error is not None for c in result.case_results)
