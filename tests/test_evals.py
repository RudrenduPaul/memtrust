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


# ---------------------------------------------------------------------------
# Contradiction eval -- the most important eval in the repo
# ---------------------------------------------------------------------------


def test_contradiction_dataset_loads() -> None:
    cases = load_contradiction_dataset()
    assert len(cases) == 5
    assert all(isinstance(c, ContradictionCase) for c in cases)


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
    adapter = NoUpdateFakeAdapter()
    result = run_contradiction_eval(adapter)
    assert result.not_applicable_rate == 1.0
    assert adapter.store_calls == 0


def test_failing_adapter_records_error_without_crashing() -> None:
    adapter = FailingFakeAdapter()
    result = run_contradiction_eval(adapter)
    assert len(result.case_results) == 5
    assert all(c.error is not None for c in result.case_results)
    assert result.scored_cases == []
    assert result.flagged_rate is None


@pytest.mark.parametrize(
    ("has_initial", "has_updated", "adapter_signal", "expected"),
    [
        (True, True, ConflictSignal.NOT_APPLICABLE, ConflictSignal.FLAGGED),
        (False, True, ConflictSignal.NOT_APPLICABLE, ConflictSignal.SILENT_OVERWRITE),
        (True, False, ConflictSignal.NOT_APPLICABLE, ConflictSignal.SERVED_STALE),
        (False, False, ConflictSignal.NOT_APPLICABLE, ConflictSignal.NOT_APPLICABLE),
        (False, False, ConflictSignal.FLAGGED, ConflictSignal.NOT_APPLICABLE),
    ],
)
def test_classify_case_matrix(
    has_initial: bool,
    has_updated: bool,
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
    records = (
        [MemoryRecord(memory_id="m0", content=" ".join(content_parts))] if content_parts else []
    )
    query_result = QueryResult(records=records, conflict_signal=adapter_signal, latency_ms=0.1)
    signal, got_initial, got_updated = classify_case(case, query_result)
    assert signal == expected
    assert got_initial == has_initial
    assert got_updated == has_updated


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
