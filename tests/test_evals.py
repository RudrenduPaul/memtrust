"""Eval runner tests. All run against the bundled synthetic fixtures in
tests/fixtures/ through in-memory fake adapters -- no real backend or LLM
API calls. This is what proves the harness's scoring logic works, fully
offline and deterministically, independent of whether any live vendor
credentials are ever configured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from memtrust.adapters.base import (
    BackendAPIError,
    ConflictSignal,
    DeleteResult,
    EmbeddingDriftSignal,
    ExtractionQualitySignal,
    MemoryBackendAdapter,
    MemoryRecord,
    MigrationFailureResult,
    QueryResult,
    RankingSignal,
    RetrievalWarning,
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
from memtrust.evals.crash_recovery import (
    CrashRecoveryCase,
    CrashRecoverySignal,
    classify_crash_recovery_case,
    run_crash_recovery_eval,
)
from memtrust.evals.crash_recovery import (
    load_dataset as load_crash_recovery_dataset,
)
from memtrust.evals.embedding_drift import (
    classify_embedding_drift_record,
    run_embedding_drift_eval,
)
from memtrust.evals.embedding_drift import (
    load_dataset as load_embedding_drift_dataset,
)
from memtrust.evals.extraction_quality import (
    ExtractionQualityCase,
    classify_extraction_case,
    classify_feedback_loop_case,
    run_extraction_quality_eval,
)
from memtrust.evals.extraction_quality import (
    load_dataset as load_extraction_quality_dataset,
)
from memtrust.evals.locomo import load_dataset as load_locomo_dataset
from memtrust.evals.locomo import load_exclude_question_ids as load_locomo_exclude_question_ids
from memtrust.evals.locomo import run_locomo
from memtrust.evals.longmemeval import load_dataset as load_longmemeval_dataset
from memtrust.evals.longmemeval import run_longmemeval
from memtrust.evals.migration_rollback import (
    MigrationRollbackCase,
    MigrationRollbackSignal,
    classify_migration_rollback_case,
    run_migration_rollback_eval,
)
from memtrust.evals.migration_rollback import (
    load_dataset as load_migration_rollback_dataset,
)
from memtrust.evals.ranking_quality import (
    RankingQualityCase,
    RankingQualitySeedRecord,
    classify_ranking_case,
    run_ranking_quality_eval,
)
from memtrust.evals.ranking_quality import (
    load_dataset as load_ranking_quality_dataset,
)
from memtrust.evals.resource_sync_safety import (
    ResourceSyncSignal,
    classify_resource_sync_file,
    run_resource_sync_eval,
)
from memtrust.evals.resource_sync_safety import (
    load_dataset as load_resource_sync_dataset,
)
from memtrust.scoring.llm_judge import JudgeVerdict, LLMJudge


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


class RankingInsertionOrderFakeAdapter(MemoryBackendAdapter):
    """Always returns records in the order they were stored, completely
    ignoring any ranking-relevant metadata -- models a backend whose
    ranking primitive has degenerated to plain insertion order, the exact
    mempalace/mempalace#1733 shape (Kartalops's 0/45,969-drawers finding)
    regardless of whether the ranking field is constant, absent, or even
    varied on the stored records."""

    name = "fake-ranking-insertion-order"
    env_var = "FAKE_API_KEY"

    def __init__(self) -> None:
        self._store: dict[str, list[MemoryRecord]] = {}
        self._counter = 0

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        self._counter += 1
        memory_id = f"m{self._counter}"
        record = MemoryRecord(memory_id=memory_id, content=content, metadata=metadata or {})
        self._store.setdefault(session_id, []).append(record)
        return StoreResult(memory_id=memory_id, latency_ms=0.1)

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        records = self._store.get(session_id, [])[:top_k]
        return QueryResult(
            records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        return UpdateResult(memory_id=memory_id, acknowledged=True, latency_ms=0.1)

    def delete(self, memory_id: str) -> DeleteResult:
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=0.1)


class RankingSortedByImportanceFakeAdapter(RankingInsertionOrderFakeAdapter):
    """Sorts returned records by descending `importance` metadata value --
    models a backend whose ranking signal genuinely works. This is the
    negative control the new eval must NOT flag."""

    name = "fake-ranking-sorted"

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        records = sorted(
            self._store.get(session_id, []),
            key=lambda r: float(r.metadata.get("importance", "0")),
            reverse=True,
        )
        return QueryResult(
            records=records[:top_k], conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )


class RankingReversedFakeAdapter(RankingInsertionOrderFakeAdapter):
    """Returns records in ascending `importance` order -- models a backend
    that has a genuine, varying per-record signal available and still
    doesn't order by it (ORDER_INCONSISTENT, distinct from
    MISSING_ORDERING_KEY)."""

    name = "fake-ranking-reversed"

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        records = sorted(
            self._store.get(session_id, []),
            key=lambda r: float(r.metadata.get("importance", "0")),
        )
        return QueryResult(
            records=records[:top_k], conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )


class CrashRecoveryFakeAdapter(MemoryBackendAdapter):
    """Models the exact volcengine/OpenViking#2644 shape (contributor
    yeyitech): a local vectordb's `_recover()` silently skips rebuilding
    the search index on server-process restart when index files are
    missing but store data exists, so queries silently return nothing
    for data that was never actually lost.

    Maintains two separate in-memory structures on purpose: `_store` (the
    underlying persisted data, what raw_store_contains() reads) and
    `_index` (what query() reads from). `simulate_crash_restart()` here
    drops `_index` and leaves `_store` untouched -- the bug itself."""

    name = "fake-crash-recovery-index-loss"
    env_var = "FAKE_API_KEY"
    supports_crash_recovery_simulation = True

    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}
        self._index: dict[str, dict[str, str]] = {}
        self._counter = 0

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        self._counter += 1
        memory_id = f"m{self._counter}"
        self._store.setdefault(session_id, {})[memory_id] = content
        self._index.setdefault(session_id, {})[memory_id] = content
        return StoreResult(memory_id=memory_id, latency_ms=0.1)

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        matches = [
            MemoryRecord(memory_id=mid, content=c)
            for mid, c in self._index.get(session_id, {}).items()
            if query.lower() in c.lower()
        ][:top_k]
        return QueryResult(
            records=matches, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        return UpdateResult(memory_id=memory_id, acknowledged=True, latency_ms=0.1)

    def delete(self, memory_id: str) -> DeleteResult:
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=0.1)

    def simulate_crash_restart(self) -> None:
        # The bug: index files are treated as missing/unrecoverable, and
        # _recover() never rebuilds them from the store that survived.
        self._index = {}

    def raw_store_contains(self, session_id: str, memory_id: str) -> bool:
        return memory_id in self._store.get(session_id, {})


class CrashRecoveryCleanFakeAdapter(CrashRecoveryFakeAdapter):
    """Models a backend whose `_recover()` correctly rebuilds the index
    from surviving store data on restart -- the negative control this
    eval must NOT flag as INDEX_LOST_DATA_SURVIVED."""

    name = "fake-crash-recovery-clean"

    def simulate_crash_restart(self) -> None:
        self._index = {sid: dict(records) for sid, records in self._store.items()}


class CrashRecoveryDataLostFakeAdapter(CrashRecoveryFakeAdapter):
    """Models a backend that loses the underlying store data itself on
    restart, not just the index -- a different, more severe failure than
    #2644's shape, and this eval must classify it as DATA_LOST rather
    than conflating it with INDEX_LOST_DATA_SURVIVED."""

    name = "fake-crash-recovery-data-lost"

    def simulate_crash_restart(self) -> None:
        self._index = {}
        self._store = {}


class CrashRecoveryFailingFakeAdapter(CrashRecoveryFakeAdapter):
    """A crash-recovery-capable adapter whose store() call itself fails --
    exercises run_crash_recovery_eval()'s BackendAPIError handling path,
    distinct from the unsupported-adapter skip path."""

    name = "fake-crash-recovery-failing"

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        raise BackendAPIError(self.name, "simulated network failure")


class MigrationRollbackFakeAdapter(MemoryBackendAdapter):
    """Models the exact MemPalace/mempalace#1028 shape (GitHub user
    eldar702): an unguarded `shutil.rmtree()`-then-`shutil.move()` swap at
    the end of a migration deletes the old backup FIRST, then attempts to
    move new data into place -- if that move step fails partway (e.g. a
    simulated cross-device EXDEV error, represented here simply as "the
    move never completes"), the old backup is already gone and there is
    nothing left to recover.

    `simulate_migration_failure()` reproduces this precisely: it deletes
    `_store[session_id]` (the "old backup") BEFORE the simulated move
    would land any new data, and never restores it -- the bug itself."""

    name = "fake-migration-rollback-unguarded-swap"
    env_var = "FAKE_API_KEY"
    supports_migration_rollback_simulation = True

    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}
        self._counter = 0

    def _contains(self, session_id: str, content: str) -> bool:
        return any(content.lower() in c.lower() for c in self._store.get(session_id, {}).values())

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        self._counter += 1
        memory_id = f"m{self._counter}"
        self._store.setdefault(session_id, {})[memory_id] = content
        return StoreResult(memory_id=memory_id, latency_ms=0.1)

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        matches = [
            MemoryRecord(memory_id=mid, content=c)
            for mid, c in self._store.get(session_id, {}).items()
            if query.lower() in c.lower()
        ][:top_k]
        return QueryResult(
            records=matches, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        return UpdateResult(memory_id=memory_id, acknowledged=True, latency_ms=0.1)

    def delete(self, memory_id: str) -> DeleteResult:
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=0.1)

    def simulate_migration_failure(self, session_id: str, content: str) -> MigrationFailureResult:
        store_result = self.store(session_id, content)
        # The bug: rmtree() deletes the old backup FIRST, before move()
        # has landed anything new. Simulate move() failing partway (e.g.
        # EXDEV) by simply never restoring what rmtree() just deleted.
        del self._store[session_id]
        recoverable = self._contains(session_id, content)
        return MigrationFailureResult(
            session_id=session_id,
            memory_id=store_result.memory_id,
            content=content,
            original_data_recoverable=recoverable,
        )


class MigrationRollbackRenameAsideFakeAdapter(MigrationRollbackFakeAdapter):
    """Models MemPalace/mempalace#935's real fix: a "rename-aside" swap
    that renames the new data into place first, keeps the old backup
    renamed-aside (never deleted up front), and only deletes the backup
    after independently confirming the swap succeeded. The negative
    control this eval must NOT flag as DATA_LOST."""

    name = "fake-migration-rollback-rename-aside"

    def simulate_migration_failure(self, session_id: str, content: str) -> MigrationFailureResult:
        store_result = self.store(session_id, content)
        # The fix: keep a renamed-aside copy of the old backup instead of
        # deleting it up front. The simulated move step fails here (same
        # EXDEV interruption as the buggy adapter above) BEFORE the
        # commit step that would delete the backup ever runs, so the
        # renamed-aside copy is restored back into place.
        backup = dict(self._store.get(session_id, {}))
        self._store[session_id] = backup
        recoverable = self._contains(session_id, content)
        return MigrationFailureResult(
            session_id=session_id,
            memory_id=store_result.memory_id,
            content=content,
            original_data_recoverable=recoverable,
        )


class MigrationRollbackFailingFakeAdapter(MigrationRollbackFakeAdapter):
    """A migration-rollback-capable adapter whose store() call itself
    fails -- exercises run_migration_rollback_eval()'s BackendAPIError
    handling path, distinct from the unsupported-adapter skip path."""

    name = "fake-migration-rollback-failing"

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        raise BackendAPIError(self.name, "simulated network failure")


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


class EmbeddingDriftCorruptingFakeAdapter(MemoryBackendAdapter):
    """Reproduces the volcengine/OpenViking#1523 bug shape in-memory:
    store() reads the fixture-level `embedding_model_label` metadata tag,
    and whenever a session's *active* label changes (i.e. a "migration"),
    every previously-stored record carrying a *different* label silently
    stops being searchable -- modeling an in-place vector-index overwrite
    with no dimension/model validation. No exception is raised and no
    signal is returned anywhere; the record's content is technically still
    held (unlike a real deletion), it simply can never be matched by
    query() again. Records stored under the label already active when they
    were written are never affected -- only a genuine label change
    triggers the corruption, so a same-label "migration" (see fixture case
    mt-embed-003) never affects retrievability, matching the honest scope
    of what this eval is designed to catch."""

    name = "fake-embedding-drift-corrupting"
    env_var = "FAKE_API_KEY"

    def __init__(self) -> None:
        self._records: dict[str, list[dict[str, Any]]] = {}
        self._active_label: dict[str, str] = {}
        self._counter = 0

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        del mode
        metadata = metadata or {}
        label = metadata.get("embedding_model_label", "unknown")
        records = self._records.setdefault(session_id, [])
        prev_label = self._active_label.get(session_id)
        if prev_label is not None and label != prev_label:
            # Simulated in-place vector-index overwrite: every record
            # carrying a different label than the newly-active one silently
            # stops being searchable -- no exception, no signal.
            for record in records:
                if record["label"] != label:
                    record["searchable"] = False
        self._active_label[session_id] = label
        self._counter += 1
        memory_id = f"m{self._counter}"
        records.append(
            {"memory_id": memory_id, "content": content, "label": label, "searchable": True}
        )
        return StoreResult(memory_id=memory_id, latency_ms=0.1)

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        del mode
        matches = [
            MemoryRecord(
                memory_id=str(r["memory_id"]),
                content=str(r["content"]),
                embedding_model=str(r["label"]),
            )
            for r in self._records.get(session_id, [])
            if r["searchable"] and query.lower() in str(r["content"]).lower()
        ][:top_k]
        return QueryResult(
            records=matches, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        return UpdateResult(memory_id=memory_id, acknowledged=True, latency_ms=0.1)

    def delete(self, memory_id: str) -> DeleteResult:
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=0.1)


class EmbeddingDriftCleanFakeAdapter(MemoryBackendAdapter):
    """Models a backend whose vector store correctly validates/segregates
    embedding-model dimensions: switching the fixture's
    `embedding_model_label` metadata tag mid-session never affects
    retrievability of previously-stored records. This is the negative
    control -- the embedding-drift eval must NOT flag EMBEDDING_DRIFT
    against this adapter."""

    name = "fake-embedding-drift-clean"
    env_var = "FAKE_API_KEY"

    def __init__(self) -> None:
        self._records: dict[str, list[MemoryRecord]] = {}
        self._counter = 0

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        del mode
        metadata = metadata or {}
        self._counter += 1
        memory_id = f"m{self._counter}"
        record = MemoryRecord(
            memory_id=memory_id,
            content=content,
            embedding_model=metadata.get("embedding_model_label"),
        )
        self._records.setdefault(session_id, []).append(record)
        return StoreResult(memory_id=memory_id, latency_ms=0.1)

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        del mode
        matches = [
            r for r in self._records.get(session_id, []) if query.lower() in r.content.lower()
        ][:top_k]
        return QueryResult(
            records=matches, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        return UpdateResult(memory_id=memory_id, acknowledged=True, latency_ms=0.1)

    def delete(self, memory_id: str) -> DeleteResult:
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=0.1)


class NoExtractionGateFakeAdapter(MemoryBackendAdapter):
    """Retains every stored item indiscriminately and returns everything
    stored under a session on every query, regardless of content -- models
    a backend with no effective extraction-quality gate at all, matching
    mem0's real reported behavior per mem0ai/mem0#4573 (jamebobob's audit
    found 97.8% of 10,134 stored entries were junk that should never have
    been kept). Also used as the negative control for the feedback-loop
    duplication tests: a plain store() call here always adds exactly one
    record, so re-storing recalled text should never trigger unexpected
    growth."""

    name = "fake-no-extraction-gate"
    env_var = "FAKE_API_KEY"

    def __init__(self) -> None:
        self._store: dict[str, list[MemoryRecord]] = {}
        self._counter = 0

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        self._counter += 1
        memory_id = f"m{self._counter}"
        record = MemoryRecord(memory_id=memory_id, content=content, metadata=metadata or {})
        self._store.setdefault(session_id, []).append(record)
        return StoreResult(memory_id=memory_id, latency_ms=0.1)

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        records = self._store.get(session_id, [])[:top_k]
        return QueryResult(
            records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        return UpdateResult(memory_id=memory_id, acknowledged=True, latency_ms=0.1)

    def delete(self, memory_id: str) -> DeleteResult:
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=0.1)


class GatedExtractionFakeAdapter(NoExtractionGateFakeAdapter):
    """Models a backend WITH an extraction-quality gate: refuses to
    persist any item whose `category` metadata (threaded through by
    `run_extraction_quality_eval` -- see evals/extraction_quality.py)
    names a junk category. This is a stand-in for "a backend that
    correctly filters junk," not a claim about any real vendor's actual
    LLM-driven extraction pipeline -- no adapter in this repo talks to a
    live one. Its purpose is to prove the eval's classification logic
    correctly credits a backend that filters junk while still retaining
    valid content."""

    name = "fake-gated-extraction"

    _JUNK_CATEGORIES = frozenset(
        {
            "boot_file_restating",
            "cron_heartbeat_noise",
            "system_dump",
            "hallucinated_profile",
            "other_junk",
        }
    )

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        metadata = metadata or {}
        self._counter += 1
        memory_id = f"m{self._counter}"
        if metadata.get("category") in self._JUNK_CATEGORIES:
            # The gate rejects this item outright -- never persisted, so
            # it can never come back on a later query().
            return StoreResult(memory_id=memory_id, latency_ms=0.1)
        record = MemoryRecord(memory_id=memory_id, content=content, metadata=metadata)
        self._store.setdefault(session_id, []).append(record)
        return StoreResult(memory_id=memory_id, latency_ms=0.1)


class FeedbackLoopDuplicatingFakeAdapter(NoExtractionGateFakeAdapter):
    """Models jamebobob's exact production mechanism (mem0ai/mem0#4573):
    the first time a given piece of content is stored it is written once,
    same as any normal backend -- but if that *exact* content is stored a
    second time (as happens when previously-recalled content is fed back
    in and re-extracted), the write fans out into several duplicate
    records instead of one. This is the generalized shape of his real
    808-duplicate finding, scaled down to a small, deterministic fanout
    for a fast unit test."""

    name = "fake-feedback-loop-duplicating"

    #: Number of records a *repeat* store() of the same content produces.
    #: >1 so the eval's growth check (single re-store adds at most 1
    #: record) reliably fires in tests -- the real bug produced 808 from
    #: one call, but the classification logic only cares that growth
    #: exceeded the expected-per-call maximum, not the exact multiplier.
    _DUPLICATE_FANOUT = 5

    def __init__(self) -> None:
        super().__init__()
        self._seen_content: dict[str, set[str]] = {}

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        seen = self._seen_content.setdefault(session_id, set())
        is_repeat = content in seen
        seen.add(content)
        fanout = self._DUPLICATE_FANOUT if is_repeat else 1
        first_id = ""
        for i in range(fanout):
            self._counter += 1
            memory_id = f"m{self._counter}"
            self._store.setdefault(session_id, []).append(
                MemoryRecord(memory_id=memory_id, content=content, metadata=metadata or {})
            )
            if i == 0:
                first_id = memory_id
        return StoreResult(memory_id=first_id, latency_ms=0.1)


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


class DegradedRetrievalFakeAdapter(RecallAllFakeAdapter):
    """Simulates the MemPalace/mempalace#1005 shape at the eval layer:
    store()/query() both succeed, and query() returns real, non-empty
    records -- but the response also carries a RetrievalWarning, the
    "backend warned me it under-delivered, but returned SOME records"
    failure mode that ConflictSignal.EMPTY_OR_LOST structurally cannot
    see (it only fires on zero records)."""

    name = "fake-degraded-retrieval"

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        contents = self._store.get(session_id, [])[-top_k:]
        records = [MemoryRecord(memory_id=f"m{i}", content=c) for i, c in enumerate(contents)]
        return QueryResult(
            records=records,
            conflict_signal=ConflictSignal.NOT_APPLICABLE,
            latency_ms=0.1,
            degraded_retrieval=RetrievalWarning(
                warnings=["hnsw drift detected"], available_in_scope=50
            ),
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
    assert len(cases) == 4
    assert all(len(c.seed_files) >= 2 for c in cases)


def test_resource_sync_dataset_has_a_case_with_real_two_level_nesting() -> None:
    """volcengine/OpenViking#1703's own examples were entities/people/ and
    preferences/{user_id}/ -- at least one fixture case must mirror that
    shape (>=2 real directory levels in path_suffix, not just a single
    origin-folder prefix) so the eval can actually exercise nested-path
    storage, not just flat single-level files under a shared prefix."""
    cases = load_resource_sync_dataset()
    nested = [sf for case in cases for sf in case.seed_files if sf.path_suffix.count("/") >= 2]
    assert nested, "expected at least one seed file with >=2-level real nesting"


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
    ("present_before", "present_after", "content_matches", "indexed_after", "expected"),
    [
        (True, True, True, None, ResourceSyncSignal.PRESERVED),
        (True, True, None, None, ResourceSyncSignal.PRESERVED),
        (True, False, None, None, ResourceSyncSignal.DELETED_USER_FILE),
        (True, True, False, None, ResourceSyncSignal.OVERWRITTEN_UNCHANGED),
        (False, False, None, None, ResourceSyncSignal.NOT_APPLICABLE),
        (False, True, True, None, ResourceSyncSignal.PRESERVED),
        # indexed_after=False -- present on disk, but no query() call ever
        # returned a record for this path at all: the volcengine/
        # OpenViking#1703 "never indexed" shape, distinct from
        # OVERWRITTEN_UNCHANGED (which requires a record to have been
        # found with the wrong content).
        (True, True, False, False, ResourceSyncSignal.NESTED_CONTENT_UNINDEXED),
        (True, True, None, False, ResourceSyncSignal.NESTED_CONTENT_UNINDEXED),
        # indexed_after=True with matching content still classifies as
        # PRESERVED -- indexed_after alone does not override a genuine
        # content match.
        (True, True, True, True, ResourceSyncSignal.PRESERVED),
        # A file never observed present before the resync still has
        # nothing meaningful to classify, regardless of indexed_after.
        (False, False, None, False, ResourceSyncSignal.NOT_APPLICABLE),
    ],
)
def test_classify_resource_sync_file_matrix(
    present_before: bool,
    present_after: bool,
    content_matches: bool | None,
    indexed_after: bool | None,
    expected: ResourceSyncSignal,
) -> None:
    signal = classify_resource_sync_file(
        present_before, present_after, content_matches, indexed_after
    )
    assert signal == expected


def test_classify_resource_sync_file_matrix_backward_compatible_three_arg_call() -> None:
    """Callers that only pass the first three positional args (the
    pre-existing signature) must keep getting the pre-existing behavior --
    indexed_after_resync defaults to None, which never triggers
    NESTED_CONTENT_UNINDEXED."""
    assert (
        classify_resource_sync_file(True, True, False) == ResourceSyncSignal.OVERWRITTEN_UNCHANGED
    )
    assert classify_resource_sync_file(True, True, True) == ResourceSyncSignal.PRESERVED
    assert classify_resource_sync_file(True, False, None) == ResourceSyncSignal.DELETED_USER_FILE


class NestedIndexSkipFakeAdapter(MemoryBackendAdapter):
    """Models volcengine/OpenViking#1703 directly: trigger_resync() never
    deletes or overwrites anything (every path survives on the
    filesystem-mirror side, exactly like the real bug -- index_resource()
    skipped subdirectories during *reindex*, it did not touch storage),
    but query() only ever returns records for paths nested one level deep
    or shallower. Any path nested two or more directory levels deep
    (path_suffix.count("/") >= 2, matching #1703's own entities/people/
    and preferences/{user_id}/ examples) is silently excluded from every
    query() response -- present on disk, never searchable. This is
    deliberately not a deletion: list_resource_paths() reports these
    paths as present both before and after the resync."""

    name = "fake-nested-index-skip"
    env_var = "FAKE_API_KEY"
    supports_update = True
    supports_resource_sync = True

    def __init__(self) -> None:
        self._files: dict[str, dict[str, str]] = {}

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        metadata = metadata or {}
        resource_path = metadata.get("resource_path", content[:12])
        path = f"{session_id}/{resource_path}"
        self._files.setdefault(session_id, {})[path] = content
        return StoreResult(memory_id=path, latency_ms=0.1)

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        matches = [
            MemoryRecord(memory_id=path, content=content)
            for path, content in self._files.get(session_id, {}).items()
            # Only paths nested at most one directory level deep (relative
            # to the session/prefix) get indexed -- mirrors #1703's actual
            # skip-every-subdirectory shape.
            if path.removeprefix(f"{session_id}/").count("/") <= 1
            and query.lower() in content.lower()
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
        return None  # never deletes or overwrites -- only the index lags


def test_resource_sync_detects_nested_content_unindexed_matching_issue_1703() -> None:
    adapter = NestedIndexSkipFakeAdapter()
    result = run_resource_sync_eval(adapter)
    assert result.skipped is False

    nested_results = [f for f in result.file_results if f.path_suffix.count("/") >= 2]
    shallow_results = [f for f in result.file_results if f.path_suffix.count("/") < 2]

    assert nested_results, "fixture must actually seed a >=2-level-nested file"
    assert shallow_results, "fixture must actually seed a shallow (<2-level) file"
    assert all(f.signal == ResourceSyncSignal.NESTED_CONTENT_UNINDEXED for f in nested_results)
    assert all(f.present_after_resync is True for f in nested_results)
    assert all(f.signal == ResourceSyncSignal.PRESERVED for f in shallow_results)
    assert result.nested_content_unindexed_rate == len(nested_results) / len(result.scored_files)
    # Not a deletion -- user_file_deletion_rate stays 0 even though
    # search is broken for the nested files.
    assert result.user_file_deletion_rate == 0.0


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


def test_longmemeval_sets_degraded_retrieval_when_backend_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend that returns real, non-empty records but also signals it
    under-delivered (MemPalace/mempalace#1005's shape) must be tagged
    degraded_retrieval=True and records_empty=False at the same time --
    proving this is tracked separately from the empty-response case."""
    monkeypatch.delenv("MEMTRUST_JUDGE_API_KEY", raising=False)
    adapter = DegradedRetrievalFakeAdapter()
    judge = LLMJudge()
    result = run_longmemeval(adapter, judge)
    assert len(result.case_results) == 3
    assert all(c.degraded_retrieval for c in result.case_results)
    assert all(not c.records_empty for c in result.case_results)
    assert result.n_degraded_retrieval == 3


def test_longmemeval_degraded_retrieval_false_when_backend_clean() -> None:
    adapter = RecallAllFakeAdapter()
    judge = LLMJudge()
    result = run_longmemeval(adapter, judge)
    assert all(not c.degraded_retrieval for c in result.case_results)
    assert result.n_degraded_retrieval == 0


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
    # 3 non-adversarial + 1 adversarial (category 5) case -- see
    # locomo_sample.json and docs/methodology.md's cat-5 note.
    assert len(conversations[0]["qa"]) == 4


def test_locomo_runs_offline_and_reports_no_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMTRUST_JUDGE_API_KEY", raising=False)
    adapter = RecallAllFakeAdapter()
    judge = LLMJudge()
    result = run_locomo(adapter, judge)
    assert len(result.case_results) == 4
    assert result.accuracy is None
    assert result.non_adversarial_accuracy is None


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
    assert len(result.case_results) == 4
    assert all(c.records_empty for c in result.case_results)
    assert result.n_records_empty == 4


def test_locomo_records_empty_false_when_records_returned() -> None:
    adapter = RecallAllFakeAdapter()
    judge = LLMJudge()
    result = run_locomo(adapter, judge)
    assert all(not c.records_empty for c in result.case_results)
    assert result.n_records_empty == 0


def test_locomo_sets_degraded_retrieval_when_backend_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same distinction as LongMemEval: a backend that returns real,
    non-empty records but also signals under-delivered retrieval must be
    tagged degraded_retrieval=True, tracked separately from records_empty."""
    monkeypatch.delenv("MEMTRUST_JUDGE_API_KEY", raising=False)
    adapter = DegradedRetrievalFakeAdapter()
    judge = LLMJudge()
    result = run_locomo(adapter, judge)
    assert len(result.case_results) == 4
    assert all(c.degraded_retrieval for c in result.case_results)
    assert all(not c.records_empty for c in result.case_results)
    assert result.n_degraded_retrieval == 4


def test_locomo_degraded_retrieval_false_when_backend_clean() -> None:
    adapter = RecallAllFakeAdapter()
    judge = LLMJudge()
    result = run_locomo(adapter, judge)
    assert all(not c.degraded_retrieval for c in result.case_results)
    assert result.n_degraded_retrieval == 0


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
    assert set(by_cat.keys()) == {"single-hop", "temporal", "multi-hop", "adversarial"}
    assert all(v == 1.0 for v in by_cat.values())
    judge.close()


def test_locomo_non_adversarial_accuracy_excludes_adversarial_cases(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """Proves the cat-5 exclusion is real, not a no-op: the adversarial
    case is graded INCORRECT (the backend confidently answers an
    unanswerable question, exactly the failure mode category 5 is
    designed to catch) while the 3 non-adversarial cases are graded
    CORRECT. `.accuracy` must fold the adversarial miss into its
    denominator (3/4 = 75%); `.non_adversarial_accuracy` must exclude it
    entirely (3/3 = 100%) -- if the two numbers were equal here, the
    exclusion would not be doing anything."""
    monkeypatch.setenv("MEMTRUST_JUDGE_API_KEY", "test-key")
    adapter = RecallAllFakeAdapter()
    judge = LLMJudge()

    def _judge_callback(request: Any) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        if "kitchen" in prompt:
            content = "INCORRECT\nThe backend fabricated an answer to an unanswerable question."
        else:
            content = "CORRECT\nmatches"
        return httpx.Response(
            status_code=200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    httpx_mock.add_callback(
        _judge_callback,
        method="POST",
        url="https://api.deepseek.com/chat/completions",
        is_reusable=True,
    )
    result = run_locomo(adapter, judge)

    assert len(result.graded_cases) == 4
    assert result.accuracy == pytest.approx(0.75)
    assert result.non_adversarial_accuracy == pytest.approx(1.0)
    assert result.accuracy != result.non_adversarial_accuracy
    assert result.accuracy_by_category()["adversarial"] == 0.0
    judge.close()


def test_locomo_exclude_question_ids_removes_flagged_cases(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """Proves exclude_question_ids genuinely removes a case from scoring
    (never queries or judges it, and it never counts toward accuracy),
    the mechanism a caller would use to score against a corrected
    ground truth once they have a verified list of known-bad question
    IDs (e.g. from an audit like dial481/locomo-audit)."""
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

    conversations = load_locomo_dataset()
    conv_id = conversations[0]["conversation_id"]
    # The multi-hop case is the 3rd qa entry (index 2) in the fixture --
    # exclude it as if an audit had flagged its ground truth as wrong.
    flagged_question_id = f"{conv_id}::2"

    result = run_locomo(adapter, judge, exclude_question_ids={flagged_question_id})

    excluded = [c for c in result.case_results if c.question_id == flagged_question_id]
    assert len(excluded) == 1
    assert excluded[0].excluded_ground_truth is True
    assert excluded[0].verdict == JudgeVerdict.NOT_RUN

    # 4 total cases recorded, but only 3 actually scored -- the excluded
    # one is neither queried nor judged nor counted.
    assert len(result.case_results) == 4
    assert len(result.graded_cases) == 3
    assert result.n_excluded_ground_truth == 1
    assert all(c.question_id != flagged_question_id for c in result.graded_cases)
    judge.close()


def test_locomo_handles_backend_failure() -> None:
    adapter = FailingFakeAdapter()
    judge = LLMJudge()
    result = run_locomo(adapter, judge)
    assert len(result.case_results) == 4
    assert all(c.error is not None for c in result.case_results)


def test_load_exclude_question_ids_from_json_file(tmp_path: Path) -> None:
    path = tmp_path / "exclude.json"
    path.write_text('["mt-locomo-001::2", "mt-locomo-004::0"]')
    assert load_locomo_exclude_question_ids(path) == {"mt-locomo-001::2", "mt-locomo-004::0"}


def test_load_exclude_question_ids_from_text_file(tmp_path: Path) -> None:
    path = tmp_path / "exclude.txt"
    path.write_text(
        "\n".join(
            [
                "# known ground-truth errors from dial481/locomo-audit-style review",
                "mt-locomo-001::2",
                "",
                "mt-locomo-004::0",
            ]
        )
    )
    assert load_locomo_exclude_question_ids(path) == {"mt-locomo-001::2", "mt-locomo-004::0"}


# ---------------------------------------------------------------------------
# Ranking-Quality eval -- distinct from ConflictSignal, closes the
# mempalace/mempalace#1733 gap (Kartalops): correct content, silently
# degenerate order.
# ---------------------------------------------------------------------------


def test_ranking_quality_dataset_loads() -> None:
    cases = load_ranking_quality_dataset()
    assert len(cases) == 4
    assert all(isinstance(c, RankingQualityCase) for c in cases)


def test_ranking_quality_reproduces_1733_constant_importance() -> None:
    """The exact bug shape: every record shares the same `importance`
    value, so a backend that (like MemPalace's real ingest path) never
    writes a real signal degenerates to insertion order. Must fire."""
    adapter = RankingInsertionOrderFakeAdapter()
    result = run_ranking_quality_eval(adapter)
    case_result = next(c for c in result.case_results if c.case.case_id == "mt-rank-001")
    assert case_result.error is None
    assert case_result.signal == RankingSignal.MISSING_ORDERING_KEY
    assert case_result.matches_insertion_order is True


def test_ranking_quality_reproduces_1733_field_never_written() -> None:
    """Even stronger form of #1733: the field isn't merely constant, it
    was never written by any ingest path at all -- same observable
    symptom, same signal."""
    adapter = RankingInsertionOrderFakeAdapter()
    result = run_ranking_quality_eval(adapter)
    case_result = next(c for c in result.case_results if c.case.case_id == "mt-rank-004")
    assert case_result.error is None
    assert case_result.signal == RankingSignal.MISSING_ORDERING_KEY
    assert case_result.field_values == [None, None, None]


def test_ranking_quality_does_not_false_positive_on_genuine_signal() -> None:
    """Negative control this eval must get right: genuinely varied
    importance values that a correctly-behaving backend orders by
    descending value must NOT be flagged as MISSING_ORDERING_KEY."""
    adapter = RankingSortedByImportanceFakeAdapter()
    result = run_ranking_quality_eval(adapter)
    case_result = next(c for c in result.case_results if c.case.case_id == "mt-rank-002")
    assert case_result.error is None
    assert case_result.signal == RankingSignal.SIGNAL_DRIVEN
    present = [v for v in case_result.field_values if v is not None]
    assert present == sorted(present, reverse=True)


def test_ranking_quality_flags_order_inconsistent_when_signal_exists_but_unused() -> None:
    """A backend can carry a genuinely varying signal and still not order
    by it -- distinct from MISSING_ORDERING_KEY, and this eval must tell
    the two apart rather than lumping them into one bucket."""
    adapter = RankingReversedFakeAdapter()
    result = run_ranking_quality_eval(adapter)
    case_result = next(c for c in result.case_results if c.case.case_id == "mt-rank-003")
    assert case_result.error is None
    assert case_result.signal == RankingSignal.ORDER_INCONSISTENT


def test_ranking_quality_handles_backend_failure() -> None:
    adapter = FailingFakeAdapter()
    result = run_ranking_quality_eval(adapter)
    assert len(result.case_results) == 4
    assert all(c.error is not None for c in result.case_results)
    assert all(c.signal == RankingSignal.NOT_APPLICABLE for c in result.case_results)


def test_ranking_quality_eval_result_rates_ignore_errored_cases() -> None:
    adapter = RankingInsertionOrderFakeAdapter()
    result = run_ranking_quality_eval(adapter)
    assert result.missing_ordering_key_rate is not None
    assert 0.0 <= result.missing_ordering_key_rate <= 1.0
    assert len(result.scored_cases) == len(result.case_results)


def _ranking_case(ranking_field: str = "importance") -> RankingQualityCase:
    return RankingQualityCase(
        case_id="rank-direct-test",
        session_id="s",
        query="q",
        ranking_field=ranking_field,
        records=[
            RankingQualitySeedRecord(content="a", metadata={ranking_field: "0.9"}),
            RankingQualitySeedRecord(content="b", metadata={ranking_field: "0.5"}),
            RankingQualitySeedRecord(content="c", metadata={ranking_field: "0.1"}),
        ],
    )


def test_classify_ranking_case_all_identical_is_missing_ordering_key() -> None:
    case = _ranking_case()
    records = [
        MemoryRecord(memory_id="m1", content="a", metadata={"importance": "0.5"}),
        MemoryRecord(memory_id="m2", content="b", metadata={"importance": "0.5"}),
        MemoryRecord(memory_id="m3", content="c", metadata={"importance": "0.5"}),
    ]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, values, matches = classify_ranking_case(case, query_result, ["m1", "m2", "m3"])
    assert signal == RankingSignal.MISSING_ORDERING_KEY
    assert values == [0.5, 0.5, 0.5]
    assert matches is True


def test_classify_ranking_case_field_missing_entirely_is_missing_ordering_key() -> None:
    case = _ranking_case()
    records = [
        MemoryRecord(memory_id="m1", content="a"),
        MemoryRecord(memory_id="m2", content="b"),
    ]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, values, matches = classify_ranking_case(case, query_result, ["m1", "m2"])
    assert signal == RankingSignal.MISSING_ORDERING_KEY
    assert values == [None, None]


def test_classify_ranking_case_partial_field_coverage_is_missing_ordering_key() -> None:
    """A field present on some but not all returned records is just as
    unreliable a signal as a constant/absent one -- no complete real
    per-record signal exists either way."""
    case = _ranking_case()
    records = [
        MemoryRecord(memory_id="m1", content="a", metadata={"importance": "0.9"}),
        MemoryRecord(memory_id="m2", content="b"),
        MemoryRecord(memory_id="m3", content="c", metadata={"importance": "0.1"}),
    ]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, values, matches = classify_ranking_case(case, query_result, ["m1", "m2", "m3"])
    assert signal == RankingSignal.MISSING_ORDERING_KEY


def test_classify_ranking_case_varied_and_sorted_is_signal_driven() -> None:
    case = _ranking_case()
    records = [
        MemoryRecord(memory_id="m1", content="a", metadata={"importance": "0.9"}),
        MemoryRecord(memory_id="m2", content="b", metadata={"importance": "0.5"}),
        MemoryRecord(memory_id="m3", content="c", metadata={"importance": "0.1"}),
    ]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, values, matches = classify_ranking_case(case, query_result, ["m3", "m2", "m1"])
    assert signal == RankingSignal.SIGNAL_DRIVEN
    assert matches is False


def test_classify_ranking_case_varied_but_unsorted_is_order_inconsistent() -> None:
    case = _ranking_case()
    records = [
        MemoryRecord(memory_id="m1", content="a", metadata={"importance": "0.1"}),
        MemoryRecord(memory_id="m2", content="b", metadata={"importance": "0.9"}),
        MemoryRecord(memory_id="m3", content="c", metadata={"importance": "0.5"}),
    ]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, values, matches = classify_ranking_case(case, query_result, ["m1", "m2", "m3"])
    assert signal == RankingSignal.ORDER_INCONSISTENT


def test_classify_ranking_case_fewer_than_two_records_is_not_applicable() -> None:
    case = _ranking_case()
    records = [MemoryRecord(memory_id="m1", content="a", metadata={"importance": "0.9"})]
    query_result = QueryResult(
        records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, values, matches = classify_ranking_case(case, query_result, ["m1"])
    assert signal == RankingSignal.NOT_APPLICABLE


def test_classify_ranking_case_zero_records_is_not_applicable() -> None:
    case = _ranking_case()
    query_result = QueryResult(
        records=[], conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, values, matches = classify_ranking_case(case, query_result, [])
    assert signal == RankingSignal.NOT_APPLICABLE
    assert values == []


def test_ranking_quality_registered_in_eval_list() -> None:
    from memtrust.cli import ALL_EVALS

    assert "ranking_quality" in ALL_EVALS


# ---------------------------------------------------------------------------
# Embedding-Drift/Consistency eval -- closes a gap none of the other evals
# in this file can see: a record broken not by the query touching it, but
# by a *different*, earlier store() call that happened to migrate
# embedding models. Modeled on volcengine/OpenViking#1523
# (A0nameless0man). Every test here runs against fake, in-memory adapters
# -- see evals/embedding_drift.py's module docstring and
# docs/methodology.md for why this cannot be adapter-native against any
# real backend in this repo.
# ---------------------------------------------------------------------------


def test_embedding_drift_dataset_loads() -> None:
    cases = load_embedding_drift_dataset()
    assert len(cases) == 3
    case_ids = {c.case_id for c in cases}
    assert case_ids == {"mt-embed-001", "mt-embed-002", "mt-embed-003"}


def test_embedding_drift_dataset_has_a_same_label_case() -> None:
    """mt-embed-003 seeds model_a_label == model_b_label -- a "migration"
    that never actually changes embedding model. Even a buggy adapter must
    not flag drift here, since nothing about the embedding model changed."""
    cases = {c.case_id: c for c in load_embedding_drift_dataset()}
    same_label_case = cases["mt-embed-003"]
    assert same_label_case.model_a_label == same_label_case.model_b_label


def test_embedding_drift_detects_drift_matching_issue_1523() -> None:
    """The exact bug shape: a fake adapter that silently corrupts
    pre-migration vectors in place (no dimension/model validation) when a
    second embedding-model label appears in the same session. Every
    model-A record in mt-embed-001 and mt-embed-002 (distinct labels) must
    be confirmed retrievable before the migration and flagged
    EMBEDDING_DRIFT after it."""
    adapter = EmbeddingDriftCorruptingFakeAdapter()
    result = run_embedding_drift_eval(adapter)

    drifted = [r for r in result.record_results if r.case_id in {"mt-embed-001", "mt-embed-002"}]
    assert drifted
    for record_result in drifted:
        assert record_result.error is None
        assert record_result.retrievable_before_migration is True
        assert record_result.retrievable_after_migration is False
        assert record_result.signal == EmbeddingDriftSignal.EMBEDDING_DRIFT

    assert result.drift_rate is not None
    assert result.drift_rate > 0.0


def test_embedding_drift_no_drift_when_model_label_does_not_change() -> None:
    """mt-embed-003's model_a_label == model_b_label -- even the buggy
    corrupting adapter must not flag drift here, since its corruption logic
    only triggers on a genuine label change."""
    adapter = EmbeddingDriftCorruptingFakeAdapter()
    result = run_embedding_drift_eval(adapter)
    case_records = [r for r in result.record_results if r.case_id == "mt-embed-003"]
    assert case_records
    for record_result in case_records:
        assert record_result.error is None
        assert record_result.signal == EmbeddingDriftSignal.CLEAN


def test_embedding_drift_does_not_false_positive_on_clean_migration() -> None:
    """Negative control this eval must get right: a backend that validates/
    segregates embedding dimensions correctly must never be flagged, across
    every case in the fixture."""
    adapter = EmbeddingDriftCleanFakeAdapter()
    result = run_embedding_drift_eval(adapter)

    assert result.record_results
    for record_result in result.record_results:
        assert record_result.error is None
        assert record_result.retrievable_before_migration is True
        assert record_result.retrievable_after_migration is True
        assert record_result.signal == EmbeddingDriftSignal.CLEAN

    assert result.drift_rate == 0.0
    assert result.clean_rate == 1.0


def test_embedding_drift_handles_backend_failure() -> None:
    adapter = FailingFakeAdapter()
    result = run_embedding_drift_eval(adapter)
    # 3 cases: 3 + 2 + 1 model-A records total.
    assert len(result.record_results) == 6
    assert all(r.error is not None for r in result.record_results)
    assert all(r.signal == EmbeddingDriftSignal.NOT_APPLICABLE for r in result.record_results)


def test_embedding_drift_eval_result_rates_ignore_errored_cases() -> None:
    adapter = EmbeddingDriftCorruptingFakeAdapter()
    result = run_embedding_drift_eval(adapter)
    assert result.drift_rate is not None
    assert 0.0 <= result.drift_rate <= 1.0
    assert len(result.scored_records) == len(result.record_results)


def test_classify_embedding_drift_record_never_retrievable_is_not_applicable() -> None:
    """A record that was never observed retrievable before any migration
    step has no valid baseline -- must never be misattributed to drift."""
    assert classify_embedding_drift_record(False, False) == EmbeddingDriftSignal.NOT_APPLICABLE
    assert classify_embedding_drift_record(False, True) == EmbeddingDriftSignal.NOT_APPLICABLE


def test_classify_embedding_drift_record_lost_after_migration_is_drift() -> None:
    assert classify_embedding_drift_record(True, False) == EmbeddingDriftSignal.EMBEDDING_DRIFT


def test_classify_embedding_drift_record_retrievable_both_times_is_clean() -> None:
    assert classify_embedding_drift_record(True, True) == EmbeddingDriftSignal.CLEAN


def test_embedding_drift_registered_in_eval_list() -> None:
    from memtrust.cli import ALL_EVALS

    assert "embedding_drift" in ALL_EVALS


# ---------------------------------------------------------------------------
# Crash-Recovery eval -- volcengine/OpenViking#2644 (yeyitech): a local
# vectordb's `_recover()` silently skips rebuilding the search index on
# server-process restart when index files are missing but store data
# exists, so queries silently return nothing after a crash/restart even
# though the data was never actually lost. memtrust's adapters have zero
# real process-lifecycle control over any live backend (see
# evals/crash_recovery.py's module docstring), so this eval targets the
# STRUCTURAL failure shape against a fake adapter that models it -- it
# proves the harness's classification logic works, not that any live
# OpenViking instance currently has this bug.
# ---------------------------------------------------------------------------


def test_crash_recovery_dataset_loads() -> None:
    cases = load_crash_recovery_dataset()
    assert len(cases) == 3
    assert all(isinstance(c, CrashRecoveryCase) for c in cases)


def test_crash_recovery_skips_cleanly_for_unsupported_adapter() -> None:
    adapter = RecallAllFakeAdapter()
    result = run_crash_recovery_eval(adapter)
    assert result.skipped is True
    assert result.skip_reason is not None
    assert result.case_results == []


def test_crash_recovery_detects_index_lost_data_survived_matching_issue_2644() -> None:
    """The exact bug shape: after the simulated crash/restart, query()
    returns nothing for every case even though raw_store_contains()
    independently confirms the data survived. Must fire on every case."""
    adapter = CrashRecoveryFakeAdapter()
    result = run_crash_recovery_eval(adapter)
    assert len(result.case_results) == 3
    assert all(c.error is None for c in result.case_results)
    assert all(c.present_before_crash for c in result.case_results)
    assert all(not c.queryable_after_crash for c in result.case_results)
    assert all(c.raw_store_contains_after_crash is True for c in result.case_results)
    assert all(
        c.signal == CrashRecoverySignal.INDEX_LOST_DATA_SURVIVED for c in result.case_results
    )
    assert result.index_lost_data_survived_rate == 1.0
    assert result.recovered_rate == 0.0
    assert result.data_lost_rate == 0.0


def test_crash_recovery_does_not_false_positive_on_clean_recovery() -> None:
    """Negative control this eval must get right: a backend that
    correctly rebuilds its index from surviving store data on restart
    must NOT be flagged as INDEX_LOST_DATA_SURVIVED."""
    adapter = CrashRecoveryCleanFakeAdapter()
    result = run_crash_recovery_eval(adapter)
    assert len(result.case_results) == 3
    assert all(c.signal == CrashRecoverySignal.RECOVERED for c in result.case_results)
    assert result.recovered_rate == 1.0
    assert result.index_lost_data_survived_rate == 0.0


def test_crash_recovery_flags_data_lost_distinct_from_index_lost() -> None:
    """A backend that loses the underlying data itself, not just the
    index, is a different and more severe failure -- must be classified
    DATA_LOST, never conflated with INDEX_LOST_DATA_SURVIVED."""
    adapter = CrashRecoveryDataLostFakeAdapter()
    result = run_crash_recovery_eval(adapter)
    assert all(c.raw_store_contains_after_crash is False for c in result.case_results)
    assert all(c.signal == CrashRecoverySignal.DATA_LOST for c in result.case_results)
    assert result.data_lost_rate == 1.0
    assert result.index_lost_data_survived_rate == 0.0


def test_crash_recovery_handles_backend_failure() -> None:
    adapter = CrashRecoveryFailingFakeAdapter()
    result = run_crash_recovery_eval(adapter)
    assert len(result.case_results) == 3
    assert all(c.error is not None for c in result.case_results)
    assert all(c.signal == CrashRecoverySignal.NOT_APPLICABLE for c in result.case_results)


def test_crash_recovery_eval_result_rates_ignore_errored_cases() -> None:
    adapter = CrashRecoveryFakeAdapter()
    result = run_crash_recovery_eval(adapter)
    assert result.index_lost_data_survived_rate is not None
    assert 0.0 <= result.index_lost_data_survived_rate <= 1.0
    assert len(result.scored_cases) == len(result.case_results)


def test_classify_crash_recovery_case_recovered() -> None:
    assert classify_crash_recovery_case(True, True, None) == CrashRecoverySignal.RECOVERED


def test_classify_crash_recovery_case_index_lost_data_survived() -> None:
    signal = classify_crash_recovery_case(True, False, True)
    assert signal == CrashRecoverySignal.INDEX_LOST_DATA_SURVIVED


def test_classify_crash_recovery_case_data_lost() -> None:
    assert classify_crash_recovery_case(True, False, False) == CrashRecoverySignal.DATA_LOST


def test_classify_crash_recovery_case_never_present_before_is_not_applicable() -> None:
    assert classify_crash_recovery_case(False, False, None) == CrashRecoverySignal.NOT_APPLICABLE
    assert classify_crash_recovery_case(False, False, True) == CrashRecoverySignal.NOT_APPLICABLE


def test_classify_crash_recovery_case_no_raw_store_evidence_is_not_applicable() -> None:
    """present_before_crash True, queryable_after_crash False, but the
    eval never got an independent raw_store_contains() read -- not
    enough evidence to call this either INDEX_LOST_DATA_SURVIVED or
    DATA_LOST."""
    assert classify_crash_recovery_case(True, False, None) == CrashRecoverySignal.NOT_APPLICABLE


def test_crash_recovery_registered_in_eval_list() -> None:
    from memtrust.cli import ALL_EVALS

    assert "crash_recovery" in ALL_EVALS


def test_migration_rollback_dataset_loads() -> None:
    cases = load_migration_rollback_dataset()
    assert len(cases) == 3
    assert all(isinstance(c, MigrationRollbackCase) for c in cases)


def test_migration_rollback_skips_cleanly_for_unsupported_adapter() -> None:
    adapter = RecallAllFakeAdapter()
    result = run_migration_rollback_eval(adapter)
    assert result.skipped is True
    assert result.skip_reason is not None
    assert result.case_results == []


def test_migration_rollback_detects_data_lost_matching_issue_1028() -> None:
    """The exact MemPalace/mempalace#1028 bug shape: the unguarded
    rmtree()-then-move() swap deletes the old backup before the move
    completes, so an independent post-failure query() finds nothing.
    Must fire on every case."""
    adapter = MigrationRollbackFakeAdapter()
    result = run_migration_rollback_eval(adapter)
    assert len(result.case_results) == 3
    assert all(c.error is None for c in result.case_results)
    assert all(c.original_data_recoverable is False for c in result.case_results)
    assert all(c.signal == MigrationRollbackSignal.DATA_LOST for c in result.case_results)
    assert result.data_lost_rate == 1.0
    assert result.restored_rate == 0.0


def test_migration_rollback_does_not_false_positive_on_rename_aside_fix() -> None:
    """Negative control this eval must get right: a backend that
    implements MemPalace/mempalace#935's rename-aside swap must NOT be
    flagged as DATA_LOST -- the original data must be classified
    RESTORED."""
    adapter = MigrationRollbackRenameAsideFakeAdapter()
    result = run_migration_rollback_eval(adapter)
    assert len(result.case_results) == 3
    assert all(c.original_data_recoverable is True for c in result.case_results)
    assert all(c.signal == MigrationRollbackSignal.RESTORED for c in result.case_results)
    assert result.restored_rate == 1.0
    assert result.data_lost_rate == 0.0


def test_migration_rollback_classification_uses_independent_query_not_adapter_self_report() -> None:
    """The classification's ground truth must be this eval's own
    independent post-failure query() observation, not the adapter's
    self-reported MigrationFailureResult.original_data_recoverable flag
    threaded through only for diagnostics -- see this module's docstring.
    For both fake adapters here, the two should agree; this asserts the
    field the eval actually reports is set and consistent."""
    lost_adapter = MigrationRollbackFakeAdapter()
    lost_result = run_migration_rollback_eval(lost_adapter)
    assert all(
        c.adapter_reported_recoverable == c.original_data_recoverable
        for c in lost_result.case_results
    )

    restored_adapter = MigrationRollbackRenameAsideFakeAdapter()
    restored_result = run_migration_rollback_eval(restored_adapter)
    assert all(
        c.adapter_reported_recoverable == c.original_data_recoverable
        for c in restored_result.case_results
    )


def test_migration_rollback_handles_backend_failure() -> None:
    adapter = MigrationRollbackFailingFakeAdapter()
    result = run_migration_rollback_eval(adapter)
    assert len(result.case_results) == 3
    assert all(c.error is not None for c in result.case_results)
    assert all(c.signal == MigrationRollbackSignal.NOT_APPLICABLE for c in result.case_results)
    assert all(c.original_data_recoverable is None for c in result.case_results)
    assert all(c.adapter_reported_recoverable is None for c in result.case_results)


def test_migration_rollback_eval_result_rates_ignore_errored_cases() -> None:
    adapter = MigrationRollbackFakeAdapter()
    result = run_migration_rollback_eval(adapter)
    assert result.data_lost_rate is not None
    assert 0.0 <= result.data_lost_rate <= 1.0
    assert len(result.scored_cases) == len(result.case_results)


def test_classify_migration_rollback_case_restored() -> None:
    assert classify_migration_rollback_case(True) == MigrationRollbackSignal.RESTORED


def test_classify_migration_rollback_case_data_lost() -> None:
    assert classify_migration_rollback_case(False) == MigrationRollbackSignal.DATA_LOST


def test_migration_rollback_registered_in_eval_list() -> None:
    from memtrust.cli import ALL_EVALS

    assert "migration_rollback" in ALL_EVALS


# ---------------------------------------------------------------------------
# Extraction-quality-at-scale eval -- mem0ai/mem0#4573 (jamebobob)
# ---------------------------------------------------------------------------


def test_extraction_quality_dataset_loads() -> None:
    cases, feedback_cases = load_extraction_quality_dataset()
    assert len(cases) == 15
    assert all(isinstance(c, ExtractionQualityCase) for c in cases)
    assert len(feedback_cases) == 2
    junk = [c for c in cases if not c.should_be_stored]
    valid = [c for c in cases if c.should_be_stored]
    assert len(junk) == 12
    assert len(valid) == 3
    # every one of jamebobob's real junk categories is represented
    categories = {c.category for c in junk}
    assert {
        "boot_file_restating",
        "cron_heartbeat_noise",
        "system_dump",
        "hallucinated_profile",
    } <= categories


def test_extraction_quality_no_gate_backend_retains_junk_at_high_rate() -> None:
    """The core regression this eval exists to catch: a backend with no
    extraction-quality gate at all (mem0's real reported behavior per
    mem0ai/mem0#4573 -- 97.8% of 10,134 stored entries were junk) must
    show a high junk_retained_rate. This fake retains everything
    indiscriminately, so every junk case should come back RETAINED_JUNK."""
    adapter = NoExtractionGateFakeAdapter()
    result = run_extraction_quality_eval(adapter)
    assert result.junk_retained_rate == 1.0
    assert result.junk_rejected_rate == 0.0
    assert result.valid_retained_rate == 1.0
    assert result.valid_lost_rate == 0.0
    junk_results = [c for c in result.case_results if not c.case.should_be_stored]
    assert junk_results  # sanity: fixture actually has junk cases
    assert all(c.signal == ExtractionQualitySignal.RETAINED_JUNK for c in junk_results)


def test_extraction_quality_gated_backend_rejects_junk_at_low_rate() -> None:
    """The positive case: a backend with a real extraction-quality gate
    (here, one keyed off the case's `category` metadata) should show a
    low junk_retained_rate and a high junk_rejected_rate, while still
    retaining every genuinely valuable item."""
    adapter = GatedExtractionFakeAdapter()
    result = run_extraction_quality_eval(adapter)
    assert result.junk_retained_rate == 0.0
    assert result.junk_rejected_rate == 1.0
    assert result.valid_retained_rate == 1.0
    assert result.valid_lost_rate == 0.0
    junk_results = [c for c in result.case_results if not c.case.should_be_stored]
    assert all(c.signal == ExtractionQualitySignal.REJECTED_JUNK for c in junk_results)
    valid_results = [c for c in result.case_results if c.case.should_be_stored]
    assert all(c.signal == ExtractionQualitySignal.RETAINED_VALID for c in valid_results)


def test_extraction_quality_over_aggressive_filter_flags_lost_valid() -> None:
    """The necessary counterweight: a backend that filters everything
    (including genuinely valuable content) must not be rewarded for a
    perfect junk_rejected_rate -- it should show a high valid_lost_rate
    too. A backend that discards every real user's content would score
    perfectly on junk-rejection alone if this axis didn't exist."""

    class RejectEverythingFakeAdapter(NoExtractionGateFakeAdapter):
        name = "fake-reject-everything"

        def store(
            self, session_id: str, content: str, metadata: dict[str, str] | None = None
        ) -> StoreResult:
            self._counter += 1
            return StoreResult(memory_id=f"m{self._counter}", latency_ms=0.1)

    adapter = RejectEverythingFakeAdapter()
    result = run_extraction_quality_eval(adapter)
    assert result.junk_rejected_rate == 1.0
    assert result.valid_lost_rate == 1.0
    assert result.valid_retained_rate == 0.0


def test_extraction_quality_feedback_loop_duplication_fires() -> None:
    """Reproduces jamebobob's exact 808-duplicate mechanism at small
    scale: store a seed item, query it back (simulating recall into an
    agent's context), re-store that exact recalled text as new input, and
    check whether the record count grew by more than the single re-store
    call should have added. Must fire FEEDBACK_LOOP_DUPLICATE against a
    backend whose write path fans a repeat store() out into duplicates."""
    adapter = FeedbackLoopDuplicatingFakeAdapter()
    result = run_extraction_quality_eval(adapter)
    assert result.feedback_loop_results
    assert result.feedback_loop_duplicate_rate == 1.0
    for fb_result in result.feedback_loop_results:
        assert fb_result.error is None
        assert fb_result.signal == ExtractionQualitySignal.FEEDBACK_LOOP_DUPLICATE
        assert fb_result.records_after_first_store == 1
        assert fb_result.records_after_second_store > fb_result.records_after_first_store + 1


def test_extraction_quality_feedback_loop_no_duplication_on_clean_backend() -> None:
    """Negative control: a backend that just stores each call once (no
    duplication bug) must NOT be flagged -- one seed store plus one
    re-store legitimately grows the matching count by exactly one."""
    adapter = NoExtractionGateFakeAdapter()
    result = run_extraction_quality_eval(adapter)
    assert result.feedback_loop_results
    for fb_result in result.feedback_loop_results:
        assert fb_result.error is None
        assert fb_result.signal == ExtractionQualitySignal.NO_UNEXPECTED_GROWTH
        assert fb_result.records_after_second_store == fb_result.records_after_first_store + 1


def test_extraction_quality_handles_backend_failure() -> None:
    adapter = FailingFakeAdapter()
    result = run_extraction_quality_eval(adapter)
    assert len(result.case_results) == 15
    assert all(c.error is not None for c in result.case_results)
    assert all(c.signal == ExtractionQualitySignal.NOT_APPLICABLE for c in result.case_results)
    assert len(result.feedback_loop_results) == 2
    assert all(c.error is not None for c in result.feedback_loop_results)
    assert all(
        c.signal == ExtractionQualitySignal.NOT_APPLICABLE for c in result.feedback_loop_results
    )
    assert result.junk_retained_rate is None
    assert result.feedback_loop_duplicate_rate is None


def test_classify_extraction_case_retained_junk() -> None:
    case = ExtractionQualityCase(
        case_id="t1",
        session_id="s",
        query="q",
        content="Heartbeat check: all systems nominal.",
        category="cron_heartbeat_noise",
        should_be_stored=False,
    )
    query_result = QueryResult(
        records=[MemoryRecord(memory_id="m1", content="Heartbeat check: all systems nominal.")],
        conflict_signal=ConflictSignal.NOT_APPLICABLE,
        latency_ms=0.1,
    )
    signal, retrieved = classify_extraction_case(case, query_result)
    assert signal == ExtractionQualitySignal.RETAINED_JUNK
    assert retrieved is True


def test_classify_extraction_case_rejected_junk() -> None:
    case = ExtractionQualityCase(
        case_id="t2",
        session_id="s",
        query="q",
        content="Heartbeat check: all systems nominal.",
        category="cron_heartbeat_noise",
        should_be_stored=False,
    )
    query_result = QueryResult(
        records=[], conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, retrieved = classify_extraction_case(case, query_result)
    assert signal == ExtractionQualitySignal.REJECTED_JUNK
    assert retrieved is False


def test_classify_extraction_case_lost_valid() -> None:
    case = ExtractionQualityCase(
        case_id="t3",
        session_id="s",
        query="q",
        content="User prefers async standups.",
        category="valid_content",
        should_be_stored=True,
    )
    query_result = QueryResult(
        records=[], conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
    )
    signal, retrieved = classify_extraction_case(case, query_result)
    assert signal == ExtractionQualitySignal.LOST_VALID
    assert retrieved is False


def test_classify_feedback_loop_case_no_growth_is_clean() -> None:
    assert classify_feedback_loop_case(1, 1) == ExtractionQualitySignal.NO_UNEXPECTED_GROWTH


def test_classify_feedback_loop_case_expected_single_growth_is_clean() -> None:
    assert classify_feedback_loop_case(1, 2) == ExtractionQualitySignal.NO_UNEXPECTED_GROWTH


def test_classify_feedback_loop_case_excess_growth_is_flagged() -> None:
    assert classify_feedback_loop_case(1, 6) == ExtractionQualitySignal.FEEDBACK_LOOP_DUPLICATE


def test_extraction_quality_registered_in_eval_list() -> None:
    from memtrust.cli import ALL_EVALS

    assert "extraction_quality" in ALL_EVALS
