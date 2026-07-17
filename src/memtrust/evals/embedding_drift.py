"""MemTrust's embedding-drift/consistency eval.

Every other eval in this package classifies a *single* store()/query()
round trip (or, for resource-sync-safety, a single seed-file's before/after
state across one resync call). This eval targets a failure mode none of
them can see: a record that was perfectly fine when it was stored can be
silently broken by a *later, unrelated* store() call for a *different*
record, if that later call happens to migrate embedding models. No single
query() response carries any evidence of this -- the record that broke
wasn't touched by the query that reveals the breakage, it was broken
earlier by a completely different write.

Motivating case: volcengine/OpenViking#1523 (contributor A0nameless0man).
An embedder migration silently degrades search quality mid-migration:
switching embedding models overwrites previously-stored vectors in place
with no dimension/model validation, so records embedded under the old
model can stop being retrievable once new-model writes start landing, with
no exception and no signal anywhere that this happened.

**Honest scope -- read this before trusting an EMBEDDING_DRIFT result.**
This eval cannot be built adapter-natively against any real backend in this
repo. `OpenVikingAdapter.query()` (the adapter this bug report names) talks
to OpenViking's documented `/v1/search` response shape, which this build's
research pass did not find to expose any per-record embedding-model or
embedding-dimension field at all (see openviking_adapter.py's module
docstring) -- and no other adapter in this repo exposes one either. A real
backend genuinely cannot tell this eval "this record was embedded by model
A" on its own; `MemoryRecord.embedding_model`/`embedding_dims`
(adapters/base.py) exist so an adapter COULD report this if a future
backend's API surfaced it, but as of this writing none does.

So this eval is built at the harness level instead, the same way
resource_sync_safety.py's NESTED_CONTENT_UNINDEXED signal and
ranking_quality.py's MISSING_ORDERING_KEY signal are: it drives the shared
store()/query() interface with a fixture-level construct -- a plain string
metadata tag (`embedding_model_label`) the fixture assigns to each seeded
record, not a real embedder concept any adapter has to understand -- and
observes whether the adapter's OWN behavior across two store() calls
reproduces the exact bug shape (in-place overwrite with no dimension
validation): records seeded under one label become unrecoverable once
records seeded under a second label are stored into the same session.

Every unit test covering this eval (tests/test_evals.py) runs against fake,
in-memory adapters purpose-built to reproduce that exact bug shape (or to
migrate cleanly, as the negative control) -- never against a live
OpenViking instance or any other live backend, since no live backend
exposes what this eval would need to be adapter-native. See
docs/methodology.md for the same caveat stated in the project's single
source of truth for what is measured vs. simulated.

Design:

  1. Store N records under the case's `model_a_label` embedding-model tag.
  2. Confirm each is retrievable (query for its own content) -- this is the
     eval's baseline, established BEFORE any migration step runs.
  3. Store the case's `model_b_records` under a *different* embedding-model
     tag, `model_b_label`, into the same session -- this is the "migration."
  4. Re-query for each model-A record's own content.
  5. Classify each model-A record from its before/after retrievability (see
     `classify_embedding_drift_record` below) -- never from a bare
     self-report, same non-negotiable convention every other eval in this
     package follows.

Requiring step 2's confirmed baseline is what keeps this eval honest about
"normal recall variance": a record that was never retrievable in the first
place (a generic recall miss with nothing to do with any migration) is
recorded NOT_APPLICABLE, never misattributed to drift. Only a record that
was provably retrievable before the migration step and stops being
retrievable strictly after it is classified EMBEDDING_DRIFT.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memtrust.adapters.base import (
    BackendAPIError,
    EmbeddingDriftSignal,
    MemoryBackendAdapter,
)

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "embedding_drift_cases.json"
)


@dataclass
class EmbeddingDriftSeedRecord:
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class EmbeddingDriftCase:
    case_id: str
    session_id: str
    model_a_label: str
    """Fixture-level tag standing in for "the embedding model in use before
    the migration." Passed through to `adapter.store()`'s generic
    `metadata` dict under the `embedding_model_label` key -- a plain string
    every adapter already knows how to accept and store, not a real
    embedder concept any adapter has to implement."""
    model_b_label: str
    """Fixture-level tag for "the embedding model switched to mid-migration."
    """
    model_a_records: list[EmbeddingDriftSeedRecord]
    """Records stored under `model_a_label` before the simulated migration
    -- these are the records this eval checks for drift."""
    model_b_records: list[EmbeddingDriftSeedRecord]
    """Records stored under `model_b_label` after the model-A records --
    this is what simulates "switching to model B for new stores." Their own
    retrievability is not scored; they exist purely to trigger whatever the
    adapter's own store() does when a second embedding-model label appears
    in the same session."""


@dataclass
class EmbeddingDriftRecordResult:
    case_id: str
    content: str
    model_a_label: str
    model_b_label: str
    retrievable_before_migration: bool
    retrievable_after_migration: bool
    signal: EmbeddingDriftSignal
    error: str | None = None


@dataclass
class EmbeddingDriftEvalResult:
    backend_name: str
    dataset_path: str
    record_results: list[EmbeddingDriftRecordResult] = field(default_factory=list)

    @property
    def scored_records(self) -> list[EmbeddingDriftRecordResult]:
        return [r for r in self.record_results if r.error is None]

    def _fraction(self, signal: EmbeddingDriftSignal) -> float | None:
        scored = self.scored_records
        if not scored:
            return None
        matching = sum(1 for r in scored if r.signal == signal)
        return matching / len(scored)

    @property
    def drift_rate(self) -> float | None:
        """Fraction of model-A records that were confirmed retrievable
        before the migration step and became unretrievable afterward --
        the headline metric this eval exists to surface, and the one that
        would flag volcengine/OpenViking#1523's exact shape."""
        return self._fraction(EmbeddingDriftSignal.EMBEDDING_DRIFT)

    @property
    def clean_rate(self) -> float | None:
        return self._fraction(EmbeddingDriftSignal.CLEAN)

    @property
    def not_applicable_rate(self) -> float | None:
        return self._fraction(EmbeddingDriftSignal.NOT_APPLICABLE)


def load_dataset(path: Path | str = DEFAULT_FIXTURE_PATH) -> list[EmbeddingDriftCase]:
    data = json.loads(Path(path).read_text())
    cases: list[dict[str, Any]] = data["cases"]
    return [
        EmbeddingDriftCase(
            case_id=c["case_id"],
            session_id=c["session_id"],
            model_a_label=c["model_a_label"],
            model_b_label=c["model_b_label"],
            model_a_records=[
                EmbeddingDriftSeedRecord(content=r["content"], metadata=r.get("metadata", {}))
                for r in c["model_a_records"]
            ],
            model_b_records=[
                EmbeddingDriftSeedRecord(content=r["content"], metadata=r.get("metadata", {}))
                for r in c["model_b_records"]
            ],
        )
        for c in cases
    ]


def classify_embedding_drift_record(
    retrievable_before_migration: bool,
    retrievable_after_migration: bool,
) -> EmbeddingDriftSignal:
    """Classify a single model-A record's outcome from its before/after
    retrievability.

    Only ever returns EMBEDDING_DRIFT when the record was provably
    retrievable before the migration step -- a record that was never
    observed retrievable in the first place has no valid baseline and is
    classified NOT_APPLICABLE instead, so an ordinary recall miss unrelated
    to any migration can never be misattributed to drift.
    """
    if not retrievable_before_migration:
        return EmbeddingDriftSignal.NOT_APPLICABLE
    if not retrievable_after_migration:
        return EmbeddingDriftSignal.EMBEDDING_DRIFT
    return EmbeddingDriftSignal.CLEAN


def _is_retrievable(adapter: MemoryBackendAdapter, session_id: str, content: str) -> bool:
    query_result = adapter.query(session_id, content, top_k=5)
    return any(content.lower() in r.content.lower() for r in query_result.records)


def run_embedding_drift_eval(
    adapter: MemoryBackendAdapter,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
) -> EmbeddingDriftEvalResult:
    cases = load_dataset(dataset_path)
    result = EmbeddingDriftEvalResult(backend_name=adapter.name, dataset_path=str(dataset_path))

    for case in cases:
        try:
            # (1) Store every model-A record under the case's model_a_label.
            for seed in case.model_a_records:
                metadata = {**seed.metadata, "embedding_model_label": case.model_a_label}
                adapter.store(case.session_id, seed.content, metadata=metadata)

            # (2) Confirm baseline retrievability BEFORE any migration --
            # this is what lets EMBEDDING_DRIFT be distinguished from
            # ordinary recall variance below.
            retrievable_before = {
                seed.content: _is_retrievable(adapter, case.session_id, seed.content)
                for seed in case.model_a_records
            }

            # (3) Simulate switching to model B for new stores.
            for seed in case.model_b_records:
                metadata = {**seed.metadata, "embedding_model_label": case.model_b_label}
                adapter.store(case.session_id, seed.content, metadata=metadata)

            # (4) Re-check every model-A record after the migration step.
            for seed in case.model_a_records:
                retrievable_after = _is_retrievable(adapter, case.session_id, seed.content)
                signal = classify_embedding_drift_record(
                    retrievable_before[seed.content], retrievable_after
                )
                result.record_results.append(
                    EmbeddingDriftRecordResult(
                        case_id=case.case_id,
                        content=seed.content,
                        model_a_label=case.model_a_label,
                        model_b_label=case.model_b_label,
                        retrievable_before_migration=retrievable_before[seed.content],
                        retrievable_after_migration=retrievable_after,
                        signal=signal,
                    )
                )
        except BackendAPIError as exc:
            for seed in case.model_a_records:
                result.record_results.append(
                    EmbeddingDriftRecordResult(
                        case_id=case.case_id,
                        content=seed.content,
                        model_a_label=case.model_a_label,
                        model_b_label=case.model_b_label,
                        retrievable_before_migration=False,
                        retrievable_after_migration=False,
                        signal=EmbeddingDriftSignal.NOT_APPLICABLE,
                        error=str(exc),
                    )
                )
            continue

    return result
