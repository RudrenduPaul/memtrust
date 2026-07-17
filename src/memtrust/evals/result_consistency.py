"""MemTrust's repeated-query determinism eval.

Every other eval in this package classifies a SINGLE query() response, or
compares two responses separated by some other event (a crash, a resync,
a migration). None of them can see a backend that is internally
non-deterministic: each individual response can look perfectly well-formed
-- real records, real content, no error -- while the *set* of records
returned for the exact same query, against data that never changed,
varies from call to call.

Motivating case: volcengine/OpenViking#204 (contributor ponsde, a repeat
contributor to this project). `search()`/`find()` returned non-
deterministic result sets for identical repeated queries: 5 runs of the
same query returned an average pairwise Jaccard similarity of 0.11 over
the returned memory/resource URIs, and a later, more rigorous 3-query x
5-run test found some query/method combinations sharing ZERO common URIs
across all 5 runs. ponsde self-diagnosed two candidate root causes through
extensive hands-on follow-up: an embedding-dimension mismatch (a
production vector collection created at `Dimension:3072` while the active
embedding config specified `1024`), and/or non-deterministic graph
traversal inside the HNSW ANN search implementation itself. This eval
reuses ponsde's own methodology directly: issue the identical query N
times against unchanged fixture data, and compute the average pairwise
(consecutive-run) Jaccard similarity over the returned record ids.

See ConsistencySignal in adapters/base.py for the taxonomy this eval
scores against.

Honest limitation (same convention as every other eval's module docstring
in this package): this eval can only measure what it can observe through
`query()` -- it has no access to OpenViking's internal ANN index state or
embedding-config metadata, so it cannot itself distinguish ponsde's two
candidate root causes from each other, or from an unrelated source of
non-determinism (e.g. a backend that intentionally randomizes tie-breaks).
It proves the harness can *detect* the same result-instability shape
ponsde documented, using his own methodology, against any adapter pointed
at it -- not that any specific root cause is present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memtrust.adapters.base import BackendAPIError, ConsistencySignal, MemoryBackendAdapter

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "result_consistency_cases.json"
)

#: Default number of times each case's query is repeated. ponsde's own
#: repro used 5 runs per query; kept as this eval's default for direct
#: comparability with his reported numbers.
DEFAULT_N_REPEATS = 5

#: Default minimum average pairwise Jaccard similarity for a case to be
#: classified CONSISTENT. Not 1.0 (perfect overlap every run): a
#: genuinely ranked-and-truncated top_k result can reasonably reorder
#: near a score tie without that being a real bug -- this tolerance is
#: deliberately generous compared to ponsde's own measured 0.11 average
#: for the broken case, so a backend has to be substantially unstable to
#: fail this threshold.
DEFAULT_CONSISTENCY_THRESHOLD = 0.9


@dataclass
class ConsistencySeedRecord:
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ConsistencyCase:
    case_id: str
    session_id: str
    query: str
    records: list[ConsistencySeedRecord]
    n_repeats: int = DEFAULT_N_REPEATS
    threshold: float = DEFAULT_CONSISTENCY_THRESHOLD


@dataclass
class ConsistencyCaseResult:
    case: ConsistencyCase
    signal: ConsistencySignal
    average_jaccard: float | None
    """Average pairwise (consecutive-run) Jaccard similarity over the
    result-id sets from each successful query() call. None when fewer
    than 2 calls succeeded."""
    n_successful_runs: int
    result_id_sets: list[list[str]]
    """Each successful run's returned record-id set, in call order, kept
    for audit purposes -- e.g. to see exactly which ids appeared/vanished
    between runs."""
    error: str | None = None


@dataclass
class ConsistencyEvalResult:
    backend_name: str
    dataset_path: str
    case_results: list[ConsistencyCaseResult] = field(default_factory=list)

    @property
    def scored_cases(self) -> list[ConsistencyCaseResult]:
        return [c for c in self.case_results if c.error is None]

    def _fraction(self, signal: ConsistencySignal) -> float | None:
        scored = self.scored_cases
        if not scored:
            return None
        matching = sum(1 for c in scored if c.signal == signal)
        return matching / len(scored)

    @property
    def consistent_rate(self) -> float | None:
        return self._fraction(ConsistencySignal.CONSISTENT)

    @property
    def inconsistent_rate(self) -> float | None:
        """Fraction of cases whose repeated, identical query against
        unchanged data produced meaningfully different result sets --
        the headline metric this eval exists to surface, and the exact
        volcengine/OpenViking#204 shape."""
        return self._fraction(ConsistencySignal.INCONSISTENT)


def load_dataset(path: Path | str = DEFAULT_FIXTURE_PATH) -> list[ConsistencyCase]:
    data = json.loads(Path(path).read_text())
    cases: list[dict[str, Any]] = data["cases"]
    return [
        ConsistencyCase(
            case_id=c["case_id"],
            session_id=c["session_id"],
            query=c["query"],
            n_repeats=c.get("n_repeats", DEFAULT_N_REPEATS),
            threshold=c.get("threshold", DEFAULT_CONSISTENCY_THRESHOLD),
            records=[
                ConsistencySeedRecord(content=r["content"], metadata=r.get("metadata", {}))
                for r in c["records"]
            ],
        )
        for c in cases
    ]


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two id sets. Two empty sets are treated
    as perfectly similar (1.0) -- "returned nothing, twice" is a
    consistency-neutral result for THIS metric (a separate eval,
    evals/contradiction.py's EMPTY_OR_LOST handling, already covers
    "returned nothing when it should not have")."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def classify_consistency_case(
    result_id_sets: list[set[str]],
    threshold: float = DEFAULT_CONSISTENCY_THRESHOLD,
) -> tuple[ConsistencySignal, float | None]:
    """Classify a case's outcome from its list of per-run result-id sets,
    using ponsde's own consecutive-pair Jaccard-similarity methodology
    (volcengine/OpenViking#204's own hand-built repro script compares run
    i against run i+1, not every pair against every other pair).

    Returns (signal, average_jaccard). average_jaccard is None when fewer
    than 2 runs are available to form a pair -- NOT_APPLICABLE in that
    case, never guessed at either way.
    """
    if len(result_id_sets) < 2:
        return ConsistencySignal.NOT_APPLICABLE, None
    pairwise = [
        _jaccard(result_id_sets[i], result_id_sets[i + 1]) for i in range(len(result_id_sets) - 1)
    ]
    average = sum(pairwise) / len(pairwise)
    if average >= threshold:
        return ConsistencySignal.CONSISTENT, average
    return ConsistencySignal.INCONSISTENT, average


def _result_ids(records: list[Any]) -> set[str]:
    """Prefer memory_id when the adapter reports a real one; fall back to
    the record's own content as an id-surrogate for adapters/fixtures
    where memory_id can be empty on some responses -- same
    "id-first, content-fallback" convention verify_store() already uses
    in adapters/base.py."""
    ids: set[str] = set()
    for record in records:
        ids.add(record.memory_id if record.memory_id else f"content:{record.content}")
    return ids


def run_result_consistency_eval(
    adapter: MemoryBackendAdapter,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
) -> ConsistencyEvalResult:
    cases = load_dataset(dataset_path)
    result = ConsistencyEvalResult(backend_name=adapter.name, dataset_path=str(dataset_path))

    for case in cases:
        try:
            for seed in case.records:
                adapter.store(case.session_id, seed.content, metadata=seed.metadata)

            result_id_sets: list[set[str]] = []
            for _ in range(case.n_repeats):
                query_result = adapter.query(case.session_id, case.query, top_k=len(case.records))
                result_id_sets.append(_result_ids(query_result.records))
        except BackendAPIError as exc:
            result.case_results.append(
                ConsistencyCaseResult(
                    case=case,
                    signal=ConsistencySignal.NOT_APPLICABLE,
                    average_jaccard=None,
                    n_successful_runs=0,
                    result_id_sets=[],
                    error=str(exc),
                )
            )
            continue

        signal, average = classify_consistency_case(result_id_sets, case.threshold)
        result.case_results.append(
            ConsistencyCaseResult(
                case=case,
                signal=signal,
                average_jaccard=average,
                n_successful_runs=len(result_id_sets),
                result_id_sets=[sorted(s) for s in result_id_sets],
            )
        )

    return result
