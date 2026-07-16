"""MemTrust's ranking/relevance-quality eval.

`evals/contradiction.py`'s `ConflictSignal` taxonomy classifies whether
returned *content* is correct after a contradiction. This eval targets a
structurally different failure mode that `ConflictSignal` cannot see at
all: the returned content can be fully correct, and the backend can still
be silently broken at ORDERING it.

Motivating case: mempalace/mempalace#1733 (GitHub user Kartalops, found
while validating memtrust against real MemPalace usage). `mempalace/
layers.py`'s `Layer1.generate()` sorts drawers by `importance`/
`emotional_weight`/`weight`, but no ingest path in the real package ever
writes those keys -- confirmed 0/45,969 drawers on a real palace. So
`importance` silently defaults to a constant, the "ranked by importance"
sort degenerates to plain insertion order, and a `wake-up`/recall call
documented to return "high importance, recent" moments instead silently
returns the oldest moments first. Every individual returned drawer is a
real, correctly-stored memory -- there is no contradiction anywhere in
this bug, which is exactly why `ConflictSignal` structurally cannot flag
it: that taxonomy only classifies cases where a fact was stored, then
recontradicted, then queried. This bug has no contradiction step at all.

Design principle (same as evals/contradiction.py's classify_case and
evals/resource_sync_safety.py's classify_resource_sync_file): classifying
a case never blindly trusts what the adapter reports about its own
ranking behavior (`QueryResult.ranking_signal`, see adapters/base.py's
RankingSignal and adapters/mempalace_adapter.py's `_classify_ranking_signal`
for how an adapter derives that self-report). This eval owns the ground
truth for what order each case's records were stored in and what values
they carried for the case's declared `ranking_field`, and computes the
final signal from that directly:

  * If the ranking field is missing from every returned record, or present
    but carrying the identical value on every returned record, no real
    per-record signal exists to have driven the order -- classified
    MISSING_ORDERING_KEY regardless of what the adapter self-reports.
  * If the field carries genuinely varied values, this eval checks whether
    the actual returned order is sorted by descending value. If it is,
    the field is a genuine ranking signal -- SIGNAL_DRIVEN. If it is not,
    a real signal exists and the backend still is not ordering by it --
    ORDER_INCONSISTENT, a distinct and equally worth-flagging bug from
    MISSING_ORDERING_KEY.
  * Fewer than 2 returned records, or zero records at all, gives this eval
    nothing to compare -- NOT_APPLICABLE.

Honest limitation (see docs/methodology.md for the full write-up): this
eval can only ever prove "no real per-record signal was observed driving
this response's order." It cannot always distinguish "the backend
genuinely has no meaningful variation to rank by for this query" from
"the backend forgot to populate the ranking field" -- both produce the
identical observable symptom (every returned record shares one value).
MISSING_ORDERING_KEY is the correct, honest name for what is actually
detected: the *absence of a driving signal*, not a proven claim about the
backend's internal cause. Kartalops's mempalace#1733 finding (0/45,969
drawers with a real value, confirmed by direct inspection of a live
palace) is the strong form of this; this eval's black-box query-response
view alone would only ever justify the weaker, still-honest claim above.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memtrust.adapters.base import (
    BackendAPIError,
    MemoryBackendAdapter,
    QueryResult,
    RankingSignal,
)

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "ranking_quality_cases.json"
)


@dataclass
class RankingQualitySeedRecord:
    """One record to store before querying, in the order it appears in
    the case's `records` list -- that storage order is this eval's ground
    truth for "insertion order," which `classify_ranking_case` compares
    the actual returned order against."""

    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RankingQualityCase:
    case_id: str
    session_id: str
    query: str
    ranking_field: str
    """Which metadata key this case is testing as the ranking-relevant
    field (e.g. "importance") -- see RankingSignal in adapters/base.py."""
    records: list[RankingQualitySeedRecord]


@dataclass
class RankingQualityCaseResult:
    case: RankingQualityCase
    signal: RankingSignal
    adapter_reported_signal: RankingSignal | None
    field_values: list[float | None]
    """The case's `ranking_field` value read off each returned record, in
    returned order, parsed as float where possible (None where missing or
    unparseable) -- the evidence `signal` above was computed from."""
    matches_insertion_order: bool | None
    """Whether the returned record order exactly matches the order the
    case's records were stored in. None when this could not be determined
    (e.g. returned memory_ids don't correspond to any stored id)."""
    retrieved_content: str
    error: str | None = None


@dataclass
class RankingQualityEvalResult:
    backend_name: str
    dataset_path: str
    case_results: list[RankingQualityCaseResult] = field(default_factory=list)

    @property
    def scored_cases(self) -> list[RankingQualityCaseResult]:
        return [c for c in self.case_results if c.error is None]

    def _fraction(self, signal: RankingSignal) -> float | None:
        scored = self.scored_cases
        if not scored:
            return None
        matching = sum(1 for c in scored if c.signal == signal)
        return matching / len(scored)

    @property
    def signal_driven_rate(self) -> float | None:
        return self._fraction(RankingSignal.SIGNAL_DRIVEN)

    @property
    def missing_ordering_key_rate(self) -> float | None:
        """Fraction of cases where no real per-record ranking signal was
        observed -- the headline metric this eval exists to surface, and
        the one that would have caught mempalace/mempalace#1733's exact
        shape (0/45,969 drawers ever getting a real `importance` value)."""
        return self._fraction(RankingSignal.MISSING_ORDERING_KEY)

    @property
    def order_inconsistent_rate(self) -> float | None:
        return self._fraction(RankingSignal.ORDER_INCONSISTENT)

    @property
    def not_applicable_rate(self) -> float | None:
        return self._fraction(RankingSignal.NOT_APPLICABLE)


def load_dataset(path: Path | str = DEFAULT_FIXTURE_PATH) -> list[RankingQualityCase]:
    data = json.loads(Path(path).read_text())
    cases: list[dict[str, Any]] = data["cases"]
    return [
        RankingQualityCase(
            case_id=c["case_id"],
            session_id=c["session_id"],
            query=c["query"],
            ranking_field=c["ranking_field"],
            records=[
                RankingQualitySeedRecord(
                    content=r["content"],
                    metadata=r.get("metadata", {}),
                )
                for r in c["records"]
            ],
        )
        for c in cases
    ]


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def classify_ranking_case(
    case: RankingQualityCase,
    query_result: QueryResult,
    stored_ids_in_order: list[str],
) -> tuple[RankingSignal, list[float | None], bool | None]:
    """Classify a single case's outcome from the adapter's query response.

    Returns (final_signal, field_values_in_returned_order,
    matches_insertion_order). Never trusts
    `query_result.ranking_signal` (the adapter's own self-report) as the
    final answer -- see this module's docstring for why -- it is only
    threaded through by the caller for transparency/comparison in the
    result record.
    """
    records = query_result.records
    field_values = [_parse_float(r.metadata.get(case.ranking_field)) for r in records]

    matches_insertion_order: bool | None = None
    if records and stored_ids_in_order:
        returned_ids = [r.memory_id for r in records]
        known_ids = [rid for rid in returned_ids if rid in stored_ids_in_order]
        if known_ids:
            expected = [rid for rid in stored_ids_in_order if rid in returned_ids]
            matches_insertion_order = known_ids == expected

    if len(records) < 2:
        return RankingSignal.NOT_APPLICABLE, field_values, matches_insertion_order

    present_values = [v for v in field_values if v is not None]

    if len(present_values) < len(field_values) or len(set(present_values)) <= 1:
        # The field is missing from at least one returned record, or is
        # present everywhere but identical -- either way, no real
        # per-record signal exists to have driven the order. This is the
        # exact mempalace/mempalace#1733 shape.
        return RankingSignal.MISSING_ORDERING_KEY, field_values, matches_insertion_order

    is_sorted_descending = all(
        a >= b for a, b in zip(present_values, present_values[1:], strict=False)
    )
    if is_sorted_descending:
        return RankingSignal.SIGNAL_DRIVEN, field_values, matches_insertion_order
    return RankingSignal.ORDER_INCONSISTENT, field_values, matches_insertion_order


def run_ranking_quality_eval(
    adapter: MemoryBackendAdapter,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
) -> RankingQualityEvalResult:
    cases = load_dataset(dataset_path)
    result = RankingQualityEvalResult(backend_name=adapter.name, dataset_path=str(dataset_path))

    for case in cases:
        stored_ids: list[str] = []
        try:
            for seed in case.records:
                store_result = adapter.store(case.session_id, seed.content, metadata=seed.metadata)
                stored_ids.append(store_result.memory_id)
            query_result = adapter.query(case.session_id, case.query, top_k=len(case.records))
        except BackendAPIError as exc:
            result.case_results.append(
                RankingQualityCaseResult(
                    case=case,
                    signal=RankingSignal.NOT_APPLICABLE,
                    adapter_reported_signal=None,
                    field_values=[],
                    matches_insertion_order=None,
                    retrieved_content="",
                    error=str(exc),
                )
            )
            continue

        final_signal, field_values, matches_insertion_order = classify_ranking_case(
            case, query_result, stored_ids
        )
        result.case_results.append(
            RankingQualityCaseResult(
                case=case,
                signal=final_signal,
                adapter_reported_signal=query_result.ranking_signal,
                field_values=field_values,
                matches_insertion_order=matches_insertion_order,
                retrieved_content=" ".join(r.content for r in query_result.records),
            )
        )

    return result
