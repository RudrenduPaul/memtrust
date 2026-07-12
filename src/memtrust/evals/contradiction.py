"""MemTrust's original multi-hop contradiction-detection eval.

This is the differentiating wedge described in docs/methodology.md and
the project README: LongMemEval and LoCoMo both measure recall -- can the
backend remember a fact you told it earlier. Neither measures what a
backend does when two stored facts conflict, which is the question that
actually matters once a memory system is running underneath a production
agent. If a user says a meeting is at 2pm and later says it moved to 3pm,
does the backend:

  * FLAGGED          -- surface the conflict (return both values, an
                         explicit "this was updated" marker, or otherwise
                         make the contradiction visible to the caller)
  * SILENT_OVERWRITE -- replace the old fact with the new one and give no
                         signal a prior, different value ever existed
  * SERVED_STALE     -- return the old fact and give no signal that a
                         newer, conflicting fact was stored since
  * NOT_APPLICABLE   -- the backend has no update primitive this eval can
                         exercise (MemoryBackendAdapter.supports_update is
                         False); recorded explicitly, never silently
                         dropped from the results table

Design principle (see [redacted] [redacted]): classification is
never a blind pass-through of what an adapter *claims* happened. Each
case's final verdict cross-checks the adapter-reported ConflictSignal
against the actual retrieved content (does it contain the old value, the
new value, or both) -- so a vendor's own optimistic self-report cannot
silently become the eval's score. See `classify_case` below for the exact
logic, which is the part of this file most worth reading closely before
trusting a published contradiction-detection number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memtrust.adapters.base import (
    BackendAPIError,
    ConflictSignal,
    MemoryBackendAdapter,
    QueryResult,
)

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "contradiction_cases.json"
)


@dataclass
class ContradictionCase:
    case_id: str
    session_id: str
    subject: str
    initial_fact: str
    contradicting_fact: str
    query: str
    initial_value: str
    updated_value: str


@dataclass
class ContradictionCaseResult:
    case: ContradictionCase
    signal: ConflictSignal
    adapter_reported_signal: ConflictSignal | None
    contains_initial_value: bool
    contains_updated_value: bool
    retrieved_content: str
    error: str | None = None


@dataclass
class ContradictionEvalResult:
    backend_name: str
    dataset_path: str
    case_results: list[ContradictionCaseResult] = field(default_factory=list)

    @property
    def scored_cases(self) -> list[ContradictionCaseResult]:
        return [c for c in self.case_results if c.error is None]

    def _fraction(self, signal: ConflictSignal) -> float | None:
        scored = self.scored_cases
        if not scored:
            return None
        matching = sum(1 for c in scored if c.signal == signal)
        return matching / len(scored)

    @property
    def flagged_rate(self) -> float | None:
        return self._fraction(ConflictSignal.FLAGGED)

    @property
    def silent_overwrite_rate(self) -> float | None:
        return self._fraction(ConflictSignal.SILENT_OVERWRITE)

    @property
    def served_stale_rate(self) -> float | None:
        return self._fraction(ConflictSignal.SERVED_STALE)

    @property
    def not_applicable_rate(self) -> float | None:
        return self._fraction(ConflictSignal.NOT_APPLICABLE)


def load_dataset(path: Path | str = DEFAULT_FIXTURE_PATH) -> list[ContradictionCase]:
    data = json.loads(Path(path).read_text())
    cases: list[dict[str, Any]] = data["cases"]
    return [
        ContradictionCase(
            case_id=c["case_id"],
            session_id=c["session_id"],
            subject=c["subject"],
            initial_fact=c["initial_fact"],
            contradicting_fact=c["contradicting_fact"],
            query=c["query"],
            initial_value=c["initial_value"],
            updated_value=c["updated_value"],
        )
        for c in cases
    ]


def classify_case(
    case: ContradictionCase, query_result: QueryResult
) -> tuple[ConflictSignal, bool, bool]:
    """Classify a single case's outcome from the adapter's query response.

    Returns (final_signal, contains_initial_value, contains_updated_value).

    Cross-checks the adapter's self-reported `conflict_signal` against the
    actual retrieved text rather than trusting it outright, and -- where
    the text alone is ambiguous -- consults per-record adapter metadata
    (`MemoryRecord.metadata`) as corroborating evidence rather than a bare
    self-report. A bi-temporal backend like Graphiti/Zep stamps a
    superseded edge's `invalid_at` field in that metadata even when the
    edge's raw text doesn't literally contain the case's old-value string
    (paraphrased extraction) or when the fixed-size top-k window makes the
    plain substring signal unreliable. That per-record marker is concrete
    adapter-reported evidence, unlike the adapter's bare top-level
    `conflict_signal` enum, which is never trusted on its own:

      * If the retrieved content contains BOTH the old and new value, the
        contradiction is visible in the response regardless of what the
        adapter claims -- classified FLAGGED.
      * If it contains only the new value, that normally reads as a
        silent overwrite -- SILENT_OVERWRITE. But if any retrieved record
        carries invalidation metadata (e.g. Graphiti's `invalid_at`), the
        backend did preserve the old fact bi-temporally even though this
        eval's literal substring match on the case's old-value string
        didn't pick it up -- that corroborated signal reclassifies the
        case as FLAGGED instead of a false-negative silent overwrite.
      * If it contains only the old value, the backend never picked up
        the update at all from the caller's perspective -- SERVED_STALE.
      * If it contains neither (backend returned nothing relevant, or
        genuinely has no update primitive): if any retrieved record
        carries invalidation metadata, that is still concrete evidence of
        bi-temporal preservation even though neither value's literal text
        matched -- FLAGGED. Otherwise there is genuinely nothing to read,
        from either text or metadata, so this falls back to
        NOT_APPLICABLE. An adapter's bare top-level `conflict_signal`
        claim (e.g. FLAGGED) with no corroborating record metadata and no
        matching text is not credible evidence on its own and is not
        upgraded to a passing score.
    """
    content = " ".join(r.content for r in query_result.records).lower()
    has_initial = case.initial_value.lower() in content
    has_updated = case.updated_value.lower() in content
    has_invalidation_metadata = any(r.metadata.get("invalid_at") for r in query_result.records)

    if has_initial and has_updated:
        return ConflictSignal.FLAGGED, has_initial, has_updated
    if has_updated and not has_initial:
        if has_invalidation_metadata:
            return ConflictSignal.FLAGGED, has_initial, has_updated
        return ConflictSignal.SILENT_OVERWRITE, has_initial, has_updated
    if has_initial and not has_updated:
        return ConflictSignal.SERVED_STALE, has_initial, has_updated

    # Neither value is present in the retrieved text. This used to
    # collapse to NOT_APPLICABLE regardless of what the adapter reported
    # (dead code -- see git history). It now genuinely differentiates on
    # concrete, per-record adapter metadata rather than the adapter's
    # bare self-reported enum.
    if has_invalidation_metadata:
        return ConflictSignal.FLAGGED, has_initial, has_updated
    return ConflictSignal.NOT_APPLICABLE, has_initial, has_updated


def run_contradiction_eval(
    adapter: MemoryBackendAdapter,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
) -> ContradictionEvalResult:
    cases = load_dataset(dataset_path)
    result = ContradictionEvalResult(backend_name=adapter.name, dataset_path=str(dataset_path))

    if not adapter.supports_update:
        for case in cases:
            result.case_results.append(
                ContradictionCaseResult(
                    case=case,
                    signal=ConflictSignal.NOT_APPLICABLE,
                    adapter_reported_signal=None,
                    contains_initial_value=False,
                    contains_updated_value=False,
                    retrieved_content="",
                    error=None,
                )
            )
        return result

    for case in cases:
        try:
            store_result = adapter.store(case.session_id, case.initial_fact)
            adapter.update(case.session_id, store_result.memory_id, case.contradicting_fact)
            query_result = adapter.query(case.session_id, case.query, top_k=5)
        except BackendAPIError as exc:
            result.case_results.append(
                ContradictionCaseResult(
                    case=case,
                    signal=ConflictSignal.NOT_APPLICABLE,
                    adapter_reported_signal=None,
                    contains_initial_value=False,
                    contains_updated_value=False,
                    retrieved_content="",
                    error=str(exc),
                )
            )
            continue

        final_signal, has_initial, has_updated = classify_case(case, query_result)
        result.case_results.append(
            ContradictionCaseResult(
                case=case,
                signal=final_signal,
                adapter_reported_signal=query_result.conflict_signal,
                contains_initial_value=has_initial,
                contains_updated_value=has_updated,
                retrieved_content=" ".join(r.content for r in query_result.records),
            )
        )

    return result
