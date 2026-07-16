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
  * EMPTY_OR_LOST    -- the backend DOES have an update primitive and the
                         store()/update()/query() calls all completed
                         without error, but the query came back with zero
                         records -- a silent empty-success, distinct from
                         NOT_APPLICABLE, and never folded into an ordinary
                         miss

Design principle: classification is never a blind pass-through of what an
adapter *claims* happened. Each
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
    metadata: dict[str, str] = field(default_factory=dict)
    """Optional non-fact structured key/value pairs stored alongside
    `initial_fact` (see `run_contradiction_eval` below, which threads this
    into `adapter.store()`'s own `metadata` parameter). Empty for every
    pre-existing case -- this is a purely additive field. Its purpose is
    to exercise the harness's `MemoryRecord.attributes` boundary
    end-to-end: a backend whose query() response echoes structured
    per-record properties (e.g. self-hosted graphiti-core's
    `EntityEdge.attributes`) has somewhere for this eval to observe them,
    instead of every case only ever carrying plain fact text. See
    tests/fixtures/contradiction_cases.json for the case that uses this,
    and docs/methodology.md for the honesty caveat: graphiti-core's real
    `add_episode()` has no generic metadata parameter to receive this
    (confirmed against source, see
    zep_graphiti_selfhosted_adapter.py's module docstring), so that
    specific adapter accepts-and-ignores it, same as every no-op `mode`
    parameter elsewhere in this codebase -- this field threading through
    the harness is not itself proof any adapter surfaces it back out.
    """


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

    @property
    def empty_or_lost_rate(self) -> float | None:
        return self._fraction(ConflictSignal.EMPTY_OR_LOST)

    @property
    def edge_integrity_violation_rate(self) -> float | None:
        return self._fraction(ConflictSignal.EDGE_INTEGRITY_VIOLATION)


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
            metadata=c.get("metadata", {}),
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
      * If it contains neither and the adapter returned zero records at
        all, that is not a genuine "no update primitive" case -- this
        function is only ever called for adapters where
        MemoryBackendAdapter.supports_update is True (see
        run_contradiction_eval below, which short-circuits every case to
        NOT_APPLICABLE before classify_case is ever invoked when it is
        False). A capable backend that completed the store/update/query
        calls with no error and came back with nothing is the "call
        succeeded but silently produced nothing" failure mode --
        classified EMPTY_OR_LOST, distinct from NOT_APPLICABLE, and never
        silently folded into an ordinary miss.
      * If it contains neither but the adapter DID return records (just
        not ones containing either value): if any retrieved record
        carries invalidation metadata, that is still concrete evidence of
        bi-temporal preservation even though neither value's literal text
        matched -- FLAGGED. Otherwise this falls back to NOT_APPLICABLE (a
        genuine "this eval could not observe anything meaningful here").
        An adapter's bare top-level `conflict_signal` claim with no
        corroborating record metadata and no matching text is not
        credible evidence on its own and is not upgraded to a passing
        score. This branch used to collapse to NOT_APPLICABLE
        unconditionally regardless of metadata (dead code -- see git
        history); it now genuinely differentiates on concrete, per-record
        adapter metadata.
      * Structural check, evaluated before all of the above: if ANY
        retrieved record is edge-shaped (its `raw` fragment carries both
        a `source_node_uuid` and a `target_node_uuid` key -- the property
        names graphiti_core's `EntityEdge` writes) but at least one of
        those two values is missing/falsy, the case is classified
        EDGE_INTEGRITY_VIOLATION regardless of what the value-level text
        match would otherwise say. A structurally broken edge (no
        endpoints) is a more fundamental failure than "which value did
        the text contain" -- see ConflictSignal.EDGE_INTEGRITY_VIOLATION
        for the two real graphiti-core bugs (getzep/graphiti#1013,
        #1001) this exists to catch if reproduced against an affected
        version. Records from backends that don't model edges at all
        (no such keys present in `raw`) never trigger this.
    """
    content = " ".join(r.content for r in query_result.records).lower()
    has_initial = case.initial_value.lower() in content
    has_updated = case.updated_value.lower() in content
    has_invalidation_metadata = any(r.metadata.get("invalid_at") for r in query_result.records)

    if any(_edge_endpoints_missing(r) for r in query_result.records):
        return ConflictSignal.EDGE_INTEGRITY_VIOLATION, has_initial, has_updated

    if has_initial and has_updated:
        return ConflictSignal.FLAGGED, has_initial, has_updated
    if has_updated and not has_initial:
        if has_invalidation_metadata:
            return ConflictSignal.FLAGGED, has_initial, has_updated
        return ConflictSignal.SILENT_OVERWRITE, has_initial, has_updated
    if has_initial and not has_updated:
        return ConflictSignal.SERVED_STALE, has_initial, has_updated

    # Neither value is present in the retrieved text.
    if not query_result.records:
        # The backend returned zero records for a capable, no-error call.
        # This is a silent empty-success, not "no update primitive" --
        # see ConflictSignal.EMPTY_OR_LOST for the distinction.
        return ConflictSignal.EMPTY_OR_LOST, has_initial, has_updated

    # Records came back, just none of them matched either value as plain
    # text. This used to collapse straight to NOT_APPLICABLE regardless of
    # what the adapter reported (dead code -- see git history). It now
    # genuinely differentiates on concrete, per-record adapter metadata: a
    # bi-temporal backend can still have preserved the old fact even when
    # neither value's literal text matched.
    if has_invalidation_metadata:
        return ConflictSignal.FLAGGED, has_initial, has_updated
    return ConflictSignal.NOT_APPLICABLE, has_initial, has_updated


def _edge_endpoints_missing(record: object) -> bool:
    """True if `record.raw` is edge-shaped (carries both a
    `source_node_uuid` and a `target_node_uuid` key -- the property names
    graphiti_core's `EntityEdge` writes on every relationship, see
    `zep_graphiti_selfhosted_adapter.py`) but at least one of those two
    values is missing or falsy. Records whose `raw` fragment has neither
    key at all (i.e. this backend doesn't model edges, or this record
    isn't edge-shaped) never trigger this -- it is a structural check on
    edge-shaped records only, not a generic "does this record have an id"
    heuristic.
    """
    raw = getattr(record, "raw", None)
    if not isinstance(raw, dict):
        return False
    if "source_node_uuid" not in raw and "target_node_uuid" not in raw:
        return False
    return not raw.get("source_node_uuid") or not raw.get("target_node_uuid")


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
            # `metadata` carries the case's optional non-fact structured
            # properties (see ContradictionCase.metadata) through to the
            # adapter's own store() -- empty dict for every pre-existing
            # case, so `metadata or None` preserves every adapter's
            # existing no-metadata call shape exactly.
            store_result = adapter.store(
                case.session_id, case.initial_fact, metadata=case.metadata or None
            )
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
