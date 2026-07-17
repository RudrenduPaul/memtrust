"""memtrust's temporal-KG as_of point-in-time boundary eval for
MemPalaceAdapter's `kg_add()`/`kg_invalidate()`/`kg_query()` primitives.

Motivating case: MemPalace/mempalace#1913, fixed by merged PR#1914
(contributor ggettert; `gh pr view 1914 --repo MemPalace/mempalace
--comments`). `mempalace/knowledge_graph.py`'s `_temporal_filter_sql` used a
*closed* interval on both ends (`valid_from <= as_of AND valid_to >=
as_of`), so a fact whose `valid_to` equals the query's `as_of` instant still
matched. When a caller hand-rolls a fact change as `kg_invalidate(ended=T)`
immediately followed by `kg_add(valid_from=T)` at the identical boundary
instant `T` -- the exact pattern MemPalace's own pre-fix `PALACE_PROTOCOL`
wake-up guidance told every agent to do -- an `as_of=T` query matched BOTH
the old, just-ended fact and the new, just-started fact simultaneously, so a
single-valued predicate (e.g. "uses_model") reported two contradictory
values at once with no error or warning. PR#1914's real fix switched the
upper-bound comparison to strict (`valid_to > as_of`), i.e. a half-open
interval `[valid_from, valid_to)`: a fact that ended exactly at the query
instant no longer matches, so only its successor does. PR#1914 also added a
new `supersede()` primitive (`mempalace_kg_supersede`) as the *preferred*
way to make this kind of change atomically -- this eval deliberately does
NOT use it, because `MemPalaceAdapter` does not wire `tool_kg_supersede` at
all (only `tool_kg_add`/`tool_kg_invalidate`/`tool_kg_query` are confirmed
real and wired -- see mempalace_adapter.py's module docstring). This eval
exercises exactly the failure-prone hand-rolled pattern PR#1914's own
PALACE_PROTOCOL fix warns callers away from, because that pattern is what
`MemPalaceAdapter`'s current, narrower capability surface actually produces
-- adding `kg_supersede` support to this adapter is a separate, later
change, out of scope here.

Scope note -- what this eval is and is not. This is a genuinely scoped
fix for ONE of the two capabilities a prior backlog item bundled together
("Add MemPalace temporal-KG as_of point-in-time queries" +
"drawer neighbor-expansion scoping"). This eval covers only the temporal-KG
as_of half. Drawer neighbor-expansion/parent_drawer_id-leak scoping is a
separate, larger, still-deferred capability this eval does not touch.

Honest scope on live verification (same convention every eval module in
this package states plainly, see docs/methodology.md): the real `mempalace`
PyPI package is not installed in this build environment, and PR#1914 itself
had not shipped in a released `mempalace` version as of this adapter's
"live-verified at version 3.5.0" build (the fix lands under CHANGELOG.md's
`## [Unreleased]` section, above the `## [3.5.0]` entry -- confirmed by
reading the merged PR's own diff). This eval's own test suite
(tests/test_temporal_kg_boundary.py) therefore proves the *classification
logic* is correct against two from-scratch fake `_MCPToolsProtocol`
implementations that reproduce the real, confirmed pre-#1914 (closed-
interval) and post-#1914 (half-open) `_temporal_filter_sql` comparison
exactly -- not against a live MemPalace instance running either version.

Design:

  1. `kg_add(subject, predicate, old_object, valid_from=<seed>)` -- seed the
     fact this case will supersede.
  2. `kg_invalidate(subject, predicate, old_object, ended=<boundary>)` --
     end it at a known instant.
  3. `kg_add(subject, predicate, new_object, valid_from=<boundary>)` -- the
     hand-rolled handover, sharing the exact same boundary instant as (2).
  4. `kg_query(subject, as_of=<boundary>, direction="outgoing")` -- query at
     the exact shared instant.
  5. Classify the response two ways: `MemPalaceAdapter.kg_query()`'s own
     self-reported `boundary_signal` (see adapters/base.py's
     `TemporalBoundarySignal` and mempalace_adapter.py's
     `_classify_boundary_signal`), and this eval's own, independently
     written `classify_temporal_kg_boundary_case()` below, which re-derives
     the verdict directly from the raw `KGFact` list `kg_query()` returned
     (how many distinct objects exist for the queried `(subject,
     predicate)`). The eval's own classification -- never the adapter's
     self-report -- is what `TemporalKGBoundaryCaseResult.signal` records;
     `adapter_reported_signal` is kept only for comparison/transparency,
     the same "claim, not proof" convention evals/ranking_quality.py's
     `classify_ranking_case()` already establishes for
     `RankingSignal.SIGNAL_DRIVEN`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from memtrust.adapters.base import BackendAPIError, TemporalBoundarySignal
from memtrust.adapters.mempalace_adapter import KGQueryResult, MemPalaceAdapter


@dataclass
class TemporalKGBoundaryCase:
    case_id: str
    subject: str
    predicate: str
    old_object: str
    """Value of the fact before the boundary."""
    new_object: str
    """Value of the fact from the boundary onward -- the hand-rolled
    successor, added via kg_add(valid_from=boundary), sharing the exact
    same instant as the kg_invalidate(ended=boundary) call that closed
    old_object."""
    seed_valid_from: str
    """When old_object's fact starts being true, well before `boundary`."""
    boundary: str
    """The single shared instant used for BOTH kg_invalidate(ended=...)
    and the successor kg_add(valid_from=...), and then queried via
    kg_query(as_of=...) -- the exact MemPalace/mempalace#1913 shape."""


def default_cases(subject_prefix: str | None = None) -> list[TemporalKGBoundaryCase]:
    """One case exercising the exact PR#1914 boundary shape. `subject` is
    randomized by default (uuid4-suffixed) -- see mempalace_adapter.py's
    module docstring "KNOWLEDGE-GRAPH STORAGE IGNORES MEMPALACE_PALACE_PATH"
    section: the real KG store is one fixed, environment-global file with
    no per-run isolation, so a fixed literal subject would make this case's
    assertions invalid against any prior run's leftover facts (the same
    randomization tests/test_adapters.py's own
    `test_real_mempalace_kg_add_invalidate_query_round_trip` already uses).
    """
    subject = subject_prefix or f"memtrust-temporal-kg-boundary-{uuid.uuid4().hex}"
    return [
        TemporalKGBoundaryCase(
            case_id="mt-tkb-001",
            subject=subject,
            predicate="uses_model",
            old_object="claude-opus-4-7",
            new_object="claude-opus-4-8",
            seed_valid_from="2026-05-01T00:00:00Z",
            boundary="2026-06-02T12:00:00Z",
        ),
    ]


@dataclass
class TemporalKGBoundaryCaseResult:
    case: TemporalKGBoundaryCase
    signal: TemporalBoundarySignal
    """This eval's own, independently-derived verdict -- see
    `classify_temporal_kg_boundary_case()` below. Never copied straight
    from the adapter's self-report."""
    adapter_reported_signal: TemporalBoundarySignal | None
    """`KGQueryResult.boundary_signal` as MemPalaceAdapter.kg_query()
    itself reported it, kept for comparison/transparency only. `None` when
    the call sequence failed before a query response ever came back."""
    objects_at_boundary: list[str]
    """Every distinct `object` value returned for `(case.subject,
    case.predicate)` at `case.boundary` -- the evidence `signal` was
    computed from. Length 2+ is the literal double-count."""
    error: str | None = None

    @property
    def self_report_agrees(self) -> bool:
        return self.adapter_reported_signal == self.signal


@dataclass
class TemporalKGBoundaryEvalResult:
    backend_name: str
    case_results: list[TemporalKGBoundaryCaseResult] = field(default_factory=list)

    @property
    def scored_cases(self) -> list[TemporalKGBoundaryCaseResult]:
        return [r for r in self.case_results if r.error is None]

    def _fraction(self, signal: TemporalBoundarySignal) -> float | None:
        scored = self.scored_cases
        if not scored:
            return None
        return sum(1 for r in scored if r.signal == signal) / len(scored)

    @property
    def double_count_rate(self) -> float | None:
        """Fraction of cases where an as_of query at the exact boundary
        instant returned both the superseded fact and its successor -- the
        headline metric this eval exists to surface, and the one that
        would flag MemPalace/mempalace#1913's exact shape against a
        pre-#1914 deployment."""
        return self._fraction(TemporalBoundarySignal.DOUBLE_COUNT)

    @property
    def clean_rate(self) -> float | None:
        return self._fraction(TemporalBoundarySignal.CLEAN)

    @property
    def not_applicable_rate(self) -> float | None:
        return self._fraction(TemporalBoundarySignal.NOT_APPLICABLE)

    @property
    def self_report_agreement_rate(self) -> float | None:
        """Fraction of scored cases where MemPalaceAdapter.kg_query()'s own
        self-reported boundary_signal matched this eval's independently
        derived verdict -- a low rate here would mean the adapter's
        self-classification (adapters/mempalace_adapter.py's
        `_classify_boundary_signal`) itself has a bug, distinct from
        whether the *backend* double-counts."""
        scored = self.scored_cases
        if not scored:
            return None
        return sum(1 for r in scored if r.self_report_agrees) / len(scored)


def classify_temporal_kg_boundary_case(
    case: TemporalKGBoundaryCase, query_result: KGQueryResult
) -> tuple[TemporalBoundarySignal, list[str]]:
    """Independently classify one case's outcome straight from the raw
    `KGFact` list `query_result.facts` carries -- deliberately NOT calling
    or trusting `query_result.boundary_signal` (the adapter's own
    self-report; see TemporalKGBoundarySignal in adapters/base.py for why
    an eval must never treat a signal field as the final answer on its
    own). Returns (signal, distinct_objects_returned).

    Only facts matching this case's exact `(subject, predicate)` on the
    `outgoing` direction are considered -- `kg_query()` was called with
    `direction="outgoing"`, so every returned fact should already match,
    but this filters defensively rather than assuming the vendor response
    never carries an unrelated fact.
    """
    matching = [
        f
        for f in query_result.facts
        if f.direction == "outgoing" and f.subject == case.subject and f.predicate == case.predicate
    ]
    objects = sorted({f.object for f in matching})
    if not matching:
        return TemporalBoundarySignal.NOT_APPLICABLE, objects
    if len(objects) >= 2:
        return TemporalBoundarySignal.DOUBLE_COUNT, objects
    return TemporalBoundarySignal.CLEAN, objects


def run_temporal_kg_boundary_eval(
    adapter: MemPalaceAdapter,
    cases: list[TemporalKGBoundaryCase] | None = None,
) -> TemporalKGBoundaryEvalResult:
    resolved_cases = cases if cases is not None else default_cases()
    result = TemporalKGBoundaryEvalResult(backend_name=adapter.name)

    for case in resolved_cases:
        try:
            add_old = adapter.kg_add(
                case.subject, case.predicate, case.old_object, valid_from=case.seed_valid_from
            )
            if not add_old.success:
                raise BackendAPIError(
                    adapter.name, add_old.error or "kg_add(old_object) reported failure"
                )

            invalidate = adapter.kg_invalidate(
                case.subject, case.predicate, case.old_object, ended=case.boundary
            )
            if not invalidate.success:
                raise BackendAPIError(
                    adapter.name, invalidate.error or "kg_invalidate() reported failure"
                )

            add_new = adapter.kg_add(
                case.subject, case.predicate, case.new_object, valid_from=case.boundary
            )
            if not add_new.success:
                raise BackendAPIError(
                    adapter.name, add_new.error or "kg_add(new_object) reported failure"
                )

            query_result = adapter.kg_query(case.subject, as_of=case.boundary, direction="outgoing")
        except BackendAPIError as exc:
            result.case_results.append(
                TemporalKGBoundaryCaseResult(
                    case=case,
                    signal=TemporalBoundarySignal.NOT_APPLICABLE,
                    adapter_reported_signal=None,
                    objects_at_boundary=[],
                    error=str(exc),
                )
            )
            continue

        signal, objects = classify_temporal_kg_boundary_case(case, query_result)
        result.case_results.append(
            TemporalKGBoundaryCaseResult(
                case=case,
                signal=signal,
                adapter_reported_signal=query_result.boundary_signal,
                objects_at_boundary=objects,
            )
        )

    return result
