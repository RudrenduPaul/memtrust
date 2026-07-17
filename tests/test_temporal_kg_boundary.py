"""Tests for evals/temporal_kg_boundary.py and
MemPalaceAdapter.kg_query()'s new `boundary_signal` self-report
(adapters/mempalace_adapter.py's `_classify_boundary_signal`,
adapters/base.py's `TemporalBoundarySignal`).

Kept in a dedicated file rather than folded into test_adapters.py/
test_evals.py -- same convention tests/test_episode_temporal_leak.py
already establishes for a capability that is MemPalace-specific, not part
of the shared `MemoryBackendAdapter` interface every other eval in
test_evals.py exercises through store()/query().

The real `mempalace` package is not installed in this environment (see
mempalace_adapter.py's module docstring and docs/methodology.md), and
MemPalace/mempalace#1914's real fix had not shipped in a released version
as of this adapter's "live-verified at 3.5.0" build (the fix sits under
CHANGELOG.md's `## [Unreleased]` heading in the real repo, confirmed via
`gh pr diff 1914 --repo MemPalace/mempalace`). So every test here exercises
this adapter's own logic against `_RealSemanticsFakeKGTools` below -- a
from-scratch fake purpose-built to reproduce the REAL, confirmed
`_temporal_filter_sql` upper-bound comparison exactly (both the pre-#1914
closed-interval bug and the post-#1914 half-open fix, selected via
`boundary_mode`), not memtrust's own guess at what a KG backend does.
Unlike `test_adapters.py`'s `FakeMCPTools.tool_kg_query()` (which performs
NO as_of filtering at all and just echoes every stored fact back
regardless of `as_of`), this fake actually implements the boundary
comparison, because that exact comparison is the bug this eval exists to
detect. These tests prove the classification logic is correct given each
response shape; they do not prove either shape matches a live MemPalace
instance -- see this module's own eval docstring for the full caveat.
"""

from __future__ import annotations

from typing import Any

import pytest

from memtrust.adapters.base import BackendAPIError, TemporalBoundarySignal
from memtrust.adapters.mempalace_adapter import KGFact, KGQueryResult, MemPalaceAdapter
from memtrust.evals.temporal_kg_boundary import (
    TemporalKGBoundaryCase,
    classify_temporal_kg_boundary_case,
    default_cases,
    run_temporal_kg_boundary_eval,
)


class _RealSemanticsFakeKGTools:
    """Reproduces the real, confirmed `_temporal_filter_sql` upper-bound
    comparison from `gh pr diff 1914 --repo MemPalace/mempalace`:

      * ``boundary_mode="closed"``  -- pre-#1914: ``valid_to >= as_of``
        (a fact ending exactly at the query instant still matches).
      * ``boundary_mode="half_open"`` -- post-#1914 (PR#1914's real fix):
        ``valid_to > as_of``, strict (a fact ending exactly at the query
        instant no longer matches).

    Every timestamp this test module uses is a fixed-width canonical UTC
    ISO-8601 string (``YYYY-MM-DDTHH:MM:SSZ``), so plain string comparison
    is chronologically correct -- the same property the real
    `_temporal_start_key`/`_temporal_end_key` normalization in
    `mempalace/knowledge_graph.py` exists to guarantee for the real,
    wider variety of caller-supplied formats (date-only, etc.); this fake
    doesn't need that normalization because every case in this test file
    only ever passes already-canonical datetimes.
    """

    def __init__(self, boundary_mode: str = "half_open") -> None:
        assert boundary_mode in ("half_open", "closed")
        self._boundary_mode = boundary_mode
        self._facts: list[dict[str, Any]] = []

    def tool_kg_add(
        self,
        subject: str,
        predicate: str,
        object: str,  # noqa: A002 - mirrors the real tool_kg_add() parameter name
        valid_from: str | None = None,
        valid_to: str | None = None,
        source_closet: str | None = None,
        source_file: str | None = None,
        source_drawer_id: str | None = None,
    ) -> dict[str, Any]:
        triple_id = f"t_{subject}_{predicate}_{object}_{len(self._facts)}"
        self._facts.append(
            {
                "triple_id": triple_id,
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "confidence": 1.0,
                "source_closet": source_closet,
            }
        )
        return {
            "success": True,
            "triple_id": triple_id,
            "fact": f"{subject} → {predicate} → {object}",
        }

    def tool_kg_invalidate(
        self,
        subject: str,
        predicate: str,
        object: str,  # noqa: A002 - mirrors the real tool_kg_invalidate() parameter name
        ended: str | None = None,
    ) -> dict[str, Any]:
        resolved = ended or "2026-01-01T00:00:00Z"
        for f in self._facts:
            if (
                f["subject"] == subject
                and f["predicate"] == predicate
                and f["object"] == object
                and f["valid_to"] is None
            ):
                f["valid_to"] = resolved
        return {"success": True, "fact": f"{subject} → {predicate} → {object}", "ended": resolved}

    def _matches_as_of(self, fact: dict[str, Any], as_of: str) -> bool:
        valid_from = fact["valid_from"]
        valid_to = fact["valid_to"]
        if valid_from is not None and valid_from > as_of:
            return False
        if valid_to is not None:
            if self._boundary_mode == "closed":
                if valid_to < as_of:
                    return False
            else:
                if valid_to <= as_of:
                    return False
        return True

    def tool_kg_query(
        self, entity: str, as_of: str | None = None, direction: str = "both"
    ) -> dict[str, Any]:
        facts = []
        for f in self._facts:
            if direction in ("outgoing", "both") and f["subject"] == entity:
                item = {**f, "direction": "outgoing"}
            elif direction in ("incoming", "both") and f["object"] == entity:
                item = {**f, "direction": "incoming"}
            else:
                continue
            if as_of is not None and not self._matches_as_of(f, as_of):
                continue
            item["current"] = f["valid_to"] is None
            facts.append(item)
        return {"entity": entity, "as_of": as_of, "facts": facts, "count": len(facts)}


def _case() -> TemporalKGBoundaryCase:
    return TemporalKGBoundaryCase(
        case_id="mt-tkb-test",
        subject="Bot",
        predicate="uses_model",
        old_object="claude-opus-4-7",
        new_object="claude-opus-4-8",
        seed_valid_from="2026-05-01T00:00:00Z",
        boundary="2026-06-02T12:00:00Z",
    )


# ---------------------------------------------------------------------------
# _classify_boundary_signal() -- MemPalaceAdapter.kg_query()'s self-report
# ---------------------------------------------------------------------------


def test_kg_query_self_reports_double_count_against_pre_1914_closed_interval_fake() -> None:
    """The exact MemPalace/mempalace#1913 shape: hand-rolled invalidate()
    + add() at one shared boundary instant, queried against a fake that
    reproduces the real PRE-#1914 closed-interval `_temporal_filter_sql`
    -- the query must return BOTH values and self-report DOUBLE_COUNT."""
    case = _case()
    adapter = MemPalaceAdapter(mcp_tools=_RealSemanticsFakeKGTools(boundary_mode="closed"))

    adapter.kg_add(case.subject, case.predicate, case.old_object, valid_from=case.seed_valid_from)
    adapter.kg_invalidate(case.subject, case.predicate, case.old_object, ended=case.boundary)
    adapter.kg_add(case.subject, case.predicate, case.new_object, valid_from=case.boundary)

    result = adapter.kg_query(case.subject, as_of=case.boundary, direction="outgoing")

    objects = sorted(f.object for f in result.facts)
    assert objects == sorted([case.old_object, case.new_object])
    assert result.boundary_signal == TemporalBoundarySignal.DOUBLE_COUNT


def test_kg_query_self_reports_clean_against_post_1914_half_open_fake() -> None:
    """The identical hand-rolled sequence, but against a fake reproducing
    the real POST-#1914 half-open fix -- the boundary instant excludes the
    ended fact, so only the successor is returned and self-report is
    CLEAN. The negative control proving no false positive on a fixed
    backend."""
    case = _case()
    adapter = MemPalaceAdapter(mcp_tools=_RealSemanticsFakeKGTools(boundary_mode="half_open"))

    adapter.kg_add(case.subject, case.predicate, case.old_object, valid_from=case.seed_valid_from)
    adapter.kg_invalidate(case.subject, case.predicate, case.old_object, ended=case.boundary)
    adapter.kg_add(case.subject, case.predicate, case.new_object, valid_from=case.boundary)

    result = adapter.kg_query(case.subject, as_of=case.boundary, direction="outgoing")

    objects = sorted(f.object for f in result.facts)
    assert objects == [case.new_object]
    assert result.boundary_signal == TemporalBoundarySignal.CLEAN


def test_kg_query_boundary_signal_not_applicable_without_as_of() -> None:
    case = _case()
    adapter = MemPalaceAdapter(mcp_tools=_RealSemanticsFakeKGTools(boundary_mode="half_open"))
    adapter.kg_add(case.subject, case.predicate, case.old_object, valid_from=case.seed_valid_from)

    result = adapter.kg_query(case.subject)

    assert result.boundary_signal == TemporalBoundarySignal.NOT_APPLICABLE


def test_kg_query_boundary_signal_not_applicable_when_no_facts_returned() -> None:
    adapter = MemPalaceAdapter(mcp_tools=_RealSemanticsFakeKGTools(boundary_mode="half_open"))

    result = adapter.kg_query("nobody-ever-added", as_of="2026-06-02T12:00:00Z")

    assert result.facts == []
    assert result.boundary_signal == TemporalBoundarySignal.NOT_APPLICABLE


def test_kg_query_boundary_signal_ignores_unrelated_multi_valued_predicate() -> None:
    """Two genuinely different, simultaneously-true facts for the same
    predicate (never invalidated, no shared boundary instant at all) must
    NOT be flagged DOUBLE_COUNT -- only the literal "one ended here,
    another started here" shared-instant shape counts. Guards against
    `_classify_boundary_signal` over-triggering on any predicate that
    simply has 2+ values."""
    adapter = MemPalaceAdapter(mcp_tools=_RealSemanticsFakeKGTools(boundary_mode="half_open"))
    adapter.kg_add("Bot", "speaks_language", "english", valid_from="2026-01-01T00:00:00Z")
    adapter.kg_add("Bot", "speaks_language", "spanish", valid_from="2026-01-01T00:00:00Z")

    result = adapter.kg_query("Bot", as_of="2026-06-02T12:00:00Z", direction="outgoing")

    assert sorted(f.object for f in result.facts) == ["english", "spanish"]
    assert result.boundary_signal == TemporalBoundarySignal.CLEAN


# ---------------------------------------------------------------------------
# classify_temporal_kg_boundary_case() -- the eval's OWN, independent verdict
# ---------------------------------------------------------------------------


def test_classify_case_double_count_from_raw_facts() -> None:
    case = _case()
    query_result = KGQueryResult(
        entity=case.subject,
        as_of=case.boundary,
        facts=[
            KGFact(
                direction="outgoing",
                subject=case.subject,
                predicate=case.predicate,
                object=case.old_object,
                valid_from=case.seed_valid_from,
                valid_to=case.boundary,
                confidence=1.0,
                source_closet=None,
                current=False,
            ),
            KGFact(
                direction="outgoing",
                subject=case.subject,
                predicate=case.predicate,
                object=case.new_object,
                valid_from=case.boundary,
                valid_to=None,
                confidence=1.0,
                source_closet=None,
                current=True,
            ),
        ],
        count=2,
        latency_ms=1.0,
        # Deliberately wrong self-report -- proves classify_temporal_kg_boundary_case()
        # derives its own verdict from `facts`, never from this field.
        boundary_signal=TemporalBoundarySignal.CLEAN,
    )

    signal, objects = classify_temporal_kg_boundary_case(case, query_result)

    assert signal == TemporalBoundarySignal.DOUBLE_COUNT
    assert objects == sorted([case.old_object, case.new_object])


def test_classify_case_clean_from_raw_facts() -> None:
    case = _case()
    query_result = KGQueryResult(
        entity=case.subject,
        as_of=case.boundary,
        facts=[
            KGFact(
                direction="outgoing",
                subject=case.subject,
                predicate=case.predicate,
                object=case.new_object,
                valid_from=case.boundary,
                valid_to=None,
                confidence=1.0,
                source_closet=None,
                current=True,
            ),
        ],
        count=1,
        latency_ms=1.0,
        boundary_signal=TemporalBoundarySignal.CLEAN,
    )

    signal, objects = classify_temporal_kg_boundary_case(case, query_result)

    assert signal == TemporalBoundarySignal.CLEAN
    assert objects == [case.new_object]


def test_classify_case_not_applicable_when_no_matching_facts() -> None:
    case = _case()
    query_result = KGQueryResult(
        entity=case.subject, as_of=case.boundary, facts=[], count=0, latency_ms=1.0
    )

    signal, objects = classify_temporal_kg_boundary_case(case, query_result)

    assert signal == TemporalBoundarySignal.NOT_APPLICABLE
    assert objects == []


# ---------------------------------------------------------------------------
# run_temporal_kg_boundary_eval() -- full pipeline
# ---------------------------------------------------------------------------


def test_run_eval_detects_double_count_against_pre_1914_backend() -> None:
    """Positive case: the eval must actually DETECT a deliberately-broken
    (pre-#1914, closed-interval) fake response end to end."""
    case = _case()
    adapter = MemPalaceAdapter(mcp_tools=_RealSemanticsFakeKGTools(boundary_mode="closed"))

    result = run_temporal_kg_boundary_eval(adapter, cases=[case])

    assert len(result.case_results) == 1
    case_result = result.case_results[0]
    assert case_result.error is None
    assert case_result.signal == TemporalBoundarySignal.DOUBLE_COUNT
    assert case_result.adapter_reported_signal == TemporalBoundarySignal.DOUBLE_COUNT
    assert case_result.self_report_agrees is True
    assert sorted(case_result.objects_at_boundary) == sorted([case.old_object, case.new_object])
    assert result.double_count_rate == 1.0
    assert result.clean_rate == 0.0
    assert result.self_report_agreement_rate == 1.0


def test_run_eval_no_false_positive_against_post_1914_backend() -> None:
    """Negative control: the identical hand-rolled sequence against a
    fixed (post-#1914, half-open) backend must classify CLEAN, never
    DOUBLE_COUNT -- proving the eval doesn't over-trigger."""
    case = _case()
    adapter = MemPalaceAdapter(mcp_tools=_RealSemanticsFakeKGTools(boundary_mode="half_open"))

    result = run_temporal_kg_boundary_eval(adapter, cases=[case])

    assert len(result.case_results) == 1
    case_result = result.case_results[0]
    assert case_result.error is None
    assert case_result.signal == TemporalBoundarySignal.CLEAN
    assert case_result.adapter_reported_signal == TemporalBoundarySignal.CLEAN
    assert case_result.self_report_agrees is True
    assert case_result.objects_at_boundary == [case.new_object]
    assert result.clean_rate == 1.0
    assert result.double_count_rate == 0.0


def test_run_eval_records_not_applicable_when_kg_add_fails() -> None:
    class RejectingTools(_RealSemanticsFakeKGTools):
        def tool_kg_add(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"success": False, "error": "sanitizer rejected this value"}

    case = _case()
    adapter = MemPalaceAdapter(mcp_tools=RejectingTools(boundary_mode="half_open"))

    result = run_temporal_kg_boundary_eval(adapter, cases=[case])

    assert len(result.case_results) == 1
    case_result = result.case_results[0]
    assert case_result.error is not None
    assert case_result.signal == TemporalBoundarySignal.NOT_APPLICABLE
    assert case_result.adapter_reported_signal is None
    # A failed case is excluded from scored_cases -- rates below should not
    # divide by a case that never produced a real verdict.
    assert result.scored_cases == []
    assert result.double_count_rate is None


def test_run_eval_wraps_vendor_exception_as_not_applicable() -> None:
    class RaisingTools(_RealSemanticsFakeKGTools):
        def tool_kg_query(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("vendor exploded")

    case = _case()
    adapter = MemPalaceAdapter(mcp_tools=RaisingTools(boundary_mode="half_open"))

    result = run_temporal_kg_boundary_eval(adapter, cases=[case])

    assert result.case_results[0].signal == TemporalBoundarySignal.NOT_APPLICABLE
    assert result.case_results[0].error is not None


def test_default_cases_randomizes_subject_to_avoid_shared_kg_state() -> None:
    """See mempalace_adapter.py's module docstring: the real KG store has
    no per-run isolation, so default_cases() must not hardcode a fixed
    subject a prior run could have already written facts under."""
    first = default_cases()
    second = default_cases()
    assert first[0].subject != second[0].subject


def test_default_cases_accepts_explicit_subject_prefix() -> None:
    cases = default_cases(subject_prefix="fixed-subject-for-a-test")
    assert cases[0].subject == "fixed-subject-for-a-test"


def test_backend_api_error_is_reraised_normally_for_unrelated_setup() -> None:
    """Sanity check the eval's exception handling doesn't accidentally
    swallow a genuine BackendAPIError raised somewhere other than the
    call sequence it wraps (e.g. an adapter with no configured mcp_tools
    at all raises BackendNotConfiguredError from __init__, well before
    run_temporal_kg_boundary_eval could even be called)."""
    with pytest.raises(BackendAPIError):
        adapter = MemPalaceAdapter(mcp_tools=_RealSemanticsFakeKGTools())
        adapter._get_mcp_tools = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
            BackendAPIError("mempalace", "boom")
        )
        adapter.kg_add("s", "p", "o")
