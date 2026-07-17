"""MemTrust's extraction-quality-at-scale eval.

`evals/contradiction.py`'s module docstring is explicit about what it is:
"MemTrust's original multi-hop **contradiction-detection** eval" -- a
single stored fact, contradicted by a later one, then queried once. This
eval is the structurally distinct counterpart the backlog asked for: not
"did the backend correctly handle one contradicted fact," but "across many
independently stored items, did the backend retain content it should
never have kept, and does re-storing recalled content silently create
duplicates." Neither LongMemEval, LoCoMo, the contradiction eval, the
compression eval, the ranking-quality eval, nor the resource-sync-safety
eval measures this -- they either measure recall of facts worth
remembering, or the shape/order/fidelity of what comes back. None of them
asks whether the backend should have stored the item at all.

Motivating case: mem0ai/mem0#4573 (GitHub user jamebobob). A 32-day real
production audit found 97.8% of 10,134 stored mem0 entries were junk the
extraction pipeline should never have persisted:

  * boot-file restating (~52.7%) -- the agent's own startup/config text
    (tool lists, persona hashes, context-window sizes) re-stored as if it
    were a new memory every session.
  * cron/heartbeat noise (~11.5%) -- routine liveness-check output
    ("all systems nominal") with no durable content.
  * system dumps (~8.2%) -- raw tracebacks and error-response payloads.
  * hallucinated profiles (~5.2%) -- attributes about the user the model
    invented (favorite colors, pets, hobbies) that the user never
    actually stated.

On top of the base junk-retention problem, jamebobob also documented a
feedback-loop case: a single hallucinated memory, once recalled back into
an agent's context, got re-extracted and re-stored as "new" input -- and
that one re-store fanned out into 808 duplicate records, not one. See
`ExtractionQualitySignal.FEEDBACK_LOOP_DUPLICATE` in adapters/base.py.

Design principle (same as every other eval in this repo): classification
never trusts a backend's own claims about what it stored -- it stores the
seed content, queries it back, and checks the actual retrieved text
against the case's ground-truth `should_be_stored` label, which memtrust
itself assigns and which no adapter ever sees or influences.

Honest limitation -- read this before trusting a junk-retention number.
Every fake adapter this eval's own test suite runs against is
hand-written specifically to model one of two behaviors: "retains
everything indiscriminately" (matching mem0's real reported behavior per
jamebobob's audit) or "filters by a category tag the eval itself passes
in `store()`'s metadata." That second fake is a stand-in for "a backend
with *some* extraction-quality gate," not a claim that any adapter in
this repo talks to a live mem0 instance's real LLM-driven extraction
pipeline -- no adapter here does. This eval and its fixture
(`tests/fixtures/extraction_quality_cases.json`) prove the
*classification logic* is correct against those fakes. Neither has been
run against a live mem0 instance at jamebobob's real 10,000+ entry scale;
see docs/methodology.md for the full caveat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memtrust.adapters.base import (
    BackendAPIError,
    ExtractionQualitySignal,
    MemoryBackendAdapter,
    QueryResult,
)

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "extraction_quality_cases.json"
)

#: A single feedback-loop re-store() call is expected to add exactly one
#: new matching record (or zero, if the backend dedups the recalled text
#: against what it already has). Any growth beyond that is unexpected --
#: see classify_feedback_loop_case() below and
#: ExtractionQualitySignal.FEEDBACK_LOOP_DUPLICATE's docstring for why
#: this is the generalized shape of jamebobob's 808-duplicate finding.
_MAX_EXPECTED_GROWTH_PER_RESTORE = 1


@dataclass
class ExtractionQualityCase:
    case_id: str
    session_id: str
    query: str
    content: str
    category: str
    """Which junk (or valid) shape this case models -- e.g.
    "boot_file_restating", "cron_heartbeat_noise", "system_dump",
    "hallucinated_profile", "other_junk", or "valid_content". Threaded
    into `adapter.store()`'s `metadata` so a fake adapter modeling a
    category-aware extraction gate has something concrete to key off of
    -- see tests/test_evals.py's GatedExtractionFakeAdapter."""
    should_be_stored: bool
    """Ground truth this eval owns, never read from or influenced by the
    adapter under test: True for genuinely valuable content that a
    correctly-behaving backend should retain, False for every junk
    category above."""


@dataclass
class ExtractionQualityCaseResult:
    case: ExtractionQualityCase
    signal: ExtractionQualitySignal
    retrieved: bool
    """Whether the case's `content` was found (as a case-insensitive
    substring) in the joined text of the query() response issued right
    after storing it."""
    retrieved_content: str
    error: str | None = None


@dataclass
class FeedbackLoopCase:
    case_id: str
    session_id: str
    seed_content: str
    unique_marker: str
    """A substring unique to `seed_content` (and expected to survive into
    whatever text the backend echoes back on query()) used to count
    matching records after each step -- robust to a backend that
    paraphrases or otherwise doesn't echo the seed text verbatim, as long
    as the marker itself survives."""
    recall_query: str


@dataclass
class FeedbackLoopCaseResult:
    case: FeedbackLoopCase
    signal: ExtractionQualitySignal
    records_after_first_store: int
    """Count of returned records whose content contains `unique_marker`
    immediately after the seed content is stored once -- expected to be
    exactly 1 for any backend that neither drops nor duplicates a fresh
    write."""
    records_after_second_store: int
    """Same count after the recalled text is re-stored as a second,
    independent store() call. Growth beyond
    records_after_first_store + 1 means that single re-store() call
    produced more than one new matching record -- see
    classify_feedback_loop_case()."""
    recalled_text: str
    """The text this eval re-stored as the second call -- the content of
    whatever record matched `unique_marker` after the first store()+
    query(), falling back to the original seed content if the backend
    returned nothing matching (which itself already means no duplication
    occurred, since there was nothing to re-store from)."""
    error: str | None = None


@dataclass
class ExtractionQualityEvalResult:
    backend_name: str
    dataset_path: str
    case_results: list[ExtractionQualityCaseResult] = field(default_factory=list)
    feedback_loop_results: list[FeedbackLoopCaseResult] = field(default_factory=list)

    @property
    def scored_cases(self) -> list[ExtractionQualityCaseResult]:
        return [c for c in self.case_results if c.error is None]

    @property
    def scored_junk_cases(self) -> list[ExtractionQualityCaseResult]:
        return [c for c in self.scored_cases if not c.case.should_be_stored]

    @property
    def scored_valid_cases(self) -> list[ExtractionQualityCaseResult]:
        return [c for c in self.scored_cases if c.case.should_be_stored]

    def _fraction(
        self, cases: list[ExtractionQualityCaseResult], signal: ExtractionQualitySignal
    ) -> float | None:
        if not cases:
            return None
        matching = sum(1 for c in cases if c.signal == signal)
        return matching / len(cases)

    @property
    def junk_retained_rate(self) -> float | None:
        """Fraction of junk cases (should_be_stored=False) that were still
        retrievable after being stored -- the headline metric this eval
        exists to surface, and the one that would reproduce jamebobob's
        97.8% junk-retention finding if run against a live, ungated
        backend."""
        return self._fraction(self.scored_junk_cases, ExtractionQualitySignal.RETAINED_JUNK)

    @property
    def junk_rejected_rate(self) -> float | None:
        return self._fraction(self.scored_junk_cases, ExtractionQualitySignal.REJECTED_JUNK)

    @property
    def valid_retained_rate(self) -> float | None:
        return self._fraction(self.scored_valid_cases, ExtractionQualitySignal.RETAINED_VALID)

    @property
    def valid_lost_rate(self) -> float | None:
        """Fraction of genuinely valuable content that did NOT survive the
        write path -- the necessary counterweight to junk_rejected_rate: a
        backend that discards everything would score perfectly on
        junk-rejection while failing every real user, and this metric is
        what would catch that."""
        return self._fraction(self.scored_valid_cases, ExtractionQualitySignal.LOST_VALID)

    @property
    def scored_feedback_loop_results(self) -> list[FeedbackLoopCaseResult]:
        return [c for c in self.feedback_loop_results if c.error is None]

    @property
    def feedback_loop_duplicate_rate(self) -> float | None:
        scored = self.scored_feedback_loop_results
        if not scored:
            return None
        matching = sum(
            1 for c in scored if c.signal == ExtractionQualitySignal.FEEDBACK_LOOP_DUPLICATE
        )
        return matching / len(scored)


def load_dataset(
    path: Path | str = DEFAULT_FIXTURE_PATH,
) -> tuple[list[ExtractionQualityCase], list[FeedbackLoopCase]]:
    data = json.loads(Path(path).read_text())
    cases: list[dict[str, Any]] = data["cases"]
    feedback_cases: list[dict[str, Any]] = data.get("feedback_loop_cases", [])
    return (
        [
            ExtractionQualityCase(
                case_id=c["case_id"],
                session_id=c["session_id"],
                query=c["query"],
                content=c["content"],
                category=c["category"],
                should_be_stored=c["should_be_stored"],
            )
            for c in cases
        ],
        [
            FeedbackLoopCase(
                case_id=c["case_id"],
                session_id=c["session_id"],
                seed_content=c["seed_content"],
                unique_marker=c["unique_marker"],
                recall_query=c["recall_query"],
            )
            for c in feedback_cases
        ],
    )


def classify_extraction_case(
    case: ExtractionQualityCase, query_result: QueryResult
) -> tuple[ExtractionQualitySignal, bool]:
    """Classify a single case's outcome from the adapter's query response.

    Returns (final_signal, retrieved). `retrieved` is True if `case.content`
    appears as a case-insensitive substring anywhere in the joined text of
    `query_result.records` -- the same plain-substring convention
    `evals/contradiction.py`'s `classify_case` uses for its own value
    checks, chosen for the same reason: it is the one check every backend
    can be held to without assuming any vendor-specific metadata shape.
    """
    content = " ".join(r.content for r in query_result.records).lower()
    retrieved = case.content.lower() in content

    if case.should_be_stored:
        signal = (
            ExtractionQualitySignal.RETAINED_VALID
            if retrieved
            else ExtractionQualitySignal.LOST_VALID
        )
    else:
        signal = (
            ExtractionQualitySignal.RETAINED_JUNK
            if retrieved
            else ExtractionQualitySignal.REJECTED_JUNK
        )
    return signal, retrieved


def classify_feedback_loop_case(
    records_after_first_store: int, records_after_second_store: int
) -> ExtractionQualitySignal:
    """Classify a feedback-loop case from its two record-count
    observations. A single re-store() call legitimately adds at most one
    new matching record (zero if the backend dedups); any growth beyond
    that is unexpected duplication -- the generalized shape of
    jamebobob's exact 808-duplicate finding (mem0ai/mem0#4573), where one
    re-extracted recall fanned out into hundreds of stored copies instead
    of one.
    """
    growth = records_after_second_store - records_after_first_store
    if growth > _MAX_EXPECTED_GROWTH_PER_RESTORE:
        return ExtractionQualitySignal.FEEDBACK_LOOP_DUPLICATE
    return ExtractionQualitySignal.NO_UNEXPECTED_GROWTH


def run_extraction_quality_eval(
    adapter: MemoryBackendAdapter,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
) -> ExtractionQualityEvalResult:
    cases, feedback_cases = load_dataset(dataset_path)
    result = ExtractionQualityEvalResult(backend_name=adapter.name, dataset_path=str(dataset_path))

    for case in cases:
        try:
            adapter.store(case.session_id, case.content, metadata={"category": case.category})
            query_result = adapter.query(case.session_id, case.query, top_k=10)
        except BackendAPIError as exc:
            result.case_results.append(
                ExtractionQualityCaseResult(
                    case=case,
                    signal=ExtractionQualitySignal.NOT_APPLICABLE,
                    retrieved=False,
                    retrieved_content="",
                    error=str(exc),
                )
            )
            continue

        final_signal, retrieved = classify_extraction_case(case, query_result)
        result.case_results.append(
            ExtractionQualityCaseResult(
                case=case,
                signal=final_signal,
                retrieved=retrieved,
                retrieved_content=" ".join(r.content for r in query_result.records),
            )
        )

    for fb_case in feedback_cases:
        try:
            adapter.store(fb_case.session_id, fb_case.seed_content)
            recall_result = adapter.query(fb_case.session_id, fb_case.recall_query, top_k=50)
            marker = fb_case.unique_marker.lower()
            matching_after_first = [r for r in recall_result.records if marker in r.content.lower()]
            records_after_first = len(matching_after_first)
            recalled_text = (
                matching_after_first[0].content if matching_after_first else fb_case.seed_content
            )

            adapter.store(fb_case.session_id, recalled_text)
            final_result = adapter.query(fb_case.session_id, fb_case.recall_query, top_k=1000)
            records_after_second = sum(
                1 for r in final_result.records if marker in r.content.lower()
            )
        except BackendAPIError as exc:
            result.feedback_loop_results.append(
                FeedbackLoopCaseResult(
                    case=fb_case,
                    signal=ExtractionQualitySignal.NOT_APPLICABLE,
                    records_after_first_store=0,
                    records_after_second_store=0,
                    recalled_text="",
                    error=str(exc),
                )
            )
            continue

        final_signal = classify_feedback_loop_case(records_after_first, records_after_second)
        result.feedback_loop_results.append(
            FeedbackLoopCaseResult(
                case=fb_case,
                signal=final_signal,
                records_after_first_store=records_after_first,
                records_after_second_store=records_after_second,
                recalled_text=recalled_text,
            )
        )

    return result
