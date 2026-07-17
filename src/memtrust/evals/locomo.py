"""LoCoMo-style multi-session conversational memory eval runner.

LoCoMo (github.com/snap-research/locomo) tests multi-session, multi-day
conversational memory: can the backend answer questions that require
recalling and reasoning across several separate sessions, not just within
one long context window. This runner loads a dataset file matching the
published schema (conversation with speaker_a/speaker_b,
session_<n>/session_<n>_date_time turn lists, and a qa list of
{question, answer, category, evidence}) and, for each conversation:

  1. Replays every session's turns into the backend via store(), scoped
     to a session_id derived from conversation_id.
  2. For every QA pair, calls query() with the question.
  3. Grades the returned content against the expected answer using the
     configured LLM judge, and records the LoCoMo category (single-hop,
     multi-hop, temporal, open-domain, adversarial) alongside the verdict
     so results can be broken down by reasoning type, not just aggregated.

The bundled tests/fixtures/locomo_sample.json is a small, explicitly
synthetic sample matching the real dataset's schema -- see its top-level
"_note" field and docs/methodology.md for exactly what is synthetic here
versus what would run against the real, full public dataset given network
access to download it. The real `locomo10.json` (snap-research/locomo) is
not bundled or auto-fetched here -- it is not memtrust's dataset to
redistribute -- but once downloaded, `memtrust run --locomo-dataset-path
<path>` or `run_locomo(..., dataset_path=<path>)` runs against it directly;
`load_dataset()` below validates the file and raises an actionable error
if it is missing or does not match the expected schema.

## Category 5 / adversarial questions, and headline accuracy

The real LoCoMo benchmark's public release documents its QA set as 1,986
questions total: 1,540 "regular" questions across categories 1-4
(single-hop, multi-hop, temporal, open-domain) plus a 446-question
adversarial category 5, deliberately designed to have no answer in the
conversation (the correct behavior is to recognize the question is
unanswerable, not to produce a confident wrong answer). An independent
audit (dial481/locomo-audit, referenced from mempalace/mempalace#29 and
#875) found this adversarial subset gets folded into vendors' headline
accuracy numbers without disclosure more often than not, and separately
catalogued 99 ground-truth labeling errors in the released dataset.

`LoCoMoResult.accuracy` intentionally keeps its old meaning -- every
graded case, all categories included -- so nothing that already reads
`.accuracy` silently changes behavior. `LoCoMoResult.non_adversarial_accuracy`
is the new, additive number: the same computation restricted to categories
other than "adversarial", mirroring the real benchmark's own 1,540/446
split. Both numbers are always computed and both are surfaced (CLI output,
JSON report, `accuracy_by_category()`) so a reader sees the distinction
instead of a single blended figure.

## Known-bad ground-truth exclusion

`run_locomo()` accepts an optional `exclude_question_ids` parameter -- a
set of question IDs to exclude from scoring entirely, for callers who have
a corrected list of known ground-truth errors (e.g. derived from an audit
like dial481/locomo-audit's 99 flagged cases). This repo does not ship
dial481's specific ID list -- it was not independently verified against
his source data -- so the set defaults to empty and every case is scored
normally until a caller supplies one. See `load_exclude_question_ids()`
for one way to load such a list from a file, and docs/methodology.md for
how a real corrected list would be plugged in.

The published LoCoMo schema does not include a `question_id` field on
each QA entry, so this runner derives a stable one per case as
`f"{conversation_id}::{index_in_conversation}"` unless the dataset
explicitly provides `qa["question_id"]`. A real, corrected exclusion list
must use IDs in that same shape (or supply `question_id` on the source
QA entries) to match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memtrust.adapters.base import BackendAPIError, MemoryBackendAdapter
from memtrust.scoring.llm_judge import JudgeVerdict, LLMJudge

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "locomo_sample.json"
)

#: The real LoCoMo benchmark's category label for its adversarial,
#: deliberately-unanswerable question set (published as "category 5" /
#: 446 of the dataset's 1,986 total questions). Matched against
#: LoCoMoCaseResult.category exactly, the same string this runner already
#: threads through from the dataset's own `qa[*].category` field.
ADVERSARIAL_CATEGORY = "adversarial"


@dataclass
class LoCoMoCaseResult:
    conversation_id: str
    category: str
    question: str
    expected_answer: str
    actual_answer: str
    verdict: JudgeVerdict
    reasoning: str
    question_id: str = ""
    """Stable per-case identifier: the dataset's own `qa["question_id"]`
    if present, otherwise f"{conversation_id}::{index_in_conversation}".
    This is the value `exclude_question_ids` matches against."""
    records_empty: bool = False
    """True when adapter.query() completed without error but returned zero
    records for this question -- see LongMemEvalCaseResult.records_empty
    and ConflictSignal.EMPTY_OR_LOST for the same distinction applied
    here: a silent empty-success from the backend is not the same
    diagnostic as a judge-graded wrong answer."""
    degraded_retrieval: bool = False
    """True when adapter.query() completed without error and returned at
    least one record, but the backend's own response signaled it
    under-delivered anyway -- see LongMemEvalCaseResult.degraded_retrieval
    and adapters/base.py's RetrievalWarning (confirmed against the real,
    merged MemPalace/mempalace#1005 PR diff). Distinct from records_empty:
    a case can have records_empty=False and degraded_retrieval=True at
    the same time, separating "backend warned us, we surfaced it" from
    "backend silently returned wrong or incomplete facts with no signal
    at all."""
    excluded_ground_truth: bool = False
    """True when this case's question_id was listed in the caller's
    exclude_question_ids set -- the case is still recorded (so n_cases
    stays honest) but is never queried or judged, and is excluded from
    every accuracy computation the same way a NOT_RUN case is."""
    error: str | None = None


@dataclass
class LoCoMoResult:
    backend_name: str
    dataset_path: str
    case_results: list[LoCoMoCaseResult] = field(default_factory=list)

    @property
    def graded_cases(self) -> list[LoCoMoCaseResult]:
        return [
            c for c in self.case_results if c.verdict != JudgeVerdict.NOT_RUN and c.error is None
        ]

    @staticmethod
    def _accuracy_over(cases: list[LoCoMoCaseResult]) -> float | None:
        if not cases:
            return None
        correct = sum(1 for c in cases if c.verdict == JudgeVerdict.CORRECT)
        return correct / len(cases)

    @property
    def accuracy(self) -> float | None:
        """Headline accuracy across every graded case, all categories
        included -- adversarial (category 5) questions included, the
        same meaning this property has always had. See
        `non_adversarial_accuracy` for the number that excludes them."""
        return self._accuracy_over(self.graded_cases)

    @property
    def non_adversarial_cases(self) -> list[LoCoMoCaseResult]:
        """Graded cases outside the adversarial category -- the subset
        the real LoCoMo benchmark's own 1,540-question figure covers."""
        return [c for c in self.graded_cases if c.category != ADVERSARIAL_CATEGORY]

    @property
    def non_adversarial_accuracy(self) -> float | None:
        """Accuracy computed only over non-adversarial categories
        (single-hop, multi-hop, temporal, open-domain), excluding the
        adversarial/category-5 subset the same way the real LoCoMo
        benchmark's 1,540/446 split does. This is the number that should
        be quoted as headline accuracy when adversarial questions are
        present but a like-for-like comparison against the benchmark's
        non-adversarial figure is wanted -- see docs/methodology.md."""
        return self._accuracy_over(self.non_adversarial_cases)

    @property
    def n_records_empty(self) -> int:
        """Count of cases where the backend call succeeded but returned
        zero records -- see LoCoMoCaseResult.records_empty."""
        return sum(1 for c in self.case_results if c.records_empty)

    @property
    def n_degraded_retrieval(self) -> int:
        """Count of cases where the backend's own response signaled
        under-delivered (but non-empty) retrieval -- see
        LoCoMoCaseResult.degraded_retrieval."""
        return sum(1 for c in self.case_results if c.degraded_retrieval)

    @property
    def n_excluded_ground_truth(self) -> int:
        """Count of cases excluded from scoring via exclude_question_ids
        -- see LoCoMoCaseResult.excluded_ground_truth."""
        return sum(1 for c in self.case_results if c.excluded_ground_truth)

    def accuracy_by_category(self) -> dict[str, float | None]:
        categories = {c.category for c in self.case_results}
        out: dict[str, float | None] = {}
        for cat in categories:
            graded = [c for c in self.graded_cases if c.category == cat]
            out[cat] = self._accuracy_over(graded)
        return out


def load_dataset(path: Path | str = DEFAULT_FIXTURE_PATH) -> list[dict[str, Any]]:
    """Load a LoCoMo-schema dataset file: either the bundled synthetic
    fixture (the default) or a real `locomo10.json` downloaded from
    snap-research/locomo -- see docs/methodology.md's "LoCoMo" section
    for the download link and schema, and `memtrust run
    --locomo-dataset-path` for the CLI entry point that plugs a real
    download in without writing custom Python.

    Raises `ValueError` with an actionable message (not a bare
    `FileNotFoundError`/`JSONDecodeError`/`KeyError`) when the file is
    missing, is not valid JSON, or does not match the expected
    top-level `{"conversations": [...]}` shape -- schema mismatches are
    the most common real-world failure mode when pointing this at a
    hand-downloaded file, and the previous bare exceptions gave no clue
    what shape was actually expected."""
    resolved = Path(path)
    if not resolved.exists():
        raise ValueError(
            f"LoCoMo dataset file not found: {resolved}. Download locomo10.json from "
            "https://github.com/snap-research/locomo and pass its path via "
            "`memtrust run --locomo-dataset-path` (CLI) or "
            "`run_locomo(..., dataset_path=...)` (Python) -- see docs/methodology.md's "
            '"LoCoMo" section for the download link and expected schema.'
        )
    try:
        data = json.loads(resolved.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LoCoMo dataset file at {resolved} is not valid JSON ({exc}). Expected the "
            'real locomo10.json\'s top-level {"conversations": [...]} shape -- see '
            'docs/methodology.md\'s "LoCoMo" section.'
        ) from exc
    if not isinstance(data, dict) or "conversations" not in data:
        raise ValueError(
            f"LoCoMo dataset file at {resolved} is missing the expected top-level "
            '"conversations" key. This loader expects the real locomo10.json shape: '
            '{"conversations": [{"conversation_id": ..., "session_1": [...], '
            '"qa": [...]}, ...]} -- see docs/methodology.md\'s "LoCoMo" section for the '
            "full schema."
        )
    conversations = data["conversations"]
    if not isinstance(conversations, list):
        raise ValueError(
            f'LoCoMo dataset file at {resolved}: "conversations" must be a list, got '
            f"{type(conversations).__name__}."
        )
    return conversations


def _iter_sessions(conversation: dict[str, Any]) -> list[list[dict[str, Any]]]:
    sessions = []
    n = 1
    while f"session_{n}" in conversation:
        sessions.append(conversation[f"session_{n}"])
        n += 1
    return sessions


def load_exclude_question_ids(path: Path | str) -> set[str]:
    """Load a known-bad-ground-truth question-id exclusion list from a
    file, for use as `run_locomo(..., exclude_question_ids=...)`.

    Accepts either a JSON file containing a top-level list of ID strings
    (e.g. `["mt-locomo-001::3", "mt-locomo-004::0"]`) or a plain text
    file with one question ID per line (blank lines and lines starting
    with "#" are ignored, so a maintainer can annotate why an ID is
    excluded). This is the mechanism a real corrected list -- e.g. one
    derived from an audit like dial481/locomo-audit's 99 flagged
    ground-truth errors -- would be plugged in through; no such list
    ships with this repo (see module docstring)."""
    text = Path(path).read_text()
    stripped = text.strip()
    if stripped.startswith("["):
        ids = json.loads(stripped)
        return {str(i) for i in ids}
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def run_locomo(
    adapter: MemoryBackendAdapter,
    judge: LLMJudge,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
    exclude_question_ids: set[str] | None = None,
) -> LoCoMoResult:
    """Run the LoCoMo eval.

    `exclude_question_ids` is an optional set of question IDs (matching
    LoCoMoCaseResult.question_id -- see module docstring for the ID
    shape) to skip scoring for entirely: no adapter.query() or judge
    call is made, and the case is recorded with
    excluded_ground_truth=True so it never counts toward `accuracy`,
    `non_adversarial_accuracy`, or `accuracy_by_category()`, the same
    way a NOT_RUN case doesn't. Use this to score a real run against a
    corrected ground truth once a caller has a verified list of known
    ground-truth errors (see `load_exclude_question_ids()`)."""
    exclude_ids = exclude_question_ids or set()
    conversations = load_dataset(dataset_path)
    result = LoCoMoResult(backend_name=adapter.name, dataset_path=str(dataset_path))

    for conversation in conversations:
        conv_id = conversation["conversation_id"]
        session_id = f"locomo-{conv_id}"
        try:
            for session in _iter_sessions(conversation):
                for turn in session:
                    adapter.store(session_id, f"{turn['speaker']}: {turn['text']}")
        except BackendAPIError as exc:
            for idx, qa in enumerate(conversation.get("qa", [])):
                question_id = str(qa.get("question_id") or f"{conv_id}::{idx}")
                result.case_results.append(
                    LoCoMoCaseResult(
                        conversation_id=conv_id,
                        category=qa.get("category", "unknown"),
                        question=qa["question"],
                        expected_answer=qa["answer"],
                        actual_answer="",
                        verdict=JudgeVerdict.NOT_RUN,
                        reasoning="",
                        question_id=question_id,
                        excluded_ground_truth=question_id in exclude_ids,
                        error=str(exc),
                    )
                )
            continue

        for idx, qa in enumerate(conversation.get("qa", [])):
            question_id = str(qa.get("question_id") or f"{conv_id}::{idx}")
            if question_id in exclude_ids:
                result.case_results.append(
                    LoCoMoCaseResult(
                        conversation_id=conv_id,
                        category=qa.get("category", "unknown"),
                        question=qa["question"],
                        expected_answer=qa["answer"],
                        actual_answer="",
                        verdict=JudgeVerdict.NOT_RUN,
                        reasoning=(
                            "Excluded via exclude_question_ids: known ground-truth "
                            "error in the source dataset, not scored."
                        ),
                        question_id=question_id,
                        excluded_ground_truth=True,
                    )
                )
                continue
            try:
                query_result = adapter.query(session_id, qa["question"], top_k=5)
            except BackendAPIError as exc:
                result.case_results.append(
                    LoCoMoCaseResult(
                        conversation_id=conv_id,
                        category=qa.get("category", "unknown"),
                        question=qa["question"],
                        expected_answer=qa["answer"],
                        actual_answer="",
                        verdict=JudgeVerdict.NOT_RUN,
                        reasoning="",
                        question_id=question_id,
                        error=str(exc),
                    )
                )
                continue

            # actual_answer is the raw retrieved-record content, judged directly -- there
            # is no answer-generation step here. This is "retrieval-graded accuracy," not
            # the official LoCoMo leaderboard's generate+judge QA-accuracy measurement. See
            # docs/methodology.md's "Retrieval-graded accuracy vs. generated-answer accuracy"
            # section before comparing this metric to a leaderboard figure.
            actual_answer = " ".join(r.content for r in query_result.records)
            judge_result = judge.judge_answer(qa["question"], qa["answer"], actual_answer)
            result.case_results.append(
                LoCoMoCaseResult(
                    conversation_id=conv_id,
                    category=qa.get("category", "unknown"),
                    question=qa["question"],
                    expected_answer=qa["answer"],
                    actual_answer=actual_answer,
                    verdict=judge_result.verdict,
                    reasoning=judge_result.reasoning,
                    question_id=question_id,
                    records_empty=not query_result.records,
                    degraded_retrieval=query_result.degraded_retrieval is not None,
                )
            )

    return result
