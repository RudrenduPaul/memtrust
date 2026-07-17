"""LongMemEval-style long-horizon recall eval runner.

LongMemEval (Wu et al., ICLR 2025, github.com/xiaowu0162/LongMemEval) asks
whether a chat assistant can correctly recall a fact injected many turns
earlier in a long conversation. This runner loads a dataset file matching
the published schema (question_id, question_type, question, answer,
haystack_sessions -- a list of chat sessions, each a list of
{role, content} turns) and, for each example:

  1. Replays every haystack session into the backend via store(), one
     call per turn, scoped to a session_id derived from question_id so
     concurrent examples never collide.
  2. Calls query() with the example's question.
  3. Grades the returned content against the expected answer using the
     configured LLM judge (semantic match, not exact string match --
     "Baxter the golden retriever" should count as correct for an answer
     of "Baxter").

The bundled tests/fixtures/longmemeval_sample.json is a small, explicitly
synthetic sample matching the real dataset's schema -- see its top-level
"_note" field and docs/methodology.md for exactly what is synthetic here
versus what would run against the real, full public dataset given network
access to download it.

## top_k vs. corpus size

`run_longmemeval()` queries every case with a fixed `top_k` (see
DEFAULT_TOP_K below). A benchmark row is only measuring real retrieval
quality when the backend has to rank `top_k` results out of a corpus
meaningfully larger than `top_k` -- if a case's corpus (the haystack
turns actually stored for that question) is no bigger than `top_k`,
"return the right answer" degenerates into "return everything you have,"
which any backend can do regardless of ranking quality. This is the same
shape contributor jtatum flagged in MemPalace's own benchmark scripts:
`n_results >= corpus_size` trivializes recall into "rank over a corpus
you can already see all of." This repo could not independently verify a
specific upstream issue number for that finding, so it is credited here
by name rather than cited with a fabricated `repo#N` link -- unlike every
other "Motivating case" note in this codebase (see docs/methodology.md).

Structurally, this mirrors `evals/locomo.py`'s `non_adversarial_accuracy`:
both exist so a caller sees a benchmark-integrity concern instead of a
single blended number. The two mechanisms differ because the concerns
differ -- LoCoMo's adversarial subset is a category of *questions* to
exclude from an aggregate, known ahead of time from the dataset; a
corpus-size-vs-top_k mismatch is a per-case *retrieval setup* fact that
only exists once haystack turns are actually stored, so LongMemEval
surfaces it as a flag on `LongMemEvalCaseResult`
(`top_k_exceeds_corpus`) plus an aggregate count
(`LongMemEvalResult.n_top_k_exceeds_corpus`) rather than a second
accuracy property -- it is a disclosure a caller checks before trusting
a high accuracy number, not a subset LongMemEval's own published schema
says to exclude from scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memtrust.adapters.base import BackendAPIError, MemoryBackendAdapter
from memtrust.scoring.llm_judge import JudgeVerdict, LLMJudge

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "longmemeval_sample.json"
)

#: Default `top_k` passed to every `adapter.query()` call in
#: `run_longmemeval()`. Pulled out as a named constant (rather than the
#: literal `5` in two places) specifically so it can be compared against
#: each case's corpus size for the `top_k_exceeds_corpus` guard below --
#: see the module docstring's "top_k vs. corpus size" section.
DEFAULT_TOP_K = 5


@dataclass
class LongMemEvalCaseResult:
    question_id: str
    question_type: str
    question: str
    expected_answer: str
    actual_answer: str
    verdict: JudgeVerdict
    reasoning: str
    records_empty: bool = False
    """True when adapter.query() completed without error but returned zero
    records for this question. A judge then grades an empty actual_answer
    as an ordinary miss (wrong/incorrect) same as any other wrong answer --
    this field exists so a run can distinguish "the model reasoned about
    retrieved content and got it wrong" from "the backend silently gave
    back nothing to reason about," which is a diagnostically different
    failure the underlying vendor call may share across many questions.
    See adapters/base.py's ConflictSignal.EMPTY_OR_LOST for the analogous
    signal in the contradiction eval."""
    degraded_retrieval: bool = False
    """True when adapter.query() completed without error and returned at
    least one record, but the backend's own response signaled it
    under-delivered anyway (see adapters/base.py's RetrievalWarning,
    confirmed against the real, merged MemPalace/mempalace#1005 PR diff).
    This is distinct from records_empty above: a case can have
    records_empty=False and degraded_retrieval=True at the same time --
    the backend returned *something* to grade, but warned it wasn't the
    full picture. Separating the two lets a report distinguish "backend
    warned us, we surfaced it" from "backend silently returned wrong or
    incomplete facts with no signal at all," which the judge's verdict on
    its own cannot tell apart."""
    corpus_size: int = 0
    """Number of haystack turns actually stored for this case (one
    store() call per `role == "user"` turn across every haystack
    session) before query() was called -- the real size of the corpus
    `top_k` is being asked to rank over for this specific case. See
    `top_k_exceeds_corpus` below and the module docstring's "top_k vs.
    corpus size" section."""
    top_k_exceeds_corpus: bool = False
    """True when this case's corpus_size is less than or equal to the
    top_k value used for query() (DEFAULT_TOP_K unless a caller passes a
    different one). When True, a high accuracy verdict on this case does
    not by itself demonstrate genuine retrieval quality -- top_k already
    covers the entire corpus, so "return the right answer" reduces to
    "return everything," which any backend can do regardless of ranking
    quality. See the module docstring's "top_k vs. corpus size" section
    for the motivating gaming shape this guards against."""
    error: str | None = None


@dataclass
class LongMemEvalResult:
    backend_name: str
    dataset_path: str
    case_results: list[LongMemEvalCaseResult] = field(default_factory=list)

    @property
    def graded_cases(self) -> list[LongMemEvalCaseResult]:
        return [
            c for c in self.case_results if c.verdict != JudgeVerdict.NOT_RUN and c.error is None
        ]

    @property
    def accuracy(self) -> float | None:
        graded = self.graded_cases
        if not graded:
            return None
        correct = sum(1 for c in graded if c.verdict == JudgeVerdict.CORRECT)
        return correct / len(graded)

    @property
    def judge_unavailable(self) -> bool:
        return len(self.graded_cases) == 0 and len(self.case_results) > 0

    @property
    def n_records_empty(self) -> int:
        """Count of cases where the backend call succeeded but returned
        zero records -- see LongMemEvalCaseResult.records_empty."""
        return sum(1 for c in self.case_results if c.records_empty)

    @property
    def n_degraded_retrieval(self) -> int:
        """Count of cases where the backend's own response signaled
        under-delivered (but non-empty) retrieval -- see
        LongMemEvalCaseResult.degraded_retrieval."""
        return sum(1 for c in self.case_results if c.degraded_retrieval)

    @property
    def n_top_k_exceeds_corpus(self) -> int:
        """Count of cases whose corpus was no bigger than top_k -- see
        LongMemEvalCaseResult.top_k_exceeds_corpus and the module
        docstring's "top_k vs. corpus size" section. A non-zero count
        here means at least one case in this run cannot distinguish
        genuine retrieval quality from a backend simply returning its
        entire (small) corpus."""
        return sum(1 for c in self.case_results if c.top_k_exceeds_corpus)


def load_dataset(path: Path | str = DEFAULT_FIXTURE_PATH) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    examples: list[dict[str, Any]] = data["examples"]
    return examples


def run_longmemeval(
    adapter: MemoryBackendAdapter,
    judge: LLMJudge,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
) -> LongMemEvalResult:
    examples = load_dataset(dataset_path)
    result = LongMemEvalResult(backend_name=adapter.name, dataset_path=str(dataset_path))

    for example in examples:
        session_id = f"longmemeval-{example['question_id']}"
        # Computed from the dataset directly (not from a running counter of
        # successful store() calls) so it reflects the intended corpus size
        # for this case even if store() fails partway through -- see the
        # module docstring's "top_k vs. corpus size" section.
        corpus_size = sum(
            1
            for session in example["haystack_sessions"]
            for turn in session
            if turn["role"] == "user"
        )
        top_k_exceeds_corpus = corpus_size <= DEFAULT_TOP_K
        try:
            for session in example["haystack_sessions"]:
                for turn in session:
                    if turn["role"] == "user":
                        adapter.store(session_id, turn["content"])
            query_result = adapter.query(session_id, example["question"], top_k=DEFAULT_TOP_K)
        except BackendAPIError as exc:
            result.case_results.append(
                LongMemEvalCaseResult(
                    question_id=example["question_id"],
                    question_type=example["question_type"],
                    question=example["question"],
                    expected_answer=example["answer"],
                    actual_answer="",
                    verdict=JudgeVerdict.NOT_RUN,
                    reasoning="",
                    corpus_size=corpus_size,
                    top_k_exceeds_corpus=top_k_exceeds_corpus,
                    error=str(exc),
                )
            )
            continue

        # actual_answer is the raw retrieved-record content, judged directly -- there is
        # no answer-generation step here. This is "retrieval-graded accuracy," not the
        # official LongMemEval leaderboard's generate+judge QA-accuracy measurement. See
        # docs/methodology.md's "Retrieval-graded accuracy vs. generated-answer accuracy"
        # section before comparing this metric to a leaderboard figure.
        actual_answer = " ".join(r.content for r in query_result.records)
        judge_result = judge.judge_answer(example["question"], example["answer"], actual_answer)
        result.case_results.append(
            LongMemEvalCaseResult(
                question_id=example["question_id"],
                question_type=example["question_type"],
                question=example["question"],
                expected_answer=example["answer"],
                actual_answer=actual_answer,
                verdict=judge_result.verdict,
                reasoning=judge_result.reasoning,
                records_empty=not query_result.records,
                degraded_retrieval=query_result.degraded_retrieval is not None,
                corpus_size=corpus_size,
                top_k_exceeds_corpus=top_k_exceeds_corpus,
            )
        )

    return result
