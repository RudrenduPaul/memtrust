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
        try:
            for session in example["haystack_sessions"]:
                for turn in session:
                    if turn["role"] == "user":
                        adapter.store(session_id, turn["content"])
            query_result = adapter.query(session_id, example["question"], top_k=5)
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
            )
        )

    return result
