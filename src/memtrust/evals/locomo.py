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
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "locomo_sample.json"
)


@dataclass
class LoCoMoCaseResult:
    conversation_id: str
    category: str
    question: str
    expected_answer: str
    actual_answer: str
    verdict: JudgeVerdict
    reasoning: str
    records_empty: bool = False
    """True when adapter.query() completed without error but returned zero
    records for this question -- see LongMemEvalCaseResult.records_empty
    and ConflictSignal.EMPTY_OR_LOST for the same distinction applied
    here: a silent empty-success from the backend is not the same
    diagnostic as a judge-graded wrong answer."""
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

    @property
    def accuracy(self) -> float | None:
        graded = self.graded_cases
        if not graded:
            return None
        correct = sum(1 for c in graded if c.verdict == JudgeVerdict.CORRECT)
        return correct / len(graded)

    @property
    def n_records_empty(self) -> int:
        """Count of cases where the backend call succeeded but returned
        zero records -- see LoCoMoCaseResult.records_empty."""
        return sum(1 for c in self.case_results if c.records_empty)

    def accuracy_by_category(self) -> dict[str, float | None]:
        categories = {c.category for c in self.case_results}
        out: dict[str, float | None] = {}
        for cat in categories:
            graded = [c for c in self.graded_cases if c.category == cat]
            if not graded:
                out[cat] = None
                continue
            correct = sum(1 for c in graded if c.verdict == JudgeVerdict.CORRECT)
            out[cat] = correct / len(graded)
        return out


def load_dataset(path: Path | str = DEFAULT_FIXTURE_PATH) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    conversations: list[dict[str, Any]] = data["conversations"]
    return conversations


def _iter_sessions(conversation: dict[str, Any]) -> list[list[dict[str, Any]]]:
    sessions = []
    n = 1
    while f"session_{n}" in conversation:
        sessions.append(conversation[f"session_{n}"])
        n += 1
    return sessions


def run_locomo(
    adapter: MemoryBackendAdapter,
    judge: LLMJudge,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
) -> LoCoMoResult:
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
            for qa in conversation.get("qa", []):
                result.case_results.append(
                    LoCoMoCaseResult(
                        conversation_id=conv_id,
                        category=qa.get("category", "unknown"),
                        question=qa["question"],
                        expected_answer=qa["answer"],
                        actual_answer="",
                        verdict=JudgeVerdict.NOT_RUN,
                        reasoning="",
                        error=str(exc),
                    )
                )
            continue

        for qa in conversation.get("qa", []):
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
                        error=str(exc),
                    )
                )
                continue

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
                    records_empty=not query_result.records,
                )
            )

    return result
