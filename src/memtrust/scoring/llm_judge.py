"""LLM-graded scoring for eval outputs that need semantic judgment (does
this recalled answer actually match the expected fact, allowing for
paraphrase) rather than exact string match.

Model-configurable via environment variables:
  MEMTRUST_JUDGE_API_KEY   -- required to actually run a judged eval
  MEMTRUST_JUDGE_MODEL     -- default: deepseek-chat
  MEMTRUST_JUDGE_BASE_URL  -- default: https://api.deepseek.com
                              (OpenAI-compatible chat-completions shape;
                              point this at any OpenAI-compatible endpoint,
                              including a Gemini OpenAI-compat proxy, to
                              use a different judge model)

If no API key is configured, judge_answer() returns a JudgeResult with
verdict=NOT_RUN and an explicit reason -- it never fabricates a score.
Every eval runner that calls this module must propagate NOT_RUN as "could
not be graded," not as a failing score, so an unconfigured judge cannot be
mistaken for a backend that failed the eval.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

import httpx

from memtrust.scoring.cost_tracker import CostTracker

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"

#: Exact judge prompt template, published here and mirrored in
#: docs/methodology.md so every scored run is reproducible from the repo
#: alone -- no prompt lives only in a maintainer's head.
JUDGE_PROMPT_TEMPLATE = """You are grading whether a memory system's recalled answer is factually \
equivalent to the expected answer. Ignore differences in phrasing, tense, or extra \
detail -- grade only whether the core fact matches.

Question asked: {question}
Expected answer: {expected}
System's actual answer: {actual}

Respond with exactly one word on the first line: CORRECT, INCORRECT, or PARTIAL.
On the second line, give a one-sentence reason.
"""


class JudgeVerdict(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"
    NOT_RUN = "not_run"
    """The judge could not run -- no API key configured, or the call
    failed. Never averaged into a score; callers must report this
    verdict's cases as "could not be graded," separate from the scored
    cases."""


@dataclass
class JudgeResult:
    verdict: JudgeVerdict
    reasoning: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class LLMJudge:
    """Thin, provider-agnostic client for the configured judge model.

    Constructed once per run and passed a shared CostTracker so every
    judged eval call's estimated cost rolls up into one printed total.
    """

    def __init__(self, cost_tracker: CostTracker | None = None, timeout: float = 30.0) -> None:
        self.api_key = os.environ.get("MEMTRUST_JUDGE_API_KEY")
        self.model = os.environ.get("MEMTRUST_JUDGE_MODEL", DEFAULT_MODEL)
        self.base_url = os.environ.get("MEMTRUST_JUDGE_BASE_URL", DEFAULT_BASE_URL)
        self.cost_tracker = cost_tracker if cost_tracker is not None else CostTracker()
        self._timeout = timeout
        self._http: httpx.Client | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
        return self._http

    def judge_answer(self, question: str, expected: str, actual: str) -> JudgeResult:
        if not self.is_configured:
            return JudgeResult(
                verdict=JudgeVerdict.NOT_RUN,
                reasoning=(
                    "No judge API key configured (set MEMTRUST_JUDGE_API_KEY). "
                    "This eval could not be graded -- reporting as not-run, "
                    "not as a failing score."
                ),
            )

        prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, expected=expected, actual=actual)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 100,
        }
        try:
            resp = self._client().post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            return JudgeResult(
                verdict=JudgeVerdict.NOT_RUN,
                reasoning=(
                    f"Judge API call failed: {exc}. Reporting as not-run, not a failing score."
                ),
                model=self.model,
            )

        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        self.cost_tracker.record(
            label=f"judge:{question[:40]}",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        first_line = text.splitlines()[0].strip().upper() if text else ""
        reasoning = text.splitlines()[1].strip() if len(text.splitlines()) > 1 else text
        if "CORRECT" in first_line and "INCORRECT" not in first_line:
            verdict = JudgeVerdict.CORRECT
        elif "PARTIAL" in first_line:
            verdict = JudgeVerdict.PARTIAL
        else:
            verdict = JudgeVerdict.INCORRECT
        return JudgeResult(
            verdict=verdict,
            reasoning=reasoning,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
