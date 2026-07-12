"""Scoring pipeline tests: cost tracker arithmetic and the LLM judge's
no-crash NOT_RUN fallback plus its parsing logic against a mocked HTTP
response (no real network / vendor credentials used).
"""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from memtrust.scoring.cost_tracker import CostTracker
from memtrust.scoring.llm_judge import JudgeVerdict, LLMJudge

# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


def test_cost_tracker_known_model_pricing() -> None:
    tracker = CostTracker()
    entry = tracker.record(
        "case-1", "deepseek-chat", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert entry.cost_usd == pytest.approx(0.28 + 0.42)
    assert tracker.total_cost_usd == pytest.approx(0.70)


def test_cost_tracker_unknown_model_uses_default_pricing() -> None:
    tracker = CostTracker()
    entry = tracker.record("case-1", "some-new-model", input_tokens=1_000_000, output_tokens=0)
    assert entry.cost_usd == pytest.approx(0.50)


def test_cost_tracker_accumulates_across_entries() -> None:
    tracker = CostTracker()
    tracker.record("a", "deepseek-chat", 1_000_000, 0)
    tracker.record("b", "deepseek-chat", 1_000_000, 0)
    assert tracker.total_cost_usd == pytest.approx(0.56)
    assert tracker.total_input_tokens == 2_000_000


def test_cost_tracker_summary_lines_empty() -> None:
    tracker = CostTracker()
    lines = tracker.summary_lines()
    assert len(lines) == 1
    assert "$0.00" in lines[0]


def test_cost_tracker_summary_lines_nonempty() -> None:
    tracker = CostTracker()
    tracker.record("a", "deepseek-chat", 1000, 1000)
    lines = tracker.summary_lines()
    assert any("deepseek-chat" in line for line in lines)


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


def test_judge_not_run_when_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMTRUST_JUDGE_API_KEY", raising=False)
    judge = LLMJudge()
    assert judge.is_configured is False
    result = judge.judge_answer("q", "expected", "actual")
    assert result.verdict == JudgeVerdict.NOT_RUN
    assert "MEMTRUST_JUDGE_API_KEY" in result.reasoning


def test_judge_configured_when_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMTRUST_JUDGE_API_KEY", "test-key")
    judge = LLMJudge()
    assert judge.is_configured is True


def test_judge_parses_correct_verdict(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("MEMTRUST_JUDGE_API_KEY", "test-key")
    judge = LLMJudge()
    httpx_mock.add_response(
        method="POST",
        url="https://api.deepseek.com/chat/completions",
        json={
            "choices": [{"message": {"content": "CORRECT\nThe dog's name matches."}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        },
    )
    result = judge.judge_answer("What is my dog's name?", "Baxter", "Baxter the golden retriever")
    assert result.verdict == JudgeVerdict.CORRECT
    assert result.input_tokens == 50
    assert judge.cost_tracker.total_cost_usd > 0
    judge.close()


def test_judge_parses_incorrect_verdict(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("MEMTRUST_JUDGE_API_KEY", "test-key")
    judge = LLMJudge()
    httpx_mock.add_response(
        method="POST",
        url="https://api.deepseek.com/chat/completions",
        json={
            "choices": [{"message": {"content": "INCORRECT\nNo match found."}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        },
    )
    result = judge.judge_answer("q", "expected", "totally different")
    assert result.verdict == JudgeVerdict.INCORRECT
    judge.close()


def test_judge_parses_partial_verdict(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("MEMTRUST_JUDGE_API_KEY", "test-key")
    judge = LLMJudge()
    httpx_mock.add_response(
        method="POST",
        url="https://api.deepseek.com/chat/completions",
        json={
            "choices": [{"message": {"content": "PARTIAL\nClose but missing detail."}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        },
    )
    result = judge.judge_answer("q", "expected", "half right")
    assert result.verdict == JudgeVerdict.PARTIAL
    judge.close()


def test_judge_returns_not_run_on_http_failure(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("MEMTRUST_JUDGE_API_KEY", "test-key")
    judge = LLMJudge()
    httpx_mock.add_response(status_code=500)
    result = judge.judge_answer("q", "expected", "actual")
    assert result.verdict == JudgeVerdict.NOT_RUN
    assert "failed" in result.reasoning.lower()
    judge.close()


def test_judge_respects_custom_model_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMTRUST_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("MEMTRUST_JUDGE_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("MEMTRUST_JUDGE_BASE_URL", "https://example.com")
    judge = LLMJudge()
    assert judge.model == "gemini-2.5-flash-lite"
    assert judge.base_url == "https://example.com"
