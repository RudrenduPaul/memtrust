"""Scoring pipeline tests: cost tracker arithmetic and the LLM judge's
no-crash NOT_RUN fallback plus its parsing logic against a mocked HTTP
response (no real network / vendor credentials used).
"""

from __future__ import annotations

import threading
import time

import pytest
from pytest_httpx import HTTPXMock

from memtrust.scoring.cost_tracker import CostEntry, CostTracker
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
# CostTracker.record_embed_calls() / EmbedCallEntry -- vendor-side
# embedder-call/cost attribution, distinct from record()'s judge-LLM
# spend. See mem0_direct_adapter.py's _CountingEmbedder and
# evals/embedder_cost.py (spike-spiegel-21, mem0ai/mem0#1900).
# ---------------------------------------------------------------------------


def test_cost_tracker_record_embed_calls_known_provider_pricing() -> None:
    tracker = CostTracker()
    entry = tracker.record_embed_calls(
        "mem0_direct:store:s1", "openai", call_count=1, estimated_tokens=1_000_000
    )
    assert entry.cost_usd == pytest.approx(0.02)
    assert entry.call_count == 1
    assert tracker.total_embed_calls == 1
    assert tracker.total_embed_cost_usd == pytest.approx(0.02)


def test_cost_tracker_record_embed_calls_fastembed_is_genuinely_free() -> None:
    tracker = CostTracker()
    entry = tracker.record_embed_calls(
        "mem0_direct:store:s1", "fastembed", call_count=3, estimated_tokens=5_000_000
    )
    assert entry.cost_usd == 0.0


def test_cost_tracker_record_embed_calls_unknown_provider_uses_default_price() -> None:
    tracker = CostTracker()
    entry = tracker.record_embed_calls(
        "x", "some-new-provider", call_count=1, estimated_tokens=1_000_000
    )
    assert entry.cost_usd == pytest.approx(0.10)


def test_cost_tracker_embed_calls_accumulate_independently_of_judge_entries() -> None:
    tracker = CostTracker()
    tracker.record("judge-case", "deepseek-chat", 1000, 1000)
    tracker.record_embed_calls("embed-1", "openai", call_count=2, estimated_tokens=2_000_000)
    assert len(tracker.entries) == 1
    assert len(tracker.embed_entries) == 1
    assert tracker.total_embed_calls == 2
    # The two spend categories never get merged into each other's totals.
    assert tracker.total_cost_usd != tracker.total_embed_cost_usd


def test_cost_tracker_summary_lines_includes_embed_call_line() -> None:
    tracker = CostTracker()
    tracker.record_embed_calls("x", "openai", call_count=4, estimated_tokens=4_000_000)
    lines = tracker.summary_lines()
    assert any("Vendor embedder calls" in line for line in lines)
    assert any("openai" in line for line in lines)


def test_cost_tracker_summary_lines_empty_when_no_entries_of_either_kind() -> None:
    tracker = CostTracker()
    lines = tracker.summary_lines()
    assert len(lines) == 1
    assert "$0.00" in lines[0]


class _NonAtomicEntryList(list[CostEntry]):
    """Test double standing in for ``CostTracker.entries``.

    Plain CPython ``list.append()`` happens to be a single, GIL-atomic
    bytecode op, so hammering it with threads doesn't reliably expose a
    missing lock on this interpreter -- a bare (unlocked) append test was
    verified by hand to pass even with ``CostTracker``'s lock removed.
    This double instead does a realistic read-copy-mutate-write-back
    append (the shape a future refactor of ``entries`` could easily take),
    with a scheduler yield in the middle of the window, so a race is
    actually observable if the surrounding code fails to serialize
    access.
    """

    def append(self, item: CostEntry) -> None:
        snapshot = list(self)
        time.sleep(0)  # yield so another thread can interleave right here
        snapshot.append(item)
        self[:] = snapshot


def test_cost_tracker_record_is_thread_safe() -> None:
    """Preventive concurrency test: memtrust's runners are sequential today,
    but CostTracker.record() is now guarded by a lock so it stays safe if
    concurrent eval execution is added later (see the class docstring and
    the OpenViking PR #3091 precedent that motivated this).

    ``entries`` is swapped for ``_NonAtomicEntryList`` so the test proves
    CostTracker's own lock -- not incidental atomicity of CPython's
    built-in ``list.append`` -- is what keeps concurrent record() calls
    from dropping entries. This was confirmed to fail reliably (final
    count short of the expected total) when the ``with self._lock:``
    guard is removed from record(), and to pass reliably with it in
    place.
    """
    tracker = CostTracker()
    tracker.entries = _NonAtomicEntryList()
    num_threads = 16
    records_per_thread = 25
    expected_total = num_threads * records_per_thread

    def worker() -> None:
        for _ in range(records_per_thread):
            tracker.record("case", "deepseek-chat", 100, 100)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(tracker.entries) == expected_total
    assert tracker.total_input_tokens == expected_total * 100


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
