"""Tracks estimated token/cost per run so `memtrust run` can print a cost
summary. Pricing is an approximate, dated estimate -- not a billing
guarantee. See PRICING_LAST_VERIFIED below and docs/methodology.md.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

#: Last time the per-1M-token prices below were checked against the
#: provider's published pricing page. Prices change; treat this table as
#: an estimate for cost-awareness, not an invoice.
PRICING_LAST_VERIFIED = "2026-07-11"

#: USD per 1,000,000 tokens, (input, output). Approximate, from each
#: provider's public pricing page as of PRICING_LAST_VERIFIED.
MODEL_PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "deepseek-chat": (0.28, 0.42),
    "deepseek-reasoner": (0.56, 1.68),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
}

DEFAULT_UNKNOWN_MODEL_PRICE = (0.50, 1.50)


@dataclass
class CostEntry:
    label: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """Accumulates cost entries across a `memtrust run` invocation.

    memtrust's eval runners are sequential today, so ``_lock`` guards a
    mutation that is not actually contended yet. It's here preemptively:
    a real, merged vendor bug (volcengine/OpenViking PR #3091) shipped a
    benchmark tool that silently ignored its own --concurrency flag,
    quietly invalidating its self-reported numbers. Keeping this shared
    list append synchronized now means that if/when memtrust adds
    concurrent eval execution, cost accounting can't silently drop or
    corrupt entries the way an unsynchronized append would.
    """

    entries: list[CostEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, label: str, model: str, input_tokens: int, output_tokens: int) -> CostEntry:
        price_in, price_out = MODEL_PRICING_PER_MILLION_TOKENS.get(
            model, DEFAULT_UNKNOWN_MODEL_PRICE
        )
        cost = (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out
        entry = CostEntry(
            label=label,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
        )
        with self._lock:
            self.entries.append(entry)
        return entry

    @property
    def total_cost_usd(self) -> float:
        return round(sum(e.cost_usd for e in self.entries), 6)

    @property
    def total_input_tokens(self) -> int:
        return sum(e.input_tokens for e in self.entries)

    @property
    def total_output_tokens(self) -> int:
        return sum(e.output_tokens for e in self.entries)

    def summary_lines(self) -> list[str]:
        if not self.entries:
            return [
                "Cost: $0.00 (no LLM-judged evals ran -- structural evals only, "
                "or judge not configured)"
            ]
        lines = [
            f"Cost: ${self.total_cost_usd:.4f} estimated "
            f"({self.total_input_tokens:,} in / {self.total_output_tokens:,} out tokens, "
            f"pricing last verified {PRICING_LAST_VERIFIED})"
        ]
        by_model: dict[str, float] = {}
        for e in self.entries:
            by_model[e.model] = by_model.get(e.model, 0.0) + e.cost_usd
        for model, cost in sorted(by_model.items()):
            lines.append(f"  {model}: ${cost:.4f}")
        return lines
