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

#: Approximate USD per 1,000,000 embedded input characters, keyed by
#: embedder provider name (matching `Mem0DirectAdapter.SUPPORTED_EMBEDDER_
#: PROVIDERS`'s naming) -- NOT per-token. Vendor embedder-call
#: instrumentation (see mem0_direct_adapter.py's `_CountingEmbedder`) only
#: has each `embed()`/`embed_batch()` call's raw input text available, not
#: the vendor's own tokenizer's real token count, so a char/4 heuristic is
#: applied at the call site before this table's price is used -- the same
#: "estimate, not an invoice" caveat `MODEL_PRICING_PER_MILLION_TOKENS`
#: above already carries, extended here to vendor-side embedding spend
#: rather than memtrust's own judge-LLM spend. Approximate, from each
#: provider's public embedding-pricing page as of PRICING_LAST_VERIFIED.
#: `fastembed` runs a local ONNX model with no per-call vendor API cost at
#: all, hence $0 -- not an "unknown, guess" $0, a genuine $0.
EMBEDDER_PRICING_PER_MILLION_TOKENS: dict[str, float] = {
    "openai": 0.02,  # text-embedding-3-small, OpenAI's cheapest/default embedding model
    "aws_bedrock": 0.02,  # Amazon Titan Text Embeddings V2
    "gemini": 0.00,  # no public per-token embedding price line-item as of PRICING_LAST_VERIFIED
    "fastembed": 0.00,  # local ONNX model, genuinely no vendor API call
}

DEFAULT_UNKNOWN_EMBEDDER_PRICE = 0.10


@dataclass
class CostEntry:
    label: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class EmbedCallEntry:
    """Vendor-side embedder-call/cost attribution -- distinct from
    `CostEntry` above, which tracks only memtrust's own judge-LLM spend.
    See `CostTracker.record_embed_calls()` and
    mem0_direct_adapter.py's `_CountingEmbedder` for where this gets
    populated. Motivating case: spike-spiegel-21 (rank 104,
    mem0ai/mem0#1900) landed a merged fix eliminating a duplicate
    embedding-API call in an earlier mem0 architecture when content was
    unchanged after search -- memtrust had zero vendor-side embedder-call
    instrumentation anywhere to observe whether that class of waste is
    still happening in the currently installed mem0ai version's own
    pipeline. See evals/embedder_cost.py.
    """

    label: str
    provider: str
    call_count: int
    estimated_tokens: int
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
    corrupt entries the way an unsynchronized append would. The same
    `_lock` guards `embed_entries` below.
    """

    entries: list[CostEntry] = field(default_factory=list)
    embed_entries: list[EmbedCallEntry] = field(default_factory=list)
    """Vendor-side embedder-call entries -- kept as a separate list from
    `entries` above (rather than folded into the same list/shape) because
    the two track fundamentally different spend categories: `entries` is
    memtrust's own judge-LLM cost (real input/output token counts from a
    real LLM API response), `embed_entries` is vendor embedder-API call
    count/cost attribution derived from a char-length heuristic, never a
    real token count the vendor itself reported. Conflating the two into
    one list/one set of properties would misrepresent the second
    category's evidentiary strength as equal to the first's."""
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

    def record_embed_calls(
        self, label: str, provider: str, call_count: int, estimated_tokens: int
    ) -> EmbedCallEntry:
        """Record vendor-side embedder-call/cost attribution for a single
        adapter operation (typically one `Mem0DirectAdapter.store()` call)
        -- distinct from `record()` above, which is memtrust's own
        judge-LLM spend only. See `EmbedCallEntry`'s docstring for why
        this is a separate list/method rather than reusing `record()`'s
        shape.
        """
        price = EMBEDDER_PRICING_PER_MILLION_TOKENS.get(provider, DEFAULT_UNKNOWN_EMBEDDER_PRICE)
        cost = (estimated_tokens / 1_000_000) * price
        entry = EmbedCallEntry(
            label=label,
            provider=provider,
            call_count=call_count,
            estimated_tokens=estimated_tokens,
            cost_usd=round(cost, 6),
        )
        with self._lock:
            self.embed_entries.append(entry)
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

    @property
    def total_embed_calls(self) -> int:
        return sum(e.call_count for e in self.embed_entries)

    @property
    def total_embed_cost_usd(self) -> float:
        return round(sum(e.cost_usd for e in self.embed_entries), 6)

    def summary_lines(self) -> list[str]:
        if not self.entries and not self.embed_entries:
            return [
                "Cost: $0.00 (no LLM-judged evals ran -- structural evals only, "
                "or judge not configured)"
            ]
        lines: list[str] = []
        if self.entries:
            lines.append(
                f"Cost: ${self.total_cost_usd:.4f} estimated "
                f"({self.total_input_tokens:,} in / {self.total_output_tokens:,} out tokens, "
                f"pricing last verified {PRICING_LAST_VERIFIED})"
            )
            by_model: dict[str, float] = {}
            for e in self.entries:
                by_model[e.model] = by_model.get(e.model, 0.0) + e.cost_usd
            for model, cost in sorted(by_model.items()):
                lines.append(f"  {model}: ${cost:.4f}")
        if self.embed_entries:
            lines.append(
                f"Vendor embedder calls: {self.total_embed_calls:,} "
                f"(~${self.total_embed_cost_usd:.4f} estimated, pricing last verified "
                f"{PRICING_LAST_VERIFIED})"
            )
            by_provider: dict[str, int] = {}
            for embed_entry in self.embed_entries:
                by_provider[embed_entry.provider] = (
                    by_provider.get(embed_entry.provider, 0) + embed_entry.call_count
                )
            for provider, calls in sorted(by_provider.items()):
                lines.append(f"  {provider}: {calls:,} calls")
        return lines
